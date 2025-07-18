"""
A script for sampling from a diffusion model for paired image-to-image translation.
(Modified for t1n to t1c translation, compatible with a specific bratsloader)
"""

import argparse
import nibabel as nib
import numpy as np
import os
import pathlib
import random
import sys
import torch as th
import torch.nn.functional as F

# Add the project root to the Python path
sys.path.append(".")

from guided_diffusion import (dist_util,
                              logger)
from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.script_util import (model_and_diffusion_defaults, create_model_and_diffusion,
                                          add_dict_to_argparser, args_to_dict)
from DWT_IDWT.DWT_IDWT_layer import IDWT_3D, DWT_3D

def main():
    args = create_argparser().parse_args()
    seed = args.seed
    dist_util.setup_dist(devices=args.devices)
    logger.configure()

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    diffusion.mode = 'i2i'
    logger.log("Load model from: {}".format(args.model_path))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())

    if args.dataset == 'brats':
        # Use the modified BRATSVolumes loader for evaluation
        ds = BRATSVolumes(args.data_dir, mode='eval')

    datal = th.utils.data.DataLoader(ds,
                                     batch_size=args.batch_size,
                                     num_workers=12,
                                     shuffle=False,)

    model.eval()
    idwt = IDWT_3D("haar")
    dwt = DWT_3D("haar")

    # Set random seeds for reproducibility
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # ★★ 修正点1: enumerateを使い、ループのインデックス(i)を取得する ★★
    # ★★ DataLoaderが返すタプルを target_batch と cond_batch で受け取る ★★
    for i, (target_batch, cond_batch) in enumerate(datal):
        
        # Since batch size is 1, get the first element from the list
        target = target_batch.to(dist_util.dev())
        cond_1 = cond_batch['cond_1'].to(dist_util.dev())

        # ★★ 修正点2: 被験者IDの代わりに連番でフォルダ名を作成する ★★
        subj = f"Sample_{i:04d}"
        print(f"Processing: {subj}")

        # Create conditioning vector from t1n only
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(cond_1)
        cond = th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)

        # Prepare noise
        noise = th.randn(args.batch_size, 8, 112, 112, 80).to(dist_util.dev())

        model_kwargs = {}

        sample_fn = diffusion.p_sample_loop

        # Run sampling
        sample = sample_fn(model=model,
                           shape=noise.shape,
                           noise=noise,
                           cond=cond,
                           clip_denoised=args.clip_denoised,
                           model_kwargs=model_kwargs)

        # Inverse DWT to bring the generated image back to the pixel space
        B, _, D, H, W = sample.size()
        sample = idwt(sample[:, 0, :, :, :].view(B, 1, D, H, W) * 3.,
                      sample[:, 1, :, :, :].view(B, 1, D, H, W),
                      sample[:, 2, :, :, :].view(B, 1, D, H, W),
                      sample[:, 3, :, :, :].view(B, 1, D, H, W),
                      sample[:, 4, :, :, :].view(B, 1, D, H, W),
                      sample[:, 5, :, :, :].view(B, 1, D, H, W),
                      sample[:, 6, :, :, :].view(B, 1, D, H, W),
                      sample[:, 7, :, :, :].view(B, 1, D, H, W))

        # Post-processing
        sample[sample <= 0] = 0
        sample[sample >= 1] = 1
        sample[cond_1 == 0] = 0 # Mask out non-brain regions

        if len(sample.shape) == 5:
            sample = sample.squeeze(dim=1)

        sample = sample[:, :, :, :155]

        if len(target.shape) == 5:
            target = target.squeeze(dim=1)

        target = target[:, :, :, :155]

        # Create output directories
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        # ★★ 修正点3: 連番のフォルダパスを作成する ★★
        subj_dir = os.path.join(args.output_dir, subj)
        pathlib.Path(subj_dir).mkdir(parents=True, exist_ok=True)

        # Save generated and target images
        for i_img in range(sample.shape[0]):
            output_name = os.path.join(subj_dir, 'sample_t1c_from_t1n.nii.gz')
            img = nib.Nifti1Image(sample.detach().cpu().numpy()[i_img, :, :, :], np.eye(4))
            nib.save(img=img, filename=output_name)
            print(f'Saved to {output_name}')

            output_name = os.path.join(subj_dir, 'target_t1c.nii.gz')
            img = nib.Nifti1Image(target.detach().cpu().numpy()[i_img, :, :, :], np.eye(4))
            nib.save(img=img, filename=output_name)

def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        data_mode='validation',
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        class_cond=False,
        sampling_steps=0,
        model_path="",
        devices=[0],
        output_dir='./results',
        mode='default',
        renormalize=False,
        image_size=256,
        half_res_crop=False,
        concat_coords=False,
    )
    # Remove the 'contr' argument as it's no longer needed
    defaults.update({k:v for k, v in model_and_diffusion_defaults().items() if 'contr' not in k}) 
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()