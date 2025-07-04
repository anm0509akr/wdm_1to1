import sys
import os
sys.path.append(os.getcwd())
import argparse

# wandbのimportを削除

import torch.distributed as dist
import torch.utils.tensorboard
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        resume_step=0,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset='brats',
        summary_writer=None,
        mode='i2i',
        loss_level='image'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()

    # wandb.init()の呼び出しを削除
    
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    
    # args.diffusion_steps を追加した状態は維持
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion, args.diffusion_steps)

    logger.log("creating data loader...")
    if args.dataset == 'brats':
        ds = BRATSVolumes(args.data_dir, mode=args.mode, gen_type=None)
        sampler = DistributedSampler(
            ds,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank()
        )
        data = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            sampler=sampler,
            drop_last=True,
        )
    else:
        raise ValueError('Dataset {} not implemented'.format(args.dataset))

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        in_channels=args.in_channels,
        image_size=args.image_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        resume_step=args.resume_step,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        dataset=args.dataset,
        summary_writer=args.summary_writer,
        mode=args.mode,
        loss_level=args.loss_level,
        contr=None
    ).run_loop()


if __name__ == "__main__":
    main()