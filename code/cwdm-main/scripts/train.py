"""
A script for training a diffusion model for paired image-to-image translation.
"""

import argparse
import numpy as np
import random
import sys
import torch as th

sys.path.append(".")
sys.path.append("..")

from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
# 'create_model_and_diffusion' は不要なので削除
from guided_diffusion.script_util import (create_model, model_and_diffusion_defaults, 
                                          args_to_dict, add_dict_to_argparser)
from guided_diffusion.train_util import TrainLoop
from guided_diffusion.bratsloader import BRATSVolumes
from torch.utils.tensorboard import SummaryWriter

from guided_diffusion.flow_matching import FlowMatching

def main():
    args = create_argparser().parse_args()
    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    summary_writer = None
    if args.use_tensorboard:
        logdir = None
        if args.tensorboard_path:
            logdir = args.tensorboard_path
        summary_writer = SummaryWriter(log_dir=logdir)
        summary_writer.add_text(
            'config',
            '\n'.join([f'--{k}={repr(v)} <br/>' for k, v in vars(args).items()])
        )
        logger.configure(dir=summary_writer.get_logdir())
    else:
        logger.configure()

    dist_util.setup_dist(devices=args.devices)

    logger.log("Creating model and flow matching...")
    
    # --- ⬇️ ここがエラーを根本的に解決する修正箇所です ⬇️ ---

    # 1. create_modelが受け取れる引数のリスト（ホワイトリスト）を定義します
    model_args_keys = [
        'image_size', 'in_channels', 'num_channels', 'out_channels', 
        'num_res_blocks', 'attention_resolutions', 'dropout', 'channel_mult', 
        'use_checkpoint', 'use_fp16', 'num_heads', 'num_head_channels', 
        'use_scale_shift_norm', 'resblock_updown', 'use_new_attention_order', 
        'dims', 'num_groups', 'renormalize', 'additive_skips', 'use_freq', 
        'bottleneck_attention', 'resample_2d'
    ]
    
    # 2. 全ての引数を含む辞書を作成します
    all_args_dict = args_to_dict(args, model_and_diffusion_defaults().keys())
    
    # 3. ホワイトリストに含まれるキーだけを抽出して、モデル用の引数辞書を作成します
    model_kwargs = {key: all_args_dict[key] for key in model_args_keys if key in all_args_dict}
    
    # 4. これで、クリーンになった引数でU-Netモデルを作成できます
    model = create_model(**model_kwargs)

    # --- ⬆️ 修正はここまで ⬆️ ---

    model.to(dist_util.dev())
    
    # FlowMatchingクラスのインスタンスを作成
    fm = FlowMatching(num_timesteps=args.diffusion_steps)
    
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, fm, maxt=args.diffusion_steps)


    if args.dataset == 'brats':
        ds = BRATSVolumes(args.data_dir, mode='train')

    datal = th.utils.data.DataLoader(ds,
                                      batch_size=args.batch_size,
                                      num_workers=args.num_workers,
                                      shuffle=True,)

    logger.log("Start training...")
    TrainLoop(
        model=model,
        flow_matching=fm,
        data=datal,
        schedule_sampler=schedule_sampler,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        resume_step=args.resume_step,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        dataset=args.dataset,
        summary_writer=summary_writer,
        contr=args.contr,
    ).run_loop()

def create_argparser():
    # この関数は変更なしでOK
    defaults = dict(
        seed=0,
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',
        resume_step=0,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset='brats',
        use_tensorboard=True,
        tensorboard_path='',
        devices=[0],
        dims=3,
        learn_sigma=False,
        num_groups=32,
        channel_mult="1,2,2,4,4",
        in_channels=16,
        out_channels=8,
        bottleneck_attention=False,
        num_workers=0,
        renormalize=True,
        additive_skips=False,
        use_freq=False,
        contr='t1n',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()