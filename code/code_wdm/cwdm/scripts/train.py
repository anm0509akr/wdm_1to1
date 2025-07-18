"""
A script for training a diffusion model for paired image-to-image translation.
"""
import os
import argparse
import numpy as np
import random
import sys
import torch 
import torch.distributed as dist
import torch.utils.data

sys.path.append(".")
sys.path.append("..")

from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (model_and_diffusion_defaults, create_model_and_diffusion,
                                          args_to_dict, add_dict_to_argparser)
from guided_diffusion.train_util import TrainLoop
from guided_diffusion.bratsloader import BRATSVolumes
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    args = create_argparser().parse_args()
    if not hasattr(args, "local_rank") or args.local_rank == 0:
        print(f"[DEBUG] num_workers = {args.num_workers}")
    seed = args.seed
    torch.manual_seed(seed)
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

    logger.log("Creating model and diffusion...")
    arguments = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(**arguments)

    # logger.log("Number of trainable parameters: {}".format(np.array([np.array(p.shape).prod() for p in model.parameters()]).sum()))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    model = model.to(torch.device(f"cuda:{local_rank}"))
    model = DDP(model,
            device_ids=[local_rank]
            )
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=1000)

    if args.dataset == 'brats':
        ds = BRATSVolumes(args.data_dir, mode='train')

    per_gpu_bs = max(1, args.batch_size // dist.get_world_size())
    sampler = DistributedSampler(
        ds,
        shuffle=True,
        rank=dist_util.get_rank(),
        num_replicas=dist.get_world_size(),
    )
    # DataLoader は 1 つで OK
    loader = torch.utils.data.DataLoader(
        ds,
        sampler=sampler,
        batch_size=per_gpu_bs,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    

    logger.log("Start training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=loader,
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
        summary_writer=summary_writer,
        mode='i2i',
        contr=args.contr,
        local_logdir=args.local_logdir,
    ).run_loop()


def create_argparser():
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
        tensorboard_path='',  # set path to existing logdir for resuming
        devices=[0],
        dims=3,
        learn_sigma=False,
        num_groups=32,
        channel_mult="1,2,2,4,4",
        in_channels=8,
        out_channels=8,
        bottleneck_attention=False,
        num_workers=0,
        mode='default',
        renormalize=True,
        additive_skips=False,
        use_freq=False,
        contr='t1n',
        local_logdir='./logs',
        ca_down_factor=1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    
    
    
    return parser


if __name__ == "__main__":
    main()
