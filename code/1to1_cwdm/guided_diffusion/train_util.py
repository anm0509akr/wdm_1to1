import functools
import os
import copy
import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.optim import AdamW
import torch.cuda.amp as amp

# tqdmをインポート
from tqdm import tqdm

from . import dist_util, logger
from .resample import LossAwareSampler, UniformSampler

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        resume_step,
        use_fp16,
        fp16_scale_growth,
        schedule_sampler,
        weight_decay,
        lr_anneal_steps,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.iterdatal = iter(data)
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = resume_step
        self.resume_step = resume_step
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()
        self.device = dist_util.dev()

        self._load_and_sync_parameters()
        self.grad_scaler = amp.GradScaler(enabled=self.use_fp16)
        
        self.ema_params = [
            copy.deepcopy(list(self.model.parameters())) for _ in range(len(self.ema_rate))
        ]
        
        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        if resume_step:
            self._load_optimizer_state()
            for rate, params in zip(self.ema_rate, self.ema_params):
                self._load_ema_parameters(rate)

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=self.device
                    )
                )
        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(list(self.model.parameters()))
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"ema_{rate}_{(self.resume_step):06d}.pt"
        )
        if bf.exists(ema_checkpoint) and dist.get_rank() == 0:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=self.device
            )
            ema_params = [state_dict[f"arr_{i}"] for i in range(len(ema_params))]

        dist_util.sync_params(ema_params)
        self.ema_params[self.ema_rate.index(rate)] = ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt_{(self.resume_step):06d}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=self.device
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        pbar = tqdm(range(self.lr_anneal_steps - self.step), initial=self.step, total=self.lr_anneal_steps, dynamic_ncols=True)
        for _ in pbar:
            try:
                batch, cond = next(self.iterdatal)
            except StopIteration:
                self.iterdatal = iter(self.data)
                batch, cond = next(self.iterdatal)
            
            loss_dict = self.run_step(batch, cond)
            
            if dist.get_rank() == 0:
                pbar.set_postfix({"loss": loss_dict["loss"].item()})

            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
            
            self.step += 1
            if self.step >= self.lr_anneal_steps:
                break
        
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        loss_dict = self.forward_backward(batch, cond)
        self.grad_scaler.unscale_(self.opt)
        self.opt.step()
        self._update_ema()
        self.grad_scaler.update()
        self._anneal_lr()
        self.log_step()
        return loss_dict

    def forward_backward(self, batch, cond):
        self.opt.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro_batch = batch[i : i + self.microbatch].to(self.device)
            micro_cond = {
                k: v[i : i + self.microbatch].to(self.device) for k, v in cond.items()
            }
            
            t, weights = self.schedule_sampler.sample(micro_batch.shape[0], self.device)

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.model,
                micro_batch,
                t,
                model_kwargs=micro_cond,
            )

            with amp.autocast(enabled=self.use_fp16):
                losses = compute_losses()

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v.detach() for k, v in losses.items()}
            )
            self.grad_scaler.scale(loss).backward()
        return losses

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            for p_main, p_ema in zip(self.model.parameters(), params):
                p_ema.copy_(p_main.lerp(p_ema, rate))

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = self.step / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step)
        logger.logkv("samples", (self.step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = {}
            for i, p in enumerate(params):
                state_dict[f"arr_{i}"] = p
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(self.model.state_dict(), f)

        save_checkpoint(0, self.model.parameters())
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)
        
        dist.barrier()


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


# ================================================================
# ★★★ ここからが script_util.py から移動してきたヘルパー関数群 ★★★
# ================================================================

def get_blob_logdir():
    return logger.get_dir()

def find_resume_checkpoint():
    return None

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = os.path.basename(filename)
    split = split.split(".")[-2]
    split = split.split("_")[-1]
    reversed_split = []
    for c in reversed(split):
        if not c.isdigit():
            break
        reversed_split.append(c)
    split = ''.join(reversed(reversed_split))
    split = ''.join(c for c in split if c.isdigit())
    try:
        return int(split)
    except ValueError:
        return 0