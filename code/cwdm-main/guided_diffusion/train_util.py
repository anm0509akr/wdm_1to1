import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
import torch.utils.tensorboard
from torch.optim import AdamW
import torch.cuda.amp as amp

import itertools
from tqdm import tqdm
import time
import datetime

from . import dist_util, logger
# from .resample import LossAwareSampler, UniformSampler # --- 変更点 1: 不要なインポートを削除 ---
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D

INITIAL_LOG_LOSS_SCALE = 20.0

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img

class TrainLoop:
    def __init__(
        self,
        *,
        model, # U-Netモデル
        flow_matching, # FlowMatchingオブジェクト
        schedule_sampler, # Schedule Sampler
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        resume_step,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        weight_decay=0.0,
        lr_anneal_steps=0,
        **kwargs, # その他の引数
    ):
        self.model = model
        self.flow_matching = flow_matching
        self.schedule_sampler = schedule_sampler
        self.data = data
        self.datal = data # 変数名を統一
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
        self.resume_step = resume_step
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()
        self.device = dist_util.dev()

        self._load_and_sync_parameters()
        
        # GradScalerの初期化
        self.grad_scaler = amp.GradScaler(enabled=self.use_fp16)
        
        # オプティマイザとEMAパラメータの準備
        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.ema_params = [copy.deepcopy(list(self.model.parameters())) for _ in self.ema_rate]
        
        if self.resume_step:
            self._load_optimizer_state()
            # EMAパラメータもロード
            for rate, params in zip(self.ema_rate, self.ema_params):
                self._load_ema_parameters(rate)

        # DWT/IDWT層
        self.dwt = DWT_3D('haar')
        self.idwt = IDWT_3D('haar')
        
    def _load_and_sync_parameters(self):
        # ユーザー提供のコードを尊重し、簡略化
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(resume_checkpoint, map_location=self.device)
            )
        dist_util.sync_params(self.model.parameters())

    def _load_optimizer_state(self):
        # ユーザー提供のコードを尊重
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt")
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(opt_checkpoint, map_location=self.device)
            self.opt.load_state_dict(state_dict)
            
            
    def _load_ema_parameters(self, rate):
        ema_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = bf.join(
            bf.dirname(ema_checkpoint), f"ema_{rate}_{(self.resume_step):06d}.pt"
        )
        if bf.exists(ema_checkpoint):
            logger.log(f"loading EMA state from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(ema_checkpoint, map_location=self.device)
            # self.ema_params に直接ロードする
            # この部分はstate_dictの形式に合わせて調整が必要な場合があります
            # ここでは、ema_params[i]がパラメータのリストであることを想定しています
            ema_rate_index = self.ema_rate.index(rate)
            for i, p in enumerate(self.ema_params[ema_rate_index]):
                p.data.copy_(state_dict[f"arr_{i}"])



    def run_loop(self):
        # ユーザー提供のrun_loopを尊重しつつ、ステップの進行を修正
        pbar = tqdm(range(self.lr_anneal_steps), initial=self.resume_step, dynamic_ncols=True)
        self.step = self.resume_step
        while self.step < self.lr_anneal_steps:
            try:
                batch, cond = next(self.iterdatal)
            except StopIteration:
                self.iterdatal = iter(self.datal)
                batch, cond = next(self.iterdatal)
            
            self.run_step(batch, cond)
            
            pbar.update(1)
            # pbarのloss表示はrun_step内でlog_step経由で行う
            
            if self.step % self.log_interval == 0:
                loss_avg = logger.getkvs()['loss/loss']
                pbar.set_postfix({"loss": loss_avg})
                logger.dumpkvs()

            if self.step % self.save_interval == 0:
                self.save()

            self.step += 1
            
        pbar.close()
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.grad_scaler.unscale_(self.opt)
        # th.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) # 必要に応じて勾配クリッピング
        self.grad_scaler.step(self.opt)
        self.grad_scaler.update()
        self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.opt.zero_grad()
        
        for i in range(0, batch.shape[0], self.microbatch):
            micro_batch = batch[i : i + self.microbatch].to(self.device)
            micro_cond_dict = {
                k: v[i : i + self.microbatch].to(self.device) for k, v in cond.items()
            }
            
            # --- ⬇️ ここが重要な修正箇所です ⬇️ ---
            
            # ターゲット画像 (t1c) をWavelet変換
            x_1_wav_components = self.dwt(micro_batch)
            # dwtが返す全テンソルを単純に結合します
            x_1_wav = th.cat(x_1_wav_components, dim=1)
            
            # 条件画像 (t1n) をWavelet変換
            cond_image = micro_cond_dict['cond_1']
            cond_wav_components = self.dwt(cond_image)
            # こちらも同様に結合します
            cond_wav = th.cat(cond_wav_components, dim=1)

            # --- ⬆️ 修正はここまで ⬆️ ---

            x_0_wav = th.randn_like(x_1_wav)
            t, weights = self.schedule_sampler.sample(micro_batch.shape[0], self.device)
            model_kwargs = {"cond": cond_wav}

            with amp.autocast(enabled=self.use_fp16):
                losses = self.flow_matching.training_losses(
                    model=self.model,
                    x_0=x_0_wav, 
                    x_1=x_1_wav, 
                    t=t,
                    model_kwargs=model_kwargs
                )
            
            loss = (losses["loss"] * weights).mean()
            log_loss_dict(self.flow_matching, t, losses)
            self.grad_scaler.scale(loss).backward()

            
    @th.no_grad() # 勾配計算を無効化することを明示的に指示
    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            for p_main, p_ema in zip(self.model.parameters(), params):
                # p_ema の値を直接変更せず、新しい値を計算してからコピーする
                p_ema.copy_(p_ema * rate + p_main * (1 - rate))
    
    # _anneal_lr, log_step, save メソッドはユーザー提供のものを尊重（適宜修正）
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
        # EMAモデルを保存するロジック（必要に応じて）
        def save_checkpoint(state_dict, filename):
            if dist.get_rank() == 0:
                logger.log(f"saving model to {filename}...")
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)
        
        # U-Netモデルの保存
        save_checkpoint(self.model.state_dict(), f"model_{self.step:06d}.pt")
        # オプティマイザの保存
        save_checkpoint(self.opt.state_dict(), f"opt_{self.step:06d}.pt")
        # EMAモデルの保存
        for i, rate in enumerate(self.ema_rate):
            # EMAパラメータはリストなので、state_dict形式に変換して保存
            ema_state_dict = {f"arr_{j}": p.data for j, p in enumerate(self.ema_params[i])}
            save_checkpoint(ema_state_dict, f"ema_{rate}_{self.step:06d}.pt")


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


def get_blob_logdir():
    return logger.get_dir()


def find_resume_checkpoint():
    return None


def log_loss_dict(flow_matching, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(f"loss/{key}", values.mean().item())
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / flow_matching.num_timesteps)
            logger.logkv_mean(f"loss/{key}_q{quartile}", sub_loss)