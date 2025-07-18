# code/cwdm-main/guided_diffusion/train_util.py

import functools
import os
import copy
import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.optim import AdamW
import torch.cuda.amp as amp

from tqdm import tqdm

from . import dist_util, logger
from .resample import LossAwareSampler, UniformSampler
# ★★★ モデルの引数を辞書に変換するために必要な関数をインポート ★★★
from .script_util import args_to_dict, model_and_diffusion_defaults
from DWT_IDWT.DWT_IDWT_layer import DWT_3D

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        flow_matching, # 以前のdiffusionから変更
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
        dataset,      # train.pyから渡される引数
        summary_writer,
        contr,
        args,         # ★★★【修正点1】args を引数に追加 ★★★
    ):
        self.model = model
        self.flow_matching = flow_matching # flow_matchingを保持
        self.data = data
        self.iterdatal = iter(self.data)
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
        self.schedule_sampler = schedule_sampler or UniformSampler(flow_matching)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.args = args # ★★★【修正点2】self.args として引数を保持 ★★★

        self.step = resume_step
        self.resume_step = resume_step
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()
        self.device = dist_util.dev()

        self.dwt = DWT_3D("haar").to(self.device)

        self._load_and_sync_parameters()
        self.grad_scaler = amp.GradScaler(enabled=self.use_fp16)
        
        self.ema_params = [
            copy.deepcopy(list(self.model.parameters())) for _ in range(len(self.ema_rate))
        ]
        
        self.opt = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # チェックポイントからの再開処理
        if self.resume_step:
            self._load_optimizer_state()
            for rate, params in zip(self.ema_rate, self.ema_params):
                self._load_ema_parameters(rate)

    def _load_and_sync_parameters(self):
        # この関数はチェックポイントからモデルの重みのみを読み込む
        # 引数は読み込まないため、新規学習時はコマンドライン引数が使われる
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                # サンプリング時とは異なり、学習再開時は重みのみを読み込む
                # 引数が変わっている場合は、新しい構造のモデルに読み込もうとする
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=self.device
                    )['model_state_dict'] # 辞書構造からの読み込みに変更
                )
        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        # EMAの再開ロジック (変更なし)
        # ...
        pass

    def _load_optimizer_state(self):
        # オプティマイザの再開ロジック (変更なし)
        # ...
        pass
        
    def run_loop(self):
        # tqdmを使った学習ループ (変更なし)
        pbar = tqdm(range(self.lr_anneal_steps - self.step), initial=self.step, total=self.lr_anneal_steps, dynamic_ncols=True)
        for _ in pbar:
            try:
                # Flow-Matchingではcondは不要かもしれないが、元のデータローダ構造を維持
                batch, cond = next(self.iterdatal)
            except StopIteration:
                self.iterdatal = iter(self.data)
                batch, cond = next(self.iterdatal)
            
            self.run_step(batch, cond)
            
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
        self.forward_backward(batch, cond)
        self.grad_scaler.unscale_(self.opt)
        self.opt.step()
        self._update_ema()
        self.grad_scaler.update()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        """
        最も重要な変更箇所です。
        この関数全体を以下の内容に置き換えてください。
        """
        self.opt.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # 1. 生の画像データのミニバッチを取得します
            micro_batch_img = batch[i : i + self.microbatch].to(self.device)
            # 条件画像は辞書に入っているので、'cond_1'キーでアクセスします
            micro_cond_img = cond['cond_1'][i : i + self.microbatch].to(self.device)
            
            # 2. ターゲットと条件の両方をWavelet変換します
            with th.no_grad():
                # ターゲット画像 (例: t1c) -> Wavelet係数
                target_wav = th.cat(self.dwt(micro_batch_img), dim=1)
                # 条件画像 (例: t1n) -> Wavelet係数
                cond_wav = th.cat(self.dwt(micro_cond_img), dim=1)

            # 3. Flow-Matchingのためにタイムステップとデータを準備します
            t, weights = self.schedule_sampler.sample(target_wav.shape[0], self.device)
            
            # フローの始点: Wavelet係数と同じ形状のランダムノイズ
            x_0_wav = th.randn_like(target_wav)
            # フローの終点: ターゲットのWavelet係数
            x_1_wav = target_wav
            # モデルへの条件: 条件画像のWavelet係数
            model_kwargs_wav = {"cond_wav": cond_wav}

            # 4. 正しいWavelet領域のデータを使って損失関数を呼び出します
            compute_losses = functools.partial(
                self.flow_matching.training_losses,
                model=self.model,
                x_0=x_0_wav,
                x_1=x_1_wav,
                t=t,
                model_kwargs=model_kwargs_wav,
            )

            with amp.autocast(enabled=self.use_fp16):
                losses = compute_losses()

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.flow_matching, t, {k: v.detach() for k, v in losses.items()}
            )
            self.grad_scaler.scale(loss).backward()

    def _update_ema(self):
        # 💡【ここが最後の修正点です】💡
        # EMAの更新処理全体を`with th.no_grad():`で囲みます。
        # これにより、このブロック内の計算では勾配が追跡されなくなり、
        # in-place操作が安全に実行できるようになります。
        with th.no_grad():
            for rate, params in zip(self.ema_rate, self.ema_params):
                for p_main, p_ema in zip(self.model.parameters(), params):
                    p_ema.copy_(p_main.lerp(p_ema, rate))
        # --- ✅ 修正完了 ✅ ---

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
        """
        モデルの引数、重み、オプティマイザの状態、EMAの重みを保存する。
        """
        if dist.get_rank() != 0:
            return

        def _save_checkpoint(data, filename):
            """ヘルパー関数でチェックポイントを保存"""
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(data, f)

        logger.log(f"saving model and optimizer state at step {self.step}...")

        # 1. メインのモデルとオプティマイザの状態を保存
        model_args_dict = args_to_dict(self.args, model_and_diffusion_defaults().keys())
        save_data = {
            'model_args': model_args_dict,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'step': self.step,
            'args': self.args, # デバッグや完全な再現のために全引数も保存
        }
        _save_checkpoint(save_data, f"model_{(self.step):06d}.pt")

        # 2. EMAモデルの重みを保存
        for rate, params in zip(self.ema_rate, self.ema_params):
            # 現在のモデルのstate_dictをコピーして、EMAパラメータで上書き
            ema_state_dict = copy.deepcopy(self.model.state_dict())
            for name, p_ema in zip(ema_state_dict.keys(), params):
                # .data を使ってテンソルの実体にアクセス
                ema_state_dict[name].copy_(p_ema.data)
            
            # EMAモデルは引数を持たないので、state_dictのみを保存
            _save_checkpoint(
                {"model_state_dict": ema_state_dict, "model_args": model_args_dict},
                f"ema_{rate}_{(self.step):06d}.pt"
            )
            
        logger.log("save complete.")
        dist.barrier() # 他のプロセスと同期

def log_loss_dict(process, ts, losses): # diffusionからprocessに変更
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # The term 'num_timesteps' may not exist in FlowMatching.
        # Adjust if necessary.
        if hasattr(process, 'num_timesteps'):
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / process.num_timesteps)
                logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

# ヘルパー関数 (変更なし)
def get_blob_logdir():
    return logger.get_dir()

def find_resume_checkpoint():
    return None

def parse_resume_step_from_filename(filename):
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