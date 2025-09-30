# guided_diffusion/train_util_flow.py
import os
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .losses_flow import compute_cfm_loss  # band別メトリクスを返す実装を想定
from guided_diffusion.train_logger import TrainLogger  # CSV/TB ロガー


def save_ckpt(path: str, model: torch.nn.Module, opt: torch.optim.Optimizer,
              epoch: int, step: int, extra: Optional[Dict[str, Any]] = None):
    """
    チェックポイントを保存。model/opt の state_dict に加え、epoch/step/extra も保存。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "epoch": epoch,
        "step": step,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


@dataclass
class TrainConfig:
    # 学習率
    lr: float = 1e-4
    # iteration-first
    max_steps: Optional[int] = None          # 総更新回数（指定時はこちら優先）
    log_interval: int = 100                  # 何 step ごとにログ表示/記録
    ckpt_interval: int = 10_000              # 何 step ごとに ckpt 保存
    lr_anneal_steps: int = 0                 # >0 なら線形に 0 までアニーリング
    # epoch fallback（max_steps 未指定時のみ有効）
    epochs: int = 0
    # Flow Matching オプション
    sigma: float = 0.0                       # CFM ノイズ量
    subband_weight: Optional[torch.Tensor] = None  # [8] or [8*C] の重み
    # 出力先
    save_dir: str = "./runs/flow"


class TrainLoopFlow:
    """
    iteration-first（max_steps）を基本に回す学習ループ。
    - compute_cfm_loss() が返す 'loss' に対して backward
    - 学習曲線は CSV/TensorBoard に記録（TrainLogger 経由）
    - サブバンド別メトリクス（mse/ene/cos）があればログに含める
    """
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 loader: DataLoader, device: torch.device, cfg: TrainConfig):
        self.model = model.to(device)
        self.opt = optimizer
        self.loader = loader
        self.device = device
        self.cfg = cfg

        os.makedirs(self.cfg.save_dir, exist_ok=True)
        self.logger = TrainLogger(save_dir=self.cfg.save_dir)  # CSV: train_log.csv, TB: tb/
        self.global_step = 0
        self.epoch = 0

        # subband weight をデバイスへ
        self._sbw = None
        if cfg.subband_weight is not None:
            self._sbw = cfg.subband_weight.detach().clone()

    @torch.no_grad()
    def _grad_norm(self) -> float:
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                total += float(torch.sum(g * g).item())
        return total ** 0.5

    def _apply_lr_anneal(self):
        """
        線形アニーリング：
          lr_anneal_steps > 0 のとき、step=0 で lr、step=lr_anneal_steps で 0 になるよう線形に低下
        """
        if self.cfg.lr_anneal_steps and self.cfg.lr_anneal_steps > 0:
            frac = max(0.0, 1.0 - self.global_step / float(self.cfg.lr_anneal_steps))
            for pg in self.opt.param_groups:
                pg["lr"] = self.cfg.lr * frac

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        1 step 学習。
        batch 仕様（collate 済み）：
          - batch['target']: [B, 8*C, D/2, H/2, W/2] もしくは 空間 [B,1,D,H,W]（loss側で DWT する実装に合わせる）
          - batch['cond']  : wavelet/空間いずれも OK（loss 側が dict/Tensor 両対応）
        """
        x_gt = batch["target"].to(self.device)
        cond = batch.get("cond", None)
        if cond is not None:
            # cond は Tensor でも dict でも可（loss が処理）
            if isinstance(cond, torch.Tensor):
                cond = cond.to(self.device)
            elif isinstance(cond, dict):
                # dict なら内部テンソルを to(device)（必要なもののみでもOK）
                cond = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in cond.items()}

        sbw = self._sbw.to(self.device) if self._sbw is not None else None

        # ---- Forward & Loss ----
        out = compute_cfm_loss(self.model, x1=x_gt, cond=cond,
                               sigma=self.cfg.sigma, sb_weight=sbw)
        loss: torch.Tensor = out["loss"]

        # ---- Backward ----
        self._apply_lr_anneal()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()

        # ---- ログ値の収集 ----
        scalars = {
            "loss": float(loss.item()),
            "lr": float(self.opt.param_groups[0]["lr"]),
            "grad_norm": float(self._grad_norm()),
        }
        # band 別メトリクス（あれば）
        for k, v in out.items():
            if k == "loss":
                continue
            # float 化できるものだけ拾う（Tensor なら .item()）
            if isinstance(v, (float, int)):
                scalars[k] = float(v)
            elif torch.is_tensor(v) and v.dim() == 0:
                scalars[k] = float(v.item())

        self.global_step += 1
        return scalars

    def _log_step(self, pbar: tqdm, scalars: Dict[str, Any]):
        """
        tqdm 表示 + CSV/TensorBoard へ記録
        """
        # tqdm 表示（主要指標のみ短く）
        pbar.set_postfix_str(
            f"loss={scalars.get('loss', float('nan')):.6f}, "
            f"lr={scalars.get('lr', float('nan')):.2e}, "
            f"gn={scalars.get('grad_norm', float('nan')):.2f}"
        )

        # CSV/TB 記録
        self.logger.log_step(
            step=self.global_step,
            elapsed_sec=time.time() - self.logger.t0,
            scalars=scalars,
        )

    def run(self, save_dir: Optional[str] = None):
        """
        学習本体。
        max_steps が与えられていれば iteration-first、
        無ければ epochs を回す（後方互換）。
        """
        if save_dir is None:
            save_dir = self.cfg.save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.model.train()

        # ---- iteration-first ----
        if self.cfg.max_steps is not None and self.cfg.max_steps > 0:
            it_loader = iter(self.loader)
            total = int(self.cfg.max_steps)
            pbar = tqdm(total=total, desc="Training (steps)", leave=True)
            while self.global_step < total:
                try:
                    batch = next(it_loader)
                except StopIteration:
                    it_loader = iter(self.loader)
                    batch = next(it_loader)

                scalars = self._train_step(batch)

                if self.global_step % self.cfg.log_interval == 0:
                    self._log_step(pbar, scalars)

                # ckpt
                if self.global_step > 0 and self.global_step % self.cfg.ckpt_interval == 0:
                    save_ckpt(os.path.join(save_dir, f"step{self.global_step}.pt"),
                              self.model, self.opt, self.epoch, self.global_step)
                    save_ckpt(os.path.join(save_dir, "last.pt"),
                              self.model, self.opt, self.epoch, self.global_step)

                pbar.update(1)

            pbar.close()
            # 終了時にも last 更新
            save_ckpt(os.path.join(save_dir, "last.pt"),
                      self.model, self.opt, self.epoch, self.global_step)
            return

        # ---- epoch fallback（max_steps 未指定時のみ）----
        epochs = max(1, int(self.cfg.epochs))
        for ep in range(1, epochs + 1):
            self.epoch = ep
            pbar = tqdm(self.loader, desc=f"Epoch {ep}", leave=False)
            for batch in pbar:
                scalars = self._train_step(batch)
                if self.global_step % self.cfg.log_interval == 0:
                    self._log_step(pbar, scalars)

            # epoch ごとに ckpt を落とす
            save_ckpt(os.path.join(save_dir, f"epoch{ep}.pt"),
                      self.model, self.opt, ep, self.global_step)
            save_ckpt(os.path.join(save_dir, "last.pt"),
                      self.model, self.opt, ep, self.global_step)
