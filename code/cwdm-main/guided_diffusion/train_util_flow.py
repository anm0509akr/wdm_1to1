# guided_diffusion/train_util_flow.py
import os
from dataclasses import dataclass
from typing import Optional
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .losses_flow import compute_cfm_loss
from guided_diffusion.train_logger import TrainLogger


def save_ckpt(path: str, model, opt, epoch: int, step: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch, "step": step},
        path,
    )


@dataclass
class TrainConfig:
    save_dir: str = "./runs"           # ★ 追加：ロガー/ckpt の保存先
    lr: float = 1e-4
    # --- iteration-first ---
    max_steps: Optional[int] = None    # 総ステップ数（これがあれば優先）
    log_interval: int = 100            # 何 step ごとにログ
    ckpt_interval: int = 10_000        # 何 step ごとに ckpt
    lr_anneal_steps: int = 0           # >0 なら線形に 0 までアニーリング
    # --- epoch fallback (互換用途) ---
    epochs: int = 0                    # max_steps が無い場合のみ使用
    sigma: float = 0.0
    subband_weight: Optional[torch.Tensor] = None


class TrainLoopFlow:
    def __init__(self, model, optimizer, loader: DataLoader, device, cfg: TrainConfig):
        self.model = model.to(device)
        self.opt = optimizer
        self.loader = loader
        self.device = device
        self.cfg = cfg
        self.logger = TrainLogger(save_dir=cfg.save_dir)  # ★ CSV/TensorBoard
        self.global_step = 0

    @torch.no_grad()
    def _grad_norm(self) -> float:
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                total += float(g.pow(2).sum())
        return total ** 0.5

    def run(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.model.train()

        sbw = self.cfg.subband_weight
        if sbw is not None:
            sbw = sbw.to(self.device)

        # =============== epoch 互換（max_steps 未指定時のみ） ===============
        max_steps = self.cfg.max_steps
        if max_steps is None or max_steps <= 0:
            for epoch in range(1, max(1, self.cfg.epochs) + 1):
                pbar = tqdm(self.loader, desc=f"Epoch {epoch}", leave=False)
                for it, batch in enumerate(pbar):
                    loss_val, grad_norm = self._train_step(batch, sbw)  # ★ 2値を受け取る
                    if self.global_step % self.cfg.log_interval == 0:
                        lr = self.opt.param_groups[0]["lr"]
                        self.logger.log(step=self.global_step, loss=loss_val, lr=lr, grad_norm=grad_norm)
                        pbar.set_postfix(loss=f"{loss_val:.6f}", lr=f"{lr:.2e}", gn=f"{grad_norm:.2f}")
                # epoch ごとに ckpt
                save_ckpt(os.path.join(save_dir, f"epoch{epoch}.pt"), self.model, self.opt, epoch, self.global_step)
                save_ckpt(os.path.join(save_dir, "last.pt"), self.model, self.opt, epoch, self.global_step)
            self.logger.close()
            return

        # =============== iteration-first ===============
        it_loader = iter(self.loader)
        pbar = tqdm(total=max_steps, desc="Training (steps)", leave=True)
        while self.global_step < max_steps:
            try:
                batch = next(it_loader)
            except StopIteration:
                it_loader = iter(self.loader)
                batch = next(it_loader)

            loss_val, grad_norm = self._train_step(batch, sbw)  # ★ 2値を受け取る

            if self.global_step % self.cfg.log_interval == 0:
                lr = self.opt.param_groups[0]["lr"]
                self.logger.log(step=self.global_step, loss=loss_val, lr=lr, grad_norm=grad_norm)
                pbar.set_postfix(loss=f"{loss_val:.6f}", lr=f"{lr:.2e}", gn=f"{grad_norm:.2f}")

            # ckpt（step 基準）
            if self.global_step > 0 and self.global_step % self.cfg.ckpt_interval == 0:
                save_ckpt(os.path.join(save_dir, f"step{self.global_step}.pt"), self.model, self.opt, 0, self.global_step)
                save_ckpt(os.path.join(save_dir, "last.pt"), self.model, self.opt, 0, self.global_step)

            pbar.update(1)

        pbar.close()
        # 終了時も last を更新
        save_ckpt(os.path.join(save_dir, "last.pt"), self.model, self.opt, 0, self.global_step)
        self.logger.close()  # ★ flush

    def _train_step(self, batch, sbw):
        x_gt = batch["target"].to(self.device)
        cond = batch.get("cond", None)
        if cond is not None:
            cond = cond.to(self.device)

        out = compute_cfm_loss(self.model, x1=x_gt, cond=cond, sigma=self.cfg.sigma, sb_weight=sbw)
        loss = out["loss"]

        # --- lr anneal（線形） ---
        if self.cfg.lr_anneal_steps and self.cfg.lr_anneal_steps > 0:
            frac = max(0.0, 1.0 - self.global_step / float(self.cfg.lr_anneal_steps))
            for pg in self.opt.param_groups:
                pg["lr"] = self.cfg.lr * frac

        # --- 逆伝播/更新 ---
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = self._grad_norm()  # ★ ここで勾配ノルムを測る（clip前でも後でもOK）
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()

        self.global_step += 1
        return float(loss.item()), float(grad_norm)  # ★ loss と grad_norm を返す
