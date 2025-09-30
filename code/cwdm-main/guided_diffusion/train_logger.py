# guided_diffusion/train_logger.py
from __future__ import annotations
import os, csv, time
from typing import Optional

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None  # TensorBoard が無ければ無視

class TrainLogger:
    """
    学習中のメトリクスを CSV と TensorBoard に記録する軽量ロガー。
    - CSV: save_dir/train_log.csv
    - TensorBoard: save_dir/tb/ （torch.utils.tensorboard があれば）
    """
    def __init__(self, save_dir: str, csv_name: str = "train_log.csv", tb_subdir: str = "tb"):
        os.makedirs(save_dir, exist_ok=True)
        self.csv_path = os.path.join(save_dir, csv_name)
        self.start_time = time.time()
        self._init_csv()

        self.tb = None
        if SummaryWriter is not None:
            try:
                self.tb = SummaryWriter(log_dir=os.path.join(save_dir, tb_subdir))
            except Exception:
                self.tb = None  # 書けなければ無視

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "elapsed_sec", "loss", "lr", "grad_norm"])

    def log(self, step: int, loss: float, lr: float, grad_norm: Optional[float] = None):
        elapsed = time.time() - self.start_time
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([step, elapsed, float(loss), float(lr), float(grad_norm or 0.0)])

        if self.tb is not None:
            try:
                self.tb.add_scalar("train/loss", loss, step)
                self.tb.add_scalar("train/lr", lr, step)
                if grad_norm is not None:
                    self.tb.add_scalar("train/grad_norm", grad_norm, step)
            except Exception:
                pass

    def close(self):
        if self.tb is not None:
            try:
                self.tb.flush()
                self.tb.close()
            except Exception:
                pass
