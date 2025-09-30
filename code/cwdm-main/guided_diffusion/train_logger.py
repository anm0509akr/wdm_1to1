# guided_diffusion/train_logger.py
import csv, os, time
from typing import Dict, Optional
import torch

class TrainLogger:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.t0 = time.time()
        self.csv_path = os.path.join(self.save_dir, "train_log.csv")
        self._csv_file = None
        self._csv_writer = None
        self.tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(self.save_dir, "tb")
            os.makedirs(tb_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=tb_dir)
        except Exception:
            pass

    def _ensure_csv(self, fieldnames):
        # 毎回 fieldnames が変わっても安全に動くよう簡易に作り直す
        if self._csv_writer is None:
            write_header = not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0
            self._csv_file = open(self.csv_path, "a", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
            if write_header:
                self._csv_writer.writeheader()
        else:
            # 既存 writer のヘッダと違う場合は作り直す
            if set(self._csv_writer.fieldnames) != set(fieldnames):
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None
                self._ensure_csv(fieldnames)

    @staticmethod
    def _flatten_scalars(scalars: Dict, prefix: str = "") -> Dict[str, float]:
        """dict/テンソルをフラット化して {str: float} のみにする。非スカラは無視。"""
        out: Dict[str, float] = {}
        for k, v in scalars.items():
            key = f"{prefix}{k}"
            if isinstance(v, (int, float)):
                out[key] = float(v)
            elif isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    out[key] = float(v.item())
                # 多要素テンソルはログしない
            elif isinstance(v, dict):
                out.update(TrainLogger._flatten_scalars(v, prefix=key + "/"))
            else:
                try:
                    out[key] = float(v)  # 変換できれば採用
                except Exception:
                    pass  # それ以外は無視
        return out

    def log_step(self, step: int, **scalars) -> None:
        # フラット化＆数値化
        flat = {"step": float(step)}
        flat.update(self._flatten_scalars(scalars))

        # CSV
        fieldnames = list(flat.keys())
        self._ensure_csv(fieldnames)
        self._csv_writer.writerow(flat)
        self._csv_file.flush()

        # TensorBoard
        if self.tb_writer is not None:
            for k, v in flat.items():
                if k == "step":
                    continue
                try:
                    self.tb_writer.add_scalar(k, v, global_step=step)
                except Exception:
                    pass

    def close(self):
        if self._csv_file is not None:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception:
                pass
            self._csv_file = None
        if self.tb_writer is not None:
            try:
                self.tb_writer.flush()
                self.tb_writer.close()
            except Exception:
                pass
            self.tb_writer = None
