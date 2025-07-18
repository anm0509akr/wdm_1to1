# code/cwdm-main/guided_diffusion/flow_matching.py

import torch
import torch.nn as nn
from tqdm import tqdm

class FlowMatching(nn.Module):
    def __init__(self, num_timesteps=1000):
        super().__init__()
        self.num_timesteps = num_timesteps

    def get_vector_field(self, x_0, x_1):
        """
        最も単純な線形補間の場合、ベクトル場は x_1 - x_0 となります。
        """
        return x_1 - x_0

    def training_losses(self, model, x_0, x_1, t, model_kwargs=None):
        """
        Flow-Matchingの損失を計算します。
        """
        if model_kwargs is None:
            model_kwargs = {}

        cond_wavelet = model_kwargs.get("cond_wav")
        if cond_wavelet is None:
            raise ValueError("条件Wavelet 'cond_wav' が model_kwargs の中に見つかりません")

        t_reshaped = t.view(-1, 1, 1, 1, 1)
        x_t = (1 - t_reshaped) * x_0 + t_reshaped * x_1

        true_vector_field = self.get_vector_field(x_0, x_1)
        model_input = torch.cat([x_t, cond_wavelet], dim=1)
        predicted_vector_field = model(model_input, t)
        
        # --- 💡【ここが最後の修正点です】💡 ---

        # 1. reduction='none' を指定して、損失をピクセルごと、サンプルごとに計算します。
        #    これにより、loss_per_pixelの形状は (B, C, D, H, W) のようになります。
        loss_per_pixel = nn.functional.mse_loss(
            predicted_vector_field, true_vector_field, reduction='none'
        )

        # 2. バッチ次元(dim=0)以外の全ての次元で平均を取ります。
        #    これにより、最終的なlossの形状は (B,) となり、
        #    各サンプルの平均損失が格納された1次元の配列が得られます。
        loss = loss_per_pixel.mean(dim=list(range(1, len(loss_per_pixel.shape))))
        
        # --- ✅ 修正完了 ✅ ---

        return {"loss": loss}


    @torch.no_grad()
    def sample_loop(self, model, shape, cond, device='cuda', ode_steps=100):
        """
        学習済みモデルから画像を生成します。
        """
        x_t = torch.randn(shape, device=device)
        time_steps = torch.linspace(0, 1, ode_steps + 1, device=device)
        
        for i in tqdm(range(ode_steps), desc="Sampling with Flow-Matching", leave=False):
            t_start, t_end = time_steps[i], time_steps[i+1]
            dt = t_end - t_start
            
            model_input = torch.cat([x_t, cond], dim=1)
            t_batch = t_start.expand(x_t.size(0))
            v_t = model(model_input, t_batch)
            
            x_t = x_t + v_t * dt

        return x_t