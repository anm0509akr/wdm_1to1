import torch
import torch.nn as nn
from torchdiffeq import odeint

class FlowMatching(nn.Module):
    def __init__(self, num_timesteps=1000):
        super().__init__()
        self.num_timesteps = num_timesteps



    def get_vector_field(self, x_0, x_1):
        """
        データペア (x_0, x_1) から真のベクトル場を計算します。
        最も単純な線形補間の場合、ベクトル場は x_1 - x_0 となります。
        """
        return x_1 - x_0

    def training_losses(self, model, x_0, x_1, t, model_kwargs):
        """
        Flow-Matchingの損失を計算します。
        - model: U-Netモデル
        - x_0: フローの開始点 (ノイズのWavelet係数)
        - x_1: フローの終了点 (ターゲットt1cのWavelet係数)
        - t: 時間 (0から1)
        - model_kwargs: {'cond': 条件t1nのWavelet係数}
        """
        # 1. t に従って x_t を生成 (線形補間)
        t_reshaped = t.view(-1, 1, 1, 1, 1) # 5Dテンソルに対応
        x_t = (1 - t_reshaped) * x_0 + t_reshaped * x_1

        # 2. 真のベクトル場を計算
        true_vector_field = self.get_vector_field(x_0, x_1)

        # --- ⬇️ ここがエラーを解決する最後の修正です ⬇️ ---

        # 3. U-Netモデルへの入力を作成します。
        #    現在の画像 x_t と条件画像のWavelet係数 model_kwargs['cond'] を
        #    チャンネル次元(dim=1)で結合します。
        model_input = torch.cat([x_t, model_kwargs['cond']], dim=1)
        
        # 4. 結合したテンソルと時間tをモデルに渡して、ベクトル場を予測します。
        predicted_vector_field = model(model_input, t)
        
        # --- ⬆️ 修正はここまで ⬆️ ---

        # 5. 損失（MSE）を計算
        loss = nn.functional.mse_loss(
            predicted_vector_field, true_vector_field, reduction='none'
        )
        loss = loss.mean(dim=list(range(1, len(loss.shape))))

        return {"loss": loss}

    @torch.no_grad()
    def sample_loop(self, shape, cond, device='cuda'):
        """
        学習済みモデルから画像を生成します。
        """
        # 初期状態 (例: 標準正規分布からのサンプリング)
        x_t = torch.randn(shape, device=device)

        # ODEソルバーを使って t=0 から t=1 まで積分する
        # ここでは、タイムステップを離散化して単純なオイラー法で近似する例を示す
        # より高精度なソルバー（例: torchdiffeq.odeint）の使用を推奨
        time_steps = torch.linspace(0, 1, 1000).to(device) # 例として1000ステップ
        for i in range(len(time_steps) - 1):
            t_start, t_end = time_steps[i], time_steps[i+1]
            dt = t_end - t_start
            
            # 現在の時刻 t でベクトル場を予測
            v_t = model(x_t, t_start.expand(x_t.size(0)), cond)
            
            # オイラー法で次のステップの画像を計算
            x_t = x_t + v_t * dt

        return x_t # 生成された画像 x_1