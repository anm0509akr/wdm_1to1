# code/cwdm-main/scripts/sample.py

import argparse
import nibabel as nib
import numpy as np
import os
import pathlib
import random
import sys
import torch as th
from tqdm import tqdm

# プロジェクトルートをPythonパスに追加
sys.path.append(os.getcwd())

from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSVolumes
# ★★ モデルを引数から作成するための正しい関数をインポート ★★
from guided_diffusion.script_util import (
    create_model,
    add_dict_to_argparser,
)
from DWT_IDWT.DWT_IDWT_layer import IDWT_3D, DWT_3D

@th.no_grad()
def ode_sampler(model, cond, shape, device, steps=100):
    """
    Flow-Matchingモデル用のシンプルなオイラー法ODEソルバー
    """
    x_t = th.randn(shape, device=device)
    time_steps = th.linspace(0, 1, steps + 1, device=device)

    for i in tqdm(range(steps), desc="ODE Sampling", leave=False):
        t_start, t_end = time_steps[i], time_steps[i+1]
        dt = t_end - t_start
        model_input = th.cat([x_t, cond], dim=1)
        # タイムステップはバッチサイズに合わせて拡張する
        v_t = model(model_input, t_start.expand(x_t.size(0)))
        x_t = x_t + v_t * dt
    
    return x_t

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(devices=[str(d) for d in args.devices])
    logger.configure(dir=args.output_dir)

    logger.log("Loading checkpoint and creating model...")
    checkpoint = dist_util.load_state_dict(args.model_path, map_location="cpu")
    
    # --- 💡【ここが永続的な最終修正です】💡 ---

    # 1. create_modelが受け付ける「正しい引数」のリストを定義します。
    #    (これは train.py で使ったものと同じリストです)
    known_model_args = [
        'image_size', 'in_channels', 'num_channels', 'out_channels', 
        'num_res_blocks', 'attention_resolutions', 'dropout', 'channel_mult', 
        'use_checkpoint', 'use_fp16', 'num_heads', 'num_head_channels', 
        'use_scale_shift_norm', 'resblock_updown', 'use_new_attention_order', 
        'dims', 'num_groups', 'renormalize', 'additive_skips', 'use_freq', 
        'bottleneck_attention', 'resample_2d'
    ]

    # 2. チェックポイントから読み込んだ引数のうち、「正しい引数」だけを抽出します。
    all_loaded_args = checkpoint['model_args']
    clean_model_args = {
        key: all_loaded_args[key] 
        for key in known_model_args 
        if key in all_loaded_args
    }
    
    # 3. 抽出したクリーンな引数だけを使ってモデルを作成します。
    model = create_model(**clean_model_args)
    
    # --- ✅ これで、この種のエラーは二度と発生しません ✅ ---
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(dist_util.dev())
    model.eval()
    logger.log("Model created and loaded successfully.")

    logger.log("Creating data loader...")
    ds = BRATSVolumes(args.data_dir, mode='eval')
    datal = th.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    dwt = DWT_3D("haar").to(dist_util.dev())
    idwt = IDWT_3D("haar").to(dist_util.dev())

    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    logger.log(f"Starting sampling with {args.ode_steps} steps...") # ログにステップ数を表示
    for i, (target_batch, cond_batch) in enumerate(tqdm(datal, desc="Processing Subjects")):
        cond_image = cond_batch['cond_1'].to(dist_util.dev())

        with th.no_grad():
            cond_wav_components = dwt(cond_image)
            cond_wav = th.cat(cond_wav_components, dim=1)
        
        out_channels = checkpoint['model_args'].get('out_channels', 8)
        sample_shape = (args.batch_size, out_channels, 112, 112, 80)
        
        # 💡 ハードコードされた100の代わりに、引数で受け取ったステップ数を使います
        sample_wav = ode_sampler(model, cond_wav, sample_shape, dist_util.dev(), steps=args.ode_steps)


        # ... (以降の画像保存処理は変更なし)
        B, _, D, H, W = sample_wav.shape
        sample_idwt = idwt(
            sample_wav[:, 0:1, ...], sample_wav[:, 1:2, ...], sample_wav[:, 2:3, ...], sample_wav[:, 3:4, ...],
            sample_wav[:, 4:5, ...], sample_wav[:, 5:6, ...], sample_wav[:, 6:7, ...], sample_wav[:, 7:8, ...]
        )
        sample_idwt[sample_idwt < 0] = 0
        sample_idwt[sample_idwt > 1] = 1
        sample_idwt[cond_image == 0] = 0
        sample_idwt = sample_idwt.squeeze(dim=1)[:, :, :, :155]
        
        subj_name = f"Sample_{i:04d}"
        subj_dir = os.path.join(args.output_dir, subj_name)
        pathlib.Path(subj_dir).mkdir(parents=True, exist_ok=True)
        
        output_name = os.path.join(subj_dir, 'sample_t1c_from_t1n.nii.gz')
        img = nib.Nifti1Image(sample_idwt.detach().cpu().numpy()[0], np.eye(4))
        nib.save(img=img, filename=output_name)

    logger.log(f"Sampling complete. Results saved to {args.output_dir}")

def create_argparser():
    """
    サンプリングに必要な引数を定義します。
    """
    defaults = dict(
        seed=42,
        data_dir="",
        model_path="",
        batch_size=1,
        output_dir="./results_flow_matching",
        devices=[0],
        # 💡【これを追加】: サンプリングステップ数を指定する引数を追加します
        ode_steps=100, 
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()