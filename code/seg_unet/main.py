import os
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from monai.networks.nets import SwinUNETR

# 作成済みの他ファイルからインポート
from dataset import BraTSDataset
from utils import CombinedLoss, train_one_epoch, evaluate, plot_learning_curve

def main(config):
    # --- 1. デバイスの設定 ---
    # exportコマンドで指定されるため、コードはシンプルにこれでOK
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"✅ PyTorch is running on GPU: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")

    # --- 2. データセットの準備 ---
    train_patient_ids = [d for d in os.listdir(config['train_data_dir']) if os.path.isdir(os.path.join(config['train_data_dir'], d))]
    val_patient_ids = [d for d in os.listdir(config['val_data_dir']) if os.path.isdir(os.path.join(config['val_data_dir'], d))]
    
    train_dataset = BraTSDataset(data_dir=config['train_data_dir'], patient_ids=train_patient_ids)
    val_dataset = BraTSDataset(data_dir=config['val_data_dir'], patient_ids=val_patient_ids)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    print(f"学習データ数: {len(train_dataset)}, 検証データ数: {len(val_dataset)}")

    # --- 3. モデル、損失関数、オプティマイザの初期化 ---
    if config['model']['name'] == 'SwinUNETR':
        model = SwinUNETR(
            in_channels=4,
            out_channels=3,
            feature_size=config['model']['feature_size'],
            use_checkpoint=True,
        ).to(device)
        print(f"モデルとして SwinUNETR (feature_size={config['model']['feature_size']}) を使用します。")
    else:
        raise ValueError(f"未対応のモデル名です: {config['model']['name']}")
        
    loss_fn = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # --- 4. 学習ループ ---
    best_val_dice, history = 0.0, {'train_loss': [], 'val_loss': [], 'val_dice': []}
    run_name = config['run_name']
    print(f"--- 実行名: {run_name} ---")

    for epoch in range(config['epochs']):
        print(f"--- Epoch {epoch+1}/{config['epochs']} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_dice = evaluate(model, val_loader, loss_fn, device)
        print(f"Avg Training Loss: {train_loss:.4f}, Avg Validation Loss: {val_loss:.4f}, Avg Validation Dice: {val_dice:.4f}")
        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss); history['val_dice'].append(val_dice)
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_path = f'best_model_{run_name}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"✨ ベストモデルを保存しました: {save_path} (Dice: {best_val_dice:.4f})")

    print("学習が完了しました。")
    final_save_path = f'final_model_{run_name}.pth'
    torch.save(model.state_dict(), final_save_path)
    print(f"✅ 最終モデルを保存しました: {final_save_path}")
    plot_learning_curve(history, file_path=f'learning_curve_{run_name}.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="設定ファイル(.yaml)へのパス")
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    main(config)