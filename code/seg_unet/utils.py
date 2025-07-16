import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class DiceLoss(nn.Module):
    """
    ソフトなDice損失を計算するクラス。
    """
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # ロジットをシグモイド関数で確率に変換
        probs = torch.sigmoid(logits)
        
        # チャンネル方向に平滑化
        probs = probs.view(probs.size(0), probs.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)

        intersection = (probs * targets).sum(2)
        union = probs.sum(2) + targets.sum(2)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # 全てのクラス、全てのバッチで平均を取る
        return 1 - dice.mean()

class CombinedLoss(nn.Module):
    """
    Dice損失とBCEWithLogitsLossを組み合わせた複合損失。
    """
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        dice_loss = self.dice_loss(logits, targets)
        bce_loss = self.bce_loss(logits, targets)
        
        # 2つの損失を重み付けして合計
        total_loss = (self.dice_weight * dice_loss) + (self.bce_weight * bce_loss)
        return total_loss

def calculate_dice_score(logits, targets, smooth=1e-5):
    """
    評価用のDice係数を計算する。
    入力はロジットとターゲットテンソル。
    
    Returns:
        float: 3クラスのDiceスコアの平均値。
    """
    # ロジットを確率に変換し、0.5を閾値としてバイナリマスクを作成
    preds = (torch.sigmoid(logits) > 0.5).float()
    
    dice_scores = []
    # 各クラス（WT, TC, ET）ごとにDiceを計算
    for i in range(preds.size(1)): # クラスのチャンネルをループ
        pred_class = preds[:, i, ...].contiguous().view(-1)
        target_class = targets[:, i, ...].contiguous().view(-1)
        
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())
        
    return np.mean(dice_scores) # 3クラスの平均を返す

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """
    1エポック分の学習を実行する関数。
    """
    model.train()
    running_loss = 0.0
    
    # プログレスバーを表示
    pbar = tqdm(loader, desc="Training")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        # 勾配をリセット
        optimizer.zero_grad()
        
        # 順伝播
        outputs = model(images)
        
        # 損失計算
        loss = loss_fn(outputs, labels)
        
        # 逆伝播
        loss.backward()
        
        # パラメータ更新
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item()) # プログレスバーに現在の損失を表示
        
    return running_loss / len(loader)

def evaluate(model, loader, loss_fn, device):
    """
    モデルの評価を行う関数。
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    
    pbar = tqdm(loader, desc="Evaluating")
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # 損失とDiceスコアを計算
            loss = loss_fn(outputs, labels)
            dice = calculate_dice_score(outputs, labels)
            
            total_loss += loss.item()
            total_dice += dice
            pbar.set_postfix(dice=dice)

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    
    return avg_loss, avg_dice


def plot_learning_curve(history, file_path='learning_curve.png'):
    """
    学習曲線（損失とDiceスコア）をプロットし、ファイルに保存する関数。
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle('Learning Curve', fontsize=16)

    # --- 損失のプロット ---
    ax1.plot(epochs, history['train_loss'], 'o-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'o-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # --- Diceスコアのプロット ---
    ax2.plot(epochs, history['val_dice'], 'o-', label='Validation Dice Score', color='r')
    ax2.set_title('Validation Dice Score')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Dice Score')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(file_path)
    print(f"学習曲線グラフを {file_path} として保存しました。")

# --- 使用例 ---
if __name__ == '__main__':
    import numpy as np

    # ダミーデータで損失関数と評価指標の動作確認
    print("--- 損失関数と評価指標のテスト ---")
    device = torch.device("cpu")
    
    # 予測値（ロジット）と正解ラベルを生成
    # (B, C, D, H, W)
    dummy_logits = torch.randn(2, 3, 16, 16, 16, device=device)
    dummy_labels = (torch.rand(2, 3, 16, 16, 16, device=device) > 0.5).float()

    # 損失関数
    loss_func = CombinedLoss()
    loss_val = loss_func(dummy_logits, dummy_labels)
    print(f"Combined Loss: {loss_val.item():.4f}")

    # 評価指標
    dice_score = calculate_dice_score(dummy_logits, dummy_labels)
    print(f"Dice Score: {dice_score:.4f}")
    
    