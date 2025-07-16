import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Convolution => [BatchNorm] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffZ // 2, diffZ - diffZ // 2,
                        diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    """
    標準的な3D U-Netモデル。
    BraTSデータセット用に、入力4チャンネル、出力3クラスを想定。
    
    Args:
        in_channels (int): 入力画像のチャンネル数 (デフォルト: 4)
        n_classes (int): 出力セグメンテーションマップのクラス数 (デフォルト: 3)
        bilinear (bool): Trueの場合、Upsample層を使用。Falseの場合、ConvTranspose3dを使用。
    """
    def __init__(self, in_channels=4, n_classes=3, bilinear=True):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # エンコーダ部分 (Downscaling path)
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        factor = 2 if bilinear else 1
        self.down4 = Down(256, 512 // factor)

        # デコーダ部分 (Upscaling path)
        self.up1 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128 // factor, bilinear)
        self.up3 = Up(128, 64 // factor, bilinear)
        self.up4 = Up(64, 32, bilinear)
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        # エンコーダ
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # デコーダとスキップ接続
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 最終出力
        logits = self.outc(x)
        
        # 損失関数にBCEWithLogitsLossなどを使う場合はlogitsをそのまま返す
        # Sigmoidを適用して確率として扱いたい場合は以下を有効化
        # return torch.sigmoid(logits)
        
        return logits


# --- 動作確認用のテストコード ---
if __name__ == '__main__':
    # GPUが利用可能かチェック
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # モデルのインスタンス化 (入力4ch, 出力3ch)
    model = UNet3D(in_channels=4, n_classes=3).to(device)

    # ダミーの入力テンソルを作成
    # (Batch, Channels, Depth, Height, Width)
    # BraTSのパッチサイズを想定
    dummy_input = torch.randn(1, 4, 128, 128, 128).to(device)

    print(f"Input shape: {dummy_input.shape}")

    # モデルにダミーデータを入力
    try:
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        # 期待される出力形状は (1, 3, 128, 128, 128)
        assert dummy_input.shape[2:] == output.shape[2:]
        print("モデルのフォワードパスが正常に動作しました。")
    except Exception as e:
        print(f"モデルのフォワードパスでエラーが発生しました: {e}")