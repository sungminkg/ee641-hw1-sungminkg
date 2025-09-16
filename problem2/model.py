import torch
import torch.nn as nn
import torch.nn.functional as F


# Heatmap-based network
class HeatmapNet(nn.Module):
    def __init__(self, num_keypoints=5):
        super().__init__()
        self.num_keypoints = num_keypoints

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)   # 128 -> 64
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)   # 64 -> 32
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)   # 32 -> 16
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)   # 16 -> 8
        )

        # Decoder
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 8 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),   # 16 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2),   # 32 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(32, num_keypoints, kernel_size=1)   # [B, K, 64, 64]

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)   # [B, 32, 64, 64]
        x2 = self.enc2(x1)  # [B, 64, 32, 32]
        x3 = self.enc3(x2)  # [B, 128, 16, 16]
        x4 = self.enc4(x3)  # [B, 256, 8, 8]

        # Decoder with skip connections
        d4 = self.dec4(x4)                          # [B, 128, 16, 16]
        d4 = torch.cat([d4, x3], dim=1)             # [B, 128+128=256, 16, 16]

        d3 = self.dec3(d4)                          # [B, 64, 32, 32]
        d3 = torch.cat([d3, x2], dim=1)             # [B, 64+64=128, 32, 32]

        d2 = self.dec2(d3)                          # [B, 32, 64, 64]
        out = self.final(d2)                        # [B, num_keypoints, 64, 64]

        return out




# Regression-based network
class RegressionNet(nn.Module):
    def __init__(self, num_keypoints=5):
        super().__init__()
        self.num_keypoints = num_keypoints

        # Encoder (same as HeatmapNet)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)   # 128 -> 64
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)   # 64 -> 32
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)   # 32 -> 16
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)   # 16 -> 8
        )

        # Regression head
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_keypoints * 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Encoder
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)   # [B, 256, 8, 8]

        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))  # [B, 256, 1, 1]
        x = torch.flatten(x, 1)               # [B, 256]

        # Fully connected head
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))        # [B, num_keypoints*2], normalized to [0,1]

        return x
