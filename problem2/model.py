import torch
import torch.nn as nn
import torch.nn.functional as F

class HeatmapNet(nn.Module):
    def __init__(self, num_keypoints=5):
        """
        Initialize the heatmap regression network.
        
        Args:
            num_keypoints: Number of keypoints to detect
        """
        super().__init__()
        self.num_keypoints = num_keypoints
        
        # Encoder (downsampling path)
        # Input: [batch, 1, 128, 128]
        # Progressively downsample to extract features
        
        # Decoder (upsampling path)
        # Progressively upsample back to heatmap resolution
        # Output: [batch, num_keypoints, 64, 64]
        
        # Skip connections between encoder and decoder
        
        # ---------------- Encoder ----------------
        # Input: [B, 1, 128, 128]
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)   # [B, 32, 64, 64]

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)   # [B, 64, 32, 32]

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)   # [B, 128, 16, 16]

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(2)   # [B, 256, 8, 8]

        # ---------------- Decoder ----------------
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # [B,128,16,16]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),   # [B,64,32,32]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2),   # [B,32,64,64]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Final conv layer to get heatmaps
        self.final = nn.Conv2d(32, num_keypoints, kernel_size=1)  # [B,K,64,64]

    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, 1, 128, 128]
            
        Returns:
            heatmaps: Tensor of shape [batch, num_keypoints, 64, 64]
        """
        
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        # Decoder with skip connections
        d4 = self.deconv4(p4)               # [B,128,16,16]
        d4 = torch.cat([d4, p3], dim=1)     # concat skip

        d3 = self.deconv3(d4)               # [B,64,32,32]
        d3 = torch.cat([d3, p2], dim=1)     # concat skip

        d2 = self.deconv2(d3)               # [B,32,64,64]

        # Final prediction
        out = self.final(d2)                # [B,K,64,64]

        return out
    


class RegressionNet(nn.Module):
    def __init__(self, num_keypoints=5):
        """
        Initialize the direct regression network.
        
        Args:
            num_keypoints: Number of keypoints to detect
        """
        super().__init__()
        self.num_keypoints = num_keypoints
        
        # Use same encoder architecture as HeatmapNet
        # But add global pooling and fully connected layers
        # Output: [batch, num_keypoints * 2]
        
        # ---------------- Encoder ----------------
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)   # [B, 32, 64, 64]

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)   # [B, 64, 32, 32]

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2)   # [B, 128, 16, 16]

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(2)   # [B, 256, 8, 8]

        # ---------------- Regression Head ----------------
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # [B, 256, 1, 1]

        self.fc1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, num_keypoints * 2),
            nn.Sigmoid()   # → 確保輸出在 [0, 1]
        )
        
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, 1, 128, 128]
            
        Returns:
            coords: Tensor of shape [batch, num_keypoints * 2]
                   Values in range [0, 1] (normalized coordinates)
        """
        
        # Encoder
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.pool4(x)   # [B, 256, 8, 8]

        # Global Average Pooling
        x = self.global_pool(x)        # [B, 256, 1, 1]
        x = torch.flatten(x, 1)        # [B, 256]

        # Fully connected layers
        x = self.fc1(x)                # [B, 128]
        x = self.fc2(x)                # [B, 64]
        coords = self.fc3(x)           # [B, K*2] in [0,1]

        return coords