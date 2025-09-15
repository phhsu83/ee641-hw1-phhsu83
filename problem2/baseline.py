import csv
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

from dataset import KeypointDataset
from model import HeatmapNet
from torch.utils.data import DataLoader
import torch.optim as optim
import copy
from tqdm import tqdm


# --------------------- skip skip ---------------------
class HeatmapNetNoSkip(nn.Module):
    def __init__(self, num_keypoints=5):

        super().__init__()
        self.num_keypoints = num_keypoints
        

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
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # [B,64,32,32]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),   # [B,32,64,64]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Final conv layer to get heatmaps
        self.final = nn.Conv2d(32, num_keypoints, kernel_size=1)  # [B,K,64,64]

    
    def forward(self, x):

        
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
        # d4 = torch.cat([d4, p3], dim=1)     # concat skip

        d3 = self.deconv3(d4)               # [B,64,32,32]
        # d3 = torch.cat([d3, p2], dim=1)     # concat skip

        d2 = self.deconv2(d3)               # [B,32,64,64]

        # Final prediction
        out = self.final(d2)                # [B,K,64,64]

        return out
    
def train_heatmap_model(model, train_loader, val_loader, num_epochs=5):
    """
    Train the heatmap-based model.
    
    Uses MSE loss between predicted and target heatmaps.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Training loop
    # Log losses and save best model
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    log = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        # ---------- Training ----------
        model.train()
        train_loss = 0.0
        for imgs, heatmaps in tqdm(train_loader, desc="Train_heatmap", leave=False):  # heatmaps: [B,K,H,W]
            imgs, heatmaps = imgs.to(device), heatmaps.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)

            # 🔧 這裡關鍵：對齊到 target 的空間
            if outputs.shape[-2:] != heatmaps.shape[-2:]:
                outputs = F.interpolate(outputs, size=heatmaps.shape[-2:], mode="bilinear", align_corners=False)


            loss = criterion(outputs, heatmaps)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, heatmaps in tqdm(val_loader, desc="Valid_heatmap", leave=False):
                imgs, heatmaps = imgs.to(device), heatmaps.to(device)
                outputs = model(imgs)

                if outputs.shape[-2:] != heatmaps.shape[-2:]:
                    outputs = F.interpolate(outputs, size=heatmaps.shape[-2:], mode="bilinear", align_corners=False)
                    
                loss = criterion(outputs, heatmaps)
                val_loss += loss.item() * imgs.size(0)

        val_loss /= len(val_loader.dataset)

        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_loss)



        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())


    # Load best model and save
    model.load_state_dict(best_model_wts)

    return model

def ablation_study():
    """
    Conduct ablation studies on key hyperparameters.
    
    Experiments to run:
    1. Effect of heatmap resolution (32x32 vs 64x64 vs 128x128)
    2. Effect of Gaussian sigma (1.0, 2.0, 3.0, 4.0)
    3. Effect of skip connections (with vs without)
    """
    # Run experiments and save results

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32


    results = {"heatmap_resolution": {}, "sigma": {}, "skip_connections": {}}


    def build_loaders(hm_res: int, sigma: float):
        # train/val（heatmap 標靶，用於訓練與挑 best）
        train_ds_hm = KeypointDataset(
            image_dir="datasets/keypoints/train", annotation_file="datasets/keypoints/train_annotations.json",
            output_type="heatmap", heatmap_size=hm_res, sigma=sigma
        )
        val_ds_hm = KeypointDataset(
            image_dir="datasets/keypoints/val", annotation_file="datasets/keypoints/val_annotations.json",
            output_type="heatmap", heatmap_size=hm_res, sigma=sigma
        )
        train_loader_hm = DataLoader(train_ds_hm, batch_size=batch_size, shuffle=True)
        val_loader_hm = DataLoader(val_ds_hm, batch_size=batch_size, shuffle=False)

        return train_loader_hm, val_loader_hm

    # 1) heatmap resolution
    for hm_size in [32, 64, 128]:
        train_ld, val_ld = build_loaders(hm_size, 2.0)
        model = HeatmapNet(num_keypoints=5)
        model = train_heatmap_model(model, train_ld, val_ld)

        criterion, loss_sum, n = nn.MSELoss(), 0.0, 0
        model.eval()
        with torch.no_grad():
            for imgs, targets in val_ld:
                imgs, targets = imgs.to(device), targets.to(device)
                out = model(imgs)
                if out.shape[-2:] != targets.shape[-2:]:
                    out = nn.functional.interpolate(out, size=targets.shape[-2:], mode="bilinear", align_corners=False)

                loss_sum += criterion(out, targets).item() * imgs.size(0)
                n += imgs.size(0)
        
        results["heatmap_resolution"][str(hm_size)] = loss_sum / max(n, 1)

    # 2) sigma
    for s in [1.0, 2.0, 3.0, 4.0]:
        train_ld, val_ld = build_loaders(64, s)
        model = HeatmapNet(num_keypoints=5)
        model = train_heatmap_model(model, train_ld, val_ld)

        criterion, loss_sum, n = nn.MSELoss(), 0.0, 0
        model.eval()
        with torch.no_grad():
            for imgs, targets in val_ld:
                imgs, targets = imgs.to(device), targets.to(device)
                out = model(imgs)

                if out.shape[-2:] != targets.shape[-2:]:
                    out = nn.functional.interpolate(out, size=targets.shape[-2:], mode="bilinear", align_corners=False)
                loss_sum += criterion(out, targets).item() * imgs.size(0)
                n += imgs.size(0)
        results["sigma"][str(s)] = loss_sum / max(n, 1)

    # 3) skip connections
    train_ld, val_ld = build_loaders(64, 2.0)
    model_with = HeatmapNet(num_keypoints=5)
    model_with  = train_heatmap_model(model_with,  train_ld, val_ld)
    model_without = HeatmapNetNoSkip(num_keypoints=5)
    model_without = train_heatmap_model(model_without, train_ld, val_ld)
    
    # val MSE
    def eval_mse(m, loader):
        crit, s, n = nn.MSELoss(), 0.0, 0
        m.eval()
        with torch.no_grad():
            for imgs, targets in loader:
                imgs, targets = imgs.to(device), targets.to(device)
                out = m(imgs)
                if out.shape[-2:] != targets.shape[-2:]:
                    out = nn.functional.interpolate(out, size=targets.shape[-2:], mode="bilinear",)
                s += crit(out, targets).item() * imgs.size(0)
                n += imgs.size(0)
        return s / max(n, 1)

    results["skip_connections"]["with"] = eval_mse(model_with, val_ld)
    results["skip_connections"]["without"] = eval_mse(model_without, val_ld)
    

    # (a) resolution
    xs = sorted(results["heatmap_resolution"].keys(), key=lambda z: int(z))
    ys = [results["heatmap_resolution"][k] for k in xs]
    plt.figure(); plt.plot([int(k) for k in xs], ys, marker="o")
    plt.xlabel("Heatmap size"); plt.ylabel("Val MSE (lower better)")
    plt.grid(True, ls="--", alpha=0.4); plt.tight_layout()
    plt.savefig("results/visualizations/ablation_heatmap_resolution.png", dpi=180); plt.close()

    # (b) sigma
    xs = sorted(results["sigma"].keys(), key=lambda z: float(z))
    ys = [results["sigma"][k] for k in xs]
    plt.figure(); plt.plot([float(k) for k in xs], ys, marker="o")
    plt.xlabel("Gaussian sigma"); plt.ylabel("Val MSE (lower better)")
    plt.grid(True, ls="--", alpha=0.4); plt.tight_layout()
    plt.savefig("results/visualizations/ablation_sigma.png", dpi=180); plt.close()

    # (c) skip
    plt.figure()
    plt.bar(["with", "without"],
            [results["skip_connections"]["with"], results["skip_connections"]["without"]])
    plt.ylabel("Val MSE (lower better)")
    plt.tight_layout(); plt.savefig("results/visualizations/ablation_skip.png", dpi=180); plt.close()





def analyze_failure_cases(model, test_loader):
    """
    Identify and visualize failure cases.
    
    Find examples where:
    1. Heatmap succeeds but regression fails
    2. Regression succeeds but heatmap fails
    3. Both methods fail
    """
    pass

    # this function is in evaluate.py
    

    
def main():

    ablation_study()

if __name__ == '__main__':
    main()
