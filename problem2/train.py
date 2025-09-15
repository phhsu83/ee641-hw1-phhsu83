import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json

import os
import copy
from tqdm import tqdm

from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet

import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def save_heatmaps(imgs, pred_heatmaps, epoch, save_dir, max_images=4):
    """
    Save visualizations of predicted heatmaps over input images.

    imgs: Tensor [B, 3, H, W]
    pred_heatmaps: Tensor [B, K, H, W]
    """
    os.makedirs(save_dir, exist_ok=True)
    imgs = imgs.cpu()
    pred_heatmaps = pred_heatmaps.cpu()

    for i in range(min(len(imgs), max_images)):
        img = TF.to_pil_image(imgs[i])  # original RGB image
        fig, axs = plt.subplots(1, pred_heatmaps.shape[1] + 1, figsize=(15, 5))

        axs[0].imshow(img)
        axs[0].set_title("Input Image")
        axs[0].axis("off")

        for k in range(pred_heatmaps.shape[1]):
            axs[k + 1].imshow(pred_heatmaps[i][k], cmap='hot')
            axs[k + 1].set_title(f"Pred Heatmap {k}")
            axs[k + 1].axis("off")

        save_path = os.path.join(save_dir, f"epoch{epoch:03d}_sample{i}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


def train_heatmap_model(model, train_loader, val_loader, num_epochs=30):
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
                loss = criterion(outputs, heatmaps)
                val_loss += loss.item() * imgs.size(0)

        val_loss /= len(val_loader.dataset)

        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_loss)

        print(f"[Epoch {epoch+1}/{num_epochs}] Heatmap Train Loss: {train_loss:.4f}, Heatmap Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())


        # === Save predicted heatmaps for visualization ===
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:  # 每5個epoch存一次
            sample_imgs, sample_heatmaps = next(iter(val_loader))
            sample_imgs = sample_imgs.to(device)
            model.eval()
            with torch.no_grad():
                sample_preds = model(sample_imgs)
            save_heatmaps(
                sample_imgs, sample_preds, epoch + 1,
                save_dir="results/visualizations"
            )

    # Load best model and save
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "results/heatmap_model.pth")

    return log


def train_regression_model(model, train_loader, val_loader, num_epochs=30):
    """
    Train the direct regression model.
    
    Uses MSE loss between predicted and target coordinates.
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
        for imgs, coords in tqdm(val_loader, desc="Train_regression", leave=False):  # coords: [B,K*2]
            imgs, coords = imgs.to(device), coords.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, coords)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, coords in tqdm(val_loader, desc="Valid_regression", leave=False):
                imgs, coords = imgs.to(device), coords.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, coords)
                val_loss += loss.item() * imgs.size(0)

        val_loss /= len(val_loader.dataset)

        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_loss)

        print(f"[Epoch {epoch+1}/{num_epochs}] Reg Train Loss: {train_loss:.4f}, Reg Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    # Load best model and save
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "results/regression_model.pth")

    return log

def main():
    # Train both models with same data
    # Save training logs for comparison
    
    os.makedirs("results", exist_ok=True)
    log_path = "results/training_log.json"

    # Configuration
    batch_size = 32
    # learning_rate = 0.001
    # num_epochs = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    # === Dataset & Loader ===
    train_dataset_heatmap = KeypointDataset(
        image_dir="datasets/keypoints/train",
        annotation_file="datasets/keypoints/train_annotations.json",
        output_type="heatmap"
    )
    val_dataset_heatmap = KeypointDataset(
        image_dir="datasets/keypoints/val",
        annotation_file="datasets/keypoints/val_annotations.json",
        output_type="heatmap"
    )

    train_loader_heatmap = DataLoader(train_dataset_heatmap, batch_size=batch_size, shuffle=True)
    val_loader_heatmap = DataLoader(val_dataset_heatmap, batch_size=batch_size, shuffle=False)


    train_dataset_reg = KeypointDataset(
        image_dir="datasets/keypoints/train",
        annotation_file="datasets/keypoints/train_annotations.json",
        output_type="regression"
    )
    val_dataset_reg = KeypointDataset(
        image_dir="datasets/keypoints/val",
        annotation_file="datasets/keypoints/val_annotations.json",
        output_type="regression"
    )

    train_loader_reg = DataLoader(train_dataset_reg, batch_size=batch_size, shuffle=True)
    val_loader_reg = DataLoader(val_dataset_reg, batch_size=batch_size, shuffle=False)




    # Train Heatmap model
    heatmap_model = HeatmapNet(num_keypoints=5)
    heatmap_log = train_heatmap_model(heatmap_model, train_loader_heatmap, val_loader_heatmap)

    # Train Regression model
    regression_model = RegressionNet(num_keypoints=5)
    regression_log = train_regression_model(regression_model, train_loader_reg, val_loader_reg)

    # Save logs for comparison
    training_log = {
        "heatmap_model": heatmap_log,
        "regression_model": regression_log
    }

    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=4)



if __name__ == '__main__':
    main()