import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json

import os
from tqdm import tqdm

from dataset import ShapeDetectionDataset
from model import MultiScaleDetector      
from loss import DetectionLoss       
from utils import generate_anchors


def collate_fn(batch): 

    images, targets = zip(*batch)       # tuple of images, tuple of targets
    images = torch.stack(images, dim=0).contiguous()
    return images, list(targets)




def train_epoch(model, dataloader, criterion, optimizer, device, anchors):
    """Train for one epoch."""
    model.train()
    # Training loop
    

    running_loss = 0.0
    n_batches = 0

    for images, targets in tqdm(dataloader, desc="Train", leave=False):
        images = images.to(device)

        optimizer.zero_grad()

        # dataset.py forward
        predictions = model(images)

        # loss.py
        loss_dict = criterion(predictions, targets, anchors=anchors)

        loss = loss_dict["loss_total"]
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


        n_batches += 1


    avg_loss = running_loss / max(n_batches, 1)
    
    return avg_loss



def validate(model, dataloader, criterion, device, anchors):
    """Validate the model."""
    model.eval()
    # Validation loop
    

    running_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Valid", leave=False):
            images = images.to(device)
            
            predictions = model(images)
            
            loss_dict = criterion(predictions, targets, anchors=anchors)


            running_loss += loss_dict["loss_total"].item()
            
            n_batches += 1

    avg_loss = running_loss / max(n_batches, 1)
    
    return avg_loss



def main():
    # Configuration
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize dataset, model, loss, optimizer
    # Training loop with logging
    # Save best model and training log
    
    os.makedirs("results", exist_ok=True)
    log_path = "results/training_log.json"


    # === Dataset & Loader ===
    train_dataset = ShapeDetectionDataset(
        image_dir="datasets/detection/train",
        annotation_file="datasets/detection/train_annotations.json",
    )
    val_dataset = ShapeDetectionDataset(
        image_dir="datasets/detection/val",
        annotation_file="datasets/detection/val_annotations.json",
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


    # === Model / Loss / Optimizer ===
    model = MultiScaleDetector(num_classes=3, num_anchors=3).to(device)
    criterion = DetectionLoss(num_classes=3)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # === Anchors ===
    feature_map_sizes = [(56, 56), (28, 28), (14, 14)]
    anchor_scales = [[16, 24, 32], [48, 64, 96], [96, 128, 192]]
    anchors = generate_anchors(feature_map_sizes, anchor_scales, image_size=224)
    anchors = [a.to(device) for a in anchors]


    # === Training Loop ===
    best_val_loss = float("inf")
    log = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, anchors)
        val_loss = validate(model, val_loader, criterion, device, anchors)

        print(f"[Epoch {epoch + 1}/{num_epochs}], Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        log["epoch"].append(epoch + 1)
        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "results/best_model.pth")

    # Save training log
    with open("results/training_log.json", "w") as f:
        json.dump(log, f, indent=2)





if __name__ == '__main__':
    main()