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
    """Custom collate function for detection dataset."""
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)


def train_epoch(model, dataloader, criterion, optimizer, device, anchors):
    model.train()
    total_loss = 0.0
    for images, targets in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        batch_targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        preds = model(images)
        loss_dict = criterion(preds, batch_targets, anchors)
        loss = loss_dict["loss_total"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device, anchors):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validating", leave=False):
            images = images.to(device)
            batch_targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            preds = model(images)
            loss_dict = criterion(preds, batch_targets, anchors)
            total_loss += loss_dict["loss_total"].item()
    return total_loss / len(dataloader)


def main():
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_img_dir = "../datasets/detection/train"
    train_ann_file = "../datasets/detection/train_annotations.json"
    val_img_dir = "../datasets/detection/val"
    val_ann_file = "../datasets/detection/val_annotations.json"

    results_dir = "problem1/results"
    os.makedirs(results_dir, exist_ok=True)

    train_dataset = ShapeDetectionDataset(train_img_dir, train_ann_file)
    val_dataset = ShapeDetectionDataset(val_img_dir, val_ann_file)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = MultiScaleDetector(num_classes=3, num_anchors=3).to(device)
    criterion = DetectionLoss(num_classes=3)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    feature_map_sizes = [(56, 56), (28, 28), (14, 14)]
    anchor_scales = [[16, 24, 32], [48, 64, 96], [96, 128, 192]]
    anchors = generate_anchors(feature_map_sizes, anchor_scales, image_size=224)
    anchors = [a.to(device) for a in anchors]

    best_val_loss = float("inf")
    log = {"train_loss": [], "val_loss": []}

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, anchors)
        val_loss = validate(model, val_loader, criterion, device, anchors)

        log["train_loss"].append(train_loss)
        log["val_loss"].append(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pth"))

    with open(os.path.join(results_dir, "training_log.json"), "w") as f:
        json.dump(log, f)


if __name__ == '__main__':
    main()
