import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os

from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet


def train_heatmap_model(model, train_loader, val_loader, num_epochs=30, device="cuda", results_dir="results"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = model.to(device)
    best_val_loss = float("inf")
    log = {"heatmap": {"train": [], "val": []}}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)

        log["heatmap"]["train"].append(train_loss)
        log["heatmap"]["val"].append(val_loss)

        print(f"[Heatmap] Epoch {epoch+1}/{num_epochs} "
              f"Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(results_dir, "heatmap_model.pth"))

    return log


def train_regression_model(model, train_loader, val_loader, num_epochs=30, device="cuda", results_dir="results"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model = model.to(device)
    best_val_loss = float("inf")
    log = {"regression": {"train": [], "val": []}}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= len(val_loader.dataset)

        log["regression"]["train"].append(train_loss)
        log["regression"]["val"].append(val_loss)

        print(f"[Regression] Epoch {epoch+1}/{num_epochs} "
              f"Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(results_dir, "regression_model.pth"))

    return log


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # results directory setup
    results_dir = os.path.join("results")
    vis_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # dataset paths
    train_img_dir = "../datasets/keypoints/train"
    train_ann_file = "../datasets/keypoints/train_annotations.json"
    val_img_dir = "../datasets/keypoints/val"
    val_ann_file = "../datasets/keypoints/val_annotations.json"

    # datasets
    train_dataset_heatmap = KeypointDataset(train_img_dir, train_ann_file, output_type="heatmap")
    val_dataset_heatmap = KeypointDataset(val_img_dir, val_ann_file, output_type="heatmap")
    train_dataset_reg = KeypointDataset(train_img_dir, train_ann_file, output_type="regression")
    val_dataset_reg = KeypointDataset(val_img_dir, val_ann_file, output_type="regression")

    # dataloaders
    train_loader_heatmap = DataLoader(train_dataset_heatmap, batch_size=32, shuffle=True)
    val_loader_heatmap = DataLoader(val_dataset_heatmap, batch_size=32, shuffle=False)
    train_loader_reg = DataLoader(train_dataset_reg, batch_size=32, shuffle=True)
    val_loader_reg = DataLoader(val_dataset_reg, batch_size=32, shuffle=False)

    # models
    heatmap_model = HeatmapNet(num_keypoints=5)
    regression_model = RegressionNet(num_keypoints=5)

    # training
    heatmap_log = train_heatmap_model(
        heatmap_model, train_loader_heatmap, val_loader_heatmap,
        device=device, results_dir=results_dir
    )
    reg_log = train_regression_model(
        regression_model, train_loader_reg, val_loader_reg,
        device=device, results_dir=results_dir
    )

    # save logs
    log = {}
    log.update(heatmap_log)
    log.update(reg_log)
    with open(os.path.join(results_dir, "training_log.json"), "w") as f:
        json.dump(log, f, indent=2)


if __name__ == '__main__':
    main()
