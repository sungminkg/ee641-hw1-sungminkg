import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import json

from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet


def extract_keypoints_from_heatmaps(heatmaps):
    B, K, H, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.view(B, K, -1)
    max_indices = torch.argmax(heatmaps_reshaped, dim=-1)
    xs = (max_indices % W).float()
    ys = (max_indices // W).float()
    coords = torch.stack([xs / W, ys / H], dim=-1)
    return coords


def compute_pck(predictions, ground_truths, thresholds, normalize_by='bbox'):
    preds = predictions.cpu().numpy()
    gts = ground_truths.cpu().numpy()
    dists = np.linalg.norm(preds - gts, axis=-1)

    if normalize_by == "bbox":
        min_xy = np.min(gts, axis=1)
        max_xy = np.max(gts, axis=1)
        diag = np.linalg.norm(max_xy - min_xy, axis=-1)
        norm = diag[:, None] + 1e-8
    elif normalize_by == "torso":
        head = gts[:, 0, :]
        feet_mid = (gts[:, 3, :] + gts[:, 4, :]) / 2.0
        torso_len = np.linalg.norm(head - feet_mid, axis=-1)
        norm = torso_len[:, None] + 1e-8
    else:
        raise ValueError(f"Unknown normalize_by: {normalize_by}")

    dists_norm = dists / norm
    pck_values = {}
    for th in thresholds:
        correct = (dists_norm <= th).astype(np.float32)
        acc = correct.mean()
        pck_values[th] = acc
    return pck_values


def plot_pck_curves(pck_heatmap, pck_regression, save_path):
    thresholds = sorted(pck_heatmap.keys())
    plt.figure(figsize=(6, 5))
    plt.plot(thresholds, [pck_heatmap[t] for t in thresholds],
             marker='o', label="HeatmapNet")
    plt.plot(thresholds, [pck_regression[t] for t in thresholds],
             marker='s', label="RegressionNet")
    plt.xlabel("Threshold (normalized distance)")
    plt.ylabel("PCK")
    plt.title("PCK Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def visualize_predictions(image, pred_keypoints, gt_keypoints, save_path):
    if torch.is_tensor(image):
        image = image.squeeze().cpu().numpy()
    H, W = image.shape
    pred = pred_keypoints * np.array([W, H])
    gt = gt_keypoints * np.array([W, H])
    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap="gray")
    plt.scatter(gt[:, 0], gt[:, 1], c="g", marker="o", label="GT")
    plt.scatter(pred[:, 0], pred[:, 1], c="r", marker="x", label="Pred")
    plt.legend()
    plt.axis("off")
    plt.savefig(save_path)
    plt.close()


def visualize_heatmaps(heatmaps, save_path):
    """Save first sample's heatmaps as a grid."""
    if torch.is_tensor(heatmaps):
        heatmaps = heatmaps[0].cpu().numpy()  # [K,H,W]
    K = heatmaps.shape[0]
    fig, axes = plt.subplots(1, K, figsize=(12, 3))
    for i in range(K):
        axes[i].imshow(heatmaps[i], cmap="hot")
        axes[i].axis("off")
        axes[i].set_title(f"K{i}")
    plt.savefig(save_path)
    plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # setup results dirs
    results_dir = "results"
    vis_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # dataset paths
    val_img_dir = "../datasets/keypoints/val"
    val_ann_file = "../datasets/keypoints/val_annotations.json"
    val_dataset = KeypointDataset(val_img_dir, val_ann_file, output_type="heatmap")
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # load models
    heatmap_model = HeatmapNet(num_keypoints=5).to(device)
    regression_model = RegressionNet(num_keypoints=5).to(device)
    if os.path.exists(os.path.join(results_dir, "heatmap_model.pth")):
        heatmap_model.load_state_dict(torch.load(os.path.join(results_dir, "heatmap_model.pth"), map_location=device))
    if os.path.exists(os.path.join(results_dir, "regression_model.pth")):
        regression_model.load_state_dict(torch.load(os.path.join(results_dir, "regression_model.pth"), map_location=device))
    heatmap_model.eval()
    regression_model.eval()

    preds_heatmap, preds_reg, gts, images, last_heatmap = [], [], [], [], None
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            out_hm = heatmap_model(imgs)
            out_reg = regression_model(imgs)
            coords_hm = extract_keypoints_from_heatmaps(out_hm)
            coords_reg = out_reg.view(out_reg.size(0), -1, 2)
            preds_heatmap.append(coords_hm.cpu())
            preds_reg.append(coords_reg.cpu())
            if targets.ndim == 4:
                gt_coords = extract_keypoints_from_heatmaps(targets)
            else:
                gt_coords = targets.view(targets.size(0), -1, 2)
            gts.append(gt_coords.cpu())
            images.append(imgs.cpu())
            last_heatmap = out_hm  # save last batch heatmap

    preds_heatmap = torch.cat(preds_heatmap)
    preds_reg = torch.cat(preds_reg)
    gts = torch.cat(gts)
    images = torch.cat(images)

    # PCK curves
    thresholds = [0.05, 0.1, 0.2]
    pck_heatmap = compute_pck(preds_heatmap, gts, thresholds)
    pck_regression = compute_pck(preds_reg, gts, thresholds)
    plot_pck_curves(pck_heatmap, pck_regression, save_path=os.path.join(vis_dir, "pck_curves.png"))

    # Sample predictions
    for i in range(min(5, len(images))):
        visualize_predictions(images[i], preds_heatmap[i].numpy(), gts[i].numpy(),
                              save_path=os.path.join(vis_dir, f"sample_{i}_heatmap.png"))
        visualize_predictions(images[i], preds_reg[i].numpy(), gts[i].numpy(),
                              save_path=os.path.join(vis_dir, f"sample_{i}_regression.png"))

    # Heatmaps
    if last_heatmap is not None:
        visualize_heatmaps(last_heatmap, os.path.join(vis_dir, "predicted_heatmaps.png"))

    # Save results
    pck_heatmap = {k: float(v) for k, v in pck_heatmap.items()}
    pck_regression = {k: float(v) for k, v in pck_regression.items()}

    with open("results/pck_results.json", "w") as f:
        json.dump({"heatmap": pck_heatmap, "regression": pck_regression}, f, indent=2)


if __name__ == "__main__":
    main()
