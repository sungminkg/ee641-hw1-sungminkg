import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json

from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet
from evaluate import extract_keypoints_from_heatmaps, compute_pck, visualize_predictions


def ablation_study(dataset, model_class, device="cuda"):
    """
    Conduct ablation studies on key hyperparameters.
    """
    results = {"heatmap_resolution": {}, "sigma": {}, "skip_connections": {}}

    # 1. Heatmap resolution
    for res in [32, 64, 128]:
        dataset.heatmap_size = res
        dataset.sigma = 2.0
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        model = model_class(num_keypoints=5).to(device)
        model.eval()

        preds, gts = [], []
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            with torch.no_grad():
                outputs = model(imgs)

            # Predictions
            if outputs.ndim == 4:  # heatmap
                coords = extract_keypoints_from_heatmaps(outputs)
            else:
                coords = outputs.view(outputs.size(0), -1, 2)
            preds.append(coords.cpu())

            # Ground truth
            if targets.ndim == 4:  # heatmap GT
                gt_coords = extract_keypoints_from_heatmaps(targets)
            else:
                gt_coords = targets.view(targets.size(0), -1, 2)
            gts.append(gt_coords.cpu())

        preds, gts = torch.cat(preds), torch.cat(gts)
        pck = compute_pck(preds, gts, thresholds=[0.05, 0.1, 0.2])
        results["heatmap_resolution"][res] = pck

    # 2. Gaussian sigma
    for sigma in [1.0, 2.0, 3.0, 4.0]:
        dataset.heatmap_size = 64
        dataset.sigma = sigma
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        model = model_class(num_keypoints=5).to(device)
        model.eval()

        preds, gts = [], []
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            with torch.no_grad():
                outputs = model(imgs)

            if outputs.ndim == 4:
                coords = extract_keypoints_from_heatmaps(outputs)
            else:
                coords = outputs.view(outputs.size(0), -1, 2)
            preds.append(coords.cpu())

            if targets.ndim == 4:
                gt_coords = extract_keypoints_from_heatmaps(targets)
            else:
                gt_coords = targets.view(targets.size(0), -1, 2)
            gts.append(gt_coords.cpu())

        preds, gts = torch.cat(preds), torch.cat(gts)
        pck = compute_pck(preds, gts, thresholds=[0.05, 0.1, 0.2])
        results["sigma"][sigma] = pck

    # 3. Skip connections
    class NoSkipHeatmapNet(HeatmapNet):
        def __init__(self, num_keypoints=5):
            super().__init__(num_keypoints=num_keypoints)
            self.dec4 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )
            self.dec3 = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
            self.dec2 = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.final = nn.Conv2d(32, num_keypoints, kernel_size=1)

        def forward(self, x):
            x1 = self.enc1(x)
            x2 = self.enc2(x1)
            x3 = self.enc3(x2)
            x4 = self.enc4(x3)

            d4 = self.dec4(x4)
            d3 = self.dec3(d4)
            d2 = self.dec2(d3)
            out = self.final(d2)
            return out

    for skip in [True, False]:
        model = HeatmapNet(num_keypoints=5).to(device) if skip else NoSkipHeatmapNet(num_keypoints=5).to(device)
        model.eval()

        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        preds, gts = [], []
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            with torch.no_grad():
                outputs = model(imgs)

            coords = extract_keypoints_from_heatmaps(outputs)
            preds.append(coords.cpu())

            if targets.ndim == 4:
                gt_coords = extract_keypoints_from_heatmaps(targets)
            else:
                gt_coords = targets.view(targets.size(0), -1, 2)
            gts.append(gt_coords.cpu())

        preds, gts = torch.cat(preds), torch.cat(gts)
        pck = compute_pck(preds, gts, thresholds=[0.05, 0.1, 0.2])
        results["skip_connections"]["with_skip" if skip else "no_skip"] = pck

    os.makedirs("results", exist_ok=True)
    with open("results/ablation_results.json", "w") as f:
        results_float = {
            k: {kk: {str(th): float(vv) for th, vv in vv.items()} for kk, vv in sub.items()}
            for k, sub in results.items()
        }
        json.dump(results_float, f, indent=2)

    return results


def analyze_failure_cases(heatmap_model, regression_model, test_loader, device="cuda"):
    """
    Identify and visualize failure cases.
    """
    heatmap_model.eval()
    regression_model.eval()

    fail_dir = os.path.join("results", "visualizations")
    os.makedirs(fail_dir, exist_ok=True)

    case_counter = {"heatmap_only": [], "regression_only": [], "both_fail": []}
    threshold = 0.1  # fixed threshold

    for idx, (imgs, targets) in enumerate(test_loader):
        imgs, targets = imgs.to(device), targets.to(device)
        with torch.no_grad():
            heatmap_out = heatmap_model(imgs)
            reg_out = regression_model(imgs)

        heatmap_coords = extract_keypoints_from_heatmaps(heatmap_out)
        reg_coords = reg_out.view(reg_out.size(0), -1, 2)

        if targets.ndim == 4:
            gt_coords = extract_keypoints_from_heatmaps(targets)
        else:
            gt_coords = targets.view(targets.size(0), -1, 2)

        for i in range(imgs.size(0)):
            h_dist = torch.norm(heatmap_coords[i] - gt_coords[i], dim=-1).mean().item()
            r_dist = torch.norm(reg_coords[i] - gt_coords[i], dim=-1).mean().item()
            h_ok, r_ok = h_dist < threshold, r_dist < threshold

            if h_ok and not r_ok:
                case_counter["heatmap_only"].append((idx, i))
                visualize_predictions(
                    imgs[i].cpu(),
                    heatmap_coords[i].cpu().numpy(),
                    gt_coords[i].cpu().numpy(),
                    os.path.join(fail_dir, f"fail_heatmap_only_{idx}_{i}.png")
                )
            elif r_ok and not h_ok:
                case_counter["regression_only"].append((idx, i))
                visualize_predictions(
                    imgs[i].cpu(),
                    reg_coords[i].cpu().numpy(),
                    gt_coords[i].cpu().numpy(),
                    os.path.join(fail_dir, f"fail_regression_only_{idx}_{i}.png")
                )
            elif not h_ok and not r_ok:
                case_counter["both_fail"].append((idx, i))
                visualize_predictions(
                    imgs[i].cpu(),
                    reg_coords[i].cpu().numpy(),
                    gt_coords[i].cpu().numpy(),
                    os.path.join(fail_dir, f"fail_both_{idx}_{i}.png")
                )

    return case_counter


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths
    test_img_dir = "../datasets/keypoints/val"
    test_ann_file = "../datasets/keypoints/val_annotations.json"

    # Dataset & Loader
    test_dataset = KeypointDataset(test_img_dir, test_ann_file, output_type="heatmap")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Load trained models
    heatmap_model = HeatmapNet(num_keypoints=5).to(device)
    regression_model = RegressionNet(num_keypoints=5).to(device)

    if os.path.exists(os.path.join("results", "heatmap_model.pth")):
        heatmap_model.load_state_dict(torch.load(os.path.join("results", "heatmap_model.pth"), map_location=device))
    if os.path.exists(os.path.join("results", "regression_model.pth")):
        regression_model.load_state_dict(torch.load(os.path.join("results", "regression_model.pth"), map_location=device))

    # Run ablation study
    print("Running ablation study...")
    ablation_results = ablation_study(test_dataset, HeatmapNet, device=device)
    print("Ablation Results:", ablation_results)

    # Analyze failure cases
    print("Analyzing failure cases...")
    failures = analyze_failure_cases(heatmap_model, regression_model, test_loader, device=device)
    print("Failure Cases:", failures)


if __name__ == "__main__":
    main()
