import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader

from dataset import ShapeDetectionDataset
from model import MultiScaleDetector
from utils import generate_anchors, compute_iou

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

def compute_ap(predictions, ground_truths, iou_threshold=0.5):
    all_scores = []
    all_matches = []
    total_gts = 0

    for pred_boxes, gt_boxes in zip(predictions, ground_truths):
        scores = [s for s, _, _ in pred_boxes]
        boxes = torch.tensor([b for _, b, _ in pred_boxes], dtype=torch.float32)
        gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)

        if gt_boxes.numel() == 0:
            continue
        total_gts += len(gt_boxes)

        if len(boxes) == 0:
            continue

        ious = compute_iou(boxes, gt_boxes)
        max_iou, _ = ious.max(dim=1)

        matches = (max_iou >= iou_threshold).int().tolist()
        all_scores.extend(scores)
        all_matches.extend(matches)

    if len(all_scores) == 0 or total_gts == 0:
        return 0.0

    sorted_idx = np.argsort(-np.array(all_scores))
    matches = np.array(all_matches)[sorted_idx]

    tp = np.cumsum(matches)
    fp = np.cumsum(1 - matches)

    recalls = tp / (total_gts + 1e-6)
    precisions = tp / (tp + fp + 1e-6)

    ap = np.trapz(precisions, recalls)
    return ap


def decode_predictions(preds, anchors, num_classes=3, conf_thresh=0.5, image_size=224):
    decoded = []
    for pred, anchor in zip(preds, anchors):
        B, C, H, W = pred.shape
        pred = pred.permute(0, 2, 3, 1).contiguous()
        pred = pred.view(B, -1, 5 + num_classes)

        obj_scores = torch.sigmoid(pred[..., 4])
        class_scores = torch.softmax(pred[..., 5:], dim=-1)
        scores, labels = class_scores.max(dim=-1)
        scores = scores * obj_scores

        tx, ty, tw, th = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3]

        anchors_cx = (anchor[:, 0] + anchor[:, 2]) / 2
        anchors_cy = (anchor[:, 1] + anchor[:, 3]) / 2
        anchors_w = anchor[:, 2] - anchor[:, 0]
        anchors_h = anchor[:, 3] - anchor[:, 1]

        cx = anchors_cx + torch.sigmoid(tx) * anchors_w
        cy = anchors_cy + torch.sigmoid(ty) * anchors_h
        w = anchors_w * torch.exp(torch.clamp(tw, max=4.0))   
        h = anchors_h * torch.exp(torch.clamp(th, max=4.0))

        x1 = torch.clamp(cx - w / 2, 0, image_size - 1)
        y1 = torch.clamp(cy - h / 2, 0, image_size - 1)
        x2 = torch.clamp(cx + w / 2, 0, image_size - 1)
        y2 = torch.clamp(cy + h / 2, 0, image_size - 1)

        boxes = torch.stack([x1, y1, x2, y2], dim=-1)

        mask = scores > conf_thresh
        decoded.append([
            (s.item(), b.tolist(), l.item())
            for s, b, l, m in zip(scores.view(-1), boxes.view(-1, 4),
                                  labels.view(-1), mask.view(-1)) if m
        ])
    return decoded


def visualize_detections(image, predictions, ground_truths, save_path):
    fig, ax = plt.subplots(1, figsize=(6, 6))

    img_np = image.cpu().numpy()
    if img_np.shape[0] == 3:  # (C, H, W)
        img_np = np.transpose(img_np, (1, 2, 0))  # -> (H, W, C)

    ax.imshow(img_np)

    for box in ground_truths:
        x1, y1, x2, y2 = box
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                         linewidth=2, edgecolor="g", facecolor="none")
        ax.add_patch(rect)

    for score, box, *_ in predictions: 
        x1, y1, x2, y2 = box
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                         linewidth=2, edgecolor="r", facecolor="none", linestyle="--")
        ax.add_patch(rect)
        ax.text(x1, y1, f"{score:.2f}", color="r", fontsize=6)

    plt.axis("off")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def visualize_anchor_grid(anchors, save_path):
    fig, ax = plt.subplots(1, figsize=(6, 6))
    for box in anchors[::500]:
        x1, y1, x2, y2 = box
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1,
                         linewidth=0.5, edgecolor="b", facecolor="none")
        ax.add_patch(rect)
    ax.set_xlim(0, 224)
    ax.set_ylim(0, 224)
    ax.invert_yaxis()
    plt.title("Anchor Coverage")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def analyze_scale_performance(model, dataloader, anchors):
    model.eval()
    scale_stats = {1: [], 2: [], 3: []}

    with torch.no_grad():
        for images, _ in dataloader:
            if isinstance(images, (tuple, list)):
                images = torch.stack(images, dim=0)  
            images = images.to(device)
            preds = model(images)

            for scale_idx, (pred, anchor) in enumerate(zip(preds, anchors), start=1):
                B, C, H, W = pred.shape
                pred = pred.permute(0, 2, 3, 1).contiguous()
                pred = pred.view(B, -1, 5 + model.num_classes)

                obj_scores = torch.sigmoid(pred[..., 4])
                max_scores, _ = obj_scores.max(dim=1)
                scale_stats[scale_idx].extend(max_scores.cpu().tolist())

    fig, ax = plt.subplots()
    for s in scale_stats:
        ax.hist(scale_stats[s], bins=20, alpha=0.5, label=f"Scale {s}")
    ax.set_xlabel("Objectness score")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    plt.savefig("problem1/results/visualizations/scale_analysis.png")
    plt.close()


if __name__ == "__main__":
    val_dataset = ShapeDetectionDataset(
        image_dir="../datasets/detection/val",
        annotation_file="../datasets/detection/val_annotations.json"
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            collate_fn=lambda b: tuple(zip(*b)))

    model = MultiScaleDetector(num_classes=3, num_anchors=3).to(device)
    model.load_state_dict(torch.load("problem1/results/best_model.pth", map_location=device))
    model.eval()

    feature_map_sizes = [(56, 56), (28, 28), (14, 14)]
    anchor_scales = [[16, 24, 32], [48, 64, 96], [96, 128, 192]]
    anchors = generate_anchors(feature_map_sizes, anchor_scales, image_size=224)
    anchors = [a.to(device) for a in anchors]

    os.makedirs("problem1/results/visualizations/detections", exist_ok=True)
    os.makedirs("problem1/results/visualizations/anchors", exist_ok=True)

    for i in range(10):
        img, target = val_dataset[i]
        with torch.no_grad():
            preds = model(img.unsqueeze(0).to(device))
        decoded = decode_predictions(preds, anchors, num_classes=3, conf_thresh=0.5)
        visualize_detections(img, decoded[0], target["boxes"],
                             f"problem1/results/visualizations/detections/val_{i}.png")

    for i, anc in enumerate(anchors, start=1):
        visualize_anchor_grid(anc.cpu().numpy(),
                              f"problem1/results/visualizations/anchors/scale{i}.png")

    analyze_scale_performance(model, val_loader, anchors)

    print("Evaluation & visualizations saved to problem1/results/visualizations/")
