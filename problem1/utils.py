import torch
import numpy as np

def generate_anchors(feature_map_sizes, anchor_scales, image_size=224):
    """
    Generate anchors for multiple feature maps.
    
    Args:
        feature_map_sizes: List of (H, W) tuples for each feature map
        anchor_scales: List of lists, scales for each feature map
        image_size: Input image size
        
    Returns:
        anchors: List of tensors, each of shape [num_anchors, 4]
                 in [x1, y1, x2, y2] format
    """
    all_anchors = []

    for (fm_h, fm_w), scales in zip(feature_map_sizes, anchor_scales):
        anchors = []
        stride_y = image_size / fm_h
        stride_x = image_size / fm_w

        # For each cell in feature map
        for i in range(fm_h):
            for j in range(fm_w):
                cx = (j + 0.5) * stride_x
                cy = (i + 0.5) * stride_y
                for s in scales:
                    # Aspect ratio = 1:1 only
                    x1 = cx - s / 2
                    y1 = cy - s / 2
                    x2 = cx + s / 2
                    y2 = cy + s / 2
                    anchors.append([x1, y1, x2, y2])
        anchors = torch.tensor(anchors, dtype=torch.float32)
        all_anchors.append(anchors)
    
    return all_anchors


def compute_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: Tensor of shape [N, 4]
        boxes2: Tensor of shape [M, 4]
        
    Returns:
        iou: Tensor of shape [N, M]
    """
    N = boxes1.size(0)
    M = boxes2.size(0)

    # Expand dimensions for broadcasting
    b1 = boxes1.unsqueeze(1).expand(N, M, 4)  # [N, M, 4]
    b2 = boxes2.unsqueeze(0).expand(N, M, 4)  # [N, M, 4]

    # Intersection
    inter_x1 = torch.max(b1[..., 0], b2[..., 0])
    inter_y1 = torch.max(b1[..., 1], b2[..., 1])
    inter_x2 = torch.min(b1[..., 2], b2[..., 2])
    inter_y2 = torch.min(b1[..., 3], b2[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Areas
    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])

    union = area1 + area2 - inter_area
    iou = inter_area / union.clamp(min=1e-6)

    return iou


def match_anchors_to_targets(anchors, target_boxes, target_labels, 
                            pos_threshold=0.5, neg_threshold=0.3):
    """
    Match anchors to ground truth boxes.
    
    Args:
        anchors: Tensor of shape [num_anchors, 4]
        target_boxes: Tensor of shape [num_targets, 4]
        target_labels: Tensor of shape [num_targets]
        pos_threshold: IoU threshold for positive anchors
        neg_threshold: IoU threshold for negative anchors
        
    Returns:
        matched_labels: Tensor of shape [num_anchors]
                       (0: background, 1-N: classes)
        matched_boxes: Tensor of shape [num_anchors, 4]
        pos_mask: Boolean tensor indicating positive anchors
        neg_mask: Boolean tensor indicating negative anchors
    """
    device = anchors.device
    num_anchors = anchors.size(0)

    matched_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
    matched_boxes = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)

    if target_boxes.numel() == 0:  # no ground truth
        pos_mask = torch.zeros(num_anchors, dtype=torch.bool, device=device)
        neg_mask = torch.ones(num_anchors, dtype=torch.bool, device=device)
        return matched_labels, matched_boxes, pos_mask, neg_mask

    # Ensure targets are on the same device
    target_boxes = target_boxes.to(device)
    target_labels = target_labels.to(device)

    iou = compute_iou(anchors, target_boxes)

    # Best gt for each anchor
    max_iou, max_idx = iou.max(dim=1)

    pos_mask = max_iou >= pos_threshold
    neg_mask = max_iou < neg_threshold

    # Assign positive anchors
    matched_boxes[pos_mask] = target_boxes[max_idx[pos_mask]]
    matched_labels[pos_mask] = target_labels[max_idx[pos_mask]] + 1  # 1~C

    return matched_labels, matched_boxes, pos_mask, neg_mask