import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import match_anchors_to_targets

class DetectionLoss(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")

    def forward(self, predictions, targets, anchors):
        device = predictions[0].device
        batch_size = predictions[0].size(0)

        total_obj_loss = 0.0
        total_cls_loss = 0.0
        total_loc_loss = 0.0

        for b in range(batch_size):
            target_boxes = targets[b]["boxes"].to(device)
            target_labels = targets[b]["labels"].to(device)

            for pred, anchor in zip(predictions, anchors):
                B, C, H, W = pred.shape
                pred = pred[b]  # [C, H, W]
                pred = pred.permute(1, 2, 0).contiguous().view(-1, 5 + self.num_classes)
                # [H*W*A, 5+C]

                # anchor matching
                matched_labels, matched_boxes, pos_mask, neg_mask = match_anchors_to_targets(
                    anchor.to(device), target_boxes, target_labels
                )

                # objectness
                obj_pred = pred[:, 4]
                obj_target = torch.zeros_like(obj_pred)
                obj_target[pos_mask] = 1.0
                obj_loss = self.bce(obj_pred, obj_target)

                # hard negative mining
                neg_selected = self.hard_negative_mining(obj_loss.detach(), pos_mask, neg_mask, ratio=3)
                selected_mask = pos_mask | neg_selected
                if selected_mask.any():
                    obj_loss = obj_loss[selected_mask].mean()
                else:
                    obj_loss = torch.tensor(0.0, device=device)

                # classification (only positives)
                if pos_mask.any():
                    cls_pred = pred[pos_mask, 5:]
                    cls_target = matched_labels[pos_mask] - 1 
                    if cls_target.numel() > 0:
                        cls_loss = self.ce(cls_pred, cls_target).mean()
                    else:
                        cls_loss = torch.tensor(0.0, device=device)
                else:
                    cls_loss = torch.tensor(0.0, device=device)

                # localization (only positives)
                if pos_mask.any():
                    loc_pred = pred[pos_mask, :4]
                    loc_target = matched_boxes[pos_mask]
                    if loc_target.numel() > 0:
                        loc_loss = self.smooth_l1(loc_pred, loc_target).mean()
                    else:
                        loc_loss = torch.tensor(0.0, device=device)
                else:
                    loc_loss = torch.tensor(0.0, device=device)

                total_obj_loss += obj_loss
                total_cls_loss += cls_loss
                total_loc_loss += loc_loss

        # average over batch
        total_obj_loss /= batch_size
        total_cls_loss /= batch_size
        total_loc_loss /= batch_size

        loss_total = total_obj_loss + total_cls_loss + 2.0 * total_loc_loss

        return {
            "loss_obj": total_obj_loss,
            "loss_cls": total_cls_loss,
            "loss_loc": total_loc_loss,
            "loss_total": loss_total,
        }

    def hard_negative_mining(self, loss, pos_mask, neg_mask, ratio=3):
        num_pos = pos_mask.sum().item()
        num_neg = min(int(num_pos * ratio), neg_mask.sum().item())

        if num_neg <= 0:
            return torch.zeros_like(neg_mask, dtype=torch.bool)

        neg_losses = loss[neg_mask]
        if neg_losses.numel() == 0:
            return torch.zeros_like(neg_mask, dtype=torch.bool)

        _, idx = torch.topk(neg_losses, num_neg)
        neg_indices = neg_mask.nonzero(as_tuple=False).squeeze()
        selected_neg_mask = torch.zeros_like(neg_mask, dtype=torch.bool)
        selected_neg_mask[neg_indices[idx]] = True

        return selected_neg_mask
