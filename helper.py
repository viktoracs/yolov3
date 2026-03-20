import torch
import math
import cv2
import matplotlib.pyplot as plt
import torchvision.ops as ops
from logger import logger

# The pairwise IoU calculation compares every predicted bounding box with every GT box and computes an (NxM) IoU matrix.
def compute_iou(pred_boxes, true_boxes, pairwise=False):
    """
    pred_boxes: predicted bboxes (tensor of shape [B, S, S, N, 4] or [N, 4])
    true_boxes: GT bboxes (tensor of shape [B, S, S, N, 4] or [M, 4])
    pairwise: IF true, computes pairwise IoU (NxM), otherwise computes element-wise IoU
    return: IoU scores (tensor of shape [B, S, S, N] or [N, M]).
    """
    if pairwise:
        """
        PAIRWISE IOU:               
        
        Use cases: 
            Anchor matching (find best anchor per GT box)
            Evaluation (match predictions to GT)

        Examples:
        # 2 predicted boxes
        pred_boxes = torch.tensor([
            [0, 0, 2, 2],
            [1, 1, 3, 3]
        ])  # Shape: [2, 4]

        # 3 ground truth boxes
        gt_boxes = torch.tensor([
            [0, 0, 1, 1],
            [1, 1, 2, 2],
            [2, 2, 4, 4]
        ])  # Shape: [3, 4]

        iou_matrix = compute_iou(pred_boxes, gt_boxes, pairwise=True)

        ELEMENTWISE IOU:     
       
        Use cases: 
            Loss computation (compare model output with matched GT)
            Per-anchor per-cell predictions
         
        Examples:    
        # 2 matching boxes (e.g. model prediction vs GT per anchor)
        pred_boxes = torch.tensor([
            [0, 0, 2, 2],
            [1, 1, 3, 3]
        ])  # Shape: [2, 4]

        gt_boxes = torch.tensor([
            [0, 0, 1, 1],
            [1, 1, 2, 2]
        ])  # Shape: [2, 4]

        iou_scores = compute_iou(pred_boxes, gt_boxes, pairwise=False)

        Analogy:
            3 people (predictions) trying to grab the best-fitting 4 shoes (GT).
            Each person tries all 4 shoes. That's pairwise.
            Just check if each person’s assigned shoe fits. That’s elementwise.

        Pairwise = “Which one matches best?”
        Elementwise = “How well does this assigned pair match?”
        """
        
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        true_x1, true_y1, true_x2, true_y2 = true_boxes[:, 0], true_boxes[:, 1], true_boxes[:, 2], true_boxes[:, 3]

        # Compute intersection
        inter_x1 = torch.max(pred_x1[:, None], true_x1[None, :])
        inter_y1 = torch.max(pred_y1[:, None], true_y1[None, :])
        inter_x2 = torch.min(pred_x2[:, None], true_x2[None, :])
        inter_y2 = torch.min(pred_y2[:, None], true_y2[None, :])

        # Ensures a valid intersection (x2 > x1 and y2 > y1) - If the boxes do not overlap, intersection area is set to 0
        valid_intersection = (inter_x2 > inter_x1) & (inter_y2 > inter_y1)
        inter_area = valid_intersection * ((inter_x2 - inter_x1) * (inter_y2 - inter_y1))

        # Compute union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        true_area = (true_x2 - true_x1) * (true_y2 - true_y1)
        
        pred_area = pred_area.unsqueeze(1)  # [N, 1]
        true_area = true_area.unsqueeze(0)  # [1, M]

        union_area = pred_area + true_area - inter_area

        return inter_area / (union_area + 1e-6)
    
    else: 
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes[..., 0], pred_boxes[..., 1], pred_boxes[..., 2], pred_boxes[..., 3]
        true_x1, true_y1, true_x2, true_y2 = true_boxes[..., 0], true_boxes[..., 1], true_boxes[..., 2], true_boxes[..., 3]

        # Safety check: ensure both tensors have matching shapes
        if pred_boxes.shape != true_boxes.shape:
            true_x1 = true_x1.expand_as(pred_x1)
            true_y1 = true_y1.expand_as(pred_y1)
            true_x2 = true_x2.expand_as(pred_x2)
            true_y2 = true_y2.expand_as(pred_y2)

        # Compute intersection
        inter_x1 = torch.max(pred_x1, true_x1)
        inter_y1 = torch.max(pred_y1, true_y1)
        inter_x2 = torch.min(pred_x2, true_x2)
        inter_y2 = torch.min(pred_y2, true_y2)

        valid_intersection = (inter_x2 > inter_x1) & (inter_y2 > inter_y1)
        inter_area = valid_intersection * ((inter_x2 - inter_x1) * (inter_y2 - inter_y1))

        # Compute union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        true_area = (true_x2 - true_x1) * (true_y2 - true_y1)
        
        # Union Area = area of box 1 + area of box 2 − intersection area
        union_area = pred_area + true_area - inter_area

        # Safety check: avoid division by zero
        return inter_area / (union_area + 1e-6)
    
# Generates the targets for training by assigning GT boxes to anchors and scales    
def generate_yolo_targets_global(gt_boxes, class_labels, anchors, grid_sizes, num_classes): 
    """
    Called with normalized_boxes and scaled_anchors from train.py
    
    gt_boxes: tensor of shape [B, N, 4] with normalized GT boxes (cx, cy, w, h)
    anchors: list of all 9 anchors in grid units
    
    1. Assigns GT boxes to the right anchor and scale.
    2. Computes offsets using YOLO’s formulation (tw = log(w/pw)).
    3. Writes the objectness, coordinates and one-hot class vector to the target tensor.
    4. Uses IoU to select the best matching anchor among all 9.
    """
    
    batch_size, num_boxes = gt_boxes.shape[0], gt_boxes.shape[1]
    num_anchors_per_scale = 3

    targets = [
        torch.zeros((batch_size, num_anchors_per_scale, S, S, 5 + num_classes), device=gt_boxes.device)
        for S in grid_sizes
    ]

    for b in range(batch_size):
        for i in range(num_boxes):
            cx_n, cy_n, w_n, h_n = gt_boxes[b, i]  # Store normalized values 
            cx, cy, w, h = cx_n, cy_n, w_n, h_n
            class_id = class_labels[b, i].long()
           
            # Safety check for invalid class IDs
            if class_id == -1:
                continue
            
            # Safety check for invalid boxes
            if w <= 0 or h <= 0:
                continue  

            # Compute IoU in grid units instead of normalized space. Compare GT vs anchor shapes centered at (0,0) shape match (width/height) not location-dependent.
            gt_ious = []
            for scale_idx, S in enumerate(grid_sizes):
                w_grid = w * S
                h_grid = h * S
                scale_anchors = anchors[scale_idx * num_anchors_per_scale : (scale_idx + 1) * num_anchors_per_scale]
                gt_corners_shape = torch.tensor([[-w_grid/2, -h_grid/2, w_grid/2, h_grid/2]], device=gt_boxes.device)
                anchor_corners = torch.tensor(
                    [[-aw/2, -ah/2, aw/2, ah/2] for aw, ah in scale_anchors],
                    device=gt_boxes.device
                )
                ious = compute_iou(gt_corners_shape, anchor_corners, pairwise=True).squeeze(0)
                for a_i, iou in enumerate(ious):
                    gt_ious.append((iou.item(), scale_idx, a_i))

            # Pick best anchor across all scales
            _, scale_idx, anchor_idx = max(gt_ious, key=lambda x: x[0])

            S = grid_sizes[scale_idx]

            # Compute grid cell indices
            grid_x = int(cx * S)
            grid_y = int(cy * S)

            # Clamp indices into valid range
            grid_x = max(min(grid_x, S - 1), 0)
            grid_y = max(min(grid_y, S - 1), 0)

            rel_x = cx * S - grid_x
            rel_y = cy * S - grid_y

            # Width/height offsets in grid units
            aw, ah = anchors[scale_idx * num_anchors_per_scale + anchor_idx]
            rel_w = torch.log(w * S / aw + 1e-6)
            rel_h = torch.log(h * S / ah + 1e-6)

            # Safety check: clamp values to avoid extreme offsets
            rel_w = rel_w.clamp(min=-5.0, max=5.0)
            rel_h = rel_h.clamp(min=-5.0, max=5.0)

            # Safety check: warn for invalid class IDs
            if not (0 <= class_id < num_classes):
                logger.warning(f"[W] Invalid class ID. Skip.")
                continue

            # Assign targets
            # Where to look: grid_x, grid_y | Which anchor should predict: anchor_idx at scale_idx | How to offset: rel_x, rel_y, rel_w, rel_h | What class: class_id         
            targets[scale_idx][b, anchor_idx, grid_y, grid_x, 0:4] = torch.tensor(
                [rel_x, rel_y, rel_w, rel_h], device=gt_boxes.device
            )
            targets[scale_idx][b, anchor_idx, grid_y, grid_x, 4] = 1.0 # Objectness
            targets[scale_idx][b, anchor_idx, grid_y, grid_x, 5 + int(class_id.item())] = 1.0 # One-hot class vector

    return targets

