import torch
import torch.nn.functional as F
from logger import logger

# Functions to create the ignore mask based on IoU thresholding
# Ignore mask is used to avoid penalizing predictions that have high IoU with any GT box (don't count as FPs). Ergo don't use only the highest-IoU anchor and punish the rest.
# Symptom: the loss heatmaps showed objectness blobs around GT boxes, because the model tried to predict objects there, but was penalized as FPs (for all non-assigned anchors)
# Why not mentioned in the study? ChatGPT: "Because it was considered an implementation detail, not a conceptual contribution and in Darknet it was already obvious to the authors."
"""
A useful mental analogy. 

Imagine grading students:
- You give one student full marks.
- You punish all students who sat near that student even though they answered similarly.

What happens?
- One perfect score.
- Everyone else clustered around "barely pass".
- No clear distinction of who actually knows the material.
"""

# YOLO predicts in grid space, but must be judged in pixel space, because IoU lives there. This is a slight deviation from YOLOv3.

# ------------------------------
# Convert from center to corner format (can be refactored based on train.py)
def xywh_to_xyxy(xywh):
    x, y, w, h = xywh.unbind(-1)
    return torch.stack([
        x - w / 2,
        y - h / 2,
        x + w / 2,
        y + h / 2
    ], dim=-1)

# Can be refactored with torchvision.ops.box_iou or helper.py's compute_iou() function (does practically the same)
def box_iou_xyxy(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    b1 = boxes1[:, None, :]
    b2 = boxes2[None, :, :]

    inter_x1 = torch.max(b1[..., 0], b2[..., 0])
    inter_y1 = torch.max(b1[..., 1], b2[..., 1])
    inter_x2 = torch.min(b1[..., 2], b2[..., 2])
    inter_y2 = torch.min(b1[..., 3], b2[..., 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    area1 = (b1[..., 2] - b1[..., 0]).clamp(min=0) * (b1[..., 3] - b1[..., 1]).clamp(min=0)
    area2 = (b2[..., 2] - b2[..., 0]).clamp(min=0) * (b2[..., 3] - b2[..., 1]).clamp(min=0)

    return inter / (area1 + area2 - inter + 1e-9)
# ------------------------------

def yolo_loss(pred, target, anchors, num_classes, scale_name="unknown"):
    """
    NOTES:

    Losses:
    1. Localization loss (xy_loss + wh_loss)
    2. Objectness loss (confidence loss for object-containing cells) -> TP / FN
    3. No-object loss (confidence loss for empty cells) -> FP / TN
    4. Classification loss (for object-containing cells)

    REMEMBER: 
    For each GT box:
        -> Compare against all 9 anchors and pick the best anchor by IoU
        -> The anchor determines the scale/grid/anchor assignment (this ensures each object is detected at the right resolution)

    Used loss types:    

    | Component            | Goal                         | Loss Function Used         | Type                       |
    | -------------------- | ---------------------------- | -------------------------- | -------------------------- |
    | Box (x, y, w, h)     | Predict accurate boxes       | MSE (in log-space for w/h) | Regression                 |
    | Objectness           | Is there an object?          | BCE with logits            | Binary classification      |
    | No-object            | Avoid false positives        | BCE with logits            | Binary classification      |
    | Classification       | Predict correct class (0-79) | BCE with logits            | Multi-class classification |
    """

    # Log anchor occupancy for debugging
    anchor_occupancy = [f"[I] Anchor {anchor_id}={(target[:, anchor_id, ..., 4] > 0).sum().item()}" 
                    for anchor_id in range(target.shape[1])]
    logger.info(f"[I] Anchor occupancy scale {scale_name}: " + " ".join(anchor_occupancy))
    logger.info(f"[I] Entry check (positives in target): {(target[..., 4] > 0).sum().item()} at scale {scale_name}")

    # Apply the same clamping for target and GT 
    LOG_WH_CLAMP = 6.0

    # Safet check: one-hot vector size
    one_hot = target[..., 5:]
    if one_hot.shape[-1] > num_classes:
        logger.error(f"[E] One-hot vector too large: {one_hot.shape[-1]} > {num_classes}")
    elif one_hot.shape[-1] < num_classes:
        logger.error(f"[E] One-hot vector too small: {one_hot.shape[-1]} < {num_classes}")

    # Safety check: valid class indices in target
    obj_mask = target[..., 4] > 0
    if obj_mask.any():
        class_indices = one_hot[obj_mask].argmax(dim=-1)
        if (class_indices >= num_classes).any():
            logger.error(f"[E] Invalid GT class indices found: {class_indices[class_indices >= num_classes]}")

    batch_size, C, S, _ = pred.shape
    
    num_anchors = len(anchors)

    # Safety check: expected channels
    expected_C = num_anchors * (5 + num_classes)
    assert C == expected_C, f"[I] Expected {expected_C} channels, got {C}"
    
    # Reshape from [B, 255, S, S] to [B, 3, S, S, 85] - because YOLOv3's predictions are structured per anchor and the loss logic assumes per-anchor indexing
    pred = pred.view(batch_size, num_anchors, 5 + num_classes, S, S).permute(0, 1, 3, 4, 2).contiguous()

    """
    if not torch.isfinite(pred).all():
        pred = torch.nan_to_num(pred, nan=0.0, posinf=10.0, neginf=-10.0)
    """

    pred = pred.float()
    target = target.float()

    # Detect and handle NaNs in model outputs
    if not torch.isfinite(pred).all():
        bad = (~torch.isfinite(pred)).sum().item()
        logger.error(f"[E][{scale_name}] Non-finite values in pred: count={bad}")
        print(f"[E][{scale_name}] Non-finite values in pred: count={bad}")
        # raise RuntimeError(f"Non-finite pred at {scale_name}")

    # Objectness
    pred_conf = pred[..., 4:5].clone()
    pred_conf = pred_conf.clamp(-10.0, 10.0)
    
    # Class logits 
    pred_class = pred[..., 5:].clone()
    pred_class = pred_class.clamp(-10.0, 10.0)
    
    # GT
    target_boxes = target[..., 0:4]
    target_conf = target[..., 4:5].float()
    target_class = (target[..., 5:] == 1.0).float()
   
    # This checks whether there is a GT object assigned at each anchor/grid cell (boolean mask)
    object_mask = target_conf[..., 0] > 0

    # Safety check: ensure target_class shape matches num_classes
    assert target_class.shape[-1] == num_classes, f"[E] Target class shape mismatch: {target_class.shape[-1]} vs {num_classes}"

    # Prepare the anchor box sizes to be broadcasted into the shape of pred_boxes
    xy = torch.sigmoid(pred[..., 0:2])
    
    anchor_tensor = torch.tensor(anchors, dtype=torch.float32, device=pred.device).view(1, num_anchors, 1, 1, 2) 
  
    # Decode from tw/th to width/height (grid-relative)
    # tw_th = target_boxes[..., 2:4].clamp(min=-2.0, max=2.0) # clamping for consistency (-2.0, 2.0) -> aligned with YOLO_with_ResNet50.py
    # target_wh = anchor_tensor * torch.exp(tw_th)  # shape: [B, A, S, S, 2]
    
    # safe_target_wh = target_wh.clamp(min=1e-3)
    # encoded_target_wh = torch.log(safe_target_wh / anchor_tensor + 1e-6)  

    # Add this early to verify getting positives
    num_pos = object_mask.sum().clamp(min=1.0)

    # Decode predicted tw/th into width/height in pixels (prediction)
    # Predicted offsets by the model (log scale NOT pixel scale)
    pred_tw_th = pred[..., 2:4] # .clamp(min=-2.0, max=2.0) # clamping for consistency (2.0, -2.0) -> aligned with YOLO_with_ResNet50.py  

    # FIX: both prediction and target are already log-space
    # log_pred_wh = pred_tw_th
    # log_target_wh = target_boxes[..., 2:4].clamp(-2.0, 2.0)
    log_pred_wh = pred_tw_th.clamp(-LOG_WH_CLAMP, LOG_WH_CLAMP)
    log_target_wh = target_boxes[..., 2:4].clamp(-LOG_WH_CLAMP, LOG_WH_CLAMP)

    # pred_wh = anchor_tensor * torch.exp(pred_tw_th.clamp(-6.0, 6.0)) # grid-relative 
    # pred_wh = anchor_tensor * torch.exp(log_pred_wh)
    pred_wh = anchor_tensor * torch.exp(log_pred_wh.clamp(-LOG_WH_CLAMP, LOG_WH_CLAMP))
    gt_wh = anchor_tensor * torch.exp(log_target_wh)
    
    pred_boxes = torch.cat([xy, pred_wh], dim=-1)
    
    img_size = 416  # input image size
    stride = img_size / S

    grid_y, grid_x = torch.meshgrid(
        torch.arange(S, device=pred.device),
        torch.arange(S, device=pred.device),
        indexing="ij"
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, S, S, 2).float()

   
    # pred_xy_px = (xy + grid) * stride
    
    # Decode centers to pixel space (YOLOv3-style supervision) 
    # Stable pixel-space decoding: gradients flow through xy only, not grid  
    pred_xy_px = (xy * stride) + (grid * stride).detach()
    gt_xy_px   = (target_boxes[..., 0:2] + grid) * stride

    # _______
    # LOSSES:
    # _______

    # Delta in pixels (prevents multiplying the gradients by the stride)
    delta_xy_px = pred_xy_px - gt_xy_px

    # Normalize by stride so each scale contributes similarly
    delta_xy = delta_xy_px / stride

    # 1. Localization loss
    xy_loss = torch.sum(
        object_mask.unsqueeze(-1).float()
        * (delta_xy ** 2)
    ) / num_pos # Computed only for the positive anchors 

    if object_mask.any():
        logger.info(
            f"[XY][{scale_name}] center error px mean="
            f"{torch.norm(delta_xy_px[object_mask], dim=-1).mean():.2f}"
        )

    # Log: pixel-space center error (matches xy_loss) - The model is trained in grid space, but the loss is computed in pixel space (xy_loss = pixel space, wh_loss = log space)
    # Larger objects produced larger absolute x/y gradients. This caused the numerical instability (NaN). Removed AMP (only FP32). 
    center_error_px = torch.norm(
        pred_xy_px - gt_xy_px,
        dim=-1
    )

    if object_mask.any():
        logger.info(
            f"[I] Center offset (px) mean = "
            f"{center_error_px[object_mask].mean().item():.2f}, "
            f"max = {center_error_px[object_mask].max().item():.2f}"
        )

    # Decode GT tw/th (into width/height in pixels)
    # gt_tw_th = target_boxes[..., 2:4].clamp(min=-2.0, max=2.0) # clamping for consistency (2.0, -2.0) -> aligned with YOLO_with_ResNet50.py
    # target_wh = anchor_tensor * torch.exp(gt_tw_th) # grid-relative

    # Log: width/height alignment
    # gt_wh = anchor_tensor * torch.exp(log_target_wh)
    # wh_error_px = torch.norm((pred_wh - target_wh) * stride, dim=-1)
    wh_error_px = torch.norm((pred_wh - gt_wh) * stride, dim=-1)

    if object_mask.any():
        logger.info(f"[I] Mean wh error = {wh_error_px[object_mask].mean().item():.2f} (pixels)")
    else:
        logger.info(f"[I] Mean wh error = n/a (no positives) (pixels)")

     # Log: the model's predicted and GT tw/th for positive matches
    if object_mask.any():
        # pred_tw_pos = pred_tw_th[..., 0][object_mask]
        # pred_th_pos = pred_tw_th[..., 1][object_mask]
        pred_tw_pos = log_pred_wh[..., 0][object_mask]
        pred_th_pos = log_pred_wh[..., 1][object_mask]
        # gt_tw_pos = gt_tw_th[..., 0][object_mask]
        # gt_th_pos = gt_tw_th[..., 1][object_mask]
        gt_tw_pos = log_target_wh[..., 0][object_mask]
        gt_th_pos = log_target_wh[..., 1][object_mask]


        logger.info(f"[I] Pred tw -> min={pred_tw_pos.min():.4f}, max={pred_tw_pos.max():.4f}, mean={pred_tw_pos.mean():.4f}")
        logger.info(f"[I] Pred th -> min={pred_th_pos.min():.4f}, max={pred_th_pos.max():.4f}, mean={pred_th_pos.mean():.4f}")
        logger.info(f"[I] GT   tw -> min={gt_tw_pos.min():.4f}, max={gt_tw_pos.max():.4f}, mean={gt_tw_pos.mean():.4f}")
        logger.info(f"[I] GT   th -> min={gt_th_pos.min():.4f}, max={gt_th_pos.max():.4f}, mean={gt_th_pos.mean():.4f}")

    # Log-space loss (safe version = scale-normalized)
    # log_pred_wh = torch.log(torch.clamp(pred_wh / anchor_tensor, min=1e-6))
    # log_target_wh = torch.log(torch.clamp(target_wh / anchor_tensor, min=1e-6))
    
    # Using MSE on log-space width/height 
    # (pred - target) provides the same gradient as the standard MSE formula (target - pred) - MSE is symmetric
    # (pred - target) matches autograd's convention (gradients flow through preds)
    wh_loss = torch.sum(
        object_mask.unsqueeze(-1).float() * (log_pred_wh - log_target_wh) ** 2
    ) / num_pos # Computed only for the positive anchors 

    if object_mask.any():
        logger.info(
            f"[WH][{scale_name}] pred_tw mean="
            f"{log_pred_wh[object_mask].mean():.3f}, "
            f"gt_tw mean="
            f"{log_target_wh[object_mask].mean():.3f}"
        )

    box_loss = xy_loss + wh_loss 

    # Safety check: handle NaNs in box_loss
    if not torch.isfinite(box_loss):
        logger.error("[E] box_loss is NaN!")
        box_loss = pred_conf.new_tensor(0.0)
    
    # Log: inspect objectness prediction at GT locations
    with torch.no_grad():
        obj_probs = torch.sigmoid(pred_conf[..., 0]) # Shape: [B, A, S, S]
        obj_probs_at_gt = obj_probs[object_mask]

        if obj_probs_at_gt.numel() > 0:
            logger.info(f"[I] Objectness @ GT min: {obj_probs_at_gt.min():.4f}, "
                f"max: {obj_probs_at_gt.max():.4f}, mean: {obj_probs_at_gt.mean():.4f}")
        else:
            logger.info("[W] Objectness - No GT-matched objectness locations in this batch")

    # Safe normalization (avoid division by zero)
    # num_pos = object_mask.sum().clamp(min=1.0)     
    
    # Using BCE for objectness and classification
    
    # 2. Objectness loss (TP)
    if object_mask.any():
        pred_obj = pred_conf[object_mask].float().clamp(-10,10)
        tgt_obj  = target_conf[object_mask].float()
        obj_loss = F.binary_cross_entropy_with_logits(pred_obj, tgt_obj, reduction="sum") / num_pos # Computed only for anchors assigned to a GT object 
    else:
        obj_loss = pred_conf.new_tensor(0.0)

    # ----------------------
    # YOLO ignore-IoU logic
    # ----------------------
    ignore_thresh = 0.5

    # pred_xy_px = (xy + grid) * stride
    
    # Decode predicted boxes (pixel space)
    # Reuse stable pixel-space centers (grid must not influence gradients!)
    # pred_xy_px = (xy * stride) + (grid * stride).detach() # Already declared above
    pred_wh_px = pred_wh * stride
    with torch.no_grad():
        pred_boxes_xyxy = xywh_to_xyxy(torch.cat([pred_xy_px, pred_wh_px], dim=-1))

    # Safety check:
    if torch.any(target_boxes[..., 0:2] < 0) or torch.any(target_boxes[..., 0:2] > 1):
        logger.warning("[W] GT xy offsets outside [0,1] — target encoding mismatch")

    # Decode GT boxes (pixel space)
    gt_xy_px = (target_boxes[..., 0:2] + grid) * stride
    # gt_wh_px = target_wh * stride
    gt_wh_px = gt_wh * stride
    gt_boxes_xyxy = xywh_to_xyxy(
        torch.cat([gt_xy_px, gt_wh_px], dim=-1)
    )

    # -------------------------
    # IoU @ GT diagnostics
    # -------------------------
    with torch.no_grad():
        iou_vals = []

        for b in range(batch_size):
            # GT boxes for this image
            gt_b = gt_boxes_xyxy[b][object_mask[b]] # [N_gt, 4]
            if gt_b.numel() == 0:
                continue

            # Predicted boxes for this image
            pred_b = pred_boxes_xyxy[b].view(-1, 4) # [A*S*S, 4]

            # IoU between all preds and GTs
            ious = box_iou_xyxy(pred_b, gt_b) # [A*S*S, N_gt]

            # Best IoU per GT
            best_iou_per_gt = ious.max(dim=0).values
            iou_vals.append(best_iou_per_gt)

        if len(iou_vals) > 0:
            iou_vals = torch.cat(iou_vals)
            logger.info(
                f"[IoU@GT][{scale_name}] "
                f"mean={iou_vals.mean():.3f}, "
                f"median={iou_vals.median():.3f}, "
                f"max={iou_vals.max():.3f}"
            )
        else:
            logger.info(f"[IoU@GT][{scale_name}] no GT boxes in batch")


    ignore_mask = torch.zeros_like(object_mask, dtype=torch.bool)

    for b in range(batch_size):
        gt_b = gt_boxes_xyxy[b][object_mask[b]]
        if gt_b.numel() == 0:
            continue

        pred_b = pred_boxes_xyxy[b].view(-1, 4)
        ious = box_iou_xyxy(pred_b, gt_b)
        best_iou = ious.max(dim=1).values.view(num_anchors, S, S)

        ignore_mask[b] = best_iou > ignore_thresh

    # 3. No-object loss (negative anchors - FP) - old faulty way
    """
    no_obj_loss = F.binary_cross_entropy_with_logits(
        pred_conf[~object_mask], target_conf[~object_mask], reduction="sum"
    ) / num_neg
    """
    
    # 3. No-object loss (negative anchors - FP) - new way with ignore mask (don't penalize any anchors that have high IoU with GT boxes)
    noobj_mask = (~object_mask) & (~ignore_mask)

    # Log: objectness statistics (now masks are meaningful)
    with torch.no_grad():
        if object_mask.any():
            obj_pos_mean = torch.sigmoid(pred_conf[object_mask]).mean().item()
        else:
            obj_pos_mean = float("nan")

        if noobj_mask.any():
            obj_neg_mean = torch.sigmoid(pred_conf[noobj_mask]).mean().item()
        else:
            obj_neg_mean = float("nan")

        logger.info(
            f"[I][{scale_name}] obj@pos={obj_pos_mean:.4f}, obj@neg={obj_neg_mean:.4f}"
        )

    # No-object loss
    """
    if noobj_mask.any():
        no_obj_loss = F.binary_cross_entropy_with_logits(
            pred_conf[noobj_mask], target_conf[noobj_mask], reduction="mean"
        )
    else:
        no_obj_loss = torch.tensor(0.0, device=pred.device)
    """
    # num_noobj = number of grid × anchor positions that are true background
    # "If there are 0 no-object cells, pretend there is 1." -> clamp ensures that "normalize by count" never divides by zero -> NaN
    # TN = the inverse of the object and ignore mask
    num_noobj = noobj_mask.sum().clamp(min=1.0)

    # No-object loss (safe)
    if noobj_mask.any():
        pred_noobj = pred_conf[noobj_mask].float().clamp(-10, 10)
        tgt_noobj  = target_conf[noobj_mask].float()
        
        # Computed for anchors not assigned to objects
        no_obj_loss = F.binary_cross_entropy_with_logits(
            pred_noobj, tgt_noobj, reduction="sum"
        ) / num_noobj # Divide the loss by the number of TN not by the TP (object-containing) cells. Otherwise the divisor will be tiny -> NaN for no-obj loss (gradient descent doesn’t understand intention, only magnitude).
    else:
        no_obj_loss = pred_conf.new_tensor(0.0)

    # Log: inspect raw class predictions before loss (how confident the model is about class predictions)
    with torch.no_grad():
        pred_probs = torch.sigmoid(pred_class)  # convert logits to probabilities
        pred_max_probs, pred_class_idx = pred_probs.max(dim=-1)  # per anchor and grid prediction

    unique_pred_classes = torch.unique(pred_class_idx)
    logger.info(f"[I] Predicted class indices (argmax): {unique_pred_classes}")
    logger.info(f"[I] Max class probabilities - min: {pred_max_probs.min():.4f}, max: {pred_max_probs.max():.4f}")

    if torch.any(target_conf > 0):
        
        # GT class indices from one-hot encoding vectors
        gt_class_indices = target[..., 5:].argmax(dim=-1)       # [B, A, S, S]
        gt_labels = gt_class_indices[object_mask]               # [N]

        # Predicted probabilities from logits
        pred_class_logits = pred_class[object_mask]             # [N, C]
        pred_probs = torch.sigmoid(pred_class_logits)
        pred_labels = pred_probs.argmax(dim=-1)                 # [N]

        logger.info(f"\n[I] GT class indices: {gt_labels}")
        logger.info(f"[I] Predicted class indices: {pred_labels}")
        logger.info(f"[I] Predicted max probs: {pred_probs.max(dim=-1).values}")

    # 4. Classification loss (normalized over positive cells) - gather predicted logits and GT class indices at positive anchors
    pred_class_active = pred_class[object_mask]  # [N_pos, num_classes]
    target_onehot_active = target[..., 5:][object_mask].float() # [N_pos, C] one-hot

    """
    logger.info(f"[I] One-hot encoded vectors representing the true object class for each matched anchor box during training:")
    logger.info(f"[I] GT class indices for: {target_onehot_active.tolist()}")
    logger.info(f"[I] Predicted classes for each matched anchor:")
    logger.info(f"[I] Predicted class argmax: {pred_class_active.argmax(dim=-1).tolist()}")
    """

    # Safety check: ensure true one-hot
    target_onehot_active = (target_onehot_active > 0.5).float()

    # Safety check: clamp logits (BCE stability), intended for early training!
    # pred_class_active = pred_class_active.clamp(-10, 10)

    # Number of positive classes (for the experiment)
    num_pos_anchors = pred_class_active.shape[0]

    if pred_class_active.numel() == 0:
        class_loss = pred_class.new_tensor(0.0)
    else:
        # class_loss = F.binary_cross_entropy_with_logits(pred_class_active, target_onehot_active, reduction="mean")
        # logger.info(f"[I] Cross entropy GT classes: {target_onehot_active.tolist()}")
        # logger.info(f"[I] Cross entropy predicted classes: {pred_class_active.argmax(dim=-1).tolist()}")
        class_loss = F.binary_cross_entropy_with_logits(pred_class_active, target_onehot_active, reduction="sum")
        class_loss = class_loss / (num_pos_anchors + 1e-6) # Only computed for positive anchors (this is practically num_pos -> needs refactoring)

    

    # Interim loss logging for tracking during training
    logger.info(f"[I] Box Loss: {box_loss.item():.4f}, Objectness Loss: {obj_loss.item():.4f}, "
          f"No-Object Loss: {no_obj_loss.item():.4f}, Class Loss: {class_loss.item():.4f}")
    print(f"[I] Box Loss: {box_loss.item():.4f}, Objectness Loss: {obj_loss.item():.4f}, "
          f"No-Object Loss: {no_obj_loss.item():.4f}, Class Loss: {class_loss.item():.4f}")
    
    # ________________________
    # LOSS WEIGHTS (lambdas)
    # ________________________
    
    # Boosts box regression, since "where?" is more important than "what?" in the beginning (this is a zero-sum game for gradient sharing - how to spend the budget?)
    lambda_box = 5.0

    # Objectness is the "glue" between regression and classification. Scales the loss on positive anchors (make true object anchors confident).      
    lambda_obj = 1.0

    # Encourages the model to predicted objects (more FPs) instead of playing safe ("There is no object." - FN) in the early phase of the training.
    # Scales the loss on background anchors (make background anchors unconfident).
    lambda_noobj = 1.0
    
    # The model must learn first where objects are (regression + objectness) before it can learn what they are (classification).
    # Overweighting classification too early risks destabilizing anchor matching and objectness learning.
    lambda_cls = 1.0

    # Weighted total loss
    total_loss = (
        lambda_box * box_loss +
        lambda_obj * obj_loss +
        lambda_noobj * no_obj_loss +
        lambda_cls * class_loss
    ) # / batch_size 
    # Removed after epoch 130, since the loss values were already normalized (the classification branch got much less gradients)
    # Loss magnitude should be independent of batch size

    # Prevent logit saturation (overlapping boxes with 1.0 confidence -> NMS cannot rank) -> regularization on logits
    # Why? Because I apply Adam + OneCycle with YOLOv3 style implementation (more aggressive than the original SGD).
    """
    logit_penalty = 1e-4 * (
        pred_class.pow(2).mean() +
        pred_conf.pow(2).mean()
    )
    total_loss += logit_penalty
    """

    # Introduce only at inference time, so that NMS can make a difference (doesn't affect the training).
    """
    T = 1.5
    class_probs = torch.sigmoid(class_logits / T)
    scores = obj_probs * class_probs
    """

    # Safety check: replace NaNs with zero
    if not torch.isfinite(total_loss):
        # logger.warning("[W] Total loss is NaN. Zeroing out.")
        # total_loss = pred_conf.new_tensor(0.0)
        raise RuntimeError("[E] NaN loss detected!")

    # Safety check: intermediate losses (legacy code - not really useful in late, converged training stages)
    for name, val in {
        "box_loss": box_loss, "obj_loss": obj_loss,
        "no_obj_loss": no_obj_loss, "class_loss": class_loss
    }.items():
        if not torch.isfinite(val):
            logger.warning(f"[W] {name} is non-finite!")

    return total_loss