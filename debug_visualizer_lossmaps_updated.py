"""
Minimal YOLOv3 Debug Visualizer (consistent with yolo_loss.py).
Safe to run during training.

Loads a checkpoint, picks an image, generates:
    - Target objectness heatmaps (52/26/13)
    - Predicted objectness heatmaps (52/26/13)
    - Predicted class-confidence heatmaps
    - Decoded prediction overlay on the image
    - Loss heatmaps computed in the SAME coordinate system as yolo_loss.py:
        xy loss in pixel space ((sigmoid(xy)+grid) * stride)
        wh loss in log-space using anchors + exp(tw/th)
        obj/cls maps shown for POSITIVES ONLY (debug sanity)
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import warnings

from YOLO_with_ResNet50 import YOLOv3
from data_loader import COCO_Dataset
from helper import generate_yolo_targets_global
from torchvision import transforms

warnings.filterwarnings("ignore", category=FutureWarning)


# -----------------------------
# Utility
# -----------------------------
def to_numpy_image(t):
    """Undo normalization and convert tensor -> numpy image."""
    img = t.detach().cpu().permute(1, 2, 0).numpy()
    img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


def plot_heatmap(data, title):
    plt.figure(figsize=(5, 5))
    plt.imshow(data, cmap="magma")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.show()


def visualize_loss_heatmap(loss_map, title):
    loss_map = loss_map.detach().cpu().numpy()
    plt.figure(figsize=(5, 5))
    plt.imshow(loss_map, cmap="inferno")
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Loss maps (aligned with yolo_loss.py)
# -----------------------------
def visualize_loss_maps(output, target, anchors, num_classes, scale_name, img_size=416):
    """
    output: [B, C, S, S]  raw model output for one scale
    target: [B, A, S, S, 5+num_classes]  from generate_yolo_targets_global
    anchors: 3 anchors for THIS scale, GRID-RELATIVE (pixel/stride), e.g. scaled_anchors[:3]
    """

    B, C, S, _ = output.shape
    A = 3
    stride = img_size / S

    # target [B,A,S,S,...] -> [B,S,S,A,...]
    target = target.permute(0, 2, 3, 1, 4).contiguous()

    # pred [B,C,S,S] -> [B,S,S,A,5+num_classes]
    pred = (
        output
        .permute(0, 2, 3, 1)
        .contiguous()
        .view(B, S, S, A, 5 + num_classes)
    )

    # anchors: grid-relative (pixel/stride)
    anchors_t = torch.tensor(anchors, device=output.device, dtype=torch.float32).view(1, 1, 1, A, 2)

    # grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(S, device=output.device),
        torch.arange(S, device=output.device),
        indexing="ij"
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, S, S, 1, 2).float()

    # -----------------
    # Predictions (YOLO)
    # -----------------
    pred_xy = torch.sigmoid(pred[..., 0:2])                 # [0,1] within cell
    pred_twth = pred[..., 2:4].clamp(-4.0, 4.0)             # stabilize
    pred_obj = pred[..., 4]                                 # logits
    pred_cls = pred[..., 5:].clamp(-10, 10)                 # stabilize BCE debug

    # Pixel-space centers (matches your yolo_loss.py)
    pred_xy_px = (pred_xy + grid) * stride                  # [B,S,S,A,2]

    # wh in grid units then pixels
    pred_wh_grid = anchors_t * torch.exp(pred_twth)         # [B,S,S,A,2]
    pred_wh_px = pred_wh_grid * stride

    # -------------
    # Targets (YOLO)
    # -------------
    tgt_xy = target[..., 0:2]                               # offsets in [0,1] within cell
    tgt_twth = target[..., 2:4].clamp(-4.0, 4.0)            # targets store log-space tw/th
    tgt_obj = target[..., 4]                                # 0/1
    tgt_cls = target[..., 5:]                               # one-hot

    obj_mask = tgt_obj > 0

    # Pixel-space GT centers (same formula)
    tgt_xy_px = (tgt_xy + grid) * stride

    # GT wh in grid units then pixels
    tgt_wh_grid = anchors_t * torch.exp(tgt_twth)
    tgt_wh_px = tgt_wh_grid * stride

    # quick debug prints
    print(f"[I][{scale_name}] positives:", int(obj_mask.sum().item()))

    # -------------------------
    # LOSS COMPONENTS (per-cell)
    # -------------------------

    # XY loss in pixel space
    xy_loss = ((pred_xy_px - tgt_xy_px) ** 2).sum(-1)       # [B,S,S,A]
    xy_loss = xy_loss * obj_mask

    # WH loss in log-space (consistent with your yolo_loss.py intent)
    log_pred = torch.log(torch.clamp(pred_wh_grid / anchors_t, min=1e-6))
    log_tgt  = torch.log(torch.clamp(tgt_wh_grid  / anchors_t, min=1e-6))
    wh_loss = ((log_pred - log_tgt) ** 2).sum(-1)           # [B,S,S,A]
    wh_loss = wh_loss * obj_mask

    box_loss = xy_loss + wh_loss

    # Objectness loss: POSITIVES ONLY (debug clarity; your training uses ignore-mask on negatives)
    obj_loss = F.binary_cross_entropy_with_logits(pred_obj, tgt_obj, reduction="none")
    obj_loss = obj_loss * obj_mask

    # Class loss: POSITIVES ONLY
    cls_loss = F.binary_cross_entropy_with_logits(pred_cls, tgt_cls, reduction="none").sum(-1)
    cls_loss = cls_loss * obj_mask

    # Reduce anchors -> grid maps
    box_map = box_loss[0].sum(dim=-1)   # [S,S]
    obj_map = obj_loss[0].sum(dim=-1)   # [S,S]
    cls_map = cls_loss[0].sum(dim=-1)   # [S,S]

    # Weighted total map (similar to your training lambdas)
    total_map = 5.0 * box_map + 2.0 * obj_map + 1.0 * cls_map

    visualize_loss_heatmap(box_map,   f"{scale_name} Box Loss (pos-only)")
    visualize_loss_heatmap(obj_map,   f"{scale_name} Obj Loss (pos-only)")
    visualize_loss_heatmap(cls_map,   f"{scale_name} Class Loss (pos-only)")
    visualize_loss_heatmap(total_map, f"{scale_name} TOTAL Loss (weighted, pos-only)")


# -----------------------------
# 1: Visualize TARGET OBJECTNESS
# -----------------------------
def visualize_target_objectness(targets, scale_name):
    # targets shape: [B, 3, S, S, 5+num_classes]
    obj = targets[..., 4].cpu().numpy()       # [B,3,S,S]
    obj = obj[0].max(axis=0)                  # [S,S]
    plot_heatmap(obj, f"TARGET Objectness – {scale_name}")


# -----------------------------
# 2: Visualize PREDICTED OBJECTNESS
# -----------------------------
def visualize_pred_objectness(output, num_classes, scale_name):
    # output shape: [B, 3*(5+num_classes), S, S]
    B, C, S, _ = output.shape

    pred = (
        output
        .permute(0, 2, 3, 1)                  # (B,S,S,C)
        .contiguous()
        .view(B, S, S, 3, 5 + num_classes)    # (B,S,S,3,5+C)
    )

    obj = torch.sigmoid(pred[..., 4]).cpu().numpy()  # (B,S,S,3)
    obj = obj[0].max(axis=-1)                        # (S,S)
    plot_heatmap(obj, f"PRED Objectness – {scale_name}")


# -----------------------------
# 3: Visualize PREDICTED CLASS CONFIDENCE
# -----------------------------
def visualize_pred_class_confidence(output, num_classes, scale_name):
    B, C, S, _ = output.shape

    pred = (
        output
        .permute(0, 2, 3, 1)                  # (B,S,S,C)
        .contiguous()
        .view(B, S, S, 3, 5 + num_classes)    # (B,S,S,3,5+C)
    )

    cls = torch.sigmoid(pred[..., 5:])        # (B,S,S,3,C)
    cls_max = cls.max(dim=-1).values          # (B,S,S,3)
    cls_map = cls_max[0].max(dim=-1).values.cpu().numpy()  # (S,S)
    plot_heatmap(cls_map, f"PRED Class Confidence – {scale_name}")


# -----------------------------
# 4: Visualize FINAL DECODED PREDICTIONS
# -----------------------------
def visualize_final_predictions(orig_img, predictions, coco_classes):
    img = orig_img.copy()
    plt.figure(figsize=(7, 7))
    plt.imshow(img)
    ax = plt.gca()

    for box, score, label in zip(predictions["boxes"], predictions["scores"], predictions["labels"]):
        x1, y1, x2, y2 = box.cpu().numpy().tolist()
        cls = coco_classes.get(int(label), str(int(label)))

        ax.add_patch(
            plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="cyan",
                linewidth=1,
            )
        )
        ax.text(
            x1,
            max(0, y1 - 5),
            f"{cls}: {score:.2f}",
            color="yellow",
            fontsize=8,
            bbox=dict(facecolor="black", alpha=0.6),
        )

    plt.title("Final Predictions (After NMS)")
    plt.tight_layout()
    plt.show()


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="yolov3_checkpoint_last_epoch.pth")
    parser.add_argument("--image-id", type=int, default=None)
    parser.add_argument("--random", action="store_true")
    args = parser.parse_args()

    # Load COCO dataset (val)
    coco_root = r"C:\Users\viktor.acs\Downloads\coco_dataset"
    val_img_dir = os.path.join(coco_root, "val2017")
    val_ann_file = os.path.join(coco_root, "annotations", "instances_val2017.json")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((416, 416)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = COCO_Dataset(val_img_dir, val_ann_file, transform=transform)

    # Select image
    if args.random:
        idx = np.random.randint(0, len(dataset))
    elif args.image_id is not None:
        idx = dataset.image_ids.index(args.image_id)
    else:
        idx = 0

    sample = dataset[idx]
    while sample is None:
        print(f"[W] dataset[{idx}] returned None. Choosing another index...")
        idx = np.random.randint(0, len(dataset))
        sample = dataset[idx]

    img_t, boxes_t, labels_t, orig_size, image_id = sample
    print(f"[I] Loaded image ID {image_id}")

    print("[I] boxes_t min/max:", boxes_t.min().item(), boxes_t.max().item())
    print("[I] labels_t unique:", torch.unique(labels_t))
    print("[I] orig_size (H, W):", orig_size)

    # Load model
    num_classes = 80  # fixed

    # K-means anchors (pixel units)
    anchors = [
        [19.39769772, 24.12491592], [40.91803117, 76.44313414], [113.50188071, 69.27488936],
        [71.71268453, 161.95265297], [105.08463053, 285.58598963], [191.49338089, 161.76429439],
        [351.54059713, 159.67305114], [224.41982096, 331.08204223], [381.16142647, 359.7244103]
    ]

    model = YOLOv3(num_classes=num_classes, anchors=anchors)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # Build class name dict
    coco_classes = {dataset.coco_id_to_model_index[c["id"]]: c["name"]
                    for c in dataset.coco.loadCats(dataset.coco.getCatIds())}

    # Prepare scaled anchors (grid-relative: pixel/stride)
    strides = [8, 16, 32]
    scaled_anchors = (
        [(w / 8,  h / 8)  for w, h in anchors[:3]] +
        [(w / 16, h / 16) for w, h in anchors[3:6]] +
        [(w / 32, h / 32) for w, h in anchors[6:]]
    )
    print("[I] scaled_anchors small:", scaled_anchors[:3])
    print("[I] scaled_anchors med:  ", scaled_anchors[3:6])
    print("[I] scaled_anchors large:", scaled_anchors[6:])

    # Generate targets for this one image
    pad_boxes = boxes_t.unsqueeze(0)     # [1,N,4] in resized 416 space
    pad_labels = labels_t.unsqueeze(0)   # [1,N]

    # Convert corner -> normalized center format using ORIGINAL size (consistent with your latest fix)
    x1 = pad_boxes[..., 0]
    y1 = pad_boxes[..., 1]
    x2 = pad_boxes[..., 2]
    y2 = pad_boxes[..., 3]

    # The decode_predictions() function only handles 416x416 correctly after the decode fix
    H, W = 416, 416
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    w  = (x2 - x1) / W
    h  = (y2 - y1) / H

    norm_boxes = torch.stack([cx, cy, w, h], dim=-1)

    print("[I] norm_boxes min/max:", norm_boxes.min().item(), norm_boxes.max().item())

    t_small, t_med, t_large = generate_yolo_targets_global(
        gt_boxes=norm_boxes,
        class_labels=pad_labels,
        anchors=scaled_anchors,
        grid_sizes=[52, 26, 13],
        num_classes=num_classes
    )

    # target objectness maps
    visualize_target_objectness(t_small, "52x52")
    visualize_target_objectness(t_med,  "26x26")
    visualize_target_objectness(t_large, "13x13")

    # Model predictions
    with torch.no_grad():
        img_in = img_t.unsqueeze(0)  # [1,3,416,416]
        outputs = model(img_in)

    # predicted maps
    visualize_pred_objectness(outputs[0], num_classes, "52x52")
    visualize_pred_objectness(outputs[1], num_classes, "26x26")
    visualize_pred_objectness(outputs[2], num_classes, "13x13")

    visualize_pred_class_confidence(outputs[0], num_classes, "52x52")
    visualize_pred_class_confidence(outputs[1], num_classes, "26x26")
    visualize_pred_class_confidence(outputs[2], num_classes, "13x13")

    # Decode predictions (must use scaled_anchors if your decode expects grid-relative)
    # The function only handles 416x416 correctly after the decode fix
    preds = model.decode_predictions(
        outputs,
        anchors=scaled_anchors,
        num_classes=num_classes,
        image_w = 416,
        image_h = 416,
        conf_threshold=0.5,
        nms_threshold=0.5
    )[0]

    # Print first few positive target locations
    print("[I] t_small obj locations:", torch.nonzero(t_small[..., 4] > 0)[:10])
    print("[I] t_med   obj locations:", torch.nonzero(t_med[..., 4] > 0)[:10])
    print("[I] t_large obj locations:", torch.nonzero(t_large[..., 4] > 0)[:10])

    orig_img = to_numpy_image(img_t)
    visualize_final_predictions(orig_img, preds, coco_classes)

    print("[I] Debug visualization completed.")

    # IMPORTANT: loss maps must use SCALED anchors (grid-relative), not raw pixel anchors
    visualize_loss_maps(outputs[0], t_small, scaled_anchors[:3],  num_classes, "52x52")
    visualize_loss_maps(outputs[1], t_med,   scaled_anchors[3:6], num_classes, "26x26")
    visualize_loss_maps(outputs[2], t_large, scaled_anchors[6:],  num_classes, "13x13")


if __name__ == "__main__":
    main()
