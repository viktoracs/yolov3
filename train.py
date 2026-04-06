import os
import gc
import torch
import time
import tracemalloc
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.backends.cudnn as cudnn
import warnings
from tqdm import tqdm
from pycocotools.coco import COCO
from sklearn.cluster import KMeans
from PIL import Image
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
from torch.optim.lr_scheduler import OneCycleLR
from YOLO_with_ResNet50 import YOLOv3
from data_loader import COCO_Dataset
from yolo_loss import yolo_loss
from evaluate import run_evaluation_after_training
from helper import compute_iou, generate_yolo_targets_global
from logger import logger

# Latest fixes and improvements:
"""
| Component                              
| ---------------------------
| K-means anchors
| Corner–center bug fix (target generator)
| tw/th clamping
| YOLO-style biases
| FPN (P3/P4/P5)
| Center loss (x,y) in pixel space
| Stable loss weights
| Ignore mask to overlapping anchors
| FP32 instead of FP16 for numerical stability (deleted AMP)
| Correct class normalization (/num_pos_class, removed /batch_size = double normalization)
"""

# ======
# INTRO:
# ======

# Python (Conda) virtual environment: "yolov3_env"

# Suppress expendable_segments warning on Windows (it works on Ubuntu)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="expandable_segments not supported")

# Helper (directly in the "yolov3_env" terminal after typing "python")
# Forces the epoch number manually if needed (after a crash).
"""
import torch
ckpt = torch.load("yolov3_checkpoint_last_epoch.pth", map_location="cpu")
ckpt["epoch"] = 20  # restore correct counter
torch.save(ckpt, "yolov3_checkpoint_last_epoch.pth")
print("Epoch number corrected to 20")
"""

# Checks the actual epoch number for the saved model:
"""
import torch
ckpt = torch.load("yolov3_checkpoint_last_epoch.pth", map_location="cpu")
print(ckpt.keys())   
print("Epoch:", ckpt.get('epoch'))
"""

# Regarding the benchmarking: COCO-style mAP@[0.5:0.95] is much stricter than mAP@[0.5] (aka Pascal VOC style)
# Differences:
# Pascal VOC metric (mAP@[0.5]) - 20 classes, IoU threshold fixed at 0.5
# COCO metric (mAP@[0.5:0.95]) - 80 classes, IoU thresholds from 0.5 to 0.95 in increments of 0.05 (10 thresholds total)

# This makes cuDNN auto-select the fastest kernels for the fixed input size (416×416). Only safe if the input resolution is constant.
cudnn.benchmark = True


# ==============
# CURRENT SETUP:
# ==============

"""
-> subset_size = None
-> num_epochs = the actual starting point
-> validation interval = every x epochs
-> batch_size = 32
-> accumulation_steps = 1

Initial default loss weights:
lambda_box = 5.0
lambda_obj = 2.0
lambda_noobj = 1.0 
lambda_cls = 1.0   

Optimizer = Adam(model.parameters(), lr=1e-3) - Why? 

Because of the "zebra" overfit results (sanity test before going for full training):

    Zebra Overfit 3 models:
    -> Old LR, with uneven loss curve - mAP 0.9, high confidence score
    -> New LR 1e-4 with very smooth loss curve - mAP 0.8, lower confidence score
    -> New LR 1e-3 with more even loss curve - mAP 0.9, higher confidence score (selected for the first full training):

    LR:
    Notation    Decimal Equivalent	    Description
    1e-1	    0.1	                    Large LR (often too aggressive)
    1e-2	    0.01	                Still relatively large
    1e-3	    0.001	                Standard for many deep learning setups (like the current config)
    1e-4	    0.0001	                Smaller (slower but safer convergence)
    1e-5	    0.00001	                Very small (often used for fine-tuning)

"""

# ==========
# FUNCTIONS:
# ==========

# K-means for anchor box calculation from COCO annotations (only for demo, there is a separate k_means_anchor_calculator.py file).
# It reads all annotations (data['annotations']) and extracts every bbox. 
# Call it using batch_size = 1, otherwise collate_fn() will pad the boxes and distort the results. max_images is an optional parameter (max limit). 

def compute_anchors(dataset, num_clusters=9, target_size=416, max_images=None): 

    bboxes = []
    count = 0

    for image, boxes, labels, orig_size, _ in tqdm(dataset, desc="Collecting boxes for K-means"):
        orig_h, orig_w = orig_size

        for box in boxes:
            x_min, y_min, x_max, y_max = box.tolist()
            w = (x_max - x_min)
            h = (y_max - y_min)

            # Normalize and scale to target resolution (e.g. 416x416)
            norm_w = (w / orig_w) * target_size
            norm_h = (h / orig_h) * target_size

            bboxes.append([norm_w, norm_h])

        count += 1
        if max_images is not None and count >= max_images:
            break

    bboxes = np.array(bboxes)
    logger.info(f"[I] Total GT boxes collected: {len(bboxes)}")

    if len(bboxes) < num_clusters:
        raise ValueError(f"[E] Not enough bounding boxes to form {num_clusters} clusters")

    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(bboxes)
    anchors = kmeans.cluster_centers_

    # Sort anchors by area
    anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]

    logger.info("\n[I] Final sorted anchors:")
    for i, (w, h) in enumerate(anchors):
        logger.info(f"[I] Anchor {i}: width = {w:.1f}, height = {h:.1f}, area = {w*h:.1f}")

    return anchors.tolist()

# Custom collate function to handle variable number of bboxes. Why? PyTorch needs same size tensors in a batch.
def collate_fn(batch):
    """
    param batch: is a tuple of these 5 tensors:
    - images: C, H, W
    - boxes: x_min, y_min, x_max, y_max (corner format)
    - labels: tensor of class labels corresponding to bounding boxes
    - original_sizes: (height, width) of each image
    - image_ids: image IDs

    After zip(*batch), the structure becomes:
    batch = [(img1, boxes1, labels1, orig_size1, id1), (img2, boxes2, labels2, orig_size2, id2)]
    - images = (img1, img2)
    - boxes = (boxes1, boxes2)
    - labels = (labels1, labels2)
    - original_sizes = (orig_size1, orig_size2)
    - image_ids = (id1, id2)
    """

    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        raise ValueError("[E] All items in the batch were filtered out.")

    images, boxes, labels, original_sizes, image_ids = zip(*batch)

    # Safety check: handle single int values (e.g. 480 treat as [480, 480] - square image)
    fixed_original_sizes = []
    for os in original_sizes:
        if isinstance(os, tuple) and len(os) == 2:
            fixed_original_sizes.append(os)
        elif isinstance(os, list) and len(os) == 2:
            fixed_original_sizes.append(tuple(os))
        elif isinstance(os, int):
            logger.warning(f"[W] Detected int instead of tuple for original_size: {os}")
            fixed_original_sizes.append((os, os))
        else:
            logger.error(f"[E] Invalid original_size in collate_fn: {os}")
            fixed_original_sizes.append((416, 416))

    # Convert labels to tensors
    # label.clone() - deep copy: prevents modifications to the original label
    # detach() - detaches the tensor from the current computation graph (no gradient tracking)
    # long() - casts the tensor to torch.int64, the standard dtype for class labels in PyTorch
    processed_labels = [label.clone().detach().long() for label in labels]
    
    # Stack images
    images = torch.stack(images, dim=0)

    # Padding logic
    max_boxes = max(box.shape[0] for box in boxes) # Finds the maximum number of bounding boxes across all images in the batch
    padded_boxes = torch.zeros((len(boxes), max_boxes, 4), dtype=torch.float32) # Creates a zero-filled tensor to hold all bounding boxes
    padded_labels = torch.full((len(labels), max_boxes), fill_value=-1, dtype=torch.long) # Creates a tensor filled with -1 to hold class labels (Why? Person is class 0, so -1 indicates padded values)

    for i in range(len(boxes)):
        if boxes[i].shape[0] > 0:
            padded_boxes[i, :boxes[i].shape[0], :] = boxes[i].clone().detach().float()
            padded_labels[i, :labels[i].shape[0]] = processed_labels[i]

    padded_labels = padded_labels.to(torch.int64)

    return images, padded_boxes, padded_labels, fixed_original_sizes, image_ids
    
# Normalize GT bboxes
def normalize_boxes(boxes, image_width, image_height):
    """
    Normalizes GT bboxes relative to the image dimensions.
    
    boxes: GT bboxes in (x_min, y_min, x_max, y_max) corner format
    image_width: width of the images
    image_height: height of the images
    
    return: normalized bounding boxes in (cx, cy, w, h) format - the model expects it (values in [0,1])
    
    YOLO format: class_id | x_center | y_center | width | height
    """
    normalized_boxes = boxes.clone()

    # Convert from (x_min, y_min, x_max, y_max) to (cx, cy, w, h)
    normalized_boxes[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2  # cx = (x_min + x_max) / 2   (cx = x the center of the bbox 1. coordinate)
    normalized_boxes[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2  # cy = (y_min + y_max) / 2  (cy = y the center of the bbox 2. coordinate)
    normalized_boxes[..., 2] = boxes[..., 2] - boxes[..., 0] # calculating the width
    normalized_boxes[..., 3] = boxes[..., 3] - boxes[..., 1] # calculating the height


    # Normalize to the range [0, 1]
    normalized_boxes[..., 0] /= image_width
    normalized_boxes[..., 1] /= image_height
    normalized_boxes[..., 2] /= image_width
    normalized_boxes[..., 3] /= image_height
    
    # Safety check: clamp values to ensure they are within [0, 1], should it it overlap the picture
    normalized_boxes[..., :4] = torch.clamp(normalized_boxes[..., :4], 0.0, 1.0)

    # Returns a tensor
    return normalized_boxes

def main():

    # Enables expandable segments (reduces fragmentation). Added as environment variable but it doesn't work on Windows. Prevents GPU OOM.
    """
    print("[I] PYTORCH_CUDA_ALLOC_CONF =", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
    
    # Apply these to surpress warnings if used on Windows:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="expandable_segments not supported")
    """
    # ==============
    # Path settings:
    # ==============

    # To ensure reproducibility across runs
    # torch.manual_seed(42)

    # COCO dataset
    coco_data_dir = r"C:\\Users\\viktor.acs\\Downloads\\coco_dataset"

    # Training dataset and DataLoader
    train_annotations_file = os.path.join(coco_data_dir, 'annotations', 'instances_train2017.json')
    train_image_dir = os.path.join(coco_data_dir, 'train2017')

    # Validation dataset and DataLoader. 
    # For overfitting on one image, comment these two lines:
    val_annotations_file = os.path.join(coco_data_dir, 'annotations', 'instances_val2017.json')
    val_image_dir = os.path.join(coco_data_dir, 'val2017')

    # For overfitting on one image, uncomment these two lines:
    # val_annotations_file = train_annotations_file  # For overfitting on one image found in the training set
    # val_image_dir = train_image_dir

    # ========
    # Anchors:
    # ========

    # Default anchors for YOLOv3 (based on the study): 1. width, 2. height
    """
    anchors = [
        [10, 13], [16, 30], [33, 23],  # Small-scale objects
        [30, 61], [62, 45], [59, 119], # Medium-scale objects
        [116, 90], [156, 198], [373, 326]  # Large-scale objects
    ]
    """
    # K-means generated by k_means_anchor_calculator.py on the resized COCO dataset (416x416):
    anchors = [
        [19.39769772, 24.12491592], [40.91803117, 76.44313414], [113.50188071, 69.27488936],
        [71.71268453, 161.95265297], [105.08463053, 285.58598963], [191.49338089, 161.76429439],
        [351.54059713, 159.67305114], [224.41982096, 331.08204223], [381.16142647, 359.7244103]
    ]

    # =============================
    # Model checkpointing strategy:
    # =============================
    #
    # 1. "Best" checkpoint -> yolov3_general_checkpoint_best.pth
    #    - Saved during training "only when mAP improves" over previous best.
    #    - Used as the preferred checkpoint to resume training by default.
    #
    # 2. "Last" checkpoint -> yolov3_general_checkpoint_last.pth
    #    - Saved at the "end of training", regardless of performance.
    #    - Used as a fallback if no "best" checkpoint exists.
    #
    # 3. "Last epoch" checkpoint -> yolov3_checkpoint_last_epoch.pth
    #    - Saves the training state after every epoch (to have a fallback to continue from -> manual resume).
    #
    # =========================================================
    # Loading strategy before training (without manual resume):
    # =========================================================
    # - IF best checkpoint exists -> resume from best model
    # - ELSE IF last checkpoint exists -> resume from last training state
    # - ELSE -> initialize a new model and start from scratch
    #
    # Contents of each checkpoint:
    # {
    #     'epoch': (int) epoch to resume from next time
    #     'model_state_dict': model weights
    #     'optimizer_state_dict': optimizer state (e.g. momentum, LR)
    #     'scheduler_state_dict': LR scheduler state (e.g. position in cycle/decay) -> ONLY IF use_scheduler = True 
    #     'best_mAP': best validation score seen so far
    # }
    # ===================================

    # num_epochs = the actual epoch where we are (additional_epochs = how much more to train)
    num_epochs = 100
    accumulation_steps = 1

    checkpoint_best_path = os.path.join(os.getcwd(), "yolov3_general_checkpoint_best.pth")
    checkpoint_last_path = os.path.join(os.getcwd(), "yolov3_general_checkpoint_last.pth")
    manual_ckpt_path = os.path.join(os.getcwd(), "yolov3_general_checkpoint_last.pth")

    # Training loop
    def train(
            model, 
            train_dataloader, 
            optimizer, 
            scheduler, 
            device, 
            num_epochs, 
            accumulation_steps, 
            coco_gt_path, 
            num_classes, 
            scaled_anchors, 
            coco_classes,
            anchors,
            start_epoch=0, 
            best_mAP=0.0
        ):

        # Minimum mAP threshold for saving
        min_mAP_threshold = 0.001
        
        # train() is a built-in function of the NN module. The YOLOv3 class doesn't need to implement it explicitly.
        model.train()
        
        # Early stopping variables for avoding overfitting. It is not applied at the moment, as "epochs_without_improvement = 0" is never greater than the current "patience". 
        # Implement later if needed: IF epoch N's AVG loss > epoch (N-1)'s AVG loss -> epochs_without_improvement + 1
        patience = 10
        epochs_without_improvement = 0  # Ensure initialization here 
        
        # For visualization
        loss_history = []

        # To store memory usage after each epoch
        memory_usage_history = []  

        # To track CUDA memory usage
        cuda_memory_history = []

        # To track IoU
        iou_history = []

        # Enable memory tracking
        tracemalloc.start()
        
        epoch = None

        print(f"\n[I] Starting training loop...\n")

        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0
            optimizer.zero_grad()  # Clear gradients at the start
            start_epoch_time = time.time() 

            # original_sizes is a tuple of integers - not a tensor - cannot be moved to the GPU
            for step, (images, boxes, labels, _, _) in enumerate(train_dataloader):

                # Ensure labels remain unchanged (no need to remap since "data_loader.py" already mapped them).
                labels = labels.to(device)  
                images = images.to(device)
                boxes = boxes.to(device)

                # Safety check: Fix inverted box coordinates
                for b in range(boxes.shape[0]):
                    for i in range(boxes.shape[1]):
                        if labels[b, i] == -1:
                            continue  # Skip padded boxes

                        x_min, y_min, x_max, y_max = boxes[b, i]

                        if x_min > x_max:
                            boxes[b, i, 0], boxes[b, i, 2] = x_max, x_min
                        if y_min > y_max:
                            boxes[b, i, 1], boxes[b, i, 3] = y_max, y_min

                # Mask padded values. Only valid GT boxes are passed into normalization and target generation steps.
                valid_mask = labels != -1 # valid_mask: A tensor with the same shape as labels, marking valid entries
                boxes[~valid_mask] = 0  # ~valid_mask: The inverse of valid_mask, marking padded (invalid) boxes with zeros

                # boxes: (B, N, 4) in absolute 416×416 pixel space
                # labels: (B, N)

                # The normalize_boxes() function was designed for this part. Needs refactoring.
                # Convert xy xy -> normalized cx cy wh for each image in batch
                B, N, _ = boxes.shape
                normalized_boxes = torch.zeros((B, N, 4), device=boxes.device)

                img_h = img_w = 416.0

                x_min = boxes[..., 0]
                y_min = boxes[..., 1]
                x_max = boxes[..., 2]
                y_max = boxes[..., 3]

                # Convert corner to center format and normalize
                # IMPORTANT: generate_yolo_targets_global() expects normalized cx, cy, w, h
                cx = (x_min + x_max) / 2.0 / img_w
                cy = (y_min + y_max) / 2.0 / img_h
                w  = (x_max - x_min) / img_w
                h  = (y_max - y_min) / img_h

                normalized_boxes[..., 0] = cx
                normalized_boxes[..., 1] = cy
                normalized_boxes[..., 2] = w
                normalized_boxes[..., 3] = h
 
                # Generate targets
                targets_small, targets_medium, targets_large = generate_yolo_targets_global(
                    gt_boxes=normalized_boxes,   # [B, N, 4]
                    class_labels=labels,          # [B, N]
                    anchors=scaled_anchors,   
                    grid_sizes=[52, 26, 13],
                    num_classes=num_classes
                )

                for name, target in zip(["small", "medium", "large"], [targets_small, targets_medium, targets_large]):
                    class_slice = target[..., 5:]
                    bbox_offsets = target[..., 0:4]
                        
                    logger.info(f"\n=== DEBUG [{name.upper()}] ===")
                    logger.info(f"[I] Unique class values (should be 0 or 1): {torch.unique(class_slice)}")
                    logger.info(f"[I] BBox Offsets [tx, ty, tw, th] - min: {bbox_offsets.min():.4f}, max: {bbox_offsets.max():.4f}")

                    # Safety checks for NaN and Inf values in targets    
                    assert not torch.isnan(target).any(), f"[E] NaN detected in {name} target tensor!"
                    assert not torch.isinf(target).any(), f"[E] Inf detected in {name} target tensor!"

                # HARD STOP if input is bad (catches NaNs coming from dataset/augmentations)
                if not torch.isfinite(images).all():
                    bad = (~torch.isfinite(images)).sum().item()
                    print(f"[E] Non-finite values in IMAGES: count={bad}")
                    raise RuntimeError("Non-finite images batch (dataset/augmentations)")

                if not torch.isfinite(targets_small).all() or not torch.isfinite(targets_medium).all() or not torch.isfinite(targets_large).all():
                    print("[E] Non-finite values in TARGETS")
                    raise RuntimeError("Non-finite targets batch (label encoding)")

                # Check weights before forward
                for name, param in model.named_parameters():
                    if not torch.isfinite(param).all():
                        print(f"[E] NaN in weights BEFORE forward: {name}")
                        raise RuntimeError("Weights corrupted before forward")

                outputs = model(images)

                # HARD STOP if forward pass is bad (this catches model/AMP/BN blow-ups)
                if isinstance(outputs, (list, tuple)):
                    for i, out in enumerate(outputs):
                        if not torch.isfinite(out).all():
                            bad = (~torch.isfinite(out)).sum().item()
                            print(f"[E] Non-finite values in MODEL OUTPUT[{i}]: count={bad}")
                            raise RuntimeError(f"Non-finite model output at head {i}")
                else:
                    if not torch.isfinite(outputs).all():
                        bad = (~torch.isfinite(outputs)).sum().item()
                        print(f"[E] Non-finite values in MODEL OUTPUT: count={bad}")
                        raise RuntimeError("Non-finite model output")

                # Each tensor represents predictions for all grid cells at a specific scale.
                # Raw model output shape: (B, 255, S, S)
                # Reshaped format for loss: (B, 3, S, S, 85) where 85 = 5 + num_classes
                
                # For the logger objectness and class probabilities -> model.last_predictions = [] - non-loss use only
                reshaped_outputs = [] 
                for out in outputs:
                    B, _, H, W = out.shape
                    reshaped = out.view(B, 3, 5 + num_classes, H, W).permute(0, 1, 3, 4, 2).contiguous()
                    reshaped_outputs.append(reshaped)
    
                # Loss calculation - outputs are in the order: small (52x52), medium (26x26), large (13x13)
                
                # SMALL SCALE
                small_one_hot = targets_small[..., 5:]
                small_classes = small_one_hot.argmax(dim=-1)
                small_obj_mask = targets_small[..., 4] > 0
                small_gt_classes = small_classes[small_obj_mask]
                logger.info(f"[I] Unique Small GT Class Indices Before Loss: {torch.unique(small_gt_classes)}")
                small_loss = yolo_loss(outputs[0], targets_small, scaled_anchors[:3], num_classes, scale_name="small")

                # MEDIUM SCALE
                medium_one_hot = targets_medium[..., 5:]
                medium_classes = medium_one_hot.argmax(dim=-1)
                medium_obj_mask = targets_medium[..., 4] > 0
                medium_gt_classes = medium_classes[medium_obj_mask]
                logger.info(f"[I] Unique Medium GT Class Indices Before Loss: {torch.unique(medium_gt_classes)}")
                medium_loss = yolo_loss(outputs[1], targets_medium, scaled_anchors[3:6], num_classes, scale_name="medium")

                # LARGE SCALE
                large_one_hot = targets_large[..., 5:]
                large_classes = large_one_hot.argmax(dim=-1)
                large_obj_mask = targets_large[..., 4] > 0
                large_gt_classes = large_classes[large_obj_mask]
                logger.info(f"[I] Unique Large GT Class Indices Before Loss: {torch.unique(large_gt_classes)}")
                large_loss = yolo_loss(outputs[2], targets_large, scaled_anchors[6:], num_classes, scale_name="large")
                
                # FINAL LOSS
                loss = (small_loss + medium_loss + large_loss) / accumulation_steps
                logger.info(f"[I] Final Loss ([small + medium + large] / accumulation steps): {loss}\n")
                print(f"[I] Batch {step + 1} Final Loss ([small + medium + large] / accumulation steps): {loss}\n")

                model.last_predictions = []
                for output in reshaped_outputs:
                    if output.requires_grad:
                        output.retain_grad()
                    model.last_predictions.append(output)

                # Inspect objectness and class predictions at GT anchors
                with torch.no_grad():
                    for output, target_tensor, scale_name in zip(outputs, [targets_small, targets_medium, targets_large], ["small", "medium", "large"]):
                        pred_H, pred_W = output.shape[-2:]
                        target_H, target_W = target_tensor.shape[-3:-1]

                        if (pred_H, pred_W) != (target_H, target_W):
                            logger.info(f"[I] Scale {scale_name}: Pred shape={output.shape}, Target shape={target_tensor.shape}")
                            logger.info(f"[W] Shape mismatch: pred {(pred_H, pred_W)}, target {(target_H, target_W)}")
                            continue

                        B, C, H, W = output.shape
                        pred_output = output.view(B, 3, -1, H, W).permute(0, 1, 3, 4, 2)

                        pred_conf = torch.sigmoid(pred_output[..., 4])
                        pred_cls = pred_output[..., 5:]
                        target_conf = target_tensor[..., 4]
                        gt_mask = target_conf == 1

                        if gt_mask.sum() > 0:
                            logger.info(f"[I] Matched GT anchors found at scale {scale_name}")
                            obj_at_gt = pred_conf[gt_mask]
                            logger.info(f"[I] GT objectness min: {obj_at_gt.min():.4f}, max: {obj_at_gt.max():.4f}, mean: {obj_at_gt.mean():.4f}")

                            # Legacy code: Zebra overfit check
                            # gt_class_logits = pred_cls[gt_mask]
                            # gt_class_probs = torch.sigmoid(gt_class_logits) # Convert logits to probabilities
                            # zebra_probs = gt_class_probs[:, 22] 
                            # print(f"[GT Zebra Probs] min: {zebra_probs.min():.4f}, max: {zebra_probs.max():.4f}, mean: {zebra_probs.mean():.4f}") # Zebra overfit check

                        else:
                            logger.info(f"[I] Scale {scale_name.upper()}] no GT anchors matched.")

                # Safety check: skip batch if loss is NaN or has no gradient
                if (not torch.isfinite(loss)) or (not loss.requires_grad):
                    logger.error("[E] NaN DETECTED - total loss is NaN or has no grad. Skipping this batch.")
                    
                    object_mask_small = targets_small[..., 4] > 0
                    corrupted_classes = targets_small[..., 5:].argmax(dim=-1)[object_mask_small]
                    logger.info(f"[I] Classes in this batch: {torch.unique(corrupted_classes)}")
                    
                    optimizer.zero_grad(set_to_none=True)
                    continue

                # Track CUDA memory after forward pass
                allocated = torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
                max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024  # Convert to MB
                cuda_memory_history.append((allocated, max_allocated))
                
                # Backward pass and optimization
                loss.backward()

                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                                       
                    # Training becomes smoother especially with OneCycle LR                
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    
                    optimizer.step()
                    
                    # Only step scheduler if optimizer actually updated
                    if scheduler is not None:
                        scheduler.step()

                        if step % 200 == 0:
                            print(f"[LR] epoch={epoch}, step={step}, "
                                f"lr={optimizer.param_groups[0]['lr']}")

                    optimizer.zero_grad(set_to_none=True)

                    # Check weights AFTER real update
                    for name, param in model.named_parameters():
                        if not torch.isfinite(param).all():
                            print(f"[E] NaN in weights AFTER step: {name}")
                            raise RuntimeError("Weights corrupted after optimizer step")

                    # Track CUDA memory after optimizer step
                    allocated = torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
                    max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024  # Convert to MB
                    cuda_memory_history.append((allocated, max_allocated))
                    
                # Track epoch loss (don't poison the epoch average with inf/nan)
                if torch.isfinite(loss):
                    epoch_loss += loss.item()
                else:
                    logger.warning("[W] Skipping non-finite batch loss for epoch average.")


                # Record CPU memory usage
                current, peak = tracemalloc.get_traced_memory()
                memory_usage_history.append((current / 1024 / 1024, peak / 1024 / 1024))  # Convert to MB

                # Record CUDA memory usage
                allocated = torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
                max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024  # Convert to MB
                cuda_memory_history.append((allocated, max_allocated))

            # Log fix
            num_batches = len(train_dataloader)
            
            avg_epoch_loss = epoch_loss / max(1, num_batches)
            loss_history.append(avg_epoch_loss)
            relative_epoch = epoch - start_epoch + 1
            total_epochs = num_epochs - start_epoch
            logger.info(f"[I] Epoch [{relative_epoch}/{total_epochs}], Average Epoch Loss: {avg_epoch_loss:.4f}")
            print(f"[I] Epoch [{relative_epoch}/{total_epochs}], Average Epoch Loss: {avg_epoch_loss:.4f}")

            # Append a placeholder for IoU so the plot has correct length even if not decoded every epoch
            iou_history.append(None) 

            # ========== SAVE LAST CHECKPOINT (every epoch) ==========
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_mAP': best_mAP
            }, "yolov3_checkpoint_last_epoch.pth")
            print(f"[I] Saved checkpoint for epoch {epoch+1} (last).")
    
            # ==========
            # VALIDATION
            # ==========
            # Validate after x epochs to save resources
            if (epoch + 1) % 100 == 0:  
                logger.info(f"[I] Epoch {epoch + 1} - Running validation...")
                print(f"[I] Epoch {epoch + 1} - Running validation...")
                current_mAP = run_evaluation_after_training(model, val_dataloader, device, val_annotations_file, coco_classes=coco_classes)
                if current_mAP is not None:
                    logger.info(f"[I] Validation mAP@[.5:.95]: {current_mAP:.4f}")
                    print(f"[I] Validation mAP@[.5:.95]: {current_mAP:.4f}")
                else:
                    logger.warning("[W] Validation mAP could not be computed.")
                    print("[W] Validation mAP could not be computed.")

                # Explicit garbage collection after validation
                gc.collect()
                torch.cuda.empty_cache()
                
                # Save the best model (based on mAP)
                if current_mAP > best_mAP and current_mAP > min_mAP_threshold:
                    best_mAP = current_mAP
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'best_mAP': best_mAP
                    }, "yolov3_general_checkpoint_best.pth")
                    print(f"[I] Full checkpoint saved with mAP: {best_mAP:.4f}")

                # Early stopping check (not actiavely used at the moment)
                if epochs_without_improvement >= patience:
                    print(f"[W] Early stopping triggered after {patience} epochs without improvement.")
                    break

            # ================================================================
            # Enhanced 3x3 prediction visualization every x epochs for 416x416
            # ================================================================
            if (epoch + 1) % 100 == 0:
                print(f"[I] Visualizing enhanced 3x3 predictions after epoch {epoch + 1}...")

                model.eval()

                with torch.no_grad():
                    outputs = model(images)

                    # Extract the GT class for debug visualization (first valid label) -> see at "Enhanced 3x3 Prediction Visualization Every x Epochs"
                    gt_label_tensor = labels[0]  # Shape: [max_boxes] padded with -1
                    valid_mask = gt_label_tensor >= 0
                    valid_labels = gt_label_tensor[valid_mask]
                    if valid_labels.numel() > 0:
                        prob_target = valid_labels[0].item()
                    else:
                        prob_target = None
                    
                    # Logit inspection on large scale for logging / debugging
                    last_output = outputs[-1]  # Use large scale output (13x13)
                    B, _, H, W = last_output.shape
                    num_anchors = 3
                    reshaped = last_output.view(B, num_anchors, 85, H, W).permute(0, 3, 4, 1, 2)  # shape: (B, H, W, anchors, 85)

                    if prob_target is not None:
                        logits_class = reshaped[..., 5 + prob_target]
                        class_name = coco_classes.get(prob_target, f"class {prob_target}")
                        score = torch.sigmoid(logits_class)
                        logger.info(f"[I] Raw logits ({class_name}) - max: {logits_class.max().item():.4f}, mean: {logits_class.mean().item():.4f}, >0.5: {(score > 0.5).float().mean().item() * 100:.2f}%")

                    predictions = model.decode_predictions(
                        outputs, 
                        anchors=scaled_anchors, 
                        num_classes=num_classes,
                        image_w=416, 
                        image_h=416,
                        conf_threshold=0.7, # Set the threshold for visualization (overrides the default parameter)
                        nms_threshold=0.5, # Set the threshold for visualization (overrides the default parameter)
                        debug_force_class=None, 
                    )
                
                scales = [52, 26, 13]  # Small, medium, large grid sizes
                strides = [8, 16, 32]  # YOLO strides
                anchor_colors = ["r", "g", "b"]

                image_np = images[0].permute(1, 2, 0).cpu().numpy()
                
                # Not used, but kept for reference
                scale_x = 416.0 / orig_w 
                scale_y = 416.0 / orig_h

                pred = predictions[0] if predictions else {"boxes": [], "scores": [], "labels": []}

                pred_boxes_rescaled = pred["boxes"] # Also used in the final prediction image

                image_np = (image_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
                image_np = (image_np * 255).astype("uint8")

                fig, axes = plt.subplots(3, 3, figsize=(18, 18))

                # Compute absolute anchor sizes
                absolute_anchors = [
                    (w * stride, h * stride) 
                    for stride, anchor_group in zip(strides, [scaled_anchors[:3], scaled_anchors[3:6], scaled_anchors[6:]]) 
                    for w, h in anchor_group
                ]

                # Compute IoU between predictions and GT box
                if len(gt_boxes) > 0 and len(pred["boxes"]) > 0:
                    gt_box_tensor = torch.tensor(gt_boxes, dtype=torch.float32, device=pred["boxes"].device)  # shape: [1, 4]
                    pred_boxes_tensor = pred["boxes"]  # [N, 4]

                    ious = compute_iou(pred_boxes_tensor, gt_box_tensor, pairwise=True)  # [N, 1]
                    best_iou = ious.max().item()
                    logger.info(f"[I] Best IoU with GT box after epoch {epoch + 1}: {best_iou:.4f}")
                    iou_history.append(best_iou)
                    iou_history[-1] = best_iou  # Overwrite placeholder from this epoch
                else:
                    logger.warning(f"[W] No GT boxes or predictions available")
                    iou_history.append(0.0)

                logger.info("\n[I] Predicted boxes and class scores:")
                for box, score, label in zip(pred_boxes_rescaled, pred["scores"], pred["labels"]):
                    logger.info(f"Model Index: {int(label)}, Score: {score:.2f}")


                for i, scale in enumerate(scales):
                    stride = strides[i]

                    for j in range(3):
                        ax = axes[i, j]
                        ax.imshow(image_np)
                        image_h, image_w = image_np.shape[:2]
                        ax.set_xlim([0, image_w])
                        ax.set_ylim([image_h, 0])
                        ax.set_xticks([0, 416])
                        ax.set_yticks([0, 416])
                        ax.set_title(f"Scale {scale}, Anchor {j+1}", fontsize=12)

                        #  Safety check: remove padded boxes based on label == -1
                        gt_boxes_tensor = boxes[0]  # shape: [max_boxes, 4]
                        gt_labels_tensor = labels[0]  # shape: [max_boxes]

                        # Only keep boxes with valid labels
                        valid_mask = gt_labels_tensor != -1
                        gt_boxes_cleaned = gt_boxes_tensor[valid_mask].cpu().tolist()

                        # Draw GT boxes
                        for gt_box in gt_boxes_cleaned:
                            x_min, y_min, x_max, y_max = gt_box
                            ax.add_patch(patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                                        linewidth=1, edgecolor="yellow", facecolor="none", linestyle="dashed"))
                            center_x = (x_min + x_max) / 2
                            center_y = (y_min + y_max) / 2
                            ax.plot(center_x, center_y, marker="o", color="yellow", markersize=5)

                        # Draw Anchor Box
                        anchor_w, anchor_h = absolute_anchors[i * 3 + j]
                        grid_x, grid_y = 5, 5
                        anchor_x = (grid_x + 0.5) * stride
                        anchor_y = (grid_y + 0.5) * stride
                        ax.add_patch(patches.Rectangle(
                            (anchor_x - anchor_w / 2, anchor_y - anchor_h / 2), anchor_w, anchor_h,
                            linewidth=1, edgecolor=anchor_colors[j], facecolor="none"
                        ))

                        # Draw predictions (cyan)
                        for box, score, label in zip(pred_boxes_rescaled, pred["scores"], pred["labels"]):
                            x_min, y_min, x_max, y_max = box.cpu().numpy()
                            model_index = int(label)
                            class_name = coco_classes.get(model_index, "Unknown")
                            label_position_y = max(y_min - 5, 10)
                            class_label = f"{class_name} ({score:.2f})"

                            ax.add_patch(patches.Rectangle(
                                (x_min, y_min), x_max - x_min, y_max - y_min,
                                linewidth=1, edgecolor="cyan", facecolor="none"
                            ))

                            ax.text(x_min, label_position_y, class_label, color="white", fontsize=8,
                                    bbox=dict(facecolor="black", alpha=0.75, edgecolor="none", boxstyle="round,pad=0.2"))
                # =============================
                # END of 3×3 visualization loop
                # =============================

                # Final prediction image
                print("[I] Final prediction image - Rendering summary view (NMS-filtered predictions only)...")

                image_np_clean = images[0].permute(1, 2, 0).cpu().numpy()
                image_np_clean = (image_np_clean * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
                image_np_clean = np.clip(image_np_clean * 255, 0, 255).astype("uint8")

                fig_final, ax_final = plt.subplots(1, 1, figsize=(8, 8))
                ax_final.imshow(image_np_clean)
                final_h, final_w = image_np_clean.shape[:2]
                ax_final.set_xlim([0, final_w])
                ax_final.set_ylim([final_h, 0])
                ax_final.set_xticks([])
                ax_final.set_yticks([])
                ax_final.set_title(f"[I] Final Prediction After Epoch {epoch + 1}", fontsize=14)

                # Draw GT box (optional)
                for gt_box in gt_boxes_cleaned:
                    x_min, y_min, x_max, y_max = gt_box
                    ax_final.add_patch(patches.Rectangle(
                        (x_min, y_min), x_max - x_min, y_max - y_min,
                        linewidth=1, edgecolor="yellow", facecolor="none", linestyle="dashed"
                    ))

                # Draw only NMS-filtered predictions
                for box, score, label in zip(pred_boxes_rescaled, pred["scores"], pred["labels"]):
                    x_min, y_min, x_max, y_max = box.cpu().numpy()
                    model_index = int(label)
                    class_name = coco_classes.get(model_index, f"class {model_index}")
                    label_y = max(y_min - 5, 10)

                    ax_final.add_patch(patches.Rectangle(
                        (x_min, y_min), x_max - x_min, y_max - y_min,
                        linewidth=1, edgecolor="cyan", facecolor="none"
                    ))

                    ax_final.text(x_min, label_y, f"{class_name} ({score:.2f})", color="white", fontsize=9, bbox=dict(facecolor="black", alpha=0.7, edgecolor="none", boxstyle="round,pad=0.2"))

                plt.tight_layout()
                plt.show()
                model.train()  # Switch back to training mode
                
            # Duration of the epoch in seconds
            end_epoch_time = time.time()
            duration = end_epoch_time - start_epoch_time
            logger.info(f"[I] Epoch {epoch+1} took {duration:.2f} seconds.")
            print(f"[I] Epoch {epoch+1} took {duration:.2f} seconds.")
            
        # Stop memory tracking    
        tracemalloc.stop()

        # Save final model checkpoint (after last epoch - regardless of mAP).
        if epoch is not None:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_mAP': best_mAP
            }, "yolov3_general_checkpoint_last.pth")
            print("[I] SAVE: Final checkpoint saved after training.")
        else:
            print("[I] Training loop never ran — no checkpoint saved.")
        
        return loss_history, memory_usage_history, cuda_memory_history, iou_history

    def reset_weights(m):
        if isinstance(m, nn.Conv2d):
            m.reset_parameters()

    # ===========
    # MAIN LOGIC:
    # ===========

    # Define transformations for preprocessing (target generation)
    """
    This pipeline:
    1. Converts the image (and only the image) to a PIL format (GT boxes are scaled to 416x416 in the COCO_Dataset class).
    2. Resizes the image to maintain aspect ratio (shortest side = 416) and pads it to 416x416.
    3. Converts the image to a PyTorch tensor.
    4. Normalizes pixel values to match ImageNet pretraining (ResNet-50 was trained on ImageNet).
    """
    
    # No augmentation
    val_transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert the NumPy arrays into PIL images
        transforms.Resize((416, 416), interpolation=Image.BILINEAR),  # Resize the image directly to 416x416
        transforms.ToTensor(),  # Convert the image to tensor (C, H, W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Light photometric augmentation
    # Double color augmentation (ColorJitter) if applied, since data_loader.py already handles this (HSV)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert the NumPy arrays into PIL images
        transforms.Resize((416, 416), interpolation=Image.BILINEAR),  # Resize the image directly to 416x416
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)], p=0.7),
        transforms.ToTensor(),  # Convert the image to tensor (C, H, W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Geometric augmentations (flipping, etc.) change the pixels, thus the bboxes must be aligned accordingly (handled in data_loader.py)
    
    train_dataset = COCO_Dataset(
        image_dir=train_image_dir, 
        annotation_file=train_annotations_file, 
        transform=val_transform,
        subset_size=None,  # Set to "1" for memorizing one image (zebra overfit test)
        fixed_image_id=None # Use the fixed image ID to ensure the same image is used for training and validation (zebra overfit test)
    )

    # stride (in pixels) = 416 / grid_size
    strides = [8, 16, 32]  

    # Small (52x52) -> Stride 8, Medium (26x26) -> Stride 16, Large (13x13) -> Stride 32
    # scaled_anchors are provided in grid units

    scaled_anchors = [
        (w / strides[0], h / strides[0]) for w, h in anchors[:3]   # Small scale (52x52)
    ] + [
        (w / strides[1], h / strides[1]) for w, h in anchors[3:6]  # Medium scale (26x26)
    ] + [
        (w / strides[2], h / strides[2]) for w, h in anchors[6:]   # Large scale (13x13)
    ]

    img_dim = 416.0

    # Normalize anchors to [0, 1] - not used in training (just for reference) 
    normalized_anchors = [(w / img_dim, h / img_dim) for w, h in anchors]  

    # Reverse mapping from model index (0–79) back to COCO category IDs (just for reference)
    model_index_to_coco_id = {v: k for k, v in train_dataset.coco_id_to_model_index.items()}

    # Extract COCO category IDs from the COCO_dataset instance
    coco_category_ids = list(train_dataset.coco_id_to_model_index.keys())

    # Hardcode or set dynamically the number of classes
    num_classes = 80 # dynamic version: len(coco_category_ids)

    # Extract mapping
    coco_id_to_model_index = train_dataset.coco_id_to_model_index 

    # Create a dictionary to map model indices to object names (use correct model indices)
    coco_classes = {coco_id_to_model_index[cat["id"]]: cat["name"] for cat in train_dataset.coco.loadCats(coco_category_ids)}
    """
    Example:

    Part 1:
    [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "bicycle"},
    {"id": 3, "name": "car"},
    {"id": 5, "name": "airplane"}
    ]

    Part 2 (COCO ID -> model indices):
    coco_id_to_model_index = {1: 0, 2: 1, 3: 2, 5: 3}

    Part 3 (coco_classes):
    {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "airplane"
    }
    """

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        collate_fn=collate_fn, 
        num_workers=6, # Check the benchmarking results down below
        pin_memory=True, # Speed up CPU to GPU transfer
        persistent_workers=True # Small wins (The dataloader workers stay alive across epochs, so the startup overhead is gone. The effect is more visible for many short epochs.)
    )

    # Legacy code: train_dataset.show_fixed_image_with_gt_bb() # Only for a single image (zebra overfit test)

    # Load validation dataset
    val_dataset = COCO_Dataset(
        image_dir=val_image_dir, 
        annotation_file=val_annotations_file, 
        transform=val_transform,
        subset_size=None,  
        fixed_image_id=None
    )

    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=6,
        pin_memory=True, 
        persistent_workers=True
    )

    logger.info(f"[I] Check mapping train_dataset.coco_id_to_model_index: {train_dataset.coco_id_to_model_index}")
    logger.info(f"[I] Sorted keys: {sorted(train_dataset.coco_id_to_model_index.keys())}")              
    logger.info(f"[I] Check mapping val_dataset.coco_id_to_model_index: {val_dataset.coco_id_to_model_index}")
    logger.info(f"[I] Sorted keys: {sorted(val_dataset.coco_id_to_model_index.keys())}")   
    
    # Initialize the model and the optimizer (with the default anchors in absolute, raw pixel units - see below)
    model = YOLOv3(num_classes=num_classes, anchors=anchors)

    """
    Anchor boxes must be carefully scaled to match the coordinate system of each stage in the pipeline:

    Component                       | Expected Anchor Format                               | Reason
    ------------------------        |------------------------------------                  |---------------------------------------------------------------------
    YOLOv3 model init               | Raw pixel anchors                                    | Used to initialize the head modules (not involved in matching logic).
    generate_yolo_targets_global()  | Grid-relative "scaled_anchors" (pixel / stride)      | GT boxes are in pixel units, but model predicts offsets in grid units.
    yolo_loss()                     | Grid-relative "scaled_anchors" (pixel / stride))     | Predictions and targets live in the same grid-relative space.
    decode_predictions()            | Grid-relative "scaled_anchors" (pixel / stride)      | Predictions are in grid-relative space, need to map back to pixel space.
    """
    # Print for debugging
    logger.info("[I] Anchor structure and scales:")
    for i, (aw, ah) in enumerate(model.anchors):
        scale = ["small", "medium", "large"][i // 3]
        logger.info(f"[I] Anchor {i} → Scale: {scale}, Size: ({aw} × {ah})")

    # Keep it only if you are explicitly starting from a clean slate (e.g. new experiment) or wrap it:
    if not os.path.exists(checkpoint_best_path) and not os.path.exists(checkpoint_last_path):
        model.apply(reset_weights)
        print("[I] No checkpoints found — model weights were reset.")

    # Print the number of output channels in each detection head
    for name, param in model.named_parameters():
        if "det_head" in name and "weight" in name:
            print(f"[I] Checking the channels {name}: out_channels = {param.shape[0]}")

    # ===========
    # Defining LR
    # START

    """
    Examples:

    # Updates the weights of the model during training using an adaptive LR (lr = 0.0001) - good for object detection
    optimizer = Adam(model.parameters(), lr=1e-4) # lr=1e-3, weight_decay=5e-4 or 0.0

    # Reduces the LR every 10 epochs, multiplies lr by 0.1 at each step (0.0001 -> 0.00001 -> etc.) - good for fine-tuning 
    # e.g. scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
    """
   
    # Adam is a type of gradient descent (like SGD) with adaptive learning rate (based on past squared gradients), helps faster convergence.
    # optimizer = Adam(model.parameters(), lr=1e-3)

    optimizer = Adam(model.parameters(), lr=3e-4)

    # Manual switch to turn OneCycle on / off
    use_scheduler = True
    
    """
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * num_epochs

    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,               
        total_steps=total_steps,   # Required
        pct_start=0.1,             # Warmup phase = 10% of training
        anneal_strategy='cos',     # Cosine decay after warmup
        div_factor=10.0,           # Initial LR = max_lr / div_factor
        final_div_factor=100.0     # Final LR = max_lr / final_div_factor
    )
    """

    # Makes computations faster and reduce GPU memory usage (mixed precision training), but fragile (not applied anymore)
    # scaler = torch.amp.GradScaler() 
    
    # Defining LR
    # END
    # ===========

    # Move model to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[I] Used device: ", device)
    model.to(device)

    # Dummy input test
    print("[I] Running dummy input test...")
    dummy_input = torch.randn(1, 3, 416, 416).to(device)  # Batch size 1 image, 3 channels, 416x416 resolution
    with torch.no_grad():  # No gradient calculation needed for this test
        outputs = model(dummy_input)
    print("[I] Dummy input test PASSED.")
    # print(f"[I] Model output shapes: {[output.shape for output in outputs]}")
    assert outputs[0].shape[2:] == (52, 52), f"[E] Small scale output mismatch: {outputs[0].shape}"
    assert outputs[1].shape[2:] == (26, 26), f"[E] Medium scale output mismatch: {outputs[1].shape}"
    assert outputs[2].shape[2:] == (13, 13), f"[E] Large scale output mismatch: {outputs[2].shape}"
    print("[I] All detection head shapes are correct: (52×52, 26×26, 13×13)")

    # =============================
    # Load checkpoint (prefer best)
    # =============================
    checkpoint_path = None
    start_epoch = 0
    best_mAP = 0.0

    additional_epochs = 0

    # Manual checkpoint control
    force_manual_resume = False # Set True only when want to resume from a manually fixed checkpoint
    manual_ckpt_path = os.path.join(os.getcwd(), "yolov3_checkpoint_last_epoch.pth")

    # Manual switch (Option 1 will switch it anyway)
    manual_loaded = False

    # Option 1: Force resume from manually fixed checkpoint (epoch counter restore - see the commands in the header) if "yolov3_checkpoint_last_epoch.pth" needs to be used due to training crash.
    if force_manual_resume and os.path.exists(manual_ckpt_path):
        print(f"[I] Forcing resume from manually fixed checkpoint: {manual_ckpt_path}")
        checkpoint = torch.load(manual_ckpt_path, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)

        start_epoch = checkpoint.get("epoch", 0)
        best_mAP = checkpoint.get("best_mAP", 0.0)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        manual_loaded = True # Prevent reloading below
        print(f"[I] Loaded from checkpoint: epoch {start_epoch}, best mAP (logged so far): {best_mAP:.4f}")
      
        if start_epoch >= num_epochs:
            num_epochs = start_epoch + additional_epochs
            print(f"[I] Resuming from epoch {start_epoch}. Updated num_epochs to {num_epochs} for continued training.")
    else:
        print("[I] Manual resume not set — continue with best/last checkpoint logic.")

    # Option 2: Standard checkpoint logic (best/last)
    if not manual_loaded: # Only True if manual_loaded is False
        if os.path.exists(checkpoint_best_path):
            checkpoint_path = checkpoint_best_path
            print("[I] MODEL STATE - Found best checkpoint. Resuming from best model...")
        elif os.path.exists(checkpoint_last_path):
            checkpoint_path = checkpoint_last_path
            print("[I] MODEL STATE - Best model not found. Resuming from last model...")
        else:
            print("[I] MODEL STATE - No checkpoint found. Starting training from scratch...")

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
              
            start_epoch = checkpoint.get('epoch', 0)
            best_mAP = checkpoint.get('best_mAP', 0.0)
            print(f"[I] Loaded from checkpoint: epoch {start_epoch}, best mAP (logged so far): {best_mAP:.4f}")
            
            # Extend training duration automatically
            if start_epoch >= num_epochs:
                num_epochs = start_epoch + additional_epochs
                print(f"[I] Resuming from epoch {start_epoch}. Updated num_epochs to {additional_epochs} for continued training.")

    # ===============================================================
    # CREATE OneCycleLR (always after optimizer and checkpoint load)
    # ===============================================================

    # steps_per_epoch = len(train_dataloader)
    steps_per_epoch = (len(train_dataloader) + accumulation_steps - 1) // accumulation_steps

    remaining_epochs = num_epochs - start_epoch
    total_steps = steps_per_epoch * remaining_epochs
    
    # The original one
    """
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0
    )
    """
    if use_scheduler:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=3e-4,
            total_steps=total_steps,
            pct_start=0.2,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1000.0
        )
    else:
        scheduler = None
    
    # The scheduler for OneCycle is planned for the predefined training run.
    # If manual resume is needed, scheduler is loaded from checkpoint.
    if force_manual_resume and os.path.exists(manual_ckpt_path) and use_scheduler:
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("[I] Scheduler state restored from checkpoint.")

    if use_scheduler:
        print(f"[I] OneCycleLR initialized with total_steps={total_steps} "f"(epochs={remaining_epochs}, steps/epoch={steps_per_epoch})") 
    else:
        print("[I] Scheduler disabled - constant LR")

    """
    Verify output shapes. This ensures the first output tensor (outputs[0]) has the shape (1, 255, 52, 52). Here:
    1 = batch size
    255 = number of output channels (255 = 3 anchors × (4 bbox coordinates + 1 confidence score + 80 classes))
    52x52 = spatial dimensions of the feature map
    """
    assert outputs[0].shape == (1, 255, 52, 52), f"[E] Small scale output mismatch: {outputs[0].shape}"
    assert outputs[1].shape == (1, 255, 26, 26), f"[E] Medium scale output mismatch: {outputs[1].shape}"
    assert outputs[2].shape == (1, 255, 13, 13), f"[E] Large scale output mismatch: {outputs[2].shape}"
    logger.info("[I] All output shapes - raw tensor from the model (B, 255, S, S) -  are correct!")

    # ===================================================================
    # Initial Debug Target Assignment - 3x3 Visualization Before Training
    # ===================================================================
    try:
        first_batch = next(iter(train_dataloader))  # Get the first batch explicitly
        images, boxes, labels, original_sizes, image_ids = first_batch  # Unpack batch data
    except StopIteration:
        images, boxes, labels, original_sizes, image_ids = None, None, None, None, None

    if images is not None:
        gt_boxes = boxes[0].cpu().numpy()  # Take all GT boxes

        # Convert tensor to NumPy image for visualization
        image_np = images[0].permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
        image_np = (image_np * 255).astype("uint8")
        
        # ========================================================================================================
        # 3x3 Grid Visualization Setup - visualize anchors for each GT box (drawn around the center of the GT box) 
        # ========================================================================================================
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))

        scales = [52, 26, 13]  # Small, Medium, Large grid sizes
        strides = [8, 16, 32]  # Stride values (8 for 52x52, 16 for 26x26, 32 for 13x13)
        
        absolute_anchors = []
        for stride, anchor_group in zip(strides, [scaled_anchors[:3], scaled_anchors[3:6], scaled_anchors[6:]]):
            for w, h in anchor_group:
                absolute_anchors.append((w * stride, h * stride)) 
        
        # Iterate over 3 scales (small, medium, large)
        for i, grid_size in enumerate(scales):  
            stride = strides[i]
            
            # Iterate over 3 anchor boxes per scale
            for j in range(3):  
                ax = axes[i, j]
                
                # Ensure every subplot gets a correctly sized 416x416 image
                ax.imshow(image_np, extent=[0, 416, 416, 0]) 
                ax.set_title(f"Scale {grid_size}, Anchor {j+1}")
                ax.axis("off")
        
                # Draw feature map grid
                cell_size = 416 / grid_size  # Compute the size of each grid cell
                for x in range(grid_size + 1):  # Vertical lines
                    ax.axvline(x * cell_size, color="black", linestyle="--", linewidth=0.5)
                for y in range(grid_size + 1):  # Horizontal lines
                    ax.axhline(y * cell_size, color="black", linestyle="--", linewidth=0.5)

                # GT + anchor overlay (inside scale + anchor loop)
                for gt_box, model_index in zip(gt_boxes, labels[0].cpu().numpy()):
                    if model_index == -1:
                        continue

                    x_min, y_min, x_max, y_max = gt_box
                    class_name = coco_classes.get(int(model_index), "Unknown")

                    # Draw GT box
                    ax.add_patch(patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                                linewidth=1, edgecolor="yellow", facecolor="none", linestyle="dashed"))

                    # Draw label
                    label_position_y = max(y_min - 5, 10)
                    ax.text(x_min, label_position_y, class_name, color="white", fontsize=8,
                            bbox=dict(facecolor="black", alpha=0.75, edgecolor="none", boxstyle="round,pad=0.2"))

                    # Anchor info
                    anchor_w, anchor_h = absolute_anchors[i * 3 + j]
                    gt_w = x_max - x_min
                    gt_h = y_max - y_min
                    iou = compute_iou(torch.tensor([0, 0, anchor_w, anchor_h]), torch.tensor([0, 0, gt_w, gt_h]))
                    logger.info(f"[I] Anchor {j+1} at Scale {grid_size}: (w={anchor_w:.2f}px, h={anchor_h:.2f}px) - IoU with GT: {iou:.4f}")

                    # Anchor position (aligned to GT center)
                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2
                    grid_x = int(center_x // stride)
                    grid_y = int(center_y // stride)
                    anchor_x = (grid_x + 0.5) * stride
                    anchor_y = (grid_y + 0.5) * stride

                    ax.add_patch(patches.Rectangle((anchor_x - anchor_w / 2, anchor_y - anchor_h / 2),
                                                anchor_w, anchor_h,
                                                linewidth=1, edgecolor="cyan", facecolor="none"))

                    # Center dot
                    ax.plot(center_x, center_y, marker="o", color="yellow", markersize=5)

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.1, hspace=0.1)
        plt.tight_layout()
        plt.show()

    print("[I] Check YOLOv3 num_classes =", model.num_classes)
    for head in [model.det_head_small, model.det_head_medium, model.det_head_large]:
        print("[I] Sanity check - detection head out_channels =", head.out_channels)

    # Benchmarking for "num_workers". Result so far:
    
    # num_workers = 4
    # [I] DataLoader warmup: 10 batches in 0.72 seconds
    
    # num_workers = 6
    # [I] DataLoader warmup: 10 batches in 0.67 seconds
     
    # num_workers = 8
    # [I] DataLoader warmup: 10 batches in 0.76 seconds

    start = time.time()
    for i, (images, _, _, _, _) in enumerate(train_dataloader):
        if i == 10:  # just load 10 batches
            break
    print(f"[I] DataLoader warmup for benchmarking num_workers: 10 batches in {time.time() - start:.2f} seconds")

    # Start training (epochs and accumulation steps are set manually at the moment)
    loss_history, memory_usage_history, cuda_memory_history, iou_history = train(model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,                              
        device=device,
        num_epochs=num_epochs,
        accumulation_steps=accumulation_steps,
        coco_gt_path=train_annotations_file,
        num_classes=num_classes,
        scaled_anchors=scaled_anchors,     
        coco_classes=coco_classes,        
        anchors=anchors,                 
        start_epoch=start_epoch,
        best_mAP=best_mAP
    )

    # Plot epoch-level loss
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, label="Epoch Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("YOLOv3 Training Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot memory usage history
    memory_usage_current = [x[0] for x in memory_usage_history]
    memory_usage_peak = [x[1] for x in memory_usage_history]

    plt.figure(figsize=(8, 4))
    plt.plot(memory_usage_current, label="Current CPU Memory Usage (MB)", color="blue")
    plt.plot(memory_usage_peak, label="Peak CPU Memory Usage (MB)", color="red")
    plt.xlabel("Training Step")
    plt.ylabel("Memory (MB)")
    plt.title("CPU Memory Usage During Training")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting CUDA memory usage
    cuda_allocated = [x[0] for x in cuda_memory_history]
    cuda_max_allocated = [x[1] for x in cuda_memory_history]

    plt.figure(figsize=(8, 4))
    plt.plot(cuda_allocated, label="CUDA Allocated Memory (MB)", color="green")
    plt.plot(cuda_max_allocated, label="CUDA Max Allocated Memory (MB)", color="orange")
    plt.xlabel("Training Step")
    plt.ylabel("Memory (MB)")
    plt.title("CUDA Memory Usage During Training")
    plt.legend()
    plt.grid(True)
    plt.show()

# Wrapped into def main()
if __name__ == "__main__":
    main()