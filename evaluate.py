import os
import gc
import copy
import torch
import json
from logger import logger
from torchvision.ops import box_iou
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def run_evaluation_after_training(model, val_loader, device, coco_gt_path, coco_classes=None):    
    logger.info("[I] Starting evaluation...")
    print("[I] Starting evaluation...")
    output_json_path = "predictions.json"
    mAP = evaluate_model(model, val_loader, device, coco_gt_path, output_json_path, coco_classes=coco_classes)

    if mAP is None:
        logger.warning("[W] Validation mAP could not be computed.")
        print("[W] Validation mAP could not be computed.")
    else:
        logger.info(f"[I] Validation mAP@[0.5:0.95]: {float(mAP):.4f}")
        print(f"[I] Validation mAP@[0.5:0.95]: {float(mAP):.4f}")
    return mAP

def evaluate_model(model, data_loader, device, coco_gt_path, output_json_path, coco_classes=None):
    coco_gt = COCO(coco_gt_path)
    dataset = data_loader.dataset
    
    logger.info(f"[I] coco_id_to_model_index: {dataset.coco_id_to_model_index}")
    logger.info(f"[I] coco_id_to_model_index: {dataset.coco_id_to_model_index}")
    logger.info(f"[I] Sorted keys: {sorted(dataset.coco_id_to_model_index.keys())}")

    coco_category_ids = sorted(coco_gt.getCatIds())
    model_to_coco_map = {
        dataset.coco_id_to_model_index[cid]: cid
        for cid in coco_category_ids
        if cid in dataset.coco_id_to_model_index
    }

    # Optional check: review round-trip consistency
    inverse_map_check_passed = True
    for model_index, coco_id in model_to_coco_map.items():
        mapped_index = dataset.coco_id_to_model_index.get(coco_id, -999)
        if mapped_index != model_index:
            logger.error(f"[E] Mapping mismatch: model index {model_index} -> COCO ID {coco_id} -> Back to model index {mapped_index}")
            inverse_map_check_passed = False
    if inverse_map_check_passed:
        logger.info("[I] Mapping check PASSED: model_to_coco_map is consistent with the training mapping")
    else:
        raise AssertionError("[E] Label mapping between training and evaluation is inconsistent")

    for name, param in model.named_parameters():
        if "det_head" in name and "weight" in name:
            logger.info(f"[I] Sanity check {name} → out_channels = {param.shape[0]} (should be {3 * (model.num_classes + 5)})")

    model.eval()

    # Store predictions for each image
    outputs = [] 

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # images, _, _, original_sizes, _ = batch
            images, _, _, original_sizes, image_ids_batch = batch
            images = images.to(device)

            batch_image_ids = image_ids_batch 

            raw_outputs = model(images)

            for scale_output in raw_outputs:
                logger.info(f"[I] Output shape: {scale_output.shape}")

            if isinstance(raw_outputs, (list, tuple)):
                assert len(raw_outputs) % 3 == 0, f"[W] Unexpected number of outputs: {len(raw_outputs)}"
                raw_outputs = [
                    torch.cat([raw_outputs[i] for i in range(j, len(raw_outputs), 3)], dim=0)
                    for j in range(3)
                ]
        
            # Safety check
            assert len(raw_outputs) == 3, f"[E] Model returned {len(raw_outputs)} scales. Expected 3."
            """
            batch_image_ids = dataset.image_ids[
                batch_idx * data_loader.batch_size:
                batch_idx * data_loader.batch_size + len(images)
            ]
            """
            batch_image_dims = [dataset.coco.imgs[image_id] for image_id in batch_image_ids]

            logger.info(f"[I] Eval batch {batch_idx}: image_ids = {batch_image_ids}")
            logger.info(f"[I] Number of images in batch: {len(images)}")

            for i in range(min(len(images), len(batch_image_ids))):
                scales = [raw_outputs[s][i].unsqueeze(0) for s in range(3)]
                img_info = batch_image_dims[i]
                image_w = img_info["width"]
                image_h = img_info["height"]

                # Load GT annotation for current image
                image_id = batch_image_ids[i]
                ann_ids = coco_gt.getAnnIds(imgIds=image_id)
                annotations = coco_gt.loadAnns(ann_ids)

                gt_tensor = None  # Default to "None" (in case no GT is found)

                strides = [8, 16, 32]
                scaled_anchors = [(aw / stride, ah / stride) 
                  for stride, (aw, ah) in zip(
                      [stride for stride in strides for _ in range(3)],
                      model.anchors
                  )]

                # Used for debugging single image IoU sanity
                # Safety check: review the format of loaded annotations
                if annotations:
                    x, y, w, h = annotations[0]['bbox']  # COCO format: [x_min, y_min, width, height]
                    x_min, y_min = x, y
                    x_max = x + w
                    y_max = y + h
                    gt_tensor = torch.tensor([[x_min, y_min, x_max, y_max]], device=device)
               
                decoded = model.decode_predictions(
                    scales,
                    anchors = scaled_anchors,
                    # anchors = model.anchors,
                    num_classes=model.num_classes,
                    # image_w=image_w,
                    # image_h=image_h,
                    image_w=416,
                    image_h=416,
                    nms_threshold=0.5,
                    conf_threshold=0.001, # Keep boxes and let NMS do the job in early training
                    debug_force_class=None,
                ) 
                """
                | Using "debug_force_class"     | Purpose                       | Safe?        |
                | ----------------------------- | ----------------------------- | ------------ |
                | train.py                      | Focus on the forced class     | Yes          |
                | evaluate.py (mAP eval)        | Forces class filter = invalid | No           |
                """

                decoded = decoded[0]

                # Sanity check:
                logger.info(f"[I] Raw decoded box sample: {decoded['boxes'][:3]}")
                # print(f"[I] Raw decoded box sample: {decoded['boxes'][:3]}")

                orig_h, orig_w = original_sizes[i]

                if len(decoded["boxes"]) > 0:
                    boxes = decoded["boxes"]

                    sx = orig_w / 416.0
                    sy = orig_h / 416.0

                    boxes[:, [0, 2]] *= sx
                    boxes[:, [1, 3]] *= sy

                    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, orig_w)
                    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, orig_h)

                    decoded["boxes"] = boxes

                if gt_tensor is not None and len(decoded["boxes"]) > 0:
                    ious = box_iou(decoded["boxes"], gt_tensor)
                    max_iou = ious.max().item()
                    logger.info(f"[I] MAX IoU in image: {max_iou:.4f}")
                    # print(f"[I] MAX IoU in image: {max_iou:.4f}")

                pred_boxes = decoded['boxes']
                pred_labels = decoded['labels']

                # Filter for high confidence negatives / garbage
                scores = decoded["scores"]
                ious = box_iou(decoded["boxes"], gt_tensor).max(dim=1).values

                high_conf = scores > 0.8
                high_conf_ious = ious[high_conf]

                logger.info(f"[EVAL] High-conf boxes: {high_conf.sum().item()}, "
                            f"mean IoU={high_conf_ious.mean().item() if high_conf_ious.numel()>0 else 0:.3f}")
                """
                print(f"[EVAL] High-conf boxes: {high_conf.sum().item()}, "
                            f"mean IoU={high_conf_ious.mean().item() if high_conf_ious.numel()>0 else 0:.3f}")
                """


                if gt_tensor is not None:
                    if pred_boxes.numel() > 0:
                        ious = box_iou(pred_boxes, gt_tensor)
                        for j in range(min(3, len(pred_boxes))):
                            max_iou, gt_idx = ious[j].max(0)
                            logger.info(f"[I] Eval diag pred #{j}: IoU = {max_iou:.4f}, GT idx = {gt_idx.item()}, Label = {pred_labels[j].item()}")
                            logger.info(f"           -> Pred box: {pred_boxes[j].tolist()}")
                            logger.info(f"           -> GT box  : {gt_tensor[gt_idx].tolist()}")
                            gt_class_id = dataset.coco_id_to_model_index[annotations[gt_idx.item()]['category_id']]
                            logger.info(f"           -> GT class: {gt_class_id}, Pred class: {pred_labels[j].item()}, Match={'YES' if gt_class_id==pred_labels[j].item() else 'NO'}")
                
                """
                if not isinstance(decoded, list) or not all(isinstance(d, dict) for d in decoded):
                    logger.error(f"[E] decode_predictions returned an unexpected type: {type(decoded)}")
                    raise TypeError("[E] decode_predictions must return a list of dictionaries.")
                """

                if not isinstance(decoded, dict):
                    logger.error(f"[E] decode_predictions returned unexpected type: {type(decoded)}")
                    raise TypeError("[E] decode_predictions must return a dict after indexing [0].")

                if len(decoded) == 0:
                    logger.warning(f"[W] decode_predictions returned an empty list for image {batch_image_ids[i]}")
                    decoded = {"boxes": [], "scores": [], "labels": []}
                else:
                    # Safe extraction from list or direct dict
                    if isinstance(decoded, list) and len(decoded) > 0:
                        decoded = decoded[0]
                    elif isinstance(decoded, dict):
                        pass
                    else:
                        logger.warning("[W] Invalid decoded format")
                        decoded = {"boxes": [], "scores": [], "labels": []}

                    # Compare first predicted box vs. GT box (if available)
                    if len(decoded["boxes"]) > 0:
                        logger.info(f"[I] Check decoded box (first): {decoded['boxes'][0]}")
                    else:
                        logger.warning("[I] Check decoded box (first): No boxes")

                    # Attempt to get GT box for this image
                    image_id = batch_image_ids[i]
                    ann_ids = coco_gt.getAnnIds(imgIds=image_id)
                    annotations = coco_gt.loadAnns(ann_ids)

                    # Already defined above, but keeping for clarity (kept from legacy code)
                    if annotations:
                        x, y, w, h = annotations[0]['bbox']  # COCO format: [x_min, y_min, width, height]
                        x_min, y_min = x, y
                        x_max = x + w
                        y_max = y + h
                        gt_box = [x_min, y_min, x_max, y_max]
                        logger.info(f"[I] Loaded GT box: {gt_box}")

                        """
                        | Purpose                            | Direction           | Operation      | Example                      |
                        | ---------------------------------- | ------------------  | -------------- | ---------------------------  |
                        | Convert GT boxes -> model input    | Original -> 416×416 | box /= scale   | GT label processing          |
                        | Convert predictions -> original    | 416×416 -> Original | box *= scale   | For COCO .json output or NMS |

                        Example:
                        scale_x = orig_w / 416.0
                        scale_y = orig_h / 416.0
                        x_min /= scale_x
                        y_min /= scale_y
                        x_max /= scale_x
                        y_max /= scale_y
                        """
                        # orig_h, orig_w = original_sizes[i]
                        
                        # gt_tensor = torch.tensor([[x_min, y_min, x_max, y_max]], device=device)

                        pred_boxes = decoded["boxes"]
                        if len(pred_boxes) > 0:
                            ious = box_iou(pred_boxes, gt_tensor).squeeze(1)  # Shape: [num_preds]
                            MAX_LOGGED_PREDS = 5  # only show top 5 predictions per image
                            found_iou = False
                            for idx, (box, score, label, iou) in enumerate(zip(pred_boxes, decoded["scores"], decoded["labels"], ious)):
                                if idx >= MAX_LOGGED_PREDS:
                                    break
                                logger.info(f"[I] Show top 5 predictions per image\n")
                                logger.info(f"[I] Pred (top 5) {idx:02d} -> IoU: {iou:.4f}, Score: {score:.2f}, Label: {label}")
                                logger.info(f"[I] Pred box: {box.tolist()}")
                                logger.info(f"[I] GT box: {gt_tensor[0].tolist()}")
                                logger.info(f"[I] IoU: {iou.item()}")
                                if iou.item() > 0.1:
                                    logger.info(f"[I] GT-aligned box found! IoU={iou.item():.4f}, Score={score:.2f}")
                                    found_iou = True
                            
                            if not found_iou:
                                logger.warning("[W] No predictions matched GT box with IoU > 0.1.")
                        else:
                            logger.warning("[W] No predicted boxes to compute IoU.")

                    #  Final validation of label indices
                    if (decoded["labels"] < 0).any() or (decoded["labels"] >= model.num_classes).any():
                        logger.error(f"[E] Invalid label(s) in evaluate_model BEFORE mapping to COCO IDs.")
                        logger.info(f"[I] Offending labels: {decoded['labels'][decoded['labels'] >= model.num_classes].tolist()}")
                        logger.info(f"[I] Label range: min={decoded['labels'].min().item()}, max={decoded['labels'].max().item()}, allowed max={model.num_classes - 1}")
                        raise ValueError("[E] Detected out-of-bounds label indices in evaluate_model.")

                outputs.append((copy.deepcopy(decoded), batch_image_ids[i], batch_image_dims[i], original_sizes[i]))

            torch.cuda.empty_cache()
            gc.collect()
           
    # ===========================================
    # Convert predictions to COCO-format JSON
    # ===========================================
    results = []

    for decoded, image_id, image_info, original_size in outputs:
        orig_h, orig_w = original_size  # From "collate_fn" (e.g. (480, 640))

        # Top-K per image. Takes the 100 highest confidence (objectness score x class score) predictions (COCOEval uses up to 100 by default anyway). K-top counts detections (not TPs).
        # Limits detections per image to keep the "predictions.jons" file light. 
        K = 100  
        if len(decoded["scores"]) > K:
            topk_idx = torch.topk(decoded["scores"], K).indices
            decoded["boxes"] = decoded["boxes"][topk_idx]
            decoded["scores"] = decoded["scores"][topk_idx]
            decoded["labels"] = decoded["labels"][topk_idx]

        for box, score, label in zip(decoded["boxes"], decoded["scores"], decoded["labels"]):

            # Now safe to convert to Python floats
            x_min, y_min, x_max, y_max = box.tolist()

            # Converts the PyTorch tensor to a standard Python list: box = tensor([x1, y1, x2, y2]) -> [x1, y1, x2, y2] (COCO JSON expects Python floats, not PyTorch tensors) - no stale tensor
            x_min, y_min, x_max, y_max = box.tolist() 

            # Safety check: clamp only during JSON export to avoid negative boxes in COCO Eval
            x_min = max(0.0, x_min)
            y_min = max(0.0, y_min)
            x_max = min(orig_w, x_max)
            y_max = min(orig_h, y_max)

            width = x_max - x_min
            height = y_max - y_min

            # Wrap clamped box in a tensor (used for debugging, optional)
            box = torch.tensor([[x_min, y_min, x_max, y_max]], device=device)

            coco_label = model_to_coco_map.get(
                int(label.item()) if isinstance(label, torch.Tensor) else int(label),
                None
            )

            # Safety check: ensure coco_label is valid                
            if coco_label is None:
                logger.error(f"[E] Invalid label: {label} not in mapping — skipping box")
                continue

            results.append({
                "image_id": image_id,
                "category_id": coco_label,
                "bbox": [x_min, y_min, width, height],
                "score": float(score)
            })

    # Prevent COCOEval crash when no predictions exist
    if not results:
        logger.warning("[W] No predictions > conf_threshold — adding dummy prediction to bypass COCOEval crash.")
        results.append({
            "image_id": coco_gt.getImgIds()[0],
            "category_id": 1,  # any valid class (e.g., person)
            "bbox": [0, 0, 1, 1],
            "score": 0.001  # low confidence
        })

    print(f"[I] Eval debug: first pred box={x_min:.1f},{y_min:.1f},{x_max:.1f},{y_max:.1f} "f"in image size {orig_w}x{orig_h}")

    # Save to file
    with open(output_json_path, "w") as f:
        f.write("[\n")
        for i, r in enumerate(results):
            json.dump(r, f)
            if i != len(results) - 1:
                f.write(",\n")
        f.write("\n]")

    print(f"[I] Saved detection results to {output_json_path}")        
    logger.info(f"[I] Saved detection results to {output_json_path}")

    # ===========================================
    # Use COCOEval for mAP computation
    # ===========================================

    coco_dt = coco_gt.loadRes(output_json_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # mAP@[0.5:0.95]
    return float(coco_eval.stats[0])