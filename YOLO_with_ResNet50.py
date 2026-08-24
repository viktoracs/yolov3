import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.ops import batched_nms
from logger import logger

"""
The model assumes 3x416x416 input. Preprocessing happens in train.py / data_loader.py.

Backbone:
- ResNet-50 pretrained on ImageNet (torchvision)
- Provides feature maps at strides 8, 16, 32
- Residual connections + BN + ReLU

Initialization:
- Backbone: ResNet-50
- YOLO heads:
    - Xavier weight init
    - Neutral bias init:
        tx, ty, tw, th = 0
        objectness = 0 (sigmoid -> 0.5)
        class logits = -4.5 (suppresses all classes initially) -> This avoids early collapse into everything = "person".

Differences from canonical YOLOv3:
- Original YOLOv3 uses Darknet-53 + SGD
- This model uses ResNet-50 + Adam
- Anchor scaling follows YOLOv3 conventions

Why this design:
- The assignment recommends pretrained backbones (ResNet-50 is available out of the box in torchvision)
- Custom fusion needed to produce YOLO-compatible 52/26/13 maps

Notes:
Interpolation acts as a smooth zoom-in or zoom-out mechanism, rather than pooling's "pick the strongest" approach (allows concaten two maps cleanly without losing info):
- "nearest"= just duplicates the nearby pixel values (no smoothing or averaging).
- "bilinear"= weighted average of 4 nearest pixels (smoother, but more compute).
- "bicubic"= weighted average of 16 nearest pixels (even smoother, but even more compute).
"""


class ConvBNLeaky(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        padding = (kernel_size - 1) // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class YOLOv3(nn.Module):
    def __init__(self, num_classes, anchors):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = len(anchors) // 3

        # Backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.stem   = nn.Sequential(*list(resnet.children())[:4])
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2  # C3 (52x52)
        self.layer3 = resnet.layer3  # C4 (26x26)
        self.layer4 = resnet.layer4  # C5 (13x13)

        # ============================================================
        # YOLOv3-style feature fusion neck
        # ============================================================

        # ------------------------------------------------------------
        # LARGE SCALE: 13×13
        # ResNet C5: 2048 channels
        # ------------------------------------------------------------
        self.large_block = nn.Sequential(
            ConvBNLeaky(2048, 512, 1),
            ConvBNLeaky(512, 1024, 3),
            ConvBNLeaky(1024, 512, 1),
        )

        self.large_pred = ConvBNLeaky(
            512,
            1024,
            3
        )

        self.det_head_large = nn.Conv2d(
            1024,
            self.num_anchors * (5 + num_classes),
            kernel_size=1
        )

        # Reduce P5 before upsampling toward the 26×26 scale
        self.large_route = ConvBNLeaky(
            512,
            256,
            1
        )


        # ------------------------------------------------------------
        # MEDIUM SCALE: 26×26
        # ResNet C4: 1024 channels
        #
        # lateral C4 = 512
        # upsampled P5 = 256
        # concat = 768
        # ------------------------------------------------------------
        self.medium_lateral = ConvBNLeaky(
            1024,
            512,
            1
        )

        self.medium_block = nn.Sequential(
            ConvBNLeaky(768, 256, 1),
            ConvBNLeaky(256, 512, 3),
            ConvBNLeaky(512, 256, 1),
        )

        self.medium_pred = ConvBNLeaky(
            256,
            512,
            3
        )

        self.det_head_medium = nn.Conv2d(
            512,
            self.num_anchors * (5 + num_classes),
            kernel_size=1
        )

        # Reduce P4 before upsampling toward the 52×52 scale
        self.medium_route = ConvBNLeaky(
            256,
            128,
            1
        )


        # ------------------------------------------------------------
        # SMALL SCALE: 52×52
        # ResNet C3: 512 channels
        #
        # lateral C3 = 256
        # upsampled P4 = 128
        # concat = 384
        # ------------------------------------------------------------
        self.small_lateral = ConvBNLeaky(
            512,
            256,
            1
        )

        self.small_block = nn.Sequential(
            ConvBNLeaky(384, 128, 1),
            ConvBNLeaky(128, 256, 3),
            ConvBNLeaky(256, 128, 1),
        )

        self.small_pred = ConvBNLeaky(
            128,
            256,
            3
        )

        self.det_head_small = nn.Conv2d(
            256,
            self.num_anchors * (5 + num_classes),
            kernel_size=1
        )

        # YOLO detection head initialization patch
        heads = [self.det_head_small, self.det_head_medium, self.det_head_large]

        for head in heads:
            # Xavier initialization (keeps early gradients stable)
            nn.init.xavier_uniform_(head.weight, gain=0.01)

            # YOLO-style bias initialization
            with torch.no_grad():
                bias = head.bias.view(self.num_anchors, 5 + num_classes)

                # Bounding box center (tx, ty) start neutral
                bias[:, 0] = 0.0    # tx
                bias[:, 1] = 0.0    # ty

                # tw, th start at 0 -> exp(0) = 1, so width/height = anchor size
                bias[:, 2] = 0.0    # tw
                bias[:, 3] = 0.0    # th

                # Objectness prior:
                # Set to -4.5 -> sigmoid(-4.5) ≈ 0.01 initial confidence.
                bias[:, 4] = -4.5   # objectness

                # Class scores suppressed initially:
                # 4.5 -> sigmoid ≈ 0.01
                bias[:, 5:] = -4.5  

                head.bias.copy_(bias.view(-1))

            head.bias.requires_grad = True

    def forward(self, x):

        # ============================================================
        # ResNet50 backbone
        # ============================================================

        x = self.stem(x)

        C2 = self.layer1(x)      # 256 channels, 104×104
        C3 = self.layer2(C2)     # 512 channels, 52×52
        C4 = self.layer3(C3)     # 1024 channels, 26×26
        C5 = self.layer4(C4)     # 2048 channels, 13×13


        # ============================================================
        # LARGE SCALE — 13×13
        # ============================================================

        P5 = self.large_block(C5)
        # [B, 512, 13, 13]

        large_features = self.large_pred(P5)
        # [B, 1024, 13, 13]

        out_large = self.det_head_large(large_features)
        # [B, 255, 13, 13]


        # Route large-scale features upward
        P5_route = self.large_route(P5)
        # [B, 256, 13, 13]

        P5_up = F.interpolate(
            P5_route,
            scale_factor=2,
            mode="nearest"
        )
        # [B, 256, 26, 26]


        # ============================================================
        # MEDIUM SCALE — 26×26
        # ============================================================

        C4_lateral = self.medium_lateral(C4)
        # [B, 512, 26, 26]

        P4 = torch.cat(
            [C4_lateral, P5_up],
            dim=1
        )
        # 512 + 256 = 768 channels

        P4 = self.medium_block(P4)
        # [B, 256, 26, 26]

        medium_features = self.medium_pred(P4)
        # [B, 512, 26, 26]

        out_medium = self.det_head_medium(medium_features)
        # [B, 255, 26, 26]


        # Route medium-scale features upward
        P4_route = self.medium_route(P4)
        # [B, 128, 26, 26]

        P4_up = F.interpolate(
            P4_route,
            scale_factor=2,
            mode="nearest"
        )
        # [B, 128, 52, 52]


        # ============================================================
        # SMALL SCALE — 52×52
        # ============================================================

        C3_lateral = self.small_lateral(C3)
        # [B, 256, 52, 52]

        P3 = torch.cat(
            [C3_lateral, P4_up],
            dim=1
        )
        # 256 + 128 = 384 channels

        P3 = self.small_block(P3)
        # [B, 128, 52, 52]

        small_features = self.small_pred(P3)
        # [B, 256, 52, 52]

        out_small = self.det_head_small(small_features)
        # [B, 255, 52, 52]


        # IMPORTANT:
        # Keep the same order expected everywhere else.
        return [
            out_small,
            out_medium,
            out_large
        ]


    """
    This function (decode_predictions) processes the model's outputs. Why necessary?
    
    The raw outputs of the YOLO model contain grid-level predictions that need to be transformed into interpretable bboxes, conf.scores and class probs.
    
    This includes:
        -> Applying sigmoid to normalize the offsets and probabilities.
        -> Decoding grid-cell-relative coordinates into absolute image-relative coordinates.
        -> Adjusting bbox sizes based on anchor boxes.
        -> Applies NMS: removes redundant predictions with high overlap (IoU) while retaining the most confident ones.
        -> Returns human-readable predictions: Produces a structured output with decoded bboxes, confidence scores and class labels.
    """
    
    # Decodes YOLOv3 model outputs into bboxes, conf.scores and class labels.
    # Stateless function (does not depend on model instance variables) 
    @staticmethod
    def decode_predictions(outputs, anchors, image_w, image_h, num_classes=80, conf_threshold=0.001, nms_threshold=0.5, debug_force_class=None):
        """
        outputs: list of tensors from YOLOv3 model (small, medium, large scales)
        anchors: list of anchor boxes for each scale
        num_classes: mumber of object classes
        image_w: original image width
        image_h: original image height
        conf_threshold: conf.threshold for filtering detections before NMS (objectness × class probability)
        nms_threshold: IoU threshold for NMS
        "debug_force_class" is an optional debug parameter
            Two possible parameters:
            1. int (the model_index, e.g. 22="zebra") - forces the model to only get predictions of the provided class, no matter what the model predicted for other classes.
            2. "None" - It will show all predictions for all classes (default setting).

        Returns:
            List of dictionaries with 'boxes', 'scores', 'labels'

        YOLO decoding formula from the paper:
        bx = sigmoid(tx) + cx
        by = sigmoid(ty) + cy
        bw = pw * exp(tw)
        bh = ph * exp(th)
  
        Symbol	    Meaning	Explanation:            Explanation:
        tx, ty	    Model outputs	                Predicted offsets for box center within a grid cell (predicted offsets from the top-left corner of the grid cell)
        tw, th	    Model outputs	                Predicted log-space width and height relative to the anchor box
        cx, cy	    Grid cell coordinates	        The top-left cell offset in the grid, e.g., (7, 14)
        pw, ph	    Anchor box width & height	    Predefined anchor dimensions (e.g. [1.25, 1.625] in grid units)
        bx, by	    Final predicted center	        Offset into the grid cell, shifted by sigmoid(tx), sigmoid(ty)
        bw, bh	    Final predicted width/height    Anchor size scaled by exp(tw/th)
        """

        device = outputs[0].device
        batch_predictions = []
        anchor_groups = [anchors[:3], anchors[3:6], anchors[6:]]  # Group anchors by scale
        strides = [8, 16, 32]  # YOLO strides (small, medium, large)
        
        # Safety check:
        assert len(outputs) == 3, f"[E] Expected 3 scales, but got {len(outputs)}."

        all_boxes = []
        all_scores = []
        all_labels = []

        for i, out in enumerate(outputs): 
            if out.shape[1] != 3 * (num_classes + 5):
                raise ValueError(f"[E] output[{i}] shape invalid: {out.shape}")

        # Safety check: grid sizes must match expected YOLO order (small, medium, large)
        expected_shapes = [52, 26, 13]
        for i, out in enumerate(outputs):
            h, w = out.shape[2:]
            assert h == expected_shapes[i], (
                f"[E] Anchor scale bug - Output[{i}] grid = {h}×{w}, expected = {expected_shapes[i]}×{expected_shapes[i]} "
                f"(scale mismatch with anchors/strides)"
            )

        # Each output tensor corresponds to one scale: 13×13, 26×26, 52×52
        for scale_idx, output in enumerate(outputs): 
            grid_size = output.shape[-1]
            stride = strides[scale_idx]

            batch_size, _, grid_h, grid_w = output.shape
            num_anchors = len(anchor_groups[scale_idx])

            expected_channels = num_anchors * (num_classes + 5)

            # Safety check: output channels must match expected
            assert output.shape[1] == expected_channels, (
                f"[E] Output channels = {output.shape[1]}, expected = {expected_channels}. "
                f"Expected invalid class predictions."
            )

            # Reshapes from (B, C, H, W) to (B, H, W, A, 85), where 85 = 4 bbox + 1 obj + 80 class scores
            output = output.permute(0, 2, 3, 1).contiguous() 

            actual_channels = output.shape[-1]
            expected_channels = num_anchors * (num_classes + 5)
            assert actual_channels == expected_channels, (
                f"[E] Output has {actual_channels} channels, expected {expected_channels} "
                f"(num_anchors={num_anchors}, num_classes={num_classes})"
            )
            
            output = output.view(batch_size, grid_h, grid_w, num_anchors, num_classes + 5) 
            
            # FIX THIS: absolute_anchors is a wrong name. decode_predictions() is only called with scaled_anchors (both from train.py and evaluate.py)
            absolute_anchors = anchor_groups[scale_idx] 
            anchor_w = torch.tensor([a[0] for a in absolute_anchors], device=output.device).view(1, 1, 1, num_anchors)
            anchor_h = torch.tensor([a[1] for a in absolute_anchors], device=output.device).view(1, 1, 1, num_anchors)

            anchor_w = anchor_w.expand(1, grid_h, grid_w, num_anchors)
            anchor_h = anchor_h.expand(1, grid_h, grid_w, num_anchors)

            # Generate grid coordinates for bbox decoding
            grid_y, grid_x = torch.meshgrid(
                torch.arange(grid_h, device=output.device).float(),
                torch.arange(grid_w, device=output.device).float(),
                indexing="ij"
            )
            # Broadcast grid coordinates to match output shape
            grid_x = grid_x.unsqueeze(0).unsqueeze(-1).expand(batch_size, grid_h, grid_w, num_anchors)
            grid_y = grid_y.unsqueeze(0).unsqueeze(-1).expand(batch_size, grid_h, grid_w, num_anchors)

            box_x = (torch.sigmoid(output[..., 0]) + grid_x) * stride 
            box_y = (torch.sigmoid(output[..., 1]) + grid_y) * stride

            # Even with a trained model, torch.exp(tw) and torch.exp(th) can explode (clamping helpes convergence, but makes it slower).
            # exp(4.0) ≈ 54.6, meaning the predicted box can be up to ~54.6× the anchor size and exp(-4.0) ≈ 0.018, meaning the predicted box can be as small as ~1.8% of the anchor size (plus, sigmoid funcion is basically flat outside ±4.0).
            # MUST BE aligned with YOLO_loss.py clamping for consistency (6.0) -> LOG_WH_CLAMP = 6.0
            tw = output[..., 2].clamp(min=-6.0, max=6.0)
            th = output[..., 3].clamp(min=-6.0, max=6.0)

            logger.info(f"[I] tw/th debug - tw min={tw.min():.2f}, max={tw.max():.2f}")
            logger.info(f"[I] tw/th debug - th min={th.min():.2f}, max={th.max():.2f}")

            # Decode width and height using anchors (pixel-space decoding 416x416)
            box_w = torch.exp(tw) * anchor_w * stride
            box_h = torch.exp(th) * anchor_h * stride

            # Optional clamping to ensure box sizes are within image dimensions
            box_w = box_w.clamp(0, image_w)
            box_h = box_h.clamp(0, image_h)

            logger.info(f"[I] Size debug decoded widths: min={box_w.min().item():.2f}, max={box_w.max().item():.2f}")
            logger.info(f"[I] Size debug decoded heights: min={box_h.min().item():.2f}, max={box_h.max().item():.2f}")

            x_min = box_x - box_w / 2
            y_min = box_y - box_h / 2
            x_max = box_x + box_w / 2
            y_max = box_y + box_h / 2

            # Safety check: clamping to ensure box sizes are within image dimensions
            x_min = x_min.clamp(0, image_w)
            y_min = y_min.clamp(0, image_h)
            x_max = x_max.clamp(0, image_w)
            y_max = y_max.clamp(0, image_h)

            # Always rescale predicted boxes from 416×416 model space back to the original image size. 
            # FIX THIS: This is fully handled in evaluate.py now, so that this part is never called. Can be removed.
            if image_w != 416 or image_h != 416:               
                scale_x = image_w / 416
                scale_y = image_h / 416
                x_min = x_min * scale_x
                x_max = x_max * scale_x
                y_min = y_min * scale_y
                y_max = y_max * scale_y

            debug_xmin = x_min.view(-1)[0].item()
            debug_ymin = y_min.view(-1)[0].item()
            debug_xmax = x_max.view(-1)[0].item()
            debug_ymax = y_max.view(-1)[0].item()
            logger.info(f"[I] Center offset debug - decoded prediction box: [{debug_xmin:.2f}, {debug_ymin:.2f}, {debug_xmax:.2f}, {debug_ymax:.2f}]")

            logger.info(f"[I] Box debug - decoded: x_min={x_min.min():.1f}-{x_min.max():.1f}, x_max={x_max.min():.1f}-{x_max.max():.1f}")
            logger.info(f"[I] Box debug - heights: {(y_max - y_min).min():.1f} to {(y_max - y_min).max():.1f}")   
            logger.info(f"[I] Box debug - widths: {(x_max - x_min).min():.1f} to {(x_max - x_min).max():.1f}")
            logger.info(f"[I] grid_x shape: {grid_x.shape}, grid_y shape: {grid_y.shape}")
            logger.info(f"[I] box_x range: {box_x.min().item():.2f} – {box_x.max().item():.2f}")
            logger.info(f"[I] image_w: {image_w}, image_h: {image_h}")
            logger.info(f"[I] Pred center: ({box_x[0, 0, 0, 0].item():.1f}, {box_y[0, 0, 0, 0].item():.1f})")                

            conf = torch.sigmoid(output[..., 4])
            
            # Safety check: review output shape before slicing class logits
            if output.shape[-1] != (num_classes + 5):
                raise ValueError(
                    f"[E] Output last dimension mismatch: expected {num_classes + 5}, got {output.shape[-1]}"
            )

            # Clamping prevents extreme logits (± inf from exploding weights) from destabilizing sigmoid or BCE loss.
            class_logits = output[..., 5:].clamp(min=-10, max=10)

            # Study: "We use logistic classifiers instead of softmax for class predictions. This lets us do multi-label classification" -> BCE with sigmoid -> each class prediction is independent, allowing multi-label detection in a single detection area (grid cell).
            class_probs = torch.sigmoid(class_logits) 

            # Active if debug_force_class is enabled
            if debug_force_class is not None:
                
                # Force only the specified class
                logger.info(f"[I] debug_force_class overriding class_probs to only class {debug_force_class}")
                class_probs = torch.zeros_like(class_probs)
                class_probs[..., debug_force_class] = 1.0
                
                # Forces the model to act as if it only sees one class
                logger.info("[I] debug_force_class overriding max_labels with debug_force_class")
                max_scores = class_probs[..., debug_force_class]
                max_labels = torch.full_like(max_scores, debug_force_class, dtype=torch.long)

            # Check that class_probs is valid
            if class_probs.shape[-1] != num_classes:
                logger.error(f"[E] Class probs shape = {class_probs.shape[-1]}, expected = {num_classes}")  

            # Safety check
            assert class_probs.shape[-1] == num_classes, f"[E] class_probs.shape[-1] = {class_probs.shape[-1]} but num_classes = {num_classes}"

            # Compute final confidence score
            max_scores, max_labels = torch.max(class_probs, dim=-1)
            final_score = conf * max_scores

            # Flatten for top-k
            flat_scores = final_score.view(-1)

            # Top-k debug block (for 3 top predictions)
            if flat_scores.numel() >= 3:
                topk_scores, topk_indices = torch.topk(flat_scores, k=3)

                for i, idx in enumerate(topk_indices):
                    pred_x = box_x.view(-1)[idx]
                    pred_y = box_y.view(-1)[idx]

                    pred_grid_x = (pred_x / stride).floor().item()
                    pred_grid_y = (pred_y / stride).floor().item()

                    logger.info(f"[I] Pred #{i}] Score = {topk_scores[i].item():.2f} | Predicted grid cell = ({pred_grid_x}, {pred_grid_y}) | Scale: {grid_size}")
            else:
                logger.info(f"[W] Skipping top-k debug: only {flat_scores.numel()} predictions available")

            # Class probability (for 5 top predictions) debug block
            logger.info("[I] Debug class probabilities - sample of per-prediction class probabilities:")
            flat_class_probs = class_probs.reshape(-1, num_classes)
            flat_labels = max_labels.reshape(-1)

            if flat_scores.numel() >= 5:
                top_scores, top_indices = torch.topk(flat_scores, k=5)
            else:
                top_scores, top_indices = flat_scores, torch.arange(flat_scores.numel())

            for i, idx in enumerate(top_indices):
                class_dist = flat_class_probs[idx]
                pred_label = flat_labels[idx].item()
                top5_classes = torch.topk(class_dist, k=5)
                logger.info(f"[I] Pred {i}: score={flat_scores[idx]:.4f}, label={pred_label}")
                logger.info(f"    → top 5 class probs: {[(j.item(), f'{s:.3f}') for j, s in zip(top5_classes.indices, top5_classes.values)]}")

            # Safety check: hard guard before using max_labels in any GPU operation
            if max_labels.max().item() >= num_classes or max_labels.min().item() < 0:
                logger.error(f"[E] max_labels out of range: max = {max_labels.max().item()}, min = {max_labels.min().item()}, num_classes = {num_classes}")
                
                # Safety clamp
                max_labels = max_labels.clamp(0, num_classes - 1)  

            assert class_probs.shape[-1] == num_classes, (
                f"[I] Mismatch: class_probs.shape[-1]={class_probs.shape[-1]} vs num_classes={num_classes}"
            )

            # Safety check: catch invalid class index predictions
            if max_labels.max().item() >= num_classes:
                logger.error(f"[E] Invalid class index detected! Max index: {max_labels.max().item()} (num_classes={num_classes})")
                logger.info(f"[I] Unique predicted labels: {max_labels.unique()}")
            
            logger.info(f"[I] Decode - Max obj_conf: {conf.max().item():.4f}, Mean: {conf.mean().item():.4f}")
            logger.info(f"[I] Decode - Max cls_conf: {class_probs.max().item():.4f}, Mean: {class_probs.mean().item():.4f}")
            logger.info(f"[I] Decode - Max total conf: {final_score.max().item():.4f}")
            
            if debug_force_class is not None:
                debug_class_scores = class_probs[..., debug_force_class]
                final_score = conf * debug_class_scores
                max_labels = torch.full_like(debug_class_scores, debug_force_class, dtype=torch.long)
                mask = final_score > conf_threshold
                logger.info(f"[I] debug_force_class = {debug_force_class}, masked {mask.sum().item()} grid cells with score > {conf_threshold}")
            else:
                mask = final_score > conf_threshold
                logger.info(f"[I] Predictions above threshold: {mask.sum().item()}")

            if final_score[mask].numel() > 0:
                logger.info(f"[I] After masking — min score: {final_score[mask].min().item():.4f}")
            else:
                logger.info(f"[I] After masking — no predictions passed the mask (class {debug_force_class}, conf > {conf_threshold})")

            # Gathering predictions
            score = final_score[mask].view(-1).float()
            label = max_labels[mask].view(-1)
            
            # Safety clamp
            label = label.clamp(0, num_classes - 1) 
            
            box = torch.stack([
                x_min[mask].view(-1),
                y_min[mask].view(-1),
                x_max[mask].view(-1),
                y_max[mask].view(-1)
            ], dim=-1).float()

            all_scores.append(score)
            all_labels.append(label)
            all_boxes.append(box)

            logger.info(f"[I] Scale {scale_idx + 1} decoding complete.")

        # Combine and apply NMS across all scales
        all_scores = torch.cat(all_scores, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_labels = all_labels.clamp(0, num_classes - 1).to(dtype=torch.int64)
        assert all_labels.dtype == torch.int64, "[W] Labels must be int64 before NMS!"
        all_boxes = torch.cat(all_boxes, dim=0)

        if all_labels.numel() == 0:
            logger.warning("[W] No labels returned (empty batch)")
            all_labels_cpu = torch.empty(0, dtype=torch.int64)
        else:
            all_labels_cpu = all_labels.detach().cpu()
            if all_labels_cpu.max().item() >= num_classes:
                logger.warning(f"[W] decode_predictions() returned invalid label ≥ {num_classes}")
            if all_labels_cpu.min().item() < 0:
                logger.warning(f"[W] decode_predictions() returned negative label!")
            logger.info(f"[I] Unique predicted labels: {all_labels_cpu.unique().tolist()}")

        # Early exit when there are no valid detections (prevents empty tensor errors in NMS)
        if all_scores.numel() == 0:
            batch_predictions.append({
                "boxes": torch.empty((0, 4), device=all_boxes.device),
                "scores": torch.empty(0, device=all_scores.device),
                "labels": torch.empty(0, dtype=torch.int64, device=all_labels.device),
            })
            return batch_predictions

        # Before NMS
        logger.info(f"[I] Total predictions before NMS: {all_boxes.shape[0]}")
        logger.info(f"[I] Unique predicted labels: {all_labels.unique().tolist()}")

        # Safety check
        if max_labels.max() >= num_classes:
            logger.error(f"[E] Max predicted label = {max_labels.max().item()}, exceeds {num_classes}")

        # Clamp labels to avoid overflow
        all_labels = all_labels.clamp(0, num_classes - 1).to(dtype=torch.int64)

        # If debug_force_class is active, only forced labels are predicted
        if debug_force_class is not None:
            logger.info("[I] Skipping NMS because debug_force_class is active")
            
            # Keep all boxes
            keep = torch.arange(all_boxes.shape[0], device=all_boxes.device)  
        else:
            keep = batched_nms(all_boxes, all_scores, all_labels, nms_threshold)
            logger.info(f"[I] Predictions kept after NMS: {len(keep)}")
            unique_labels = all_labels[keep].unique()
            logger.info(f"[I] Labels surviving NMS: {unique_labels.tolist()}")

        # After NMS
        logger.info(f"[I] Predictions kept after NMS: {len(keep)}")
        unique_labels = all_labels[keep].unique()
        logger.info(f"[I] Labels surviving NMS: {unique_labels.tolist()}")
        
        batch_predictions.append({
            "boxes": all_boxes[keep],
            "scores": all_scores[keep],
            "labels": all_labels[keep]
        })

        # Safety check: type check for all_labels
        if not torch.is_tensor(all_labels):
            logger.error("[E] all_labels is not a tensor!")
        elif all_labels.dtype != torch.int64:
            logger.error(f"[E] all_labels has wrong dtype: {all_labels.dtype}")

        all_labels_cpu = all_labels.detach().cpu()
        
        # Safety check: final label validity
        if all_labels_cpu.max().item() >= num_classes:
            logger.error(f"[E] decode_predictions() returned invalid label ≥ {num_classes}")
        elif all_labels_cpu.min().item() < 0:
            logger.error(f"[E] decode_predictions() returned negative label!")

        if all_labels_cpu.numel() > 0:
            logger.info(f"[I] Decode final check - unique predicted labels: {all_labels_cpu.unique().tolist()}")
        else:
            logger.warning("[W] Decode final check - no labels returned (empty batch)")

        # Normal exit with predictions
        return batch_predictions