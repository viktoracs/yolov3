from pycocotools.coco import COCO
from torch.utils.data import Dataset
from logger import logger
import os
import cv2
import torch
import json
import random
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

class COCO_Dataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None, subset_size=None, fixed_image_id=None):
        self.image_dir = image_dir
        self.annotation_file = annotation_file

        # Validate the annotation file
        self.validate_annotations(annotation_file)

        self.coco = COCO(annotation_file)
        self.transform = transform

        # Create COCO ID -> Model Index mapping inside COCO_Dataset
        coco_category_ids = sorted(self.coco.getCatIds())  # Get sorted COCO category IDs
        self.coco_id_to_model_index = {coco_id: idx for idx, coco_id in enumerate(coco_category_ids)} # Example: {1: 0, ... 18: 16, ...}

        # Handle fixed image case (zebra single image overfit scenario)
        if fixed_image_id is not None:
            if fixed_image_id in self.coco.imgs:
                self.image_ids = [fixed_image_id]  # Use only this image
                logger.info(f"[I] Using fixed image ID: {fixed_image_id}")
            else:
                raise ValueError(f"[E] Fixed image ID {fixed_image_id} not found in dataset!")
        else:
            self.image_ids = self.coco.getImgIds()  # ensures only valid image_ids present in GT

            # Constrain to a smaller subset if specified
            if subset_size is not None and subset_size < len(self.image_ids):
                self.image_ids = self.image_ids[:subset_size]
                logger.info(f"[I] Dataset constrained to {subset_size} samples.")
            else:
                logger.info(f"[I] Using full dataset with {len(self.image_ids)} samples.")

    def __len__(self):
        return len(self.image_ids)

    # Responsible for mapping (COCO ID -> model index)
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]  # Always the same for fixed image case
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
    
        # Ensure image exists
        if not os.path.exists(image_path):
            logger.warning(f"Warning: Image {image_path} does not exist.")
            return self.__getitem__((idx + 1) % len(self))  # Load next image if missing
    
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
        # Save original image size (H, W)
        original_size = (image.shape[0], image.shape[1])  # (Height, Width)
    
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(ann_ids)
    
        boxes = []
        labels = []
    
        for ann in annotations:
        
            coco_id = ann['category_id']
    
            if coco_id in self.coco_id_to_model_index:
                model_index = self.coco_id_to_model_index[coco_id]
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])  # Convert to corner format (x_min, y_min, x_max, y_max) for the model [COCO format uses (x_min, y_min, width, height)]
            else:
                print(f"[I] Skipping unknown COCO ID: {coco_id}")
                continue

            # Safety check: ensure it’s an integer
            assert isinstance(model_index, int), f"[E] Model Index is not an integer! Got: {type(model_index)}"
            
            if not isinstance(model_index, int) or model_index < 0 or model_index > 79:
                logger.error(f"[E] Invalid model index detected! COCO ID {coco_id} → Model Index {model_index}")
            
            labels.append(model_index)

        # Convert to NumPy arrays
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # Extended augmentations
        if boxes.shape[0] > 0:
            
            # 1: Random horizontal flip (geometrical)
            if random.random() < 0.5:
                image = cv2.flip(image, 1)
                W = original_size[1]
                x_min = boxes[:, 0].copy()
                x_max = boxes[:, 2].copy()
                boxes[:, 0] = W - x_max
                boxes[:, 2] = W - x_min

            # 2: Color augmentation (HSV) - no bbox alignment is needed
            if random.random() < 0.5:
                # Convert RGB -> HSV
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

                # Hue: small shift
                hsv[..., 0] += random.uniform(-5, 5)

                # Saturation: scale
                hsv[..., 1] *= random.uniform(0.7, 1.3)

                # Value (brightness): scale
                hsv[..., 2] *= random.uniform(0.7, 1.3)

                # Clip to valid HSV ranges
                hsv[..., 0] = np.clip(hsv[..., 0], 0, 179)
                hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
                hsv[..., 2] = np.clip(hsv[..., 2], 0, 255)

                # Convert back HSV -> RGB
                image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Apply transformations
        if self.transform:

            # Get augmented size before resizing (image is still a NumPy array here)
            # OpenCV / NumPy images are: image.shape = (Height, Width, Channels)
            H_aug, W_aug = image.shape[:2]

            # Apply torchvision transform (Resize -> ToTensor -> Normalize)
            image = self.transform(image)

            # Scale bounding boxes based on augmentation size
            scale_x = 416 / W_aug
            scale_y = 416 / H_aug

            if boxes.size > 0:
                boxes[:, 0] *= scale_x
                boxes[:, 1] *= scale_y
                boxes[:, 2] *= scale_x
                boxes[:, 3] *= scale_y
    
        # Filter out invalid boxes (zero-area boxes)
        valid_boxes = []
        valid_labels = []
        
        for box, label in zip(boxes, labels):
            if (box[2] > box[0]) and (box[3] > box[1]):  # Safety check: ensure valid width & height
                valid_boxes.append(box)
                valid_labels.append(label)

        # Safety check: will be filtered in collate_fn() anyway
        if len(valid_labels) == 0:
            return None
                
        boxes = np.array(valid_boxes)
        labels = np.array(valid_labels)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        # Final safety checks
        assert labels.ndim == 1, f"[W] Labels shape is unexpected: {labels.shape}"
        assert (labels >= 0).all() and (labels <= 79).all(), f"[W] Invalid label values: {labels}"

        return image, boxes, labels, original_size, image_id

    # Validates that the annotation file contains required COCO keys
    def validate_annotations(self, annotation_file):
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        required_keys = {'images', 'annotations', 'categories'}
        if not required_keys.issubset(data.keys()):
            raise ValueError(f"[E] Annotation file is missing required keys: {required_keys - data.keys()}")
        print("[I] Annotation file validation PASSED.")

    # Display the fixed image with GT bboxes (for verification - one image overfit scenario)
    def show_fixed_image_with_gt_bb(self):
        
        if len(self.image_ids) == 1:
            idx = 0  # Since it's always the same image
        else:
            idx = random.randint(0, len(self.image_ids) - 1)
    
        # Update to unpack 5 values
        transformed_image, scaled_boxes, labels, original_size, image_id = self[idx]  
    
        # Convert transformed image back to a displayable format (denormalization)
        image = transformed_image.permute(1, 2, 0).cpu().numpy()  # Convert from (C, H, W) to (H, W, C)
        image = (image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]  # Denormalize
        image = (image * 255).astype("uint8")  # Rescale to [0, 255]
    
        # Plot the image
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(image)
    
        # Draw bounding boxes
        for box, label in zip(scaled_boxes, labels):
            x_min, y_min, x_max, y_max = box
    
            rect = patches.Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                linewidth=1, edgecolor="yellow", facecolor="none", linestyle="dashed"
            )
            ax.add_patch(rect)
    
        plt.axis("off")
        plt.title(f"Image ID {image_id} with GT BB(s)")
        plt.show()