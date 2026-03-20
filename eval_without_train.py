import torch
import os
from PIL import Image
from torchvision import transforms
from train import run_evaluation_after_training
from YOLO_with_ResNet50 import YOLOv3
from data_loader import COCO_Dataset
from train import collate_fn
from torch.utils.data import DataLoader

# Path
coco_data_dir = r"C:\\Users\\viktor.acs\\Downloads\\coco_dataset"
val_annotations_file = os.path.join(coco_data_dir, 'annotations', 'instances_val2017.json')
val_image_dir = os.path.join(coco_data_dir, 'val2017')

# Load checkpoint
checkpoint = torch.load("yolov3_checkpoint_last_epoch.pth", map_location="cuda")
num_classes = 80

"""
anchors = [
    [10, 13], [16, 30], [33, 23],
    [30, 61], [62, 45], [59, 119],
    [116, 90], [156, 198], [373, 326]
]
"""

# K-means generate by k_means_anchor_calculator.py on the resized COCO dataset (416x416):
anchors = [
    [19.39769772, 24.12491592], [40.91803117, 76.44313414], [113.50188071, 69.27488936],
    [71.71268453, 161.95265297], [105.08463053, 285.58598963], [191.49338089, 161.76429439],
    [351.54059713, 159.67305114], [224.41982096, 331.08204223], [381.16142647, 359.7244103]
]

model = YOLOv3(num_classes=num_classes, anchors=anchors)
model.load_state_dict(checkpoint["model_state_dict"])
model.to("cuda")
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert the NumPy arrays into PIL images
    transforms.Resize((416, 416), interpolation=Image.BILINEAR),  # Resize the image directly to 416x416
    transforms.ToTensor(),  # Convert the image to tensor (C, H, W)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Build val dataloader
val_dataset = COCO_Dataset(
    image_dir=val_image_dir,
    annotation_file=val_annotations_file,
    transform=transform
)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Run eval
mAP = run_evaluation_after_training(model, val_dataloader, "cuda", val_annotations_file)
print(f"Validation mAP@[.5:.95]: {mAP:.4f}")
