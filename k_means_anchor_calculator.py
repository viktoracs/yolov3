import os
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from data_loader import COCO_Dataset   
from logger import logger

"""
To adapt the YOLOv3 detector to the actual object size distribution in the resized COCO dataset, a custom anchor generator was implemented using K-means clustering. 
Instead of relying on the original YOLOv3 anchors — which were computed on unscaled COCO images using multi-scale training - this script extracts all GT bboxes after they have been normalized to the model’s fixed 416×416 input resolution. 
It then runs K-means (k=9) on the resulting width–height pairs to produce anchors that more accurately reflect the resized dataset.

The script also filters invalid or padded boxes, flattens nested annotations and ensures that the final dataset fed to K-means has the correct (N×2) shape.
"""

def kmeans_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)

# Paths and parameters
coco_data_dir = r"C:\\Users\\viktor.acs\\Downloads\\coco_dataset"
train_annotations_file = os.path.join(coco_data_dir, "annotations", "instances_train2017.json")
train_image_dir = os.path.join(coco_data_dir, "train2017")

num_clusters = 9
target_size = 416
max_images = None 

dataset = COCO_Dataset(
    image_dir=train_image_dir,
    annotation_file=train_annotations_file,
    transform=None
)

loader = DataLoader(
    dataset, 
    batch_size=1, 
    shuffle=False, 
    collate_fn=kmeans_collate
)

bboxes = []
count = 0

for batch in tqdm(loader, desc="Collecting boxes for K-means"):

    if batch is None:
        continue

    image, boxes, labels, orig_size, _ = batch
    orig_h, orig_w = orig_size

    for box in boxes[0]:
        box = box.tolist()

        # Flatten nested lists [[x,y,x2,y2]] -> [x,y,x2,y2]
        while isinstance(box[0], list):
            box = box[0]

        # Skip padded or invalid boxes
        if len(box) != 4:
            continue
        if box == [0, 0, 0, 0]:
            continue

        x_min, y_min, x_max, y_max = box

        # Skip invalid boxes
        w = x_max - x_min
        h = y_max - y_min
        if w <= 0 or h <= 0:
            continue

        # Normalize to 416×416
        norm_w = (w / orig_w) * target_size
        norm_h = (h / orig_h) * target_size

        # Skip degenerate
        if norm_w <= 0 or norm_h <= 0:
            continue

        bboxes.append([float(norm_w), float(norm_h)])

    count += 1
    if max_images is not None and count >= max_images:
        break

logger.info(f"[I] Total GT boxes collected: {len(bboxes)}")

bboxes = np.array(bboxes)
logger.info(f"[I] Total GT boxes collected: {len(bboxes)}")

print("DEBUG: bboxes type:", type(bboxes))
print("DEBUG: bboxes shape:", bboxes.shape)
print("DEBUG: example entry:", bboxes[0])

if len(bboxes) < num_clusters:
    raise ValueError("[E] Not enough bounding boxes for clustering.")

kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(bboxes)
anchors = kmeans.cluster_centers_
anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]

logger.info("\n[I] Final sorted anchors:")
for i, (w, h) in enumerate(anchors):
    logger.info(f"[I] Anchor {i}: w={w:.1f}, h={h:.1f}")

print("\nFinal Anchors:")
print(anchors)
