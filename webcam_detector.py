import warnings
import cv2
import torch
import time
import os
from datetime import datetime
from torchvision import transforms
from YOLO_with_ResNet50 import YOLOv3


# ============================================================
# Configuration
# ============================================================

CHECKPOINT_PATH = "models/yolov3_checkpoint_last_epoch.pth"
CONF_THRESHOLD = 0.60
NMS_THRESHOLD = 0.40
SCREENSHOT_DIR = "webcam_screenshots"
VIDEO_DIR = "webcam_recordings"

os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# ============================================================
# Warnings
# ============================================================

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message="expandable_segments not supported"
)


# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[I] Device: {device}")


# ============================================================
# COCO class mapping
# ============================================================

coco_class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

coco_classes = {
    idx: name
    for idx, name in enumerate(coco_class_names)
}

# ============================================================
# Anchors
# ============================================================

num_classes = 80

anchors = [
    [19.39769772, 24.12491592],
    [40.91803117, 76.44313414],
    [113.50188071, 69.27488936],

    [71.71268453, 161.95265297],
    [105.08463053, 285.58598963],
    [191.49338089, 161.76429439],

    [351.54059713, 159.67305114],
    [224.41982096, 331.08204223],
    [381.16142647, 359.7244103],
]

strides = [8, 16, 32]

# IMPORTANT:
# decode_predictions expects grid-relative anchors.
scaled_anchors = (
    [(w / strides[0], h / strides[0]) for w, h in anchors[:3]]
    + [(w / strides[1], h / strides[1]) for w, h in anchors[3:6]]
    + [(w / strides[2], h / strides[2]) for w, h in anchors[6:]]
)


# ============================================================
# Model
# ============================================================

print(f"[I] Loading checkpoint: {CHECKPOINT_PATH}")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)

model = YOLOv3(
    num_classes=num_classes,
    anchors=anchors,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print(f"[I] Loaded checkpoint epoch: {checkpoint.get('epoch')}")


# ============================================================
# Preprocessing
#
# This matches the validation preprocessing:
# ToPILImage -> Resize(BILINEAR) -> ToTensor -> Normalize
# ============================================================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(
        (416, 416),
        interpolation=transforms.InterpolationMode.BILINEAR,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# Video input
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("[I] Webcam opened successfully.")
print("[I] Controls: Q/Esc = quit | S = screenshot | R = start/stop recording")

# ============================================================
# Inference loop
# ============================================================

processed = 0

recording = False

writer = None

print("[I] Starting video inference...")

prev_time = time.perf_counter()
fps_smooth = 0.0

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("[E] Failed to read frame from webcam.")
            break

        H, W = frame.shape[:2]

        # OpenCV frame: BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_tensor = (
            transform(frame_rgb)
            .unsqueeze(0)
            .to(device)
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        infer_start = time.perf_counter()

        # --------------------------------------------
        # Forward pass
        # --------------------------------------------
        with torch.inference_mode():
            outputs = model(input_tensor)

            if device.type == "cuda":
                torch.cuda.synchronize()

            infer_end = time.perf_counter()

            inference_ms = (infer_end - infer_start) * 1000.0

            preds = model.decode_predictions(
                outputs,
                anchors=scaled_anchors,
                num_classes=num_classes,
                image_w=416,
                image_h=416,
                conf_threshold=CONF_THRESHOLD,
                nms_threshold=NMS_THRESHOLD,
                debug_force_class=None,
            )

        pred = preds[0]

        boxes = pred["boxes"].detach().cpu().numpy()
        scores = pred["scores"].detach().cpu().numpy()
        labels = pred["labels"].detach().cpu().numpy()

        # --------------------------------------------
        # Scale boxes from 416x416 back to video frame
        # --------------------------------------------
        if len(boxes) > 0:
            sx = W / 416.0
            sy = H / 416.0

            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy

        # --------------------------------------------
        # Draw detections
        # --------------------------------------------
        for (x1, y1, x2, y2), conf, cls in zip(
            boxes,
            scores,
            labels,
        ):
            class_name = coco_classes.get(
                int(cls),
                f"id_{int(cls)}",
            )

            label_text = f"{class_name}: {conf * 100:.0f}%"

            # Green bbox / label background.
            color = (0, 255, 0)

            # Clip coordinates to frame boundaries.
            x1 = max(0, min(int(x1), W - 1))
            y1 = max(0, min(int(y1), H - 1))
            x2 = max(0, min(int(x2), W - 1))
            y2 = max(0, min(int(y2), H - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            box_w = x2 - x1
            box_h = y2 - y1

            # Adaptive text size.
            font_scale = max(
                0.3,
                min(0.65, box_h / 120.0),
            )

            thickness = 1

            (text_w, text_h), _ = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness,
            )

            # Prefer label inside bottom-left of bbox.
            if text_h + 6 < box_h and text_w + 6 < box_w:
                text_x = x1
                text_y = y2
            else:
                text_x = x1
                text_y = max(text_h + 5, y1 - 5)

            # Bounding box.
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            # Filled text background.
            bg_top = max(0, text_y - text_h - 5)
            bg_right = min(W - 1, text_x + text_w + 4)

            cv2.rectangle(
                frame,
                (text_x, bg_top),
                (bg_right, text_y),
                color,
                -1,
            )

            # Black text.
            cv2.putText(
                frame,
                label_text,
                (text_x + 2, text_y - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )

        current_time = time.perf_counter()
        frame_time = current_time - prev_time
        prev_time = current_time

        instant_fps = 1.0 / frame_time if frame_time > 0 else 0.0

        # Smooth the displayed FPS so it does not jump constantly
        if fps_smooth == 0.0:
            fps_smooth = instant_fps
        else:
            fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps

        # Model inference = how fast the neural network runs (one forward pass)
        # Pipeline FPS = how fast the whole application can process frames
        stats_text = f"Pipeline FPS: {fps_smooth:.1f} | Model inference: {inference_ms:.1f} ms"

        cv2.putText(
            frame,
            stats_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("YOLOv3 Webcam Detection", frame)

        # Write annotated frame if recording is active
        if recording and writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            print("[I] Webcam detection stopped by user.")
            break

        elif key in (ord("s"), ord("S")):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            screenshot_path = os.path.join(
                SCREENSHOT_DIR,
                f"webcam_detection_{timestamp}.jpg"
            )

            cv2.imwrite(screenshot_path, frame)

            print(f"[I] Screenshot saved to: {screenshot_path}")

        elif key in (ord("r"), ord("R")):

            if not recording:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                video_path = os.path.join(
                    VIDEO_DIR,
                    f"webcam_detection_{timestamp}.mp4"
                )

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                writer = cv2.VideoWriter(
                    video_path,
                    fourcc,
                    18.0,
                    (W, H),
                )

                if not writer.isOpened():
                    print("[E] Could not start video recording.")
                    writer = None

                else:
                    recording = True
                    print(f"[I] Recording started: {video_path}")

            else:
                recording = False

                if writer is not None:
                    writer.release()
                    writer = None

                print("[I] Recording stopped.")

finally:
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


