# A custom (FPN) YOLOv3-style object detector with ResNet-50 backbone trained on COCO and implemented in PyTorch

## This repository is intended as:
- A full end-to-end object detection implementation example
- A portfolio project demonstrating PyTorch and object detection experience

### The requirements of the Python environment are stored in requirements.txt. The model is stored separately. 

### Trained on NVIDIA GeForce RTX 4070 Ti.

### mAP score achieved:
- **mAP@[0.5:0.95]: 0.1275**
- **AP50: 0.271**
- **AP75: 0.105**

This result was obtained with:
- Fixed input size: 416x416
- Optimizer: Adam
- Scheduler: OneCycleLR
- Epochs: 100
- Simple augmentations (horizontal flip, hue, saturation, brightness)
 
### Executable files

### random_image_detector.py:

	Please start this script from the same folder where the model and project's .py files are.
	It can handle any image resolutions and accepts the following extensions: .jpg, .jpeg, .png, .bmp, .webp 
	It takes the test image from the hardcoded folders. If there are multiple images, the selection is random.

	# For COCO val images
	image_dir = r"C:\Users\viktor.acs\Downloads\coco_dataset\val2017"
	OR
	# For custom images
	image_dir = r"C:\Users\viktor.acs\Downloads\coco_dataset\test_images"

	There are no additional parameters, just simply run: python random_image_detector.py
	The image with YOLO-style predictions (prediction_result.jpg) will be saved to the same folder where the script is. 
	You can set the NMS and the confidence threshold in the "decode_predictions" function manually:
	preds = model.decode_predictions(
	    ...
	    conf_threshold=0.75,
	    nms_threshold=0.25,
	    ...

### debug_visualizer_lossmaps_updated.py:

	Please start this script from the same folder where the model and project's .py files are.
	This visualizer script takes either the reference or a random image:
	python debug_visualizer_lossmaps_updated.py
	python debug_visualizer_lossmaps_updated.py --random

### eval_without_train.py:

	Please start this script from the same folder where the model and project's .py files are.	
	There are no additional parameters, just simply run: python eval_without_train.py 
	It runs evaluation without training.
	Note: Please comment out the extended augmentations part (# Extended augmentations) in data_loader.py to get valid (non-augmented) mAP scores.

### train.py:

	Please start this script from the same folder where the model and project's .py files are.
	There are no additional parameters, just simply run: python train.py 	
	It starts the full training. Either from scratch or from a saved checkpoint.
	Note: Please check the checkpoint logic.

### k_means_anchor_calculator.py

	Please start this script from the same folder where the model and project's .py files are.
	There are no additional parameters, just simply run: python k_means_anchor_calculator.py
	It calculates the k-means anchors for the pipeline.

### Limitations:
- Performance remains below stronger reference YOLOv3 implementations
- Ranking/calibration is weaker than ideal
