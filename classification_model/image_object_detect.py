# import libraries
import torch
from PIL import Image
import sys
sys.path.append("..")

# Load YOLOv5 with PyTorch Hub (Object Detection Model)
model = torch.hub.load("ultralytics/yolov5", "yolov5s", _verbose=False)

# Define function to extract object names from image with YOLOv5 Model
def extract_objectNames(image_path):
    # open the image from path
    image = Image.open(image_path)
    
    # Perform object detection
    results = model(image)

    # Extract detected object names
    object_names = results.names
    detected_objects = {}

    for *box, conf, cls in results.xyxy[0]:  # xyxy format
        object_name = object_names[int(cls)]
        confidence = conf.item()  # Convert tensor to Python float
        detected_objects[object_name] = round(confidence, 2)

    # get the detected object names
    return detected_objects
