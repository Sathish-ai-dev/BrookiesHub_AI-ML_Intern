# detect.py

import sys
import cv2
import torch
import numpy as np

# Add YOLOv5 repo to path
sys.path.append('./yolov5')

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# Load model
device = select_device('')
model = DetectMultiBackend('yolov5s.pt', device=device)
model.eval()

def detect_objects(frame):
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img = img.to(device)

    with torch.no_grad():
        pred = model(img)
        pred = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.45)

    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return frame
