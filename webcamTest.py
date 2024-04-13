import super_gradients
from super_gradients.training import models
import cv2
import numpy
import time


model = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")
model.predict_webcam()