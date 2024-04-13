import super_gradients
from super_gradients.training import models
import cv2
import math

model = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")
model.predict("Grace1.mp4").save("Grace1Output.mp4")
