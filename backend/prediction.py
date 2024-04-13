import super_gradients
from super_gradients.training import models
import cv2
import numpy
import time
from super_gradients.training.utils.visualization.pose_estimation import PoseVisualization
import matplotlib.pyplot as plt


model = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")
#model.predict("output.mp4").save("output3.mp4")
#model = models.get("pp_lite_t_seg50", pretrained_weights="cityscapes")

prediction = model.predict("input.jpg")
print(prediction.prediction)

prediction.show()