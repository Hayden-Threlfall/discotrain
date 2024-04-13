import super_gradients
from super_gradients.training import models
import cv2
import numpy
import time


#model = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")
#model.predict("output.mp4").save("output3.mp4")
#model = models.get("pp_lite_t_seg50", pretrained_weights="cityscapes")


model = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")

prediction = model.predict("input.jpg")
print(prediction.prediction.poses)
prediction.prediction.poses[0][0] += prediction.prediction.poses[0][0]
print(prediction.prediction.poses)
prediction.show()


#file = open("output1.txt", "w+")
#file.write(str(prediction))
#file.close()
#prediction2 = model.predict("input3.jpg")