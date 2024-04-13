import super_gradients
from super_gradients.training import models
import cv2
import math

model = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")
prediction = model.predict("BaselineImages\Baseline1.jpg")

image = cv2.imread("BaselineImages\Baseline1.jpg")
pose1 = []
for i in prediction.prediction.poses[0]:
    x, y, c = i
    pose1.append((x, y))
    cv2.circle(image,(int(x), int(y)), 1, (0,0,255), -1)

#print(prediction.prediction.poses)

prediction2 = model.predict("BaselineImages\Baseline2.jpg")
pose2 = []
for i in prediction2.prediction.poses[0]:
    x, y, c = i
    pose2.append((x, y))
    cv2.circle(image,(int(x), int(y)), 3, (255,0,0), -1)

distance = 0
for i in range(0, len(pose1)):
    distance += math.dist(pose1[i], pose2[i])
print(distance)
distance = 0
for i in range(0, len(pose1)):
    distance += math.dist(pose1[i], pose1[i])
print(distance)

cv2.imshow("BaselineImages\Baseline1.jpg", image)

cv2.waitKey(0)
