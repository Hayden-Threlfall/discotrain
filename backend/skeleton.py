import super_gradients
from super_gradients.training import models
import cv2

model = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")
prediction = model.predict("BaselineImages\Baseline1.jpg")

image = cv2.imread("Baseline1.jpg")
for i in prediction.prediction.poses[0]:
    x, y, c = i
    cv2.circle(image,(int(x), int(y)), 1, (0,0,255), -1)

#print(prediction.prediction.poses)

prediction2 = model.predict("BaselineImages\Baseline2.jpg")

for i in prediction2.prediction.poses[0]:
    x, y, c = i
    cv2.circle(image,(int(x), int(y)), 3, (255,0,0), -1)

cv2.imshow("Baseline1.jpg", image)

cv2.waitKey(0)