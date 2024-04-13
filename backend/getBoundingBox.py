import super_gradients
from super_gradients.training import models
import cv2


def getBoundingBox(imagePath):
    model = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")
    prediction = model.predict(imagePath)
    x1, y1, x2, y2 = prediction.prediction.bboxes_xyxy[0]
    return int(x1), int(y1), int(x2), int(y2)

image = cv2.imread("BaselineImages\Baseline1.jpg")
x1, y1, x2, y2 = getBoundingBox(image)

cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255),3)

cv2.imshow("BaselineImages\Baseline1.jpg", image)

cv2.waitKey(0)