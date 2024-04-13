import super_gradients
from super_gradients.training import models
import math
import time

def getKeyPoints(image1):
    model = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")
    prediction = model.predict(image1)

    pose1 = []
    for i in prediction.prediction.poses[0]:
        x, y, c = i
        pose1.append((x, y))

    return pose1

def getMultipleKeyPoints(image1, image2):
    model = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")
    prediction = model.predict(image1)

    pose1 = []
    for i in prediction.prediction.poses[0]:
        x, y, c = i
        pose1.append((x, y))

    prediction2 = model.predict(image2)

    pose2 = []
    for i in prediction2.prediction.poses[0]:
        x, y, c = i
        pose2.append((x, y))

    return pose1, pose2

def getScore(pose1, pose2):
    if len(pose1) != len(pose2):
        print("ERROR: Poses have different dimensions.")
        return -1
    distance = 0
    for i in range(0, len(pose1)):
        distance += math.dist(pose1[i], pose2[i])
    return distance


def centerModels(image1, image2):
    print("TODO")

current_timestamp = time.time()
print(current_timestamp)
pose1, pose2 = getMultipleKeyPoints("BaselineImages\Baseline1.jpg","BaselineImages\Baseline2.jpg")

score = getScore(pose1, pose2)
print(score)
current_timestamp = time.time()
print(current_timestamp)