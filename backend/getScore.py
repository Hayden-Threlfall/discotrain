import super_gradients
from super_gradients.training import models
import math
import time
import cv2
import json
import numpy

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


def getDeltas(pose1, pose2):
    deltaX = pose1[5][0] - pose2[5][0]
    deltaY = pose1[5][1] - pose2[5][1]
    deltaX += pose1[6][0] - pose2[6][0]
    deltaY += pose1[6][1] - pose2[6][1]
    deltaX += pose1[11][0] - pose2[11][0]
    deltaY += pose1[11][1] - pose2[11][1]
    deltaX += pose1[12][0] - pose2[12][0]
    deltaY += pose1[12][1] - pose2[12][1]
    deltaX /= 4
    deltaY /= 4
    return deltaX, deltaY

def centerModels(pose1, pose2):
    deltaX, deltaY = getDeltas(pose1, pose2)
    print(deltaX)
    print(deltaY)
    print(pose2)
    newPose2 = []
    for x in range(0, len(pose2)):
        newPost = list(pose2[x])
        newPost[0] += deltaX
        newPost[1] += deltaY
        newPose2.append(tuple(newPost))
    print(newPose2)
    return newPose2


def displayPoints(image, points):
    for i in points:
        x, y = i
        print(int(x))
        print(int(y))
        cv2.circle(image,(int(x), int(y)), 5, (255,0,0), -1)
    cv2.imshow("Points", image)
    cv2.waitKey(0)   
    return image


def translateForLimbs(pose1, pose2):
    print("TODO")
    return pose1, pose2


def getVideoKeyPoints(videoPath):
    keyPoints = []
    cap = cv2.VideoCapture(videoPath)
    try:
        count = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if count % 20 == 0:
                keyPoints.append(getKeyPoints(frame))
            count += 1
        cap.release()
        cv2.destroyAllWindows()
    except:
        cap.release()
        cv2.destroyAllWindows()
    return keyPoints


def readKeyPoints(filePath):
    file = open(filePath, "r")
    lines = file.readlines()
    file.close()
    print(lines)
    keyPoints = []
    for x in lines:
        line = []
        x = x.replace('\n', '')
        res = x.split(")")
        for y in res:
            if y != '':
                y = y.replace("(", '')
                res2 = y.split(", ")
                print(res2)
                line.append((numpy.float32(res2[0]), numpy.float32(res2[1])))
        keyPoints.append(line)
    print(keyPoints)
    return keyPoints


def writeToFile(videoPath, keyPoints):
    file = open(videoPath.replace(".mp4", "Points.txt"), "w+")
    for x in keyPoints:
        print(x)
        for y in x:
            file.write("(" + str(y[0]) + ", " + str(y[1]) + ")")
            print("(" + str(y[0]) + ", " + str(y[1]) + ")")
                
        file.write('\n')
    file.close()
    print(keyPoints)


def showVideo(videoPath, keyPoints):
    count = 0
    counter = 0
    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
    # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if count % 20 == 0:
            image = displayPoints(frame, keyPoints[counter])
            counter += 1
        else:
            image = frame
        count += 1
        
        cv2.imshow("Points", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


keyPoints = getVideoKeyPoints("Grace1.mp4")
writeToFile("Grace1.mp4", keyPoints)
keyPoints = getVideoKeyPoints("Grace2.mp4")
writeToFile("Grace2.mp4", keyPoints)
keyPoints = getVideoKeyPoints("Andrew1.mp4")
writeToFile("Andrew1.mp4", keyPoints)
#print(keyPoints)
#keypoints = readKeyPoints("outputPoints.txt")
#showVideo("output.mp4", keypoints)


'''current_timestamp = time.time()
print(current_timestamp)
pose1, pose2 = getMultipleKeyPoints("BaselineImages\Baseline4.jpg","BaselineImages\Baseline5.jpg")

score = getScore(pose1, pose2)
print(score)


newpose2 = centerModels(pose1, pose2)
score = getScore(pose1, pose2)
print(score)

score = getScore(pose1, pose2)
print(score)
current_timestamp = time.time()
print(current_timestamp)

image = cv2.imread("BaselineImages\Baseline4.jpg")
for i in pose1:
    x, y = i

    cv2.circle(image,(int(x), int(y)), 2, (255,0,0), -1)
cv2.imshow("BaselineImages\Baseline4.jpg", image)


image4 = cv2.imread("BaselineImages\Baseline5.jpg")
for i in pose2:
    x, y = i

    cv2.circle(image4,(int(x), int(y)), 2, (0,255,0), -1)
cv2.imshow("BaselineImages\Baseline5.jpg Transformed", image4)

image2 = cv2.imread("BaselineImages\Baseline5.jpg")
for i in newpose2:
    x, y = i

    cv2.circle(image2,(int(x), int(y)), 2, (0,0,255), -1)
cv2.imshow("BaselineImages\Baseline5.jpg Transformed", image2)


image3 = cv2.imread("BaselineImages\Baseline4.jpg")
for i in pose1:
    x, y = i
    cv2.circle(image3,(int(x), int(y)), 1, (255,0,0), -1)
for i in pose2:
    x, y = i
    cv2.circle(image3,(int(x), int(y)), 2, (0,255,0), -1)
for i in newpose2:
    x, y = i
    cv2.circle(image3,(int(x), int(y)), 2, (0,0,255), -1)
cv2.imshow("BaselineImages\Baseline6.jpg", image3)

cv2.waitKey(0)'''