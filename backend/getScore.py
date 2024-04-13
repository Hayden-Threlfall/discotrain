#import super_gradients
#from super_gradients.training import models
import math
import time
import cv2
import json
import numpy
'''
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
'''
def getScore(pose1, pose2):
    if len(pose1) != len(pose2):
        if len(pose1) < len(pose2):
            count = len(pose1)
        else:
            count = len(pose2)
    else:
        count = len(pose1)
    distance = 0.0
    for i in range(0, count):
        print((int(pose1[i][0][0]), int(pose1[i][0][1])))
        print((int(pose2[i][0][0]), int(pose2[i][0][1])))
        distance += math.dist((int(pose1[i][0][0]), int(pose1[i][0][1])), (int(pose2[i][0][0]), int(pose2[i][0][1])))
        print(distance)
    return distance

def getScore2(pose1, pose2):
    if len(pose1) != len(pose2):
        if len(pose1) < len(pose2):
            count = len(pose1)
        else:
            count = len(pose2)
    else:
        count = len(pose1)
    distance = []
    for i in range(0, count):
        print((int(pose1[i][0][0]), int(pose1[i][0][1])))
        print((int(pose2[i][0][0]), int(pose2[i][0][1])))
        distance.append(math.dist((int(pose1[i][0][0]), int(pose1[i][0][1])), (int(pose2[i][0][0]), int(pose2[i][0][1]))))
        print(distance)
    return distance

def getDeltas(pose1, pose2):
    deltaX = pose1[0][5][0] - pose2[0][5][0]
    deltaY = pose1[0][5][1] - pose2[0][5][1]
    deltaX += pose1[0][6][0] - pose2[0][6][0]
    deltaY += pose1[0][6][1] - pose2[0][6][1]
    deltaX += pose1[0][11][0] - pose2[0][11][0]
    deltaY += pose1[0][11][1] - pose2[0][11][1]
    deltaX += pose1[0][12][0] - pose2[0][12][0]
    deltaY += pose1[0][12][1] - pose2[0][12][1]
    deltaX /= 4
    deltaY /= 4
    return deltaX, deltaY

def centerModels(pose1, pose2):
    deltaX, deltaY = getDeltas(pose1, pose2)
    newPose2 = []
    if len(pose1) != len(pose2):
        if len(pose1) < len(pose2):
            count = len(pose1)
        else:
            count = len(pose2)
    else:
        count = len(pose1)
    for x in range(0, count):
        newPost = list(pose2[x])
        newPost[0] += deltaX
        newPost[1] += deltaY
        newPose2.append(tuple(newPost))
    return newPose2


def displayPoints(image, points):
    for i in points:
        x, y = i
        cv2.circle(image,(int(x), int(y)), 5, (0,0,255), -1)
    return image


def translateForLimbs(pose1, pose2):
    print("TODO")
    return pose1, pose2

'''
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
'''

def readKeyPoints(filePath):
    file = open(filePath, "r")
    lines = file.readlines()
    file.close()
    keyPoints = []
    for x in lines:
        line = []
        x = x.replace('\n', '')
        res = x.split(")")
        for y in res:
            if y != '':
                y = y.replace("(", '')
                res2 = y.split(", ")
                line.append((numpy.float32(res2[0]), numpy.float32(res2[1])))
        keyPoints.append(line)
    return keyPoints


def writeToFile(videoPath, keyPoints):
    file = open(videoPath.replace(".mp4", "Points.txt"), "w+")
    for x in keyPoints:
        for y in x:
            file.write("(" + str(y[0]) + ", " + str(y[1]) + ")")
                
        file.write('\n')
    file.close()


def showVideo(videoPath, keyPoints):
    count = 0
    counter = 0
    cap = cv2.VideoCapture(videoPath)
    out = cv2.VideoWriter('GracePoints2Out.mp4', -1, 20.0, (640,480))
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
            cv2.imshow("Points", image)
            cv2.waitKey(100)   
            counter += 1
        else:
            image = frame
        count += 1
        
        cv2.imshow("Points", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def writeVideoWithScore(videoPath, keyPoints, score):
    count = 0
    counter = 0
    cap = cv2.VideoCapture(videoPath)
    out = cv2.VideoWriter('GracePoints2Out.mp4', -1, 20.0, (640,480))
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        if counter >= len(keyPoints) and counter >= len(score):
            break
    # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if count % 20 == 0:
            image = displayPoints(frame, keyPoints[counter])
            font = cv2.FONT_HERSHEY_SIMPLEX 
            org = (50, 50) 
            fontScale = 1
            color = (255, 255, 255) 
            thickness = 2
            image = cv2.putText(image, 'Score: ' + str(score[counter]), org, font, fontScale, color, thickness, cv2.LINE_AA)
            out.write(image)
            out.write(image)
            out.write(image)
            out.write(image)
            out.write(image)
            out.write(image)
            counter += 1
        else:
            image = frame
        count += 1
        
        out.write(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()


def displayBothVideos(videoPath, keyPoints, videoPath2, keyPoints2, score):
    names = [videoPath, videoPath2]
    window_titles = ["Original Video", "Copy Attempt"]


    cap = [cv2.VideoCapture(i) for i in names]

    frames = [None] * len(names)
    ret = [None] * len(names)
    count = 0
    count2 = 0
    counter = 0
    counter2 = 0

    while True:

        for i,c in enumerate(cap):
            if c is not None:
                ret[i], frames[i] = c.read()


        for i,f in enumerate(frames):
            if ret[i] is True:
                if i == 0:
                    if count % 20 == 0:
                        image = displayPoints(f, keyPoints[counter])
                        cv2.imshow(window_titles[i], image)
                        cv2.waitKey(100)   
                        counter += 1
                    else:
                        image = f
                    count += 1
                    cv2.imshow(window_titles[i], image)
                    cv2.waitKey(1)
                else:
                    if count2 % 20 == 0:
                        image2 = displayPoints(f, keyPoints2[counter2])
                        font = cv2.FONT_HERSHEY_SIMPLEX 
                        org = (50, 50) 
                        fontScale = 1
                        color = (255, 255, 255) 
                        thickness = 2
                        image2 = cv2.putText(image2, 'Score: ' + str(score[counter2]), org, font, fontScale, color, thickness, cv2.LINE_AA)
                        counter2 += 1
                        cv2.imshow(window_titles[i], image2)
                        cv2.waitKey(500)   
                    else:
                        image2 = f
                    count2 += 1
                    cv2.imshow(window_titles[i], image2)
                    cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    for c in cap:
        if c is not None:
            c.release()

    cv2.destroyAllWindows()

#keyPoints = getVideoKeyPoints("Grace1.mp4")
#writeToFile("Grace1.mp4", keyPoints)
#keyPoints = getVideoKeyPoints("Grace2.mp4")
#writeToFile("Grace2.mp4", keyPoints)
#writeToFile("Andrew1.mp4", keyPoints)
#print(keyPoints)
keypoints = readKeyPoints("Grace1Points.txt")
#showVideo("Grace1.mp4", keypoints)
keypoints2 = readKeyPoints("Grace2Points.txt")

#keypoints2 = readKeyPoints("Andrew1Points.txt")

#keypoints2 = centerModels(keypoints, keypoints2)

score = getScore2(keypoints, keypoints2)

displayBothVideos("Grace1.mp4", keypoints, "Grace2.mp4", keypoints2, score)

'''
score = getScore(keypoints, keypoints2)
print(score)

keypoints2 = centerModels(keypoints, keypoints2)

score = getScore(keypoints, keypoints2)
print(score)

current_timestamp = time.time()
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