#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import numpy as np
import time
import math
#from super_gradients.training import models
#import threading

def nothing(x):
    # any operation
    pass

# def skeleton_detect(imagePath):
#     global x1,x2,y1,y2
#     model = models.get("yolo_nas_pose_l", pretrained_weights="coco_pose")
#     prediction = model.predict(imagePath)
#     x1, y1, x2, y2 = prediction.prediction.bboxes_xyxy[0]
#     return int(x1), int(y1), int(x2), int(y2)
    
    

def crop_frame(frame, x1, y1, x2, y2):
    cropped_frame = frame[y1:y2, x1:x2]
    return cropped_frame

def findAngle(pt1, pt2):
    # Calculate the angle from pt1 to pt2 from the horizontal
    deltaY = pt2[1] - pt1[1]
    deltaX = pt2[0] - pt1[0]
    angleRad = math.atan2(deltaY, deltaX)
    angleDeg = math.degrees(angleRad)
    angleDeg = ( -1 * (angleDeg + 360)) % 360
    return angleDeg

if "__main__":
    x1 = 0
    x2 = 0
    y1 = 0
    y2 =0
    cap = cv2.VideoCapture(0)
    #(720,1280,3)

    
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("L-H", "Trackbars", 0, 120, nothing)
    cv2.createTrackbar("L-S", "Trackbars", 0, 120, nothing)
    cv2.createTrackbar("L-V", "Trackbars", 0, 50, nothing)
    cv2.createTrackbar("U-H", "Trackbars", 120, 255, nothing)
    cv2.createTrackbar("U-S", "Trackbars", 120, 255, nothing)
    cv2.createTrackbar("U-V", "Trackbars", 120, 255, nothing)
    cv2.createTrackbar("e", "Trackbars", 300, 1000, nothing) #divide by one thousands
    
    font = cv2.FONT_HERSHEY_COMPLEX
    
    
    output_file = 'output.mp4'  # Output video file name with mp4 extension
    output_file2 = 'output2.mp4'
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change codec to mp4v
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (640, 480))
    out2 = cv2.VideoWriter(output_file2, fourcc, 20.0, (640, 480))
    
    begin = time.time()
    count = time.time()
   # cY = 0
   # cX = 0
    center = []
    bash_prevention = 0
    mask = []
    first = True
    while True:
        ret, frame = cap.read()
        #print(type(frame))
        #print(frame.shape)
        frameCenter = (frame.shape[1] // 2, frame.shape[0] // 2)
                
        
        
        if ret == True:
            if not ret:
                break
        
   
            try:
        #Convert this to RGB
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            except:
                pass
        
            # if(first ):
            #     skeleton_detect(frame)
            #     first = False

            #frame = crop_frame(frame, x1, y1, x2, y2)       
            l_h = cv2.getTrackbarPos("L-H", "Trackbars")
            l_s = cv2.getTrackbarPos("L-S", "Trackbars")
            l_v = cv2.getTrackbarPos("L-V", "Trackbars")
            u_h = cv2.getTrackbarPos("U-H", "Trackbars")
            u_s = cv2.getTrackbarPos("U-S", "Trackbars")
            u_v = cv2.getTrackbarPos("U-V", "Trackbars")
            e = cv2.getTrackbarPos("e", "Trackbars")
        
        
                    
            lower_red = np.array([l_h, l_s, l_v])
            upper_red = np.array([u_h, u_s, u_v]) #150 190 229
        
            mask = cv2.inRange(frame, lower_red, upper_red) # frame was hsv
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel)

             # Perform morphological closing to fill small holes and gaps
            kernel = np.ones((5,5), np.uint8)
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
            # Invert the image
            inverted = cv2.bitwise_not(closed)
        
            # Perform morphological operations to remove isolated black spots
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(inverted, kernel, iterations=1)
            eroded = cv2.erode(dilated, kernel, iterations=1)
        
        
            
            
            # Invert the result back
            mask = cv2.bitwise_not(eroded)
            



            
            # Contours detectison
            if int(cv2.__version__[0]) > 3:
                # Opencv 4.x.x
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            else:
                # Opencv 3.x.x
                _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
            max_area = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                #Change this value here to better represent the polygon
                #Lower == More rounder
                #Upper == More sides
                approx = cv2.approxPolyDP(cnt, (e/100000)*cv2.arcLength(cnt, True), True)
                x = approx.ravel()[0]
                y = approx.ravel()[1]
        
                if area > 500: #Change this to represent full body
                    max_area = area
                
                    cv2.drawContours(frame, [approx], 0, (0, 0, 0), 5)
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        leafCenter = (cX, cY)
                        cv2.circle(hsv, (cX, cY), 5, (255, 0, 0), -1)
                        

                        #Now center the output frame based on the center of the 
                        # center the x to match the center

            try:
                center, radius = cv2.minEnclosingCircle(np.concatenate(contours))
            except:
                pass

            if(time.time()-begin > 10):
               # print("Scouting videos")
                m = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                
                out.write(mask)
                out2.write(frame)
            cv2.imshow("Frame", frame)
            cv2.imshow("Mask", mask)
            
            if cv2.waitKey(1) & 0xFF == ord('s'): 
                break
            if abs(time.time()-begin - 180):
                ret = False
            
        # Break the loop 
        else: 
            break
    
    out.release() 
    out2.release()
    cap.release()
    cv2.destroyAllWindows()
    
    #Align the body to the center using bodily moment of inertia
    
    
    
    
    
    
    
    