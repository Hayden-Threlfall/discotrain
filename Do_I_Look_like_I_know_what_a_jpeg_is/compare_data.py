#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 23:32:21 2024

@author: semoreas
"""

import cv2
import numpy as np
import sys
import math

# Function to compare black and white values of two frames
def compare_frames(frame1, frame2):
    # Convert frames to grayscale
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference between grayscale frames
    diff = cv2.absdiff(gray_frame1, gray_frame2)
    
    # Sum of pixel values of the absolute difference frame
    diff_sum = np.sum(diff)
    
    diff_sum = diff_sum/(2*(640*480))
    return diff_sum

# Paths to the videos

video1_path = 'videos/M-T.mp4'
video2_path = 'videos/M-T.mp4'

# Open the videos
cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

# Check if videos opened successfully
if not (cap1.isOpened() and cap2.isOpened()):
    print("Error: One or both videos could not be opened.")
    sys.exit()

# Loop through frames
while True:
    # Read frames
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    # Break loop if either video reaches its end
    if not (ret1 and ret2):
        break
    
    # Compare frames
    diff_sum = compare_frames(frame1, frame2)
    
    # Print the comparison result
    if diff_sum > 40:
        diff_sum = 40
    diff_sum = diff_sum * 1.66
    
    
    print("Difference in black and white values:", diff_sum)
    
    # Display the frames (optional)
    cv2.imshow('Frame 1', frame1)
    cv2.imshow('Frame 2', frame2)
    
    # Break loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release video capture objects
cap1.release()
cap2.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
