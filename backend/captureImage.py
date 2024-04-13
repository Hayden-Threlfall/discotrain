import cv2

cap = cv2.VideoCapture(0)


ret, frame = cap.read()

cv2.imwrite("Baseline1.jpg", frame)
# Display the resulting frame
cv2.imshow('Baseline', frame)
cv2.waitKey(0)