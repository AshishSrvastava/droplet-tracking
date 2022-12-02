# for passing in video via terminal 
import sys

import cv2
import numpy as np

# Input

cap = cv2.VideoCapture(sys.argv[1])
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # 1920
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 1080

# Outputting a video
# Define the codec and create VideoWriter object
# macOS: DIVX for .avi
# fourcc = cv2.VideoWriter_fourCC('D', 'I', 'V', 'X')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))

while cap.isOpened:
    ret, frame = cap.read()
    if not ret:
        print("Can't recieve frame (stream end?). Exiting...")
        break

    cv2.imshow("Original", frame)


    # Filter experimentation
    # blurred_frame = cv2.GaussianBlur(frame, (5,5), 0)
    # cv2.imshow("Blurred", blurred_frame)

    bilateral_filtered_frame = cv2.bilateralFilter(frame, 5, 175, 175)
    cv2.imshow('Bilateral', bilateral_filtered_frame)


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([63, 0, 0])
    upper_blue = np.array([179, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    cv2.imshow("Mask", mask)

    # key = cv2.waitKey(100)
    # if key == 27:
    #     break

    # be able to play and pause video
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        cv2.waitKey(-1) # wait until any key is pressed

cap.release()
cv2.destroyAllWindows()
