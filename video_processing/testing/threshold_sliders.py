import cv2
import numpy as np
import sys

# Check if a video file is provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python threshold_sliders.py <video_path>")
    sys.exit(1)

video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)

# Check if the video is opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    sys.exit(1)

# Read the first frame of the video
ret, image = cap.read()
if not ret:
    print(f"Error: Could not read the first frame of the video {video_path}")
    sys.exit(1)

cap.release()  # Close the video file

print(image.shape)

cv2.namedWindow("Sliders")
cv2.resizeWindow("Sliders", 640, 240)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_bound = np.array([63, 0, 0])
upper_bound = np.array([179, 255, 255])

def on_trackbar(*args):
    hue_min = cv2.getTrackbarPos("Hue Min", "Sliders")
    hue_max = cv2.getTrackbarPos("Hue Max", "Sliders")
    sat_min = cv2.getTrackbarPos("Sat Min", "Sliders")
    sat_max = cv2.getTrackbarPos("Sat Max", "Sliders")
    val_min = cv2.getTrackbarPos("Val Min", "Sliders")
    val_max = cv2.getTrackbarPos("Val Max", "Sliders")

    print(f"hue: ({hue_min}, {hue_max}) | sat: ({sat_min}, {sat_max}) | val: ({val_min}, {val_max})")

    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])

    imgMASK = cv2.inRange(hsv, lower, upper)
    notMask = cv2.bitwise_not(imgMASK)

    segmented_img = cv2.bitwise_and(image, image, mask=notMask)

    cv2.imshow("Mask", imgMASK)
    cv2.imshow("Masked", segmented_img)

cv2.createTrackbar("Hue Min", "Sliders", lower_bound[0], 179, on_trackbar)
cv2.createTrackbar("Hue Max", "Sliders", upper_bound[0], 179, on_trackbar)
cv2.createTrackbar("Sat Min", "Sliders", lower_bound[1], 255, on_trackbar)
cv2.createTrackbar("Sat Max", "Sliders", upper_bound[1], 255, on_trackbar)
cv2.createTrackbar("Val Min", "Sliders", lower_bound[2], 255, on_trackbar)
cv2.createTrackbar("Val Max", "Sliders", upper_bound[2], 255, on_trackbar)

on_trackbar(0)

cv2.waitKey(0)
cv2.destroyAllWindows()
