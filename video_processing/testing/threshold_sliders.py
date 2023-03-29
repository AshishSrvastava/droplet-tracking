import cv2
import numpy as np
import sys

def reset_to_defaults():
    cv2.setTrackbarPos("Hue Min", "Control Panel", lower_bound[0])
    cv2.setTrackbarPos("Hue Max", "Control Panel", upper_bound[0])
    cv2.setTrackbarPos("Sat Min", "Control Panel", lower_bound[1])
    cv2.setTrackbarPos("Sat Max", "Control Panel", upper_bound[1])
    cv2.setTrackbarPos("Val Min", "Control Panel", lower_bound[2])
    cv2.setTrackbarPos("Val Max", "Control Panel", upper_bound[2])

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

# print(image.shape) # for debugging

cv2.namedWindow("Main Window", cv2.WINDOW_NORMAL)
cv2.namedWindow("Control Panel", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Main Window", 1280, 480)
cv2.resizeWindow("Control Panel", 300, 480)

cv2.moveWindow("Main Window", 0, 0)
cv2.moveWindow("Control Panel", 1280, 0)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_bound = np.array([63, 0, 0])
upper_bound = np.array([179, 255, 255])

def on_trackbar(*args):
    hue_min = cv2.getTrackbarPos("Hue Min", "Control Panel")
    hue_max = cv2.getTrackbarPos("Hue Max", "Control Panel")
    sat_min = cv2.getTrackbarPos("Sat Min", "Control Panel")
    sat_max = cv2.getTrackbarPos("Sat Max", "Control Panel")
    val_min = cv2.getTrackbarPos("Val Min", "Control Panel")
    val_max = cv2.getTrackbarPos("Val Max", "Control Panel")

    print(f"hue: ({hue_min}, {hue_max}) | sat: ({sat_min}, {sat_max}) | val: ({val_min}, {val_max})")

    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])

    imgMASK = cv2.inRange(hsv, lower, upper)
    notMask = cv2.bitwise_not(imgMASK)

    segmented_img = cv2.bitwise_and(image, image, mask=notMask)

    imgMASK_color = cv2.cvtColor(imgMASK, cv2.COLOR_GRAY2BGR)
    grid = np.vstack((image, imgMASK_color, segmented_img))

    # Create an empty black image with the same height as the grid and the desired width for the sliders
    slider_space = np.zeros((grid.shape[0], 300, 3), dtype=np.uint8)

    # Concatenate the empty black image with the grid
    grid_with_sliders = np.hstack((grid, slider_space))

    cv2.imshow("Main Window", grid_with_sliders)


cv2.createTrackbar("Hue Min", "Control Panel", lower_bound[0], 179, on_trackbar)
cv2.createTrackbar("Hue Max", "Control Panel", upper_bound[0], 179, on_trackbar)
cv2.createTrackbar("Sat Min", "Control Panel", lower_bound[1], 255, on_trackbar)
cv2.createTrackbar("Sat Max", "Control Panel", upper_bound[1], 255, on_trackbar)
cv2.createTrackbar("Val Min", "Control Panel", lower_bound[2], 255, on_trackbar)
cv2.createTrackbar("Val Max", "Control Panel", upper_bound[2], 255, on_trackbar)

on_trackbar(0)

while True:
    key = cv2.waitKey(0)
    if key == ord('r'):
        reset_to_defaults()
    elif key == 27:  # Escape key
        break

cv2.destroyAllWindows()

