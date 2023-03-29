import cv2
import numpy as np
import sys

def reset_to_defaults():
    cv2.setTrackbarPos("Hue Min", "Controls", lower_bound[0])
    cv2.setTrackbarPos("Hue Max", "Controls", upper_bound[0])
    cv2.setTrackbarPos("Sat Min", "Controls", lower_bound[1])
    cv2.setTrackbarPos("Sat Max", "Controls", upper_bound[1])
    cv2.setTrackbarPos("Val Min", "Controls", lower_bound[2])
    cv2.setTrackbarPos("Val Max", "Controls", upper_bound[2])

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

cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Result", 640, 480)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_bound = np.array([63, 0, 0])
upper_bound = np.array([179, 255, 255])

def on_trackbar(*args):
    hue_min = cv2.getTrackbarPos("Hue Min", "Result")
    hue_max = cv2.getTrackbarPos("Hue Max", "Result")
    sat_min = cv2.getTrackbarPos("Sat Min", "Result")
    sat_max = cv2.getTrackbarPos("Sat Max", "Result")
    val_min = cv2.getTrackbarPos("Val Min", "Result")
    val_max = cv2.getTrackbarPos("Val Max", "Result")

    print(f"hue: ({hue_min}, {hue_max}) | sat: ({sat_min}, {sat_max}) | val: ({val_min}, {val_max})")

    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])

    imgMASK = cv2.inRange(hsv, lower, upper)
    notMask = cv2.bitwise_not(imgMASK)

    segmented_img = cv2.bitwise_and(image, image, mask=notMask)

    imgMASK_color = cv2.cvtColor(imgMASK, cv2.COLOR_GRAY2BGR)
    sliders = np.zeros((480, 320, 3), dtype=np.uint8)
    
    # Display images in a 2x2 grid
    img1 = image
    img2 = imgMASK_color
    img3 = segmented_img
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    h3, w3 = img3.shape[:2]
    h4, w4 = h1, int(w2/2)
    sliders_resized = cv2.resize(sliders, (w4, h4))
    
    max_h = max(h1, h2, h3, h4)
    total_w = w1 + w2 + w3 + w4
    
    new_img = np.zeros((max_h*2, total_w, 3), dtype=np.uint8)
    
    new_img[0:h1, 0:w1] = img1
    new_img[0:h2, w1:w1+w2] = img2
    new_img[0:h3, w1+w2:w1+w2+w3] = img3
    new_img[h1:max_h*2, w1+w2:w1+w2+w4] = sliders_resized
    
    cv2.imshow("Result", new_img)

cv2.createTrackbar("Hue Min", "Reset to Defaults (Press R)", lower_bound[0], 179, on_trackbar)
cv2.createTrackbar("Hue Max", "Reset to Defaults (Press R)", upper_bound[0], 179, on_trackbar)
cv2.createTrackbar("Sat Min", "Reset to Defaults (Press R)", lower_bound[1], 255, on_trackbar)
cv2.createTrackbar("Sat Max", "Reset to Defaults (Press R)", upper_bound[1], 255, on_trackbar)
cv2.createTrackbar("Val Min", "Reset to Defaults (Press R)", lower_bound[2], 255, on_trackbar)
cv2.createTrackbar("Val Max", "Reset to Defaults (Press R)", upper_bound[2], 255, on_trackbar)

on_trackbar(0)

while True:
    key = cv2.waitKey(0)
    if key == ord('r'):
        reset_to_defaults()
    elif key == 27:  # Escape key
        break

cv2.destroyAllWindows()
