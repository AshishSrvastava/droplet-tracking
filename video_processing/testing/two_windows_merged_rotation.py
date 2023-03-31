import sys
import cv2
import numpy as np
import math
import re
import time
import os

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
cv2.resizeWindow("Control Panel", 400, 600)

cv2.moveWindow("Main Window", 0, 0)
cv2.moveWindow("Control Panel", 1000, 0)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_bound = np.array([63, 0, 0])
upper_bound = np.array([179, 255, 255])


selecting = False

class CheckButton:
    def __init__(self, text, x, y, callback=None, checked=True):
        self.text = text
        self.x = x
        self.y = y
        self.callback = callback
        self.checked = checked

    def draw(self, img):
        cv2.rectangle(img, (self.x, self.y), (self.x + 20, self.y + 20), (255, 255, 255), 2)
        if not self.checked:
            cv2.rectangle(img, (self.x + 2, self.y + 2), (self.x + 18, self.y + 18), (255, 255, 255), -1)
        cv2.putText(img, self.text, (self.x + 30, self.y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def check_click(self, x, y):
        if self.x < x < self.x + 20 and self.y < y < self.y + 20:
            self.checked = not self.checked
            if self.callback:
                self.callback()
            return True
        return False
    
class PressButton:
    def __init__(self, text, x, y, callback=None, checked=False):
        self.text = text
        self.x = x
        self.y = y
        self.callback = callback
        self.checked = checked

    def draw(self, img):
        cv2.rectangle(img, (self.x, self.y), (self.x + 20, self.y + 20), (255, 255, 255), 2)
        if self.checked:
            cv2.rectangle(img, (self.x + 2, self.y + 2), (self.x + 18, self.y + 18), (0, 0, 0), -1)
        cv2.putText(img, self.text, (self.x + 30, self.y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def check_click(self, x, y):
        if self.x < x < self.x + 20 and self.y < y < self.y + 20:
            self.checked = True  # Always set checked to True when clicked
            if self.callback:
                self.callback()
            return True
        return False


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for button in check_buttons:
            if button.check_click(x, y):
                draw_control_panel()

def draw_control_panel():
    cp_img = np.zeros((480, 300, 3), dtype=np.uint8)
    for button in check_buttons:
        button.draw(cp_img)
    cv2.imshow("Control Panel", cp_img)

# Add the following lines after creating the Control Panel window
cv2.setMouseCallback("Control Panel", on_mouse)

def on_check_button(*args):
    update_main_window()

def update_main_window_with_points(x, y):
    global points, selecting

    display_img = image.copy()

    for i, point in enumerate(points):
        cv2.circle(display_img, point, 5, (0, 0, 255), -1)
        cv2.putText(display_img, f"Point {i + 1}", (point[0] + 5, point[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if selecting:
        cv2.circle(display_img, (x, y), 5, (0, 255, 0), -1)
        

    update_main_window(display_img)


def select_points(event, x, y, flags, param):
    global points, selecting, first_frame_display, image

    if event == cv2.EVENT_MOUSEMOVE:
        if selecting:
            update_main_window_with_points(x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        if selecting:
            if len(points) < 2:
                points.append((x, y))
                print(f"Point {len(points)}: ({x}, {y})")
                update_main_window_with_points(x, y)
                if len(points) == 2:
                    selecting = False

                    # Rotate the image based on the selected points
                    angle = math.degrees(math.atan2(points[1][1] - points[0][1], points[1][0] - points[0][0]))
                    rotated_img = rotate_frame(image, angle)

                    # Update the main window with the rotated image
                    update_main_window_with_points(None, None)
                    update_main_window(rotated_img)

# Create a new function called rotate_frame to perform the frame rotation
def rotate_frame(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated


def on_angle_correction_button(*args):
    global selecting, points
    selecting = True
    points = []
    print("On Angle Correction Button clicked. Select two points on the image.")


# Set the mouse callback for the "Select Points" window
cv2.setMouseCallback("Main Window", select_points)

# Initialize the points and first_frame_display variables
points = []
first_frame_display = image.copy()



check_buttons = [
    CheckButton("Show Image", 10, 10, on_check_button, checked=True),
    CheckButton("Show Mask", 10, 40, on_check_button, checked=True),
    CheckButton("Show Segmented", 10, 70, on_check_button, checked=True),
    PressButton("Angle Correction", 10, 110, on_angle_correction_button, checked=False),
]

# Update first frame display with points and green dot under cursor
def update_main_window(x=None, y=None):
    global points, image, imgMASK_color, segmented_img

    display_list = []

    if check_buttons[0].checked:
        display_list.append(image)
    if check_buttons[1].checked:
        display_list.append(imgMASK_color)
    if check_buttons[2].checked:
        display_list.append(segmented_img)

    if display_list:
        fixed_width = 720
        aspect_ratio = float(image.shape[1]) / float(image.shape[0])
        grid_height = int(fixed_width / aspect_ratio)
        grid_width = fixed_width
        grid = np.vstack(display_list)
    else:
        grid_height = 1
        grid_width = 1
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    if x is not None and y is not None:
        for i, point in enumerate(points):
            cv2.circle(grid, point, 5, (0, 0, 255), -1)
            cv2.putText(grid, f"Point {i + 1}", (point[0] + 5, point[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if selecting:
            cv2.circle(grid, (x, y), 5, (0, 255, 0), -1)

    cv2.resizeWindow("Main Window", grid_width, grid.shape[1])
    cv2.imshow("Main Window", grid)
    cv2.setMouseCallback("Main Window", select_points)


def on_trackbar(*args):
    hue_min = cv2.getTrackbarPos("Hue Min", "Control Panel")
    hue_max = cv2.getTrackbarPos("Hue Max", "Control Panel")
    sat_min = cv2.getTrackbarPos("Sat Min", "Control Panel")
    sat_max = cv2.getTrackbarPos("Sat Max", "Control Panel")
    val_min = cv2.getTrackbarPos("Val Min", "Control Panel")
    val_max = cv2.getTrackbarPos("Val Max", "Control Panel")

    # print(f"hue: ({hue_min}, {hue_max}) | sat: ({sat_min}, {sat_max}) | val: ({val_min}, {val_max})")

    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])

    imgMASK = cv2.inRange(hsv, lower, upper)
    notMask = cv2.bitwise_not(imgMASK)
    
    global segmented_img, imgMASK_color

    segmented_img = cv2.bitwise_and(image, image, mask=notMask)
    imgMASK_color = cv2.cvtColor(imgMASK, cv2.COLOR_GRAY2BGR)
    
    update_main_window()

# Control panel interactive options w/ default values

cv2.createTrackbar("Hue Min", "Control Panel", lower_bound[0], 179, on_trackbar)
cv2.createTrackbar("Hue Max", "Control Panel", upper_bound[0], 179, on_trackbar)
cv2.createTrackbar("Sat Min", "Control Panel", lower_bound[1], 255, on_trackbar)
cv2.createTrackbar("Sat Max", "Control Panel", upper_bound[1], 255, on_trackbar)
cv2.createTrackbar("Val Min", "Control Panel", lower_bound[2], 255, on_trackbar)
cv2.createTrackbar("Val Max", "Control Panel", upper_bound[2], 255, on_trackbar)

# Create control panel
draw_control_panel()

on_trackbar(0)

while True:
    key = cv2.waitKey(0)
    if key == ord('r'):
        reset_to_defaults()
    elif key == 27:  # Escape key
        break

cv2.destroyAllWindows()
