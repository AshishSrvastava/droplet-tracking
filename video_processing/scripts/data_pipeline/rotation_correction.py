import sys
import csv
import cv2
import numpy as np
import math
import re
import os

# Create folders if they do not exist
if not os.path.exists("rotated_videos"):
    os.makedirs("rotated_videos")
if not os.path.exists("position_data"):
    os.makedirs("position_data")

# Mouse event callback function
def select_points(event, x, y, flags, param):
    global points, selecting, first_frame_display

    if event == cv2.EVENT_MOUSEMOVE:
        update_first_frame_display(x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")
            update_first_frame_display(x, y)
            if len(points) == 2:
                selecting = False

# Update first frame display with points and green dot under cursor
def update_first_frame_display(x, y):
    global points, first_frame, first_frame_display

    first_frame_display = first_frame.copy()

    for i, point in enumerate(points):
        cv2.circle(first_frame_display, point, 5, (0, 0, 255), -1)
        cv2.putText(first_frame_display, f"Point {i + 1}", (point[0] + 5, point[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if selecting:
        cv2.circle(first_frame_display, (x, y), 5, (0, 255, 0), -1)

# Input
input_vid = sys.argv[1]
cap = cv2.VideoCapture(input_vid)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 1920
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 1080

# Video name handling
base_name_pattern = re.compile(r"^(.+?)_(\d{8}_\d{3}\.avi)$")
match = base_name_pattern.search(input_vid)
if match:
    base_name = match.group(1)
    identifier_suffix = match.group(2)
    print(f"base name: {base_name}")
    print(f"Suffix: {identifier_suffix}")
else:
    print("Invalid input video filename")
    sys.exit()

# Get first frame for error correction
ret, first_frame = cap.read()
first_frame_display = first_frame.copy()

# Check if angle is already in the input video filename
angle_pattern = re.compile(r"[-+]?\d+_\d+(?=_angle)")  # Updated regular expression
angle_match = angle_pattern.search(input_vid)

if angle_match:
    angle_str = angle_match.group().replace("_", ".")
    correction_angle = float(angle_str)
    print(f"Correction angle from filename: {correction_angle} radians")
else:
    # Show first frame for user to select points
    cv2.namedWindow("First Frame")
    cv2.setMouseCallback("First Frame", select_points)

    points = []
    selecting = True

    while selecting:
        cv2.imshow("First Frame", first_frame_display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            exit(0)

    cv2.destroyAllWindows()

    # Fit a line to the two points and calculate the angle
    x1, y1 = points[0]
    x2, y2 = points[1]
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    angle = round(angle, 2)
    correction_angle = -angle
    print(f"Correction angle: {correction_angle} degrees")

# Helper function to rotate a frame
def rotate_frame(frame, angle):
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(frame, M, (w, h))
    return rotated

# Modify output filenames to include the correction angle
angle_str = f"{correction_angle:.2f}".replace(".", "_") + "deg"
print(f"Angle string: {angle_str}")
print(f"Base name: {base_name} \n angle_str: {angle_str}")
output_vid = f"rotated_videos/{base_name}_{angle_str}_{identifier_suffix}"
print(f"Output video: {output_vid}")


position_data_file = f"position_data/droplet_posn_time_{angle_str}_angle.txt"

# Save the correction angle in a separate text file
with open("correction_angles.txt", "a") as file:
    file.write(f"{input_vid}\t{correction_angle}\n")

# Output video
fourcc = cv2.VideoWriter_fourcc("D", "I", "V", "X")
out = cv2.VideoWriter(output_vid, fourcc, 20.0, (1920, 1080))

# Output data for graphing
with open(position_data_file, "w") as file:
    file.write("frame \t x \t y \t width \t height \t deformation \n")

# Process the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Rotate frame using the correction angle
    frame = rotate_frame(frame, correction_angle)

    # Get current frame number
    frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    # Convert frame number to string and add leading zeros
    # Put frame number in top left corner of frame
    cv2.rectangle(frame, (0, 0), (100, 30), (0, 0, 0), -1)
    cv2.putText(frame, str(frame_num), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Put correction angle in the top right corner of frame
    angle_text = f"Angle: {correction_angle:.2f} deg"
    cv2.rectangle(frame, (int(width) - 240, 0), (int(width), 30), (0, 0, 0), -1)
    cv2.putText(frame, angle_text, (int(width) - 230, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


    # Convert to grayscale and apply Gaussian blur
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Convert to HSV
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    # Define range of color in HSV
    lower_blue = np.array([63, 0, 0])
    upper_blue = np.array([179, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours 
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # all contours
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)

    # contour filtering
    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter out contours that are too small
        if area < 1600:
            continue

    # write the contoured frame
    out.write(frame)

    # concatenate all frames into a single image for display
    # convert the 2d contour image to 3d image by copying same image to 3rd channel
    mask_3ch = cv2.merge((mask, mask, mask))
    # video_image = np.concatenate((blurred_frame, mask_3ch, frame), axis=0)
    video_image = frame
    # show the image
    cv2.imshow("Video, mask and contour", video_image)
    cv2.waitKey(1)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()