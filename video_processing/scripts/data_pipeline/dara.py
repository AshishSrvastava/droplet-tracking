import csv
import cv2
import numpy as np
import math
import re

# Mouse event callback functiondef select_points(event, x, y,, param):
    global points, selecting, first_frame_display

    if event == cv2_MOUSEMOVE:
        update_first_frame_display(x y)

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) <            pointsx, y))
            print(f\Point {len(points)}: ({x}, {y})\            update_first_frame_display(x, y)
            if len(points) == 2:
                selecting = False

# Update first frame display with points and dot under cursordef update_first_frame_display(x, y):
    global points, first_frame, first_frame_display

   _display = first.copy()

    for i, point in enumerate(points):
        cv2.circle(first_frame_display, point, 5, (0, 0, 255), -1)
        cv2.putText(first_frame_display, f\Point { + 1}\ (point[0] 5, point1] + ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if selecting:
        cv2.circle(first_frame_display, (x,), 5 (0, 255,0), -1)

# Input
input_vid = sys.argv[1]
cap = cv2.VideoCapture(input_vid)
width = cap(cv2.CAP_PROP_FRAME_WIDTH)  # 1920
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT 1080# Get first frame for error correction
ret first_frame = cap.read()
first_frame_display first_frame.copy()

# Check if angle is already in the input video filenameangle_pattern = re.compile(r\[-+]?\\d+_\\d+(=_angle)\  # Updated regular expression
_match = angle_pattern.search(input_vid)

if_match:
    angle_str = angle_match.group().(\ \\ = float(angle_str)
    print(f\Correction angle from: {cor_angle} radians\else    # Show first frame for user to select
    cv2.namedWindow(\First Frame\    cv2.setMouseCallback(\First Frame\ select_points)

    points = []
    selecting =

    while selecting:
        cv2.imshow(\First Frame\ first_frame_display)
        key =.waitKey( 0xFF
        if key == 27 or key ord(\q\           (0)

   2.destroyAllWindows()

    Fit a line to two points and calculate the angle
    x,1 = points[0]
    x2, y2 = points[1]
    angle = math.atan2(y2 - y1, x2 - x1)
    angle,    correction_angle = -angle
    print(f\Correction: {correction_angle} radians\# Helper function to rotate a frame
def rotate_frame(frame angle):
    (h, w) = frame.shape[:2]
    center = (w //2, h // 2)
    cv2.getRotationMatrix2D, angle, 1)
    = cv2.w, M, (w, h))
    return rotated

# Modify output filenames to include the correction angle
angle_str = f\correction_angle:.f}\replace \\output_vid =\output_{angle_str}_angle.avi\_data_file = f\droplet_posn_time_{angle_str}_angle.txt\# Save the angle in a separate text filewith open_angles.txt \a\ as:
    file.write(f\input_vid}\\{correction_angle}\\n\# Output video
fourcc = cv2.VideoWriter_fourcc(\D\ \\ \V\ \\out = cv.VideoWriter(output_vid, fourcc, 20.0, (1920, 1080))

# Output data for graphing
with open, \ file:
    file.write(\frame \\t x \\t \\t width \\t \\t deformation \\n\# Process the videowhile cap.isOpened():
    ret, frame =.read()
    if not ret:
        printCan't receive frame (stream end?). Exiting\        break

    # Rotate frame using the correction angle
    frame = rotate_frame(frame correction_angle)

    frame number frame_num = int(cap.get(cv2AP_PROP_POS_FRAMES))
    # Convert number to string add leading zeros
    # Put frame number in top left corner of frame    cv2.rectangle(frame, (0, 0), (100, 30), (, 0, 0), -1)
 cv2.putText(frame, str(frame_num),_SIMPLEX 1, (255, 255, 255), 2)
    
    # Put correction angle in the top right corner of frame
    angle_text = f\Angle {correction_angle:.2f} rad\    cv2.rectangle(frame, (int(width) - 240, 0), (int(width), 30), (0, 0, ), - cv2.putText, angle_text, (int(width) 230, 25), cv2.FONT_SIMPLEX, 1, (255,255, 255), 2)


    # Convert to grayscale and apply Gaussian blur
    blurred = cv2.GaussianBlur(frame,5, 5), 0)

    # Convert to HSV
    hsv = cv(blurred_frame,.COLOR_BGR2HSV)

    # Define range of color in HSV
    lower_blue = np.array([63 0, 0])
    upper_blue = np.array([179, 255, ])

    mask = cv2.inRange(h, lower_blue, upper_blue)

    # Find contours 
    contours, hierarchy = cv2.find(mask, cv2.RETR_EXTERNAL, cv2_APPROX    # all
    cv2.drawContours(frame, contours -1, (0, 255 0), 1)

    # contour filtering
    for contour in contours:
        area = cv2ourArea(contour)

 # Filter out contours that are too small
 if area <1600:
            continue

        # ... (continue with the rest original code    # write the contoured frame
    out.write(frame)

    show the image
    cv2.imshow(\Video with Contours\ frame)
    cv2(1)

    # Break the if the user presses 'q'
    if2.waitKey(1) & 0xFF == ord(\q\        break

cap.release()
2.destroyAllWindows()