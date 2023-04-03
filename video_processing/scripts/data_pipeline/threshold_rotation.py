import cv2
import numpy as np
import sys
import os
import csv
import math
import re
import argparse

# Functions from threshold_sliders_confirm.py
def reset_to_defaults():
    cv2.setTrackbarPos("Hue Min", "Reset to Defaults (Press R)", default_lower_bound[0])
    cv2.setTrackbarPos("Hue Max", "Reset to Defaults (Press R)", default_upper_bound[0])
    cv2.setTrackbarPos("Sat Min", "Reset to Defaults (Press R)", default_lower_bound[1])
    cv2.setTrackbarPos("Sat Max", "Reset to Defaults (Press R)", default_upper_bound[1])
    cv2.setTrackbarPos("Val Min", "Reset to Defaults (Press R)", default_lower_bound[2])
    cv2.setTrackbarPos("Val Max", "Reset to Defaults (Press R)", default_upper_bound[2])

def on_trackbar(*args):
    global frame
    
    frame_pos = cv2.getTrackbarPos("Frame", "Reset to Defaults (Press R)")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
    ret, frame = cap.read()
    if not ret:
        return
    
    update_segmented_image()

def update_segmented_image():
    global frame
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    hue_min = cv2.getTrackbarPos("Hue Min", "Reset to Defaults (Press R)")
    hue_max = cv2.getTrackbarPos("Hue Max", "Reset to Defaults (Press R)")
    sat_min = cv2.getTrackbarPos("Sat Min", "Reset to Defaults (Press R)")
    sat_max = cv2.getTrackbarPos("Sat Max", "Reset to Defaults (Press R)")
    val_min = cv2.getTrackbarPos("Val Min", "Reset to Defaults (Press R)")
    val_max = cv2.getTrackbarPos("Val Max", "Reset to Defaults (Press R)")

    # print(f"hue: ({hue_min}, {hue_max}) | sat: ({sat_min}, {sat_max}) | val: ({val_min}, {val_max})")

    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])

    imgMASK = cv2.inRange(hsv, lower, upper)
    notMask = cv2.bitwise_not(imgMASK)

    segmented_img = cv2.bitwise_and(frame, frame, mask=notMask)

    imgMASK_color = cv2.cvtColor(imgMASK, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((imgMASK_color, segmented_img))
    cv2.imshow("Reset to Defaults (Press R)", combined)
    
def reset_key(*args):
    global reset
    reset = True
    
def write_metadata_to_csv(video_id, video_path, angle, hue_min, hue_max, sat_min, sat_max, val_min, val_max):
    if not os.path.exists("video_metadata.csv"):
        with open("video_metadata.csv", "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile, delimiter="|")
            csv_writer.writerow(["ID", "Name", "Correction Angle (deg)", "hue_min", "hue_max", "sat_min", "sat_max", "val_min", "val_max"])

    with open("video_metadata.csv", "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter="|")
        csv_writer.writerow([video_id, video_path, angle, hue_min, hue_max, sat_min, sat_max, val_min, val_max])
        # print the data written to csv
        print(f"ID: {video_id} | Name: {video_path} | Correction Angle (deg): {angle} | hue_min: {hue_min} | hue_max: {hue_max} | sat_min: {sat_min} | sat_max: {sat_max} | val_min: {val_min} | val_max: {val_max}")

# FUNCTIONS FROM rotation_correction.py
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
                
def rotate_frame(frame, angle):
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(frame, M, (w, h))
    return rotated



def update_first_frame_display(x, y):
    global points, first_frame, first_frame_display

    first_frame_display = first_frame.copy()

    for i, point in enumerate(points):
        cv2.circle(first_frame_display, point, 5, (0, 0, 255), -1)
        cv2.putText(first_frame_display, f"Point {i + 1}", (point[0] + 5, point[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computer vision analysis for droplets")
    parser.add_argument("input_vid", help="Path to the input video file")
    args = parser.parse_args()

    # Load the video
    video_path = args.input_vid
    video_id = re.findall(r'\d+', os.path.splitext(os.path.basename(video_path))[0])[-1]
    # print(f"Video ID: {video_id}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        sys.exit()
        
    # Output video
    fourcc = cv2.VideoWriter_fourcc("D", "I", "V", "X")
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_vid = f"rotated_videos/{base_name}_rotated.avi"
    print(f"Output video: {output_vid}")
    out = cv2.VideoWriter(output_vid, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))
    
    reset = False
    
    default_lower_bound = np.array([63, 0, 0])
    default_upper_bound = np.array([179, 255, 255])
    
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Rotation correction
        first_frame = frame.copy()
        first_frame_display = first_frame.copy()
        points = []
        selecting = True
        
        cv2.namedWindow("Select Points")
        cv2.setMouseCallback("Select Points", select_points)
        
        while selecting:
            cv2.imshow("Select Points", first_frame_display)
            cv2.waitKey(1)
        
        cv2.destroyAllWindows()
        
        # Calculate the angle
        x_diff = points[1][0] - points[0][0]
        y_diff = points[1][1] - points[0][1]
        angle = math.degrees(math.atan2(y_diff, x_diff))
        angle = round(angle, 2)
        print(f"Angle: {angle} degrees")
        
        # Rotate the frame
        rotated_frame = rotate_frame(frame, angle)

        # Convert the image to HSV
        hsv = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2HSV)

        # Define the default lower and upper bounds for the threshold sliders
        default_lower_bound = np.array([63, 0, 0])
        default_upper_bound = np.array([179, 255, 255])

        # Create the window for threshold sliders
        cv2.namedWindow("Reset to Defaults (Press R)")

        # Create trackbars
        cv2.createTrackbar("Hue Min", "Reset to Defaults (Press R)", default_lower_bound[0], 179, on_trackbar)
        cv2.createTrackbar("Hue Max", "Reset to Defaults (Press R)", default_upper_bound[0], 179, on_trackbar)
        cv2.createTrackbar("Sat Min", "Reset to Defaults (Press R)", default_lower_bound[1], 255, on_trackbar)
        cv2.createTrackbar("Sat Max", "Reset to Defaults (Press R)", default_upper_bound[1], 255, on_trackbar)
        cv2.createTrackbar("Val Min", "Reset to Defaults (Press R)", default_lower_bound[2], 255, on_trackbar)
        cv2.createTrackbar("Val Max", "Reset to Defaults (Press R)", default_upper_bound[2], 255, on_trackbar)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cv2.createTrackbar("Frame", "Reset to Defaults (Press R)", 0, total_frames, on_trackbar)

        # Run the threshold sliders loop
        while True:
            key = cv2.waitKey(0)
            # key = cv2.waitKey(1)
            if key == ord("q"):
                break
            if key == ord("r"):
                reset_key()
                reset_to_defaults()
            elif key == 13: # Enter key
                hue_min = cv2.getTrackbarPos("Hue Min", "Reset to Defaults (Press R)")
                hue_max = cv2.getTrackbarPos("Hue Max", "Reset to Defaults (Press R)")
                sat_min = cv2.getTrackbarPos("Sat Min", "Reset to Defaults (Press R)")
                sat_max = cv2.getTrackbarPos("Sat Max", "Reset to Defaults (Press R)")
                val_min = cv2.getTrackbarPos("Val Min", "Reset to Defaults (Press R)")
                val_max = cv2.getTrackbarPos("Val Max", "Reset to Defaults (Press R)")

                print("\033[32m")  # Set the text color to green
                print(f"Selected values: hue: ({hue_min}, {hue_max}) | sat: ({sat_min}, {sat_max}) | val: ({val_min}, {val_max})")
                print("\033[0m")  # Reset the text color
                
                # Write the selected values to the video_metadata.csv file
                write_metadata_to_csv(video_id, video_path, angle, hue_min, hue_max, sat_min, sat_max, val_min, val_max)
                
                break
            elif key == 27: # Escape key
                break
            else:
                on_trackbar()
            
            if reset:
                reset_to_defaults()
                reset = False

        on_trackbar()

        # Rotation correction
        first_frame = frame.copy()
        first_frame_display = first_frame.copy()
        points = []
        selecting = True

        cv2.namedWindow("Select Points")
        cv2.setMouseCallback("Select Points", select_points)

        while selecting:
            cv2.imshow("Select Points", first_frame_display)
            cv2.waitKey(1)

        cv2.destroyAllWindows()

        # Calculate the angle
        x_diff = points[1][0] - points[0][0]
        y_diff = points[1][1] - points[0][1]
        angle = math.degrees(math.atan2(y_diff, x_diff))
        print(f"Angle: {angle}")

        # Rotate the image
        height, width = frame.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(frame, rotation_matrix, (width, height))

        # Show the rotated image
        cv2.imshow("Rotated Image", rotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    cap.release()
