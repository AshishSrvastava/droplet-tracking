import cv2
import numpy as np
import sys
import os
import csv
import argparse
import math

# functions from rotation_correction.py

def create_folders():
    """Create folders if they do not exist"""
    if not os.path.exists("rotated_videos"):
        os.makedirs("rotated_videos")
    if not os.path.exists("position_data"):
        os.makedirs("position_data")


def select_points(event, x, y, flags, param):
    """Mouse event callback function"""
    points, selecting, first_frame_display = param  # retrieve points from param

    if event == cv2.EVENT_MOUSEMOVE:
        update_first_frame_display(x, y, points, first_frame_display, selecting)

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 2:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")
            if len(points) == 2:
                selecting = False
                print(f"Selecting is done. Confirmed points: {points}")
            update_first_frame_display(x, y, points, first_frame_display, selecting)

    cv2.imshow("First Frame", first_frame_display)
    cv2.waitKey(1)
    return points




def update_first_frame_display(x, y, points, first_frame_display, selecting):
    """Update first frame display with points and green dot under cursor"""
    first_frame_display_copy = first_frame_display.copy()
    
    for i, point in enumerate(points):
        cv2.circle(first_frame_display_copy, point, 5, (0, 0, 255), -1)
        cv2.putText(first_frame_display_copy, f"Point {i + 1}", (point[0] + 5, point[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if selecting:
        cv2.circle(first_frame_display_copy, (x, y), 5, (0, 255, 0), -1)
    else:
        for i, point in enumerate(points):
            cv2.circle(first_frame_display_copy, point, 5, (0, 0, 255), -1)
            cv2.putText(first_frame_display_copy, f"Point {i + 1}", (point[0] + 5, point[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.circle(first_frame_display_copy, points[0], 5, (0, 0, 255), -1)
        cv2.putText(first_frame_display_copy, "Selected", (points[0][0] + 5, points[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.circle(first_frame_display_copy, points[1], 5, (0, 0, 255), -1)
        cv2.putText(first_frame_display_copy, "Selected", (points[1][0] + 5, points[1][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


def get_correction_angle(points):
    """Get the correction angle from the selected points"""
    # print(f"Points passed into get_correction_angle: {points}") # debug
    # print(f"get_correction_angle called. Points: {points}") # debug
    x1, y1 = points[0]
    x2, y2 = points[1]
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    angle = round(angle, 2)

    print(f"Correction angle: {angle} degrees")

    return angle


# Helper function to rotate a frame
def rotate_frame(frame, angle):
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(frame, M, (w, h))
    return rotated


def rotate_video(input_video_path):
    """Rotate a video and save the rotated video to the 'rotated_videos' folder"""
    points = []
    selecting = True

    # Load the video
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error: Could not open {input_video_path}")
        return

    # Get the first frame
    ret, first_frame = cap.read()

    # Make a copy of the first frame to display
    first_frame_display = first_frame.copy()

    # Show first frame for user to select points
    cv2.namedWindow("First Frame")
    cv2.setMouseCallback("First Frame", select_points, param=(points, selecting, first_frame_display)) # pass points to select_points

    while len(points) < 2:
        cv2.imshow("First Frame", first_frame_display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            exit(0)
    
    # Get the correction angle
    correction_angle = get_correction_angle(points)

    # Output video
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    angle_str = f"{correction_angle:.2f}".replace(".", "_") + "deg"
    # output_video_path = f"rotated_videos/{base_name}_{angle_str}_rotated.avi"
    output_video_path = f"rotated_videos/{base_name}_rotated.avi"
    print(f"Output video: {output_video_path}")
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_num in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Rotate the frame using the angle
        rotated_frame = rotate_frame(frame, correction_angle)

        # Write the rotated frame to the output video
        out.write(rotated_frame)

        # Calculate progress and display a progress bar     
        progress = int((frame_num / num_frames) * 100)  # Add this line to define progress
        bar_width = 50
        filled_width = int(bar_width * progress / 100)
        remaining_width = bar_width - filled_width
        print(f"[{'=' * filled_width}{' ' * remaining_width}] {progress}%\r", end="")


    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return correction_angle


# Functions from get_video_thresholds.py

def reset_to_defaults():
    cv2.setTrackbarPos("Hue Min", "Reset to Defaults (Press R)", lower_bound[0])
    cv2.setTrackbarPos("Hue Max", "Reset to Defaults (Press R)", upper_bound[0])
    cv2.setTrackbarPos("Sat Min", "Reset to Defaults (Press R)", lower_bound[1])
    cv2.setTrackbarPos("Sat Max", "Reset to Defaults (Press R)", upper_bound[1])
    cv2.setTrackbarPos("Val Min", "Reset to Defaults (Press R)", lower_bound[2])
    cv2.setTrackbarPos("Val Max", "Reset to Defaults (Press R)", upper_bound[2])
    
def on_trackbar(*args):
    hue_min = cv2.getTrackbarPos("Hue Min", "Reset to Defaults (Press R)")
    hue_max = cv2.getTrackbarPos("Hue Max", "Reset to Defaults (Press R)")
    sat_min = cv2.getTrackbarPos("Sat Min", "Reset to Defaults (Press R)")
    sat_max = cv2.getTrackbarPos("Sat Max", "Reset to Defaults (Press R)")
    val_min = cv2.getTrackbarPos("Val Min", "Reset to Defaults (Press R)")
    val_max = cv2.getTrackbarPos("Val Max", "Reset to Defaults (Press R)")

    print(f"hue: ({hue_min}, {hue_max}) | sat: ({sat_min}, {sat_max}) | val: ({val_min}, {val_max})")

    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])

    imgMASK = cv2.inRange(hsv, lower, upper)
    notMask = cv2.bitwise_not(imgMASK)

    segmented_img = cv2.bitwise_and(image, image, mask=notMask)

    imgMASK_color = cv2.cvtColor(imgMASK, cv2.COLOR_GRAY2BGR)
    combined = np.hstack((imgMASK_color, segmented_img))
    cv2.imshow("Reset to Defaults (Press R)", combined)
    
def get_video_thresholds(video_path):
    global hsv, image, lower_bound, upper_bound
    
    # Extract video ID from the video_path

    video_path = video_path
    cap = cv2.VideoCapture(video_path)
    video_id = os.path.splitext(os.path.basename(video_path))[0].split('_')[-1]

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        sys.exit(1)

    ret, image = cap.read()
    if not ret:
        print(f"Error: Could not read the first frame of the video {video_path}")
        sys.exit(1)

    cap.release()

    # print(image.shape) # debug

    cv2.namedWindow("Reset to Defaults (Press R)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Reset to Defaults (Press R)", 640, 480)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([63, 0, 0])
    upper_bound = np.array([179, 255, 255])

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
        elif key == 13:  # Enter or Return key
            hue_min = cv2.getTrackbarPos("Hue Min", "Reset to Defaults (Press R)")
            hue_max = cv2.getTrackbarPos("Hue Max", "Reset to Defaults (Press R)")
            sat_min = cv2.getTrackbarPos("Sat Min", "Reset to Defaults (Press R)")
            sat_max = cv2.getTrackbarPos("Sat Max", "Reset to Defaults (Press R)")
            val_min = cv2.getTrackbarPos("Val Min", "Reset to Defaults (Press R)")
            val_max = cv2.getTrackbarPos("Val Max", "Reset to Defaults (Press R)")

            print("\033[32m")  # Set the text color to green
            print(f"Selected values: hue: ({hue_min}, {hue_max}) | sat: ({sat_min}, {sat_max}) | val: ({val_min}, {val_max})")
            print("\033[0m")  # Reset the text color
            
            # Check if the video_metadata.csv file exists
            if not os.path.exists("video_metadata.csv"):
                with open("video_metadata.csv", "w", newline="") as csvfile:
                    csv_writer = csv.writer(csvfile, delimiter="|")
                    csv_writer.writerow(["ID", "Name", "Correction Angle (deg)", "hue_min", "hue_max", "sat_min", "sat_max", "val_min", "val_max"])
            
            # Write the selected values to the video_metadata.csv file
            with open("video_metadata.csv", "a", newline="") as csvfile:
                csv_writer = csv.writer(csvfile, delimiter="|")
                csv_writer.writerow([video_id, video_path, "", hue_min, hue_max, sat_min, sat_max, val_min, val_max])
            
            break
        elif key == 27:  # Escape key
            break

    cv2.destroyAllWindows()
    return hue_min, hue_max, sat_min, sat_max, val_min, val_max
    
def video_correction_one_pipeline(input_video_path, mode):
    video_id = os.path.splitext(os.path.basename(input_video_path))[0].split('_')[-1] 
    print(f"Thresholding video {video_id}...")
    hue_min, hue_max, sat_min, sat_max, val_min, val_max = get_video_thresholds(input_video_path)
    print(f"Performing rotation correction on video {video_id}...")
    correction_angle = rotate_video(input_video_path) 
    
    
    # Check if the video_metadata.csv file exists
    if not os.path.exists("video_metadata.csv"):
        with open("video_metadata.csv", "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile, delimiter="|")
            csv_writer.writerow(["ID", "Name", "Correction Angle (deg)", "hue_min", "hue_max", "sat_min", "sat_max", "val_min", "val_max"])

    # Write the selected values to the video_metadata.csv file
    with open("video_metadata.csv", "a", newline="") as csvfile:
        csv_writer = csv.writer(csvfile, delimiter="|")
        csv_writer.writerow([video_id, input_video_path, correction_angle, hue_min, hue_max, sat_min, sat_max, val_min, val_max])

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Correction for droplets")
    parser.add_argument("input_vid", help="Path to the input video file")
    parser.add_argument("--mode", type=int, default=1, help="Choose the mode: 1 for get_video_thresholds, 2 for rotation_correction")
    args = parser.parse_args()


    video_path = args.input_vid
    mode = args.mode

    # video_correction(video_path, mode)
    video_correction_one_pipeline(video_path, mode)