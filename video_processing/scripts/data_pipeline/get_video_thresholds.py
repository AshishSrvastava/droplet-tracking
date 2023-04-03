import cv2
import numpy as np
import sys
import os
import csv
import argparse

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
    global hsv, image
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threshold Sliders for Color Segmentation")
    parser.add_argument("input_vid", help="Path to the input video file")
    args = parser.parse_args()
    
    video_path = args.input_vid
    get_video_thresholds(video_path)