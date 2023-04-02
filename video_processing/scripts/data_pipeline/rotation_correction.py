import sys
import csv
import cv2
import numpy as np
import math
import re
import os
import argparse

def create_folders():
    """Create folders if they do not exist"""
    if not os.path.exists("rotated_videos"):
        os.makedirs("rotated_videos")
    if not os.path.exists("position_data"):
        os.makedirs("position_data")


# def select_points(event, x, y, flags, param):
#     """Mouse event callback function"""
#     points, selecting, first_frame_display = param  # retrieve points from param

#     if event == cv2.EVENT_MOUSEMOVE:
#         update_first_frame_display(x, y, points, first_frame_display, selecting)

#     if event == cv2.EVENT_LBUTTONDOWN:
#         if len(points) < 2:
#             points.append((x, y))
#             print(f"Point {len(points)}: ({x}, {y})")
#             update_first_frame_display(x, y, points, first_frame_display, selecting)
#             if len(points) == 2:
#                 selecting = False
#                 # print(f"Selecting is done. Points: {points}")
#                 return points  # return points when selecting is done

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
                print(f"Selecting is done. Points: {points}")
            update_first_frame_display(x, y, points, first_frame_display, selecting)
            


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

    cv2.imshow("First Frame", first_frame_display_copy)
    
    # for i, point in enumerate(points):
    #     cv2.circle(first_frame_display_copy, point, 5, (0, 0, 255), -1)
    #     cv2.putText(first_frame_display_copy, f"Point {i + 1}", (point[0] + 5, point[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # if selecting:
    #     cv2.circle(first_frame_display_copy, (x, y), 5, (0, 255, 0), -1)

    # cv2.imshow("First Frame", first_frame_display_copy)

        
def get_correction_angle(input_vid, first_frame_display):
    """Get the correction angle from the input video filename"""
    # global points
    points = []
    selecting = True
    
    # Check if angle is already in the input video filename
    angle_pattern = re.compile(r"[-+]?\d+_\d+(?=_angle)")  # Updated regular expression
    angle_match = angle_pattern.search(input_vid)

    if angle_match:
        angle_str = angle_match.group().replace("_", ".")
        correction_angle = float(angle_str)
        print(f"Correction angle from filename: {correction_angle} degrees")
    else:
        # Show first frame for user to select points
        cv2.namedWindow("First Frame")
        # cv2.setMouseCallback("First Frame", select_points)
        cv2.setMouseCallback("First Frame", select_points, param=(points, selecting, first_frame_display)) # pass points to select_points

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

        correction_angle = angle
        print(f"Correction angle: {correction_angle} degrees")
    
    return correction_angle
        
# Helper function to rotate a frame
def rotate_frame(frame, angle):
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(frame, M, (w, h))
    return rotated
        
def rotate_video(input_video_path):
    """Rotate a video and save the rotated video to the 'rotated_videos' folder"""
    global points, selecting
    
    # Load the video
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error: Could not open {input_video_path}")
        return
    
    # Get the first frame
    ret, first_frame = cap.read()
    
    # Make a copy of the first frame to display
    first_frame_display = first_frame.copy()
    
    # Initialize selecting and points variables
    selecting = True
    points = []

    # Get the correction angle
    correction_angle = get_correction_angle(input_video_path, first_frame_display)

    # Output video
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    angle_str = f"{correction_angle:.2f}".replace(".", "_") + "deg"
    output_video_path = f"rotated_videos/{base_name}_{angle_str}_rotated.avi"
    print(f"Output video: {output_video_path}")
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Rotate the video and write to output
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Rotate the frame using the angle
        rotated_frame = rotate_frame(frame, correction_angle)

        # Write the rotated frame to the output video
        out.write(rotated_frame)
        cv2.imshow("Rotated Video", rotated_frame)
        cv2.waitKey(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computer vision analysis for droplets")
    parser.add_argument("input_vid", help="Path to the input video file")
    args = parser.parse_args()

    # Load the video
    video_path = args.input_vid

    rotate_video(video_path)