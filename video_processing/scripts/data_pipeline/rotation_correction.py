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


def select_points(event, x, y, flags, param):
    """Mouse event callback function"""
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

def update_first_frame_display(x, y):
    """Update first frame display with points and green dot under cursor"""
    global points, first_frame, first_frame_display

    first_frame_display = first_frame.copy()

    for i, point in enumerate(points):
        cv2.circle(first_frame_display, point, 5, (0, 0, 255), -1)
        cv2.putText(first_frame_display, f"Point {i + 1}", (point[0] + 5, point[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if selecting:
        cv2.circle(first_frame_display, (x, y), 5, (0, 255, 0), -1)
        
def get_correction_angle(input_vid):
    """Get the correction angle from the input video filename"""
    global points
    
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
        correction_angle = angle
        print(f"Correction angle: {correction_angle} degrees")
    
    return correction_angle
        
def rotate_video(input_video_path):
    """Rotate a video and save the rotated video to the 'rotated_videos' folder"""
    global points, selecting
    
    # Load the video
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error: Could not open {input_video_path}")
        return

    # Get the correction angle
    correction_angle = get_correction_angle(input_video_path)

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

# # Check if angle is already in the input video filename
# angle_pattern = re.compile(r"[-+]?\d+_\d+(?=_angle)")  # Updated regular expression
# angle_match = angle_pattern.search(input_vid)

# if angle_match:
#     angle_str = angle_match.group().replace("_", ".")
#     correction_angle = float(angle_str)
#     print(f"Correction angle from filename: {correction_angle} radians")
# else:
#     # Show first frame for user to select points
#     cv2.namedWindow("First Frame")
#     cv2.setMouseCallback("First Frame", select_points)

#     points = []
#     selecting = True

#     while selecting:
#         cv2.imshow("First Frame", first_frame_display)
#         key = cv2.waitKey(1) & 0xFF
#         if key == 27 or key == ord("q"):
#             exit(0)

#     cv2.destroyAllWindows()

#         # Fit a line to the two points and calculate the angle
#     x1, y1 = points[0]
#     x2, y2 = points[1]
#     angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
#     angle = round(angle, 2)
#     correction_angle = angle
#     print(f"Correction angle: {correction_angle} degrees")

# Helper function to rotate a frame
def rotate_frame(frame, angle):
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(frame, M, (w, h))
    return rotated

# # Modify output filenames to include the correction angle
# angle_str = f"{correction_angle:.2f}".replace(".", "_") + "deg"
# print(f"Angle string: {angle_str}")
# print(f"Base name: {base_name} \n angle_str: {angle_str}")
# output_vid = f"rotated_videos/{base_name}_{angle_str}_{identifier_suffix}"
# print(f"Output video: {output_vid}")

# # Output video
# fourcc = cv2.VideoWriter_fourcc("D", "I", "V", "X")
# out = cv2.VideoWriter(output_vid, fourcc, 20.0, (1920, 1080))

# # Process the video
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break

#     # Rotate frame using the correction angle
#     frame = rotate_frame(frame, correction_angle)


#     # write the rotated frame
#     out.write(frame)

#     # show the image
#     cv2.imshow("Rotated Video", frame)
#     cv2.waitKey(1)

#     # Break the loop if the user presses 'q'
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Computer vision analysis for droplets")
#     parser.add_argument("input_vid", help="Path to the input video file")
#     args = parser.parse_args()

#     # Load the video
#     video_path = args.input_vid
#     cap = cv2.VideoCapture(video_path)

#     if not cap.isOpened():
#         print(f"Error: Could not open {video_path}")
#         sys.exit()
        
#     # Output video
#     fourcc = cv2.VideoWriter_fourcc("D", "I", "V", "X")
#     base_name = os.path.splitext(os.path.basename(video_path))[0]
#     output_vid = f"rotated_videos/{base_name}_rotated.avi"
#     print(f"Output video: {output_vid}")
#     out = cv2.VideoWriter(output_vid, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

#     # Add the rotation functionality
#     angle = correction_angle  # You can set the desired rotation angle here
#     num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     for frame_num in range(num_frames):
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Rotate the frame using the angle
#         # print(f"Angle: {angle}")
#         rotated_frame = rotate_frame(frame, angle)
# # 
#         # Convert the rotated frame to HSV
#         hsv = cv2.cvtColor(rotated_frame, cv2.COLOR_BGR2HSV)
        

#         # Write the rotated frame to the output video
#         out.write(rotated_frame)
#         cv2.imshow("Rotated Video", rotated_frame)
#         cv2.waitKey(1)
        
#         # Progress calculation
#         progress = int((frame_num / num_frames) * 100)
#         bar_width = 50
#         filled_width = int(bar_width * progress / 100)
#         remaining_width = bar_width - filled_width
#         print(f"[{'=' * filled_width}{' ' * remaining_width}] {progress}%\r", end="")

    
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
    
def main():
    """Main function"""
    # Input
    input_video_path = sys.argv[1]
    print(f"Input video: {input_video_path}")

    # Get the rotation angle from the user
    rotation_angle = get_correction_angle()
    
    # Rotate the video and save the rotated video to the 'rotated_videos' folder
    rotate_video(input_video_path, rotation_angle)

if __name__ == "__main__":
    main()