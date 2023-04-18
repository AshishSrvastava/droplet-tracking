import sys
import csv
import cv2
import numpy as np
import re
import argparse
import os

def get_upper_lower_bounds(input_vid):
    pass

def main(input_vid, position_data_file, output_video, enable_tracked_video, show_tracked_video):
    cap = cv2.VideoCapture(input_vid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 1920
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 1080

    # Extract angle from input video file name
    # angle_pattern = re.compile(r"[-+]?\d*\.\d+deg")
    # Extract angle from input video file name
    angle_pattern = re.compile(r"[-+]?\d+_\d+(?=deg)")  # Updated regular expression
    angle_match = angle_pattern.search(input_vid)
    angle_text = f"Angle: {angle_match.group(0).replace('_', '.')} deg" if angle_match else "Angle: N/A"
    


    # Output video
    # Define the codec and create VideoWriter object
    # macOS: DIVX (.avi) or MJPG (.mp4) | Windows: DIVX
    fourcc = cv2.VideoWriter_fourcc("D", "I", "V", "X")
    out = cv2.VideoWriter(output_video, fourcc, 20.0, (1920, 1080))

    # Output data for graphing
    with open(position_data_file, "w") as file:
        file.write(f"frame \t x \t y \t width \t height \t deformation \n") 
    
    # Calculate the total number of frames in the video
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        # Get current frame number
        frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        progress = int((frame_num / num_frames) * 100)
        bar_width = 50
        filled_width = int(bar_width * progress / 100)
        remaining_width = bar_width - filled_width
        print(f"[{'=' * filled_width}{' ' * remaining_width}] {progress}%\r", end="")

        
        # Put frame number in top left corner of frame
        cv2.rectangle(frame, (0, 0), (100, 30), (0, 0, 0), -1)
        cv2.putText(frame, str(frame_num), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # # Put correction angle in top right corner of frame
        # cv2.rectangle(frame, (int(frame.shape[1]) - 300, 0), (int(frame.shape[1]), 30), (0, 0, 0), -1)
        # cv2.putText(frame, angle_text, (int(frame.shape[1]) - 285, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Convert to grayscale and apply Gaussian blur
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # Convert to HSV
        hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

        # Define range of color in HSV
        lower_blue = np.array([63, 0, 0])
        upper_blue = np.array([179, 255, 255])
        # Old values for glycerol droplet
        # (56, 0, 0)
        # 179, 255, 255

        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Find contours 
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # all contours
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)

        # contour filtering
        for contour in contours:
            area = cv2.contourArea(contour)
            # For area debugging
            # print(f"Area: {area}")

            # Filter out contours that are too small
            if area < 1600:
                continue
            
            # Circularity filtering
            # Fit an ellipse to the contour
            ellipse = cv2.fitEllipse(contour)
            
            # Calculate the area and perimeter of the ellipse
            area = np.pi * ellipse[1][0] * ellipse[1][1] / 4
            perimeter = np.pi * (3 * (ellipse[1][0] + ellipse[1][1]) - np.sqrt((3 * ellipse[1][0] + ellipse[1][1]) * (ellipse[1][0] + 3 * ellipse[1][1])))
            # Calculate the circularity of the contour
            circularity = 4 * np.pi * area / (perimeter * perimeter)

            
            cv2.drawContours(frame, contour, -1, (0, 0, 255), 3)
            
            # Taylor Deformation Parameter
            # Calculate the minimum bounding rectangle of the contour
            rect = cv2.minAreaRect(contour)
            # Get corner points of rectangle
            box = cv2.boxPoints(rect)
            # convert the corner points to integers
            box = np.intp(box)
            # Draw the bounding rectangle on the frame
            cv2.drawContours(frame, [box], 0, (210, 140, 17), 2)
            # get dimensions of rectangle
            width, height = rect[1]
            L_maj, L_min = max(width, height), min(width, height)
            
            # Calculate the Taylor deformation parameter
            # L_maj = major axis length (width)
            # L_min = minor axis length (height)
            deformation = (L_maj - L_min)/(L_maj + L_min)
            # output deformation parameter to a file, along with frame number and droplet position
            M = cv2.moments(contour)
            center_x = round(M["m10"] / M["m00"])
            center_y = round(M["m01"] / M["m00"])
            with open(position_data_file, "a") as file:
                frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
                # calculate moments of contour make sure center is between y=600 and bottom of frame
                if center_y > 600 and center_y < frame.shape[0]:
                    file.write(f"{frame_num} \t {center_x} \t {center_y} \t {width} \t {height} \t {deformation} \n")
                else:
                    pass
            # Calculate image moments of the detected contour
            M = cv2.moments(contour)

            # Draw a circle based centered at centroid coordinates
            cv2.circle(
                frame,
                (round(M["m10"] / M["m00"]), round(M["m01"] / M["m00"])),
                5,
                (0, 255, 0),
                -1,
            )
            
            # Display output frame
            if show_tracked_video:
                cv2.imshow("Processed Frame", frame)
        
            
        # Break the loop if the user presses "q"
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
            
            
        if enable_tracked_video:                
            # write the contoured frame
            out.write(frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computer vision analysis for droplets")
    parser.add_argument("input_vid", help="Path to the input video file")
    parser.add_argument("--position_data_file", default=None)
    parser.add_argument("--output_video", default=None, help="Path to the output video file")
    parser.add_argument("--enable_tracked_video", default=True, action="store_true", help="Enable output of tracked video")
    parser.add_argument("--show_tracked_video", action="store_true", default=False, help="Show processed video frames")
    args = parser.parse_args()
    
    # Set default output file names
    base_filename, ext = os.path.splitext(os.path.basename(args.input_vid))
    if args.position_data_file is None:
        print(f"Position data file not specified, using default: position_data/{base_filename}.txt")
        args.position_data_file = f"position_data/{base_filename}.txt"
    if args.output_video is None and args.enable_tracked_video:
        print(f"Output video file not specified, using default: tracked_videos/{base_filename}_tracked.avi")
        args.output_video = f"tracked_videos/{base_filename}_tracked.avi"
    
    main(args.input_vid, args.position_data_file, args.output_video, args.enable_tracked_video, args.show_tracked_video)