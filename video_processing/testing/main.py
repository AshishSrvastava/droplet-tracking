import sys
import cv2
import numpy as np
import math
import re
import time

# Constants
LOWER_BLUE = np.array([63, 0, 0])
UPPER_BLUE = np.array([179, 255, 255])
MIN_CONTOUR_AREA = 1600

# Helper functions
def rotate_frame(frame, angle):
    (h, w) = frame.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(frame, M, (w, h))
    return rotated

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

def update_first_frame_display(x, y):
    global points, first_frame, first_frame_display

    first_frame_display = first_frame.copy()

    for i, point in enumerate(points):
        cv2.circle(first_frame_display, point, 5, (0, 0, 255), -1)
        cv2.putText(first_frame_display, f"Point {i + 1}", (point[0] + 5, point[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if selecting:
        cv2.circle(first_frame_display, (x, y), 5, (0, 255, 0), -1)

def get_correction_angle(input_vid):
    cap = cv2.VideoCapture(input_vid)
    if not cap.isOpened():
        raise Exception(f"Error opening video file: {input_vid}")

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    global first_frame, first_frame_display
    ret, first_frame = cap.read()
    first_frame_display = first_frame.copy()

    angle_pattern = re.compile(r"[-+]?\d+(\.\d{1,3})?_deg")
    angle_match = angle_pattern.search(input_vid)

    if angle_match:
        angle_str = angle_match.group().replace("_deg", "")
        correction_angle = float(angle_str)
        print(f"Correction angle from filename: {correction_angle:.3e} degrees")
    else:
        cv2.namedWindow("First Frame")
        cv2.setMouseCallback("First Frame", select_points)

        global points, selecting
        points = []
        selecting = True

        while selecting:
            cv2.imshow("First Frame", first_frame_display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        if len(points) != 2:
            raise Exception("Two points must be selected")

        cv2.destroyAllWindows()
        dx, dy = points[1][0] - points[0][0], points[1][1] - points[0][1]
        correction_angle = -1*math.degrees(math.atan2(dy, dx))
        correction_angle = round(correction_angle, 3)
        print(f"Correction angle from selected points: {correction_angle} degrees")

    cap.release()
    return correction_angle

def process_video(input_vid, correction_angle, output_vid, position_data_file, show_contour=False, show_centroid=False, show_rejected=False, show_bbox=False):
    cap = cv2.VideoCapture(input_vid)
    if not cap.isOpened():
        raise Exception(f"Error opening video file: {input_vid}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_vid, fourcc, fps, (width, height))

    with open(position_data_file, "w") as pos_file:
        pos_file.write("Frame, x, y\n")

        frame_counter = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1
            rotated_frame = rotate_frame(frame, correction_angle)

            # Convert frame to grayscale and apply Gaussian blur
            blurred_frame = cv2.GaussianBlur(rotated_frame, (5, 5), 0)

            # Convert to HSV
            hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)

                if area > MIN_CONTOUR_AREA:
                    if show_contour:
                        cv2.drawContours(rotated_frame, [contour], -1, (0, 255, 0), 1)

                    if show_centroid:
                        M = cv2.moments(contour)
                        center_x = round(M["m10"] / M["m00"])
                        center_y = round(M["m01"] / M["m00"])
                        cv2.circle(rotated_frame, (center_x, center_y), 5, (0, 255, 0), -1)

                    if show_bbox:
                        rect = cv2.minAreaRect(contour)
                        box = cv2.boxPoints(rect)
                        box = np.intp(box)
                        cv2.drawContours(rotated_frame, [box], 0, (210, 140, 17), 2)

                elif show_rejected:
                    cv2.drawContours(rotated_frame, [contour], -1, (0, 0, 255), 1)

            out.write(rotated_frame)

            if frame_counter % 100 == 0:
                print(f"Processed {frame_counter} frames")

            cv2.imshow("Rotated Frame", rotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    if len(sys.argv) != 2:
        print("Usage: python rotation_correction.py <input_video>")
        exit(0)

    input_vid = sys.argv[1]
    base_name = input_vid.split("/")[-1].split(".")[0]
    print(f"base_name: {base_name}")

    # Add the following line to remove any digits before "mL"
    base_name = re.sub(r'^\d*\.*\d*', '', base_name)

    correction_angle = get_correction_angle(input_vid)
    print(f"correction_angle: {correction_angle} degrees")
    angle_info = f"{'{:+.3f}'.format(correction_angle).replace('.', '_').replace('+', '')}_deg"

    print(f"angle_info: {angle_info}")
    output_vid = f"rotated_videos/{base_name}{angle_info}.avi"
    print(f"Output video: {output_vid}")

    show_contour = True
    show_centroid = True
    show_rejected = True
    show_bbox = True

    position_data_file = f"{base_name}_position_data.txt"

    process_video(input_vid, correction_angle, output_vid, position_data_file, show_contour, show_centroid, show_rejected, show_bbox)

if __name__ == "__main__":
    main()

