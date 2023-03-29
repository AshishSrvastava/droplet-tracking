# for passing in video in terminal
import sys
import csv
import cv2
import numpy as np


def main():
    # Input
    # cap = cv2.VideoCapture('0.02mL_15mlmin.avi')
    # below code requires passing in video in terminal 'python 3 video_contours.py '0.02mL_15mlmin.avi'
    input_vid = sys.argv[1]
    cap = cv2.VideoCapture(input_vid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 1920
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 1080

    # Output video
    # Define the codec and create VideoWriter object
    # macOS: DIVX (.avi) or MJPG (.mp4) | Windows: DIVX
    fourcc = cv2.VideoWriter_fourcc("D", "I", "V", "X")
    out = cv2.VideoWriter("output.avi", fourcc, 20.0, (1920, 1080))

    # Output data for graphing
    position_data_file = "droplet_posn.txt"
    with open(position_data_file, "w") as file:
        file.write(f"frame \t x \t y \n")

    while cap.isOpened:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # kernel = np.ones((5,5), np.uint8)
        # eroded_frame = cv2.erode(blurred_frame, kernel, iterations=1)
        hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([63, 0, 0])
        upper_blue = np.array([179, 255, 255])
        # (56, 0, 0)
        # 179, 255, 255

        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )

        # all contours
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)

        # contour filtering
        for contour in contours:
            area = cv2.contourArea(contour)
            print(f"Area: {area}")

            if area > 1600:
                cv2.drawContours(frame, contour, -1, (0, 0, 255), 3)

                if cv2.isContourConvex(contour):
                    print(f"Contour is convex")
                # Calculate image moments of the detected contour
                M = cv2.moments(contour)

                # Print center (debugging):
                # print("center X : '{}'".format(round(M['m10'] / M['m00'])))
                # print("center Y : '{}'".format(round(M['m01'] / M['m00'])))

                center_x = round(M["m10"] / M["m00"])
                center_y = round(M["m01"] / M["m00"])
                frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)

                if center_y > 600:
                    with open(position_data_file, "a") as file:
                        file.write(f"{int(frame_num)} \t {center_x} \t {center_y} \n")

                # Draw a circle based centered at centroid coordinates
                cv2.circle(
                    frame,
                    (round(M["m10"] / M["m00"]), round(M["m01"] / M["m00"])),
                    5,
                    (0, 255, 0),
                    -1,
                )

        # write the contoured frame
        out.write(frame)

        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(100)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
