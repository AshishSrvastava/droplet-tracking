# TODO: 
# - Calculate circularity, graph time series of contour circularity
# - angle adjustment based on channel outline
# - Taylor deformation parameter graphing over time
# - try using polydp instead of the jank ass image area thing you're using rn you dumbass
#       - https://www.authentise.com/post/detecting-circular-shapes-using-contours

# for passing in video in terminal
import sys
import csv
import cv2
import numpy as np

# Input
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
position_data_file = "droplet_posn_time.txt"
with open(position_data_file, "w") as file:
    file.write(f"frame \t x \t y \t width \t height \t deformation \n") # added deformation column to file header

while cap.isOpened:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Get current frame number
    frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    # Convert frame number to string and add leading zeros
    # Put frame number in top left corner of frame
    cv2.rectangle(frame, (0, 0), (100, 30), (0, 0, 0), -1)
    cv2.putText(frame, str(frame_num), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

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

        if area > 1600:
            cv2.drawContours(frame, contour, -1, (0, 0, 255), 3)
            
            # Calculate the minimum bounding rectangle of the contour
            rect = cv2.minAreaRect(contour)
            # Get corner points of rectangle
            box = cv2.boxPoints(rect)
            # convert the corner points to integers
            box = np.int0(box)
            # Draw the bounding rectangle on the frame
            cv2.drawContours(frame, [box], 0, (210, 140, 17), 2)
            # get dimensions of rectangle
            width, height = rect[1]
            L_maj, L_min = max(width, height), min(width, height)
            print(f"L_maj: {L_maj} \t L_min: {L_min} \n")
            
            # Calculate the Taylor deformation parameter

            # L_maj = major axis length (width)
            # L_min = minor axis length (height)
            deformation = (L_maj - L_min)/(L_maj + L_min)
            # output deformation parameter to a file, along with frame number and droplet position
            with open(position_data_file, "a") as file:
                frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
                # calculate moments of contour
                M = cv2.moments(contour)
                center_x = round(M["m10"] / M["m00"])
                center_y = round(M["m01"] / M["m00"])
                if center_y > 600:
                    file.write(f"{frame_num} \t {center_x} \t {center_y} \t {width} \t {height} \t {deformation} \n")

            if cv2.isContourConvex(contour):
                print(f"Contour is convex")
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
            
            if frame_num in [1, 131, 216]:
                cv2.imwrite(f"frame_{frame_num}.png", frame)

    # write the contoured frame
    out.write(frame)

    
    # concatenate all frames into a single image for display
    # convert the 2d contour image to 3d image by copying same image to 3rd channel
    mask_3ch = cv2.merge((mask, mask, mask))
    video_image = np.concatenate((blurred_frame, mask_3ch, frame), axis=1)
    # show the image
    cv2.imshow("Video, mask and contour", video_image)
    cv2.waitKey(1)

    key = cv2.waitKey(100)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
