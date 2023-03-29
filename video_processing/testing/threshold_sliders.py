import cv2 
import numpy as np
from pandas import concat

# camera = cv2.VideoCapture('IMG_4509.mov')
# openCV does imread in the BGR format
image = cv2.imread('test_image.png')
print(image.shape)

cv2.namedWindow("Sliders")
cv2.resizeWindow("Sliders", 640, 240)

# test to show video
# cv2.imshow('Original Image', image)

# Playing with editing the video's colorspace first to make drop detection easier
# remove B and G layers from BGR matrix
# This converts the original image matrix, it doesn't save it into a new matrix, so you'll have to run the "cv2.imshow("no red/green", image) command to show it
# image[:,:,0] = np.zeros([image.shape[0], image.shape[1]])
# image[:,:,1] = np.zeros([image.shape[0], image.shape[1]])


# cv2.imshow("no red/green", image)

# save the image
# cv2.imwrite("masked.png", image)

edges = cv2.Canny(image, 10, 100)
# cv2.imshow("Canny", edges)

# convert to hsv colorspace
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# lower and upper H, S, V bounds for blue droplet (these values were the ones that worked best for the video I was playing around with)
# can replace using the values fround using the sliders
lower_bound = np.array([0, 0, 0])
upper_bound = np.array([63, 255, 255])
# try 62 -> 131 for hue

# find the colors within the boundaries
mask = cv2.inRange(hsv, lower_bound, upper_bound)

# Segment only the detected region
segmented_img = cv2.bitwise_and(image, image, mask=mask)
cv2.imwrite("mask.bmp", mask)


# sliders for thresholding
def on_trackbar(*args):
    hue_min = cv2.getTrackbarPos("Hue Min", "Sliders")
    hue_max = cv2.getTrackbarPos("Hue Max", "Sliders")
    sat_min = cv2.getTrackbarPos("Sat Min", "Sliders")
    sat_max = cv2.getTrackbarPos("Sat Max", "Sliders")
    val_min = cv2.getTrackbarPos("Val Min", "Sliders")
    val_max = cv2.getTrackbarPos("Val Max", "Sliders")
    # this'll print the threshold values to your terminal
    print(f"hue: ({hue_min}, {hue_max}) | sat: ({sat_min}, {sat_max}) | val: ({val_min}, {val_max})")


    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])

    imgMASK = cv2.inRange(hsv, lower, upper)
    notMask = cv2.bitwise_not(imgMASK)

    segmented_img = cv2.bitwise_and(image, image, mask=notMask)

    # cv2.imshow("Output1", image)
    # cv2.imshow("Output2", hsv)
    cv2.imshow("Mask", imgMASK)
    cv2.imshow("Masked", segmented_img)



cv2.createTrackbar("Hue Min", "Sliders", 0, 179, on_trackbar)
cv2.createTrackbar("Hue Max", "Sliders", 179, 179, on_trackbar)
cv2.createTrackbar("Sat Min", "Sliders", 0, 255, on_trackbar)
cv2.createTrackbar("Sat Max", "Sliders", 255, 255, on_trackbar)
cv2.createTrackbar("Val Min", "Sliders", 0, 255, on_trackbar)
cv2.createTrackbar("Val Max", "Sliders", 255, 255, on_trackbar)




# This is the stuff I was playing with for the contors
# Find contours from the mask
# contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# output = cv2.drawContours(segmented_img, contours, -1, (0, 0, 255), 3)
# Showing the output
# cv2.imshow("Output", output)

# show some stuff
on_trackbar(0)

# Need this code to be able to exit opencv 
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
Me trying to make this goddamn script work
  +-----+
  |     |
 ( )    |
 /|\    |
  |     | 
 / \    |
        |
        |
========+======
++           ++       
||           ||
||           ||

"""