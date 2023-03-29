# Import required packages:
import cv2
import numpy as np

# Load the image and convert it to grayscale:
image = cv2.imread("test_image.png")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply cv2.threshold() to get a binary image
ret, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

cv2.imshow("thresh", thresh)

# # Using hsv mask
# hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# # lower and upper H, S, V bounds for blue droplet (these values were the ones that worked best for the video I was playing around with)
# lower_bound = np.array([0, 0, 0])
# upper_bound = np.array([73, 255, 255])

# # find the colors within the boundaries
# thresh = cv2.inRange(hsv, lower_bound, upper_bound)

# Find contours:
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Draw contours:
cv2.drawContours(image, contours, 0, (0, 255, 0), 2)

# Calculate image moments of the detected contour
M = cv2.moments(contours[0])

# Print center (debugging):
print("center X : '{}'".format(round(M['m10'] / M['m00'])))
print("center Y : '{}'".format(round(M['m01'] / M['m00'])))

# Draw a circle based centered at centroid coordinates
cv2.circle(image, (round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])), 5, (0, 255, 0), -1)

# Show image:
cv2.imshow("outline contour & centroid", image)

# Wait until a key is pressed:
cv2.waitKey(0)

# Destroy all created windows:
cv2.destroyAllWindows()