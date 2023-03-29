# Droplet Computer Vision

This project is a Python script that uses the OpenCV library to process a video of droplets moving in a channel. The script detects the droplets in the video, calculates their positions and shapes, and saves the results to a file for further analysis

This script first applies a Gaussian blur to the video frames to reduce noise. It then converts the frames to the HSV (hue, saturation, value) color space, which is better suited for detecting the colored droplets. The script then defines a range of colors in the HSV space that correspond to the color of the droplet, and uses this range to create a binary mask of the frame.

Next, the script uses the `cv2.findContours()` function to find the contours (outlines) of droplets in the frame. It filters the contours by area to remove any small contours that are not droplets. For each remaining contour, the script calculates the minimum bounding rectangle and the Taylor deformation parameter (which is a measure of the droplet's shape [Bentley and Leal](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/an-experimental-investigation-of-drop-deformation-and-breakup-in-steady-twodimensional-linear-flows/A5B375548D1C094B5C61159303B62F61))

The script then overlays the bounding rectangle on the original video frames and saves the resulting frames to the "output.avi" file. It also save the droplet positions and calculated parameters to the "droplet_posn.txt" file for further analysis

## Getting Started

These instructions will get you a copy of the project up and running on your local machine

### Prerequisites 

To run this script, you will need to have Python and the OpenCV library installed on your system. You can install Python from the [Python website](https://www.python.org/) and OpenCV using the following command:
```pip install opencv-python```

### Running the Script

To run the script, open a terminal or command prompt and navigate to the directory where the script is saved. Then, run the following command:
```python droplet_computer_vision.py <input_video>```

Replace `<input_video>` with the path to the input video file. The script will process the video and save the output to the "output.avi" file in the same directory.

## Built With

* [Python](https://www.python.org/) - The programming language used
* [OpenCV](https://opencv.org/) - The computer vision library used

## Authors

* **Ashish Srivastaa** - *2022/Jun/27* - [My Github Profile](https://github.com/AshishSrvastava)


## License

This project is licensed under the [MIT License](LICENSE.md).

