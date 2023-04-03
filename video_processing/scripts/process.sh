#!/bin/bash

# Set the input video filename
input_file=$1

# Create necessary directories if they don't exist
mkdir -p position_data rotated_videos tracked_videos

# Run video_correction.py with input file
python video_correction.py "$input_file"

# Check if video_correction.py was successful
if [ $? -ne 0 ]; then
    echo "Error: video_correction.py failed"
    exit 1
fi

# Run video_graphing.py with rotated video file
echo "Base name: $(basename "$input_file" .avi)"
rotated_file="rotated_videos/$(basename "$input_file" .avi)_rotated.avi"
python droplet_tracking.py "$rotated_file"

# Check if droplet_tracking.py was successful
if [ $? -ne 0 ]; then
    echo "Error: droplet_tracking.py failed"
    exit 1
fi

echo "Processing complete"


