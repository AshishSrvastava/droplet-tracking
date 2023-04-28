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

# Run video_correction.py with input file
python droplet_tracking.py "$input_file" --show_tracked_video

# Check if video_correction.py was successful
if [ $? -ne 0 ]; then
    echo "Error: video_correction.py failed"
    exit 1
fi