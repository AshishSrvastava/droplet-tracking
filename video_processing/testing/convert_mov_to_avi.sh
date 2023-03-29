#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <input_mov_file>"
  exit 1
fi

input_mov_file="$1"
base_name="${input_mov_file%.*}"
output_avi_file="${base_name}.avi"

ffmpeg -i "$input_mov_file" -c:v libxvid -q:v 4 -c:a libmp3lame -q:a 2 "$output_avi_file"

