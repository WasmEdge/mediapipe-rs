#!/bin/bash

set -ex

VIDEO_PATH="$(realpath "$(dirname -- "$0")")/../"
IMG_PATH="${VIDEO_PATH}/../img"

output_file="${VIDEO_PATH}/banana_clock_tabby.mp4"

img_arr=("banana.jpg" "clock.jpg" "tabby.jpg")
img_arr_len=${#img_arr[@]}

scale=640x640

# generate command
args=""
filter_complex=""
concat=""
for index in "${!img_arr[@]}"; do
  args="${args} -framerate 100 -loop 1 -t 0.01 -i ${IMG_PATH}/${img_arr[index]}"
  filter_complex="${filter_complex}[${index}:v]scale=${scale},setsar=1[v${index}];"
  concat="${concat}[v${index}]"
done
filter_complex="${filter_complex} ${concat}concat=n=${img_arr_len}:v=1:a=0"

ffmpeg -y ${args} -filter_complex "${filter_complex}" "${output_file}"
