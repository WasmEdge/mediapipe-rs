#!/usr/bin/env bash

set -ex

data_path="$(realpath "$(dirname -- "$0")")/../assets/testdata"

download_audio_data() {
  p="${data_path}/audio"
  mkdir -p "${p}"
  pushd "${p}" || exit

  url="https://storage.googleapis.com/mediapipe-assets/speech_16000_hz_mono.wav"
  curl -sLO "${url}"

  popd || exit
}

download_img_data() {
  p="${data_path}/img"
  mkdir -p "${p}"
  pushd "${p}" || exit

  urls=("https://storage.googleapis.com/mediapipe-tasks/object_detector/cat_and_dog.jpg"
        "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/thumbs_down.jpg"
        "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/victory.jpg"
        "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/thumbs_up.jpg"
        "https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/pointing_up.jpg"
        "https://storage.googleapis.com/mediapipe-assets/burger.jpg"
        "https://storage.googleapis.com/mediapipe-assets/burger_crop.jpg"
        "https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg"
  )

  for url in "${urls[@]}"; do
    curl -sLO "${url}"
  done

  curl -sL https://storage.googleapis.com/mediapipe-assets/portrait.jpg -o face.jpg
  curl -sL https://developers.google.com/static/mediapipe/images/solutions/image-classifier.jpg -o bird.jpg
  curl -sL https://storage.googleapis.com/mediapipe-tasks/image_classifier/cat.jpg -o tabby.jpg

  popd || exit
}

install_ffmpeg() {
  if which ffmpeg; then
    echo "FFMpeg is installed."
  else
    apt install ffmpeg -y
  fi
}

generate_test_video() {
  p="${data_path}/video"
  mkdir -p "${p}"
  pushd "${p}" || exit

  IMG_PATH="../img"
  output_file="./bird_burger_tabby.mp4"

  img_arr=("bird.jpg" "burger.jpg" "tabby.jpg")
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

  popd || exit
}

download_audio_data
download_img_data

install_ffmpeg
generate_test_video
