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

  popd || exit
}

#download_audio_data
download_img_data
