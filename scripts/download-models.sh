#!/usr/bin/env bash

set -ex

model_path="$(realpath "$(dirname -- "$0")")/../assets/models"

object_detection_init() {
  object_detection_dir="${model_path}/object_detection"
  mkdir -p "${object_detection_dir}"
  pushd "${object_detection_dir}"

  model_urls=("https://storage.googleapis.com/mediapipe-tasks/object_detector/efficientdet_lite0_uint8.tflite"
              "https://storage.googleapis.com/mediapipe-tasks/object_detector/efficientdet_lite0_fp32.tflite"
              "https://storage.googleapis.com/mediapipe-tasks/object_detector/efficientdet_lite2_uint8.tflite"
              "https://storage.googleapis.com/mediapipe-tasks/object_detector/efficientdet_lite2_fp32.tflite"
              "https://storage.googleapis.com/mediapipe-tasks/object_detector/mobilenetv2_ssd_256_uint8.tflite"
              "https://storage.googleapis.com/mediapipe-tasks/object_detector/mobilenetv2_ssd_256_fp32.tflite"
  )

  for url in "${model_urls[@]}"; do
    curl -sLO "${url}"
  done

  popd
}

image_classification_init() {
  image_classification_dir="${model_path}/image_classification"
  mkdir -p "${image_classification_dir}"
  pushd "${image_classification_dir}"

  model_urls=("https://storage.googleapis.com/mediapipe-tasks/image_classifier/efficientnet_lite0_uint8.tflite"
              "https://storage.googleapis.com/mediapipe-tasks/image_classifier/efficientnet_lite0_fp32.tflite"
              "https://storage.googleapis.com/mediapipe-tasks/image_classifier/efficientnet_lite2_uint8.tflite"
              "https://storage.googleapis.com/mediapipe-tasks/image_classifier/efficientnet_lite2_fp32.tflite"
  )

  for url in "${model_urls[@]}"; do
    curl -sLO "${url}"
  done

  # for custom model downloaded from kaggle
  bird_model_name="lite-model_aiy_vision_classifier_birds_V1_3"
  kaggle_tflite_filename="3.tflite"
  curl -sL "https://www.kaggle.com/api/v1/models/google/aiy/tfLite/vision-classifier-birds-v1/3/download" -o "${bird_model_name}.tar.gz"
  tar -zxvf "${bird_model_name}.tar.gz"
  mv "${kaggle_tflite_filename}" "${bird_model_name}.tflite"
  rm -rf "${bird_model_name}.tar.gz"

  popd
}

hand_landmark_detection_init() {
  hand_landmark_detection_dir="${model_path}/hand_landmark_detection"
  mkdir -p "${hand_landmark_detection_dir}"
  pushd "${hand_landmark_detection_dir}"

  model_urls=("https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/hand_landmarker.task"
  )

  for url in "${model_urls[@]}"; do
    curl -sLO "${url}"
  done

  popd
}

gesture_recognition_init() {
  gesture_recognition_dir="${model_path}/gesture_recognition"
  mkdir -p "${gesture_recognition_dir}"
  pushd "${gesture_recognition_dir}"

  model_urls=("https://storage.googleapis.com/mediapipe-tasks/gesture_recognizer/gesture_recognizer.task"
  )

  for url in "${model_urls[@]}"; do
    curl -sLO "${url}"
  done

  popd
}

image_segmentation_init() {
  image_segmentation_dir="${model_path}/image_segmentation"
  mkdir -p "${image_segmentation_dir}"
  pushd "${image_segmentation_dir}"

  model_urls=("https://storage.googleapis.com/mediapipe-tasks/image_segmenter/deeplabv3.tflite"
              "https://storage.googleapis.com/mediapipe-tasks/image_segmenter/selfie_segm_128_128_3.tflite"
  )

  for url in "${model_urls[@]}"; do
    curl -sLO "${url}"
  done

  popd
}

image_embedding_init() {
  image_embedding_dir="${model_path}/image_embedding"
  mkdir -p "${image_embedding_dir}"
  pushd "${image_embedding_dir}"

  model_urls=("https://storage.googleapis.com/mediapipe-tasks/image_embedder/mobilenet_v3_small_075_224_embedder.tflite"
              "https://storage.googleapis.com/mediapipe-tasks/image_embedder/mobilenet_v3_large_075_224_embedder.tflite"
  )

  for url in "${model_urls[@]}"; do
    curl -sLO "${url}"
  done

  popd
}

face_detection_init() {
    face_detection_dir="${model_path}/face_detection"
    mkdir -p "${face_detection_dir}"
    pushd "${face_detection_dir}"

    model_urls=("https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite"
    )

    # now models have not be released
    model_urls=()
    curl -SL "https://storage.googleapis.com/mediapipe-assets/face_detection_short_range.tflite?generation=1677044301978921" -o "face_detection_short_range.tflite"

    for url in "${model_urls[@]}"; do
      curl -sLO "${url}"
    done

    popd
}

face_landmark_init() {
    face_landmark_dir="${model_path}/face_landmark"
    mkdir -p "${face_landmark_dir}"
    pushd "${face_landmark_dir}"

    model_urls=("https://storage.googleapis.com/mediapipe-tasks/face_landmarker/face_landmarker.task"
    )

    for url in "${model_urls[@]}"; do
      curl -sLO "${url}"
    done

    popd
}


audio_classification_init() {
  audio_classification_dir="${model_path}/audio_classification"
  mkdir -p "${audio_classification_dir}"
  pushd "${audio_classification_dir}"

  model_urls=("https://storage.googleapis.com/mediapipe-tasks/audio_classifier/yamnet_audio_classifier_with_metadata.tflite"
  )

  for url in "${model_urls[@]}"; do
    curl -sLO "${url}"
  done

  popd
}

text_classification_init() {
  text_classification_dir="${model_path}/text_classification"
  mkdir -p "${text_classification_dir}"
  pushd "${text_classification_dir}"

  model_urls=("https://storage.googleapis.com/mediapipe-tasks/text_classifier/bert_text_classifier.tflite"
              "https://storage.googleapis.com/mediapipe-tasks/text_classifier/average_word_embedding.tflite"
  )

  for url in "${model_urls[@]}"; do
    curl -sLO "${url}"
  done

  popd
}

text_embedding_init() {
    text_embedding_dir="${model_path}/text_embedding"
    mkdir -p "${text_embedding_dir}"
    pushd "${text_embedding_dir}"

    model_urls=(
      "https://storage.googleapis.com/mediapipe-models/text_embedder/bert_embedder/float32/1/bert_embedder.tflite"
      "https://storage.googleapis.com/mediapipe-models/text_embedder/universal_sentence_encoder/float32/latest/universal_sentence_encoder.tflite"
    )

    for url in "${model_urls[@]}"; do
      curl -sLO "${url}"
    done

    popd
}

object_detection_init
image_classification_init
gesture_recognition_init
hand_landmark_detection_init
image_segmentation_init
image_embedding_init
face_detection_init
face_landmark_init
audio_classification_init
text_classification_init
text_embedding_init
