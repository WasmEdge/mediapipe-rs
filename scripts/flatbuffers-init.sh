#!/bin/bash

set -ex

TMP_DIR="/tmp"

# requirements: flatbuffers compiler
# such as: ```apt install flatbuffers-compiler -y```
FLAT_C="flatc"

CARGO_ROOT="$(dirname -- "$0")/../"

TF_LITE_SCHEMAS_URL=(
  "https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/tasks/metadata/metadata_schema.fbs"
  "https://raw.githubusercontent.com/google/mediapipe/master/mediapipe/tasks/metadata/image_segmenter_metadata_schema.fbs"
  "https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/schema/schema.fbs"
)
TF_LITE_OUT_DIR="/src/model/tflite/generated"

for url in "${TF_LITE_SCHEMAS_URL[@]}"; do
  fbs="${TMP_DIR}/${url##*/}"
  curl -sL "${url}" -o "${fbs}"
  ${FLAT_C} -r -o "${CARGO_ROOT}${TF_LITE_OUT_DIR}" "${fbs}"
done
