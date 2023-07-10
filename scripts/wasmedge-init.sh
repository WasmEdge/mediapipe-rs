#!/bin/bash

# Init the WasmEdge environment  (with wasi-nn plugin and tf-lite backend)

set -ex

source "$(dirname -- "$0")/env.sh"

# Must use the version after 0.13.1 or build from master branch.
WASMEDGE_VERSION=0.13.1
TFLITE_DEPS_VERSION=TF-2.12.0-CC

build_wasmedge_from_source_with_wasi_nn_tflite() {
  # install requirements
  apt update && apt install git software-properties-common libboost-all-dev llvm-14-dev liblld-14-dev cmake ninja-build gcc g++ -y

  REPO_CURL="https://github.com/WasmEdge/WasmEdge.git"
  REPO_BRANCH="master"

  git clone "${REPO_CURL}"
  pushd WasmEdge
  git checkout "${REPO_BRANCH}"
  mkdir build && pushd build
  cmake .. -G "Ninja" -DCMAKE_BUILD_TYPE=Release \
    -DWASMEDGE_PLUGIN_WASI_NN_BACKEND="TensorflowLite" -DCMAKE_INSTALL_PREFIX="${WASMEDGE_PATH}"
  ninja && ninja install

  # install tflite deps
  cp "_deps/wasmedgetensorflowdepslite-src/libtensorflowlite_c.so" "${WASMEDGE_LIB_PATH}"/

  popd
  popd
}

download_wasmedge_with_wasi_nn_tflite() {
  curl -sLO https://github.com/WasmEdge/WasmEdge/releases/download/${WASMEDGE_VERSION}/WasmEdge-${WASMEDGE_VERSION}-manylinux2014_x86_64.tar.gz
  curl -sLO https://github.com/WasmEdge/WasmEdge/releases/download/${WASMEDGE_VERSION}/WasmEdge-plugin-wasi_nn-tensorflowlite-${WASMEDGE_VERSION}-manylinux2014_x86_64.tar.gz

  tar -zxf WasmEdge-${WASMEDGE_VERSION}-manylinux2014_x86_64.tar.gz
  tar -zxf WasmEdge-plugin-wasi_nn-tensorflowlite-${WASMEDGE_VERSION}-manylinux2014_x86_64.tar.gz

  mkdir -p "${WASMEDGE_BIN_PATH}"
  mkdir -p "${WASMEDGE_LIB_PATH}"
  mkdir -p "${WASMEDGE_PLUGIN_PATH}"

  mv WasmEdge-${WASMEDGE_VERSION}-Linux/bin/* "${WASMEDGE_BIN_PATH}"/
  if [[ -d "WasmEdge-${WASMEDGE_VERSION}-Linux/lib64/wasmedge/" ]]; then
    mv WasmEdge-${WASMEDGE_VERSION}-Linux/lib64/wasmedge/* "${WASMEDGE_PLUGIN_PATH}"/
    rmdir WasmEdge-${WASMEDGE_VERSION}-Linux/lib64/wasmedge # avoid mv fail when /lib/wasmedge exists
  fi
  mv WasmEdge-${WASMEDGE_VERSION}-Linux/lib64/* "${WASMEDGE_LIB_PATH}"/
  mv libwasmedgePluginWasiNN.so "${WASMEDGE_PLUGIN_PATH}"/

  rm -r WasmEdge-${WASMEDGE_VERSION}-manylinux2014_x86_64.tar.gz WasmEdge-${WASMEDGE_VERSION}-Linux/ WasmEdge-plugin-wasi_nn-tensorflowlite-${WASMEDGE_VERSION}-manylinux2014_x86_64.tar.gz
}

download_wasmedge_tflite_deps() {
  tar_filename="WasmEdge-tensorflow-deps-TFLite-${TFLITE_DEPS_VERSION}-manylinux2014_x86_64.tar.gz"
  curl -sLO "https://github.com/second-state/WasmEdge-tensorflow-deps/releases/download/${TFLITE_DEPS_VERSION}/${tar_filename}"

  tar -zxf "${tar_filename}"

  mkdir -p "${WASMEDGE_LIB_PATH}"
  mv libtensorflowlite_c.so "${WASMEDGE_LIB_PATH}/"
  mv libtensorflowlite_flex.so "${WASMEDGE_LIB_PATH}/"

  rm "${tar_filename}"
}

mediapipe_custom_ops_init() {
  url_base="https://github.com/yanghaku/mediapipe-custom-ops/releases/download"
  tag="v0.0.1"
  dl_filename="mediapipe_custom_ops-tf2.6.0-0.0.1-manylinux2014_x86_64.tar.gz"

  curl -sLO "${url_base}/${tag}/${dl_filename}"
  tar -zxvf "${dl_filename}"
  rm "${dl_filename}"

  mv "${WASMEDGE_WASINN_CUSTOM_OPS_LIBNAME}" "${WASMEDGE_PLUGIN_WASI_NN_TFLITE_CUSTOM_OPS_PATH}"
}

#build_wasmedge_from_source_with_wasi_nn_tflite
download_wasmedge_with_wasi_nn_tflite
download_wasmedge_tflite_deps
# Use the mediapipe custom ops is only a draft PR, so I comment it and comment the test ```test_image_segmentation_model_2```.
#mediapipe_custom_ops_init
