#!/bin/bash

# Init the WasmEdge environment  (with wasi-nn plugin and tf-lite backend)

set -ex

source "$(dirname -- "$0")/env.sh"

# Because the WasmEdge 0.12.0 and 0.12.1 will cause segment fault when running,
# so we use the build-from-source method instead of downloading the release file.
# The PR WasmEdge/WasmEdge#2360 has fixed it.
# When the next version of WasmEdge has been released, we can delete the build-from-source method
# and use these download functions.
export WASMEDGE_VERSION=0.13.0-alpha.1

build_wasmedge_with_nn_tflite() {
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

wasmedge_with_nn_init() {
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

wasmedge_tflite_deps_init() {
  curl -sLO https://github.com/second-state/WasmEdge-tensorflow-deps/releases/download/${WASMEDGE_VERSION}/WasmEdge-tensorflow-deps-TFLite-${WASMEDGE_VERSION}-manylinux2014_x86_64.tar.gz

  tar -zxf WasmEdge-tensorflow-deps-TFLite-${WASMEDGE_VERSION}-manylinux2014_x86_64.tar.gz

  mv libtensorflowlite_c.so "${WASMEDGE_LIB_PATH}"/

  rm WasmEdge-tensorflow-deps-TFLite-${WASMEDGE_VERSION}-manylinux2014_x86_64.tar.gz
}

wasmedge_lib_env_init() {
  echo "${WASMEDGE_LIB_PATH}" >/etc/ld.so.conf.d/wasmedge.conf
  ldconfig
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

build_wasmedge_with_nn_tflite
#wasmedge_with_nn_init
#wasmedge_tflite_deps_init
# wasmedge_lib_env_init
# Use the mediapipe custom ops is only a draft PR, so I comment it and comment the test ```test_image_segmentation_model_2```.
#mediapipe_custom_ops_init
