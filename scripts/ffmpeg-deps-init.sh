#!/bin/bash

set -ex

# install libclang and clang
apt install -y libclang1-14 clang

FFMPEG_VERSION="v0.0.1"
FFMPEG_FILENAME="ffmpeg-6.0-wasm32-wasi-v0.0.1.tar.gz"

WASI_SDK_VERSION="wasi-sdk-19"
LIB_CLANG_RT_FILENAME="libclang_rt.builtins-wasm32-wasi-19.0.tar.gz"
WASI_SYSROOT_FILENAME="wasi-sysroot-19.0.tar.gz"

OUTPUT_ROOT="$(realpath "$(dirname -- "$0")")/../assets"
TEMP_DIR="/tmp"

download_ffmpeg_lib() {
  pushd "${TEMP_DIR}"

  BASE_URL="https://github.com/yanghaku/ffmpeg-wasm32-wasi/releases/download/"

  OUTPUT_DIR="${OUTPUT_ROOT:?}/ffmpeg-lib/"
  mkdir -p "${OUTPUT_DIR}"

  # clean old if exist
  if [[ -d "${OUTPUT_DIR:?}/include" ]]; then
    rm -rf "${OUTPUT_DIR:?}/include"
  fi
  if [[ -d "${OUTPUT_DIR:?}/lib" ]]; then
    rm -rf "${OUTPUT_DIR:?}/lib"
  fi

  curl -sLO "${BASE_URL}/${FFMPEG_VERSION}/${FFMPEG_FILENAME}"
  tar -zxvf "${FFMPEG_FILENAME}"
  folder_name=$(basename "${FFMPEG_FILENAME}" .tar.gz)
  mv "${folder_name:?}/include" "${OUTPUT_DIR:?}/include"
  mv "${folder_name:?}/lib" "${OUTPUT_DIR:?}/lib"

  rm -rf "${FFMPEG_FILENAME}" "${folder_name}"

  popd
}

download_wasi_sysroot() {
  pushd "${TEMP_DIR}"

  BASE_URL="https://github.com/WebAssembly/wasi-sdk/releases/download/"

  OUTPUT_DIR="${OUTPUT_ROOT:?}/wasi-sysroot"
  # clean old if exist
  if [[ -d "${OUTPUT_DIR:?}" ]]; then
    rm -rf "${OUTPUT_DIR:?}"
  fi

  curl -sLO "${BASE_URL}/${WASI_SDK_VERSION}/${WASI_SYSROOT_FILENAME}"
  tar -zxvf "${WASI_SYSROOT_FILENAME}"
  mv wasi-sysroot "${OUTPUT_ROOT:?}/"
  rm "${WASI_SYSROOT_FILENAME}"

  popd
}

download_lib_clang_rt() {
  pushd "${TEMP_DIR}"

  BASE_URL="https://github.com/WebAssembly/wasi-sdk/releases/download/"

  LIB_NAME="libclang_rt.builtins-wasm32.a"

  OUTPUT_DIR="${OUTPUT_ROOT:?}/clang-rt"
  mkdir -p "${OUTPUT_DIR}"
  # clean old if exist
  if [[ -f "${OUTPUT_DIR:?}/${LIB_NAME}" ]]; then
    rm -rf "${OUTPUT_DIR:?}/${LIB_NAME}"
  fi

  curl -sLO "${BASE_URL}/${WASI_SDK_VERSION}/${LIB_CLANG_RT_FILENAME}"

  folder_name=$(basename "${LIB_CLANG_RT_FILENAME}" .tar.gz)
  mkdir -p "${folder_name}"
  tar -zxvf ${LIB_CLANG_RT_FILENAME} -C "${folder_name}"
  mv "${folder_name}/lib/wasi/${LIB_NAME}" "${OUTPUT_DIR}/${LIB_NAME}"

  rm -rf "${LIB_CLANG_RT_FILENAME}" "${folder_name}"
  popd
}

download_ffmpeg_lib
download_wasi_sysroot
download_lib_clang_rt
