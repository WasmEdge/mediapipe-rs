#!/bin/bash

set -ex

source "$HOME/.cargo/env"

CURRENT="$(realpath "$(dirname -- "$0")")"
WASI_SYSROOT="${CURRENT}/../assets/wasi-sysroot"
export BINDGEN_EXTRA_CLANG_ARGS="--sysroot=${WASI_SYSROOT} --target=wasm32-wasi -fvisibility=default"

pushd "${CURRENT}/.."

# default features (audio,text,vision)
cargo test --release -- --nocapture

# ffmpeg features
# 1. audio
cargo test --test audio_classification --release --no-default-features --features="audio,ffmpeg" -- --nocapture
# 2. video
cargo test --test ffmpeg_video_input --release --no-default-features --features="vision,ffmpeg" -- --nocapture

popd
