#!/bin/bash

set -ex

source "$HOME/.cargo/env"

CURRENT="$(realpath "$(dirname -- "$0")")"
WASI_SYSROOT="${CURRENT}/../assets/wasi-sysroot"
export BINDGEN_EXTRA_CLANG_ARGS="--sysroot=${WASI_SYSROOT} --target=wasm32-wasip1 -fvisibility=default"

pushd "${CURRENT}/.."

# default features (audio,text,vision)
cargo clippy --all-targets -- -D warnings

# each task feature alone
cargo clippy --all-targets --no-default-features --features="vision" -- -D warnings
cargo clippy --all-targets --no-default-features --features="audio" -- -D warnings
cargo clippy --all-targets --no-default-features --features="text" -- -D warnings

# ffmpeg
cargo clippy --all-targets --features="ffmpeg" -- -D warnings

popd
