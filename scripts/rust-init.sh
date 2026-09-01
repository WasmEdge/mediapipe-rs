#!/bin/bash

# install curl
apt update && apt install curl -y

# install rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -q -y
source "$HOME/.cargo/env"

# add wasm32-wasip1 target
rustup target add wasm32-wasip1
