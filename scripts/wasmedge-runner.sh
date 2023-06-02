#!/bin/bash

# wrapper the wasmedge to run wasm using cargo

source "$(dirname -- "$0")/env.sh"

wasmedge --dir .:. "$@"
