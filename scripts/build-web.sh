#!/usr/bin/env bash
# Build the Rust geometry core to wasm and copy it into web/.
# Needs: rustup target add wasm32-unknown-unknown. wasm-opt is used if installed.
set -euo pipefail
cd "$(dirname "$0")/.."
RUSTFLAGS="-C target-feature=+simd128" cargo build -p campose-wasm --release --target wasm32-unknown-unknown
OUT=target/wasm32-unknown-unknown/release/campose_wasm.wasm
if command -v wasm-opt >/dev/null 2>&1; then
  wasm-opt -Os --enable-simd "$OUT" -o web/campose_core.wasm
else
  cp "$OUT" web/campose_core.wasm
fi
ls -l web/campose_core.wasm
