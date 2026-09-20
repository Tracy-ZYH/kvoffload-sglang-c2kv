#!/usr/bin/env bash
# CUDA execution profile for C2KV; model and memory settings come from the caller.
set -euo pipefail

SGLANG_SOURCE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
export PYTHONPATH="${SGLANG_SOURCE}/python${PYTHONPATH:+:${PYTHONPATH}}"

# Multi-round prefill remains eager. Decode CUDA graphs remain enabled and
# replay the dynamic C2KV projection mask.
# Radix caching uses the server default; callers can disable it explicitly.
exec "${PYTHON_BIN}" -m sglang.launch_server \
  --device cuda \
  --attention-backend flashinfer \
  --enable-c2kv \
  --enable-streaming-session \
  --page-size 1 \
  --max-running-requests 1 \
  --disable-overlap-schedule \
  --disable-piecewise-cuda-graph \
  "$@"
