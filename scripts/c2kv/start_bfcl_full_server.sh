#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR=/home/zhuyuhan/project/c2kv
SGLANG_DIR=/home/zhuyuhan/project/kvoffload-sglang
PYTHON_BIN=/home/zhuyuhan/miniconda3/envs/sglang/bin/python

MODEL_PATH="${ROOT_DIR}/checkpoints/qwen3-4b-agent-tooldoc-hardneg-npu"

# BFCL 请求里会使用这个 model id，因此 server 也直接暴露同名模型
SERVED_MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507-FC"

HOST=127.0.0.1
PORT=32000
DEVICE=7
MEM_FRACTION_STATIC=0.55

echo "Preparing Ascend environment..."

set +e
set +u
source /usr/local/Ascend/cann-8.5.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
set -e

export PYTHONPATH="${ROOT_DIR}/python:${ROOT_DIR}/python/inference:${ROOT_DIR}/agent:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1

export NO_PROXY="127.0.0.1,localhost,::1"
export no_proxy="${NO_PROXY}"

cd "${SGLANG_DIR}"

echo "=========================================="
echo "BFCL Full SGLang Server"
echo "MODEL=${MODEL_PATH}"
echo "SERVED_MODEL=${SERVED_MODEL_NAME}"
echo "DEVICE=${DEVICE}"
echo "URL=http://${HOST}:${PORT}/v1"
echo "=========================================="

SGLANG_DEBUG_MEMORY_POOL=1 \
SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 \
SGLANG_EMPTY_CACHE_INTERVAL=1 \
ASCEND_LAUNCH_BLOCKING=1 \
TASK_QUEUE_ENABLE=1 \
ASCEND_RT_VISIBLE_DEVICES="${DEVICE}" \
"${PYTHON_BIN}" -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --model-impl sglang \
  --device npu \
  --attention-backend ascend \
  --tool-call-parser qwen25 \
  --enable-c2kv \
  --dtype bfloat16 \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --host "${HOST}" \
  --port "${PORT}"
