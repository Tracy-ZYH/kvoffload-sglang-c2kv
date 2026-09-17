# CUDA execution profile for C2KV

`scripts/c2kv/start_cuda_server.sh` selects FlashInfer attention and leaves
decode CUDA graphs enabled. It uses the existing SGLang kernels and C2KV graph
eligibility checks; it does not change compression, query projection, history
selection, or recovery policy. The Ascend launcher is unchanged.

## Launch

Use the Python environment containing the CUDA build of PyTorch, FlashInfer,
and this checkout's SGLang dependencies. Pass the model and memory settings
appropriate to the checkpoint and GPU:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHON_BIN=/path/to/venv/bin/python \
  bash scripts/c2kv/start_cuda_server.sh \
  --model-path /path/to/checkpoint \
  --served-model-name c2kv-agent \
  --dtype bfloat16 \
  --c2kv-gist-type dynamic-interleave --c2kv-gist-param qkv \
  --c2kv-query-proj base \
  --context-length 131072 --max-total-tokens 131072 \
  --chunked-prefill-size 512 \
  --mem-fraction-static 0.8 --c2kv-pool-fraction 0.1 \
  --tool-call-parser qwen25 --host 127.0.0.1 --port 34000
```

The example uses the base-query, dynamic-interleave QKV checkpoint regime.
Keep the checkpoint's existing projection and compression settings when
changing only the execution backend. Capacity options above target a larger
experiment GPU; reduce them for a laptop GPU. `PYTHONPATH` selects this checkout
even when an environment previously installed another editable SGLang tree.

The profile is single-flight for isolated request latency and memory
measurement. It is not a throughput-tuned serving profile. Piecewise graphs
and overlap scheduling remain disabled; multi-round prefill remains eager.
Decode graphs replay the runtime's dynamic C2KV projection mask.

Additional server arguments are passed through. For example, append
`--attention-backend triton` to compare another CUDA backend, or
`--disable-cuda-graph` to compare eager decode.

## Prefix caching and measurement

Radix caching is enabled by the normal server default. Append
`--disable-radix-cache` when the experiment explicitly uses a no-cache policy.
Keep that policy comparable between methods. Physical persistent-history KV
and reusable C2KV gist entries are separate state mechanisms: changing the
attention kernel must not reconstruct evicted history from its raw archive.

Report total resident KV over the complete decision chain, including auxiliary
compression/summary calls and their retained cache. Generation-active KV and
cached evictable KV can be reported separately. Evictable pages still consume
resident memory; subtracting them does not measure total resident KV. Aggregate
peaks across all calls in a decision, not only its final chat response.

## Hardware and validation scope

The CUDA profile was exercised with Qwen3-4B on an RTX 4090 Laptop using PyTorch
2.9.1+cu129, Transformers 5.3.0, FlashInfer 0.6.7.post2, and sgl-kernel 0.4.1.
The existing paper benchmark integration completed real generation and
Full-prefix replay for Full, HiAgent, ACON, C2KV4, H2O, and SnapKV; persistent
H2O/SnapKV also completed a second turn without full-history re-prefill.
Those checks used the paper integration branch, including its separate
telemetry and persistent-history fixes; this PR does not import that branch.

RTX PRO 6000 Blackwell is SM120, which is included in FlashInfer 0.6.7's
hardware support. Compile/download kernels for the target GPU; do not copy
an Ada-only compiled kernel cache. PRO 6000 latency and output checks should
be run on that GPU; the laptop checks do not supply a PRO 6000 speedup factor.

- [NVIDIA compute capabilities](https://developer.nvidia.com/cuda/gpus)
- [FlashInfer 0.6.7 support and installation](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.7/README.md)

The CUDA loader also collects temporary module cycles and releases unused
allocator blocks after loading weights, before KV-pool sizing. This avoids
counting lingering FP32-to-BF16 loader allocations as live model weights.
