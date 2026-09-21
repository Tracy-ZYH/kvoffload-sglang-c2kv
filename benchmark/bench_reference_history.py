"""Small, model-free decode attention parity and timing probe."""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch


parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cuda")
parser.add_argument("--history", type=int, default=2048)
parser.add_argument("--iters", type=int, default=64)
args = parser.parse_args()

if args.device == "npu":
    import torch_npu  # noqa: F401

path = Path(__file__).resolve().parents[1] / "python/sglang/srt/mem_cache/history_kv_reference.py"
spec = importlib.util.spec_from_file_location("history_kv_reference_bench", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

device = torch.device(args.device)
dtype = torch.bfloat16 if args.device != "cpu" else torch.float32
heads, kv_heads, dim = 32, 8, 128
positions = torch.arange(args.history, device=device, dtype=torch.long)
history = module.ReferenceLayerKV(
    key=torch.randn(kv_heads, args.history, dim, device=device, dtype=dtype),
    value=torch.randn(kv_heads, args.history, dim, device=device, dtype=dtype),
    positions=positions.expand(kv_heads, -1),
)
query = torch.randn(1, heads, dim, device=device, dtype=dtype)
normal_key = torch.randn(1, kv_heads, dim, device=device, dtype=dtype)
normal_value = torch.randn(1, kv_heads, dim, device=device, dtype=dtype)
normal_positions = torch.tensor([args.history], device=device)
query_positions = normal_positions


def synchronize():
    if args.device == "cuda":
        torch.cuda.synchronize()
    elif args.device == "npu":
        torch.npu.synchronize()


def run(fast):
    return module.reference_sdpa(
        query, history, normal_key, normal_value, normal_positions,
        query_positions, scale=dim**-0.5, validate_history=False,
        decode_causal=fast,
    )


with torch.inference_mode():
    reference = run(False)
    optimized = run(True)
    synchronize()
    error = (reference.float() - optimized.float()).abs().max().item()
    results = {}
    for name, fast in (("masked", False), ("decode_causal", True)):
        for _ in range(4):
            run(fast)
        synchronize()
        start = time.perf_counter()
        for _ in range(args.iters):
            run(fast)
        synchronize()
        results[name] = (time.perf_counter() - start) * 1000 / args.iters

print(json.dumps({
    "device": args.device, "torch_version": torch.__version__,
    "history": args.history, "iters": args.iters,
    "max_abs_error": error, "ms_per_decode_layer": results,
}, sort_keys=True))
