#!/usr/bin/env python3
"""Compare C2KV dense attention with Ascend NPU attention kernels.

This script intentionally uses a small synthetic problem by default. It builds
the same C2KV keep-mask semantics used by gist extraction:
  input -> input: causal
  input -> gist: masked
  gist  -> input: own chunk + sink tokens
  gist  -> gist: causal

The dense reference uses attention_mask == True as "keep". Ascend kernels use
atten_mask == True / 1 as "mask", so the script passes ~attention_mask.
"""

from __future__ import annotations

import argparse
import math
from typing import Any

import torch


def build_c2kv_keep_mask(
    seq_len: int,
    gist_len: int,
    ratio: int,
    gist_overlap: int,
    device: torch.device,
) -> torch.Tensor:
    total_len = seq_len + gist_len
    idx = torch.arange(total_len, device=device, dtype=torch.long)
    q_idx = idx[:, None]
    kv_idx = idx[None, :]

    is_q_input = q_idx < seq_len
    is_kv_input = kv_idx < seq_len

    input_to_input = is_q_input & is_kv_input & (q_idx >= kv_idx)

    gist_j = q_idx - seq_len
    chunk_begin = gist_j * ratio - gist_overlap
    chunk_end = (gist_j + 1) * ratio
    gist_to_input = (~is_q_input) & is_kv_input & (
        ((kv_idx >= chunk_begin) & (kv_idx < chunk_end)) | (kv_idx < ratio)
    )

    gist_to_gist = (~is_q_input) & (~is_kv_input) & (q_idx >= kv_idx)

    return (input_to_input | gist_to_input | gist_to_gist).unsqueeze(0).unsqueeze(0)


def repeat_kv(x: torch.Tensor, num_heads: int, num_kv_heads: int) -> torch.Tensor:
    if num_heads == num_kv_heads:
        return x
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"Invalid GQA heads: {num_heads=} {num_kv_heads=}")
    return x.repeat_interleave(num_heads // num_kv_heads, dim=1)


def dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    keep_mask: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    k_attn = repeat_kv(k, num_heads, num_kv_heads)
    v_attn = repeat_kv(v, num_heads, num_kv_heads)
    scores = torch.matmul(q.float(), k_attn.transpose(-2, -1).float()) * scale
    scores = scores.masked_fill(~keep_mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(v_attn.dtype)
    return torch.matmul(probs, v_attn)


def unwrap_attention_output(output: Any, expected_shape: torch.Size) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if not isinstance(output, tuple):
        raise RuntimeError(f"Unexpected output type: {type(output)!r}")
    candidates = [
        item
        for item in output
        if isinstance(item, torch.Tensor) and tuple(item.shape) == tuple(expected_shape)
    ]
    if candidates:
        return candidates[0]
    candidates = [item for item in output if isinstance(item, torch.Tensor) and item.dim() == 4]
    if candidates:
        return candidates[0]
    raise RuntimeError("No 4-D attention output tensor found.")


def npu_prompt_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    npu_mask: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    import torch_npu

    output = torch_npu.npu_prompt_flash_attention(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        num_heads=num_heads,
        input_layout="BNSD",
        atten_mask=npu_mask.contiguous(),
        scale_value=scale,
        num_key_value_heads=num_kv_heads,
        sparse_mode=0,
    )
    return unwrap_attention_output(output, q.shape)


def npu_fusion_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    npu_mask: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
) -> torch.Tensor:
    import torch_npu

    k_attn = repeat_kv(k, num_heads, num_kv_heads).contiguous()
    v_attn = repeat_kv(v, num_heads, num_kv_heads).contiguous()
    output = torch_npu.npu_fusion_attention(
        q.contiguous(),
        k_attn,
        v_attn,
        num_heads,
        input_layout="BNSD",
        atten_mask=npu_mask.contiguous(),
        scale=scale,
        keep_prob=1.0,
        sparse_mode=0,
    )
    return unwrap_attention_output(output, q.shape)


def compare(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_f = actual.float().cpu()
    expected_f = expected.float().cpu()
    diff = (actual_f - expected_f).abs()
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.flatten(), expected_f.flatten(), dim=0
    )
    print(
        f"{name}: "
        f"max_abs_error={diff.max().item():.6g}, "
        f"mean_abs_error={diff.mean().item():.6g}, "
        f"cosine_similarity={cosine.item():.8f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--gist-len", type=int, default=None)
    parser.add_argument("--ratio", type=int, default=16)
    parser.add_argument("--gist-overlap", type=int, default=0)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-pfa", action="store_true")
    parser.add_argument("--skip-fusion", action="store_true")
    args = parser.parse_args()

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device)
    gist_len = args.gist_len or math.ceil(args.seq_len / args.ratio)
    total_len = args.seq_len + gist_len
    scale = args.head_dim**-0.5

    torch.manual_seed(args.seed)
    q = torch.randn(
        1, args.num_heads, total_len, args.head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        1, args.num_kv_heads, total_len, args.head_dim, device=device, dtype=dtype
    )
    v = torch.randn(
        1, args.num_kv_heads, total_len, args.head_dim, device=device, dtype=dtype
    )

    keep_mask = build_c2kv_keep_mask(
        args.seq_len, gist_len, args.ratio, args.gist_overlap, device
    )
    npu_mask = ~keep_mask

    print(
        f"shape: q={tuple(q.shape)}, k={tuple(k.shape)}, "
        f"mask={tuple(keep_mask.shape)}, keep={keep_mask.sum().item()}"
    )

    ref = dense_attention(
        q, k, v, keep_mask, scale, args.num_heads, args.num_kv_heads
    )
    compare("dense_self_check", ref, ref)

    if not args.skip_pfa:
        try:
            pfa = npu_prompt_flash_attention(
                q, k, v, npu_mask, scale, args.num_heads, args.num_kv_heads
            )
            compare("npu_prompt_flash_attention", pfa, ref)
        except Exception as exc:
            print(f"npu_prompt_flash_attention failed: {type(exc).__name__}: {exc}")

    if not args.skip_fusion:
        try:
            fusion = npu_fusion_attention(
                q, k, v, npu_mask, scale, args.num_heads, args.num_kv_heads
            )
            compare("npu_fusion_attention", fusion, ref)
        except Exception as exc:
            print(f"npu_fusion_attention failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
