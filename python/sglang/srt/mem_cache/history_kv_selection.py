"""Pure-Torch selectors for compressed history KV at a reuse boundary.

The selectors operate on native KV heads.  For GQA, attention contributed by
the query heads in a KV group is summed before selecting that KV head's paired
K/V rows.  Stored keys are expected to retain their original RoPE phase.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F


HEADWISE_HISTORY_KV_METHODS = frozenset(
    {"h2o", "snapkv_persistent", "snapkv_refresh"}
)
DEFAULT_SCORE_QUERY_CHUNK_SIZE = 64
DEFAULT_STREAMINGLLM_SINK_TOKENS = 4


def dense_headwise_recovery_indices(layer_indices, recovery_indices, *, seq_len):
    """Restore all target tokens, retaining each head's original selection.

    Different overlap with the target gives different union lengths. Existing
    dense storage needs one common length: fill shorter heads with additional
    real source tokens (most recent first), never duplicate or zero-pad K/V.
    This is explicitly a superset restore, not target-only recovery.
    """
    if not layer_indices or seq_len < 1:
        raise ValueError("non-empty layers and positive seq_len required")
    target = set(int(i) for i in recovery_indices)
    if not target or min(target) < 0 or max(target) >= seq_len:
        raise ValueError("non-empty recovery indices within source span required")
    shape = tuple(layer_indices[0].shape)
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("selection must have shape [Hkv, K]")
    unions, missing = [], []
    for indices in layer_indices:
        if tuple(indices.shape) != shape:
            raise ValueError("all layers require equal [Hkv, K] shapes")
        heads = []
        for head in indices.tolist():
            retained = set(int(i) for i in head)
            if len(retained) != shape[1] or min(retained) < 0 or max(retained) >= seq_len:
                raise ValueError("retained indices must be unique and within source span")
            heads.append(retained | target)
            missing.append(len(target - retained))
        unions.append(heads)
    length = max(len(head) for layer in unions for head in layer)
    completed, extras = [], []
    for indices, heads in zip(layer_indices, unions):
        rows = []
        for head in heads:
            extra = length - len(head)
            extras.append(extra)
            if extra:
                for i in range(seq_len - 1, -1, -1):
                    if i not in head:
                        head.add(i)
                        if len(head) == length:
                            break
            rows.append(sorted(head))
        completed.append(torch.tensor(rows, dtype=torch.long, device=indices.device))
    return completed, {
        "before_recovery_active_tokens": shape[1],
        "after_recovery_active_tokens": length,
        "recovered_segment_size": len(target),
        "restored_raw_token_count": length - shape[1],
        "target_restored_tokens_per_head_mean": sum(missing) / len(missing),
        "dense_alignment_extra_tokens_per_head_mean": sum(extras) / len(extras),
        "dense_alignment_extra_tokens_per_head_max": max(extras),
        "duplicate_raw_token_count": 0,
        "recovery_target_coverage": 1.0,
    }


def deduplicated_recovery_indices(
    retained_indices: Sequence[int],
    recovery_indices: Sequence[int],
    *,
    seq_len: int,
) -> tuple[list[int], dict[str, int]]:
    """Union a common token selection with an exact raw recovery segment.

    This helper deliberately accepts one common source-index vector only.
    Headwise methods must not collapse their per-head vectors before calling
    it, because that would silently create duplicate or wrongly positioned KV.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be positive")
    retained = sorted({int(i) for i in retained_indices})
    recovery = sorted({int(i) for i in recovery_indices})
    if any(i < 0 or i >= seq_len for i in retained + recovery):
        raise ValueError("history recovery indices are outside the source span")
    if not recovery:
        raise ValueError("history recovery indices must be non-empty")
    merged = sorted(set(retained).union(recovery))
    return merged, {
        "before_recovery_active_tokens": len(retained),
        "after_recovery_active_tokens": len(merged),
        "recovered_segment_size": len(recovery),
        "restored_raw_token_count": len(set(recovery) - set(retained)),
        "duplicate_raw_token_count": 0,
    }


def require_rotated_headwise_storage(method: str, position_mode: str) -> None:
    """Reject a shared-position storage form for headwise token origins."""

    if method in HEADWISE_HISTORY_KV_METHODS and position_mode != "rotated":
        raise ValueError(
            f"history_kv_method={method!r} selects different source positions "
            "per layer/KV head and therefore requires "
            "raw_kv_position_mode='rotated'; a shared pre-RoPE position vector "
            "cannot preserve those phases."
        )


def attention_scores_by_kv_head(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    *,
    scale: float,
    query_start: int,
    query_end: int,
    key_start: int,
    key_end: int,
    query_chunk_size: int = DEFAULT_SCORE_QUERY_CHUNK_SIZE,
) -> torch.Tensor:
    """Accumulate causal attention probabilities as ``[Hkv, selected_keys]``.

    ``query_states`` is ``[B, Hq, L, D]`` and ``key_states`` is
    ``[B, Hkv, L, D]``.  The contiguous query heads belonging to each native KV
    head are summed.  Query chunks bound temporary attention storage to
    ``O(B * Hq * query_chunk_size * L)``.
    """

    if query_states.ndim != 4 or key_states.ndim != 4:
        raise ValueError("query_states and key_states must both be rank-4 tensors")
    batch, num_query_heads, seq_len, head_dim = query_states.shape
    key_batch, num_kv_heads, key_len, key_head_dim = key_states.shape
    if (key_batch, key_len, key_head_dim) != (batch, seq_len, head_dim):
        raise ValueError(
            "query/key shape mismatch: "
            f"{tuple(query_states.shape)} vs {tuple(key_states.shape)}"
        )
    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            f"GQA head mismatch: {num_query_heads=} is not divisible by {num_kv_heads=}"
        )
    if not (0 <= query_start < query_end <= seq_len):
        raise ValueError(
            f"invalid query range [{query_start}, {query_end}) for length {seq_len}"
        )
    if not (0 <= key_start < key_end <= key_len):
        raise ValueError(
            f"invalid key range [{key_start}, {key_end}) for length {key_len}"
        )
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")

    group_size = num_query_heads // num_kv_heads
    selected_key_count = key_end - key_start
    result = torch.zeros(
        (num_kv_heads, selected_key_count),
        dtype=torch.float32,
        device=query_states.device,
    )
    key_grouped = key_states.float().unsqueeze(2)
    key_positions = torch.arange(key_len, device=query_states.device).view(
        1, 1, 1, 1, key_len
    )

    for chunk_start in range(query_start, query_end, query_chunk_size):
        chunk_end = min(query_end, chunk_start + query_chunk_size)
        query_chunk = query_states[:, :, chunk_start:chunk_end, :]
        query_grouped = query_chunk.reshape(
            batch,
            num_kv_heads,
            group_size,
            chunk_end - chunk_start,
            head_dim,
        ).float()
        logits = torch.matmul(
            query_grouped,
            key_grouped.transpose(-2, -1),
        ) * float(scale)
        query_positions = torch.arange(
            chunk_start, chunk_end, device=query_states.device
        ).view(1, 1, 1, -1, 1)
        logits = logits.masked_fill(key_positions > query_positions, float("-inf"))
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
        result.add_(
            probabilities[..., key_start:key_end].sum(dim=(0, 2, 3))
        )

    if not torch.isfinite(result).all():
        raise ValueError("history KV attention scoring produced non-finite values")
    return result


def _validate_scores_and_budget(
    scores: torch.Tensor, target_tokens: int
) -> tuple[int, int]:
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape [Hkv, L], got {tuple(scores.shape)}")
    num_heads, seq_len = scores.shape
    if num_heads <= 0 or seq_len <= 0:
        raise ValueError(f"scores must be non-empty, got {tuple(scores.shape)}")
    if not (1 <= target_tokens <= seq_len):
        raise ValueError(
            f"target_tokens must be in [1, {seq_len}], got {target_tokens}"
        )
    if not torch.isfinite(scores).all():
        raise ValueError("history KV selection received non-finite scores")
    return num_heads, seq_len


def _pool_snapkv_scores(
    scores: torch.Tensor, kernel_size: int, pooling: str
) -> torch.Tensor:
    if kernel_size <= 0:
        raise ValueError(f"kernel_size must be positive, got {kernel_size}")
    pooling = pooling.strip().lower()
    if pooling not in {"avgpool", "maxpool"}:
        raise ValueError(
            f"SnapKV pooling must be 'avgpool' or 'maxpool', got {pooling!r}"
        )
    if kernel_size == 1 or scores.shape[-1] <= 1:
        return scores
    pool = F.avg_pool1d if pooling == "avgpool" else F.max_pool1d
    pooled = pool(
        scores.unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    ).squeeze(0)
    return pooled[..., : scores.shape[-1]]


def pool_snapkv_scores_by_position(
    scores: torch.Tensor,
    canonical_positions: Sequence[int],
    kernel_size: int,
    pooling: str,
) -> torch.Tensor:
    """Pool SnapKV scores without bridging gaps left by persistent eviction."""
    positions = [int(position) for position in canonical_positions]
    if len(positions) != scores.shape[-1]:
        raise ValueError(
            "canonical_positions must match the SnapKV score length, got "
            f"{len(positions)} positions for {scores.shape[-1]} scores"
        )
    if any(right <= left for left, right in zip(positions, positions[1:])):
        raise ValueError("canonical_positions must be strictly increasing")
    if kernel_size <= 0:
        raise ValueError(f"kernel_size must be positive, got {kernel_size}")
    pooling = pooling.strip().lower()
    if pooling not in {"avgpool", "maxpool"}:
        raise ValueError(
            f"SnapKV pooling must be 'avgpool' or 'maxpool', got {pooling!r}"
        )
    if kernel_size == 1 or scores.shape[-1] <= 1:
        return scores
    if all(right == left + 1 for left, right in zip(positions, positions[1:])):
        return _pool_snapkv_scores(scores, kernel_size, pooling)

    position_to_index = {position: index for index, position in enumerate(positions)}
    padding = kernel_size // 2
    offsets = range(-padding, kernel_size - padding)
    if pooling == "avgpool":
        pooled = torch.zeros_like(scores)
    else:
        pooled = torch.full_like(scores, float("-inf"))

    for offset in offsets:
        source_indices = [
            position_to_index.get(position + offset, -1) for position in positions
        ]
        present = torch.tensor(
            [index >= 0 for index in source_indices],
            dtype=torch.bool,
            device=scores.device,
        )
        safe_indices = torch.tensor(
            [max(index, 0) for index in source_indices],
            dtype=torch.long,
            device=scores.device,
        )
        values = scores.index_select(-1, safe_indices)
        if pooling == "avgpool":
            pooled = pooled + torch.where(present, values, torch.zeros_like(values))
        else:
            pooled = torch.maximum(
                pooled,
                torch.where(
                    present,
                    values,
                    torch.full_like(values, float("-inf")),
                ),
            )

    return pooled / kernel_size if pooling == "avgpool" else pooled


def select_snapkv_indices(
    scores: torch.Tensor,
    *,
    target_tokens: int,
    recent_window: int,
    kernel_size: int,
    pooling: str,
) -> torch.Tensor:
    """Select paired K/V positions per native KV head for SnapKV prefill."""

    num_heads, seq_len = _validate_scores_and_budget(scores, target_tokens)
    if recent_window <= 0:
        raise ValueError(f"recent_window must be positive, got {recent_window}")
    if kernel_size <= 0:
        raise ValueError(f"kernel_size must be positive, got {kernel_size}")
    pooling = pooling.strip().lower()
    if pooling not in {"avgpool", "maxpool"}:
        raise ValueError(
            f"SnapKV pooling must be 'avgpool' or 'maxpool', got {pooling!r}"
        )
    recent_tokens = min(target_tokens, recent_window, seq_len)
    past_tokens = target_tokens - recent_tokens
    recent = torch.arange(
        seq_len - recent_tokens, seq_len, dtype=torch.long, device=scores.device
    ).expand(num_heads, -1)
    if past_tokens == 0:
        return recent.contiguous()

    candidate_scores = _pool_snapkv_scores(
        scores[:, : seq_len - recent_tokens], kernel_size, pooling
    )
    past = torch.topk(candidate_scores, k=past_tokens, dim=-1).indices
    past = past.sort(dim=-1).values
    return torch.cat([past, recent], dim=-1).contiguous()


def select_h2o_prefill_indices(
    scores: torch.Tensor,
    *,
    target_tokens: int,
    recent_fraction: float,
) -> torch.Tensor:
    """Apply H2O's heavy-hitter/recent split to accumulated prefill scores."""

    num_heads, seq_len = _validate_scores_and_budget(scores, target_tokens)
    if not (0.0 <= recent_fraction <= 1.0):
        raise ValueError(
            f"recent_fraction must be in [0, 1], got {recent_fraction}"
        )
    recent_tokens = max(
        1,
        min(target_tokens, int(round(target_tokens * recent_fraction))),
    )
    heavy_tokens = target_tokens - recent_tokens
    recent = torch.arange(
        seq_len - recent_tokens, seq_len, dtype=torch.long, device=scores.device
    ).expand(num_heads, -1)
    if heavy_tokens == 0:
        return recent.contiguous()

    heavy = torch.topk(
        scores[:, : seq_len - recent_tokens], k=heavy_tokens, dim=-1
    ).indices
    heavy = heavy.sort(dim=-1).values
    return torch.cat([heavy, recent], dim=-1).contiguous()


def select_streamingllm_indices(
    seq_len: int,
    *,
    target_tokens: int,
    sink_tokens: int = DEFAULT_STREAMINGLLM_SINK_TOKENS,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Keep attention sinks plus a recent suffix, reserving the final token."""

    if not (1 <= target_tokens <= seq_len):
        raise ValueError(
            f"target_tokens must be in [1, {seq_len}], got {target_tokens}"
        )
    if sink_tokens < 0:
        raise ValueError(f"sink_tokens must be non-negative, got {sink_tokens}")
    kept_sinks = min(sink_tokens, max(0, target_tokens - 1), max(0, seq_len - 1))
    kept_recent = target_tokens - kept_sinks
    sinks = torch.arange(kept_sinks, dtype=torch.long, device=device)
    recent = torch.arange(
        seq_len - kept_recent, seq_len, dtype=torch.long, device=device
    )
    return torch.cat([sinks, recent]).contiguous()


def gather_paired_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    selected_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather matching K/V rows into shared physical slots per native KV head."""

    if key.ndim != 2 or value.ndim != 2 or selected_indices.ndim != 2:
        raise ValueError("key/value must be rank 2 and selected_indices must be rank 2")
    num_heads, target_tokens = selected_indices.shape
    if num_heads <= 0 or target_tokens <= 0:
        raise ValueError("selected_indices must be non-empty")
    if key.shape[0] != value.shape[0]:
        raise ValueError(
            f"K/V token length mismatch: {key.shape[0]} != {value.shape[0]}"
        )
    if key.shape[1] % num_heads or value.shape[1] % num_heads:
        raise ValueError(
            "flattened K/V dimensions must be divisible by the number of KV heads"
        )
    if selected_indices.dtype != torch.long:
        raise ValueError("selected_indices must use torch.long")
    if selected_indices.device != key.device or value.device != key.device:
        raise ValueError("key, value, and selected_indices must use the same device")
    if int(selected_indices.min()) < 0 or int(selected_indices.max()) >= key.shape[0]:
        raise ValueError("selected_indices contain an out-of-range token position")

    key_dim = key.shape[1] // num_heads
    value_dim = value.shape[1] // num_heads
    key_by_head = key.view(key.shape[0], num_heads, key_dim).transpose(0, 1)
    value_by_head = value.view(value.shape[0], num_heads, value_dim).transpose(0, 1)
    gathered_key = torch.gather(
        key_by_head,
        1,
        selected_indices.unsqueeze(-1).expand(-1, -1, key_dim),
    )
    gathered_value = torch.gather(
        value_by_head,
        1,
        selected_indices.unsqueeze(-1).expand(-1, -1, value_dim),
    )
    return (
        gathered_key.transpose(0, 1).reshape(target_tokens, -1).contiguous(),
        gathered_value.transpose(0, 1).reshape(target_tokens, -1).contiguous(),
    )


def summarize_headwise_indices(
    layer_indices: Sequence[torch.Tensor], preview_tokens: int = 4
) -> Dict[str, object]:
    """Return bounded audit metadata without claiming one shared token set."""

    if not layer_indices:
        raise ValueError("layer_indices must be non-empty")
    if preview_tokens <= 0:
        raise ValueError(f"preview_tokens must be positive, got {preview_tokens}")
    expected_shape = tuple(layer_indices[0].shape)
    if len(expected_shape) != 2:
        raise ValueError("every layer index tensor must have shape [Hkv, K]")
    previews: List[Dict[str, object]] = []
    for layer_index, indices in enumerate(layer_indices):
        if tuple(indices.shape) != expected_shape:
            raise ValueError(
                f"layer {layer_index} index shape {tuple(indices.shape)} "
                f"!= {expected_shape}"
            )
        preview_count = min(preview_tokens, indices.shape[1])
        previews.append(
            {
                "layer": layer_index,
                "head_prefix": indices[:, :preview_count].detach().cpu().tolist(),
                "head_suffix": indices[:, -preview_count:].detach().cpu().tolist(),
            }
        )
    return {
        "selection_index_shape": [len(layer_indices), *expected_shape],
        "selection_indices_preview": previews,
        "selection_indices_preview_tokens": min(preview_tokens, expected_shape[1]),
    }
