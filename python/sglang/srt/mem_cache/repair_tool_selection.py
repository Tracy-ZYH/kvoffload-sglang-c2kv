"""Select sparse schema KV within one contiguous native repair interval."""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from sglang.srt.mem_cache.history_kv_reference import select_pyramidkv_headwise
from sglang.srt.mem_cache.history_kv_selection import select_streamingllm_indices
from sglang.srt.mem_cache.tool_kv_eviction import select_tool_h2o, select_tool_snapkv


SPARSE_REPAIR_METHODS = frozenset(
    {"streamingllm", "h2o", "snapkv_persistent", "pyramidkv"}
)


def validate_sparse_repair_partition(
    span_tokens: int,
    selectable: Optional[Sequence[int]],
    mandatory: Optional[Sequence[int]],
    target_tokens: Optional[int],
) -> tuple[list[int], list[int]]:
    """Require an exhaustive partition so no protocol token can be dropped."""

    if selectable is None:
        if mandatory is not None:
            raise ValueError("mandatory indices require selectable indices")
        return [], []
    candidates = [int(index) for index in selectable]
    if not candidates or candidates != sorted(set(candidates)):
        raise ValueError("selectable indices must be non-empty, sorted, and unique")
    if candidates[0] < 0 or candidates[-1] >= span_tokens:
        raise ValueError("selectable indices are outside the repair span")
    candidate_set = set(candidates)
    complement = [index for index in range(span_tokens) if index not in candidate_set]
    if mandatory is not None and [int(index) for index in mandatory] != complement:
        raise ValueError("mandatory indices must be the exact selectable complement")
    if target_tokens is not None and not (
        max(1, len(complement)) <= int(target_tokens) <= span_tokens
    ):
        raise ValueError(
            "history_kv_target_tokens must retain every mandatory token "
            "and cannot exceed the repair span"
        )
    return candidates, complement


def select_sparse_repair_indices(
    method: str,
    scores_by_layer: Sequence[torch.Tensor],
    selectable: Sequence[int],
    mandatory: Sequence[int],
    target_tokens: int,
    *,
    recent_window: int = 64,
    kernel_size: int = 5,
    pooling: str = "avgpool",
    h2o_recent_fraction: float = 0.5,
    num_layers: int = 1,
    device: Optional[torch.device] = None,
) -> tuple[list[torch.Tensor], dict]:
    """Return full-span relative positions, selecting schema tokens globally.

    Headwise methods may choose different schema tokens for each KV head, but
    every head retains the same mandatory protocol tokens in original order.
    """

    if method not in SPARSE_REPAIR_METHODS:
        raise ValueError(f"unsupported sparse repair method: {method}")
    candidates = [int(index) for index in selectable]
    protected = [int(index) for index in mandatory]
    candidate_keep = target_tokens - len(protected)
    if not 0 <= candidate_keep <= len(candidates):
        raise ValueError("sparse repair target does not fit selectable tokens")
    if num_layers < 1:
        raise ValueError("num_layers must be positive")
    layer_count = len(scores_by_layer) if scores_by_layer else num_layers
    score_device = scores_by_layer[0].device if scores_by_layer else device
    if score_device is None:
        score_device = torch.device("cpu")
    candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=score_device)
    protected_tensor = torch.tensor(protected, dtype=torch.long, device=score_device)

    if method == "streamingllm" or candidate_keep in {0, len(candidates)}:
        if candidate_keep == 0:
            chosen = candidate_tensor[:0]
        elif candidate_keep == len(candidates):
            chosen = candidate_tensor
        else:
            indices = select_streamingllm_indices(
                len(candidates), target_tokens=candidate_keep, device=score_device
            )
            chosen = candidate_tensor[indices]
        common = torch.cat((protected_tensor, chosen)).sort().values
        selected = [common.unsqueeze(0) for _ in range(layer_count)]
        return selected, {
            "algorithm_version": "sparse_tool_repair_v1",
            "per_head_selection": False,
            "selected_relative_indices": common.tolist(),
            "candidate_tokens": len(candidates),
            "mandatory_tokens": len(protected),
            "candidate_kept_tokens": candidate_keep,
            "target_tokens": target_tokens,
            "budget_preserved": True,
        }

    if len(scores_by_layer) != num_layers:
        raise ValueError("one score tensor per layer is required")
    span_tokens = len(candidates) + len(protected)
    head_count = scores_by_layer[0].shape[0]
    if any(
        scores.ndim != 2 or tuple(scores.shape) != (head_count, span_tokens)
        for scores in scores_by_layer
    ):
        raise ValueError("score tensors must cover every token in the repair span")
    if recent_window < 1 or kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError("recent_window and odd kernel_size must be positive")
    if pooling not in {"avgpool", "maxpool"}:
        raise ValueError("pooling must be avgpool or maxpool")

    if method == "pyramidkv":
        chosen_by_layer, pyramid_metadata = select_pyramidkv_headwise(
            [scores[:, candidate_tensor] for scores in scores_by_layer],
            target_tokens=candidate_keep,
            capacity_history_tokens=len(candidates),
            recent_window=recent_window,
            kernel_size=kernel_size,
            pooling=pooling,
        )
        # Native repair storage has one dense token count for all layers. Keep
        # each layer's PyramidKV choices and fill shorter rows with real schema
        # tokens ranked by that layer/head's score; never duplicate a KV row.
        width = max(indices.shape[1] for indices in chosen_by_layer)
        aligned = []
        for scores, indices in zip(scores_by_layer, chosen_by_layer):
            rows = []
            for head, row in enumerate(indices.tolist()):
                chosen = set(row)
                if len(chosen) < width:
                    ranked = torch.argsort(
                        scores[head, candidate_tensor], descending=True
                    ).tolist()
                    for index in ranked:
                        chosen.add(int(index))
                        if len(chosen) == width:
                            break
                rows.append(sorted(chosen))
            aligned.append(torch.tensor(rows, dtype=torch.long, device=score_device))
        chosen_by_layer = aligned
        extra_metadata = {
            **pyramid_metadata,
            "dense_native_alignment_candidate_tokens": width - candidate_keep,
            "native_repair_shared_length_approximation": True,
        }
    else:
        chosen_by_layer = []
        for scores in scores_by_layer:
            if method == "h2o":
                indices = select_tool_h2o(
                    scores, candidates, candidate_keep, h2o_recent_fraction
                )
            else:
                indices = select_tool_snapkv(
                    scores, candidates, candidate_keep, kernel_size
                )
            chosen_by_layer.append(indices)
        extra_metadata = {
            "recent_candidate_tokens": (
                int(candidate_keep * h2o_recent_fraction) if method == "h2o" else None
            ),
            "snapkv_pooling": "maxpool" if method == "snapkv_persistent" else None,
        }

    selected = []
    for candidate_indices in chosen_by_layer:
        source_indices = candidate_tensor[candidate_indices]
        mandatory_indices = protected_tensor.expand(head_count, -1)
        selected.append(torch.cat((source_indices, mandatory_indices), dim=-1).sort(dim=-1).values)
    actual = selected[0].shape[1]
    if any(indices.shape != selected[0].shape for indices in selected):
        raise RuntimeError("sparse repair selection has unequal layer shapes")
    return selected, {
        "algorithm_version": "sparse_tool_repair_v1",
        "per_head_selection": True,
        "selected_relative_indices": None,
        "candidate_tokens": len(candidates),
        "mandatory_tokens": len(protected),
        "candidate_kept_tokens": actual - len(protected),
        "physical_per_layer_tokens": [actual] * layer_count,
        "target_tokens": target_tokens,
        "budget_preserved": actual == target_tokens,
        **extra_metadata,
    }
