"""Plan schema-only KV selection while preserving the rest of a chat prompt."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F


TOOL_KV_METHODS = frozenset({"h2o", "snapkv", "streamingllm", "pyramidkv"})


def plan_tool_kv_eviction(config: Mapping, prompt_tokens: int) -> dict:
    method = str(config.get("method") or "").lower()
    if method not in TOOL_KV_METHODS:
        raise ValueError("TOOL_KV_METHOD_UNSUPPORTED")
    if prompt_tokens < 2 or int(config.get("full_prompt_tokens") or -1) != prompt_tokens:
        raise ValueError("TOOL_KV_PROMPT_LENGTH_MISMATCH")
    spans = config.get("resolved_schema_token_spans")
    raw_spans = config.get("schema_spans") or []
    if not isinstance(spans, list) or (not spans and not raw_spans):
        raise ValueError("TOOL_KV_RESOLVED_SCHEMA_SPANS_REQUIRED")
    protected = {int(index) for index in config.get("protected_schema_indices") or []}
    # A short prose value may have no wholly contained token, while its
    # protected catalog index is still valid in the source annotations.
    available = ({int(item["schema_index"]) for item in spans}
                 | {int(item["schema_index"]) for item in raw_spans})
    if protected - available:
        raise ValueError("TOOL_KV_PROTECTED_SCHEMA_UNKNOWN")
    protected_interface = set()
    for item in config.get("resolved_protected_interface_token_spans") or []:
        start, end = int(item["token_start"]), int(item["token_end"])
        if not 0 <= start < end <= prompt_tokens:
            raise ValueError("TOOL_KV_INTERFACE_SPAN_INVALID")
        protected_interface.update(range(start, end))
    evictable = set()
    all_schema = set()
    for item in spans:
        start, end = int(item["token_start"]), int(item["token_end"])
        if not 0 <= start < end < prompt_tokens:
            raise ValueError("TOOL_KV_SCHEMA_OUTSIDE_PREFILL")
        indices = set(range(start, end))
        if all_schema.intersection(indices):
            raise ValueError("TOOL_KV_SCHEMA_TOKEN_SPANS_OVERLAP")
        all_schema.update(indices)
        if int(item["schema_index"]) not in protected:
            evictable.update(indices)
    evictable.difference_update(protected_interface)
    evictable = sorted(evictable)
    evictable_count = len(evictable)
    mandatory = prompt_tokens - evictable_count
    total_target = config.get("target_resident_tokens_per_layer")
    direct_target = config.get("target_evictable_tokens_per_layer")
    if total_target is None and direct_target is None:
        raise ValueError("TOOL_KV_TARGET_REQUIRED")
    requested_keep = int(total_target) - mandatory if total_target is not None else int(direct_target)
    if direct_target is not None and total_target is not None and int(direct_target) != requested_keep:
        raise ValueError("TOOL_KV_TARGETS_DISAGREE")
    keep = min(max(0, requested_keep), evictable_count)
    protocol = config.get("resolved_tool_protocol_token_span")
    tool_scope = all_schema | protected_interface
    if protocol is not None:
        protocol_start = int(protocol["token_start"])
        protocol_end = int(protocol["token_end"])
        if not 0 <= protocol_start < protocol_end <= prompt_tokens:
            raise ValueError("TOOL_KV_PROTOCOL_SPAN_INVALID")
        tool_scope.update(range(protocol_start, protocol_end))
    tool_resident = len(tool_scope) - evictable_count + keep
    cap = config.get("max_resident_tool_tokens")
    if cap is not None:
        if tool_resident > int(cap):
            raise ValueError("TOOL_KV_TOOL_BUDGET_EXCEEDED")
    history_end = prompt_tokens - 1
    evictable_set = set(evictable)
    protected_indices = [index for index in range(history_end) if index not in evictable_set]
    recent_window = max(1, int(config.get("recent_window") or 64))
    query_start = 0 if method == "h2o" else max(0, history_end - recent_window)
    return {
        "method": "snapkv_persistent" if method == "snapkv" else method,
        "tool_kv_eviction": True,
        "history_start": 0,
        "history_end": history_end,
        "protected_history_indices": protected_indices,
        "tool_evictable_indices": evictable,
        "tool_keep_tokens": keep,
        "target_tokens": len(protected_indices) + keep,
        "selection_query_start": query_start,
        "selection_query_end": history_end,
        "selection_query_tokens": history_end - query_start,
        "selection_query_phase": "prompt_prefill_before_last_token_replay",
        "history_kv_recent_window": recent_window,
        "history_kv_kernel_size": max(1, int(config.get("kernel_size") or 5)),
        "history_kv_pooling": str(config.get("pooling") or "avgpool").lower(),
        "h2o_recent_fraction": float(config.get("h2o_recent_fraction") or 0.5),
        "streamingllm_sink_tokens": max(0, int(config.get("sink_tokens") or 4)),
        "tool_full_prompt_tokens": prompt_tokens,
        "tool_resident_tokens": mandatory + keep,
        "tool_protocol_resident_tokens": tool_resident,
        "tool_scope_full_tokens": len(tool_scope),
        "tool_protected_interface_tokens": len(protected_interface),
        "tool_max_resident_tokens": int(cap) if cap is not None else None,
        "tool_no_op": keep == evictable_count,
        "tool_budget_status": (
            "exceeds_allowance" if total_target is not None
            and mandatory + keep > int(total_target) else "within_allowance"
        ),
    }


def select_tool_h2o(scores: torch.Tensor, evictable: Sequence[int], keep: int,
                    recent_fraction: float) -> torch.Tensor:
    """Full-prompt heavy hitters plus floor(keep*fraction) recent schema tokens."""
    candidate = scores[:, list(evictable)].float()
    if not 0 <= recent_fraction <= 1:
        raise ValueError("TOOL_KV_H2O_RECENT_FRACTION_INVALID")
    recent = int(keep * recent_fraction)
    heavy = keep - recent
    suffix = torch.arange(len(evictable) - recent, len(evictable),
                          device=scores.device, dtype=torch.long).expand(scores.shape[0], -1)
    if heavy:
        prefix = torch.topk(candidate[:, :len(evictable) - recent], heavy, dim=-1).indices
        indices = torch.cat([prefix, suffix], dim=-1)
    else:
        indices = suffix
    return indices.sort(dim=-1).values


def select_tool_snapkv(scores: torch.Tensor, evictable: Sequence[int], keep: int,
                       kernel_size: int) -> torch.Tensor:
    """Pool in original prompt positions, then select only schema columns."""
    if kernel_size < 1 or kernel_size % 2 != 1:
        raise ValueError("TOOL_KV_SNAPKV_KERNEL_INVALID")
    pooled = F.max_pool1d(scores.float().unsqueeze(0), kernel_size=kernel_size,
                          stride=1, padding=kernel_size // 2).squeeze(0)
    candidates = pooled[:, list(evictable)]
    return torch.topk(candidates, keep, dim=-1).indices.sort(dim=-1).values
