from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch


@dataclass
class HistoryKVEvictionResult:
    success: bool
    error: str = ""
    method: str = ""
    runtime_status: str = ""
    old_physical_kv_slots: int = 0
    new_physical_kv_slots: int = 0
    freed_physical_slots: int = 0
    allocator_free_before: int | None = None
    allocator_free_after: int | None = None
    protected_prefix_tokens: int = 0
    history_tokens: int = 0
    kept_history_tokens: int = 0
    current_tokens: int = 0
    next_rope_position_before: int = 0
    next_rope_position_after: int = 0
    selected_history_indices: list[int] | None = None
    bytes_per_kv_token: int | None = None
    freed_kv_bytes: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "error": self.error,
            "method": self.method,
            "runtime_status": self.runtime_status,
            "old_physical_kv_slots": self.old_physical_kv_slots,
            "new_physical_kv_slots": self.new_physical_kv_slots,
            "freed_physical_slots": self.freed_physical_slots,
            "allocator_free_before": self.allocator_free_before,
            "allocator_free_after": self.allocator_free_after,
            "protected_prefix_tokens": self.protected_prefix_tokens,
            "history_tokens": self.history_tokens,
            "kept_history_tokens": self.kept_history_tokens,
            "current_tokens": self.current_tokens,
            "next_rope_position_before": self.next_rope_position_before,
            "next_rope_position_after": self.next_rope_position_after,
            "selected_history_indices": self.selected_history_indices or [],
            "bytes_per_kv_token": self.bytes_per_kv_token,
            "freed_kv_bytes": self.freed_kv_bytes,
        }


class PhysicalHistoryKVEvictor:
    """Token-level physical KV compaction for completed-history baselines.

    The request table maps logical positions to a shared physical token slot across
    all layers, so eviction is global-token based. Layer-specific PyramidKV
    capacities are therefore represented by a globalized keep set elsewhere.

    Paged allocators own complete pages. We compact surviving logical tokens into
    the request's leading pages, then free only pages that no longer contain a
    destination slot. This preserves the page allocator invariant while allowing
    arbitrary token selection within the retained tail page.
    """

    ATTENTION_SCORE_METHODS = {"h2o", "snapkv", "snapkv_persistent", "pyramidkv"}

    def __init__(
        self,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        *,
        bytes_per_kv_token: int | None = None,
    ):
        self.req_to_token_pool = req_to_token_pool
        self.allocator = token_to_kv_pool_allocator
        self.kv_cache = token_to_kv_pool_allocator.get_kvcache()
        self.bytes_per_kv_token = bytes_per_kv_token

    def evict(
        self,
        req,
        *,
        method: str,
        history_start: int,
        history_end: int,
        target_tokens: int,
        selected_history_indices: Sequence[int] | None = None,
    ) -> HistoryKVEvictionResult:
        method = (method or "").strip().lower()
        if method == "snapkv":
            method = "snapkv_persistent"
        if method == "pyramid":
            method = "pyramidkv"

        old_len = int(req.kv_committed_len)
        page_size = int(getattr(self.allocator, "page_size", 1))
        old_resident_slots = self._page_aligned_slots(old_len, page_size)
        result = HistoryKVEvictionResult(
            success=False,
            method=method,
            old_physical_kv_slots=old_resident_slots,
            next_rope_position_before=old_len + int(getattr(req, "c2kv_position_correction", 0) or 0),
        )

        if not (0 <= history_start <= history_end <= old_len):
            result.error = (
                "INVALID_HISTORY_EVICTION_RANGE: "
                f"history_start={history_start}, history_end={history_end}, old_len={old_len}"
            )
            result.runtime_status = "invalid_range"
            return result

        history_len = history_end - history_start
        result.protected_prefix_tokens = history_start
        result.history_tokens = history_len
        result.current_tokens = max(0, old_len - history_end)
        if history_len <= 0:
            result.success = True
            result.runtime_status = "physical_eviction_no_history"
            result.new_physical_kv_slots = old_resident_slots
            result.next_rope_position_after = result.next_rope_position_before
            return result

        if method in self.ATTENTION_SCORE_METHODS and selected_history_indices is None:
            result.error = (
                "ATTENTION_SCORE_SELECTION_UNAVAILABLE_IN_NORMAL_PREFILL: "
                f"method={method}"
            )
            result.runtime_status = "physical_eviction_attention_scores_unavailable"
            return result

        if selected_history_indices is None:
            selected_history_indices = self._select_streaming(history_len, target_tokens)
            runtime_status = "physical_eviction_ok"
        else:
            selected_history_indices = sorted(
                {int(x) for x in selected_history_indices if 0 <= int(x) < history_len}
            )
            runtime_status = (
                "physical_eviction_globalized"
                if method == "pyramidkv"
                else "physical_eviction_ok"
            )

        keep_abs = (
            list(range(0, history_start))
            + [history_start + i for i in selected_history_indices]
            + list(range(history_end, old_len))
        )
        keep_abs = sorted(dict.fromkeys(keep_abs))
        new_len = len(keep_abs)

        if new_len == old_len:
            result.success = True
            result.runtime_status = "physical_eviction_ok"
            result.new_physical_kv_slots = old_resident_slots
            result.kept_history_tokens = history_len
            result.selected_history_indices = list(range(history_len))
            result.next_rope_position_after = result.next_rope_position_before
            return result

        req_row = self.req_to_token_pool.req_to_token[req.req_pool_idx]
        old_slots = req_row[:old_len].to(torch.long)
        src_slots = old_slots[torch.tensor(keep_abs, dtype=torch.long, device=old_slots.device)]
        dst_slots = old_slots[:new_len]
        new_resident_slots = self._page_aligned_slots(new_len, page_size)

        # The req_to_token prefix is page-contiguous for an uncached request.
        # Keep all pages intersecting the destination prefix, including the last
        # partially populated page. PagedAllocator.free() rounds supplied token
        # indices to their owning pages, so passing old_slots[new_len:] would
        # incorrectly release that tail page.
        if new_resident_slots > old_resident_slots:
            result.error = (
                "INVALID_PAGED_COMPACTION_LENGTH: "
                f"{new_resident_slots=} exceeds {old_resident_slots=}"
            )
            result.runtime_status = "invalid_paged_compaction"
            return result
        destination_page_ids = torch.unique(dst_slots // page_size)
        old_page_ids = torch.unique(old_slots // page_size)
        keep_page_mask = torch.isin(old_page_ids, destination_page_ids)
        free_page_ids = old_page_ids[~keep_page_mask]
        # One representative slot per page is sufficient for PagedAllocator.free.
        free_slots = free_page_ids * page_size

        free_before = self._available_size()

        self._copy_kv(dst_slots, src_slots)
        if free_slots.numel() > 0:
            self.allocator.free(free_slots)
        req_row[:new_len] = dst_slots.to(req_row.dtype)
        req_row[new_len:old_len] = 0

        req.kv_committed_len = new_len
        req.kv_allocated_len = new_len
        req.already_computed = new_len
        req.c2kv_position_correction = (
            int(getattr(req, "c2kv_position_correction", 0) or 0) + old_len - new_len
        )

        free_after = self._available_size()
        result.success = True
        result.runtime_status = runtime_status
        result.new_physical_kv_slots = new_resident_slots
        result.freed_physical_slots = old_resident_slots - new_resident_slots
        result.allocator_free_before = free_before
        result.allocator_free_after = free_after
        result.kept_history_tokens = len(selected_history_indices)
        result.selected_history_indices = list(selected_history_indices)
        result.next_rope_position_after = (
            new_len + int(getattr(req, "c2kv_position_correction", 0) or 0)
        )
        result.bytes_per_kv_token = self._bytes_per_kv_token()
        if result.bytes_per_kv_token is not None:
            result.freed_kv_bytes = result.freed_physical_slots * result.bytes_per_kv_token
        return result

    @staticmethod
    def _select_streaming(history_len: int, target_tokens: int) -> list[int]:
        keep = min(history_len, max(0, int(target_tokens or 0)))
        if keep <= 0:
            return []
        return list(range(history_len - keep, history_len))

    def _copy_kv(self, dst_slots: torch.Tensor, src_slots: torch.Tensor) -> None:
        for layer_id in range(self.kv_cache.start_layer, self.kv_cache.start_layer + self.kv_cache.layer_num):
            key_buffer = self.kv_cache._get_key_buffer(layer_id)
            value_buffer = self.kv_cache._get_value_buffer(layer_id)
            key_buffer[dst_slots] = key_buffer[src_slots].clone()
            value_buffer[dst_slots] = value_buffer[src_slots].clone()

    @staticmethod
    def _page_aligned_slots(token_len: int, page_size: int) -> int:
        if token_len <= 0:
            return 0
        return ((int(token_len) + page_size - 1) // page_size) * page_size

    def _available_size(self) -> int | None:
        try:
            return int(self.allocator.available_size())
        except Exception:
            return None

    def _bytes_per_kv_token(self) -> int | None:
        # The paged buffer layout can expose a page rather than a single token
        # at index 0. The scheduler derives this from model KV heads/layers,
        # which is unambiguous and shared with the runtime peak-byte report.
        return self.bytes_per_kv_token
