from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InitLoadBackParams,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.utils.common import ceil_align

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


class _VirtualNode:
    """Sentinel node for streaming session requests.

    Passed to inc_lock_ref / dec_lock_ref so the wrapper can distinguish
    streaming-session locks (no-op) from real radix-tree locks (forwarded).
    """

    pass


@dataclass
class SessionSlot:
    """Holds KV state between streaming session turns."""

    virtual_node: _VirtualNode = field(default_factory=_VirtualNode)

    # KV pool state (None means no KV is currently held by this slot)
    req_pool_idx: Optional[int] = None
    kv_committed_len: int = 0
    kv_allocated_len: int = 0

    # First req's radix tree node (for dec_lock_ref on session close)
    last_node: Any = None
    cache_protected_len: int = 0
    swa_uuid_for_lock: Optional[str] = None

    # SWA state
    swa_evicted_seqlen: int = 0

    # C2KV physical-history eviction keeps a compact physical sequence while
    # rotary positions remain in the original logical frame. Persist the
    # correction across streaming session requests.
    c2kv_position_correction: int = 0

    # Mamba states
    mamba_pool_idx: Any = None
    mamba_ping_pong_track_buffer: Any = None
    mamba_next_track_idx: Any = None
    mamba_last_track_seqlen: Any = None
    mamba_branching_seqlen: Any = None

    @property
    def is_holding_kv(self) -> bool:
        """Whether this slot currently holds KV pool resources."""
        return self.req_pool_idx is not None

    def save_from_req(self, req: Req, is_first: bool):
        """Save KV state from a finishing request into this slot."""
        self.req_pool_idx = req.req_pool_idx
        self.kv_committed_len = req.kv_committed_len
        self.kv_allocated_len = req.kv_allocated_len
        self.swa_evicted_seqlen = req.swa_evicted_seqlen
        self.c2kv_position_correction = int(
            getattr(req, "c2kv_position_correction", 0) or 0
        )

        if is_first:
            self.last_node = req.last_node
            self.cache_protected_len = req.cache_protected_len
            self.swa_uuid_for_lock = req.swa_uuid_for_lock

        self.mamba_pool_idx = req.mamba_pool_idx
        self.mamba_ping_pong_track_buffer = req.mamba_ping_pong_track_buffer
        self.mamba_next_track_idx = req.mamba_next_track_idx
        self.mamba_last_track_seqlen = req.mamba_last_track_seqlen
        self.mamba_branching_seqlen = req.mamba_branching_seqlen

        req.req_pool_idx = None
        req.mamba_pool_idx = None

    def restore_to_req(self, req: Req):
        """Restore KV state from this slot into an incoming request."""
        req.req_pool_idx = self.req_pool_idx
        req.kv_committed_len = self.kv_committed_len
        req.kv_allocated_len = self.kv_allocated_len
        req.swa_evicted_seqlen = self.swa_evicted_seqlen
        req.c2kv_position_correction = self.c2kv_position_correction
        req.swa_uuid_for_lock = self.swa_uuid_for_lock

        req.mamba_pool_idx = self.mamba_pool_idx
        req.mamba_ping_pong_track_buffer = self.mamba_ping_pong_track_buffer
        req.mamba_next_track_idx = self.mamba_next_track_idx
        req.mamba_last_track_seqlen = self.mamba_last_track_seqlen
        req.mamba_branching_seqlen = self.mamba_branching_seqlen

        # NOTE: req_pool_idx and mamba_pool_idx are intentionally NOT cleared
        # from the slot. During chunked prefill, a request may be rejected by
        # the scheduler (e.g. budget exhausted) and retried in the next cycle.
        # Each retry calls match_prefix -> restore_to_req again, so the slot
        # must remain intact for idempotent restoration.


def _is_streaming(req: Optional[Req]) -> bool:
    return req is not None and req.session is not None and req.session.streaming


class SessionAwareCache(BasePrefixCache):
    """Decorator around any BasePrefixCache that manages streaming session KV.

    Non-streaming requests are pure pass-through. Streaming requests have their
    KV lifecycle managed by SessionSlot objects, avoiding any invasive changes
    to the scheduling pipeline.
    """

    def __init__(self, inner: BasePrefixCache):
        self.inner = inner
        self.slots: Dict[str, SessionSlot] = {}

    # -- Forward PrefixCacheTrait properties to inner cache --

    @property
    def req_to_token_pool(self):
        return self.inner.req_to_token_pool

    @req_to_token_pool.setter
    def req_to_token_pool(self, value):
        self.inner.req_to_token_pool = value

    @property
    def token_to_kv_pool_allocator(self):
        return self.inner.token_to_kv_pool_allocator

    @token_to_kv_pool_allocator.setter
    def token_to_kv_pool_allocator(self, value):
        self.inner.token_to_kv_pool_allocator = value

    @property
    def page_size(self):
        return self.inner.page_size

    @page_size.setter
    def page_size(self, value):
        self.inner.page_size = value

    @property
    def disable(self):
        return self.inner.disable

    @disable.setter
    def disable(self, value):
        self.inner.disable = value

    @property
    def metrics_collector(self):
        return self.inner.metrics_collector

    @metrics_collector.setter
    def metrics_collector(self, value):
        self.inner.metrics_collector = value

    # -- BasePrefixCache abstract methods --

    def reset(self):
        self.slots.clear()
        self.inner.reset()

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        req = params.req
        if not _is_streaming(req):
            return self.inner.match_prefix(params)

        session_id = req.session.session_id
        slot = self.slots.get(session_id)
        if slot is None or slot.req_pool_idx is None:
            return self.inner.match_prefix(params)

        slot.restore_to_req(req)

        report = getattr(req, "kv_memory_report", None)
        if isinstance(report, dict):
            report["persistent_history_session"] = True
            report["persistent_session_restore"] = True
            report["persistent_session_reused_physical_tokens"] = int(
                req.kv_committed_len
            )
            report["persistent_session_position_correction"] = int(
                req.c2kv_position_correction
            )

        # A persistent physical-history request arrives with only the exact
        # chat-template delta.  Now that the session KV length is restored,
        # split that delta into completed-history and current-query rounds.
        # This must happen after restore: before that point the physical prefix
        # length is unknown.
        config = getattr(req, "history_kv_eviction", None)
        if (
            isinstance(config, dict)
            and config.get("persistent_continuation_pending")
        ):
            from sglang.srt.managers.schedule_batch import C2KVPrefillRound

            protected = int(config.get("persistent_protected_prefix_tokens") or 0)
            delta_history = int(config.get("persistent_delta_history_tokens") or 0)
            prefix_len = int(req.kv_committed_len)
            history_end = prefix_len + delta_history
            origin_len = len(req.origin_input_ids)
            if not (0 <= protected <= prefix_len <= history_end <= origin_len):
                req.set_finish_with_abort(
                    "PERSISTENT_HISTORY_SESSION_RANGE_INVALID: "
                    f"{protected=}, {prefix_len=}, {delta_history=}, {origin_len=}"
                )
            else:
                rounds = [
                    C2KVPrefillRound(
                        list(req.origin_input_ids[:history_end]),
                        [],
                        post_history_kv_eviction=True,
                    )
                ]
                if history_end < origin_len:
                    rounds.append(
                        C2KVPrefillRound(
                            list(req.origin_input_ids[history_end:]), []
                        )
                    )
                req.c2kv_rounds = rounds
                req.c2kv_round_idx = 0
                req.c2kv_round_start_len = 0
                req.c2kv_virtual_input_ids = list(req.origin_input_ids)
                config["history_start"] = protected
                config["history_end"] = history_end
                config["persistent_prior_physical_tokens"] = prefix_len
                if isinstance(report, dict):
                    report["persistent_session_history_start"] = protected
                    report["persistent_session_history_end"] = history_end
                    report["persistent_session_delta_history_tokens"] = delta_history
                config.pop("persistent_continuation_pending", None)

        # logprob_start_len is already forced to -1 for streaming sessions
        # (in Req.init_next_round_input), so the prefix key is not truncated
        # and we can directly reuse the committed KV length.
        prefix_len = min(req.kv_committed_len, max(len(params.key.token_ids) - 1, 0))
        device_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :prefix_len
        ].to(dtype=torch.int64)

        return MatchResult(
            device_indices=device_indices,
            last_device_node=slot.virtual_node,
            last_host_node=slot.virtual_node,
            cache_protected_len=slot.cache_protected_len,
        )

    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        if not _is_streaming(req):
            return self.inner.cache_finished_req(req, is_insert=is_insert, **kwargs)

        if self._is_persistent_history_req(req):
            self._discard_persistent_decode_suffix(req)

        session_id = req.session.session_id
        slot = self.slots.get(session_id)
        is_first = slot is None
        if is_first:
            slot = SessionSlot()
            self.slots[session_id] = slot

        slot.save_from_req(req, is_first=is_first)

    @staticmethod
    def _is_persistent_history_req(req: Req) -> bool:
        hint = getattr(req, "c2kv_kv_memory_hint", None)
        return bool(
            isinstance(hint, dict)
            and isinstance(hint.get("persistent_history_session"), dict)
            and hint["persistent_history_session"].get("enabled")
        )

    def _discard_persistent_decode_suffix(self, req: Req) -> None:
        """Remove raw decode KV before stashing a persistent chat session.

        A generated tool call is parsed into structured OpenAI data and gets
        re-rendered by the chat template on the next request. Its raw decode
        tokens are therefore not a valid cache prefix. Keep only the prompt
        KV; the canonical assistant/tool serialization is part of the next
        request's delta and is prefetched normally.
        """
        prompt_len = len(req.origin_input_ids)
        committed_len = int(req.kv_committed_len)
        allocated_len = int(req.kv_allocated_len)
        if prompt_len > committed_len:
            raise RuntimeError(
                "PERSISTENT_HISTORY_SESSION_PROMPT_LONGER_THAN_KV: "
                f"{prompt_len=}, {committed_len=}, rid={req.rid}"
            )
        if committed_len > allocated_len:
            raise RuntimeError(
                "PERSISTENT_HISTORY_SESSION_COMMITTED_EXCEEDS_ALLOCATED: "
                f"{committed_len=}, {allocated_len=}, rid={req.rid}"
            )
        discarded = allocated_len - prompt_len
        reclaimed = 0
        orphaned_tail = 0
        req_row = self.req_to_token_pool.req_to_token[req.req_pool_idx]

        # Paged decode allocation is page-granular.  The allocator can own a
        # page even when the request row/logical length only exposes the
        # prompt.  Release explicitly recorded decode slots first, filtering
        # out the prompt's final partial page, which must remain resident.
        explicit_reclaimed = 0
        explicit_slots = getattr(req, "persistent_decode_cache_locs", None) or []
        if explicit_slots:
            slots = torch.stack(
                [slot.reshape(()) for slot in explicit_slots]
            ).to(dtype=torch.int64)
            min_page = (prompt_len + self.page_size - 1) // self.page_size
            page_ids = torch.div(slots, self.page_size, rounding_mode="floor")
            slots = slots[(slots > 0) & (page_ids >= min_page)]
            if slots.numel() > 0:
                pages = torch.unique(
                    torch.div(slots, self.page_size, rounding_mode="floor")
                )
                free_slots = pages[:, None] * self.page_size + torch.arange(
                    self.page_size, device=pages.device, dtype=pages.dtype
                )
                free_slots = free_slots.reshape(-1)
                self.token_to_kv_pool_allocator.free(free_slots)
                explicit_reclaimed = int(free_slots.numel())
                # Clear any request-row references into the released pages.
                for page in pages.tolist():
                    start = int(page) * self.page_size
                    end = min(start + self.page_size, req_row.shape[0])
                    req_row[start:end] = 0
                orphaned_tail = explicit_reclaimed
            req.persistent_decode_cache_locs = []
        if discarded:
            # PagedAllocator owns complete pages. Free only pages strictly
            # after the prompt's final page; the remaining partial page is
            # retained as an internal session tail and is not attention
            # visible. It is released with the session.
            free_start = ceil_align(prompt_len, self.page_size)
            if free_start < allocated_len:
                free_slots = req_row[free_start:allocated_len]
                self.token_to_kv_pool_allocator.free(free_slots)
                reclaimed = allocated_len - free_start
            req_row[prompt_len:allocated_len] = 0
            req.kv_committed_len = prompt_len
            req.kv_allocated_len = prompt_len
            req.already_computed = prompt_len

        # Some decode/over-allocation paths reserve a complete page but leave
        # kv_allocated_len at the last committed token. Such a page is not
        # visible through the session length accounting and would trip the
        # idle memory checker after ownership transfer. It is safe to inspect
        # only pages after the prompt's final page: the prompt page itself is
        # retained because it contains attention-visible KV.
        scan_start = ceil_align(prompt_len, self.page_size)
        scan_end = min(scan_start + self.page_size, req_row.shape[0])
        if scan_start < scan_end:
            tail_slots = req_row[scan_start:scan_end]
            live_tail_slots = tail_slots[tail_slots > 0]
            if live_tail_slots.numel() > 0:
                self.token_to_kv_pool_allocator.free(live_tail_slots)
                req_row[scan_start:scan_end] = 0
                orphaned_tail = self.page_size
        report = getattr(req, "kv_memory_report", None)
        if isinstance(report, dict):
            report["persistent_session_discarded_decode_kv_tokens"] = discarded
            report["persistent_session_reclaimed_decode_kv_tokens"] = reclaimed
            report["persistent_session_page_tail_tokens"] = discarded - reclaimed
            report["persistent_session_orphaned_tail_page_tokens"] = orphaned_tail
            report["persistent_session_explicit_decode_reclaimed_tokens"] = (
                explicit_reclaimed
            )
            report["persistent_session_prompt_physical_tokens"] = prompt_len

    def cache_unfinished_req(self, req: Req, **kwargs):
        if _is_streaming(req):
            # in chunked_prefill for streaming, we skip the stash path which triggers radix.
            # only the last chunk in first turn trigger a full prompt radix insert.
            if kwargs.get("chunked", False):
                kv_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, : len(req.fill_ids)
                ]
                req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
                return
            if req.session.session_id in self.slots:
                # Subsequent turns: slot exists, skip inner entirely.
                return
            # First turn (no slot): fall through to inner for lock management,
            # tree insertion, and cache_protected_len updates between chunks.
        self.inner.cache_unfinished_req(req, **kwargs)

    def evict(self, params: EvictParams) -> EvictResult:
        return self.inner.evict(params)

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        if isinstance(node, _VirtualNode):
            return IncLockRefResult()
        return self.inner.inc_lock_ref(node)

    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if isinstance(node, _VirtualNode):
            return DecLockRefResult()
        return self.inner.dec_lock_ref(node, params)

    # -- Session lifecycle --

    def release_session(self, session_id: str):
        """Release all KV resources held by a streaming session."""
        slot = self.slots.pop(session_id, None)
        if slot is None:
            return

        if slot.last_node is not None:
            if slot.swa_uuid_for_lock is not None:
                self.inner.dec_lock_ref(
                    slot.last_node,
                    DecLockRefParams(swa_uuid_for_lock=slot.swa_uuid_for_lock),
                )
            else:
                self.inner.dec_lock_ref(slot.last_node)

        if slot.is_holding_kv:
            start = slot.cache_protected_len
            end = slot.kv_allocated_len
            if start < end:
                kv_indices = self.req_to_token_pool.req_to_token[
                    slot.req_pool_idx, start:end
                ]
                self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free_slots.append(slot.req_pool_idx)

    def session_held_tokens(self) -> int:
        """Total KV tokens held by session slots, not tracked by the tree."""
        total = 0
        for slot in self.slots.values():
            if slot.is_holding_kv:
                allocated = ceil_align(slot.kv_allocated_len, self.page_size)
                total += allocated - slot.cache_protected_len
        return total

    def session_held_full_tokens(self) -> int:
        """An alias to align the naming style of SWA"""
        return self.session_held_tokens()

    def session_held_swa_tokens(self) -> int:
        """Total SWA tokens held by session slots, not tracked by the tree."""
        total = 0
        for slot in self.slots.values():
            if slot.is_holding_kv:
                allocated = ceil_align(slot.kv_allocated_len, self.page_size)
                total += allocated - max(
                    slot.cache_protected_len, slot.swa_evicted_seqlen
                )
        return total

    def session_held_req_count(self) -> int:
        """Number of req pool slots held by session slots."""
        return sum(s.is_holding_kv for s in self.slots.values())

    # -- Pass-through methods --

    def evictable_size(self):
        return self.inner.evictable_size()

    def full_evictable_size(self):
        return self.inner.full_evictable_size()

    def swa_evictable_size(self):
        return self.inner.swa_evictable_size()

    def protected_size(self):
        return self.inner.protected_size()

    def full_protected_size(self):
        return self.inner.full_protected_size()

    def swa_protected_size(self):
        return self.inner.swa_protected_size()

    def total_size(self):
        return self.inner.total_size()

    def pretty_print(self):
        return self.inner.pretty_print()

    def init_load_back(self, params: InitLoadBackParams):
        return self.inner.init_load_back(params)

    def ready_to_load_host_cache(self):
        return self.inner.ready_to_load_host_cache()

    def flush_write_through_acks(self) -> None:
        return self.inner.flush_write_through_acks()

    def check_hicache_events(self):
        return self.inner.check_hicache_events()

    def take_events(self):
        return self.inner.take_events()

    def supports_swa(self):
        return self.inner.supports_swa()

    def supports_mamba(self):
        return self.inner.supports_mamba()

    def is_chunk_cache(self):
        return self.inner.is_chunk_cache()

    def is_tree_cache(self):
        return self.inner.is_tree_cache()

    def available_and_evictable_str(self):
        return self.inner.available_and_evictable_str()

    def init_metrics_collector(self):
        return self.inner.init_metrics_collector()

    def sanity_check(self):
        # Skip inner sanity check when sessions hold tree locks, because
        # the check asserts all nodes are unlocked during idle.
        if any(s.is_holding_kv for s in self.slots.values()):
            return
        self.inner.sanity_check()

    # Forward attribute access for cache-specific methods (e.g.
    # sliding_window_size, all_values_flatten, etc.)
    def __getattr__(self, name):
        return getattr(self.inner, name)
