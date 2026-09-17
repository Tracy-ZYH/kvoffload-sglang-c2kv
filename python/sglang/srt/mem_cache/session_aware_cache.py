from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import json
import logging

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
    history_kv_resident_positions: Any = None
    history_kv_score_state: Any = None

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
        self.history_kv_resident_positions = list(getattr(req, "history_kv_resident_positions", []) or [])
        resident = set(self.history_kv_resident_positions)
        self.history_kv_score_state = {layer: {p: s for p, s in scores.items() if p in resident}
                                     for layer, scores in (getattr(req, "history_kv_score_state", {}) or {}).items()}

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
        req.history_kv_score_state = self.history_kv_score_state
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

    @staticmethod
    def owns_finished_request(req: Req) -> bool:
        """Streaming KV belongs to the session, even without radix insertion."""
        return _is_streaming(req)

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
            config = getattr(req, "history_kv_eviction", None)
            hint = getattr(req, "c2kv_kv_memory_hint", None) or {}
            if (isinstance(config, dict) and config.get("persistent_continuation")) or (
                self._is_persistent_history_req(req) and int(hint.get("persistent_session_logical_prefix_tokens", 0)) > 0
            ):
                raise RuntimeError("PERSISTENT_HISTORY_SESSION_RESIDENT_CACHE_MISSING")
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

            from sglang.srt.mem_cache.history_kv_lifecycle import (
                append_resident_positions,
                physical_history_range,
                position_summary,
                selection_query_window,
            )
            hint = req.c2kv_kv_memory_hint or {}
            logical_prefix = int(hint["persistent_session_logical_prefix_tokens"])
            canonical_len = int(hint["persistent_session_canonical_prompt_tokens"])
            prior_positions = list(slot.history_kv_resident_positions or [])
            if len(prior_positions) != int(req.kv_committed_len):
                raise RuntimeError("PERSISTENT_HISTORY_SESSION_LEDGER_LENGTH_MISMATCH")
            positions = append_resident_positions(prior_positions, logical_prefix, canonical_len)
            req.history_kv_resident_positions = positions
            canonical_scopes = config.get("persistent_canonical_scopes")
            if isinstance(canonical_scopes, list) and canonical_scopes:
                physical_scopes = []
                for scope in canonical_scopes:
                    start, end = physical_history_range(
                        positions, int(scope["start"]), int(scope["end"]))
                    mapped = dict(scope)
                    mapped["canonical_start"] = int(scope["start"])
                    mapped["canonical_end"] = int(scope["end"])
                    mapped["start"] = start
                    mapped["end"] = end
                    # A persistent scope may already contain fewer resident
                    # tokens than its nominal budget; never manufacture old
                    # candidates that were previously evicted.
                    mapped["target_tokens"] = min(
                        end - start, int(scope.get("target_tokens") or end - start))
                    physical_scopes.append(mapped)
                config["semantic_scopes"] = physical_scopes
                protected = min(scope["start"] for scope in physical_scopes)
                history_end = max(scope["end"] for scope in physical_scopes)
                config["target_tokens"] = sum(
                    int(scope["target_tokens"]) for scope in physical_scopes)
            else:
                protected, history_end = physical_history_range(positions,
                    int(config.get("persistent_protected_prefix_tokens") or 0),
                    int(config["persistent_canonical_history_end"]))
            delta_history = int(config.get("persistent_delta_history_tokens") or 0)
            prefix_len = int(req.kv_committed_len)
            origin_len = len(req.origin_input_ids)
            if not (0 <= protected <= history_end <= origin_len and len(positions) == origin_len):
                req.set_finish_with_abort(
                    "PERSISTENT_HISTORY_SESSION_RANGE_INVALID: "
                    f"{protected=}, {prefix_len=}, {delta_history=}, {origin_len=}"
                )
            else:
                method = str(config.get("method") or "").strip().lower()
                query_window = selection_query_window(
                    method,
                    prefix_len,
                    history_end,
                    origin_len,
                    config.get("history_kv_recent_window"),
                )
                if query_window is None:
                    rounds = [
                        C2KVPrefillRound(
                            list(
                                req.origin_input_ids[
                                    : max(prefix_len + 1, history_end)
                                ]
                            ),
                            [],
                            post_history_kv_eviction=True,
                        )
                    ]
                    round_end = min(
                        origin_len, max(prefix_len + 1, history_end)
                    )
                    if round_end < origin_len:
                        rounds.append(
                            C2KVPrefillRound(
                                list(req.origin_input_ids[round_end:]), []
                            )
                        )
                else:
                    query_start, query_end = query_window
                    rounds = []
                    if query_start > prefix_len:
                        rounds.append(
                            C2KVPrefillRound(
                                list(req.origin_input_ids[:query_start]), []
                            )
                        )
                        query_tokens = list(
                            req.origin_input_ids[query_start:query_end]
                        )
                    else:
                        # The first active round must include the resident
                        # prefix so prepare_c2kv_round_input can consume it.
                        query_tokens = list(req.origin_input_ids[:query_end])
                    rounds.append(
                        C2KVPrefillRound(
                            query_tokens,
                            [],
                            post_history_kv_eviction=True,
                        )
                    )
                    config.update(
                        {
                            "selection_query_start": query_start,
                            "selection_query_end": query_end,
                            "selection_query_tokens": query_end - query_start,
                            "selection_query_phase": (
                                "new_tail_prefill_before_eviction"
                            ),
                        }
                    )
                req.c2kv_rounds = rounds
                req.c2kv_round_idx = 0
                req.c2kv_round_start_len = 0
                req.c2kv_virtual_input_ids = list(req.origin_input_ids)
                config["history_start"] = protected
                config["history_end"] = history_end
                config["persistent_prior_physical_tokens"] = prefix_len
                config["resident_logical_positions"] = positions
                config["previous_resident_position_summary"] = position_summary(prior_positions)
                if isinstance(report, dict):
                    report["persistent_session_history_start"] = protected
                    report["persistent_session_history_end"] = history_end
                    report["persistent_session_delta_history_tokens"] = delta_history
                    for key in (
                        "selection_query_start",
                        "selection_query_end",
                        "selection_query_tokens",
                        "selection_query_phase",
                    ):
                        if key in config:
                            report[key] = config[key]
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
            if getattr(req, "persistent_history_eviction_failed", False):
                self._rollback_failed_persistent_request(req)
                return
            from sglang.srt.mem_cache.history_kv_lifecycle import append_resident_positions, position_summary
            positions = getattr(req, "history_kv_resident_positions", None)
            hint = req.c2kv_kv_memory_hint or {}
            if positions is None:
                slot = self.slots.get(req.session.session_id)
                previous = list(slot.history_kv_resident_positions or []) if slot else []
                positions = append_resident_positions(previous,
                    int(hint.get("persistent_session_logical_prefix_tokens", 0)),
                    int(hint.get("persistent_session_canonical_prompt_tokens", len(req.origin_input_ids))))
                req.history_kv_resident_positions = positions
            if len(positions) != len(req.origin_input_ids):
                raise RuntimeError("PERSISTENT_HISTORY_SESSION_FINISHED_LEDGER_MISMATCH")
            self._discard_persistent_decode_suffix(req)
            if isinstance(getattr(req, "kv_memory_report", None), dict):
                req.kv_memory_report["persistent_session_saved_position_summary"] = position_summary(positions)
                old_slot = self.slots.get(req.session.session_id)
                prior = list(old_slot.history_kv_resident_positions or []) if old_slot else []
                event = req.kv_memory_report.setdefault("history_kv_lifecycle", {})
                event.update({
                    **{k: hint.get(k) for k in ("episode_id", "turn_id", "step_id")},
                    "event": "session_prompt_saved", "session_id": req.session.session_id,
                    "history_kv_backend": "physical_eviction", "persistent_session_enabled": True,
                    "resident_tokens_before_append": len(prior),
                    "new_turn_tokens": int(hint.get("persistent_session_delta_tokens", 0)),
                    "resident_tokens_after_append": len(prior) + int(hint.get("persistent_session_delta_tokens", 0)),
                    "resident_tokens_after_eviction": len(positions),
                    "evicted_tokens_this_turn": len(prior) + int(hint.get("persistent_session_delta_tokens", 0)) - len(positions),
                    "previous_resident_position_summary": position_summary(prior),
                    "resident_position_summary": position_summary(positions),
                    "full_history_reprefill_performed": False,
                    "history_prefill_tokens": int((getattr(req, "history_kv_eviction", None) or {}).get("persistent_delta_history_tokens", 0)),
                    "canonical_delta_prefill_tokens": int(hint.get("persistent_session_delta_tokens", 0)),
                    "count_scope": "canonical_prompt_including_protected_system_and_current",
                })
                event.setdefault("full_history_tokens", int(hint.get("full_equivalent_history_tokens", 0)))
                event.setdefault("retained_tokens_this_turn", int(req.kv_memory_report.get("active_history_kv_tokens", 0)))
                logging.getLogger(__name__).info("HISTORY_KV_LIFECYCLE %s", json.dumps(event, sort_keys=True))

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
        """Keep canonical prompt KV; free decode pages by PHYSICAL ownership.

        Allocator page IDs are not logical prompt offsets. In particular a
        session may start at any page in the pool; comparing a physical page
        ID with ceil(prompt_len/page_size) can free the prompt itself.
        """
        prompt_len = len(req.origin_input_ids)
        committed_len = int(req.kv_committed_len)
        allocated_len = int(req.kv_allocated_len)
        if not prompt_len <= committed_len <= allocated_len:
            raise RuntimeError("PERSISTENT_HISTORY_SESSION_INVALID_FINISHED_LENGTHS")
        row = self.req_to_token_pool.req_to_token[req.req_pool_idx]
        prompt_slots = row[:prompt_len].long()
        if prompt_len and (prompt_slots <= 0).any():
            raise RuntimeError("PERSISTENT_HISTORY_SESSION_MISSING_PROMPT_SLOT")
        prompt_pages = torch.unique(prompt_slots // self.page_size)
        tail_slots = row[prompt_len:allocated_len].long()
        explicit = getattr(req, "persistent_decode_cache_locs", None) or []
        if explicit:
            tail_slots = torch.cat([tail_slots, torch.stack(
                [slot.reshape(()) for slot in explicit]).long().to(row.device)])
        tail_pages = torch.unique(tail_slots[tail_slots > 0] // self.page_size)
        free_pages = tail_pages[~torch.isin(tail_pages, prompt_pages)]
        if free_pages.numel():
            self.token_to_kv_pool_allocator.free(free_pages * self.page_size)
        row[prompt_len:allocated_len] = 0
        req.persistent_decode_cache_locs = []
        req.kv_committed_len = prompt_len
        req.kv_allocated_len = prompt_len
        req.already_computed = prompt_len
        report = getattr(req, "kv_memory_report", None)
        if isinstance(report, dict):
            report["persistent_session_discarded_decode_kv_tokens"] = allocated_len - prompt_len
            report["persistent_session_reclaimed_decode_kv_tokens"] = int(free_pages.numel()) * self.page_size
            report["persistent_session_prompt_physical_tokens"] = prompt_len
            report["persistent_session_decode_page_free_scope"] = "request_owned_pages_excluding_prompt_pages"

    def _rollback_failed_persistent_request(self, req: Req) -> None:
        """Drop this turn's partial KV while preserving the prior session.

        Selection can fail after the persistent slot was restored and some
        delta tokens were prefetched.  The normal finished-request path uses
        the full canonical prompt length and must not run in that state.  The
        prior slot still owns the valid prefix, so free only pages exclusively
        used by the appended suffix and detach the failed request.
        """

        session_id = req.session.session_id
        slot = self.slots.get(session_id)
        if slot is None or slot.req_pool_idx is None:
            # First-turn failure has no reusable prior state.  Delegate normal
            # non-session cleanup and make sure no broken session is retained.
            self.slots.pop(session_id, None)
            self.inner.cache_finished_req(req, is_insert=False)
            return

        keep_len = int(slot.kv_committed_len)
        allocated_len = int(getattr(req, "kv_allocated_len", keep_len) or 0)
        row = self.req_to_token_pool.req_to_token[slot.req_pool_idx]
        keep_slots = row[:keep_len].long()
        keep_pages = torch.unique(keep_slots[keep_slots > 0] // self.page_size)
        tail_slots = row[keep_len:max(keep_len, allocated_len)].long()
        tail_pages = torch.unique(tail_slots[tail_slots > 0] // self.page_size)
        free_pages = tail_pages[~torch.isin(tail_pages, keep_pages)]
        if free_pages.numel():
            self.token_to_kv_pool_allocator.free(free_pages * self.page_size)
        row[keep_len:max(keep_len, allocated_len)] = 0

        report = getattr(req, "kv_memory_report", None)
        if isinstance(report, dict):
            report["persistent_session_failed_turn_rolled_back"] = True
            report["persistent_session_rollback_kept_tokens"] = keep_len
            report["persistent_session_rollback_freed_pages"] = int(
                free_pages.numel())
        # SessionSlot retains ownership of the valid prefix.  Prevent generic
        # request cleanup from freeing or re-saving the same pool row.
        req.req_pool_idx = None
        req.mamba_pool_idx = None

    def cache_unfinished_req(self, req: Req, **kwargs):
        if _is_streaming(req):
            if self._is_persistent_history_req(req):
                # Physical eviction can rewrite/free these pages. Keep them
                # request/session-owned instead of inserting them into radix.
                kv_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, : len(req.fill_ids)
                ]
                req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
                req.cache_protected_len = 0
                return
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
        logging.getLogger(__name__).info("HISTORY_KV_SESSION_CLOSED %s", json.dumps({
            "session_id": session_id, "resident_tokens_released": slot.kv_allocated_len,
            "remaining_session_slots": len(self.slots)}))

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
