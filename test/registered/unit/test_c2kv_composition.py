"""CPU execution tests for composed injection, selection and session coordinates."""

from __future__ import annotations

import ast
import copy
import importlib.util
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Optional

import pytest
import torch


SRT = Path(__file__).resolve().parents[3] / "python/sglang/srt"


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, SRT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


composition = load("test_composition_contract", "mem_cache/c2kv_composition.py")
lifecycle = load("test_composition_lifecycle", "mem_cache/history_kv_lifecycle.py")
selection = load("test_composition_selection", "mem_cache/history_kv_selection.py")
native = load("test_composition_native", "mem_cache/c2kv_native_packed.py")


def extract_class(relative, name, methods=None, extra=None):
    tree = ast.parse((SRT / relative).read_text(encoding="utf-8"))
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name)
    if methods is not None:
        node.bases = []
        node.body = [item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name in methods]
    namespace = {"Optional": Optional, "List": list, "torch": torch,
                 "logger": logging.getLogger(__name__), "_is_npu": False,
                 "pool_snapkv_scores_by_position": selection.pool_snapkv_scores_by_position,
                 "_persistent_history_session_error": lambda *args: None,
                 **(extra or {})}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SRT / relative), "exec"), namespace)
    return namespace[name]


Round = extract_class("managers/schedule_batch.py", "C2KVPrefillRound")
Scheduler = extract_class("managers/scheduler.py", "Scheduler", {
    "_build_c2kv_prefill_rounds", "_compose_c2kv_history_rounds",
    "_select_history_kv_eviction_indices", "_build_pyramidkv_reference_state",
    "_advance_cached_c2kv_tool_rounds",
    "_add_c2kv_kv_memory_tokens",
})


@pytest.fixture
def engine(monkeypatch):
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.schedule_batch", SimpleNamespace(C2KVPrefillRound=Round))
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.c2kv_composition", composition)
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.history_kv_lifecycle", lifecycle)
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.c2kv_pool", SimpleNamespace(c2kv_gist_token_ids=lambda key, count: [-100 - index for index in range(count)]))
    obj = Scheduler()
    obj.page_size = 1
    obj.max_req_input_len = 1000
    obj.max_req_len = 1000
    obj._log_c2kv_token_usage = lambda *args, **kwargs: None
    obj._release_c2kv_pins = lambda req: None
    obj._C2KV_REPAIR_PLACEMENTS = ("in_place", "append_keep_ledger", "append_tail")
    entries = {"tool": SimpleNamespace(entry_type="gist", gist_len=2, original_seq_len=8, positions=[3, 7])}
    obj.c2kv_pool = SimpleNamespace(get=lambda key: entries.get(key),
        get_position_ids=lambda entry: torch.tensor(entry.positions), pin_many=lambda keys: True)
    return obj


def request(*, start=2, end=2, history_start=2, history_end=8, region="tool", method="h2o", persistent=False):
    return SimpleNamespace(
        rid="composition", origin_input_ids=list(range(12)),
        c2kv_segments=[SimpleNamespace(token_start=start, token_end=end, key_hash="tool",
            source_token_count=8, source_token_end=None, expected_token_len=None,
            region=region, repair_placement=None, repair_key_hashes=[])],
        c2kv_use_gist_projection=False, prefix_indices=[], c2kv_position_correction=0,
        sampling_params=SimpleNamespace(max_new_tokens=8),
        history_kv_eviction={"method": method, "history_start": history_start,
            "history_end": history_end, "target_tokens": 2,
            "history_kv_recent_window": 2, "persistent_session": persistent},
        c2kv_kv_memory_hint={"persistent_history_session": {"enabled": persistent}},
        kv_memory_report={}, history_kv_score_state={},
    )


def test_off_legacy_builder_keeps_rounds_and_virtual_sequence(engine):
    req = request(region=None)
    assert engine._build_c2kv_prefill_rounds(req) is None
    assert [(r.tokens, r.post_inject_seg_indices, r.post_history_kv_eviction) for r in req.c2kv_rounds] == [
        ([0, 1], [0], False), (list(range(2, 12)), [], False)]
    assert req.c2kv_virtual_input_ids == [0, 1, -100, -101] + list(range(2, 12))
    assert not hasattr(req, "history_kv_resident_positions")


@pytest.mark.parametrize("method", ["h2o", "snapkv", "pyramidkv"])
def test_initial_tools_and_history_share_rounds_and_expanded_positions(engine, method):
    req = request(method=method, persistent=True)
    assert engine._build_c2kv_prefill_rounds(req) is None
    assert req.history_kv_resident_positions == [0, 1, 5, 9] + list(range(10, 20))
    assert req.history_kv_eviction["history_start"] == 4
    assert req.history_kv_eviction["history_end"] == 10
    assert req.history_kv_eviction["canonical_history_start"] == 10
    assert req.history_kv_eviction["canonical_history_end"] == 16
    assert req.c2kv_rounds[0].post_inject_seg_indices == [0]
    assert req.c2kv_rounds[-1].tokens == [10, 11]
    assert sum(r.post_history_kv_eviction for r in req.c2kv_rounds) == 1
    assert req.c2kv_persistent_active_input_ids == req.c2kv_virtual_input_ids


def test_first_turn_without_eviction_stores_physical_session_view(engine):
    req = request(persistent=True)
    req.history_kv_eviction = None
    assert engine._build_c2kv_prefill_rounds(req) is None
    assert req.c2kv_persistent_active_input_ids == [0, 1, -100, -101] + list(range(2, 12))
    assert req.history_kv_resident_positions[-1] == 19
    assert not any(r.post_history_kv_eviction for r in req.c2kv_rounds)


def test_tools_inside_history_are_protected_and_not_charged_to_history_budget(engine):
    req = request(start=5, end=5, history_start=2, history_end=10, persistent=True)
    assert engine._build_c2kv_prefill_rounds(req) is None
    assert req.history_kv_eviction["protected_history_indices"] == [3, 4]
    req.history_kv_selection_scores = {"layers": [torch.tensor([1., 2., 3., 999., 999., 4., 5., 6., 7., 8.])]}
    selected = engine._select_history_kv_eviction_indices(req, req.history_kv_eviction)
    assert selected == [3, 4, 8, 9]
    tool_positions = {req.history_kv_resident_positions[5], req.history_kv_resident_positions[6]}
    assert not tool_positions & set(req.history_kv_score_state[0])


def test_query_collection_preserves_npu_bridge_and_evicts_once():
    descriptors = [{"positions": [3, 7]}]
    rounds = [Round([0, 1], [0]), Round([2, 3], []), Round([4, 5, 6, 7, 8, 9], [])]
    output = composition.split_rounds_at_query(rounds, descriptors, 4, 12, Round)
    assert [len(item.tokens) for item in output] == [2, 2, 6]
    assert [item.collect_history_kv_scores for item in output] == [False, True, True]
    assert [item.post_history_kv_eviction for item in output] == [False, False, True]


def test_query_window_excludes_gists_of_newly_appended_document(engine):
    req = request(start=10, end=10, history_start=2, history_end=8)
    req.history_kv_eviction["history_kv_recent_window"] = 64
    assert engine._build_c2kv_prefill_rounds(req) is None
    # Only the two real tail tokens execute Q; injected document KV cannot be
    # counted as query observations even when the requested window is longer.
    assert req.history_kv_eviction["selection_query_tokens"] == 2
    assert req.c2kv_rounds[-1].tokens == [10, 11]


def test_append_ledger_preserves_evicted_holes_and_anchors_new_tool():
    previous = [0, 1, 5, 9, 13, 18, 19]
    descriptors = [{"token_start": 9, "token_end": 9, "source_tokens": 8,
                    "positions": [3, 7]}]
    positions = composition.resident_positions(12, descriptors, previous, 20)
    assert positions == previous + [20, 21, 25, 29, 30, 31, 32]
    assert not set(range(10, 13)) & set(positions)
    assert len(positions) == 14


def test_message_boundaries_and_event_metadata_remove_same_carriers():
    hint = {"history_kv_eviction": {"history_start_message_count": 2, "history_message_count": 5},
            "history_kv_event_messages": list("abcdef")}
    composition.remap_message_metadata(hint, [1, 3], 6)
    assert hint["history_kv_eviction"] == {"history_start_message_count": 1, "history_message_count": 3}
    assert hint["history_kv_event_messages"] == list("acef")


def test_removed_carriers_renumber_events_for_real_token_span_resolver():
    events = load("test_composition_events", "mem_cache/history_kv_events.py")
    original = [
        {"message_index": 0, "role": "system", "phase": "others"},
        {"message_index": 1, "role": "user", "phase": "others"},
        {"message_index": 2, "role": "assistant", "phase": "act", "event_id": "call"},
        {"message_index": 3, "role": "user", "phase": "others"},
        {"message_index": 4, "role": "tool", "phase": "tool", "event_id": "result"},
    ]
    hint = {"history_kv_event_messages": original}
    composition.remap_message_metadata(hint, [1, 3], 5)
    retained = hint["history_kv_event_messages"]
    assert [item["message_index"] for item in retained] == [0, 1, 2]
    assert retained[-1]["event_id"] == "result"
    assert original[-1]["message_index"] == 4
    spans = events.resolve_history_kv_event_token_spans(
        total_tokens=13, message_prefix_token_counts=[0, 5, 8, 13],
        event_messages=retained,
    )
    assert spans == [
        {"role": "system", "phase": "others", "start": 0, "end": 5, "message_index": 0},
        {"role": "assistant", "phase": "act", "start": 5, "end": 8, "message_index": 1},
        {"role": "tool", "phase": "tool", "start": 8, "end": 13, "message_index": 2},
    ]


def test_session_boundary_at_new_carrier_does_not_count_it_as_cached(monkeypatch):
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.c2kv_composition", composition)
    Chat = extract_class("entrypoints/openai/serving_chat.py", "OpenAIServingChat", {"_translate_tool_session_coordinates"})
    chat = Chat()
    previous = {"token_start": 2, "token_end": 2, "source_tokens": 8, "key_hash": "a"}
    fresh = {"token_start": 12, "token_end": 12, "source_tokens": 16, "key_hash": "b"}
    chat._persistent_history_tool_segments = {"session": [previous]}
    hint = {"tool_memory_segments": [previous, fresh], "persistent_session_logical_prefix_tokens": 12,
            "persistent_session_canonical_prompt_tokens": 20,
            "history_kv_eviction": {"persistent_canonical_history_end": 16}}
    chat._translate_tool_session_coordinates(SimpleNamespace(c2kv_kv_memory_hint=hint, session_params={"id": "session"}), 12, 20)
    assert hint["persistent_session_logical_prefix_tokens"] == 20
    assert hint["persistent_session_canonical_prompt_tokens"] == 44
    assert hint["history_kv_eviction"]["persistent_canonical_history_end"] == 40


def test_cached_prefix_injects_tool_without_zero_token_forward(engine):
    req = SimpleNamespace(c2kv_tool_source_spans=[(10, 18)], c2kv_rounds=[Round([0, 1], [0]), Round([2, 3], [])],
        c2kv_round_idx=0, extend_input_len=0, kv_committed_len=2, req_pool_idx=0)
    engine.req_to_token_pool = SimpleNamespace(req_to_token=torch.arange(20).reshape(1, 20))
    engine.tree_cache = object()
    injected = []
    def inject(current, index, start):
        injected.append((index, start))
        current.kv_committed_len += 2
        return True
    engine._inject_c2kv_gist_segment = inject
    req.prepare_c2kv_round_input = lambda cache: setattr(req, "extend_input_len", 2)
    assert engine._advance_cached_c2kv_tool_rounds(req)
    assert injected == [(0, 2)]
    assert req.c2kv_round_idx == 1
    assert req.c2kv_round_start_len == 4


def test_headwise_pyramid_excludes_tool_slots_from_reference_candidates(engine, monkeypatch):
    reference = load("test_composition_reference", "mem_cache/history_kv_reference.py")
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.history_kv_reference", reference)
    telemetry = SimpleNamespace(sample=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "sglang.srt.observability", SimpleNamespace(paper_telemetry=telemetry))
    keys = torch.arange(24, dtype=torch.float32).reshape(12, 1, 2)
    engine.token_to_kv_pool_allocator = SimpleNamespace(get_kvcache=lambda: SimpleNamespace(get_kv_buffer=lambda layer: (keys, keys + 100)))
    engine.req_to_token_pool = SimpleNamespace(req_to_token=torch.arange(12).reshape(1, 12))
    req = SimpleNamespace(req_pool_idx=0, history_kv_resident_positions=list(range(12)), history_kv_reference_state=None)
    config = {"history_start": 2, "history_end": 10, "protected_history_indices": [2, 3],
              "target_tokens": 2, "history_kv_recent_window": 1, "history_kv_kernel_size": 1}
    score = {"headwise_layers": [torch.tensor([[1., 2., 999., 999., 3., 4., 5., 6.]])], "layer_ids": [0]}
    state = engine._build_pyramidkv_reference_state(req, config, score)
    assert not {4, 5} & set(state.layer(0).positions.flatten().tolist())
    assert state.layer(0).key.shape[1] == 2


def native_chunk(name, ids, start, ratio=4, tool=False):
    return {"chunk_id": name, "event_id": name, "part_index": 0,
            "source_token_start": 0, "source_token_end": len(ids), "token_ids": list(ids),
            "source_position_start": start,
            "gist_position_ids": [start + min(index + ratio, len(ids)) - 1 for index in range(0, len(ids), ratio)],
            **({"projection_set": "tool", "compression_ratio": ratio} if tool else {})}


def native_plan(**overrides):
    args = {"system_input_ids": [0, 1], "workspace_input_ids": list(range(6, 12)),
            "encoder_chunks": [native_chunk("history", list(range(2, 6)), 2)],
            "compression_chunks": [], "model_binding": {}, "packing_version": native.PACKING_VERSION,
            "raw_layout_profile": native.RAW_LAYOUT_PROFILE, "encoding_scope": "test",
            "compression_ratio": 4, "tool_binding": {"enabled": True, "identity": "checkpoint"}}
    args.update(overrides)
    return native.plan_native_packed_request(**args)


def test_anchored_t0_preserves_encoder_envelope_and_multiple_chunk_order():
    chunks = [native_chunk("tool-a", list(range(100, 108)), 6, tool=True),
              native_chunk("tool-b", list(range(200, 204)), 14, tool=True)]
    plan = native_plan(tool_gist_segments=[{"token_start": 6, "token_end": 8, "chunks": chunks}])
    assert list(plan.logical_input_ids) == list(range(12))
    assert plan.segment_boundaries == ((2, 6), (6, 8), (8, 8))
    assert len(plan.selected_handles) == 3
    assert list(plan.unique_chunks[1]["token_ids"]) == list(range(100, 108))
    assert plan.costs["canonical_position_tokens"] == 22
    assert plan.costs["anchored_tool_encoder_tokens"] == 12
    assert plan.costs["resident_kv_tokens"] == 10
    assert plan.costs["system_prefix_kv_tokens"] + plan.costs["workspace_resident_kv_tokens"] + plan.costs["gist_prefix_kv_tokens"] == 10


def test_raw_tool_workspace_slots_replace_only_named_span():
    plan = native_plan(raw_tool_segments=[{"token_start": 7, "token_end": 11,
        "token_len": 2, "repair_key_hashes": ["repair"]}])
    assert list(plan.logical_input_ids) == list(range(12))
    assert plan.costs["resident_kv_tokens"] == 7
    assert plan.costs["system_tokens"] == 2
    assert plan.costs["raw_tokens"] == 6
    assert plan.costs["workspace_resident_kv_tokens"] == 4


def test_raw_tool_cannot_overlap_compressed_history():
    with pytest.raises(ValueError, match="C2KV_NATIVE_RAW_TOOL_SPAN_OVERLAP"):
        native_plan(raw_tool_segments=[{"token_start": 3, "token_end": 5,
            "token_len": 1, "repair_key_hashes": ["repair"]}])


def test_tool_ratio_does_not_change_history_handles():
    history = native_chunk("history", [2, 3, 4, 5], 2)
    first = native_plan(encoder_chunks=[history])
    tool = native_chunk("tool", list(range(20, 28)), 6, ratio=2, tool=True)
    second = native_plan(tool_gist_segments=[{"token_start": 6, "token_end": 8, "chunk": tool}])
    assert first.selected_handles[0] == second.selected_handles[0]
    assert second.costs["anchored_tool_gist_tokens"] == 4


def test_session_restore_keeps_history_holes_and_appends_tool_rounds(engine, monkeypatch):
    Cache = extract_class("mem_cache/session_aware_cache.py", "SessionAwareCache", {"match_prefix"},
        extra={"MatchPrefixParams": object, "MatchResult": SimpleNamespace, "_is_streaming": lambda req: True})
    req = request(start=7, end=7, persistent=True)
    # A previous turn keeps seven physical positions from a twenty-token
    # source. The new document is appended immediately after that prefix.
    previous = [0, 1, 5, 9, 13, 18, 19]
    req.origin_input_ids = [0, 1, -100, -101, 13, 18, 19, 40, 41, 42]
    req.history_kv_eviction.update(persistent_continuation=True,
        persistent_protected_prefix_tokens=10, persistent_canonical_history_end=29,
        persistent_delta_history_tokens=1)
    req.c2kv_kv_memory_hint.update(persistent_session_logical_prefix_tokens=20,
        persistent_session_canonical_prompt_tokens=31)
    req.session = SimpleNamespace(session_id="session")
    assert engine._build_c2kv_prefill_rounds(req) is None
    assert req.c2kv_composition_pending
    state = {0: {13: 4.0}}
    def restore(current):
        current.kv_committed_len = 7
        current.req_pool_idx = 0
        current.c2kv_position_correction = 13
        current.c2kv_tool_source_spans = [(2, 10)]
        current.history_kv_score_state = state
    slot = SimpleNamespace(req_pool_idx=0, history_kv_resident_positions=previous,
        restore_to_req=restore, virtual_node=object(), cache_protected_len=0)
    cache = Cache()
    cache.slots = {"session": slot}
    cache.req_to_token_pool = SimpleNamespace(req_to_token=torch.arange(32).reshape(1, 32))
    cache.match_prefix(SimpleNamespace(req=req, key=SimpleNamespace(token_ids=req.origin_input_ids)))
    assert req.history_kv_resident_positions == previous + [23, 27, 28, 29, 30]
    assert req.history_kv_score_state is state
    assert req.c2kv_tool_source_spans == [(2, 10), (20, 28)]
    assert req.history_kv_eviction["protected_history_indices"] == [3, 4]
    assert req.c2kv_rounds[0].tokens == req.origin_input_ids[:7]
    assert req.c2kv_rounds[0].post_inject_seg_indices == [0]
    assert req.c2kv_rounds[-1].tokens == [41, 42]
    assert req.c2kv_rounds[-1].post_history_kv_eviction
    assert len(req.c2kv_virtual_input_ids) == len(req.history_kv_resident_positions)


def test_canonical_source_and_history_measurement_views_are_independent():
    Chat = extract_class("entrypoints/openai/serving_chat.py", "OpenAIServingChat", {
        "_paper_source_request", "_paper_history_request"}, extra={"copy": copy})
    class Request(SimpleNamespace):
        def model_dump(self):
            return copy.deepcopy(self.__dict__)
        @classmethod
        def model_validate(cls, payload):
            return cls(**payload)
    source = [{"role": "system", "content": "raw tool source with its original wrapper"},
              {"role": "user", "content": "previous history"},
              {"role": "user", "content": "current query"}]
    req = Request(messages=[SimpleNamespace(content="system"),
        SimpleNamespace(content="carrier envelope", c2kv_region="tool", c2kv_key_hash="gist"),
        SimpleNamespace(content="previous history"), SimpleNamespace(content="current query")], tools=[],
        c2kv_kv_memory_hint={"paper_measurement": {"history_start_message_count": 2,
            "history_message_count": 3, "canonical_source_messages": source, "canonical_source_tools": []},
            "history_kv_eviction": {"target_tokens": 7}})
    generation_before = copy.deepcopy(req.__dict__)
    history = Chat._paper_history_request(req)
    canonical = Chat._paper_source_request(req, req.c2kv_kv_memory_hint["paper_measurement"])
    assert canonical.messages == source
    assert canonical.tools == []
    assert [item.content for item in history.messages] == ["system", "previous history", "current query"]
    assert history.c2kv_kv_memory_hint["paper_measurement"]["history_start_message_count"] == 1
    assert history.c2kv_kv_memory_hint["paper_measurement"]["history_message_count"] == 2
    assert req.__dict__ == generation_before
    assert history.c2kv_kv_memory_hint["history_kv_eviction"]["target_tokens"] == 7


def test_tool_accounting_does_not_change_history_denominator_or_residency(engine, monkeypatch):
    accounting = load("test_composition_accounting", "managers/c2kv_kv_accounting.py")
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.c2kv_kv_accounting", accounting)
    req = SimpleNamespace(c2kv_active_region="tool", kv_memory_report={
        "full_equivalent_history_tokens": 100, "active_history_kv_tokens": 10})
    engine._add_c2kv_kv_memory_tokens(req, kind="gist", tokens=4, original_tokens=32)
    assert req.kv_memory_report["active_history_kv_tokens"] == 10
    assert req.kv_memory_report["full_equivalent_history_tokens"] == 100
    assert req.kv_memory_report["active_tool_kv_tokens"] == 4
    assert req.kv_memory_report["active_tool_gist_tokens"] == 4
    assert req.kv_memory_report["tool_encoder_source_tokens"] == 32
