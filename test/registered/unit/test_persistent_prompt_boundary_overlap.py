"""CPU regressions for persistent session prompt boundaries under overlap."""

import ast
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "python/sglang/srt/mem_cache"
SESSION_CACHE = CACHE / "session_aware_cache.py"
OUTPUT_PROCESSOR = ROOT / "python/sglang/srt/managers/scheduler_output_processor_mixin.py"


def _method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == class_name
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method_name]


def _prefill_decode_boundary(req):
    """Execute the two real assignments in the prefill-result handler."""
    tree = ast.parse(OUTPUT_PROCESSOR.read_text(encoding="utf-8"))
    handler = next(
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        and cls.name == "SchedulerOutputProcessorMixin"
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "process_batch_result_prefill"
    )
    wanted = {"reference_decode_protected_len", "reference_decode_logical_start"}
    assignments = [
        node
        for node in ast.walk(handler)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr in wanted
    ]
    assert {node.targets[0].attr for node in assignments} == wanted
    assignments.sort(key=lambda node: node.lineno)
    exec(
        compile(ast.Module(body=assignments, type_ignores=[]), str(OUTPUT_PROCESSOR), "exec"),
        {"req": req},
    )


def test_overlap_decode_reservation_does_not_enter_protected_prompt():
    # prepare_for_decode increments this mutable length before the preceding
    # prefill result is processed by the overlap scheduler.
    req = SimpleNamespace(
        origin_input_ids=list(range(128)),
        kv_committed_len=129,
        c2kv_position_correction=17,
    )
    _prefill_decode_boundary(req)
    assert req.reference_decode_protected_len == 128
    assert req.reference_decode_logical_start == 145


def test_first_turn_nonexact_session_discards_overlap_decode_and_saves_prompt(monkeypatch):
    spec = importlib.util.spec_from_file_location("overlap_history_ledger", CACHE / "history_kv_lifecycle.py")
    ledger = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ledger)
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.history_kv_lifecycle", ledger)

    discard = _method(
        SESSION_CACHE,
        "SessionAwareCache",
        "_discard_persistent_decode_suffix",
        {"torch": torch, "Req": object},
    )

    class Slot:
        def save_from_req(self, req, is_first):
            self.positions = list(req.history_kv_resident_positions)
            self.kv_committed_len = req.kv_committed_len

    finished = _method(
        SESSION_CACHE,
        "SessionAwareCache",
        "cache_finished_req",
        {
            "Req": object,
            "_is_streaming": lambda req: True,
            "SessionSlot": Slot,
            "json": json,
            "logging": logging,
        },
    )
    row = torch.zeros((1, 12), dtype=torch.int64)
    row[0, :7] = torch.arange(40, 47)
    freed = []
    owner = SimpleNamespace(
        slots={},
        req_to_token_pool=SimpleNamespace(req_to_token=row),
        token_to_kv_pool_allocator=SimpleNamespace(
            free=lambda slots: freed.extend(slots.tolist())
        ),
        page_size=1,
        _is_persistent_history_req=lambda req: True,
    )
    owner._discard_persistent_decode_suffix = lambda req: discard(owner, req)
    req = SimpleNamespace(
        session=SimpleNamespace(session_id="s", streaming=True),
        c2kv_kv_memory_hint={
            "persistent_session_logical_prefix_tokens": 0,
            "persistent_session_canonical_prompt_tokens": 5,
            "persistent_session_delta_tokens": 5,
        },
        origin_input_ids=[10, 11, 12, 13, 14],
        history_kv_resident_positions=None,
        history_kv_reference_config={"method": "pyramidkv"},
        req_pool_idx=0,
        kv_committed_len=7,
        kv_allocated_len=7,
        reference_decode_protected_len=6,
        persistent_decode_cache_locs=[],
        kv_memory_report={},
    )

    finished(owner, req)

    assert owner.slots["s"].positions == [0, 1, 2, 3, 4]
    assert owner.slots["s"].kv_committed_len == 5
    assert freed == [45, 46]
    assert row[0, 5:7].tolist() == [0, 0]
    event = req.kv_memory_report["history_kv_lifecycle"]
    assert event["resident_tokens_after_eviction"] == 5
    assert event["resident_decode_normal_tokens"] == 0


def test_exact_generated_prefix_starts_at_canonical_prompt_after_overlap():
    req = SimpleNamespace(
        origin_input_ids=[10, 11, 12],
        kv_committed_len=4,  # prompt 3 + overlapping decode reservation
        c2kv_position_correction=2,
    )
    _prefill_decode_boundary(req)
    assert req.reference_decode_protected_len == 3
    assert req.reference_decode_logical_start == 5

    row = torch.zeros((1, 8), dtype=torch.int64)
    row[0, :5] = torch.arange(40, 45)
    discard = _method(
        SESSION_CACHE,
        "SessionAwareCache",
        "_discard_persistent_decode_suffix",
        {"torch": torch, "Req": object},
    )
    owner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=row),
        token_to_kv_pool_allocator=SimpleNamespace(free=lambda _: None),
        page_size=1,
    )
    req.kv_committed_len = req.kv_allocated_len = 5
    req.history_kv_resident_positions = [0, 1, 2]
    req.history_kv_reference_config = {"method": "agentkv"}
    req.output_ids = [100, 101, 102]
    req.req_pool_idx = 0
    req.persistent_decode_cache_locs = []
    req.kv_memory_report = {}

    discard(owner, req)

    assert req.history_kv_resident_positions == [0, 1, 2, 5, 6]
    assert req.persistent_session_active_output_ids == [100, 101, 102]
    assert req.kv_memory_report["persistent_session_computed_logical_horizon"] == 7
