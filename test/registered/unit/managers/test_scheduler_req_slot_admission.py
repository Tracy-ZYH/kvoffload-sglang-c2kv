"""CPU-only admission checks against the scheduler's actual prefill method."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import MagicMock


ROOT = Path(__file__).resolve().parents[4]
SCHEDULER = ROOT / "python/sglang/srt/managers/scheduler.py"
POOL = ROOT / "python/sglang/srt/mem_cache/memory_pool.py"


def _method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        item
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name == method_name
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method_name]


class _PrefillHarness:
    def __init__(self, free_slots, max_running_requests=1):
        self.free_slots = free_slots
        self.max_running_requests = max_running_requests
        self.adder = MagicMock()
        self.adder.can_run_list = []
        self.adder.preempt_list = []
        self.adder.new_chunked_req = None
        self.adder.add_one_req.side_effect = self._add_one_req
        self.adder.add_chunked_req.side_effect = self._add_chunked_req
        self.batch = MagicMock()
        self.namespace = {
            "Optional": Optional,
            "List": List,
            "Req": object,
            "ScheduleBatch": SimpleNamespace(init_new=MagicMock(return_value=self.batch)),
            "PrefillDelayerSinglePassExecutor": object,
            "PrefillAdder": MagicMock(return_value=self.adder),
            "PrefillStats": SimpleNamespace(from_adder=MagicMock()),
            "AddReqResult": SimpleNamespace(CONTINUE="continue", NO_TOKEN="no_token"),
            "DisaggregationMode": SimpleNamespace(PREFILL="prefill"),
            "TEST_RETRACT": False,
            "set_time_batch": MagicMock(),
            "get_global_server_args": lambda: SimpleNamespace(
                pp_max_micro_batch_size=max_running_requests
            ),
        }
        self.get_num = _method(
            SCHEDULER, "Scheduler", "get_num_allocatable_reqs", self.namespace
        )
        self.prefill = _method(
            SCHEDULER, "Scheduler", "_get_new_batch_prefill_raw", self.namespace
        )
        self.scheduler = self._scheduler()

    def _add_one_req(self, req, **_kwargs):
        self.adder.can_run_list.append(req)
        return "continue"

    def _add_chunked_req(self, req):
        self.adder.can_run_list.append(req)
        return None

    def _scheduler(self):
        scheduler = SimpleNamespace(
            grammar_manager=MagicMock(),
            enable_hierarchical_cache=False,
            enable_priority_preemption=False,
            running_batch=SimpleNamespace(reqs=[], batch_is_full=False),
            waiting_queue=[],
            chunked_req=None,
            pp_size=1,
            policy=MagicMock(),
            chunked_prefill_size=512,
            enable_dynamic_chunking=False,
            page_size=1,
            tree_cache=MagicMock(),
            req_to_token_pool=SimpleNamespace(
                available_size=lambda: self.free_slots
            ),
            token_to_kv_pool_allocator=MagicMock(),
            new_token_ratio=1.0,
            max_prefill_tokens=512,
            is_mixed_chunk=False,
            priority_scheduling_preemption_threshold=0,
            max_prefill_bs=0,
            max_running_requests=self.max_running_requests,
            truncation_align_size=None,
            server_args=SimpleNamespace(prefill_max_requests=None),
            dllm_config=None,
            enable_lora=False,
            disaggregation_mode=None,
            enable_hicache_storage=False,
            model_config=MagicMock(),
            enable_overlap=False,
            spec_algorithm=None,
            enable_priority_scheduling=False,
            _advance_cached_c2kv_tool_rounds=MagicMock(return_value=True),
        )
        scheduler.grammar_manager.has_waiting_grammars.return_value = False
        scheduler.get_num_allocatable_reqs = lambda running_bs: self.get_num(
            scheduler, running_bs
        )
        return scheduler

    def run(self):
        return self.prefill(self.scheduler, None)


def _req(pool_idx=None):
    req = MagicMock()
    req.req_pool_idx = pool_idx
    req.mamba_pool_idx = None
    return req


def test_new_request_waits_until_slot_is_free():
    harness = _PrefillHarness(free_slots=0)
    req = _req()
    harness.scheduler.waiting_queue = [req]

    assert harness.run() is None
    assert harness.scheduler.waiting_queue == [req]
    assert harness.scheduler.running_batch.batch_is_full is False
    harness.adder.add_one_req.assert_not_called()
    harness.batch.prepare_for_extend.assert_not_called()

    harness.free_slots = 1
    assert harness.run() is harness.batch
    assert harness.scheduler.waiting_queue == []
    harness.adder.add_one_req.assert_called_once()
    harness.batch.prepare_for_extend.assert_called_once()


def test_staged_new_requests_cannot_overbook_free_slots():
    harness = _PrefillHarness(free_slots=1, max_running_requests=2)
    first, second = _req(), _req()
    harness.scheduler.waiting_queue = [first, second]

    assert harness.run() is harness.batch
    assert harness.adder.can_run_list == [first]
    assert harness.scheduler.waiting_queue == [second]


def test_occupied_session_and_chunked_rows_can_continue():
    harness = _PrefillHarness(free_slots=0)
    new_request, continuation = _req(), _req()
    continuation.init_next_round_input.side_effect = lambda _cache: setattr(
        continuation, "req_pool_idx", 0
    )
    harness.scheduler.waiting_queue = [new_request, continuation]

    assert harness.run() is harness.batch
    assert harness.adder.can_run_list == [continuation]
    assert harness.scheduler.waiting_queue == [new_request]

    harness = _PrefillHarness(free_slots=0)
    chunked = _req(pool_idx=0)
    harness.scheduler.chunked_req = chunked
    assert harness.run() is harness.batch
    assert harness.adder.can_run_list == [chunked]

    alloc = _method(
        POOL,
        "ReqToTokenPool",
        "alloc",
        {"Optional": Optional, "List": List, "Req": object},
    )
    chunked.is_chunked = 1
    chunked.kv_committed_len = 1
    assert alloc(SimpleNamespace(free_slots=[]), [chunked]) == [0]
