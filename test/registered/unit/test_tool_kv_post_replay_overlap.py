"""Regress the tool-KV first-token receipt against overlap decode reservation."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


OUTPUT_PROCESSOR = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/managers/scheduler_output_processor_mixin.py"
)


class _PastFirstTokenReceipt(Exception):
    """Stop after the real handler validates and records its first-token receipt."""


def _prefill_handler(paper_telemetry):
    tree = ast.parse(OUTPUT_PROCESSOR.read_text(encoding="utf-8"))
    method = next(
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef)
        and cls.name == "SchedulerOutputProcessorMixin"
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "process_batch_result_prefill"
    )
    method = copy.deepcopy(method)
    method.decorator_list = []
    method.returns = None
    for argument in (
        method.args.posonlyargs + method.args.args + method.args.kwonlyargs
    ):
        argument.annotation = None
    namespace = {"paper_telemetry": paper_telemetry}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(OUTPUT_PROCESSOR), "exec"),
        namespace,
    )
    return namespace["process_batch_result_prefill"]


def _run_first_token_prefill_result(resident_tokens_by_layer):
    observed = []

    def mark_generation_start(req, **kwargs):
        observed.append({"req": req, **kwargs})
        raise _PastFirstTokenReceipt

    handler = _prefill_handler(
        SimpleNamespace(mark_generation_start=mark_generation_start)
    )
    reference_state = SimpleNamespace(
        layers={
            layer_id: SimpleNamespace(key=torch.empty((2, 4, 8)))
            for layer_id in (0, 1)
        }
    )
    receipt = {
        "no_op": False,
        "resident_tokens_by_layer": list(resident_tokens_by_layer),
        "first_token_after_selection": False,
    }
    snapshot = {"tool_kv_eviction": dict(receipt)}
    req = SimpleNamespace(
        rid="tool-kv-overlap",
        is_retracted=False,
        finished=lambda: False,
        c2kv_rounds=[
            SimpleNamespace(post_inject_seg_indices=[], post_history_kv_eviction=True),
            SimpleNamespace(post_inject_seg_indices=[], post_history_kv_eviction=False),
        ],
        c2kv_round_idx=1,
        is_chunked=0,
        time_stats=SimpleNamespace(set_prefill_finished_time=lambda: None),
        c2kv_virtual_input_ids=list(range(125)),
        c2kv_persistent_active_input_ids=None,
        origin_input_ids=list(range(650)),
        c2kv_position_correction=525,
        history_kv_reference_config=None,
        history_kv_reference_state=reference_state,
        kv_memory_report={"tool_kv_eviction": receipt},
        history_kv_eviction_report_snapshot=snapshot,
        # The next decode was reserved before the preceding prefill result was
        # processed. Its KV is not part of the 125-token completed prompt.
        kv_committed_len=126,
        output_ids=[],
    )
    batch = SimpleNamespace(
        reqs=[req],
        return_logprob=False,
        seq_lens_cpu=torch.tensor([125]),
    )
    result = SimpleNamespace(
        copy_done=None,
        logits_output=SimpleNamespace(),
        next_token_ids=torch.tensor([17]),
        extend_input_len_per_req=[1],
        extend_logprob_start_len_per_req=[0],
    )
    scheduler = SimpleNamespace(
        is_generation=True,
        _log_c2kv_token_usage=lambda *args, **kwargs: None,
        _release_c2kv_pins=lambda *args, **kwargs: None,
        token_to_kv_pool_allocator=SimpleNamespace(
            get_kvcache=lambda: SimpleNamespace(layer_num=2)
        ),
    )
    return handler, scheduler, batch, result, req, receipt, snapshot, observed


def test_tool_kv_post_replay_receipt_uses_completed_prefill_under_overlap():
    # Each layer has 125 ordinary prompt tokens and 4 reference schema tokens.
    state = _run_first_token_prefill_result([129, 129])
    handler, scheduler, batch, result, req, receipt, snapshot, observed = state

    with pytest.raises(_PastFirstTokenReceipt):
        handler(scheduler, batch, result)

    assert req.kv_committed_len == 126
    assert batch.seq_lens_cpu.tolist() == [125]
    assert receipt["first_token_after_selection"] is True
    assert snapshot["tool_kv_eviction"]["first_token_after_selection"] is True
    assert len(observed) == 1
    assert observed[0]["normal_kv_tokens"] == 125


def test_tool_kv_post_replay_rejects_real_resident_mismatch_under_overlap():
    state = _run_first_token_prefill_result([128, 129])
    handler, scheduler, batch, result, _, receipt, snapshot, observed = state

    with pytest.raises(RuntimeError, match="TOOL_KV_POST_REPLAY_RESIDENT_MISMATCH"):
        handler(scheduler, batch, result)

    assert receipt["first_token_after_selection"] is False
    assert snapshot["tool_kv_eviction"]["first_token_after_selection"] is False
    assert observed == []
