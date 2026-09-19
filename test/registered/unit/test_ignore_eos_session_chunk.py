"""Keep the resident session prefix when ignore-EOS prefill is chunked."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/managers/schedule_policy.py"
)


def _add_ignore_eos():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    method = next(
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "PrefillAdder"
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "add_one_req_ignore_eos"
    )
    namespace = {"Req": object, "CLIP_MAX_NEW_TOKENS": 256}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["add_one_req_ignore_eos"]


@pytest.mark.parametrize("chunk_size,expected_extend", [(4, 4), (None, 9)])
def test_ignore_eos_continuation_counts_only_new_tokens(chunk_size, expected_extend):
    add_ignore_eos = _add_ignore_eos()
    budgets = []
    adder = SimpleNamespace(
        ceil_paged_tokens=lambda count: count,
        cur_rem_tokens=1000,
        rem_total_tokens=1000,
        req_states=None,
        running_batch=None,
        can_run_list=[],
        new_token_ratio=1.0,
        is_hybrid_swa=True,
        dllm_config=None,
        rem_chunk_tokens=chunk_size,
        new_chunked_req=None,
        _update_prefill_budget=lambda *args: budgets.append(args),
        budget_state=lambda: "scheduled",
    )
    req = SimpleNamespace(
        prefix_indices=[90, 91, 92],
        fill_ids=list(range(12)),
        extend_input_len=9,
        origin_input_ids=list(range(12)),
        output_ids=[],
        sampling_params=SimpleNamespace(ignore_eos=True, max_new_tokens=32),
    )
    req.set_extend_input_len = lambda count: setattr(req, "extend_input_len", count)

    assert add_ignore_eos(adder, req) == "scheduled"
    assert req.fill_ids == list(range(3 + expected_extend))
    assert req.extend_input_len == expected_extend
    assert budgets == [(3, expected_extend, 0 if chunk_size else 32)]
    assert adder.can_run_list == [req]
