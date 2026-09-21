"""Cross-turn ordering contracts for reference-history selection axes."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from sglang.srt.mem_cache.agentkv import AgentKVQueryRing  # noqa: E402
from sglang.srt.mem_cache.commitkv import (  # noqa: E402
    CommitKVConfig,
    CommitKVRuntimeState,
)
from sglang.srt.mem_cache.history_kv_reference import (  # noqa: E402
    CommitKVServingState,
    ReferenceHistoryKVState,
    ReferenceLayerKV,
    gather_reference_candidates,
    merge_reference_candidates,
)


def _scheduler_method(name: str):
    path = ROOT / "python/sglang/srt/managers/scheduler.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    scheduler = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    method = next(
        node
        for node in scheduler.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    namespace = {"Req": object, "torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class _KVPool:
    start_layer = 0

    def __init__(self, keys: list[torch.Tensor], values: list[torch.Tensor]):
        self.keys = keys
        self.values = values
        self.layer_num = len(keys)

    def get_kv_buffer(self, layer_id: int):
        return self.keys[layer_id], self.values[layer_id]


def _existing(method: str) -> ReferenceHistoryKVState:
    layer = ReferenceLayerKV(
        key=torch.tensor([[[10.0], [20.0]]]),
        value=torch.tensor([[[110.0], [120.0]]]),
        positions=torch.tensor([[10, 20]], dtype=torch.long),
    )
    return ReferenceHistoryKVState(
        method=method,
        layers={0: layer},
        expected_layer_ids=(0,),
    )


def _owner():
    keys = [torch.tensor([[[0.0]], [[5.0]], [[25.0]]])]
    values = [keys[0] + 100]
    pool = _KVPool(keys, values)
    return SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange(3, dtype=torch.long).view(1, 3)
        ),
        token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: pool),
    )


def test_merge_precedes_selection_when_external_positions_follow_normal_prefix():
    existing = _existing("agentkv").layer(0)
    normal_key = torch.tensor([[[0.0]], [[5.0]], [[25.0]]])
    normal_value = normal_key + 100

    key, value, positions = merge_reference_candidates(
        existing, normal_key, normal_value, [0, 5, 25]
    )

    assert positions.tolist() == [[0, 5, 10, 20, 25]]
    assert key[0, :, 0].tolist() == [0, 5, 10, 20, 25]
    assert value[0, :, 0].tolist() == [100, 105, 110, 120, 125]
    selected = gather_reference_candidates(
        existing,
        normal_key,
        normal_value,
        [0, 5, 25],
        torch.tensor([[4, 0, 2]], dtype=torch.long),
    )
    assert selected.positions.tolist() == [[0, 10, 25]]
    assert selected.key[0, :, 0].tolist() == [0, 10, 25]


def test_agentkv_sink_selection_uses_canonical_candidate_axis():
    ring = AgentKVQueryRing()
    ring.write_layer(
        layer_id=0,
        query=torch.ones(1, 1, 1),
        positions=torch.tensor([30], dtype=torch.long),
        stage_ids=torch.tensor([2], dtype=torch.int32),
    )
    req = SimpleNamespace(
        req_pool_idx=0,
        history_kv_runtime_state=ring,
        history_kv_reference_state=_existing("agentkv"),
        history_kv_resident_positions=[0, 5, 25],
    )

    state = _scheduler_method("_build_agentkv_reference_state")(
        _owner(),
        req,
        {"history_start": 0, "history_end": 3, "target_tokens": 2},
    )

    # AgentKV's sink rule selects the first two *canonical* tokens.  The old
    # external+normal concatenation incorrectly selected positions 10 and 20.
    assert state.layer(0).positions.tolist() == [[0, 5]]


def test_commitkv_recency_baseline_uses_canonical_candidate_axis():
    policy = CommitKVRuntimeState(
        CommitKVConfig(measurement_layer_id=0, page_size=1)
    )
    req = SimpleNamespace(
        req_pool_idx=0,
        history_kv_runtime_state=CommitKVServingState(
            policy=policy, target_tokens=2
        ),
        history_kv_reference_state=_existing("commitkv"),
        history_kv_resident_positions=[0, 5, 25],
    )

    state = _scheduler_method("_build_commitkv_reference_state")(
        _owner(),
        req,
        {"history_start": 0, "history_end": 3, "target_tokens": 2},
    )

    # The documented project baseline is newest first.  On the canonical
    # axis, the two newest resident positions are 20 and 25.
    assert state.layer(0).positions.tolist() == [[20, 25]]
