"""CPU integration contracts for CommitKV's serving capture and builder."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from sglang.srt.mem_cache.commitkv import (  # noqa: E402
    CommitKVConfig,
    CommitKVRuntimeState,
    EventPage,
)
from sglang.srt.mem_cache.history_kv_reference import (  # noqa: E402
    CommitKVServingState,
)


def _extract_method(path: Path, class_name: str, method_name: str):
    """Load one production method without importing the full serving stack."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "torch": torch,
        "ForwardBatch": object,
        "paper_telemetry": SimpleNamespace(sample=lambda *args, **kwargs: None),
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method_name]


QWEN_CAPTURE = _extract_method(
    ROOT / "python" / "sglang" / "srt" / "models" / "qwen3.py",
    "Qwen3Attention",
    "_capture_history_kv_runtime_queries",
)
QWEN_NORMAL_POSITIONS = _extract_method(
    ROOT / "python" / "sglang" / "srt" / "models" / "qwen3.py",
    "Qwen3Attention",
    "_reference_normal_positions",
)
SCHEDULER_BUILD = _extract_method(
    ROOT / "python" / "sglang" / "srt" / "managers" / "scheduler.py",
    "Scheduler",
    "_build_commitkv_reference_state",
)


class _ForwardMode:
    def __init__(self, mode: str):
        self.mode = mode

    def is_decode(self) -> bool:
        return self.mode == "decode"

    def is_extend_or_draft_extend_or_mixed(self) -> bool:
        return self.mode == "extend"


class _KVPool:
    def __init__(self, keys: list[torch.Tensor], values: list[torch.Tensor]):
        self.start_layer = 0
        self.layer_num = len(keys)
        self._keys = keys
        self._values = values

    def get_kv_buffer(self, layer_id: int):
        return self._keys[layer_id], self._values[layer_id]


def test_new_transition_is_not_consumed_while_previous_post_window_is_pending():
    policy = SimpleNamespace(
        pending=object(), config=SimpleNamespace(page_size=16)
    )
    state = CommitKVServingState(
        policy=policy,
        target_tokens=128,
        event_signature=((1, "tool", "tool", 16, 32),),
        pending_commit_id=1,
    )
    next_spans = [
        {"message_index": 1, "role": "tool", "phase": "tool", "start": 16, "end": 32},
        {"message_index": 2, "role": "tool", "phase": "tool", "start": 32, "end": 48},
    ]
    with pytest.raises(RuntimeError, match="OVERLAPPING_TRANSITIONS_UNSUPPORTED"):
        state.configure_events(next_spans)
    assert state.event_signature == ((1, "tool", "tool", 16, 32),)


def test_long_history_scans_latest_fully_resident_pages_under_project_cap():
    config = CommitKVConfig(
        window_size=1,
        page_size=1,
        max_scanned_pages=64,
        measurement_layer_id=0,
    )
    policy = CommitKVRuntimeState(config)
    state = CommitKVServingState(policy=policy, target_tokens=128)
    existing = tuple(
        (index, "tool", "tool", index, index + 1) for index in range(70)
    )
    state.event_signature = existing
    state.event_pages = tuple(
        EventPage(index, 0, index, index + 1) for index in range(70)
    )
    state.record_decode_window(
        torch.ones(1, 1, 1),
        torch.tensor([70]),
        torch.ones(1, 70, 1),
        torch.arange(70, dtype=torch.float32).reshape(1, 70, 1),
        torch.arange(70),
        scale=1.0,
    )
    assert len(state.pre_pages) == 64
    assert [page.start for page in state.pre_pages] == list(range(6, 70))
    assert state.pre_scan_metadata["scan_truncated_pages"] == 6

    next_spans = [
        {
            "message_index": index,
            "role": "tool",
            "phase": "tool",
            "start": index,
            "end": index + 1,
        }
        for index in range(71)
    ]
    state.configure_events(next_spans)
    receipt = state.receipts[-1]
    assert receipt["scanned_pages"] == 64
    assert receipt["scan_policy"] == (
        "latest_fully_resident_pages_project_convention"
    )


def _fake_attention(layer_id: int = 0):
    attention = SimpleNamespace(
        attn=SimpleNamespace(layer_id=layer_id),
        num_heads=1,
        num_kv_heads=1,
        head_dim=1,
        scaling=1.0,
    )
    attention._reference_normal_positions = QWEN_NORMAL_POSITIONS
    attention._capture_history_kv_runtime_queries = MethodType(
        QWEN_CAPTURE, attention
    )
    return attention


def _forward_batch(
    *,
    mode: str,
    config: dict,
    state: CommitKVServingState,
    position: int | list[int],
    seq_len: int,
    req_to_token: torch.Tensor,
    kv_pool: _KVPool,
):
    positions = [position] if isinstance(position, int) else list(position)
    batch = SimpleNamespace(
        history_kv_reference_configs=[config],
        history_kv_runtime_states=[state],
        history_kv_reference_states=[None],
        history_kv_resident_positions=[list(range(seq_len))],
        forward_mode=_ForwardMode(mode),
        batch_size=1,
        extend_seq_lens_cpu=([len(positions)] if mode == "extend" else None),
        input_ids=torch.arange(len(positions), dtype=torch.long),
        seq_lens=torch.tensor([seq_len], dtype=torch.long),
        req_pool_indices=torch.tensor([0], dtype=torch.long),
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        token_to_kv_pool=kv_pool,
    )
    return torch.tensor(positions, dtype=torch.long), batch


def test_qwen_commitkv_capture_excludes_prompt_and_pairs_cross_turn_windows():
    policy = CommitKVRuntimeState(
        CommitKVConfig(
            measurement_layer_id=0,
            window_size=2,
            page_size=1,
            pending_fraction=0.5,
            max_scanned_pages=8,
            max_pending_pages=2,
        )
    )
    state = CommitKVServingState(policy=policy, target_tokens=4)
    attention = _fake_attention()
    req_to_token = torch.arange(16, dtype=torch.long).view(1, -1)
    key = torch.linspace(0.1, 1.6, 16).view(16, 1, 1)
    value = torch.linspace(1.0, 16.0, 16).view(16, 1, 1)
    kv_pool = _KVPool([key], [value])
    config = {
        "method": "commitkv",
        "event_token_spans": [
            {
                "message_index": 1,
                "role": "assistant",
                "phase": "act",
                "start": 0,
                "end": 1,
            }
        ],
    }

    # Prompt/extend queries configure event boundaries but must never enter
    # either CommitKV measurement window.
    positions, batch = _forward_batch(
        mode="extend",
        config=config,
        state=state,
        position=[0, 1, 2],
        seq_len=3,
        req_to_token=req_to_token,
        kv_pool=kv_pool,
    )
    attention._capture_history_kv_runtime_queries(
        torch.ones(3, 1), torch.ones(3, 1), torch.ones(3, 1), positions, batch
    )
    assert state.pre_queries == []
    assert state.post_queries == []

    # The final W decoded queries become the rolling pre-commit window.
    for position in (3, 4):
        positions, batch = _forward_batch(
            mode="decode",
            config=config,
            state=state,
            position=position,
            seq_len=position + 1,
            req_to_token=req_to_token,
            kv_pool=kv_pool,
        )
        attention._capture_history_kv_runtime_queries(
            torch.tensor([[float(position)]]),
            torch.tensor([[float(position) / 10]]),
            torch.tensor([[float(position)]]),
            positions,
            batch,
        )
    assert torch.cat(state.pre_positions).tolist() == [3, 4]
    assert state.pre_window.query_positions.tolist() == [3, 4]

    # The next prompt exposes a newly completed tool event. It seals the old
    # rolling pre window but contributes no query to the post window.
    config["event_token_spans"] = [
        *config["event_token_spans"],
        {
            "message_index": 2,
            "role": "tool",
            "phase": "tool",
            "start": 5,
            "end": 7,
        },
    ]
    positions, batch = _forward_batch(
        mode="extend",
        config=config,
        state=state,
        position=[5, 6],
        seq_len=7,
        req_to_token=req_to_token,
        kv_pool=kv_pool,
    )
    attention._capture_history_kv_runtime_queries(
        torch.full((2, 1), 99.0),
        torch.ones(2, 1),
        torch.ones(2, 1),
        positions,
        batch,
    )
    assert state.pending_commit_id == 2
    assert state.post_queries == []
    assert policy.pending is not None
    assert policy.pending.commit_id == 2

    # Only the first W decoded queries after the prompt form the post window.
    for position in (7, 8):
        positions, batch = _forward_batch(
            mode="decode",
            config=config,
            state=state,
            position=position,
            seq_len=position + 1,
            req_to_token=req_to_token,
            kv_pool=kv_pool,
        )
        attention._capture_history_kv_runtime_queries(
            torch.tensor([[float(position)]]),
            torch.tensor([[float(position) / 10]]),
            torch.tensor([[float(position)]]),
            positions,
            batch,
        )
        if position == 7:
            assert torch.cat(state.post_positions).tolist() == [7]
            assert policy.pending is not None
            assert policy.completed_transitions == 0
    assert state.pending_commit_id is None
    assert state.post_queries == []
    assert policy.pending is None
    assert policy.completed_transitions == 1
    assert [receipt["measurement_phase"] for receipt in state.receipts] == [
        "pre_commit",
        "post_commit",
    ]
    assert torch.cat(state.pre_positions).tolist() == [7, 8]


class _FixedEffectWindow:
    def __init__(self, key_positions: list[int]):
        self.key_positions = torch.tensor(key_positions, dtype=torch.long)

    def effect(self, indices):
        return torch.tensor(0.5 + 0.01 * len(tuple(indices)))


def test_scheduler_commitkv_builder_protects_pending_and_uses_common_indices():
    policy = CommitKVRuntimeState(
        CommitKVConfig(
            measurement_layer_id=1,
            page_size=1,
            pending_fraction=0.5,
            max_pending_pages=1,
        )
    )
    policy.record_pre(
        "commit",
        [EventPage("old-action", 0, 0, 1)],
        _FixedEffectWindow([0, 1, 2, 3]),
        [0, 1, 2, 3],
        total_budget=2,
    )
    serving_state = CommitKVServingState(policy=policy, target_tokens=2)
    keys = [
        (100 * layer + torch.arange(4, dtype=torch.float32)).view(4, 1, 1)
        for layer in range(2)
    ]
    values = [item + 1000 for item in keys]
    kv_pool = _KVPool(keys, values)
    scheduler = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange(4, dtype=torch.long).view(1, -1)
        ),
        token_to_kv_pool_allocator=SimpleNamespace(
            get_kvcache=lambda: kv_pool
        ),
    )
    scheduler._build_commitkv_reference_state = MethodType(
        SCHEDULER_BUILD, scheduler
    )
    req = SimpleNamespace(
        req_pool_idx=0,
        history_kv_runtime_state=serving_state,
        history_kv_reference_state=None,
        history_kv_resident_positions=[0, 1, 2, 3],
    )
    config = {
        "history_start": 0,
        "history_end": 4,
        "target_tokens": 2,
    }

    state = scheduler._build_commitkv_reference_state(req, config)

    state.validate()
    assert state.expected_layer_ids == (0, 1)
    assert state.selection_metadata["baseline_policy"] == (
        "most_recent_first_project_convention"
    )
    assert state.selection_metadata["protected_pending_page_count"] == 1
    for layer_id, layer in state.layers.items():
        # Pending position 0 survives even though the explicit base policy is
        # most-recent-first. All heads and layers use the same [0, 3] indices.
        assert layer.positions.tolist() == [[0, 3]]
        torch.testing.assert_close(
            layer.key[:, :, 0],
            torch.tensor([[100.0 * layer_id, 100.0 * layer_id + 3.0]]),
        )
        torch.testing.assert_close(layer.value, layer.key + 1000)
    assert state.layers[0].positions.tolist() == state.layers[1].positions.tolist()
    assert config["commitkv_baseline_policy"] == (
        "most_recent_first_project_convention"
    )
