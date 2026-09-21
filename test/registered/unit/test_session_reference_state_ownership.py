import gc
import ast
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch

ROOT = Path(__file__).resolve().parents[3]
source = ROOT / "python/sglang/srt/mem_cache/session_aware_cache.py"
tree = ast.parse(source.read_text(encoding="utf-8"))
nodes = [
    node
    for node in tree.body
    if isinstance(node, ast.ClassDef)
    and node.name in {"_VirtualNode", "SessionSlot"}
]
namespace = {
    "dataclass": dataclass,
    "field": field,
    "Any": Any,
    "Optional": Optional,
    "Req": object,
}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
SessionSlot = namespace["SessionSlot"]


class _State:
    def __init__(self):
        self.tensor = torch.zeros(1)


def _req(state):
    return SimpleNamespace(
        req_pool_idx=0,
        kv_committed_len=1,
        kv_allocated_len=1,
        swa_evicted_seqlen=0,
        c2kv_position_correction=0,
        history_kv_resident_positions=[0],
        history_kv_score_state={},
        history_kv_reference_state=state,
        history_kv_reference_config={"method": "pyramidkv"},
        history_kv_runtime_state=state,
        last_node=None,
        cache_protected_len=0,
        swa_uuid_for_lock=None,
        mamba_pool_idx=None,
        mamba_ping_pong_track_buffer=None,
        mamba_next_track_idx=None,
        mamba_last_track_seqlen=None,
        mamba_branching_seqlen=None,
    )


def test_session_slot_transfer_does_not_pin_every_historical_request_state():
    slot = SessionSlot()
    first = _State()
    first_ref = weakref.ref(first)
    first_req = _req(first)
    slot.save_from_req(first_req, is_first=True)
    assert first_req.history_kv_reference_state is None
    assert first_req.history_kv_runtime_state is None

    second = _State()
    second_req = _req(second)
    slot.save_from_req(second_req, is_first=False)
    del first
    gc.collect()
    assert first_ref() is None
    assert slot.history_kv_reference_state is second
    assert slot.history_kv_runtime_state is second
