"""Recovery must not free a page still containing retained lossy history."""
import ast
from pathlib import Path
from types import SimpleNamespace

import torch


def test_shared_tail_page_survives_recovery_and_scheduler_retry():
    source = (Path(__file__).resolve().parents[3] /
              "python/sglang/srt/mem_cache/session_aware_cache.py")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "SessionAwareCache")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name == "_trim_persistent_generation_prefix")
    namespace = {"torch": torch, "SessionSlot": SimpleNamespace,
                 "Req": SimpleNamespace}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
         namespace)
    trim = namespace[node.name]
    # Physical page 10 holds both retained body and removable scaffold.
    # Page 11 holds only scaffold. Logical positions 1 and 3 remain absent.
    row = torch.tensor([[40, 41, 42, 43, 44, 45, 0, 0]])
    freed = []
    owner = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(req_to_token=row), page_size=4,
        token_to_kv_pool_allocator=SimpleNamespace(
            free=lambda indices: freed.extend(indices.tolist())))
    slot = SimpleNamespace(
        req_pool_idx=0, kv_committed_len=6, kv_allocated_len=6,
        history_kv_resident_positions=[0, 2, 4, 5, 6, 7],
        history_kv_score_state={0: {0: 1., 4: 2., 5: 3.}},
        c2kv_position_correction=2)
    req = SimpleNamespace(
        session=SimpleNamespace(session_id="recovery"), kv_memory_report={},
        c2kv_kv_memory_hint={
            "persistent_session_logical_prefix_tokens": 5,
            "persistent_session_drop_generation_prefix_tokens": 3})

    trim(owner, slot, req)
    assert freed == [44]
    assert row[0].tolist() == [40, 41, 42, 0, 0, 0, 0, 0]
    assert slot.history_kv_resident_positions == [0, 2, 4]
    assert slot.history_kv_score_state == {0: {0: 1., 4: 2.}}
    assert slot.kv_committed_len == slot.kv_allocated_len == 3
    assert slot.c2kv_position_correction == 2

    # Retrying scheduler admission must neither release page 11 again nor
    # trim another suffix from the already compressed body.
    trim(owner, slot, req)
    assert freed == [44]
    assert slot.history_kv_resident_positions == [0, 2, 4]
    assert row[0, :3].tolist() == [40, 41, 42]
