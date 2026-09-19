"""CPU-only regression for a streaming request outliving its session timeout."""

import ast
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

ROOT = Path(__file__).resolve().parents[3]


class _Slots(list):
    def tolist(self):
        return list(self)


class _ReqToToken:
    def __getitem__(self, key):
        row, span = key
        assert row == 0
        return _Slots([11, 12, 13][span])


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


def test_timeout_waits_for_running_streaming_request():
    reap = _method(
        ROOT / "python/sglang/srt/managers/session_controller.py",
        "SessionController",
        "maybe_reap",
        {"logger": logging.getLogger(__name__)},
    )
    active = SimpleNamespace(finished=lambda: False)
    session = SimpleNamespace(
        streaming=True,
        req_nodes={"rid": SimpleNamespace(req=active)},
        is_timed_out=lambda: True,
    )
    closed = []
    controller = SimpleNamespace(
        sessions={"session": session},
        _last_reap_time=0.0,
        _close=closed.append,
    )
    reap(controller, 2.0)
    assert closed == []

    active.finished = lambda: True
    reap(controller, 4.0)
    assert closed == ["session"]


def test_explicit_close_passes_active_request_for_ownership_transfer():
    class Cache:
        def __init__(self):
            self.released = []

        def release_session(self, session_id, active_req=None):
            self.released.append((session_id, active_req))

    close = _method(
        ROOT / "python/sglang/srt/managers/session_controller.py",
        "SessionController",
        "_close",
        {"SessionAwareCache": Cache},
    )
    active = SimpleNamespace(
        finished=lambda: False, session=object(), multimodal_inputs=None
    )
    session = SimpleNamespace(
        streaming=True, req_nodes={"rid": SimpleNamespace(req=active)}
    )
    cache = Cache()
    controller = SimpleNamespace(sessions={"session": session}, tree_cache=cache)
    close(controller, "session")
    assert active.session is None
    assert cache.released == [("session", active)]
    assert controller.sessions == {}


def test_explicit_close_transfers_borrowed_kv_to_active_request():
    release = _method(
        ROOT / "python/sglang/srt/mem_cache/session_aware_cache.py",
        "SessionAwareCache",
        "release_session",
        {"Optional": Optional, "Req": object, "json": json, "logging": logging},
    )
    slot = SimpleNamespace(
        req_pool_idx=0,
        is_holding_kv=True,
        cache_protected_len=0,
        kv_allocated_len=3,
        last_node="radix-node",
        swa_uuid_for_lock=None,
    )
    freed = []
    unlocked = []
    pool = SimpleNamespace(req_to_token=_ReqToToken(), free_slots=[])
    cache = SimpleNamespace(
        slots={"session": slot},
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=SimpleNamespace(
            free=lambda ids: freed.extend(ids.tolist())
        ),
        inner=SimpleNamespace(dec_lock_ref=lambda node: unlocked.append(node)),
    )
    active_req = SimpleNamespace(req_pool_idx=0, last_node="virtual-node")
    release(cache, "session", active_req=active_req)
    assert cache.slots == {}
    assert freed == []
    assert pool.free_slots == []
    assert unlocked == []
    assert active_req.req_pool_idx == 0
    assert active_req.last_node == "radix-node"
    assert active_req.session_cache_closed_during_request is True


def test_detached_request_finishes_without_inserting_physical_kv_as_canonical():
    cache_finished = _method(
        ROOT / "python/sglang/srt/mem_cache/session_aware_cache.py",
        "SessionAwareCache",
        "cache_finished_req",
        {"Req": object},
    )
    calls = []
    cache = SimpleNamespace(
        inner=SimpleNamespace(
            cache_finished_req=lambda req, is_insert, **kwargs:
                calls.append((req, is_insert, kwargs))
        )
    )
    req = SimpleNamespace(session_cache_closed_during_request=True)
    cache_finished(cache, req, is_insert=True)
    assert calls == [(req, False, {})]


def test_detached_request_stays_owned_through_release_dispatch():
    owns = _method(
        ROOT / "python/sglang/srt/mem_cache/session_aware_cache.py",
        "SessionAwareCache",
        "owns_finished_request",
        {"Req": object, "_is_streaming": lambda req: False},
    )
    assert owns(SimpleNamespace(session_cache_closed_during_request=True))


def test_close_releases_unborrowed_slot():
    release = _method(
        ROOT / "python/sglang/srt/mem_cache/session_aware_cache.py",
        "SessionAwareCache",
        "release_session",
        {"Optional": Optional, "Req": object, "json": json, "logging": logging},
    )
    slot = SimpleNamespace(
        req_pool_idx=0,
        is_holding_kv=True,
        cache_protected_len=0,
        kv_allocated_len=3,
        last_node=None,
        swa_uuid_for_lock=None,
    )
    freed = []
    pool = SimpleNamespace(req_to_token=_ReqToToken(), free_slots=[])
    cache = SimpleNamespace(
        slots={"session": slot},
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=SimpleNamespace(
            free=lambda ids: freed.extend(ids.tolist())
        ),
    )
    release(cache, "session", active_req=SimpleNamespace(req_pool_idx=None))
    assert freed == [11, 12, 13]
    assert pool.free_slots == [0]
