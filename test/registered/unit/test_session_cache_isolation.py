"""CPU-only regressions for streaming-session ownership boundaries."""

import ast
import logging
import time
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
MANAGERS = ROOT / "python/sglang/srt/managers"


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


class _Req:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.multimodal_inputs = None
        self.finished_reason = None
        self.to_finish = None

    def set_finish_with_abort(self, message):
        self.to_finish = message

    def check_finished(self):
        self.finished_reason = self.to_finish
        self.to_finish = None


def _incoming():
    return SimpleNamespace(
        rid="next",
        input_ids=[22],
        mm_inputs=None,
        session_params=SimpleNamespace(
            replace=False, drop_previous_output=False, offset=0, rid=None
        ),
        c2kv_kv_memory_hint={"persistent_history_session": {"enabled": True}},
        sampling_params=SimpleNamespace(max_new_tokens=2),
        lora_id=None,
        custom_logit_processor=None,
        stream=False,
        return_logprob=False,
        top_logprobs_num=0,
        token_ids_logprob=None,
        require_reasoning=False,
        return_hidden_states=False,
        return_routed_experts=False,
        priority=None,
        routing_key=None,
        http_worker_ipc=None,
        time_stats=None,
    )


def test_overlapping_streaming_append_keeps_previous_kv_owner():
    create = _method(
        MANAGERS / "session_controller.py",
        "Session",
        "create_req",
        {
            "time": time,
            "Req": _Req,
            "SessionReqNode": SimpleNamespace,
            "TokenizedGenerateReqInput": object,
        },
    )
    old = SimpleNamespace(session=None, finished=lambda: False)
    owner = SimpleNamespace(req=old)
    session = SimpleNamespace(streaming=True, req_nodes={"old": owner})
    old.session = session

    rejected = create(session, _incoming(), None, vocab_size=100)

    assert rejected.finished_reason == (
        "Streaming session previous request has not finished."
    )
    assert session.req_nodes == {"old": owner}
    assert old.session is session


def test_completed_streaming_append_transfers_session_node():
    create = _method(
        MANAGERS / "session_controller.py",
        "Session",
        "create_req",
        {
            "time": time,
            "Req": _Req,
            "SessionReqNode": lambda req: SimpleNamespace(req=req),
            "TokenizedGenerateReqInput": object,
        },
    )
    old = SimpleNamespace(
        session=None,
        finished=lambda: True,
        output_ids=[31, 32],
        sampling_params=SimpleNamespace(max_new_tokens=2),
        origin_input_ids=[11],
        origin_input_ids_unpadded=[11],
        multimodal_inputs=None,
    )
    session = SimpleNamespace(
        streaming=True, req_nodes={"old": SimpleNamespace(req=old)}
    )
    old.session = session

    appended = create(session, _incoming(), None, vocab_size=100)

    assert appended.origin_input_ids == [11, 31, 32, 22]
    assert old.session is None
    assert session.req_nodes["next"].req is appended
    assert "old" not in session.req_nodes


def test_flush_refuses_to_clear_live_session_slots():
    flush = _method(
        MANAGERS / "scheduler.py",
        "Scheduler",
        "flush_cache",
        {"logging": logging},
    )
    reset_calls = []
    scheduler = SimpleNamespace(
        is_fully_idle=lambda: True,
        session_controller=SimpleNamespace(sessions={"live": object()}),
        tree_cache=SimpleNamespace(reset=lambda: reset_calls.append(True)),
        waiting_queue=[],
        running_batch=SimpleNamespace(reqs=[]),
    )

    assert flush(scheduler) is False
    assert not reset_calls


def test_missing_resident_slot_aborts_one_request_and_closes_session():
    abort = _method(
        MANAGERS / "scheduler.py",
        "Scheduler",
        "_abort_missing_persistent_history_session",
        {"Req": _Req, "CloseSessionReqInput": SimpleNamespace},
    )
    events = []

    class Sessions:
        def __contains__(self, session_id):
            return session_id == "lost"

        def close(self, request):
            events.append(("close", request.session_id))

    request = _Req(
        session=SimpleNamespace(session_id="lost"),
        return_logprob=False,
        kv_memory_report={},
    )
    scheduler = SimpleNamespace(
        session_controller=Sessions(),
        _cleanup_aborted_c2kv_waiting_req=lambda req: events.append(("cleanup", req)),
        stream_output=lambda reqs, logprob: events.append(("output", reqs, logprob)),
    )

    abort(scheduler, request)

    assert request.finished_reason == "PERSISTENT_HISTORY_SESSION_RESIDENT_CACHE_MISSING"
    assert request.kv_memory_report == {
        "history_kv_runtime_status": "resident_cache_missing",
        "persistent_history_session_error": request.finished_reason,
    }
    assert events == [
        ("cleanup", request),
        ("close", "lost"),
        ("output", [request], False),
    ]
