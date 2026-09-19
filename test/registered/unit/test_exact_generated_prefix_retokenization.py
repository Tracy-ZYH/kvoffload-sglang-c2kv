"""CPU-only exact-prefix regressions for generated-token BPE differences."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest


ROOT = Path(__file__).resolve().parents[3]
SERVING = ROOT / "python/sglang/srt/entrypoints/openai/serving_chat.py"


def _method(name, namespace):
    tree = ast.parse(SERVING.read_text(encoding="utf-8"))
    node = next(
        node
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "OpenAIServingChat"
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SERVING), "exec"), namespace)
    return namespace[name]


class _Tokenizer:
    # 100 + 101 is the generated `OME` + `G`; canonical BPE uses `OMEG`.
    pieces = {
        1: "<|im_start|>assistant\n",
        2: "<|im_end|>",
        3: "<|im_start|>tool\nline1\nline2<|im_end|>",
        4: "<|im_start|>user\nchanged",
        100: "OME",
        101: "G",
        200: "OMEG",
    }

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(self.pieces[item] for item in ids)

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        if text == self.decode([1, 100, 101, 2], skip_special_tokens=False,
                               clean_up_tokenization_spaces=False):
            return [1, 200, 2]
        raise AssertionError(f"unexpected text to encode: {text!r}")


def _serving():
    namespace = {"List": List, "Optional": Optional, "Dict": Dict, "Any": Any,
                 "ChatCompletionRequest": object}
    reconcile = _method("_reconcile_exact_generated_prefix", namespace)
    prepare = _method("_prepare_persistent_history_delta", namespace)
    serving = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(tokenizer=_Tokenizer()),
        _persistent_history_sessions={"s": [1, 100, 101, 2]},
        _persistent_history_exact_output={"s": True},
        _persistent_history_computed_prefixes={"s": 4},
        _reconcile_exact_generated_prefix=lambda previous, full, mismatch, hint:
            reconcile(serving, previous, full, mismatch, hint),
        _is_persistent_history_request=lambda _: True,
    )
    return serving, prepare


def _request():
    hint = {
        "persistent_history_session": {"enabled": True, "session_id": "s"},
        "history_kv_eviction": {"method": "agentkv", "history_start": 1,
                                "history_end": 3},
        "history_kv_event_token_spans": [
            {"start": 1, "end": 3}, {"start": 3, "end": 4}
        ],
        "history_kv_event_generation_suffix_start": 3,
    }
    return SimpleNamespace(stream=False, session_params={"id": "s"},
                           c2kv_kv_memory_hint=hint)


def test_generated_bpe_is_preserved_and_message_boundaries_follow_actual_tokens():
    serving, prepare = _serving()
    request = _request()
    delta, session_id, actual = prepare(serving, request, [1, 200, 2, 3])
    assert session_id == "s"
    assert delta == [3]
    assert actual == [1, 100, 101, 2, 3]
    hint = request.c2kv_kv_memory_hint
    assert hint["history_kv_eviction"]["history_end"] == 4
    assert hint["history_kv_event_token_spans"] == [
        {"start": 1, "end": 4}, {"start": 4, "end": 5}
    ]
    assert hint["history_kv_event_generation_suffix_start"] == 4
    assert hint["persistent_session_retokenization_token_shift"] == 1
    assert hint["persistent_session_delta_tokens"] == 1
    assert hint["persistent_session_canonical_prompt_tokens"] == 5
    assert request.session_params["drop_previous_output"] is False


def test_changed_prior_assistant_text_still_fails_closed():
    serving, prepare = _serving()
    with pytest.raises(ValueError, match="PREFIX_MISMATCH"):
        prepare(serving, _request(), [1, 200, 4])
