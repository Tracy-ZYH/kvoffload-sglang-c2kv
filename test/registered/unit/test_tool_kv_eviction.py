import pytest
import torch
import json
import os
from collections.abc import Mapping

from sglang.srt.mem_cache.tool_kv_eviction import (
    plan_tool_kv_eviction,
    select_tool_h2o,
    select_tool_snapkv,
)
from sglang.srt.mem_cache.tool_kv_spans import resolve_schema_token_spans


class _Tokenizer:
    def __init__(self, pieces):
        self.pieces = pieces

    def __call__(self, text, **kwargs):
        assert "".join(self.pieces) == text
        offsets = []
        cursor = 0
        for piece in self.pieces:
            offsets.append((cursor, cursor + len(piece)))
            cursor += len(piece)
        return {"input_ids": list(range(len(self.pieces))), "offset_mapping": offsets}


def test_schema_boundaries_keep_crossing_bpe_tokens():
    rendered = "<S>H\n{\"x\":1}\n</S>"
    tokenizer = _Tokenizer(["<S>", "H\n{", '"x"', ":1", "}\n", "</S>"])
    schema = '{"x":1}'
    message = "H\n" + schema + "\n"
    spans = resolve_schema_token_spans(
        rendered_prompt=rendered,
        prompt_ids=list(range(6)),
        message_contents=[message],
        schema_spans=[{"schema_index": 0, "message_index": 0,
                       "start": 2, "end": 2 + len(schema), "text": schema}],
        tokenizer=tokenizer,
    )
    assert spans == [{"schema_index": 0, "token_start": 2, "token_end": 4}]
    protocol = resolve_schema_token_spans(
        rendered_prompt=rendered,
        prompt_ids=list(range(6)), message_contents=[message],
        schema_spans=[{"schema_index": -1, "message_index": 0,
                       "start": 0, "end": len(message), "text": message}],
        tokenizer=tokenizer, boundary_policy="overlap",
    )
    assert protocol[0]["token_start"] == 1
    assert protocol[0]["token_end"] == 5


def test_tool_plan_preserves_every_non_schema_token_and_all_native_noop():
    config = {
        "method": "h2o", "full_prompt_tokens": 8,
        "resolved_schema_token_spans": [
            {"schema_index": 0, "token_start": 1, "token_end": 3},
            {"schema_index": 1, "token_start": 4, "token_end": 6},
        ],
        "protected_schema_indices": [0],
        "resolved_tool_protocol_token_span": {"token_start": 0, "token_end": 7},
        "target_resident_tokens_per_layer": 7,
        "max_resident_tool_tokens": 6,
    }
    plan = plan_tool_kv_eviction(config, 8)
    assert plan["tool_evictable_indices"] == [4, 5]
    assert plan["protected_history_indices"] == [0, 1, 2, 3, 6]
    assert plan["tool_keep_tokens"] == 1
    assert plan["tool_protocol_resident_tokens"] == 6
    assert plan["selection_query_end"] == 7

    config["protected_schema_indices"] = [0, 1]
    config["target_resident_tokens_per_layer"] = 8
    config["max_resident_tool_tokens"] = 7
    assert plan_tool_kv_eviction(config, 8)["tool_no_op"] is True


def test_zero_keep_is_feasible_and_never_selects_history():
    config = {
        "method": "streamingllm", "full_prompt_tokens": 6,
        "resolved_schema_token_spans": [
            {"schema_index": 0, "token_start": 1, "token_end": 3},
        ],
        "target_evictable_tokens_per_layer": 0,
    }
    plan = plan_tool_kv_eviction(config, 6)
    assert plan["tool_keep_tokens"] == 0
    assert plan["protected_history_indices"] == [0, 3, 4]


def test_headwise_h2o_recent_floor_and_snapkv_original_positions():
    scores = torch.tensor([[0.0, 1.0, 9.0, 8.0, 0.0],
                           [0.0, 9.0, 1.0, 0.0, 8.0]])
    selected = select_tool_h2o(scores, [1, 2, 4], 1, 0.5)
    assert selected.tolist() == [[1], [0]]  # floor(1*.5)=0
    selected = select_tool_snapkv(
        torch.tensor([[9.0, 0.0, 0.0, 0.0, 1.0]]), [1, 4], 1, 3
    )
    assert selected.tolist() == [[0]]  # position 1 inherits position 0's score
    with pytest.raises(ValueError, match="KERNEL"):
        select_tool_snapkv(scores, [1, 4], 1, 4)


def test_checkpoint_tokenizer_tool_schema_boundary():
    checkpoint = os.environ.get("C2KV_TOOL_TEST_TOKENIZER")
    if not checkpoint:
        pytest.skip("Set C2KV_TOOL_TEST_TOKENIZER for the checkpoint tokenizer probe")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    schema = json.dumps(
        {"type": "function", "function": {"name": "lookup",
         "description": "Find a record", "parameters": {"type": "object",
         "properties": {"query": {"type": "string"}}}}},
        ensure_ascii=False, separators=(",", ":"),
    )
    system = "# Tools\n<tools>\n" + schema + "\n</tools>\nUse <tool_call>."
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "Look this up."}]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False, return_dict=False,
    )
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=False,
    )
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    span = {"schema_index": 0, "message_index": 0,
            "start": system.index(schema), "end": system.index(schema) + len(schema),
            "text": schema}
    resolved = resolve_schema_token_spans(
        rendered_prompt=rendered, prompt_ids=ids,
        message_contents=[system, "Look this up."],
        schema_spans=[span], tokenizer=tokenizer,
    )
    assert resolved[0]["token_end"] > resolved[0]["token_start"]
    assert resolved[0]["token_end"] < len(ids) - 1
