"""CPU-only contracts for the second ("tool") C2KV gist projection set.

The tool set is used only by extraction requests that name
``projection_set="tool"``.  These tests pin the two identity rules that keep
it from aliasing the history set: the extraction cache key and the native
chunk handle both change when the tool set is requested, and both stay
byte-identical to the pre-tool-set schema when it is not.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest


_HERE = os.path.dirname(os.path.abspath(__file__))
_SRT = os.path.normpath(os.path.join(_HERE, "..", "..", "..", "python", "sglang", "srt"))


def _load(name, relative):
    path = os.path.join(_SRT, *relative)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


semantics = _load("c2kv_semantics_tool_gist_under_test", ("mem_cache", "c2kv_semantics.py"))
native = _load("c2kv_native_packed_tool_gist_under_test", ("mem_cache", "c2kv_native_packed.py"))


def _write_tool_checkpoint(directory, *, step=1034, variant="T0"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "gist_param": "qkv",
                "gist_residual_type": "embed-mean",
                "gist_token_id": 151645,
                "gist_type": "dynamic-interleave",
                "hidden_size": 2560,
                "history_memory_compression_domain": "tool",
                "history_memory_variant": variant,
                "history_memory_supported_ratios": [8, 12],
                "history_memory_render_profile": "next-compression-tool-explicit-protocol-v2",
                "num_hidden_layers": 36,
            }
        ),
        encoding="utf-8",
    )
    (directory / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "parameter_version": step, "completed": True}),
        encoding="utf-8",
    )


def test_tool_gist_identity_binds_checkpoint_metadata_and_step(tmp_path):
    _write_tool_checkpoint(tmp_path / "checkpoint-1034")
    _write_tool_checkpoint(tmp_path / "copy" / "checkpoint-1034")
    _write_tool_checkpoint(tmp_path / "checkpoint-500", step=500)
    _write_tool_checkpoint(tmp_path / "t1" / "checkpoint-1034", variant="T1")

    base = semantics.c2kv_tool_gist_identity(str(tmp_path / "checkpoint-1034"))
    assert base == semantics.c2kv_tool_gist_identity(
        str(tmp_path / "copy" / "checkpoint-1034")
    ), "identity depends on metadata, not on the absolute path"
    assert base != semantics.c2kv_tool_gist_identity(str(tmp_path / "checkpoint-500"))
    assert base != semantics.c2kv_tool_gist_identity(str(tmp_path / "t1" / "checkpoint-1034"))
    assert len(base) == 64


def test_extraction_cache_key_separates_projection_sets():
    ids = [11, 12, 13]
    history_config = {"gist_type": "dynamic-interleave", "gist_param": "qkv"}
    history = semantics.compute_gist_cache_key(ids, 8, history_config)
    tool = semantics.compute_gist_cache_key(
        ids, 8, dict(history_config, projection_set="tool", projection_identity="abc")
    )
    other_tool = semantics.compute_gist_cache_key(
        ids, 8, dict(history_config, projection_set="tool", projection_identity="def")
    )
    assert history != tool
    assert tool != other_tool
    # The history key never carries the new fields, so it is unchanged.
    assert history == semantics.compute_gist_cache_key(ids, 8, dict(history_config))


def _binding():
    return native.canonical_model_binding(
        model_path="/checkpoints/checkpoint-1000",
        tokenizer_path=None,
        weight_version=None,
        dtype="bfloat16",
        kv_cache_dtype="bfloat16",
        gist_parameter_dtype="float32",
        gist_compute_dtype="bfloat16",
        gist_type="dynamic-interleave",
        gist_param="qkv",
        gist_extra_embed_num=1,
        gist_residual_type="embed-mean",
        gist_overlap=0,
        pic_enabled=False,
        pic_param="qkv",
        query_projection="base",
    )


def _chunk(token_ids, **extra):
    chunk = {
        "chunk_id": "c0",
        "event_id": "t0-tool-0-abc",
        "part_index": 0,
        "source_indices": [],
        "source_token_start": 0,
        "source_token_end": len(token_ids),
        "token_ids": list(token_ids),
    }
    chunk.update(extra)
    return chunk


_TOOL_BINDING = {"enabled": True, "identity": "f" * 64, "source": "/ckpt/T0/checkpoint-1034"}


def _handle(chunk, tool_binding=None):
    return native.canonical_chunk_handle(
        chunk,
        model_binding=_binding(),
        packing_version=native.PACKING_VERSION,
        encoding_scope="scope",
        compression_ratio=8,
        tool_binding=tool_binding,
    )


def test_history_chunk_handle_is_unchanged_by_tool_binding():
    chunk = _chunk([1, 2, 3, 4])
    assert _handle(chunk) == _handle(chunk, _TOOL_BINDING)
    assert _handle(dict(chunk, projection_set="history"), _TOOL_BINDING) == _handle(chunk)
    payload = native.canonical_chunk_payload(
        chunk,
        model_binding=_binding(),
        packing_version=native.PACKING_VERSION,
        encoding_scope="scope",
        compression_ratio=8,
        tool_binding=_TOOL_BINDING,
    )
    assert "projection_set" not in payload["chunk"]
    assert "projection_identity" not in payload


def test_tool_chunk_handle_binds_projection_set_and_identity():
    chunk = _chunk([1, 2, 3, 4])
    tool_chunk = dict(chunk, projection_set="tool")
    assert _handle(tool_chunk, _TOOL_BINDING) != _handle(chunk)
    assert _handle(tool_chunk, _TOOL_BINDING) != _handle(
        tool_chunk, dict(_TOOL_BINDING, identity="e" * 64)
    )
    with pytest.raises(ValueError, match="C2KV_TOOL_GIST_UNAVAILABLE"):
        _handle(tool_chunk)
    with pytest.raises(ValueError, match="C2KV_TOOL_GIST_UNAVAILABLE"):
        _handle(tool_chunk, {"enabled": False, "identity": None})
    with pytest.raises(ValueError, match="projection_set"):
        _handle(dict(chunk, projection_set="other"), _TOOL_BINDING)


def test_plan_lays_tool_chunks_before_history_chunks_and_keeps_projection_set():
    ratio = 8
    system = list(range(100, 110))
    tool_ids = list(range(1, 21))
    history_ids = list(range(50, 62))
    cursor = len(system)
    tool_chunk = _chunk(
        tool_ids,
        projection_set="tool",
        source_position_start=cursor,
        gist_position_ids=[
            cursor + min(start + ratio, len(tool_ids)) - 1
            for start in range(0, len(tool_ids), ratio)
        ],
    )
    cursor += len(tool_ids)
    history_chunk = _chunk(
        history_ids,
        chunk_id="h0",
        event_id="event-1",
        source_position_start=cursor,
        gist_position_ids=[
            cursor + min(start + ratio, len(history_ids)) - 1
            for start in range(0, len(history_ids), ratio)
        ],
    )
    plan = native.plan_native_packed_request(
        system_input_ids=system,
        workspace_input_ids=[7, 8, 9],
        encoder_chunks=[tool_chunk, history_chunk],
        compression_chunks=[],
        model_binding=_binding(),
        packing_version=native.PACKING_VERSION,
        raw_layout_profile=native.RAW_LAYOUT_PROFILE,
        encoding_scope="scope",
        compression_ratio=ratio,
        tool_binding=_TOOL_BINDING,
    )
    assert plan.segment_boundaries == (
        (len(system), len(system) + len(tool_ids)),
        (len(system) + len(tool_ids), len(system) + len(tool_ids) + len(history_ids)),
    )
    by_handle = {chunk["handle"]: chunk for chunk in plan.unique_chunks}
    tool_handle, history_handle = plan.selected_handles
    assert by_handle[tool_handle]["projection_set"] == "tool"
    assert "projection_set" not in by_handle[history_handle]
    assert plan.costs["gist_tokens"] == 3 + 2
