"""CPU contracts for the native packed generation sampling profiles."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest


HTTP_SERVER = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/entrypoints/http_server.py"
)


def _load_functions(*names, **bindings):
    tree = ast.parse(HTTP_SERVER.read_text(encoding="utf-8"), filename=str(HTTP_SERVER))
    functions = []
    for name in names:
        node = next(
            item
            for item in tree.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == name
        )
        node.decorator_list = []
        functions.append(node)
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "C2KVNativePackedGenerateRequest": object,
        "Request": object,
        **bindings,
    }
    exec(compile(module, str(HTTP_SERVER), "exec"), namespace)
    return namespace


SAMPLING = _load_functions("_c2kv_native_sampling_params")[
    "_c2kv_native_sampling_params"
]


def _request(profile="greedy-v1", **sampling):
    return SimpleNamespace(
        sampling_profile=profile,
        sampling_params={"max_new_tokens": 1000, **sampling},
        shadow_features=None,
    )


def test_default_greedy_profile_preserves_old_temperature_gate():
    params = SAMPLING(_request())
    assert params == {"max_new_tokens": 1000, "temperature": 0.0}
    with pytest.raises(ValueError, match="requires greedy decoding"):
        SAMPLING(_request(temperature=0.001))


def test_acebench_profile_passes_exact_sampler_without_seed_or_shadow():
    original = {
        "max_new_tokens": 1000,
        "temperature": 0.001,
        "top_p": 1.0,
        "stop_token_ids": [151645],
    }
    request = SimpleNamespace(
        sampling_profile="acebench-agent-v1",
        sampling_params=original,
        shadow_features=None,
    )
    assert SAMPLING(request) == original
    assert request.sampling_params == original


@pytest.mark.parametrize(
    "sampling,shadow,error",
    [
        ({"temperature": 0, "top_p": 1}, None, "temperature=0.001"),
        ({"temperature": 0.001, "top_p": 0.9}, None, "top_p=1"),
        ({"temperature": 0.001}, None, "top_p=1"),
        ({"temperature": 0.001, "top_p": 1, "seed": 0}, None, "seed"),
        (
            {"temperature": 0.001, "top_p": 1, "sampling_seed": 0},
            None,
            "sampling_seed",
        ),
        ({"temperature": 0.001, "top_p": 1, "top_k": 5}, None, "top_k"),
        (
            {"temperature": 0.001, "top_p": 1},
            {"enabled": False},
            "shadow_features",
        ),
    ],
)
def test_acebench_profile_rejects_other_sampler_or_detector_settings(
    sampling, shadow, error
):
    request = _request("acebench-agent-v1", **sampling)
    request.shadow_features = shadow
    with pytest.raises(ValueError, match=error):
        SAMPLING(request)


def test_capability_advertises_both_named_profiles():
    server_args = SimpleNamespace(
        dtype="float16", kv_cache_dtype="auto", enable_c2kv=True
    )
    model_config = SimpleNamespace(
        dtype="float16",
        num_hidden_layers=2,
        num_key_value_heads=1,
        head_dim=4,
        v_head_dim=4,
        hf_config=SimpleNamespace(pic_enabled=False),
    )
    manager = SimpleNamespace(
        server_args=server_args,
        model_config=model_config,
        model_path="model",
    )
    namespace = _load_functions(
        "_c2kv_native_capability",
        "_c2kv_tool_gist_capability",
        _global_state=SimpleNamespace(tokenizer_manager=manager),
        canonical_model_binding=lambda **kwargs: {
            **kwargs,
            "weight_version": "checkpoint",
        },
        _c2kv_dtype_nbytes=lambda dtype: 2,
        NATIVE_PACKED_CAPABILITY_SCHEMA="c2kv-native-packed-capability-v1",
        C2KV_NATIVE_PACKING_VERSION="history-event-v1",
        C2KV_NATIVE_RAW_LAYOUT_PROFILE="event-native-evidence-v1",
    )
    capability = namespace["_c2kv_native_capability"]()
    assert capability["sampling_profiles"] == ["greedy-v1", "acebench-agent-v1"]


def test_endpoint_forwards_acebench_sampling_to_generation_request():
    captured = {}

    async def generate_request(request, raw_request):
        captured["sampling_params"] = request.sampling_params
        yield {
            "output_ids": [42],
            "text": "answer",
            "meta_info": {"output_token_logprobs": [(-0.5, 42)]},
        }

    manager = SimpleNamespace(generate_request=generate_request)
    plan = SimpleNamespace(
        logical_input_ids=[1, 2],
        selected_handles=[],
        segment_boundaries=[],
        compression_handles=[],
        unique_chunks=[],
        costs={
            "presented_encoder_tokens": 0,
            "gist_tokens": 0,
            "system_tokens": 1,
            "gist_prefix_kv_tokens": 0,
            "raw_workspace_kv_tokens": 1,
            "resident_kv_tokens": 2,
        },
    )
    namespace = _load_functions(
        "_c2kv_native_sampling_params",
        "v1_c2kv_native_generate",
        _global_state=SimpleNamespace(tokenizer_manager=manager),
        _c2kv_native_capability=lambda: {
            "enabled": True,
            "model_binding": {"pic_enabled": False},
            "shadow_feature_layer": None,
            "kv_bytes_per_token": 4,
        },
        plan_native_packed_request=lambda **kwargs: plan,
        GenerateReqInput=lambda **kwargs: SimpleNamespace(**kwargs),
        orjson_response=lambda value: value,
        NATIVE_PACKED_RESPONSE_SCHEMA="c2kv-native-packed-response-v1",
        logger=SimpleNamespace(error=lambda *args, **kwargs: None),
        _create_error_response=lambda error: {"error": str(error)},
    )
    sampling = {
        "max_new_tokens": 1000,
        "temperature": 0.001,
        "top_p": 1,
        "stop_token_ids": [151645],
    }
    request = SimpleNamespace(
        sampling_profile="acebench-agent-v1",
        sampling_params=sampling,
        shadow_features=None,
        max_extraction_calls=0,
        encoder_chunks=[],
        compression_chunks=[],
        system_input_ids=[1],
        workspace_input_ids=[2],
        packing_version="history-event-v1",
        raw_layout_profile="event-native-evidence-v1",
        encoding_scope="current",
        compression_ratio=8,
        rid="native-1",
        session_id="session-1",
        generation_id="generation-1",
    )
    response = asyncio.run(
        namespace["v1_c2kv_native_generate"](
            request, SimpleNamespace(headers={})
        )
    )
    assert captured["sampling_params"] == sampling
    assert response["sampling_profile"] == "acebench-agent-v1"
    assert response["output_ids"] == [42]
