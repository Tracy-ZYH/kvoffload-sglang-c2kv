"""The served T0 checkpoint can supply both named gist projection sets."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.models import qwen3


def _write_config(path, domain, variant):
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 8,
                "history_memory_compression_domain": domain,
                "history_memory_variant": variant,
                "gist_type": "dynamic-interleave",
                "gist_param": "qkv",
            }
        ),
        encoding="utf-8",
    )


class _TinyModel(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.embed_tokens = nn.Embedding(2, config.hidden_size)

    def _init_c2kv(self, config, server_args):
        self.gist_embed_tokens = nn.Embedding(1, config.hidden_size, dtype=torch.float32)
        self.prepare_gist_input = lambda *args: None
        return object()

    def _init_c2kv_tool_set(self, config, tool_config, server_args):
        self.tool_gist_embed_tokens = nn.Embedding(
            1, config.hidden_size, dtype=torch.float32
        )
        self.prepare_tool_gist_input = lambda *args: None
        return object()


@pytest.mark.parametrize("same_source", [True, False])
def test_tool_gist_parameters_share_only_for_served_t0(
    tmp_path, monkeypatch, same_source
):
    served = tmp_path / "served"
    tool = served if same_source else tmp_path / "tool"
    _write_config(served, "tool" if same_source else "history", "T0" if same_source else "H0")
    if tool != served:
        _write_config(tool, "tool", "T0")
    config = SimpleNamespace(
        hidden_size=8,
        history_memory_compression_domain="tool" if same_source else "history",
        history_memory_variant="T0" if same_source else "H0",
        tie_word_embeddings=True,
    )
    server_args = SimpleNamespace(
        enable_c2kv=True,
        model_path=str(served),
        c2kv_tool_gist_weights=str(tool),
        c2kv_gist_type="dynamic-interleave",
        c2kv_gist_param="qkv",
        c2kv_shadow_feature_layer=None,
        rl_on_policy_target=None,
    )
    assert qwen3._c2kv_tool_gist_uses_served_t0(config, server_args) is same_source

    monkeypatch.setattr(qwen3, "get_global_server_args", lambda: server_args)
    monkeypatch.setattr(
        qwen3,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True, world_size=1),
    )
    monkeypatch.setattr(qwen3, "Qwen3Model", _TinyModel)
    monkeypatch.setattr(qwen3, "LogitsProcessor", lambda config: object())
    monkeypatch.setattr(qwen3, "Pooler", lambda **kwargs: object())
    model = qwen3.Qwen3ForCausalLM(config)
    history_embed = model.model.gist_embed_tokens.weight
    tool_embed = model.model.tool_gist_embed_tokens.weight
    assert history_embed.dtype == tool_embed.dtype == torch.float32
    assert (history_embed is tool_embed) is same_source
    assert (history_embed.data_ptr() == tool_embed.data_ptr()) is same_source

    allocations = []

    class TinyParallelLinear(nn.Module):
        def __init__(self, *args, params_dtype=None, **kwargs):
            super().__init__()
            allocations.append(params_dtype)
            self.weight = nn.Parameter(
                torch.empty((8, 8), dtype=params_dtype or torch.float32)
            )
            self.bias = None

    monkeypatch.setattr(qwen3, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(qwen3, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(qwen3, "get_attention_tp_rank", lambda: 0)
    monkeypatch.setattr(qwen3, "get_attention_tp_size", lambda: 1)
    monkeypatch.setattr(qwen3, "RMSNorm", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(qwen3, "QKVParallelLinear", TinyParallelLinear)
    monkeypatch.setattr(qwen3, "RowParallelLinear", TinyParallelLinear)
    monkeypatch.setattr(qwen3, "get_rope", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(qwen3, "RadixAttention", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(qwen3.torch, "compile", lambda func, **kwargs: func)
    attention = qwen3.Qwen3Attention(
        hidden_size=8,
        num_heads=2,
        num_kv_heads=2,
        tool_gist_uses_served_t0=qwen3._c2kv_tool_gist_uses_served_t0(
            config, server_args
        ),
    )
    history_qkv = attention.gist_qkv_proj.weight
    tool_qkv = attention.tool_gist_qkv_proj.weight
    assert history_qkv.dtype == tool_qkv.dtype == torch.float32
    assert (history_qkv is tool_qkv) is same_source
    assert (history_qkv.data_ptr() == tool_qkv.data_ptr()) is same_source
    assert len(allocations) == (3 if same_source else 4)
    if same_source:
        monkeypatch.setattr(
            qwen3, "_c2kv_gist_weight_files",
            lambda source: pytest.fail("The served T0 must not be loaded twice"),
        )
        assert model.load_c2kv_tool_gist_weights()["parameter_source"] == "served_checkpoint"


def test_served_t0_alias_accepts_checkpoint_symlink(tmp_path):
    served = tmp_path / "served"
    _write_config(served, "tool", "T0")
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(served, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")
    config = SimpleNamespace(
        history_memory_compression_domain="tool", history_memory_variant="T0"
    )
    args = SimpleNamespace(model_path=str(served), c2kv_tool_gist_weights=str(alias))
    assert qwen3._c2kv_tool_gist_uses_served_t0(config, args)
