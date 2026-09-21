"""Regression contracts for the common reference-history tensor route."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from sglang.srt.mem_cache.history_kv_reference import (  # noqa: E402
    ReferenceHistoryKVState,
    ReferenceLayerKV,
    gather_reference_candidates,
    gather_reference_layer,
    reference_sdpa,
)
import sglang.srt.mem_cache.history_kv_reference as reference_module  # noqa: E402


def test_reference_attention_masks_each_kv_heads_own_absolute_positions():
    history = ReferenceLayerKV(
        key=torch.zeros(2, 2, 1),
        value=torch.tensor([[[1.0], [9.0]], [[2.0], [6.0]]]),
        positions=torch.tensor([[0, 10], [0, 1]], dtype=torch.long),
    )

    output = reference_sdpa(
        torch.zeros(1, 2, 1),
        history,
        torch.empty(0, 2, 1),
        torch.empty(0, 2, 1),
        torch.empty(0, dtype=torch.long),
        torch.tensor([5], dtype=torch.long),
        scale=1.0,
    )

    # Head 0 must exclude its future key at position 10.  Head 1 sees both of
    # its keys at positions 0 and 1 and averages their values.
    torch.testing.assert_close(output, torch.tensor([[[1.0], [4.0]]]))


def test_reference_attention_uses_four_dimensional_sdpa(monkeypatch):
    observed = []
    original = reference_module.F.scaled_dot_product_attention

    def checked(q, k, v, **kwargs):
        observed.append((q.ndim, k.ndim, v.ndim, kwargs["attn_mask"].ndim))
        return original(q, k, v, **kwargs)

    monkeypatch.setattr(reference_module.F, "scaled_dot_product_attention", checked)
    history = ReferenceLayerKV(
        key=torch.zeros(1, 2, 2), value=torch.ones(1, 2, 2),
        positions=torch.tensor([[0, 1]], dtype=torch.long),
    )
    reference_sdpa(
        torch.zeros(1, 2, 2), history,
        torch.empty(0, 1, 2), torch.empty(0, 1, 2),
        torch.empty(0, dtype=torch.long), torch.tensor([1]), scale=1.0,
    )
    assert observed == [(4, 4, 4, 4)]


def test_two_turn_append_only_uses_previous_resident_kv_and_new_delta():
    source_key = torch.arange(6 * 2, dtype=torch.float32).reshape(6, 2, 1)
    source_value = source_key + 100
    first = gather_reference_layer(
        source_key,
        source_value,
        torch.arange(6, dtype=torch.long),
        torch.tensor([[0, 2, 5], [1, 3, 5]], dtype=torch.long),
    )
    delta_key = torch.tensor([[[20.0], [21.0]], [[22.0], [23.0]]])
    delta_value = delta_key + 100

    second = gather_reference_candidates(
        first,
        delta_key,
        delta_value,
        torch.tensor([6, 7], dtype=torch.long),
        # Candidate axes are exactly [previous resident, new delta] per head.
        torch.tensor([[0, 3, 4], [1, 3, 4]], dtype=torch.long),
    )

    assert second.positions.tolist() == [[0, 6, 7], [3, 6, 7]]
    assert set(second.positions[0].tolist()) <= {0, 2, 5, 6, 7}
    assert set(second.positions[1].tolist()) <= {1, 3, 5, 6, 7}
    # Raw positions evicted before this append are absent from the candidate
    # tensors, so the second selection cannot bring them back.
    assert not ({1, 3, 4} & set(second.positions[0].tolist()))
    assert not ({0, 2, 4} & set(second.positions[1].tolist()))
    second.validate()


def test_state_resident_bytes_sum_k_v_and_per_head_positions_for_all_layers():
    layer0 = ReferenceLayerKV(
        key=torch.zeros(2, 3, 4, dtype=torch.bfloat16),
        value=torch.zeros(2, 3, 4, dtype=torch.bfloat16),
        positions=torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
    )
    layer1 = ReferenceLayerKV(
        key=torch.zeros(2, 2, 4, dtype=torch.float32),
        value=torch.zeros(2, 2, 4, dtype=torch.float32),
        positions=torch.tensor([[0, 2], [1, 2]], dtype=torch.long),
    )
    state = ReferenceHistoryKVState(
        method="pyramidkv", layers={0: layer0, 1: layer1}
    )

    state.validate()
    layer0_expected = 2 * 2 * 3 * 4 * 2 + 2 * 3 * 8
    layer1_expected = 2 * 2 * 2 * 4 * 4 + 2 * 2 * 8
    assert layer0.resident_bytes == layer0_expected
    assert layer1.resident_bytes == layer1_expected
    assert state.resident_bytes == layer0_expected + layer1_expected
