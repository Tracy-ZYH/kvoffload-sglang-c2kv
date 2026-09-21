import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from sglang.srt.mem_cache.repair_tool_selection import (  # noqa: E402
    select_sparse_repair_indices,
    validate_sparse_repair_partition,
)
from sglang.srt.mem_cache.history_kv_selection import gather_paired_kv  # noqa: E402


def test_sparse_partition_requires_exact_mandatory_complement_and_budget():
    assert validate_sparse_repair_partition(7, [1, 2, 5], None, 5) == (
        [1, 2, 5], [0, 3, 4, 6]
    )
    for selectable, mandatory, target in [
        ([1, 1], None, 5),
        ([2, 1], None, 5),
        ([1, 7], None, 5),
        ([1, 2, 5], [0, 3, 6], 5),
        ([1, 2, 5], None, 3),
    ]:
        with pytest.raises(ValueError):
            validate_sparse_repair_partition(7, selectable, mandatory, target)


@pytest.mark.parametrize("method", ["streamingllm", "h2o", "snapkv_persistent"])
def test_global_schema_selection_preserves_every_protocol_token(method):
    selectable = [1, 2, 5, 6]
    mandatory = [0, 3, 4, 7]
    scores = [torch.tensor([[0, 8, 2, 0, 0, 9, 1, 0],
                            [0, 1, 7, 0, 0, 2, 8, 0]], dtype=torch.float32)]
    selected, metadata = select_sparse_repair_indices(
        method, scores if method != "streamingllm" else [],
        selectable, mandatory, target_tokens=6, num_layers=1,
        recent_window=1, kernel_size=1, h2o_recent_fraction=0.5,
    )
    assert selected[0].shape[1] == 6
    assert metadata["mandatory_tokens"] == 4
    assert metadata["budget_preserved"]
    for row in selected[0].tolist():
        assert mandatory == [index for index in row if index in mandatory]
        assert len(set(row)) == 6
        assert set(row) <= set(range(8))
    if method == "streamingllm":
        assert selected[0].tolist() == [[0, 1, 3, 4, 6, 7]]
    else:
        assert not torch.equal(selected[0][0], selected[0][1])
        original_key = torch.arange(16, dtype=torch.float32).reshape(8, 2)
        original_value = original_key + 100
        key, value = gather_paired_kv(original_key, original_value, selected[0])
        for head, row in enumerate(selected[0].tolist()):
            for mandatory_index in mandatory:
                slot = row.index(mandatory_index)
                assert key[slot, head] == original_key[mandatory_index, head]
                assert value[slot, head] == original_value[mandatory_index, head]


def test_sparse_pyramid_keeps_mandatory_tokens_and_reports_dense_cost():
    selectable = [i for i in range(1, 23) if i not in {5, 12, 18}]
    mandatory = [i for i in range(23) if i not in selectable]
    scores = [torch.stack((torch.arange(23).float(),
                           torch.arange(22, -1, -1).float()))
              for _ in range(4)]
    selected, metadata = select_sparse_repair_indices(
        "pyramidkv", scores, selectable, mandatory,
        target_tokens=len(mandatory) + 6, num_layers=4,
        recent_window=2, kernel_size=1,
    )
    assert len({indices.shape[1] for indices in selected}) == 1
    for indices in selected:
        for row in indices.tolist():
            assert set(mandatory) <= set(row)
            assert len(row) == len(set(row))
    assert metadata["native_repair_shared_length_approximation"]
    assert metadata["candidate_kept_tokens"] == selected[0].shape[1] - len(mandatory)
