import json
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/observability/paper_telemetry.py"
)
SPEC = importlib.util.spec_from_file_location("paper_telemetry_under_test", MODULE_PATH)
paper_telemetry = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = paper_telemetry
SPEC.loader.exec_module(paper_telemetry)
_PaperTelemetry = paper_telemetry._PaperTelemetry


class _Allocator:
    size = 100

    def available_size(self):
        return 70


class _C2KVPool:
    max_total_tokens = 50

    def current_tokens(self):
        return 5


def test_request_scoped_metrics_include_generation_and_history(monkeypatch, tmp_path):
    log_path = tmp_path / "paper.jsonl"
    monkeypatch.setenv("C2KV_PAPER_TELEMETRY", "1")
    monkeypatch.setenv("C2KV_PAPER_TELEMETRY_LOG", str(log_path))
    telemetry = _PaperTelemetry()
    telemetry.configure(_Allocator(), _C2KVPool(), bytes_per_kv_token=4)
    telemetry.start(
        server_request_id="server-1",
        outer_request_id="outer-1",
        phase="chat",
        kind="generation",
        whole_full_kv_tokens=100,
    )
    telemetry.set_phase("selection")
    temporary_kv = torch.zeros((2, 3), dtype=torch.float32)
    telemetry.sample(
        "selection_buffers", tensors=[temporary_kv], temporary_kv=True
    )
    req = SimpleNamespace(
        rid="server-1",
        kv_committed_len=42,
        origin_input_ids=list(range(80)),
        kv_memory_report={
            "full_equivalent_history_tokens": 60,
            "active_history_kv_tokens": 15,
            "selection_query_tokens": 7,
            "selection_query_tokens_observed": 7,
            "history_kv_runtime_status": "physical_eviction_applied",
            "history_kv_physical_eviction": {"success": True},
            "history_kv_lifecycle": {"full_history_reprefill_performed": False},
        },
    )
    telemetry.mark_generation_start(req)
    result = telemetry.finish(req=req, success=True)

    assert result["outer_request_id"] == "outer-1"
    assert result["server_request_id"] == "server-1"
    assert result["phase"] == "chat"
    assert result["generation_start"] is not None
    assert result["baseline"]["event"] == "baseline"
    assert result["peak"] is not result["baseline"]
    metrics = result["metrics"]
    assert metrics["generation_active_kv_tokens"] == 42
    assert metrics["generation_active_kv_bytes"] == 168
    assert metrics["whole_full_kv_tokens"] == 100
    assert metrics["whole_active_kv_tokens"] == 42
    assert metrics["history_full_kv_tokens"] == 60
    assert metrics["history_active_kv_tokens"] == 15
    assert metrics["temporary_extraction_recovery_peak_kv_tokens"] == 6
    assert metrics["temporary_extraction_recovery_peak_kv_bytes"] == 24
    assert metrics["full_history_reprefill"] is False
    assert metrics["selection_query_tokens_planned"] == 7
    assert metrics["selection_query_tokens_observed"] == 7
    assert metrics["history_kv_physical_eviction_success"] is True
    assert all(phase["duration_ns"] >= 0 for phase in result["phases"])
    assert {phase["name"] for phase in result["phases"]} == {
        "chat",
        "selection",
        "decode",
    }
    assert json.loads(log_path.read_text(encoding="utf-8"))["metrics"] == metrics


def test_transformed_text_history_is_not_labeled_full(monkeypatch):
    monkeypatch.setenv("C2KV_PAPER_TELEMETRY", "1")
    monkeypatch.delenv("C2KV_PAPER_TELEMETRY_LOG", raising=False)
    telemetry = _PaperTelemetry()
    telemetry.configure(_Allocator(), _C2KVPool(), bytes_per_kv_token=4)
    telemetry.start(
        server_request_id="text-1",
        outer_request_id="outer-text-1",
        phase="chat",
        kind="generation",
        whole_full_kv_tokens=None,
    )
    req = SimpleNamespace(
        rid="text-1",
        kv_committed_len=30,
        origin_input_ids=list(range(30)),
        kv_memory_report={},
        c2kv_paper_history_full_kv_tokens=None,
        c2kv_paper_history_active_kv_tokens=12,
        c2kv_paper_canonical_full_source=False,
        c2kv_paper_denominator_tokenization_duration_ns=100,
    )
    telemetry.mark_generation_start(req)
    metrics = telemetry.finish(req=req, success=True)["metrics"]
    assert metrics["whole_full_kv_tokens"] is None
    assert metrics["whole_active_kv_tokens"] == 30
    assert metrics["history_full_kv_tokens"] is None
    assert metrics["history_active_kv_tokens"] == 12
    assert metrics["canonical_full_source"] is False
    assert metrics["denominator_tokenization_duration_ns"] == 100


def test_canonical_request_boundary_overrides_runtime_placeholders(monkeypatch):
    monkeypatch.setenv("C2KV_PAPER_TELEMETRY", "1")
    telemetry = _PaperTelemetry()
    telemetry.configure(_Allocator(), _C2KVPool(), bytes_per_kv_token=4)
    telemetry.start(
        server_request_id="canonical-1",
        outer_request_id="outer-canonical-1",
        phase="chat",
        kind="generation",
        whole_full_kv_tokens=422,
    )
    req = SimpleNamespace(
        rid="canonical-1",
        kv_committed_len=134,
        kv_memory_report={
            "full_equivalent_history_tokens": 0,
            "active_history_kv_tokens": 96,
        },
        c2kv_paper_history_full_kv_tokens=384,
        c2kv_paper_history_active_kv_tokens=None,
        c2kv_paper_canonical_full_source=True,
    )
    telemetry.mark_generation_start(req)
    metrics = telemetry.finish(req=req, success=True)["metrics"]
    assert metrics["whole_full_kv_tokens"] == 422
    assert metrics["history_full_kv_tokens"] == 384
    assert metrics["history_active_kv_tokens"] == 96


def test_request_peak_uses_simultaneous_pool_plus_temporary_kv(monkeypatch):
    monkeypatch.setenv("C2KV_PAPER_TELEMETRY", "1")

    class MutableAllocator:
        size = 100
        available = 80

        def available_size(self):
            return self.available

    class EmptyC2KVPool:
        max_total_tokens = 0

        def current_tokens(self):
            return 0

    allocator = MutableAllocator()
    telemetry = _PaperTelemetry()
    telemetry.configure(allocator, EmptyC2KVPool(), bytes_per_kv_token=4)
    telemetry.start(
        server_request_id="joint-peak",
        outer_request_id="outer-joint-peak",
        phase="extraction",
        kind="c2kv_extract",
    )

    # First sample: pool20 + temporary10 = joint30.
    telemetry.sample(
        "large_temp_small_pool",
        tensors=[torch.zeros(10, dtype=torch.float32)],
        temporary_kv=True,
    )
    # Second sample: pool39 + temporary2 = joint41.  The requested peak is
    # 41, not pooled max39 + temporary max10 = the impossible value49.
    allocator.available = 61
    telemetry.sample(
        "small_temp_large_pool",
        tensors=[torch.zeros(2, dtype=torch.float32)],
        temporary_kv=True,
    )
    result = telemetry.finish(success=True)
    metrics = result["metrics"]
    assert metrics["request_peak_resident_kv_tokens"] == 41
    assert metrics["request_peak_resident_kv_bytes"] == 164
    assert metrics["request_peak_pooled_resident_kv_tokens"] == 39
    assert metrics["temporary_extraction_recovery_peak_kv_tokens"] == 10
    assert result["peak"]["event"] == "small_temp_large_pool"
    assert result["peak"]["kv"]["simultaneous_temporary_kv_tokens"] == 2


def test_nested_tensor_storage_is_deduplicated_globally():
    base = torch.zeros(8, dtype=torch.float32)
    measured = _PaperTelemetry._tensor_bytes(
        [[base[:4]], [base[4:]], base]
    )
    assert measured["logical_bytes"] == (4 + 4 + 8) * 4
    assert measured["storage_bytes"] == base.untyped_storage().nbytes()
