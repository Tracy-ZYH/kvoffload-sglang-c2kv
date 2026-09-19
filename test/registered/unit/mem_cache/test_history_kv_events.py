from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "python"))

from sglang.srt.mem_cache.history_kv_events import (
    resolve_history_kv_event_token_spans,
)

EVENTS = [
    {"role": "system", "phase": "others", "message_index": 0},
    {"role": "assistant", "phase": "act", "message_index": 1},
    {"role": "user", "phase": "tool", "message_index": 2},
]


def test_resolves_absolute_half_open_spans_after_synthetic_system_shift() -> None:
    spans = resolve_history_kv_event_token_spans(
        total_tokens=13,
        message_prefix_token_counts=[1, 5, 9, 13],
        event_messages=EVENTS,
    )
    assert spans == [
        {"role": "system", "phase": "others", "start": 1, "end": 5, "message_index": 0},
        {"role": "assistant", "phase": "act", "start": 5, "end": 9, "message_index": 1},
        {"role": "user", "phase": "tool", "start": 9, "end": 13, "message_index": 2},
    ]


@pytest.mark.parametrize(
    ("counts", "total", "events", "error"),
    [
        ([0, 5, 4, 9], 9, EVENTS, "non-decreasing"),
        ([0, 5, 9, 10], 11, EVENTS, "end at total_tokens"),
        ([0, 5, 9, 13], 13, EVENTS[:2], "one event hint"),
        ([0, 5, 9, 13], 13, [EVENTS[0], EVENTS[0], EVENTS[2]], "duplicate"),
        (
            [0, 5, 9, 13],
            13,
            [EVENTS[0], EVENTS[1], {**EVENTS[2], "message_index": 3}],
            "outside",
        ),
    ],
)
def test_rejects_ambiguous_or_out_of_range_event_boundaries(
    counts, total, events, error
) -> None:
    with pytest.raises(ValueError, match=error):
        resolve_history_kv_event_token_spans(
            total_tokens=total,
            message_prefix_token_counts=counts,
            event_messages=events,
        )
