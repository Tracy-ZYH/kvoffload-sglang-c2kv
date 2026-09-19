"""Resolve proxy message semantics to exact history-token event spans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise

HISTORY_KV_EVENT_ROLES = frozenset({"system", "user", "assistant", "tool"})
HISTORY_KV_EVENT_PHASES = frozenset({"others", "act", "tool"})


def resolve_history_kv_event_token_spans(
    *,
    total_tokens: int,
    message_prefix_token_counts: Sequence[int],
    event_messages: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return absolute half-open spans for every assembled history message.

    ``message_prefix_token_counts`` contains the token length before the first
    assembled message followed by the length after every message.  The final
    count must equal ``total_tokens``; callers exclude any assistant
    generation suffix from both values.  ``message_index`` refers to the
    assembled message list after a synthetic system message has been inserted.
    """

    if total_tokens < 0:
        raise ValueError("history KV total_tokens must be non-negative")
    boundaries = [int(item) for item in message_prefix_token_counts]
    if len(boundaries) < 2:
        raise ValueError("history KV prefix counts need initial and final boundaries")
    if boundaries[0] < 0 or boundaries[-1] != total_tokens:
        raise ValueError(
            "history KV prefix counts must start in-range and end at total_tokens"
        )
    if any(right < left for left, right in pairwise(boundaries)):
        raise ValueError("history KV prefix counts must be non-decreasing")

    message_count = len(boundaries) - 1
    if len(event_messages) != message_count:
        raise ValueError(
            "history KV needs one event hint per assembled message: "
            f"expected {message_count}, got {len(event_messages)}"
        )
    by_index: dict[int, Mapping[str, object]] = {}
    for event in event_messages:
        if not isinstance(event, Mapping):
            raise TypeError("history KV event hints must be mappings")
        raw_index = event.get("message_index")
        if type(raw_index) is not int:
            raise ValueError("history KV message_index must be an integer")
        index = raw_index
        if index < 0 or index >= message_count:
            raise ValueError(
                f"history KV message_index is outside assembled messages: {index}"
            )
        if index in by_index:
            raise ValueError(f"duplicate history KV message_index: {index}")
        role = event.get("role")
        phase = event.get("phase")
        if role not in HISTORY_KV_EVENT_ROLES:
            raise ValueError(f"unsupported history KV event role: {role!r}")
        if phase not in HISTORY_KV_EVENT_PHASES:
            raise ValueError(f"unsupported history KV event phase: {phase!r}")
        by_index[index] = event
    if set(by_index) != set(range(message_count)):
        raise ValueError("history KV event hints do not cover every assembled message")

    return [
        {
            "role": by_index[index]["role"],
            "phase": by_index[index]["phase"],
            "start": boundaries[index],
            "end": boundaries[index + 1],
            "message_index": index,
        }
        for index in range(message_count)
    ]


__all__ = [
    "HISTORY_KV_EVENT_PHASES",
    "HISTORY_KV_EVENT_ROLES",
    "resolve_history_kv_event_token_spans",
]
