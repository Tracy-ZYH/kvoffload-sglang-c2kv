"""Model-free invariants for persistent, shared-index history KV sessions.

Positions identify canonical prompt tokens, not allocator slots. Keeping this
ledger independent of token IDs detects resurrection even for repeated text.
"""
import hashlib


ATTENTION_SELECTION_METHODS = {
    "h2o",
    "snapkv",
    "snapkv_persistent",
    "pyramid",
    "pyramidkv",
}


def position_summary(positions):
    positions = list(positions)
    return {"count": len(positions), "min": min(positions, default=None),
            "max": max(positions, default=None),
            "sha256": hashlib.sha256(
                ",".join(map(str, positions)).encode()).hexdigest()}


def append_resident_positions(previous, logical_prefix, canonical_length):
    previous = list(previous)
    if previous != sorted(set(previous)) or any(p < 0 or p >= logical_prefix for p in previous):
        raise ValueError("PERSISTENT_HISTORY_INVALID_RESIDENT_POSITIONS")
    if canonical_length < logical_prefix:
        raise ValueError("PERSISTENT_HISTORY_NON_APPEND_REQUEST")
    # No full-history materialization: missing positions below logical_prefix
    # are NEVER candidates, even if the incoming text contains them.
    return previous + list(range(logical_prefix, canonical_length))


def physical_history_range(positions, history_start, history_end):
    if not 0 <= history_start <= history_end:
        raise ValueError("PERSISTENT_HISTORY_INVALID_CANONICAL_BOUNDARY")
    return (sum(p < history_start for p in positions),
            sum(p < history_end for p in positions))


def selection_query_window(
    method, prefix_len, history_end, prompt_len, recent_window
):
    """Return newly-prefilled tail queries used to score resident history."""
    method = str(method or "").strip().lower()
    if method not in ATTENTION_SELECTION_METHODS:
        return None
    prefix_len = int(prefix_len)
    history_end = int(history_end)
    prompt_len = int(prompt_len)
    recent_window = max(1, int(recent_window or 64))
    if not 0 <= prefix_len <= prompt_len:
        raise ValueError("HISTORY_KV_INVALID_SELECTION_PREFIX")
    if not 0 <= history_end <= prompt_len:
        raise ValueError("HISTORY_KV_INVALID_SELECTION_HISTORY_END")
    if prompt_len <= prefix_len:
        raise ValueError("HISTORY_KV_SELECTION_REQUIRES_NEW_QUERY")

    query_end = prompt_len
    query_start = max(prefix_len, history_end, query_end - recent_window)
    if query_start >= query_end:
        query_start = query_end - 1
    return query_start, query_end


def compact_positions(positions, history_start, history_end, selected):
    positions = list(positions)
    selected = list(selected)
    if not 0 <= history_start <= history_end <= len(positions):
        raise ValueError("PERSISTENT_HISTORY_INVALID_PHYSICAL_BOUNDARY")
    if selected != sorted(set(selected)) or any(not 0 <= i < history_end-history_start for i in selected):
        raise ValueError("PERSISTENT_HISTORY_INVALID_SELECTION")
    result = positions[:history_start] + [positions[history_start+i] for i in selected] + positions[history_end:]
    if not set(result).issubset(positions) or len(result) != len(set(result)):
        raise ValueError("PERSISTENT_HISTORY_TOKEN_RESURRECTION")
    return result
