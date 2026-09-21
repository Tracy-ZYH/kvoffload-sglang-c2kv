"""Resolve annotated tool definitions in the server's final token frame."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def resolve_schema_token_spans(
    *,
    rendered_prompt: str,
    prompt_ids: Sequence[int],
    message_contents: Sequence[str],
    schema_spans: Sequence[Mapping],
    tokenizer,
    boundary_policy: str = "inside",
) -> list[dict]:
    """Protect tokens crossing a schema boundary; count protocol by overlap."""

    if boundary_policy not in {"inside", "overlap"}:
        raise ValueError("TOOL_KV_BOUNDARY_POLICY_INVALID")

    encoded = tokenizer(
        rendered_prompt, add_special_tokens=False, return_offsets_mapping=True
    )
    token_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids, offsets = token_ids[0], offsets[0]
    if list(token_ids) != list(prompt_ids):
        raise ValueError("TOOL_KV_RENDERED_PROMPT_TOKEN_MISMATCH")
    if not schema_spans:
        raise ValueError("TOOL_KV_SCHEMA_SPANS_REQUIRED")

    content_starts = []
    cursor = 0
    for content in message_contents:
        if not isinstance(content, str):
            content_starts.append(None)
            continue
        start = rendered_prompt.find(content, cursor) if content else cursor
        if start < 0:
            raise ValueError("TOOL_KV_MESSAGE_CONTENT_NOT_IN_RENDERED_PROMPT")
        content_starts.append(start)
        cursor = start + len(content)

    resolved = []
    occupied = set()
    occupied_text = []
    for raw in schema_spans:
        if not isinstance(raw, Mapping):
            raise ValueError("TOOL_KV_SCHEMA_SPAN_INVALID")
        schema_index = int(raw["schema_index"])
        message_index = int(raw["message_index"])
        start = int(raw["start"])
        end = int(raw["end"])
        if not 0 <= message_index < len(message_contents):
            raise ValueError("TOOL_KV_SCHEMA_INDEX_INVALID")
        content = message_contents[message_index]
        if not isinstance(content, str) or not 0 <= start < end <= len(content) or content[start:end] != raw.get("text"):
            raise ValueError("TOOL_KV_SCHEMA_TEXT_MISMATCH")
        absolute_start = content_starts[message_index] + start
        absolute_end = content_starts[message_index] + end
        if any(absolute_start < right and left < absolute_end
               for left, right in occupied_text):
            raise ValueError("TOOL_KV_SCHEMA_TOKEN_SPANS_OVERLAP")
        occupied_text.append((absolute_start, absolute_end))
        if boundary_policy == "inside":
            included = [
                index for index, (left, right) in enumerate(offsets)
                if absolute_start <= left < right <= absolute_end
            ]
        else:
            included = [
                index for index, (left, right) in enumerate(offsets)
                if left < right and left < absolute_end and right > absolute_start
            ]
        if not included:
            # A short prose value can share a BPE token with surrounding
            # executable syntax. It has no evictable token; the crossing
            # token stays protected because it is absent from this result.
            if boundary_policy == "inside":
                continue
            raise ValueError("TOOL_KV_SCHEMA_HAS_NO_WHOLE_TOKENS")
        token_start = included[0]
        token_end = included[-1] + 1
        if included != list(range(token_start, token_end)) or occupied.intersection(included):
            raise ValueError("TOOL_KV_SCHEMA_TOKEN_SPANS_OVERLAP")
        occupied.update(included)
        resolved.append(
            {
                "schema_index": schema_index,
                "token_start": token_start,
                "token_end": token_end,
            }
        )
    return sorted(resolved, key=lambda item: item["token_start"])
