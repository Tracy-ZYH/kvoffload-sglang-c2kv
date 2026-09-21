"""Exercise the actual HTTP request classes, including their default fields."""
from sglang.srt.entrypoints.openai.protocol import (
    C2KVNativePackedChunk,
    C2KVNativePackedGenerateRequest,
)


def test_native_request_keeps_history_only_defaults_and_tool_extensions():
    request = C2KVNativePackedGenerateRequest(
        encoding_scope="scope", max_extraction_calls=1,
        workspace_input_ids=[1, 2], sampling_params={"max_new_tokens": 1})
    assert request.max_tool_extraction_calls is None
    assert request.raw_tool_segments == []
    assert request.tool_gist_segments == []
    assert request.paper_whole_full_kv_tokens is None
    assert request.sampling_profile == "greedy-v1"
    chunk = C2KVNativePackedChunk(
        chunk_id="tool", event_id="tool", part_index=0,
        source_token_start=0, source_token_end=1, token_ids=[1],
        projection_set="tool", compression_ratio=8)
    request.encoder_chunks = [chunk]
    restored = C2KVNativePackedGenerateRequest.model_validate(request.model_dump())
    assert restored.encoder_chunks[0].projection_set == "tool"
    assert restored.encoder_chunks[0].compression_ratio == 8
