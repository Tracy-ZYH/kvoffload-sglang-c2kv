"""Chat response regression for C2KV detector prefill features."""

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.utils import process_hidden_states_from_ret


def _request(**kwargs):
    return ChatCompletionRequest.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "return_hidden_states": True,
            **kwargs,
        }
    )


def test_full_capture_keeps_prefill_segment_for_session_detector():
    prefill = [[0.1, 0.2], [0.3, 0.4]]
    final_decode = [0.5, 0.6]
    ret = {"meta_info": {"hidden_states": [prefill, final_decode]}}
    request = _request(c2kv_return_full_hidden_states=True)

    assert process_hidden_states_from_ret(ret, request) == [prefill, final_decode]
    assert process_hidden_states_from_ret(
        {"meta_info": {"hidden_states": [prefill]}}, request
    ) == [prefill]


def test_standard_chat_hidden_state_response_keeps_last_token_contract():
    prefill = [[0.1, 0.2], [0.3, 0.4]]
    final_decode = [0.5, 0.6]
    ret = {"meta_info": {"hidden_states": [prefill, final_decode]}}

    assert process_hidden_states_from_ret(ret, _request()) == final_decode
    assert process_hidden_states_from_ret(
        ret, _request(return_hidden_states=False, c2kv_return_full_hidden_states=True)
    ) is None
