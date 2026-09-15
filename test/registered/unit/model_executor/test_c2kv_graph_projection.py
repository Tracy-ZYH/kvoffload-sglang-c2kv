import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.cuda_graph_runner import DecodeInputBuffers


class TestC2KVGraphProjectionMask(unittest.TestCase):
    @staticmethod
    def create_buffers(capacity: int) -> DecodeInputBuffers:
        buffers = object.__new__(DecodeInputBuffers)
        buffers.c2kv_use_gist_projection = torch.ones(capacity, dtype=torch.bool)
        return buffers

    def test_constructor_defaults_projection_buffer_off(self):
        buffers = DecodeInputBuffers.create(
            device=torch.device("cpu"),
            max_bs=2,
            max_num_token=2,
            hidden_size=4,
            vocab_size=8,
            dtype=torch.float32,
            dp_size=1,
            pp_size=1,
            is_encoder_decoder=False,
            require_mlp_tp_gather=False,
            seq_len_fill_value=1,
            encoder_len_fill_value=0,
            num_tokens_per_bs=1,
            cache_loc_dtype=torch.int64,
            enable_mamba_track=False,
        )

        self.assertIsNone(buffers.c2kv_use_gist_projection)

    def test_mixed_requests_and_padding_are_preserved(self):
        buffers = self.create_buffers(4)
        forward_batch = SimpleNamespace(
            c2kv_use_gist_projection=torch.tensor([True, False, True])
        )

        buffers.update_c2kv_gist_projection_mask(
            forward_batch,
            raw_num_token=3,
            graph_num_token=4,
        )

        torch.testing.assert_close(
            buffers.c2kv_use_gist_projection,
            torch.tensor([True, False, True, False]),
        )

    def test_absent_mask_resets_previous_batch(self):
        buffers = self.create_buffers(4)
        forward_batch = SimpleNamespace(c2kv_use_gist_projection=None)

        buffers.update_c2kv_gist_projection_mask(
            forward_batch,
            raw_num_token=2,
            graph_num_token=4,
        )

        torch.testing.assert_close(
            buffers.c2kv_use_gist_projection,
            torch.zeros(4, dtype=torch.bool),
        )

    def test_rejects_mask_that_does_not_match_real_tokens(self):
        buffers = self.create_buffers(4)
        forward_batch = SimpleNamespace(
            c2kv_use_gist_projection=torch.tensor([True, False])
        )

        with self.assertRaisesRegex(RuntimeError, "mask shape mismatch"):
            buffers.update_c2kv_gist_projection_mask(
                forward_batch,
                raw_num_token=3,
                graph_num_token=4,
            )


if __name__ == "__main__":
    unittest.main()
