from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention import triton_backend as triton_backend_module
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTritonMlaDcp(CustomTestCase):
    def test_mla_write_uses_dcp_aware_split_writer(self):
        backend = TritonAttnBackend.__new__(TritonAttnBackend)
        backend.token_to_kv_pool = MagicMock()
        layer = SimpleNamespace(qk_head_dim=6, v_head_dim=4)
        k = torch.arange(12, dtype=torch.float32).view(2, 1, 6)
        logical_loc = torch.tensor([40, 41])

        backend._set_mla_kv_buffer(layer, KVWriteLoc(logical_loc), k)

        args, kwargs = backend.token_to_kv_pool.set_mla_kv_buffer.call_args
        self.assertIs(args[0], layer)
        torch.testing.assert_close(args[1], logical_loc)
        torch.testing.assert_close(args[2], k[..., :4])
        torch.testing.assert_close(args[3], k[..., 4:])
        self.assertEqual(kwargs, {})

        with self.assertRaisesRegex(NotImplementedError, "unified-memory"):
            backend._set_mla_kv_buffer(
                layer,
                KVWriteLoc(logical_loc, full_loc=torch.tensor([24, 25])),
                k,
            )

    def test_hip_mla_decode_uses_direct_lse_a2a(self):
        backend = TritonAttnBackend.__new__(TritonAttnBackend)
        backend.dcp_size = 8
        backend.use_mla = True
        backend.use_pdl = False
        backend.page_size = 128
        backend.swa_v_head_dim = None
        backend.max_kv_splits = 1
        backend.token_to_kv_pool = MagicMock()

        def decode_attention_fwd(*args, **kwargs):
            args[3].zero_()
            kwargs["output_lse"].fill_(7.0)

        backend.decode_attention_fwd = MagicMock(side_effect=decode_attention_fwd)
        backend.forward_metadata = SimpleNamespace(
            window_kv_indptr=None,
            window_kv_indices=None,
            kv_indptr=torch.tensor([0, 1]),
            kv_indices=torch.tensor([0]),
            attn_logits=torch.empty(1, 16, 1, 4),
            attn_lse=torch.zeros(1, 16, 1),
            num_kv_splits=torch.ones(1, dtype=torch.int32),
            swa_attn_logits=None,
        )
        group = SimpleNamespace(
            all_gather=lambda tensor, dim: tensor.repeat(1, 8, 1),
        )
        layer = SimpleNamespace(
            tp_q_head_num=2,
            qk_head_dim=4,
            v_head_dim=4,
            logit_capping_method="tanh",
            logit_cap=0.0,
            sliding_window_size=-1,
            k_scale=None,
            v_scale=None,
            scaling=1.0,
            layer_id=0,
            xai_temperature_len=None,
        )
        q = torch.zeros(1, 8)
        reduced = torch.arange(8, dtype=torch.float32).view(1, 2, 4)

        with (
            patch.object(triton_backend_module, "_is_hip", True),
            patch.object(
                triton_backend_module,
                "get_parallel",
                return_value=SimpleNamespace(
                    dcp_comm_backend="a2a",
                    dcp_group=group,
                ),
            ),
            patch.object(
                triton_backend_module,
                "use_symmetric_memory",
                return_value=nullcontext(),
            ),
            patch.object(
                triton_backend_module,
                "dcp_a2a_lse_reduce",
                return_value=reduced,
            ) as merge,
        ):
            output = backend.forward_decode(
                q,
                None,
                None,
                layer,
                SimpleNamespace(),
                save_kv_cache=False,
            )

        self.assertIsNotNone(
            backend.decode_attention_fwd.call_args.kwargs["output_lse"]
        )
        merge.assert_called_once()
        self.assertTrue(merge.call_args.kwargs["is_lse_base_on_e"])
        self.assertEqual(merge.call_args.kwargs["comm_backend"], "a2a")
        torch.testing.assert_close(output, reduced.reshape(1, 8))


if __name__ == "__main__":
    import unittest

    unittest.main()
