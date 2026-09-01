from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt import runtime_context as rc
from sglang.srt.arg_groups import overrides as overrides_module
from sglang.srt.layers.attention import triton_backend as triton_backend_module
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.models.deepseek_common.attention_forward_methods import (
    forward_mla_rocm,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestKimiK3RocmDcp(CustomTestCase):
    @staticmethod
    def _args(**overrides):
        values = {
            "attention_backend": None,
            "dcp_comm_backend": "ag_rs",
            "dcp_replicate_q_proj": None,
            "dcp_size": 8,
            "decode_attention_backend": "triton",
            "enable_symm_mem": False,
            "enable_unified_memory": False,
            "page_size": 128,
            "prefill_attention_backend": "aiter",
            "speculative_algorithm": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_rocm_triton_dcp_uses_packed_a2a(self):
        with patch.object(overrides_module, "is_hip", return_value=True):
            declared = overrides_module._kimi_k3_overrides(self._args(), None)

        self.assertEqual(
            declared,
            {
                "dcp_comm_backend": "a2a",
                "dcp_replicate_q_proj": False,
            },
        )

    def test_non_rocm_triton_dcp_is_rejected(self):
        with (
            patch.object(overrides_module, "is_hip", return_value=False),
            self.assertRaisesRegex(AssertionError, "cutedsl_mla.*tokenspeed_mla"),
        ):
            overrides_module._kimi_k3_overrides(self._args(), None)

    def test_rocm_triton_dcp_rejects_triton_prefill(self):
        with (
            patch.object(overrides_module, "is_hip", return_value=True),
            self.assertRaisesRegex(ValueError, "prefill backend 'aiter'"),
        ):
            overrides_module._kimi_k3_overrides(
                self._args(prefill_attention_backend="triton"), None
            )

    def test_rocm_triton_dcp_rejects_speculative_decode(self):
        with (
            patch.object(overrides_module, "is_hip", return_value=True),
            self.assertRaisesRegex(ValueError, "speculative decoding"),
        ):
            overrides_module._kimi_k3_overrides(
                self._args(speculative_algorithm="EAGLE"), None
            )

    def test_rocm_triton_dcp_rejects_unified_memory(self):
        with (
            patch.object(overrides_module, "is_hip", return_value=True),
            self.assertRaisesRegex(ValueError, "unified memory pool"),
        ):
            overrides_module._kimi_k3_overrides(
                self._args(enable_unified_memory=True), None
            )

    def test_triton_backend_owns_model_dcp_decode(self):
        helper = getattr(
            forward_mla_rocm, "is_model_managed_dcp_mla_decode_phase", None
        )
        self.assertIsNotNone(helper)
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_decode=lambda: True,
                is_target_verify=lambda: False,
            )
        )
        with rc.get_parallel().override(dcp_enabled=True):
            self.assertFalse(helper(forward_batch, "triton"))
            self.assertTrue(helper(forward_batch, "cutedsl_mla"))

    def test_triton_mla_write_uses_dcp_aware_split_writer(self):
        backend = TritonAttnBackend.__new__(TritonAttnBackend)
        backend.dcp_size = 8
        backend.use_mla = True
        backend.token_to_kv_pool = MagicMock()
        layer = SimpleNamespace(qk_head_dim=6, v_head_dim=4)
        k = torch.arange(12, dtype=torch.float32).view(2, 1, 6)
        out_cache_loc = torch.tensor([40, 41])

        self.assertTrue(hasattr(backend, "_set_mla_kv_buffer"))
        backend._set_mla_kv_buffer(layer, KVWriteLoc(out_cache_loc), k)

        args, kwargs = backend.token_to_kv_pool.set_mla_kv_buffer.call_args
        self.assertIs(args[0], layer)
        torch.testing.assert_close(args[1], out_cache_loc)
        torch.testing.assert_close(args[2], k[..., :4])
        torch.testing.assert_close(args[3], k[..., 4:])
        self.assertEqual(kwargs, {})

    def test_triton_mla_write_rejects_unified_memory_translation(self):
        backend = TritonAttnBackend.__new__(TritonAttnBackend)
        backend.dcp_size = 8
        backend.use_mla = True
        backend.token_to_kv_pool = MagicMock()
        layer = SimpleNamespace(qk_head_dim=6, v_head_dim=4)
        k = torch.arange(12, dtype=torch.float32).view(2, 1, 6)
        loc_info = KVWriteLoc(
            torch.tensor([40, 41]),
            full_loc=torch.tensor([24, 25]),
        )

        self.assertTrue(hasattr(backend, "_set_mla_kv_buffer"))
        with self.assertRaisesRegex(NotImplementedError, "unified-memory"):
            backend._set_mla_kv_buffer(layer, loc_info, k)

    @staticmethod
    def _triton_decode_case():
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
            output_lse = kwargs.get("output_lse")
            if output_lse is not None:
                output_lse.fill_(7.0)
            else:
                args[7].fill_(0.0)

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
            world_size=8,
            rank_in_group=0,
            all_gather=lambda tensor, dim: tensor.repeat(1, 8, 1),
            all_reduce=lambda tensor: tensor,
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
        reduced = torch.arange(8, dtype=torch.float32).view(2, 1, 4)
        return backend, group, layer, q, reduced

    def test_triton_decode_preserves_mha_merge_for_non_a2a_hip(self):
        backend, group, layer, q, reduced = self._triton_decode_case()

        with (
            patch.object(triton_backend_module, "_is_hip", True, create=True),
            patch.object(
                triton_backend_module,
                "get_parallel",
                return_value=SimpleNamespace(
                    dcp_comm_backend="ag_rs",
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
                "cp_lse_ag_out_rs_mha",
                return_value=reduced.transpose(0, 1),
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

        merge.assert_called_once()
        self.assertIsNone(backend.decode_attention_fwd.call_args.kwargs["output_lse"])
        self.assertTrue(backend.decode_attention_fwd.call_args.kwargs["has_mla"])
        self.assertEqual(
            backend.decode_attention_fwd.call_args.kwargs["page_size"],
            backend.page_size,
        )
        self.assertEqual(
            backend.decode_attention_fwd.call_args.kwargs["use_pdl"],
            backend.use_pdl,
        )
        self.assertTrue(
            torch.equal(
                merge.call_args.args[1],
                torch.zeros_like(merge.call_args.args[1]),
            )
        )
        self.assertFalse(torch.isnan(merge.call_args.args[0]).any())
        self.assertEqual(merge.call_args.args[2], group)
        torch.testing.assert_close(output, reduced.transpose(0, 1).reshape(1, 8))

    def test_triton_mha_decode_preserves_existing_merge_on_hip(self):
        backend, group, layer, q, reduced = self._triton_decode_case()
        backend.use_mla = False

        with (
            patch.object(triton_backend_module, "_is_hip", True, create=True),
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
                create=True,
            ) as a2a_merge,
            patch.object(
                triton_backend_module,
                "cp_lse_ag_out_rs_mha",
                return_value=reduced.transpose(0, 1),
            ) as mha_merge,
        ):
            backend.forward_decode(
                q,
                None,
                None,
                layer,
                SimpleNamespace(),
                save_kv_cache=False,
            )

        a2a_merge.assert_not_called()
        mha_merge.assert_called_once()
        self.assertIsNone(backend.decode_attention_fwd.call_args.kwargs["output_lse"])

    def test_triton_decode_sanitizes_only_empty_direct_lse_rows(self):
        backend, group, layer, q, reduced = self._triton_decode_case()

        def decode_attention_fwd(*args, **kwargs):
            args[3].fill_(1.0)
            args[3][:, 0, :].fill_(float("nan"))
            args[3][:, 1, :].fill_(float("inf"))
            kwargs["output_lse"].fill_(7.0)
            kwargs["output_lse"][:, 0].fill_(-float("inf"))

        backend.decode_attention_fwd = MagicMock(side_effect=decode_attention_fwd)

        with (
            patch.object(triton_backend_module, "_is_hip", True, create=True),
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
                return_value=reduced.transpose(0, 1),
                create=True,
            ) as merge,
        ):
            backend.forward_decode(
                q,
                None,
                None,
                layer,
                SimpleNamespace(),
                save_kv_cache=False,
            )

        merged_output = merge.call_args.args[0]
        self.assertTrue(torch.equal(merged_output[:, 0, :], torch.zeros(1, 4)))
        self.assertTrue(torch.isposinf(merged_output[:, 1, :]).all())
        self.assertTrue(torch.equal(merged_output[:, 2:, :], torch.ones(1, 14, 4)))

    def test_triton_decode_uses_packed_a2a_with_natural_log_lse(self):
        backend, group, layer, q, reduced = self._triton_decode_case()

        with (
            patch.object(triton_backend_module, "_is_hip", True, create=True),
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
                return_value=reduced.transpose(0, 1),
                create=True,
            ) as merge,
            patch.object(
                triton_backend_module,
                "cp_lse_ag_out_rs_mha",
            ) as mha_merge,
        ):
            output = backend.forward_decode(
                q,
                None,
                None,
                layer,
                SimpleNamespace(),
                save_kv_cache=False,
            )

        merge.assert_called_once()
        self.assertFalse(torch.isnan(merge.call_args.args[0]).any())
        self.assertTrue(merge.call_args.kwargs["is_lse_base_on_e"])
        self.assertEqual(merge.call_args.kwargs["comm_backend"], "a2a")
        mha_merge.assert_not_called()
        torch.testing.assert_close(output, reduced.transpose(0, 1).reshape(1, 8))

    def test_triton_decode_preserves_mha_merge_on_cuda(self):
        backend, group, layer, q, reduced = self._triton_decode_case()

        with (
            patch.object(triton_backend_module, "_is_hip", False, create=True),
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
                "cp_lse_ag_out_rs_mha",
                return_value=reduced.transpose(0, 1),
            ) as mha_merge,
        ):
            backend.forward_decode(
                q,
                None,
                None,
                layer,
                SimpleNamespace(),
                save_kv_cache=False,
            )

        mha_merge.assert_called_once()
        self.assertIsNone(backend.decode_attention_fwd.call_args.kwargs["output_lse"])

    def test_internal_state_reports_allocator_token_capacity(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_total_num_tokens = 1024
        scheduler.metrics_reporter = SimpleNamespace(
            last_gen_throughput=0.0,
            spec_total_num_forward_ct=0,
        )
        scheduler.draft_worker = None
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(weight_load_mem_usage=1.0),
            graph_memory_usage={},
        )
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            size=7680, get_kvcache=lambda: SimpleNamespace(mem_usage=2.0)
        )
        scheduler.startup_available_gpu_memory_gb = 3.0
        scheduler.swa_tokens_per_layer = None
        scheduler.startup_time = {}
        scheduler.max_running_requests = 40
        scheduler.spec_algorithm = SimpleNamespace(
            is_none=lambda: True,
            is_dspark=lambda: False,
        )

        context = SimpleNamespace(resolved_server_args_dict=lambda: {})
        execution = SimpleNamespace(moe=SimpleNamespace(elastic_ep_backend=None))
        with (
            rc.get_parallel().override(dcp_enabled=True, attn_dcp_size=8),
            patch.object(scheduler_module, "get_context", return_value=context),
            patch.object(scheduler_module, "get_exec", return_value=execution),
        ):
            state = scheduler.get_internal_state(None).internal_state

        self.assertEqual(state["memory_usage"]["token_capacity"], 7680)
        self.assertEqual(scheduler.max_total_num_tokens, 1024)


if __name__ == "__main__":
    import unittest

    unittest.main()
