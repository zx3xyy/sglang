from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import overrides as overrides_module
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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

    def test_enables_rocm_triton_dcp(self):
        with patch.object(overrides_module, "is_hip", return_value=True):
            declared = overrides_module._kimi_k3_overrides(self._args(), None)

        self.assertEqual(
            declared,
            {
                "dcp_comm_backend": "a2a",
                "dcp_replicate_q_proj": False,
            },
        )

    def test_rejects_unsupported_rocm_triton_dcp_modes(self):
        cases = (
            (
                {"prefill_attention_backend": "triton"},
                ValueError,
                "prefill backend 'aiter'",
            ),
            (
                {"speculative_algorithm": "EAGLE"},
                ValueError,
                "speculative decoding",
            ),
            (
                {"enable_unified_memory": True},
                ValueError,
                "unified memory pool",
            ),
        )
        for changes, error, message in cases:
            with self.subTest(changes=changes):
                with (
                    patch.object(overrides_module, "is_hip", return_value=True),
                    self.assertRaisesRegex(error, message),
                ):
                    overrides_module._kimi_k3_overrides(self._args(**changes), None)


if __name__ == "__main__":
    import unittest

    unittest.main()
