"""Kimi-K3 fused KDA decode must match the existing unfused decode chain.

The fused kernel replaces:

    causal_conv1d_update -> kda_packed_decode -> sigmoid-gated RMSNorm

CUDA coverage spans Kimi-K3 TP8/TP16/TP32 (H = 12/6/3). ROCm coverage targets
the gfx950 TP8 path with H = 12 and BF16 recurrent state.
"""

import pytest
import torch

from sglang.kernels.ops.attention import kda_fused_decode
from sglang.kernels.ops.attention.fla.fused_norm_gate import rms_norm_gated
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_kda_packed_decode,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=45, suite="stage-b-test-1-gpu-small-amd-mi35x")

_HEAD_DIM = 128
_CONV_STATE_W = 3
_SLOTS = 8
_BATCH = 4
_HIP_HEADS = 12
_HIP_STEPS = 2


def _randn(shape, dtype, generator, scale=1.0):
    return (torch.randn(shape, device="cuda", generator=generator) * scale).to(dtype)


def _make_case(
    heads: int,
    seed: int,
    batch: int = _BATCH,
    slots: int = _SLOTS,
    state_dtype: torch.dtype = torch.float32,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    seg = heads * _HEAD_DIM
    conv_dim = 3 * seg

    # Keep magnitudes moderate so fp32 state updates stay in a stable range.
    mixed_qkv = _randn((batch, conv_dim), torch.bfloat16, generator, scale=0.2)
    a = _randn((batch, seg), torch.bfloat16, generator, scale=0.2)
    b = _randn((batch, heads), torch.bfloat16, generator, scale=0.2)
    onorm_g = _randn((batch, seg), torch.bfloat16, generator, scale=0.2)

    conv_states = _randn(
        (slots, _CONV_STATE_W, conv_dim), torch.bfloat16, generator, scale=0.2
    )
    ssm_states = _randn(
        (slots, heads, _HEAD_DIM, _HEAD_DIM), state_dtype, generator, scale=0.02
    )
    cache_indices = torch.arange(batch, device="cuda", dtype=torch.int32)

    conv_weights = _randn((conv_dim, 4), torch.float32, generator, scale=0.1)
    conv_bias = _randn((conv_dim,), torch.float32, generator, scale=0.05)
    a_log = _randn((heads,), torch.float32, generator, scale=0.1)
    dt_bias = _randn((seg,), torch.float32, generator, scale=0.1)
    onorm_weight = _randn((_HEAD_DIM,), torch.float32, generator, scale=0.1) + 1.0

    return (
        mixed_qkv,
        a,
        b,
        onorm_g,
        conv_states,
        ssm_states,
        cache_indices,
        conv_weights,
        conv_bias,
        a_log,
        dt_bias,
        onorm_weight,
    )


def _run_unfused_reference(
    mixed_qkv,
    a,
    b,
    onorm_g,
    conv_states,
    ssm_states,
    cache_indices,
    conv_weights,
    conv_bias,
    a_log,
    dt_bias,
    onorm_weight,
    lower_bound=None,
):
    batch = mixed_qkv.shape[0]
    heads = ssm_states.shape[-3]
    qkv = causal_conv1d_update(
        mixed_qkv,
        conv_states.transpose(-1, -2),
        conv_weights,
        conv_bias,
        activation="silu",
        conv_state_indices=cache_indices,
    )
    out = torch.empty((batch, 1, heads, _HEAD_DIM), dtype=torch.bfloat16, device="cuda")
    out, _ = fused_recurrent_kda_packed_decode(
        qkv,
        a,
        b,
        a_log,
        dt_bias,
        _HEAD_DIM**-0.5,
        ssm_states,
        out,
        cache_indices,
        use_qk_l2norm_in_kernel=True,
        lower_bound=lower_bound,
    )
    ref = rms_norm_gated(
        out,
        onorm_g.view(1, batch, heads, _HEAD_DIM),
        onorm_weight,
        None,
        activation="sigmoid",
        eps=1e-6,
    )
    return ref.transpose(0, 1).contiguous()


def _run_fused(
    mixed_qkv,
    a,
    b,
    onorm_g,
    conv_states,
    ssm_states,
    cache_indices,
    conv_weights,
    conv_bias,
    a_log,
    dt_bias,
    onorm_weight,
    lower_bound=None,
    transposed_weights=None,
):
    heads = ssm_states.shape[-3]
    if transposed_weights is None:
        transposed_weights = tuple(
            weight.t().contiguous()
            for weight in conv_weights.split(heads * _HEAD_DIM, dim=0)
        )
    return kda_fused_decode.kda_fused_decode(
        mixed_qkv,
        a,
        b,
        conv_states,
        *transposed_weights,
        conv_bias,
        a_log,
        dt_bias,
        onorm_g,
        onorm_weight,
        ssm_states,
        cache_indices,
        scale=_HEAD_DIM**-0.5,
        onorm_eps=1e-6,
        lower_bound=lower_bound,
    )


def _is_gfx950() -> bool:
    return bool(
        torch.cuda.is_available()
        and torch.version.hip is not None
        and torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx950"
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="requires NVIDIA CUDA",
)
@pytest.mark.parametrize(
    "heads,tp_size",
    [
        pytest.param(3, 32, id="tp32_h3"),
        pytest.param(6, 16, id="tp16_h6"),
        pytest.param(12, 8, id="tp8_h12"),
    ],
)
def test_kda_fused_decode_matches_unfused_chain(heads: int, tp_size: int):
    (
        mixed_qkv,
        a,
        b,
        onorm_g,
        conv_states,
        ssm_states,
        cache_indices,
        conv_weights,
        conv_bias,
        a_log,
        dt_bias,
        onorm_weight,
    ) = _make_case(heads=heads, seed=20260731 + tp_size)

    conv_ref = conv_states.clone()
    conv_fused = conv_states.clone()
    state_ref = ssm_states.clone()
    state_fused = ssm_states.clone()

    assert kda_fused_decode.covered(
        mixed_qkv,
        a,
        b,
        conv_fused,
        state_fused,
        cache_indices,
        onorm_g,
    )

    ref = _run_unfused_reference(
        mixed_qkv.clone(),
        a,
        b,
        onorm_g,
        conv_ref,
        state_ref,
        cache_indices,
        conv_weights,
        conv_bias,
        a_log,
        dt_bias,
        onorm_weight,
    )
    fused = _run_fused(
        mixed_qkv.clone(),
        a,
        b,
        onorm_g,
        conv_fused,
        state_fused,
        cache_indices,
        conv_weights,
        conv_bias,
        a_log,
        dt_bias,
        onorm_weight,
    )
    torch.cuda.synchronize()

    # JIT log breadcrumb for PR/CI evidence that the fused fixed-head branch ran.
    print(f"K3 fused KDA decode test used fused path: TP{tp_size}, H={heads}")

    torch.testing.assert_close(fused, ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(state_fused, state_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(conv_fused, conv_ref, rtol=0, atol=0)


@pytest.mark.skipif(not _is_gfx950(), reason="requires gfx950 ROCm")
def test_kda_fused_decode_hip_coverage():
    case = _make_case(
        heads=_HIP_HEADS,
        batch=1,
        slots=3,
        seed=20260831,
        state_dtype=torch.bfloat16,
    )
    mixed_qkv, a, b, onorm_g, conv_states, ssm_states, cache_indices = case[:7]

    assert kda_fused_decode.covered(
        mixed_qkv, a, b, conv_states, ssm_states, cache_indices, onorm_g
    )
    assert not kda_fused_decode.covered(
        mixed_qkv, a, b, conv_states, ssm_states.float(), cache_indices, onorm_g
    )
    assert not kda_fused_decode.covered(
        mixed_qkv,
        a,
        b,
        conv_states,
        ssm_states.transpose(-1, -2),
        cache_indices,
        onorm_g,
    )

    h6 = _make_case(6, 20260832, batch=1, slots=3, state_dtype=torch.bfloat16)
    assert not kda_fused_decode.covered(h6[0], h6[1], h6[2], h6[4], h6[5], h6[6], h6[3])


@pytest.mark.skipif(not _is_gfx950(), reason="requires gfx950 ROCm")
@pytest.mark.parametrize(
    "batch,use_graph",
    [
        pytest.param(1, False, id="eager_b1"),
        pytest.param(64, True, id="graph_b64"),
    ],
)
def test_kda_fused_decode_hip_matches_unfused(batch: int, use_graph: bool):
    slots = batch + 2
    case = _make_case(
        heads=_HIP_HEADS,
        batch=batch,
        slots=slots,
        seed=20260901 + batch,
        state_dtype=torch.bfloat16,
    )
    (
        mixed_qkv,
        a,
        b,
        onorm_g,
        conv_states,
        dense_state,
        cache_indices,
        conv_weights,
        conv_bias,
        a_log,
        dt_bias,
        onorm_weight,
    ) = case
    state_storage = torch.empty(
        (slots, 2, *dense_state.shape[1:]),
        dtype=dense_state.dtype,
        device=dense_state.device,
    )
    ssm_states = state_storage[:, 0]
    ssm_states.copy_(dense_state)
    state_storage[:, 1].fill_(1.25)
    state_canary = state_storage[:, 1].clone()
    conv_initial = conv_states.clone()
    state_initial = dense_state.clone()
    conv_ref = conv_states.clone()
    state_ref = dense_state.clone()
    transposed_weights = tuple(
        weight.t().contiguous()
        for weight in conv_weights.split(_HIP_HEADS * _HEAD_DIM, dim=0)
    )

    assert ssm_states.stride(0) != _HIP_HEADS * _HEAD_DIM * _HEAD_DIM
    assert kda_fused_decode.covered(
        mixed_qkv, a, b, conv_states, ssm_states, cache_indices, onorm_g
    )

    def run_fused():
        return _run_fused(
            mixed_qkv,
            a,
            b,
            onorm_g,
            conv_states,
            ssm_states,
            cache_indices,
            conv_weights,
            conv_bias,
            a_log,
            dt_bias,
            onorm_weight,
            lower_bound=-5.0,
            transposed_weights=transposed_weights,
        )

    run_fused()
    torch.cuda.synchronize()
    conv_states.copy_(conv_initial)
    ssm_states.copy_(state_initial)

    graph = None
    graph_out = None
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = run_fused()
        conv_states.copy_(conv_initial)
        ssm_states.copy_(state_initial)

    for step in range(_HIP_STEPS):
        step_case = _make_case(
            heads=_HIP_HEADS,
            batch=batch,
            slots=slots,
            seed=202609100 + batch * 10 + step,
            state_dtype=torch.bfloat16,
        )
        mixed_qkv.copy_(step_case[0])
        a.copy_(step_case[1])
        b.copy_(step_case[2])
        onorm_g.copy_(step_case[3])
        next_indices = torch.arange(batch, device="cuda", dtype=torch.int32)
        if step:
            if batch == 1:
                next_indices[0] = slots - 1
            else:
                next_indices = torch.roll(next_indices, shifts=1)
                next_indices[-1] = -1
        cache_indices.copy_(next_indices)

        ref = _run_unfused_reference(
            mixed_qkv,
            a,
            b,
            onorm_g,
            conv_ref,
            state_ref,
            cache_indices,
            conv_weights,
            conv_bias,
            a_log,
            dt_bias,
            onorm_weight,
            lower_bound=-5.0,
        )
        if graph is None:
            fused = run_fused()
        else:
            graph.replay()
            fused = graph_out
        torch.cuda.synchronize()

        torch.testing.assert_close(fused, ref, rtol=2e-2, atol=2e-2)
        assert torch.equal(ssm_states, state_ref)
        assert torch.equal(conv_states, conv_ref)
        assert torch.equal(state_storage[:, 1], state_canary)
        if batch > 1 and step:
            assert torch.count_nonzero(fused[:, -1]).item() == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
