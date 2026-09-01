#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/warp.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cstdint>

#ifndef USE_ROCM
#error "kda_fused_decode_hip.cuh requires ROCm"
#endif

namespace sglang {

constexpr int kKdaHipHeads = 12;
constexpr int kKdaHipDim = 128;
constexpr int kKdaHipSeg = kKdaHipHeads * kKdaHipDim;
constexpr int kKdaHipConvDim = 3 * kKdaHipSeg;
constexpr int kKdaHipActiveWarps = kKdaHipDim / 32;
constexpr int kKdaHipSmallBatchThreads = 512;
constexpr int kKdaHipLargeBatchThreads = 256;
constexpr int kKdaHipSmallBatchMax = 16;

struct KdaHipSum2 {
  float x;
  float y;
};

using KdaHipFloat2 = float __attribute__((ext_vector_type(2)));

SGL_DEVICE KdaHipSum2 kda_hip_block_sum2_128(float x, float y, float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (warp < kKdaHipActiveWarps) {
    x = device::warp::reduce_sum<32>(x);
    y = device::warp::reduce_sum<32>(y);
    if (lane == 0) {
      scratch[warp] = x;
      scratch[kKdaHipActiveWarps + warp] = y;
    }
  }
  __syncthreads();

  if (warp == 0) {
    x = lane < kKdaHipActiveWarps ? scratch[lane] : 0.0f;
    y = lane < kKdaHipActiveWarps ? scratch[kKdaHipActiveWarps + lane] : 0.0f;
    x = device::warp::reduce_sum<32>(x);
    y = device::warp::reduce_sum<32>(y);
    if (lane == 0) {
      scratch[0] = x;
      scratch[1] = y;
    }
  }
  __syncthreads();
  return {scratch[0], scratch[1]};
}

SGL_DEVICE float kda_hip_block_sum_128(float value, float* scratch) {
  return kda_hip_block_sum2_128(value, 0.0f, scratch).x;
}

SGL_DEVICE float kda_hip_silu(float x) {
  return x / (1.0f + __expf(-x));
}

SGL_DEVICE float kda_hip_sigmoid(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

SGL_DEVICE float kda_hip_exp(float x) {
  return __expf(x);
}

SGL_DEVICE KdaHipFloat2 kda_hip_dot8_pair(const KdaHipFloat2 (&lhs)[8], const float (&rhs)[8]) {
  // Preserve Triton's paired reduction order for bitwise state parity.
  const KdaHipFloat2 seed_lhs = {lhs[1][0], lhs[0][1]};
  const KdaHipFloat2 seed_rhs = {rhs[1], rhs[0]};
  KdaHipFloat2 sum = seed_lhs * seed_rhs;
  const KdaHipFloat2 first_lhs = {lhs[0][0], lhs[1][1]};
  const KdaHipFloat2 first_rhs = {rhs[0], rhs[1]};
  sum += first_lhs * first_rhs;
#pragma unroll
  for (int i = 2; i < 8; ++i) {
    const KdaHipFloat2 packed_rhs = {rhs[i], rhs[i]};
    sum += lhs[i] * packed_rhs;
  }
  return sum;
}

SGL_DEVICE float kda_hip_wave64_sum(float value) {
  const int lane = threadIdx.x & 63;
  const int row_lane = lane & 15;
#pragma unroll
  for (int offset = 8; offset > 0; offset >>= 1) {
    const float shifted = __shfl_up(value, offset, 16);
    if (row_lane >= offset) {
      value += shifted;
    }
  }
  const float row0 = __shfl(value, 15, 64);
  const float row2 = __shfl(value, 47, 64);
  if (lane == 31) {
    value += row0;
  } else if (lane == 63) {
    value += row2;
  }
  const float lower_half = __shfl(value, 31, 64);
  if (lane == 63) {
    value += lower_half;
  }
  return __shfl(value, 63, 64);
}

SGL_DEVICE float kda_hip_subgroup16_sum(float value) {
  int bits = __builtin_bit_cast(int, value);
  int moved = __builtin_amdgcn_update_dpp(bits, bits, 0x118, 0xf, 0xc, false);
  moved = __builtin_amdgcn_update_dpp(moved, bits, 0x108, 0xf, 0x3, false);
  value += __builtin_bit_cast(float, moved);
  bits = __builtin_bit_cast(int, value);
  moved = __builtin_amdgcn_update_dpp(bits, bits, 0x114, 0xf, 0xa, false);
  moved = __builtin_amdgcn_update_dpp(moved, bits, 0x104, 0xf, 0x5, false);
  value += __builtin_bit_cast(float, moved);
  bits = __builtin_bit_cast(int, value);
  moved = __builtin_amdgcn_update_dpp(bits, bits, 0x04e, 0xf, 0xf, false);
  value += __builtin_bit_cast(float, moved);
  bits = __builtin_bit_cast(int, value);
  moved = __builtin_amdgcn_update_dpp(bits, bits, 0x0b1, 0xf, 0xf, false);
  return value + __builtin_bit_cast(float, moved);
}

template <int kThreads, bool kUseLowerBound>
__global__ __launch_bounds__(kThreads, kThreads == 512 ? 1 : 2) void kda_fused_decode_hip_kernel(
    const bf16_t* __restrict__ mixed_qkv,
    const bf16_t* __restrict__ a,
    const bf16_t* __restrict__ b,
    bf16_t* __restrict__ conv_states,
    const float* __restrict__ w_q_t,
    const float* __restrict__ w_k_t,
    const float* __restrict__ w_v_t,
    const float* __restrict__ conv_bias,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    const bf16_t* __restrict__ onorm_g,
    const float* __restrict__ onorm_weight,
    bf16_t* __restrict__ state,
    const int32_t* __restrict__ indices,
    bf16_t* __restrict__ out,
    float scale,
    float onorm_eps,
    float lower_bound,
    int64_t mixed_stride,
    int64_t a_stride,
    int64_t b_stride,
    int64_t onorm_stride,
    int64_t conv_slot_stride,
    int64_t conv_width_stride,
    int64_t state_slot_stride) {
  static_assert(kThreads == kKdaHipSmallBatchThreads || kThreads == kKdaHipLargeBatchThreads);
  const int batch_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int tid = threadIdx.x;
  const int wave = tid >> 6;
  const int wave_lane = tid & 63;
  constexpr int kWaves = kThreads / 64;
  constexpr int kRowsPerWave = kKdaHipDim / kWaves;
  constexpr int kRowPairsPerWave = kRowsPerWave / 8;
  const int slot = indices[batch_idx];
  const int out_base = (batch_idx * kKdaHipHeads + head_idx) * kKdaHipDim;

  if (slot < 0) {
    if (tid < kKdaHipDim) {
      out[out_base + tid] = device::cast<bf16_t>(0.0f);
    }
    return;
  }

  __shared__ float shared_q[kKdaHipDim];
  __shared__ float shared_k[kKdaHipDim];
  __shared__ float shared_decay[kKdaHipDim];
  __shared__ float shared_v[kKdaHipDim];
  __shared__ float shared_o[kKdaHipDim];
  __shared__ float shared_reduce[2 * kKdaHipActiveWarps];
  __shared__ float shared_beta;

  if constexpr (kThreads == kKdaHipSmallBatchThreads) {
    if (tid < 3 * kKdaHipDim) {
      const int channel = tid / kKdaHipDim;
      const int dim_idx = tid % kKdaHipDim;
      const int head_dim_idx = head_idx * kKdaHipDim + dim_idx;
      const int channel_offset = channel * kKdaHipSeg;
      const int64_t row_base = static_cast<int64_t>(batch_idx) * mixed_stride;
      const int64_t conv_base = static_cast<int64_t>(slot) * conv_slot_stride;
      const float* weight = channel == 0 ? w_q_t : (channel == 1 ? w_k_t : w_v_t);
      float* shared_output = channel == 0 ? shared_q : (channel == 1 ? shared_k : shared_v);

      float acc = conv_bias[channel_offset + head_dim_idx];
      bf16_t conv_state[3];
#pragma unroll
      for (int tap = 0; tap < 3; ++tap) {
        const int64_t state_base = conv_base + static_cast<int64_t>(tap) * conv_width_stride;
        conv_state[tap] = conv_states[state_base + channel_offset + head_dim_idx];
        acc += device::cast<float>(conv_state[tap]) * weight[tap * kKdaHipSeg + head_dim_idx];
      }

      const bf16_t new_value = mixed_qkv[row_base + channel_offset + head_dim_idx];
      acc += device::cast<float>(new_value) * weight[3 * kKdaHipSeg + head_dim_idx];

      conv_states[conv_base + channel_offset + head_dim_idx] = conv_state[1];
      conv_states[conv_base + conv_width_stride + channel_offset + head_dim_idx] = conv_state[2];
      conv_states[conv_base + 2 * conv_width_stride + channel_offset + head_dim_idx] = new_value;
      shared_output[dim_idx] = device::cast<float>(device::cast<bf16_t>(kda_hip_silu(acc)));
    }
  } else if (tid < kKdaHipDim) {
    const int dim_idx = tid;
    const int head_dim_idx = head_idx * kKdaHipDim + dim_idx;
    const int64_t row_base = static_cast<int64_t>(batch_idx) * mixed_stride;
    const int64_t conv_base = static_cast<int64_t>(slot) * conv_slot_stride;

    float q_acc = conv_bias[head_dim_idx];
    float k_acc = conv_bias[kKdaHipSeg + head_dim_idx];
    float v_acc = conv_bias[2 * kKdaHipSeg + head_dim_idx];
    bf16_t q_state[3];
    bf16_t k_state[3];
    bf16_t v_state[3];
#pragma unroll
    for (int tap = 0; tap < 3; ++tap) {
      const int64_t state_base = conv_base + static_cast<int64_t>(tap) * conv_width_stride;
      q_state[tap] = conv_states[state_base + head_dim_idx];
      k_state[tap] = conv_states[state_base + kKdaHipSeg + head_dim_idx];
      v_state[tap] = conv_states[state_base + 2 * kKdaHipSeg + head_dim_idx];
      q_acc += device::cast<float>(q_state[tap]) * w_q_t[tap * kKdaHipSeg + head_dim_idx];
      k_acc += device::cast<float>(k_state[tap]) * w_k_t[tap * kKdaHipSeg + head_dim_idx];
      v_acc += device::cast<float>(v_state[tap]) * w_v_t[tap * kKdaHipSeg + head_dim_idx];
    }

    const bf16_t q_new = mixed_qkv[row_base + head_dim_idx];
    const bf16_t k_new = mixed_qkv[row_base + kKdaHipSeg + head_dim_idx];
    const bf16_t v_new = mixed_qkv[row_base + 2 * kKdaHipSeg + head_dim_idx];
    q_acc += device::cast<float>(q_new) * w_q_t[3 * kKdaHipSeg + head_dim_idx];
    k_acc += device::cast<float>(k_new) * w_k_t[3 * kKdaHipSeg + head_dim_idx];
    v_acc += device::cast<float>(v_new) * w_v_t[3 * kKdaHipSeg + head_dim_idx];

    conv_states[conv_base + head_dim_idx] = q_state[1];
    conv_states[conv_base + conv_width_stride + head_dim_idx] = q_state[2];
    conv_states[conv_base + 2 * conv_width_stride + head_dim_idx] = q_new;
    conv_states[conv_base + kKdaHipSeg + head_dim_idx] = k_state[1];
    conv_states[conv_base + conv_width_stride + kKdaHipSeg + head_dim_idx] = k_state[2];
    conv_states[conv_base + 2 * conv_width_stride + kKdaHipSeg + head_dim_idx] = k_new;
    conv_states[conv_base + 2 * kKdaHipSeg + head_dim_idx] = v_state[1];
    conv_states[conv_base + conv_width_stride + 2 * kKdaHipSeg + head_dim_idx] = v_state[2];
    conv_states[conv_base + 2 * conv_width_stride + 2 * kKdaHipSeg + head_dim_idx] = v_new;

    shared_q[dim_idx] = device::cast<float>(device::cast<bf16_t>(kda_hip_silu(q_acc)));
    shared_k[dim_idx] = device::cast<float>(device::cast<bf16_t>(kda_hip_silu(k_acc)));
    shared_v[dim_idx] = device::cast<float>(device::cast<bf16_t>(kda_hip_silu(v_acc)));
  }
  if (tid < 64) {
    const int dim_base = tid * 2;
    const int head_dim_base = head_idx * kKdaHipDim + dim_base;
    const int64_t gate_base = static_cast<int64_t>(batch_idx) * a_stride + head_dim_base;
    const float exp_a = kda_hip_exp(A_log[head_idx]);
#pragma unroll
    for (int elem = 0; elem < 2; ++elem) {
      const float gate_x = device::cast<float>(a[gate_base + elem]) + dt_bias[head_dim_base + elem];
      if constexpr (kUseLowerBound) {
        shared_decay[dim_base + elem] = kda_hip_exp(lower_bound * kda_hip_sigmoid(exp_a * gate_x));
      } else {
        const float softplus = gate_x <= 20.0f ? logf(1.0f + kda_hip_exp(gate_x)) : gate_x;
        shared_decay[dim_base + elem] = kda_hip_exp(-exp_a * softplus);
      }
    }
  }
  if (tid == 0) {
    shared_beta = kda_hip_sigmoid(device::cast<float>(b[static_cast<int64_t>(batch_idx) * b_stride + head_idx]));
  }
  __syncthreads();

  if (tid < 64) {
    const int dim_base = wave_lane * 2;
    const float q_sum = shared_q[dim_base] * shared_q[dim_base] + shared_q[dim_base + 1] * shared_q[dim_base + 1];
    const float k_sum = shared_k[dim_base] * shared_k[dim_base] + shared_k[dim_base + 1] * shared_k[dim_base + 1];
    const float q_total = kda_hip_wave64_sum(q_sum);
    const float k_total = kda_hip_wave64_sum(k_sum);
    const float q_norm = __builtin_amdgcn_sqrtf(q_total + 1.0e-6f);
    const float k_norm = __builtin_amdgcn_sqrtf(k_total + 1.0e-6f);
#pragma unroll
    for (int elem = 0; elem < 2; ++elem) {
      shared_q[dim_base + elem] = shared_q[dim_base + elem] / q_norm * scale;
      shared_k[dim_base + elem] = shared_k[dim_base + elem] / k_norm;
    }
  }
  __syncthreads();

  const int lane_group = wave_lane >> 4;
  const int lane_in_group = wave_lane & 15;
  const int k_base = lane_in_group * 8;
  float q[8];
  float k[8];
  float decay[8];
#pragma unroll
  for (int elem = 0; elem < 8; ++elem) {
    q[elem] = shared_q[k_base + elem];
    k[elem] = shared_k[k_base + elem];
    decay[elem] = shared_decay[k_base + elem];
  }

  const int64_t state_head_base =
      static_cast<int64_t>(slot) * state_slot_stride + static_cast<int64_t>(head_idx) * kKdaHipDim * kKdaHipDim;
  for (int row_pair = 0; row_pair < kRowPairsPerWave; ++row_pair) {
    const int row0 = wave * kRowsPerWave + lane_group + row_pair * 8;
    const int row1 = row0 + 4;
    const int64_t state_base0 = state_head_base + static_cast<int64_t>(row0) * kKdaHipDim + k_base;
    const int64_t state_base1 = state_head_base + static_cast<int64_t>(row1) * kKdaHipDim + k_base;
    KdaHipFloat2 h[8];
#pragma unroll
    for (int elem = 0; elem < 8; ++elem) {
      h[elem] = {
          device::cast<float>(state[state_base0 + elem]) * decay[elem],
          device::cast<float>(state[state_base1 + elem]) * decay[elem],
      };
    }
    const KdaHipFloat2 local_hk = kda_hip_dot8_pair(h, k);
    const KdaHipFloat2 dot_hk = {
        kda_hip_subgroup16_sum(local_hk[0]),
        kda_hip_subgroup16_sum(local_hk[1]),
    };
    const KdaHipFloat2 value = {
        (shared_v[row0] - dot_hk[0]) * shared_beta,
        (shared_v[row1] - dot_hk[1]) * shared_beta,
    };
#pragma unroll
    for (int elem = 0; elem < 8; ++elem) {
      const KdaHipFloat2 packed_k = {k[elem], k[elem]};
      // Triton uses mul+add for row%8<4, k%8==7; preserve BF16 state bits.
      if (elem == 7) {
        h[elem] = {
            value[0] * k[elem] + h[elem][0],
            __builtin_elementwise_fma(value[1], k[elem], h[elem][1]),
        };
      } else {
        h[elem] = __builtin_elementwise_fma(value, packed_k, h[elem]);
      }
      state[state_base0 + elem] = device::cast<bf16_t>(h[elem][0]);
      state[state_base1 + elem] = device::cast<bf16_t>(h[elem][1]);
    }
    const KdaHipFloat2 local_hq = kda_hip_dot8_pair(h, q);
    const KdaHipFloat2 dot_hq = {
        kda_hip_subgroup16_sum(local_hq[0]),
        kda_hip_subgroup16_sum(local_hq[1]),
    };
    if (lane_in_group == 0) {
      shared_o[row0] = dot_hq[0];
      shared_o[row1] = dot_hq[1];
    }
  }
  __syncthreads();

  const float o_value = tid < kKdaHipDim ? shared_o[tid] : 0.0f;
  const float sumsq = kda_hip_block_sum_128(o_value * o_value, shared_reduce);
  if (tid < kKdaHipDim) {
    const int64_t gate_idx = static_cast<int64_t>(batch_idx) * onorm_stride + head_idx * kKdaHipDim + tid;
    const float rstd = rsqrtf(sumsq / static_cast<float>(kKdaHipDim) + onorm_eps);
    const float gate = device::math::sigmoid_fast(device::cast<float>(onorm_g[gate_idx]));
    out[out_base + tid] = device::cast<bf16_t>(shared_o[tid] * rstd * onorm_weight[tid] * gate);
  }
}

struct KdaFusedDecodeHipKernel {
  static void
  run(const tvm::ffi::TensorView mixed_qkv,
      const tvm::ffi::TensorView a,
      const tvm::ffi::TensorView b,
      const tvm::ffi::TensorView conv_states,
      const tvm::ffi::TensorView w_q_t,
      const tvm::ffi::TensorView w_k_t,
      const tvm::ffi::TensorView w_v_t,
      const tvm::ffi::TensorView conv_bias,
      const tvm::ffi::TensorView A_log,
      const tvm::ffi::TensorView dt_bias,
      const tvm::ffi::TensorView onorm_g,
      const tvm::ffi::TensorView onorm_weight,
      const tvm::ffi::TensorView state,
      const tvm::ffi::TensorView indices,
      const tvm::ffi::TensorView out,
      double scale,
      double onorm_eps,
      double lower_bound,
      bool use_lower_bound) {
    using namespace host;

    auto B_ = SymbolicSize{"batch"};
    auto Slots_ = SymbolicSize{"pool_slots"};
    auto device = SymbolicDevice{};
    device.set_options<kDLGPU>();

    TensorMatcher({B_, kKdaHipConvDim})
        .with_dtype<bf16_t>()
        .with_device(device)
        .with_strides({-1, 1})
        .verify(mixed_qkv);
    TensorMatcher({B_, kKdaHipSeg}).with_dtype<bf16_t>().with_device(device).with_strides({-1, 1}).verify(a);
    TensorMatcher({B_, kKdaHipHeads}).with_dtype<bf16_t>().with_device(device).with_strides({-1, 1}).verify(b);
    TensorMatcher({Slots_, 3, kKdaHipConvDim})
        .with_dtype<bf16_t>()
        .with_device(device)
        .with_strides({-1, -1, 1})
        .verify(conv_states);
    TensorMatcher({4, kKdaHipSeg}).with_dtype<fp32_t>().with_device(device).with_strides({kKdaHipSeg, 1}).verify(w_q_t);
    TensorMatcher({4, kKdaHipSeg}).with_dtype<fp32_t>().with_device(device).with_strides({kKdaHipSeg, 1}).verify(w_k_t);
    TensorMatcher({4, kKdaHipSeg}).with_dtype<fp32_t>().with_device(device).with_strides({kKdaHipSeg, 1}).verify(w_v_t);
    TensorMatcher({kKdaHipConvDim}).with_dtype<fp32_t>().with_device(device).with_strides({1}).verify(conv_bias);
    TensorMatcher({kKdaHipHeads}).with_dtype<fp32_t>().with_device(device).with_strides({1}).verify(A_log);
    TensorMatcher({kKdaHipSeg}).with_dtype<fp32_t>().with_device(device).with_strides({1}).verify(dt_bias);
    TensorMatcher({B_, kKdaHipSeg}).with_dtype<bf16_t>().with_device(device).with_strides({-1, 1}).verify(onorm_g);
    TensorMatcher({kKdaHipDim}).with_dtype<fp32_t>().with_device(device).with_strides({1}).verify(onorm_weight);
    TensorMatcher({Slots_, kKdaHipHeads, kKdaHipDim, kKdaHipDim})
        .with_dtype<bf16_t>()
        .with_device(device)
        .with_strides({-1, kKdaHipDim * kKdaHipDim, kKdaHipDim, 1})
        .verify(state);
    TensorMatcher({B_}).with_dtype<int32_t>().with_device(device).with_strides({1}).verify(indices);
    TensorMatcher({B_, kKdaHipSeg}).with_dtype<bf16_t>().with_device(device).with_strides({kKdaHipSeg, 1}).verify(out);

    const int batch = static_cast<int>(B_.unwrap());
    RuntimeCheck(batch > 0, "HIP KDA fused decode requires a positive batch size");

    const auto launch = [&](auto kernel, int threads) {
      LaunchKernel(dim3(batch, kKdaHipHeads), dim3(threads), device.unwrap())(
          kernel,
          static_cast<const bf16_t*>(mixed_qkv.data_ptr()),
          static_cast<const bf16_t*>(a.data_ptr()),
          static_cast<const bf16_t*>(b.data_ptr()),
          static_cast<bf16_t*>(conv_states.data_ptr()),
          static_cast<const fp32_t*>(w_q_t.data_ptr()),
          static_cast<const fp32_t*>(w_k_t.data_ptr()),
          static_cast<const fp32_t*>(w_v_t.data_ptr()),
          static_cast<const fp32_t*>(conv_bias.data_ptr()),
          static_cast<const fp32_t*>(A_log.data_ptr()),
          static_cast<const fp32_t*>(dt_bias.data_ptr()),
          static_cast<const bf16_t*>(onorm_g.data_ptr()),
          static_cast<const fp32_t*>(onorm_weight.data_ptr()),
          static_cast<bf16_t*>(state.data_ptr()),
          static_cast<const int32_t*>(indices.data_ptr()),
          static_cast<bf16_t*>(out.data_ptr()),
          static_cast<float>(scale),
          static_cast<float>(onorm_eps),
          static_cast<float>(lower_bound),
          mixed_qkv.stride(0),
          a.stride(0),
          b.stride(0),
          onorm_g.stride(0),
          conv_states.stride(0),
          conv_states.stride(1),
          state.stride(0));
    };

    if (batch <= kKdaHipSmallBatchMax) {
      const auto kernel = use_lower_bound ? kda_fused_decode_hip_kernel<kKdaHipSmallBatchThreads, true>
                                          : kda_fused_decode_hip_kernel<kKdaHipSmallBatchThreads, false>;
      launch(kernel, kKdaHipSmallBatchThreads);
    } else {
      const auto kernel = use_lower_bound ? kda_fused_decode_hip_kernel<kKdaHipLargeBatchThreads, true>
                                          : kda_fused_decode_hip_kernel<kKdaHipLargeBatchThreads, false>;
      launch(kernel, kKdaHipLargeBatchThreads);
    }
  }
};

}  // namespace sglang
