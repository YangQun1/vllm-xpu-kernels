#pragma once

#include <cstdint>
#include <type_traits>
#include <sycl/sycl.hpp>
#include <torch/all.h>

#include "gemm.hpp"
#include "gdn_attn_utils.h"
#include "csrc/utils.h"

namespace gdn {
namespace v2 {

using namespace cute;

static constexpr int sub_group_size = 16;
static constexpr int chunk_size = gdn::chunk_size_xe2;  // 64
static constexpr int MaxThreadsPerSM = 512;
static constexpr int MaxThreadsPerSM_grf128 = 1024;
using InverseType = half_t;
static_assert(
  std::is_same_v<InverseType, half_t> ||
    std::is_same_v<InverseType, tfloat32_t>,
  "InverseType must be half_t or tfloat32_t");

// ============================================================================
// V2 Prepare Kernel (design §2.9)
// Input:  raw a [num_v_heads, total_virtual_seqlen], dt_bias [num_v_heads],
//         A_log [num_v_heads]
// Output: alpha [num_v_heads, total_virtual_seqlen] in LINEAR space (per-token)
//         beta  [num_v_heads, total_virtual_seqlen] passthrough
//
// Computes per-token: gate_i = softplus(a_i + dt_bias) * (-exp(A_log))
//                     alpha_i = exp(gate_i)
// NO cumulative sum — the main V2 kernel does log2+prefix_sum internally.
// ============================================================================
CUTE_DEVICE float
act_softplus(float x, float beta = 1.0f, float threshold = 20.0f) {
  if (beta * x < threshold) {
    return sycl::log(1.0f + sycl::exp(beta * x)) / beta;
  } else {
    return x;
  }
}

template <typename T, typename StateT>
CUTE_DEVICE void chunk_prepare_v2_kernel(
    float* alpha_out,          // [num_v_heads, total_virtual_seqlen] output: linear space
    const float* a,            // [num_v_heads, total_virtual_seqlen] raw gate param
    const float* A_log,        // [num_v_heads]
    const T* dt_bias,          // [num_v_heads]
    const int* query_start_loc,
    const int total_virtual_seqlen,
    const int batch_size,
    const int num_v_heads) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int group_id = item.get_group(1);
  int group_range = item.get_group_range(1);
  auto sg = item.get_sub_group();
  int sg_id = sg.get_group_linear_id();
  int sg_range = sg.get_group_linear_range();
  int sg_local_id = sg.get_local_linear_id();

  int total_sg_range = group_range * sg_range;
  int total_sg_id = group_id * sg_range + sg_id;

  // Distribute work across (v_head, token) pairs
  const int tokens_per_sg = sub_group_size;  // each sg thread handles 1 token
  const int total_tokens = total_virtual_seqlen;
  const int token_chunks = (total_tokens + tokens_per_sg - 1) / tokens_per_sg;
  const int total_work = num_v_heads * token_chunks;

  for (int work_id = total_sg_id; work_id < total_work; work_id += total_sg_range) {
    int v_head_id = work_id / token_chunks;
    int token_chunk_id = work_id % token_chunks;
    int token_idx = token_chunk_id * tokens_per_sg + sg_local_id;

    if (token_idx >= total_tokens) continue;

    const float A_log_exp_h = -sycl::exp(A_log[v_head_id]);
    const float dt_bias_h = static_cast<float>(dt_bias[v_head_id]);

    float a_val = a[token_idx + v_head_id * total_virtual_seqlen];
    float gate = act_softplus(a_val + dt_bias_h) * A_log_exp_h;
    float alpha_val = sycl::exp(gate);  // linear space

    alpha_out[token_idx + v_head_id * total_virtual_seqlen] = alpha_val;
  }
}

template <typename T, typename StateT>
class ChunkPrepareV2Kernel;


// ============================================================================
// Per-stage MMA policies (independently tunable per design §2.3)
// ============================================================================

// QK: Q×K^T, chunk-local attention
// [C, Dk] x [C, Dk]^T → [C, C]  (64×64 output, 32 SGs)
struct MmaPolicy_QK {
  using WGTile = Shape<_64, _64, _32>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
};

// SK: S×K^T, state readout
// [Dv, Dk] x [C, Dk]^T → [Dv, C]  (128×64 output, 32 SGs, no dv loop)
struct MmaPolicy_SK {
  using WGTile = Shape<_128, _64, _32>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
};

// Ṽ = V^T-SK
// [C, Dv]^T - [Dv, C] → [Dv, C]

// NewV: Ṽ×T^T, apply inverse
// [Dv, C] x [C, C]^T → [Dv, C]  (128×64 output, 32 SGs, no dv loop)
struct MmaPolicy_NewV { 
  using WGTile = Shape<_128, _64, _32>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
};

// O1: QxS^T, inter-chunk output
// [C, Dk] x [Dv, Dk]^T → [C, Dv]  (64×128 output, 32 SGs, no dv loop)
struct MmaPolicy_O1 {
  using WGTile = Shape<_64, _128, _32>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
};

// O2: QKxNewV^T, intra-chunk output
// [C, C] x [Dv, C]^T → [C, Dv]  (64×128 output, 32 SGs, no dv loop)
struct MmaPolicy_O2 {
  using WGTile = Shape<_64, _128, _32>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
};

// KV: NewV×K → [Dv×Dk], state update
// [Dv, C] x [C, Dk] → [Dv, Dk]  (128×128 output, 32 SGs, no dv/dk loop)
struct MmaPolicy_KV {
  using WGTile = Shape<_128, _128, _32>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
};


// KK: K×K^T → [C×C], for inverse
struct MmaPolicy_KK {
  using WGTile = Shape<_64, _64, _32>;
  using SGLayout = Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>;
};

// Inverse: 16×16 block forward substitution
struct MmaPolicy_Inverse {
  using WGTile = Shape<_16, _16, _16>;
  using SGLayout = Layout<Shape<_1, _1, _1>, Stride<_1, _1, _0>>;
};

// Inverse 32×32: bottom-left super-block (T_31/T_32/T_41/T_42) solved as one
// 32×32×32 GEMM chain cooperatively by 4 sub-groups. SGLayout<4,1,1> splits M
// so each sub-group owns an 8×32 output stripe (fp32 accumulator 1 KB ⇒ no GRF
// spill). K is the full 32 ⇒ single k-tile ⇒ resident-A gemm_STS is valid.
struct MmaPolicy_Inverse32 {
  using WGTile = Shape<_32, _32, _32>;
  using SGLayout = Layout<Shape<_4, _1, _1>, Stride<_1, _0, _0>>;
};

// ============================================================================
// Main V2 kernel struct (design §2.10)
// ============================================================================
template <typename T, typename StateT = float>
struct chunk_gated_delta_rule_v2_kernel {
  // Type aliases
  using Element = T;
  using AccType = float;

  // MMA type definitions using CuTe XE_DPAS
  using op_type = XE_DPAS_TT<8, float, cutlass::platform::remove_cv_t<T>>;
  using op_type_inverse =
      XE_DPAS_TT<8, float, cutlass::platform::remove_cv_t<InverseType>>;

  // Per-stage TiledMMA types
  using MmaQK = typename TiledMMAHelper<
      MMA_Atom<op_type>,
      Layout<typename MmaPolicy_QK::WGTile>,
      typename MmaPolicy_QK::SGLayout>::TiledMMA;

  using MmaKK = typename TiledMMAHelper<
      MMA_Atom<op_type>,
      Layout<typename MmaPolicy_KK::WGTile>,
      typename MmaPolicy_KK::SGLayout>::TiledMMA;

  using MmaSK = typename TiledMMAHelper<
      MMA_Atom<op_type>,
      Layout<typename MmaPolicy_SK::WGTile>,
      typename MmaPolicy_SK::SGLayout>::TiledMMA;

  using MmaNewV = typename TiledMMAHelper<
      MMA_Atom<op_type>,
      Layout<typename MmaPolicy_NewV::WGTile>,
      typename MmaPolicy_NewV::SGLayout>::TiledMMA;

  using MmaO1 = typename TiledMMAHelper<
      MMA_Atom<op_type>,
      Layout<typename MmaPolicy_O1::WGTile>,
      typename MmaPolicy_O1::SGLayout>::TiledMMA;

  using MmaO2 = typename TiledMMAHelper<
      MMA_Atom<op_type>,
      Layout<typename MmaPolicy_O2::WGTile>,
      typename MmaPolicy_O2::SGLayout>::TiledMMA;

  using MmaKV = typename TiledMMAHelper<
      MMA_Atom<op_type>,
      Layout<typename MmaPolicy_KV::WGTile>,
      typename MmaPolicy_KV::SGLayout>::TiledMMA;

  using MmaInverse = typename TiledMMAHelper<
      MMA_Atom<op_type_inverse>,
      Layout<typename MmaPolicy_Inverse::WGTile>,
      typename MmaPolicy_Inverse::SGLayout>::TiledMMA;

  using MmaInverse32 = typename TiledMMAHelper<
      MMA_Atom<op_type_inverse>,
      Layout<typename MmaPolicy_Inverse32::WGTile>,
      typename MmaPolicy_Inverse32::SGLayout>::TiledMMA;

  // ============================================================================
  // Parameters
  // ============================================================================
  T* core_attn_out;        // [total_seqlen, num_v_heads, head_v_dim]
  const T* q;              // [total_virtual_seqlen, num_k_heads, head_k_dim]
  const T* k;              // [total_virtual_seqlen, num_k_heads, head_k_dim]
  const T* v;              // [total_virtual_seqlen, num_v_heads, head_v_dim]
  const float* alpha;      // [num_v_heads, total_virtual_seqlen] (linear space)
  const float* beta;       // [num_v_heads, total_virtual_seqlen]
  StateT* ssm_state;       // [cache_batch_size, num_v_heads, head_v_dim, head_k_dim]
  int ssm_state_stride_0;
  const int* query_start_loc;   // [batch_size + 1]
  const int* cache_indices;     // [batch_size]
  const bool* has_initial_state; // [batch_size] or nullptr
  const int* token_indx;        // [total_seqlen] or nullptr
  // Global scratch buffers
  T* qk_buf;              // [num_v_heads, total_virtual_seqlen, chunk_size]
  InverseType* l_buf;   // [num_v_heads, total_virtual_seqlen, chunk_size] inverse workspace
  T* t_buf;               // [num_v_heads, total_virtual_seqlen, chunk_size] T_out buffer (Element)
  T* sk_buf;              // [num_v_heads, total_virtual_seqlen, head_v_dim] or reuse
  T* newv_buf;            // [num_v_heads, total_virtual_seqlen, head_v_dim] or reuse
  float* cumsum_log_buf;  // [num_v_heads, total_virtual_seqlen] alpha cumsum_log channel
  float* cumprod_buf;     // [num_v_heads, total_virtual_seqlen] alpha cumprod channel

  int batch_size;
  int total_seqlen;
  int total_virtual_seqlen;
  int num_k_heads;
  int head_k_dim;
  int num_v_heads;
  int head_v_dim;

  // ============================================================================
  // SLM layout
  // - alpha channels: cumsum_log, cumprod
  // - inverse scratch: load/save buffers for SLM inverse path
  // ============================================================================
  static constexpr int slm_alpha_size = chunk_size * 2;
  static constexpr int slm_cumsum_log_offset = 0;
  static constexpr int slm_cumprod_offset = chunk_size;
  static constexpr int slm_inverse_load_offset = slm_alpha_size;
  static constexpr int slm_inverse_save_offset =
      slm_inverse_load_offset + chunk_size * chunk_size;
  static constexpr int slm_inverse_size = chunk_size * chunk_size * 2;
  static constexpr int slm_total_size = slm_alpha_size + slm_inverse_size;

  // ============================================================================
  // Alpha 3-channel preprocessing (design §2.7 item 2)
  // Computes cumsum_log, cumprod from alpha (linear space) into SLM AND persists
  // both channels to global scratch (cumsum_log_buf / cumprod_buf) in the same
  // pass, so the recurrent back kernel can reload them via load_alpha_channels
  // instead of recomputing the log2/scan/exp2 (design: Option A). Folding the
  // store here avoids a second SLM-read pass over the freshly written channels.
  // ============================================================================
  CUTE_DEVICE void compute_alpha_channels(
      float* slm_ptr,
      const float* alpha_chunk_ptr,
      int current_chunk_size,
      int v_head_id,
      int chunk_offset,
      sycl::nd_item<3>& item) const {
    auto sg = item.get_sub_group();
    int sg_local_id = sg.get_local_linear_id();
    int local_id = item.get_local_linear_id();
    int local_range = item.get_local_range(2);

    float* cumsum_log_ptr = slm_ptr + slm_cumsum_log_offset;
    float* cumprod_ptr = slm_ptr + slm_cumprod_offset;
    float* g_cumsum_log = cumsum_log_buf +
        static_cast<int64_t>(v_head_id) * total_virtual_seqlen + chunk_offset;
    float* g_cumprod = cumprod_buf +
        static_cast<int64_t>(v_head_id) * total_virtual_seqlen + chunk_offset;
    // Each sub-group element handles chunk_size/sub_group_size elements
    constexpr int local_num = chunk_size / sub_group_size;
    float log_local[local_num] = {};
    float log_local_sum = 0.0f;

    // Load alpha and convert to log2 space
    CUTE_UNROLL
    for (int c = 0; c < local_num; ++c) {
      int idx = sg_local_id * local_num + c;
      float alpha_val = (idx < current_chunk_size) ? alpha_chunk_ptr[idx] : 1.0f;
      // Clamp alpha to avoid log of zero/negative
      alpha_val = sycl::fmax(alpha_val, 1e-10f);
      log_local[c] = sycl::log2(alpha_val);
      log_local_sum += log_local[c];
    }

    // Inclusive prefix sum across sub-group
    log_local_sum =
        sycl::inclusive_scan_over_group(sg, log_local_sum, sycl::plus<float>());

    // Write cumsum_log (prefix sum of log2(alpha))
    CUTE_UNROLL
    for (int c = local_num - 1; c >= 0; --c) {
      int idx = sg_local_id * local_num + c;
      cumsum_log_ptr[idx] = log_local_sum;
      log_local_sum -= log_local[c];
    }

    item.barrier(sycl::access::fence_space::local_space);

    // Compute cumprod = exp2(cumsum_log) and persist the valid (non-padding)
    // channels to global. Padding elements (e >= current_chunk_size) are written
    // to global in the next loop instead, so each global slot is written once.
    CUTE_UNROLL
    for (int e = local_id; e < chunk_size; e += local_range) {
      float cs_log = cumsum_log_ptr[e];
      float cp = sycl::exp2(cs_log);
      cumprod_ptr[e] = cp;
      if (e < current_chunk_size) {
        g_cumsum_log[e] = cs_log;
        g_cumprod[e] = cp;
      }
    }

    // Zero out elements beyond current_chunk_size (SLM + global).
    CUTE_UNROLL
    for (int e = current_chunk_size + local_id; e < chunk_size; e += local_range) {
      float cs_pad = cumsum_log_ptr[current_chunk_size - 1];
      cumsum_log_ptr[e] = cs_pad;
      cumprod_ptr[e] = 0.0f;
      g_cumsum_log[e] = cs_pad;
      g_cumprod[e] = 0.0f;
    }

    item.barrier(sycl::access::fence_space::local_space);
  }

  // ============================================================================
  // Reload the alpha channels (cumsum_log, cumprod) from global scratch into
  // SLM. Replaces the redundant compute_alpha_channels call in the recurrent
  // back kernel; the channels were already produced by the front kernel.
  // Ends with a local barrier so the loaded SLM is visible work-group-wide.
  // ============================================================================
  CUTE_DEVICE void load_alpha_channels(
      float* slm_ptr,
      int v_head_id,
      int chunk_offset,
      sycl::nd_item<3>& item) const {
    int local_id = item.get_local_linear_id();
    int local_range = item.get_local_range(2);
    float* cumsum_log_ptr = slm_ptr + slm_cumsum_log_offset;
    float* cumprod_ptr = slm_ptr + slm_cumprod_offset;
    const float* g_cumsum_log = cumsum_log_buf +
        static_cast<int64_t>(v_head_id) * total_virtual_seqlen + chunk_offset;
    const float* g_cumprod = cumprod_buf +
        static_cast<int64_t>(v_head_id) * total_virtual_seqlen + chunk_offset;
    CUTE_UNROLL
    for (int e = local_id; e < chunk_size; e += local_range) {
      cumsum_log_ptr[e] = g_cumsum_log[e];
      cumprod_ptr[e] = g_cumprod[e];
    }
    item.barrier(sycl::access::fence_space::local_space);
  }

# if 1 // Not used in the current V2 path, keep for reference/debug only
  // ============================================================================
  // Inverse algorithm: 16×16 block forward substitution (design §2.5)
  // T_out = L^{-1} · diag(β), where L is a unit-lower matrix (I+strict_lower)
  // Algorithm: V1's block back-substitution with beta fused at the end.
  // NOTE(yangqun): This legacy path shows random precision instability in unit
  // tests under the V2 pipeline. Keep it for reference/debug only; do not use
  // as the default inverse path until the numerical issue is fixed.
  // ============================================================================
  CUTE_DEVICE void compute_inverse(
      T* T_out_ptr,                 // output [chunk_size × chunk_size], Element
      InverseType* L_ptr,           // in-place [chunk_size × chunk_size]: L -> T_pure
      const float* beta_chunk_ptr,
      sycl::nd_item<3>& item) const {
    int local_id = item.get_local_linear_id();
    int local_range = item.get_local_range(2);
    auto sg = item.get_sub_group();
    int sg_id = sg.get_group_linear_id();
    int sg_local_id = sg.get_local_linear_id();

    MmaInverse mma_inv{};

    // ===== Step 1: 4 diagonal 16×16 blocks: forward substitution (pure inverse, no beta) =====
    // Distribute the 4 independent diagonal blocks across the first 4 sub-groups
    // (sg_id 0..3). Previously every sub-group recomputed all 4 blocks (8x
    // redundant work + write races on the same global addresses); now each
    // block is computed exactly once by its owning sub-group.
    if (sg_id < 4) {
      int i = sg_id;
      int offset = i * 16;
      InverseType* A_ptr_xx = L_ptr + offset * chunk_size + offset;
      float A_local[16];
      float A_sum;

      CUTE_UNROLL
      for (int e = 0; e < sg_local_id + 1; ++e) {
        A_local[e] = 0.0f;
      }

      InverseType A_load[16];
      CUTE_UNROLL
      for (int e = 0; e < sg_local_id; ++e) {
        A_load[e] = A_ptr_xx[sg_local_id * chunk_size + e];
      }

      // Forward substitution
      CUTE_UNROLL
      for (int mm_idx = 1; mm_idx < 16; ++mm_idx) {
        CUTE_UNROLL
        for (int nn_idx = 0; nn_idx < mm_idx; ++nn_idx) {
          float send_value = static_cast<float>(A_load[nn_idx]);
          float receive_value = sycl::group_broadcast(sg, send_value, mm_idx);
          if (sg_local_id == nn_idx) {
            A_local[mm_idx] = receive_value;
          }
        }
      }

      CUTE_UNROLL
      for (int mm_idx = 1; mm_idx < 16; ++mm_idx) {
        A_sum = 0.0f;
        float A_other[16];
        CUTE_UNROLL
        for (int e = 1; e < mm_idx + 1; ++e) {
          A_other[e] = sycl::group_broadcast(sg, A_local[mm_idx], e);
        }

        CUTE_UNROLL
        for (int e = 1; e < mm_idx + 1; ++e) {
          A_sum += A_local[e] * A_other[e];
        }

        A_local[mm_idx] = -A_local[mm_idx] - A_sum;
      }

      // Write PURE inverse (no beta) — matching V1 exactly
      InverseType* T_ptr_xx = L_ptr + offset * chunk_size + offset;
      CUTE_UNROLL
      for (int e = 0; e < 16; ++e) {
        if (e == sg_local_id) {
          T_ptr_xx[e * chunk_size + sg_local_id] = static_cast<InverseType>(1.0f);
        } else if (e > sg_local_id) {
          T_ptr_xx[e * chunk_size + sg_local_id] =
              static_cast<InverseType>(A_local[e]);
        } else {
          T_ptr_xx[e * chunk_size + sg_local_id] = static_cast<InverseType>(0.0f);
        }
      }
    }

    item.barrier(sycl::access::fence_space::global_and_local);

    // ===== Step 2: 6 off-diagonal block GEMMs (V1's back-substitution formula) =====
    // Deps Chain 1:
    //    T_21 = -(T_22 × KK_21 × T_11)
    //    T_31 = -T_33 × (KK_31 × T_11 + KK_32 × T_21)
    //    T_41 = -T_44 × (KK_41 × T_11 + KK_42 × T_21 + KK_43 × T_31)
    // Deps Chain 2:
    //    T_32 = -(T_33 × KK_32 × T_22)
    //    T_42 = -T_44 × (KK_42 × T_22 + KK_43 × T_32)
    // Deps Chain 3:
    //    T_43 = -(T_44 × KK_43 × T_33)

    auto A_XX_tensor_shape = make_shape(16, 16);

    // Step 2 uses subgroup-relative slicing (sg_local_id, 0..15) instead of the
    // global local_id (0..127), so that each owning sub-group partitions the
    // 16x16 MMA/copy correctly. For now the whole of Step 2 is still executed
    // only by sub-group 0 (if (sg_id == 0)); the workgroup barriers between
    // blocks stay at workgroup scope so all sub-groups participate.
    auto thr_mma_inv = mma_inv.get_slice(sg_local_id);
    auto wg_tile_inv = mma_inv.tile_mnk();

    // Identity tensor + partition for C accumulator and copy-D
    Tensor cC_inv = make_identity_tensor(A_XX_tensor_shape);
    Tensor gC_inv = local_tile(cC_inv, wg_tile_inv, make_coord(0, 0, 0), Step<_1, _1, X>{});
    auto tCrC_inv = thr_mma_inv.partition_sg_fragment_C(gC_inv);
    // Second C accumulator: holds the left-multiplied result (D x inner) for
    // multi-term blocks, computed entirely in registers (no global round-trip).
    auto tCrC_acc = thr_mma_inv.partition_sg_fragment_C(gC_inv);

    // A partition for register-to-register transfer (for gemm_STS)
    Tensor cA_inv = make_identity_tensor(A_XX_tensor_shape);
    Tensor gA_inv = local_tile(cA_inv, select<0, 2>(wg_tile_inv), make_coord(0, _));
    auto tCrA_inv = thr_mma_inv.partition_sg_fragment_A(gA_inv(_, _, 0));
    // B partition: multi-term blocks reorder inner^T (C-fragment) into a B
    // operand so gemm_TSS_sg can left-multiply by the diagonal block.
    Tensor gB_inv = local_tile(cA_inv, select<1, 2>(wg_tile_inv), make_coord(0, _));
    auto tCrB_inv = thr_mma_inv.partition_sg_fragment_B(gB_inv(_, _, 0));

    // Sub-block pointers
    auto T_ptr_11 = L_ptr;
    auto T_ptr_21 = L_ptr + 16 * chunk_size;
    auto T_ptr_22 = L_ptr + 16 * chunk_size + 16;
    auto T_ptr_31 = L_ptr + 32 * chunk_size;
    auto T_ptr_32 = L_ptr + 32 * chunk_size + 16;
    auto T_ptr_33 = L_ptr + 32 * chunk_size + 32;
    auto T_ptr_41 = L_ptr + 48 * chunk_size;
    auto T_ptr_42 = L_ptr + 48 * chunk_size + 16;
    auto T_ptr_43 = L_ptr + 48 * chunk_size + 32;
    auto T_ptr_44 = L_ptr + 48 * chunk_size + 48;

    // L sub-block pointers (off-diagonal blocks are same as strict-lower part)
    auto L_ptr_21 = L_ptr + 16 * chunk_size;
    auto L_ptr_31 = L_ptr + 32 * chunk_size;
    auto L_ptr_32 = L_ptr + 32 * chunk_size + 16;
    auto L_ptr_41 = L_ptr + 48 * chunk_size;
    auto L_ptr_42 = L_ptr + 48 * chunk_size + 16;
    auto L_ptr_43 = L_ptr + 48 * chunk_size + 32;

    // Diagonal block tensors (row-major for A operand of left-multiplication)
    auto T_22_tensor = make_tensor(make_gmem_ptr(T_ptr_22),
        make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
    auto T_33_tensor = make_tensor(make_gmem_ptr(T_ptr_33),
        make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
    auto T_44_tensor = make_tensor(make_gmem_ptr(T_ptr_44),
        make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));

    // Transposed tensors for B operand
    auto T_11_tensor_T = make_tensor(make_gmem_ptr(T_ptr_11),
        make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
    auto T_21_tensor_T = make_tensor(make_gmem_ptr(T_ptr_21),
        make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
    auto T_22_tensor_T = make_tensor(make_gmem_ptr(T_ptr_22),
        make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
    auto T_31_tensor_T = make_tensor(make_gmem_ptr(T_ptr_31),
        make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
    auto T_32_tensor_T = make_tensor(make_gmem_ptr(T_ptr_32),
        make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
    auto T_33_tensor_T = make_tensor(make_gmem_ptr(T_ptr_33),
        make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));

    // Row-major tensors for output (copy_D targets)
    auto T_21_tensor = make_tensor(make_gmem_ptr(T_ptr_21),
        make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
    auto T_31_tensor = make_tensor(make_gmem_ptr(T_ptr_31),
        make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
    auto T_32_tensor = make_tensor(make_gmem_ptr(T_ptr_32),
        make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
    auto T_41_tensor = make_tensor(make_gmem_ptr(T_ptr_41),
        make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
    auto T_42_tensor = make_tensor(make_gmem_ptr(T_ptr_42),
        make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
    auto T_43_tensor = make_tensor(make_gmem_ptr(T_ptr_43),
        make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));

    // Transposed T_41/T_42 (read back the stored inner as B operand)
    auto T_41_tensor_T = make_tensor(make_gmem_ptr(T_ptr_41),
        make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
    auto T_42_tensor_T = make_tensor(make_gmem_ptr(T_ptr_42),
        make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));

    // KK row-major tensors (A operand of the inner KK_a×T_b products)
    auto KK_31_tensor = make_tensor(make_gmem_ptr(L_ptr_31),
        make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
    auto KK_32_tensor = make_tensor(make_gmem_ptr(L_ptr_32),
        make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
    auto KK_41_tensor = make_tensor(make_gmem_ptr(L_ptr_41),
        make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
    auto KK_42_tensor = make_tensor(make_gmem_ptr(L_ptr_42),
        make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
    auto KK_43_tensor = make_tensor(make_gmem_ptr(L_ptr_43),
        make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));

    // KK transposed tensors (B operand for the single-term blocks)
    auto KK_21_tensor_T = make_tensor(make_gmem_ptr(L_ptr_21),
        make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
    auto KK_32_tensor_T = make_tensor(make_gmem_ptr(L_ptr_32),
        make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
    auto KK_43_tensor_T = make_tensor(make_gmem_ptr(L_ptr_43),
        make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));

    // ---- Chain-bound solve: one dependency chain pinned per sub-group ----
    // Every cross-level handoff inside a chain (T_21->T_31->T_41 on sg0,
    // T_32->T_42 on sg1) is produced and consumed by the SAME sub-group, so it
    // only needs a sub-group barrier (the producing/consuming lanes are all in
    // that sub-group), NOT a work-group barrier. The three chains therefore run
    // fully concurrently with no inter-level rendezvous; a single work-group
    // barrier afterwards makes every block visible work-group-wide before
    // Step 3. Numerically identical to the per-level version (same GEMM and
    // truncation sequence); only the synchronization is relaxed.
    //   chain A (sg0): T_21 -> T_31 -> T_41
    //   chain B (sg1): T_32 -> T_42
    //   chain C (sg2): T_43
    if (sg_id == 0) {
      // ----- T_21 = -(T_22 × KK_21 × T_11) -----
      auto copy_D_21 = get_block_2d_copy_D<void>(mma_inv, T_21_tensor);
      auto thr_copy_D_21 = copy_D_21.get_slice(sg_local_id);
      auto tCrD_21 = thr_copy_D_21.partition_sg_fragment_S(gC_inv);
      auto tCgD_21 = thr_copy_D_21.partition_D(gC_inv);
      clear(tCrC_inv);
      gemm_TTS_sg(T_22_tensor, KK_21_tensor_T, tCrC_inv, 0, 0, mma_inv);
      reorder(tCrC_inv, tCrA_inv);
      clear(tCrC_inv);
      gemm_STS_sg(tCrA_inv, T_11_tensor_T, tCrC_inv, 0, 0, mma_inv);
      CUTE_UNROLL
      for (int i = 0; i < tCrC_inv.size(); ++i) {
        tCrC_inv(i) *= -1.0f;
      }
      reorder(tCrC_inv, tCrD_21);
      copy(copy_D_21, tCrD_21, tCgD_21);

      // T_21 just written to global; make it visible to this sub-group's lanes
      // before T_31/T_41 read it back transposed (intra-sub-group handoff).
      sycl::group_barrier(sg);

      // ----- T_31 = -T_33 × (KK_31×T_11 + KK_32×T_21) -----
      auto copy_D_31 = get_block_2d_copy_D<void>(mma_inv, T_31_tensor);
      auto thr_copy_D_31 = copy_D_31.get_slice(sg_local_id);
      auto tCrD_31 = thr_copy_D_31.partition_sg_fragment_S(gC_inv);
      auto tCgD_31 = thr_copy_D_31.partition_D(gC_inv);
      // step 1: inner^T = T_11^T×KK_31^T + T_21^T×KK_32^T (fp32 accumulate).
      //   gemm_TTS_sg(A,B): C[m,n] = Σ_k A[m,k]·B[n,k]. With A = T_xx^T and
      //   B = KK_xx, this yields the inner-product transpose accumulated in C.
      clear(tCrC_inv);
      gemm_TTS_sg(T_11_tensor_T, KK_31_tensor, tCrC_inv, 0, 0, mma_inv);
      gemm_TTS_sg(T_21_tensor_T, KK_32_tensor, tCrC_inv, 0, 0, mma_inv);
      reorder(tCrC_inv, tCrB_inv);
      // step 2: T_31 = -T_33 × inner = -T_33 × (inner^T)^T, fully in registers.
      clear(tCrC_acc);
      gemm_TSS_sg(T_33_tensor, tCrB_inv, tCrC_acc, 0, 0, mma_inv);
      CUTE_UNROLL
      for (int i = 0; i < tCrC_acc.size(); ++i) {
        tCrC_acc(i) *= -1.0f;
      }
      reorder(tCrC_acc, tCrD_31);
      copy(copy_D_31, tCrD_31, tCgD_31);

      // T_31 written; make it visible before T_41 reads it back transposed.
      sycl::group_barrier(sg);

      // ----- T_41 = -T_44 × (KK_41×T_11 + KK_42×T_21 + KK_43×T_31) -----
      auto copy_D_41 = get_block_2d_copy_D<void>(mma_inv, T_41_tensor);
      auto thr_copy_D_41 = copy_D_41.get_slice(sg_local_id);
      auto tCrD_41 = thr_copy_D_41.partition_sg_fragment_S(gC_inv);
      auto tCgD_41 = thr_copy_D_41.partition_D(gC_inv);
      // step 1: inner^T = T_11^T×KK_41^T + T_21^T×KK_42^T + T_31^T×KK_43^T
      clear(tCrC_inv);
      gemm_TTS_sg(T_11_tensor_T, KK_41_tensor, tCrC_inv, 0, 0, mma_inv);
      gemm_TTS_sg(T_21_tensor_T, KK_42_tensor, tCrC_inv, 0, 0, mma_inv);
      gemm_TTS_sg(T_31_tensor_T, KK_43_tensor, tCrC_inv, 0, 0, mma_inv);
      reorder(tCrC_inv, tCrB_inv);
      // step 2: T_41 = -T_44 × inner, fully in registers.
      clear(tCrC_acc);
      gemm_TSS_sg(T_44_tensor, tCrB_inv, tCrC_acc, 0, 0, mma_inv);
      CUTE_UNROLL
      for (int i = 0; i < tCrC_acc.size(); ++i) {
        tCrC_acc(i) *= -1.0f;
      }
      reorder(tCrC_acc, tCrD_41);
      copy(copy_D_41, tCrD_41, tCgD_41);
    } else if (sg_id == 1) {
      // ----- T_32 = -(T_33 × KK_32 × T_22) -----
      auto copy_D_32 = get_block_2d_copy_D<void>(mma_inv, T_32_tensor);
      auto thr_copy_D_32 = copy_D_32.get_slice(sg_local_id);
      auto tCrD_32 = thr_copy_D_32.partition_sg_fragment_S(gC_inv);
      auto tCgD_32 = thr_copy_D_32.partition_D(gC_inv);
      clear(tCrC_inv);
      gemm_TTS_sg(T_33_tensor, KK_32_tensor_T, tCrC_inv, 0, 0, mma_inv);
      reorder(tCrC_inv, tCrA_inv);
      clear(tCrC_inv);
      gemm_STS_sg(tCrA_inv, T_22_tensor_T, tCrC_inv, 0, 0, mma_inv);
      CUTE_UNROLL
      for (int i = 0; i < tCrC_inv.size(); ++i) {
        tCrC_inv(i) *= -1.0f;
      }
      reorder(tCrC_inv, tCrD_32);
      copy(copy_D_32, tCrD_32, tCgD_32);

      // T_32 written; make it visible before T_42 reads it back transposed.
      sycl::group_barrier(sg);

      // ----- T_42 = -T_44 × (KK_42×T_22 + KK_43×T_32) -----
      auto copy_D_42 = get_block_2d_copy_D<void>(mma_inv, T_42_tensor);
      auto thr_copy_D_42 = copy_D_42.get_slice(sg_local_id);
      auto tCrD_42 = thr_copy_D_42.partition_sg_fragment_S(gC_inv);
      auto tCgD_42 = thr_copy_D_42.partition_D(gC_inv);
      // step 1: inner^T = T_22^T×KK_42^T + T_32^T×KK_43^T (fp32 accumulate)
      clear(tCrC_inv);
      gemm_TTS_sg(T_22_tensor_T, KK_42_tensor, tCrC_inv, 0, 0, mma_inv);
      gemm_TTS_sg(T_32_tensor_T, KK_43_tensor, tCrC_inv, 0, 0, mma_inv);
      reorder(tCrC_inv, tCrB_inv);
      // step 2: T_42 = -T_44 × inner, fully in registers.
      clear(tCrC_acc);
      gemm_TSS_sg(T_44_tensor, tCrB_inv, tCrC_acc, 0, 0, mma_inv);
      CUTE_UNROLL
      for (int i = 0; i < tCrC_acc.size(); ++i) {
        tCrC_acc(i) *= -1.0f;
      }
      reorder(tCrC_acc, tCrD_42);
      copy(copy_D_42, tCrD_42, tCgD_42);
    } else if (sg_id == 2) {
      // ----- T_43 = -(T_44 × KK_43 × T_33) -----
      auto copy_D_43 = get_block_2d_copy_D<void>(mma_inv, T_43_tensor);
      auto thr_copy_D_43 = copy_D_43.get_slice(sg_local_id);
      auto tCrD_43 = thr_copy_D_43.partition_sg_fragment_S(gC_inv);
      auto tCgD_43 = thr_copy_D_43.partition_D(gC_inv);
      clear(tCrC_inv);
      gemm_TTS_sg(T_44_tensor, KK_43_tensor_T, tCrC_inv, 0, 0, mma_inv);
      reorder(tCrC_inv, tCrA_inv);
      clear(tCrC_inv);
      gemm_STS_sg(tCrA_inv, T_33_tensor_T, tCrC_inv, 0, 0, mma_inv);
      CUTE_UNROLL
      for (int i = 0; i < tCrC_inv.size(); ++i) {
        tCrC_inv(i) *= -1.0f;
      }
      reorder(tCrC_inv, tCrD_43);
      copy(copy_D_43, tCrD_43, tCgD_43);
    }

    item.barrier(sycl::access::fence_space::global_and_local);

    // ===== Step 3: Fuse beta and cast to Element =====
    // T_out[row, col] = cast<Element>(T_pure[row, col] × β[col])
    for (int col = local_id; col < chunk_size; col += local_range) {
      float bv = beta_chunk_ptr[col];
      CUTE_UNROLL
      for (int row = 0; row < chunk_size; ++row) {
        T_out_ptr[row * chunk_size + col] =
            static_cast<T>(static_cast<float>(L_ptr[row * chunk_size + col]) * bv);
      }
    }

    item.barrier(sycl::access::fence_space::global_and_local);
  }
# endif

  // ============================================================================
  // Inverse algorithm (SLM path): migrated from V1 chunk_inverse_kernel style.
  // Keeps current compute_inverse untouched for comparison/debug purposes.
  // T_out = L^{-1} · diag(β), where L is unit-lower (I+strict_lower).
  // ============================================================================
  CUTE_DEVICE void compute_inverse_slm(
      T* T_out_ptr,                 // output [chunk_size × chunk_size], Element
      InverseType* L_ptr,           // input [chunk_size × chunk_size]: unit-lower L source
      const float* beta_chunk_ptr,
      float* slm_ptr,
      sycl::nd_item<3>& item) const {
    int local_id = item.get_local_linear_id();
    int local_range = item.get_local_range(2);

    auto sg = item.get_sub_group();
    int sg_id = sg.get_group_linear_id();
    int sg_range = sg.get_group_linear_range();
    int sg_local_id = sg.get_local_linear_id();

    float* A_ptr_load = slm_ptr + slm_inverse_load_offset;
    float* A_ptr_save = slm_ptr + slm_inverse_save_offset;

    // Load strict-lower part of L into SLM (float), and initialize diag of inverse to 1.
    CUTE_UNROLL
    for (int m_idx = sg_id; m_idx < chunk_size; m_idx += sg_range) {
      CUTE_UNROLL
      for (int n_idx = sg_local_id; n_idx < m_idx; n_idx += sub_group_size) {
        A_ptr_load[m_idx * chunk_size + n_idx] =
            static_cast<float>(L_ptr[m_idx * chunk_size + n_idx]);
      }
    }

    CUTE_UNROLL
    for (int idx = local_id; idx < chunk_size; idx += local_range) {
      A_ptr_save[idx * chunk_size + idx] = 1.0f;
    }

    item.barrier(sycl::access::fence_space::local_space);

    // Forward substitution for lower-triangular inverse.
    CUTE_UNROLL
    for (int n_idx = local_id; n_idx < chunk_size; n_idx += local_range) {
      CUTE_UNROLL
      for (int m_idx = n_idx + 1; m_idx < chunk_size; ++m_idx) {
        float sum = A_ptr_load[m_idx * chunk_size + n_idx];
        CUTE_UNROLL
        for (int loop_idx = n_idx + 1; loop_idx < m_idx; ++loop_idx) {
          sum += A_ptr_save[loop_idx * chunk_size + n_idx] *
                 A_ptr_load[m_idx * chunk_size + loop_idx];
        }
        A_ptr_save[m_idx * chunk_size + n_idx] = -sum;
      }
    }

    item.barrier(sycl::access::fence_space::local_space);

    // Fuse beta and cast to Element output.
    for (int col = local_id; col < chunk_size; col += local_range) {
      float bv = beta_chunk_ptr[col];
      CUTE_UNROLL
      for (int row = 0; row < chunk_size; ++row) {
        T_out_ptr[row * chunk_size + col] =
            static_cast<T>(
                static_cast<float>(A_ptr_save[row * chunk_size + col]) * bv);
      }
    }
  }

  // ============================================================================
  // Compute loop body: recurrent (batch, head) kernel, processes one chunk.
  //
  // Notation: C = chunk_size (64), Dk = head_k_dim (128), Dv = head_v_dim (128)
  //
  // Data flow per chunk:
  //   Stage 1: Load cumsum_log / cumprod from global scratch → SLM
  //   Stage 2+4: QK [C,C] = Q [C,Dk] × K [C,Dk]^T, fused causal mask + gate
  //             SK [Dv,C] = S [Dv,Dk] × K [C,Dk]^T · diag(cumprod)
  //             (K loaded once, shared via gemm_TTS_fused_2A)
  //   ── barrier ──
  //   Stage 5: NewV [Dv,C] = (V^T [Dv,C] - SK [Dv,C]) × T [C,C]^T
  //   ── barrier ──
  //   Stage 6: O [C,Dv] = diag(cumprod) · Q [C,Dk] × S [Dv,Dk]^T
  //                      + QK [C,C] × NewV [Dv,C]^T
  //   ── barrier ──
  //   Stage 7: S' [Dv,Dk] = e^(g_last) · S + decay(NewV) [Dv,C] × K^T [Dk,C]^T
  //
  // Stages 4-7 are recurrent: they read/write S (ssm_state) sequentially.
  // Stages 4,5,6 tile along Dv (for dv in 0..Dv/C-1), fully independent.
  // Stage 7 tiles along both Dv and Dk.
  // ============================================================================
  template <typename IsFirstBlock>
  CUTE_DEVICE void compute_loop_body(
      int chunk_id,
      int chunk_offset,        // global chunk start in virtual seqlen
      int out_chunk_offset,    // output chunk start in actual seqlen
      int current_chunk_size,  // actual tokens in this chunk
      int v_head_id,
      int kv_head_id,
      IsFirstBlock is_first_block,
      bool has_prev_state,
      float* slm_ptr,
      StateT* ssm_state_ptr,
      sycl::nd_item<3>& item) const {
    int local_id = item.get_local_linear_id();
    int local_range = item.get_local_range(2);
    auto sg = item.get_sub_group();
    int sg_local_id = sg.get_local_linear_id();

    const int kv_ratio = num_v_heads / num_k_heads;

    float* cumsum_log_ptr = slm_ptr + slm_cumsum_log_offset;
    float* cumprod_ptr = slm_ptr + slm_cumprod_offset;
    // =========================================================================
    // Stage 1: Alpha preprocessing (reload channels produced by front kernel)
    // The cumsum_log / cumprod channels were already computed (log2/scan/exp2)
    // by compute_front_body and persisted to global scratch; here we only load
    // them into SLM to keep the log2/scan/exp2 off the recurrent critical path.
    // =========================================================================
    load_alpha_channels(slm_ptr, v_head_id, chunk_offset, item);

    // =========================================================================
    // Stage 2: QK = Q×K^T with fused causal mask + gate scaling
    // [C, Dk] x [C, Dk]^T → [C, C]
    // =========================================================================
    auto q_ptr_chunk =
        q + static_cast<int64_t>(chunk_offset) * num_k_heads * head_k_dim +
        kv_head_id * head_k_dim;
    auto k_ptr_chunk =
        k + static_cast<int64_t>(chunk_offset) * num_k_heads * head_k_dim +
        kv_head_id * head_k_dim;

    // Q [C, Dk], K [C, Dk] — row-major with inter-head stride
    auto Q_shape = make_shape(chunk_size, head_k_dim);
    auto Q_tensor = make_tensor(
        make_gmem_ptr(q_ptr_chunk),
        make_layout(Q_shape, make_stride(head_k_dim * num_k_heads, _1{})));
    auto K_shape = make_shape(chunk_size, head_k_dim);
    auto K_tensor = make_tensor(
        make_gmem_ptr(k_ptr_chunk),
        make_layout(K_shape, make_stride(head_k_dim * num_k_heads, _1{})));

    // QK [C, C] buffer for this chunk
    auto QK_ptr = qk_buf +
                  static_cast<int64_t>(v_head_id) * total_virtual_seqlen * chunk_size +
                  chunk_offset * chunk_size;
    auto QK_shape = make_shape(chunk_size, chunk_size);
    auto QK_tensor = make_tensor(
        make_gmem_ptr(QK_ptr),
        make_layout(QK_shape, make_stride(chunk_size, _1{})));

    // T [C, C] buffer: T = L^{-1}·diag(β), produced by front kernel
    auto T_ptr = t_buf +
           static_cast<int64_t>(v_head_id) * total_virtual_seqlen * chunk_size +
           chunk_offset * chunk_size;

    // --- Stage 2+4 fused: QK = Q×K^T, SK = S×K^T (shared K load) ---
    auto v_ptr_chunk =
        v + static_cast<int64_t>(chunk_offset) * num_v_heads * head_v_dim +
        v_head_id * head_v_dim;

    // SK and NewV buffers (needed by later stages)
    auto SK_ptr = sk_buf +
                  static_cast<int64_t>(v_head_id) * total_virtual_seqlen * head_v_dim +
                  chunk_offset * head_v_dim;
    auto NewV_ptr = newv_buf +
                    static_cast<int64_t>(v_head_id) * total_virtual_seqlen * head_v_dim +
                    chunk_offset * head_v_dim;

    {
      MmaQK mma_qk{};
      auto thr_mma_qk = mma_qk.get_slice(local_id);
      auto wg_tile_qk = mma_qk.tile_mnk();

      // QK layout constants for post-processing
      static constexpr auto tile_m_qk = get<0>(wg_tile_qk);
      static constexpr auto tile_n_qk = get<1>(wg_tile_qk);
      static constexpr auto ATOM_M_QK = get<1>(typename MmaQK::ThrLayoutVMNK{}.shape());
      static constexpr auto ATOM_N_QK = get<2>(typename MmaQK::ThrLayoutVMNK{}.shape());
      static constexpr auto SG_M_QK = tile_m_qk / ATOM_M_QK;
      static constexpr auto SG_N_QK = tile_n_qk / ATOM_N_QK;

      auto sg_local_m_coord = cutlass::get_sub_group_id() / ATOM_N_QK;
      auto sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N_QK;
      int m_sg_start = sg_local_m_coord * SG_M_QK;
      int n_sg_start = sg_local_n_coord * SG_N_QK;

      // QK accumulator + store setup
      Tensor cQK = make_identity_tensor(QK_shape);
      Tensor gQK_C = local_tile(cQK, wg_tile_qk, make_coord(0, 0, 0), Step<_1, _1, X>{});
      auto copy_QK_d = get_block_2d_copy_D<void>(mma_qk, QK_tensor);
      auto thr_copy_QK_d = copy_QK_d.get_slice(local_id);
      auto tCrQK_d = thr_copy_QK_d.partition_sg_fragment_S(gQK_C);
      auto tCgQK_d = thr_copy_QK_d.partition_D(gQK_C);
      auto tSrQK = thr_mma_qk.partition_sg_fragment_C(gQK_C);
      clear(tSrQK);

      // do_sk is WG-uniform → both paths hit same barrier count
      if (!is_first_block || has_prev_state) {
        // Fused path: QK + SK sharing K loads
        MmaSK mma_sk{};
        auto thr_mma_sk = mma_sk.get_slice(local_id);
        auto wg_tile_sk = mma_sk.tile_mnk();

        // S [Dv, Dk] — ssm state
        auto S_shape = make_shape(head_v_dim, head_k_dim);
        auto S_tensor = make_tensor(
            make_gmem_ptr(ssm_state_ptr),
            make_layout(S_shape, make_stride(head_k_dim, _1{})));

        // SK [Dv, C] — output buffer
        auto SK_shape = make_shape(head_v_dim, chunk_size);
        auto SK_tensor = make_tensor(
            make_gmem_ptr(SK_ptr),
            make_layout(SK_shape, make_stride(chunk_size, _1{})));

        Tensor cSK = make_identity_tensor(SK_shape);
        Tensor gSK_C = local_tile(cSK, wg_tile_sk, make_coord(0, 0, 0), Step<_1, _1, X>{});
        auto copy_SK_d = get_block_2d_copy_D<void>(mma_sk, SK_tensor);
        auto thr_copy_SK_d = copy_SK_d.get_slice(local_id);
        auto tCrSK_d = thr_copy_SK_d.partition_sg_fragment_S(gSK_C);
        auto tCgSK_d = thr_copy_SK_d.partition_D(gSK_C);
        auto tSrSK = thr_mma_sk.partition_sg_fragment_C(gSK_C);
        clear(tSrSK);

        // Single fused GEMM: QK = Q×K^T, SK = S×K^T (K loaded once)
        gemm_TTS_fused_2A(Q_tensor, S_tensor, K_tensor,
                          tSrQK, tSrSK, 0, 0, 0, mma_qk, mma_sk);

        // SK post-processing: scale by cumprod(alpha)
        static constexpr auto tile_n_sk = get<1>(wg_tile_sk);
        static constexpr auto ATOM_N_SK = get<2>(typename MmaSK::ThrLayoutVMNK{}.shape());
        static constexpr auto SG_N_SK = tile_n_sk / ATOM_N_SK;
        static constexpr auto ATOM_M_SK = get<1>(typename MmaSK::ThrLayoutVMNK{}.shape());
        static constexpr auto SG_M_SK = get<0>(wg_tile_sk) / ATOM_M_SK;

        auto sg_local_n_coord_sk = cutlass::get_sub_group_id() % ATOM_N_SK;
        int n_sg_start_sk = sg_local_n_coord_sk * SG_N_SK;

        CUTE_UNROLL
        for (int sn = 0; sn < SG_N_SK / sub_group_size; ++sn) {
          int n_idx = n_sg_start_sk + sn * sub_group_size + sg_local_id;
          float cp = cumprod_ptr[n_idx];
          CUTE_UNROLL
          for (int sm = 0; sm < SG_M_SK; ++sm) {
            tSrSK(sn * SG_M_SK + sm) *= cp;
          }
        }

        reorder(tSrSK, tCrSK_d);
        copy(copy_SK_d, tCrSK_d, tCgSK_d);
      } else {
        // No state: just QK, no SK needed
        gemm_TTS(Q_tensor, K_tensor, tSrQK, 0, 0, mma_qk);
      }

      // QK post-processing: causal mask + gate scaling
      CUTE_UNROLL
      for (int sn = 0; sn < SG_N_QK / sub_group_size; ++sn) {
        int n_idx = n_sg_start + sn * sub_group_size + sg_local_id;
        CUTE_UNROLL
        for (int sm = 0; sm < SG_M_QK; ++sm) {
          int m_idx = m_sg_start + sm;
          float gate_scale = sycl::exp2(cumsum_log_ptr[m_idx] - cumsum_log_ptr[n_idx]);
          tSrQK(sn * SG_M_QK + sm) *= gate_scale;
          if (m_idx < n_idx) {
            tSrQK(sn * SG_M_QK + sm) = 0.0f;
          }
        }
      }

      reorder(tSrQK, tCrQK_d);
      copy(copy_QK_d, tCrQK_d, tCgQK_d);
    }

    item.barrier(sycl::access::fence_space::global_and_local);

    // =========================================================================
    // Stage 5: NewV = (V^T - SK) × T^T
    // Ṽ [Dv, C] = V^T [Dv, C] - SK [Dv, C]
    // NewV [Dv, C] = Ṽ [Dv, C] x T [C, C]^T
    // =========================================================================
    {
      MmaNewV mma_newv{};
      auto thr_mma_newv = mma_newv.get_slice(local_id);
      auto wg_tile_newv = mma_newv.tile_mnk();

      // T [C, C] — B operand for gemm_TTS (row-major)
      auto T_shape = make_shape(chunk_size, chunk_size);
      auto T_tensor = make_tensor(
          make_gmem_ptr(T_ptr),
          make_layout(T_shape, make_stride(chunk_size, _1{})));

      // V^T [Dv, C] — transposed view of V (stride-1 along Dv)
      auto VT_shape = make_shape(head_v_dim, chunk_size);
      auto VT_tensor = make_tensor(
          make_gmem_ptr(v_ptr_chunk),
          make_layout(VT_shape, make_stride(_1{}, head_v_dim * num_v_heads)));

      // NewV [Dv, C] — output buffer
      auto NewV_shape = make_shape(head_v_dim, chunk_size);
      auto NewV_tensor = make_tensor(
          make_gmem_ptr(NewV_ptr),
          make_layout(NewV_shape, make_stride(chunk_size, _1{})));

      Tensor cNewV = make_identity_tensor(NewV_shape);
      auto copy_NewV_d = get_block_2d_copy_D<void>(mma_newv, NewV_tensor);
      auto thr_copy_NewV_d = copy_NewV_d.get_slice(local_id);

      {
        Tensor gNewV_C = local_tile(cNewV, wg_tile_newv, make_coord(0, 0, 0), Step<_1, _1, X>{});
        auto tCrNewV_d = thr_copy_NewV_d.partition_sg_fragment_S(gNewV_C);
        auto tCgNewV_d = thr_copy_NewV_d.partition_D(gNewV_C);
        auto tSrNewV = thr_mma_newv.partition_sg_fragment_C(gNewV_C);
        clear(tSrNewV);

        if (!is_first_block || has_prev_state) {
          // NewV = (V^T - SK) × T^T — fused subtract-then-GEMM
          auto SK_shape_s5 = make_shape(head_v_dim, chunk_size);
          auto SK_tensor_s5 = make_tensor(
              make_gmem_ptr(SK_ptr),
              make_layout(SK_shape_s5, make_stride(chunk_size, _1{})));

          gemm_TTS_sub_2A(VT_tensor, SK_tensor_s5, T_tensor, tSrNewV, 0, 0, mma_newv);
        } else {
          // No previous state: SK=0, NewV = V^T × T^T
          gemm_TTS(VT_tensor, T_tensor, tSrNewV, 0, 0, mma_newv);
        }

        reorder(tSrNewV, tCrNewV_d);
        copy(copy_NewV_d, tCrNewV_d, tCgNewV_d);
      }
    }

    item.barrier(sycl::access::fence_space::global_and_local);

    // =========================================================================
    // Stage 6: Output O = O1 + O2
    //   O1 = diag(cumprod) · Q × S^T  (inter-chunk, skip if first & no state)
    //        [C, Dk] x [Dv, Dk]^T → [C, Dv], then scale rows by cumprod[m]
    //   O2 = QK × NewV^T              (intra-chunk)
    //        [C, C] x [Dv, C]^T → [C, Dv]
    // =========================================================================
    {
      MmaO1 mma_o1{};
      MmaO2 mma_o2{};
      auto thr_mma_o1 = mma_o1.get_slice(local_id);
      auto wg_tile_o1 = mma_o1.tile_mnk();

      // Stage 6 currently accumulates O1 and O2 into the same fragment before
      // one final store. Keep O1/O2 fragment-compatible while using mma_o2
      // explicitly for O2 GEMM.
      static_assert(
          std::is_same_v<MmaO1, MmaO2>,
          "Stage 6 shared-accumulator path requires MmaO1 and MmaO2 to match");

      static constexpr auto tile_m_o1 = get<0>(wg_tile_o1);
      static constexpr auto tile_n_o1 = get<1>(wg_tile_o1);
      static constexpr auto ATOM_M_O1 = get<1>(typename MmaO1::ThrLayoutVMNK{}.shape());
      static constexpr auto ATOM_N_O1 = get<2>(typename MmaO1::ThrLayoutVMNK{}.shape());
      static constexpr auto SG_M_O1 = tile_m_o1 / ATOM_M_O1;
      static constexpr auto SG_N_O1 = tile_n_o1 / ATOM_N_O1;

      auto sg_local_m_coord_o1 = cutlass::get_sub_group_id() / ATOM_N_O1;
      auto sg_local_n_coord_o1 = cutlass::get_sub_group_id() % ATOM_N_O1;
      int m_sg_start_o1 = sg_local_m_coord_o1 * SG_M_O1;
      int n_sg_start_o1 = sg_local_n_coord_o1 * SG_N_O1;

      // S [Dv, Dk] — B operand for O1
      auto S_shape = make_shape(head_v_dim, head_k_dim);
      auto S_tensor = make_tensor(
          make_gmem_ptr(ssm_state_ptr),
          make_layout(S_shape, make_stride(head_k_dim, _1{})));

      // NewV [Dv, C] — B operand for O2
      auto NewV_shape = make_shape(head_v_dim, chunk_size);
      auto NewV_tensor = make_tensor(
          make_gmem_ptr(NewV_ptr),
          make_layout(NewV_shape, make_stride(chunk_size, _1{})));

      // QK [C, C] — A operand for O2
      auto QK_shape_s6 = make_shape(chunk_size, chunk_size);
      auto QK_tensor_s6 = make_tensor(
          make_gmem_ptr(QK_ptr),
          make_layout(QK_shape_s6, make_stride(chunk_size, _1{})));

      auto O_ptr_chunk =
          core_attn_out +
          (token_indx ? token_indx[out_chunk_offset] : out_chunk_offset) *
              num_v_heads * head_v_dim +
          v_head_id * head_v_dim;
      // O [C, Dv] output tensor (actual shape [current_chunk_size, head_v_dim])
      auto O_tensor_shape = make_shape(current_chunk_size, head_v_dim);
      auto O_tensor = make_tensor(
          make_gmem_ptr(O_ptr_chunk),
          make_layout(O_tensor_shape, make_stride(num_v_heads * head_v_dim, _1{})));

      Tensor cO = make_identity_tensor(O_tensor.shape());
      auto copy_O_d = get_block_2d_copy_D<void>(mma_o1, O_tensor);
      auto thr_copy_O_d = copy_O_d.get_slice(local_id);

      {
        Tensor gO_C = local_tile(cO, wg_tile_o1, make_coord(0, 0, 0), Step<_1, _1, X>{});
        auto tCrO_d = thr_copy_O_d.partition_sg_fragment_S(gO_C);
        auto tCgO_d = thr_copy_O_d.partition_D(gO_C);
        auto tSrO = thr_mma_o1.partition_sg_fragment_C(gO_C);
        clear(tSrO);

        // O1: diag(cumprod) · Q × S^T (inter-chunk)
        // gemm_TTS(Q, S): C[c, dv] = Σ_dk Q[c, dk] × S[dv, dk]
        //   A=Q [C, Dk], B=S [Dv, Dk], wg_m=0, wg_n=0
        if (!is_first_block || has_prev_state) {
          gemm_TTS(Q_tensor, S_tensor, tSrO, 0, 0, mma_o1);

          // Scale by cumprod per-token (M dimension = tokens)
          CUTE_UNROLL
          for (int sm = 0; sm < SG_M_O1; ++sm) {
            int m_idx = m_sg_start_o1 + sm;
            float cps = cumprod_ptr[m_idx];
            CUTE_UNROLL
            for (int sn = 0; sn < SG_N_O1 / sub_group_size; ++sn) {
              tSrO(sn * SG_M_O1 + sm) *= cps;
            }
          }
        }

        // O2: QK × NewV^T (intra-chunk)
        // gemm_TTS(QK, NewV): C[c, dv] = Σ_c' QK[c, c'] × NewV[dv, c']
        //   A=QK [C, C], B=NewV [Dv, C], wg_m=0, wg_n=0
        gemm_TTS(QK_tensor_s6, NewV_tensor, tSrO, 0, 0, mma_o2);

        reorder(tSrO, tCrO_d);
        copy(copy_O_d, tCrO_d, tCgO_d);
      }
    }

    // Barrier between Stage 6 and Stage 7: Stage 6 reads cumprod_ptr
    // (SLM) and ssm_state (global). Stage 7 overwrites cumprod_ptr
    // (reused as decay_slm) and writes ssm_state. Without this barrier,
    // fast sub-groups entering Stage 7 can corrupt data still being read
    // by slow sub-groups in Stage 6.
    item.barrier(sycl::access::fence_space::global_and_local);

    // =========================================================================
    // Stage 7: State update S' = e^(g_last) · S + decay(NewV) × K
    //   decay(NewV)_j = e^(g_last - g_j) · NewV_j
    //   gemm_TTS_k_multi(NewV, K^T, decay): S'[dv, dk] = Σ_c decay[c]·NewV[dv,c]·K^T[dk,c]
    //   [Dv, C] x [Dk, C]^T → [Dv, Dk], with per-k scaling by decay[c]
    // =========================================================================
    {
      MmaKV mma_kv{};
      auto thr_mma_kv = mma_kv.get_slice(local_id);
      auto wg_tile_kv = mma_kv.tile_mnk();

      static constexpr auto tile_n_kv = get<1>(wg_tile_kv);
      static constexpr auto ATOM_N_KV = get<2>(typename MmaKV::ThrLayoutVMNK{}.shape());
      static constexpr auto SG_N_KV = tile_n_kv / ATOM_N_KV;
      static constexpr auto ATOM_M_KV = get<1>(typename MmaKV::ThrLayoutVMNK{}.shape());
      static constexpr auto SG_M_KV = get<0>(wg_tile_kv) / ATOM_M_KV;

      // g_last = cumsum_log at the last valid position
      float g_last = cumsum_log_ptr[current_chunk_size - 1];
      float g_last_decay = sycl::exp2(g_last);  // exp2 since we use log2

      // K^T [Dk, C] — transposed view of K (stride-1 along Dk)
      auto KT_shape = make_shape(head_k_dim, chunk_size);
      auto KT_tensor = make_tensor(
          make_gmem_ptr(k_ptr_chunk),
          make_layout(KT_shape, make_stride(_1{}, head_k_dim * num_k_heads)));

      // NewV [Dv, C]
      auto NewV_shape = make_shape(head_v_dim, chunk_size);
      auto NewV_tensor = make_tensor(
          make_gmem_ptr(NewV_ptr),
          make_layout(NewV_shape, make_stride(chunk_size, _1{})));

      // S [Dv, Dk]
      auto S_shape = make_shape(head_v_dim, head_k_dim);
      auto S_tensor = make_tensor(
          make_gmem_ptr(ssm_state_ptr),
          make_layout(S_shape, make_stride(head_k_dim, _1{})));

      Tensor cS = make_identity_tensor(S_shape);
      auto copy_S_c = get_block_2d_copy_C<void>(mma_kv, S_tensor);
      auto copy_S_d = get_block_2d_copy_D<void>(mma_kv, S_tensor);
      auto thr_copy_S_c = copy_S_c.get_slice(local_id);
      auto thr_copy_S_d = copy_S_d.get_slice(local_id);

      {
        Tensor gS_C = local_tile(cS, wg_tile_kv, make_coord(0, 0, 0), Step<_1, _1, X>{});
        auto tCrS_d = thr_copy_S_d.partition_sg_fragment_S(gS_C);
        auto tCgS_d = thr_copy_S_d.partition_D(gS_C);
        auto tSrS = thr_mma_kv.partition_sg_fragment_C(gS_C);

        // Seed with e^(g_last) * S_prev
        if (!is_first_block || has_prev_state) {
          auto tCgS_c = thr_copy_S_c.partition_S(gS_C);
          auto tCrS_c = thr_copy_S_c.partition_sg_fragment_D(gS_C);
          copy(copy_S_c, tCgS_c, tCrS_c);
          reorder(tCrS_c, tSrS);
          CUTE_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            tSrS(i) *= g_last_decay;
          }
        } else {
          clear(tSrS);
        }

        // S' += Σ_c decay[c] · NewV[dv,c] · K^T[dk,c]
        // decay[c] = exp2(g_last - cumsum_log[c])
        float* decay_slm = cumprod_ptr;  // Reuse since output already written
        CUTE_UNROLL
        for (int e = local_id; e < chunk_size; e += local_range) {
          decay_slm[e] = sycl::exp2(g_last - cumsum_log_ptr[e]);
        }
        item.barrier(sycl::access::fence_space::local_space);

        gemm_TTS_k_multi(NewV_tensor, KT_tensor, tSrS, 0, 0, mma_kv, decay_slm);

        reorder(tSrS, tCrS_d);
        copy(copy_S_d, tCrS_d, tCgS_d);
      }
    }
  }

  // ============================================================================
  // Front kernel body: per-chunk L build + inverse (chunk-parallel).
  // Computes, for ONE chunk: alpha channels (Stage 1) -> L = I +
  // strict_lower(e^(g_i-g_j)·β_i · K×K^T) (Stage 2 KK only) -> T = L^{-1}·diag(β)
  // (Stage 3 inverse) into t_buf. This is the work that was previously fused
  // into compute_loop_body; it is hoisted out so it can run with a
  // chunk×head grid for much higher occupancy than the (batch,head) grid of
  // the recurrent back kernel.
  // ============================================================================
  CUTE_DEVICE void compute_front_body(
      int chunk_offset,
      int current_chunk_size,
      int v_head_id,
      int kv_head_id,
      float* slm_ptr,
      sycl::nd_item<3>& item) const {
    int local_id = item.get_local_linear_id();
    auto sg = item.get_sub_group();
    int sg_local_id = sg.get_local_linear_id();

    float* cumsum_log_ptr = slm_ptr + slm_cumsum_log_offset;

    const float* alpha_chunk_ptr =
        alpha + static_cast<int64_t>(v_head_id) * total_virtual_seqlen +
        chunk_offset;
    const float* beta_chunk_ptr =
        beta + static_cast<int64_t>(v_head_id) * total_virtual_seqlen +
        chunk_offset;

    // Stage 1: alpha preprocessing — compute cumsum_log / cumprod into SLM and
    // persist both channels to global scratch (cumsum_log_buf / cumprod_buf) so
    // the recurrent back kernel can reload them (design: Option A).
    compute_alpha_channels(
        slm_ptr, alpha_chunk_ptr, current_chunk_size, v_head_id, chunk_offset,
        item);

    // K tensor for this chunk.
    auto k_ptr_chunk =
        k + static_cast<int64_t>(chunk_offset) * num_k_heads * head_k_dim +
        kv_head_id * head_k_dim;
    auto K_tensor_shape = make_shape(chunk_size, head_k_dim);
    auto K_tensor = make_tensor(
        make_gmem_ptr(k_ptr_chunk),
        make_layout(K_tensor_shape, make_stride(head_k_dim * num_k_heads, _1{})));

    // L workspace (InverseType) for this chunk. The inverse (T = L^{-1}·diag(β))
    // is produced by a separate single-subgroup inverse kernel (run_inverse),
    // which reads this l_buf and writes t_buf.
    auto L_ptr = l_buf +
           static_cast<int64_t>(v_head_id) * total_virtual_seqlen * chunk_size +
           chunk_offset * chunk_size;

    // Stage 2 (KK only): build L = I + strict_lower(e^(g_i-g_j)·β_i · K×K^T).
    {
      MmaKK mma_kk{};
      auto thr_mma = mma_kk.get_slice(local_id);
      auto wg_tile = mma_kk.tile_mnk();

      static constexpr auto tile_m = get<0>(wg_tile);
      static constexpr auto tile_n = get<1>(wg_tile);
      static constexpr auto ATOM_M = get<1>(typename MmaKK::ThrLayoutVMNK{}.shape());
      static constexpr auto ATOM_N = get<2>(typename MmaKK::ThrLayoutVMNK{}.shape());
      static constexpr auto SG_M = tile_m / ATOM_M;
      static constexpr auto SG_N = tile_n / ATOM_N;

      auto sg_local_m_coord = cutlass::get_sub_group_id() / ATOM_N;
      auto sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N;
      int m_sg_start = sg_local_m_coord * SG_M;
      int n_sg_start = sg_local_n_coord * SG_N;

      auto KK_tensor_shape = make_shape(chunk_size, chunk_size);
      auto KK_tensor = make_tensor(
          make_gmem_ptr(L_ptr),  // write L (InverseType) for inverse stage
          make_layout(KK_tensor_shape, make_stride(chunk_size, _1{})));

      Tensor cKK = make_identity_tensor(KK_tensor_shape);
      Tensor gKK_C = local_tile(cKK, wg_tile, make_coord(0, 0, 0), Step<_1, _1, X>{});
      auto copy_KK_d = get_block_2d_copy_D<void>(mma_kk, KK_tensor);
      auto thr_copy_KK_d = copy_KK_d.get_slice(local_id);
      auto tCrKK_d = thr_copy_KK_d.partition_sg_fragment_S(gKK_C);
      auto tCgKK_d = thr_copy_KK_d.partition_D(gKK_C);
      auto tSrKK = thr_mma.partition_sg_fragment_C(gKK_C);

      clear(tSrKK);
      gemm_TTS(K_tensor, K_tensor, tSrKK, 0, 0, mma_kk);

      // Hoist the m-indexed SLM/global reads out of the (sn, sm) epilogue loop.
      // cumsum_log[m_idx] and beta[m_idx] depend only on sm, but the original
      // loop re-read them once per sn (SG_N/16 times each) — that was the source
      // of the ~29x SLM read traffic. Preload them once into registers; the
      // n-indexed cumsum_log is also read just once per sn instead of per sm.
      float cs_m[SG_M];
      float beta_m[SG_M];
      CUTE_UNROLL
      for (int sm = 0; sm < SG_M; ++sm) {
        int m_idx = m_sg_start + sm;
        cs_m[sm] = cumsum_log_ptr[m_idx];
        beta_m[sm] = beta_chunk_ptr[m_idx];
      }

      CUTE_UNROLL
      for (int sn = 0; sn < SG_N / sub_group_size; ++sn) {
        int n_idx = n_sg_start + sn * sub_group_size + sg_local_id;
        float cs_n = cumsum_log_ptr[n_idx];
        CUTE_UNROLL
        for (int sm = 0; sm < SG_M; ++sm) {
          int m_idx = m_sg_start + sm;
          if (m_idx > n_idx) {
            // Keep this in LOG space: gate_scale = exp2(cumsum_log[m] -
            // cumsum_log[n]) accumulates only the m..n decay, so the exponent
            // stays bounded and well-conditioned. The algebraically-equal
            // linear form cumprod[m]/cumprod[n] is NOT safe: cumprod[n] =
            // exp2(cumsum_log[n]) with cumsum_log = prefix-sum of log2(alpha<=0)
            // underflows to 0 for strong decay (e.g. exp2(-212) in fp32),
            // giving x/0 = Inf/NaN in the valid (non-padding) region.
            float gate_scale = sycl::exp2(cs_m[sm] - cs_n);
            tSrKK(sn * SG_M + sm) *= gate_scale * beta_m[sm];
          } else if (m_idx == n_idx) {
            tSrKK(sn * SG_M + sm) = 1.0f;
          } else {
            tSrKK(sn * SG_M + sm) = 0.0f;
          }
        }
      }

      reorder(tSrKK, tCrKK_d);
      copy(copy_KK_d, tCrKK_d, tCgKK_d);
    }
  }

  // ============================================================================
  // Front kernel entry: chunk×head persistent grid (mirrors V1 inverse kernel).
  // group(1) encodes (chunk_global * num_v_heads + v_head); each work-group
  // strides over chunks by global_chunk_range to cover the variable-length
  // virtual sequence.
  // ============================================================================
  void run_front(sycl::nd_item<3> item, float* slm_ptr) const {
    const int v_head_id = item.get_group(1) % num_v_heads;
    int chunk_id = item.get_group(1) / num_v_heads;
    const int global_chunk_range = item.get_group_range(1) / num_v_heads;

    const int kv_ratio = num_v_heads / num_k_heads;
    const int kv_head_id = v_head_id / kv_ratio;

    int pre_chunks = 0;
    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
      const int seq_start_offset = query_start_loc[batch_id];
      const int seq_end_offset = query_start_loc[batch_id + 1];
      const int seq_len = seq_end_offset - seq_start_offset;
      const int current_chunks = (seq_len + chunk_size - 1) / chunk_size;
      const int cumsum_chunks = pre_chunks + current_chunks;

      if (chunk_id >= cumsum_chunks) {
        pre_chunks = cumsum_chunks;
        continue;
      }

      while (chunk_id < cumsum_chunks) {
        const int local_chunk = chunk_id - pre_chunks;
        const int chunk_offset = chunk_id * chunk_size;
        int current_chunk_size = chunk_size;
        if ((local_chunk + 1) * chunk_size > seq_len) {
          current_chunk_size = seq_len - local_chunk * chunk_size;
        }
        compute_front_body(
            chunk_offset, current_chunk_size, v_head_id, kv_head_id,
            slm_ptr, item);
        chunk_id += global_chunk_range;
      }
      pre_chunks = cumsum_chunks;
    }
  }

  // ============================================================================
  // Inverse body (single sub-group, barrier-free): migrated from V1's
  // chunk_inverse_opt_kernel. Inverts L in place (l_buf) into L^{-1}, then fuses
  // beta and casts to Element into t_buf. Runs with 16 threads (1 sub-group) per
  // work-group so it is embarrassingly parallel across chunks — NO work-group
  // barriers (intra-block dependencies handled by sub-group broadcast + program
  // order / scoreboard), unlike the 128-thread DPAS path.
  // ============================================================================
  CUTE_DEVICE void compute_inverse_opt_body(
      InverseType* L_ptr,
      T* T_out_ptr,
      const float* beta_chunk_ptr,
      sycl::nd_item<3>& item) const {
    int local_id = item.get_local_linear_id();
    auto sg = item.get_sub_group();
    int sg_local_id = sg.get_local_linear_id();
    int sg_id = local_id / 16;  // 0..3: which of the 4 cooperating sub-groups

    MmaInverse mma{};
    auto wg_tile = mma.tile_mnk();

    auto A_ptr = L_ptr;

    // ----- Step 1: invert the 4 diagonal 16×16 blocks (forward substitution) --
    // 4-sub-group parallel: sub-group sg_id inverts diagonal block sg_id. The
    // blocks are independent and the forward substitution is sub-group-scoped
    // (group_broadcast within sg), so all four run concurrently with no barrier.
    {
      int i = sg_id;
      int offset = i * 16;
      InverseType* A_ptr_xx = A_ptr + offset * chunk_size + offset;
      float A_local[16];
      float A_other[16];
      float A_sum;
      CUTE_UNROLL
      for (int e = 0; e < sg_local_id + 1; ++e) {
        A_local[e] = 0.0f;
      }

      InverseType A_load[16];
      CUTE_UNROLL
      for (int e = 0; e < sg_local_id; ++e) {
        A_load[e] = A_ptr_xx[sg_local_id * chunk_size + e];
      }

      CUTE_UNROLL
      for (int mm_idx = 1; mm_idx < 16; ++mm_idx) {
        CUTE_UNROLL
        for (int nn_idx = 0; nn_idx < mm_idx; ++nn_idx) {
          float send_value = static_cast<float>(A_load[nn_idx]);
          float receive_value = sycl::group_broadcast(sg, send_value, mm_idx);
          if (sg_local_id == nn_idx) {
            A_local[mm_idx] = receive_value;
          }
        }
      }

      CUTE_UNROLL
      for (int mm_idx = 1; mm_idx < 16; ++mm_idx) {
        A_sum = 0.0f;
        CUTE_UNROLL
        for (int e = 1; e < mm_idx + 1; ++e) {
          A_other[e] = sycl::group_broadcast(sg, A_local[mm_idx], e);
        }

        CUTE_UNROLL
        for (int e = 1; e < mm_idx + 1; ++e) {
          A_sum += A_local[e] * A_other[e];
        }

        A_local[mm_idx] = -A_local[mm_idx] - A_sum;
      }

      CUTE_UNROLL
      for (int e = sg_local_id + 1; e < 16; ++e) {
        A_ptr_xx[e * chunk_size + sg_local_id] =
            static_cast<InverseType>(A_local[e]);
      }
    }
    // diagonal blocks now complete & visible work-group-wide
    item.barrier(sycl::access::fence_space::global_and_local);

    // ----- Step 2: T_21 (sub-group 0) and T_43 (sub-group 2) in parallel ------
    auto A_ptr_11 = A_ptr;
    auto A_ptr_21 = A_ptr + 16 * chunk_size;
    auto A_ptr_22 = A_ptr + 16 * chunk_size + 16;
    auto A_ptr_31 = A_ptr + 32 * chunk_size;
    auto A_ptr_33 = A_ptr + 32 * chunk_size + 32;
    auto A_ptr_43 = A_ptr + 48 * chunk_size + 32;
    auto A_ptr_44 = A_ptr + 48 * chunk_size + 48;

    auto A_XX_tensor_shape = make_shape(16, 16);

    // Per-sub-group 16×16 fragments: sliced by sg_local_id so the running
    // sub-group's 16 lanes own a full 16×16 block (T_21 on sg0, T_43 on sg2).
    Tensor cA = make_identity_tensor(A_XX_tensor_shape);
    Tensor cC = make_identity_tensor(A_XX_tensor_shape);
    Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(0, _));
    Tensor gC = local_tile(cC, wg_tile, make_coord(0, 0, 0), Step<_1, _1, X>{});
    auto thr_mma_sg = mma.get_slice(sg_local_id);
    auto tCrA = thr_mma_sg.partition_sg_fragment_A(gA(_, _, 0));
    auto tCrC = thr_mma_sg.partition_sg_fragment_C(gC);

if (sg_id == 0) {
      // T_21 = -(T_22 × KK_21 × T_11)
      auto A_11_tensor_T = make_tensor(make_gmem_ptr(A_ptr_11),
          make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
      auto A_21_tensor = make_tensor(make_gmem_ptr(A_ptr_21),
          make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
      auto A_21_tensor_T = make_tensor(make_gmem_ptr(A_ptr_21),
          make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
      auto A_22_tensor = make_tensor(make_gmem_ptr(A_ptr_22),
          make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
      auto copy_D_21 = get_block_2d_copy_D<void>(mma, A_21_tensor);
      auto thr_copy_D_21 = copy_D_21.get_slice(sg_local_id);
      auto tCrD_21 = thr_copy_D_21.partition_sg_fragment_S(gC);
      auto tCgD_21 = thr_copy_D_21.partition_D(gC);
      clear(tCrC);
      gemm_TTS_sg(A_22_tensor, A_21_tensor_T, tCrC, 0, 0, mma);
      reorder(tCrC, tCrA);
      clear(tCrC);
      gemm_STS_sg(tCrA, A_11_tensor_T, tCrC, 0, 0, mma);
      CUTE_UNROLL
      for (int i = 0; i < tCrC.size(); ++i) {
        tCrC(i) *= -1.0f;
      }
      reorder(tCrC, tCrD_21);
      copy(copy_D_21, tCrD_21, tCgD_21);
    } else if (sg_id == 2) {
      // T_43 = -(T_44 × KK_43 × T_33)
      auto A_33_tensor_T = make_tensor(make_gmem_ptr(A_ptr_33),
          make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
      auto A_43_tensor = make_tensor(make_gmem_ptr(A_ptr_43),
          make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
      auto A_43_tensor_T = make_tensor(make_gmem_ptr(A_ptr_43),
          make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
      auto A_44_tensor = make_tensor(make_gmem_ptr(A_ptr_44),
          make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
      auto copy_D_43 = get_block_2d_copy_D<void>(mma, A_43_tensor);
      auto thr_copy_D_43 = copy_D_43.get_slice(sg_local_id);
      auto tCrD_43 = thr_copy_D_43.partition_sg_fragment_S(gC);
      auto tCgD_43 = thr_copy_D_43.partition_D(gC);
      clear(tCrC);
      gemm_TTS_sg(A_44_tensor, A_43_tensor_T, tCrC, 0, 0, mma);
      reorder(tCrC, tCrA);
      clear(tCrC);
      gemm_STS_sg(tCrA, A_33_tensor_T, tCrC, 0, 0, mma);
      CUTE_UNROLL
      for (int i = 0; i < tCrC.size(); ++i) {
        tCrC(i) *= -1.0f;
      }
      reorder(tCrC, tCrD_43);
      copy(copy_D_43, tCrD_43, tCgD_43);
    }
    // T_A (top-left 32×32) and T_D (bottom-right 32×32) now complete & visible
    item.barrier(sycl::access::fence_space::global_and_local);

    // ----- Step 3: bottom-left 32×32 super-block T_C, cooperative 4 sub-groups
    // T_C = -(T_D · Q) · T_A. Compute N = T_D·Q FIRST so the intermediate N
    // plays the A role in the 2nd GEMM: reorder(C→A) is a no-transpose relayout
    // AND, under SGLayout<4,1,1> (splits M), N's rows stay in the sub-group that
    // produced them ⇒ no cross-sub-group movement. Each sub-group owns an 8×32
    // output stripe ⇒ fp32 accumulator 1 KB ⇒ no GRF spill.
    //   gemm helpers: C[m,n] = Σ_k A[m,k]·B[n,k].
    MmaInverse32 mma32{};
    auto thr_mma32 = mma32.get_slice(local_id);
    auto wg_tile32 = mma32.tile_mnk();

    auto SB_shape = make_shape(32, 32);
    auto TD_tensor = make_tensor(make_gmem_ptr(A_ptr_33),  // T_D [m,k]
        make_layout(SB_shape, make_stride(chunk_size, _1{})));
    auto Q_tensor_T = make_tensor(make_gmem_ptr(A_ptr_31),  // Q^T ⇒ B[n,k]=Q[k,n]
        make_layout(SB_shape, make_stride(_1{}, chunk_size)));
    auto TA_tensor_T = make_tensor(make_gmem_ptr(A_ptr_11),  // T_A^T ⇒ B[n,k]=T_A[k,n]
        make_layout(SB_shape, make_stride(_1{}, chunk_size)));

    Tensor cA32 = make_identity_tensor(SB_shape);
    Tensor cC32 = make_identity_tensor(SB_shape);
    Tensor gA32 = local_tile(cA32, select<0, 2>(wg_tile32), make_coord(0, _));
    Tensor gC32 =
        local_tile(cC32, wg_tile32, make_coord(0, 0, 0), Step<_1, _1, X>{});
    auto tCrA32 = thr_mma32.partition_sg_fragment_A(gA32(_, _, 0));
    auto tCrC32 = thr_mma32.partition_sg_fragment_C(gC32);

    // Load T_D (bottom-right 32×32 = the D^{-1} output block) ONCE into an
    // accumulator-layout fragment. It feeds GEMM1 as the A operand AND is the
    // source of the D^{-1} output block, so we beta-fuse + store it right here
    // instead of re-reading it from l_buf in Step 4 ("don't read twice"). The
    // 32-wide store is a full 64-byte cache line, cutting partial writes.
    auto copy_C_TD = get_block_2d_copy_C<void>(mma32, TD_tensor);
    auto thr_copy_C_TD = copy_C_TD.get_slice(local_id);
    auto tCgC_TD = thr_copy_C_TD.partition_S(gC32);
    auto tCrC_TD_load = thr_copy_C_TD.partition_sg_fragment_D(gC32);
    copy(copy_C_TD, tCgC_TD, tCrC_TD_load);
    auto accTD = thr_mma32.partition_sg_fragment_C(gC32);
    reorder(tCrC_TD_load, accTD);  // T_D in accumulator layout (float)

    // Make the unscaled half A-copy for GEMM1 FIRST (before β-scaling accTD),
    // then store the D^{-1} block so accTD dies before GEMM1's accumulator
    // (tCrC32) goes live — peak live fragments drop 3→2 and the store latency
    // overlaps GEMM1 compute.
    reorder(accTD, tCrA32);  // T_D as A operand (C→A, no transpose), unscaled

    // Fuse the D^{-1} output block (rows 32-63, cols 32-63): beta-scale T_D's
    // columns (global cols 32..63), cast, store 32-wide. No -1 sign (a diagonal
    // super-block inverse keeps its sign). accTD is in the C-accumulator layout
    // ⇒ element sn*8+sm maps to column sn*16+sg_local_id (SGLayout<4,1,1>).
    {
      auto TD_out_ptr = T_out_ptr + 32 * chunk_size + 32;
      auto TD_out_tensor = make_tensor(make_gmem_ptr(TD_out_ptr),
          make_layout(SB_shape, make_stride(chunk_size, _1{})));
      CUTE_UNROLL
      for (int sn = 0; sn < 2; ++sn) {
        float bv = beta_chunk_ptr[32 + sn * sub_group_size + sg_local_id];
        CUTE_UNROLL
        for (int sm = 0; sm < 8; ++sm) {
          accTD(sn * 8 + sm) *= bv;
        }
      }
      auto copy_D_TD = get_block_2d_copy_D<void>(mma32, TD_out_tensor);
      auto thr_copy_D_TD = copy_D_TD.get_slice(local_id);
      auto tCrD_TD = thr_copy_D_TD.partition_sg_fragment_S(gC32);
      auto tCgD_TD = thr_copy_D_TD.partition_D(gC32);
      reorder(accTD, tCrD_TD);
      copy(copy_D_TD, tCrD_TD, tCgD_TD);
    }

    // GEMM1: N = T_D · Q   (C[m,n]=Σ_k T_D[m,k]·Q[k,n])
    clear(tCrC32);
    gemm_STS(tCrA32, Q_tensor_T, tCrC32, 0, 0, mma32);

    // reorder N (C-accumulator) into the A operand — no transpose, intra-sg.
    reorder(tCrC32, tCrA32);
    clear(tCrC32);
    // GEMM2: T_C = N · T_A  (C[m,n]=Σ_k N[m,k]·T_A[k,n])
    gemm_STS(tCrA32, TA_tensor_T, tCrC32, 0, 0, mma32);

    // Fuse Step-4 (beta-scale + cast) directly here: T_C is terminal (no later
    // inverse step reads it), so write the bottom-left 32×32 straight to t_buf
    // instead of round-tripping through l_buf. The 32-wide rows are full 64-byte
    // cache lines (vs the four half-line 16×16 epilogue stores), cutting partial
    // writes. beta scales columns; the -1 sign is folded in. Fragment element
    // index = sn*SG_M + sm (SG_M=8 rows/sg; SG_N=32 ⇒ sn∈{0,1} selects column
    // n = sn*16 + sg_local_id) — same layout convention as the front KK epilogue.
    {
      auto TC_out_ptr = T_out_ptr + 32 * chunk_size;  // rows 32-63, cols 0-31
      auto TC_out_tensor = make_tensor(make_gmem_ptr(TC_out_ptr),
          make_layout(SB_shape, make_stride(chunk_size, _1{})));
      auto copy_D_TCo = get_block_2d_copy_D<void>(mma32, TC_out_tensor);
      auto thr_copy_D_TCo = copy_D_TCo.get_slice(local_id);
      auto tCrD_TCo = thr_copy_D_TCo.partition_sg_fragment_S(gC32);
      auto tCgD_TCo = thr_copy_D_TCo.partition_D(gC32);
      CUTE_UNROLL
      for (int sn = 0; sn < 2; ++sn) {
        float bv = beta_chunk_ptr[sn * sub_group_size + sg_local_id];
        CUTE_UNROLL
        for (int sm = 0; sm < 8; ++sm) {
          tCrC32(sn * 8 + sm) *= -bv;
        }
      }
      reorder(tCrC32, tCrD_TCo);
      copy(copy_D_TCo, tCrD_TCo, tCgD_TCo);
    }
    // No barrier needed between Step 3 and Step 4: Step 3 writes only t_buf
    // (rows 32-63), while Step 4 reads l_buf rows 0-31 (T_A — produced in
    // Steps 1-2 and already visible from the pre-Step-3 barrier) and writes
    // t_buf rows 0-31. The l_buf reads and t_buf writes are disjoint from
    // anything Step 3 touches, so there is no cross-step hazard.

    // ----- Step 4: fuse beta + cast for T_A (top-left 32×32) -> t_buf ---------
    // Only T_A remains: T_C (bottom-left) and T_D (bottom-right) were beta-fused
    // and stored in Step 3. T_A = 4 blocks (block-rows 0-1 × block-cols 0-1),
    // spread one-per-sub-group across ALL 4 sub-groups (sg0→(0,0), sg1→(0,1),
    // sg2→(1,0), sg3→(1,1)) so the epilogue is a single fully-parallel step
    // instead of 2 sub-groups doing 2 blocks each. The strict-upper-right 32×32
    // (rows 0-31, cols 32-63) of L^{-1} is all zeros and t_buf is zero-init
    // (torch::zeros) per launch, so those blocks are skipped — no store needed.
    {
      auto blk_shape = make_shape(_16{}, _16{});
      Tensor cC3 = make_identity_tensor(blk_shape);
      Tensor gC3 = local_tile(cC3, wg_tile, make_coord(0, 0, 0), Step<_1, _1, X>{});
      auto tCrC3 = thr_mma_sg.partition_sg_fragment_C(gC3);

      int bi = sg_id / 2;  // block-row 0 or 1
      int bj = sg_id % 2;  // block-col 0 or 1
      {
        // column owned by this lane within block-column bj
        float bv = beta_chunk_ptr[bj * 16 + sg_local_id];
        InverseType* L_blk = A_ptr + (bi * 16) * chunk_size + bj * 16;
        T* T_blk = T_out_ptr + (bi * 16) * chunk_size + bj * 16;
        auto L_blk_tensor = make_tensor(make_gmem_ptr(L_blk),
            make_layout(blk_shape, make_stride(chunk_size, _1{})));
        auto T_blk_tensor = make_tensor(make_gmem_ptr(T_blk),
            make_layout(blk_shape, make_stride(chunk_size, _1{})));

        auto copy_c3 = get_block_2d_copy_C<void>(mma, L_blk_tensor);
        auto copy_d3 = get_block_2d_copy_D<void>(mma, T_blk_tensor);
        auto thr_c3 = copy_c3.get_slice(sg_local_id);
        auto thr_d3 = copy_d3.get_slice(sg_local_id);

        // block-2d load L^{-1} block -> float accumulator fragment
        auto tCgC3 = thr_c3.partition_S(gC3);
        auto tCrC3_load = thr_c3.partition_sg_fragment_D(gC3);
        copy(copy_c3, tCgC3, tCrC3_load);
        reorder(tCrC3_load, tCrC3);

        // fuse beta per column (constant per lane), reorder to store fragment
        CUTE_UNROLL
        for (int i = 0; i < tCrC3.size(); ++i) {
          tCrC3(i) *= bv;
        }
        auto tCrD3 = thr_d3.partition_sg_fragment_S(gC3);
        auto tCgD3 = thr_d3.partition_D(gC3);
        reorder(tCrC3, tCrD3);
        copy(copy_d3, tCrD3, tCgD3);
      }
    }
  }

  // ============================================================================
  // Inverse kernel entry: chunk×head persistent grid, 1 sub-group per chunk.
  // Reads l_buf (L from run_front), writes t_buf (T = L^{-1}·diag(β)).
  // ============================================================================
  void run_inverse(sycl::nd_item<3> item) const {
    const int v_head_id = item.get_group(1) % num_v_heads;
    int chunk_id = item.get_group(1) / num_v_heads;
    const int global_chunk_range = item.get_group_range(1) / num_v_heads;

    int pre_chunks = 0;
    for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
      const int seq_start_offset = query_start_loc[batch_id];
      const int seq_end_offset = query_start_loc[batch_id + 1];
      const int seq_len = seq_end_offset - seq_start_offset;
      const int current_chunks = (seq_len + chunk_size - 1) / chunk_size;
      const int cumsum_chunks = pre_chunks + current_chunks;

      if (chunk_id >= cumsum_chunks) {
        pre_chunks = cumsum_chunks;
        continue;
      }

      while (chunk_id < cumsum_chunks) {
        const int chunk_offset = chunk_id * chunk_size;
        auto L_ptr = l_buf +
            static_cast<int64_t>(v_head_id) * total_virtual_seqlen * chunk_size +
            chunk_offset * chunk_size;
        auto T_ptr = t_buf +
            static_cast<int64_t>(v_head_id) * total_virtual_seqlen * chunk_size +
            chunk_offset * chunk_size;
        const float* beta_chunk_ptr =
            beta + static_cast<int64_t>(v_head_id) * total_virtual_seqlen +
            chunk_offset;
        compute_inverse_opt_body(L_ptr, T_ptr, beta_chunk_ptr, item);
        chunk_id += global_chunk_range;
      }
      pre_chunks = cumsum_chunks;
    }
  }

  // ============================================================================
  // Kernel entry with SLM (called from launcher)
  // ============================================================================
  void operator()(
      sycl::nd_item<3> item,
      float* slm_ptr) const {
    int current_batch_id = item.get_group(0);
    int v_head_id = item.get_group(1);
    int local_id = item.get_local_linear_id();
    int local_range = item.get_local_range(2);

    auto sg = item.get_sub_group();
    int sg_local_id = sg.get_local_linear_id();

    const int kv_ratio = num_v_heads / num_k_heads;
    const int kv_head_id = v_head_id / kv_ratio;

    int seq_start_offset = query_start_loc[current_batch_id];
    int seq_end_offset = query_start_loc[current_batch_id + 1];
    int seq_len = seq_end_offset - seq_start_offset;
    int num_chunks = (seq_len + chunk_size - 1) / chunk_size;

    bool initial_state = has_initial_state ? has_initial_state[current_batch_id] : false;

    StateT* ssm_state_ptr =
        ssm_state +
        static_cast<int64_t>(cache_indices[current_batch_id]) * ssm_state_stride_0 +
        v_head_id * head_v_dim * head_k_dim;

    int pre_chunks = 0;
    for (int b = 0; b < current_batch_id; ++b) {
      int b_seq_len = query_start_loc[b + 1] - query_start_loc[b];
      pre_chunks += (b_seq_len + chunk_size - 1) / chunk_size;
    }

    // Chunk loop with first/last specialization
    for (int chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {
      int chunk_offset = (pre_chunks + chunk_id) * chunk_size;
      int out_chunk_offset = seq_start_offset + chunk_id * chunk_size;
      int current_chunk_size = chunk_size;
      if ((chunk_id + 1) * chunk_size > seq_len) {
        current_chunk_size = seq_len - chunk_id * chunk_size;
      }

      bool has_prev_state = (chunk_id != 0) || initial_state;

      if (chunk_id == 0) {
        compute_loop_body(
            chunk_id, chunk_offset, out_chunk_offset, current_chunk_size,
            v_head_id, kv_head_id,
            cute::true_type{},
            has_prev_state,
            slm_ptr,
            ssm_state_ptr, item);
      } else {
        compute_loop_body(
            chunk_id, chunk_offset, out_chunk_offset, current_chunk_size,
            v_head_id, kv_head_id,
            cute::false_type{},
            has_prev_state,
            slm_ptr,
            ssm_state_ptr, item);
      }
    }
  }

  // ============================================================================
  // Launch configuration
  // ============================================================================
  static sycl::nd_range<3> get_nd_range(int batch_size, int num_v_heads) {
    // One workgroup per (batch, v_head) pair
    // Use the largest MMA policy (KV: 128×128, 32 SGs = 512 threads)
    MmaKV mma_kv{};
    int threads_per_wg = size(mma_kv);
    sycl::range<3> local(1, 1, threads_per_wg);
    sycl::range<3> global(batch_size, num_v_heads, 1);
    return sycl::nd_range<3>{global * local, local};
  }

  // Front kernel launch config: chunk×head persistent grid. Same thread count
  // (128) as the main kernel so the KK build and inverse partition identically;
  // the grid is sized to fill the device (mirrors V1's inverse kernel).
  static sycl::nd_range<3> get_front_nd_range(int num_v_heads) {
    MmaKK mma_kk{};
    int threads_per_wg = size(mma_kk);
    int sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    int num_wg =
        (sm_count * MaxThreadsPerSM_grf128 / threads_per_wg + num_v_heads - 1) /
        num_v_heads * num_v_heads;
    sycl::range<3> local(1, 1, threads_per_wg);
    sycl::range<3> global(1, num_wg, 1);
    return sycl::nd_range<3>{global * local, local};
  }

  // Inverse kernel launch config: 1 sub-group (16 threads) per work-group so
  // each work-group solves one chunk's inverse with NO work-group barriers
  // (mirrors V1's ChunkInverseOptKernel). chunk×head persistent grid.
  static sycl::nd_range<3> get_inverse_nd_range(int num_v_heads) {
    // 4 sub-groups per work-group (64 threads) cooperate on one chunk's 64×64
    // inverse: parallel diagonal inverses + parallel epilogue + a cooperative
    // 32×32 super-block (SGLayout<4,1,1>, no GRF spill).
    MmaInverse32 mma_inv{};
    int threads_per_wg = size(mma_inv);
    int sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    // grf128 (8 threads/EU) so the latency-bound inverse gets 2x the resident
    // threads vs grf256; the inverse uses few registers (16 threads, a handful
    // of 16x16 blocks) so it does not need the 256-GRF budget.
    int num_wg =
        (sm_count * MaxThreadsPerSM_grf128 / threads_per_wg + num_v_heads - 1) /
        num_v_heads * num_v_heads;
    sycl::range<3> local(1, 1, threads_per_wg);
    sycl::range<3> global(1, num_wg, 1);
    return sycl::nd_range<3>{global * local, local};
  }

  static int get_slm_size() {
    return slm_total_size;
  }

  // Front kernel only touches the alpha channels (cumsum_log / cumprod); it does
  // NOT use the inverse scratch region. Allocating the full slm_total_size there
  // (33 KB) would pin ~64x more SLM than needed and crush occupancy (dispatch
  // resource stalls). Hand it just the alpha region so many more work-groups can
  // reside per Xe-core and hide the load/barrier latency.
  static int get_front_slm_size() {
    return slm_alpha_size;
  }
};

// ============================================================================
// Kernel class names for SYCL
// ============================================================================
template <typename T, typename StateT>
class ChunkGDNV2Kernel;

template <typename T, typename StateT>
class ChunkGDNV2FrontKernel;

template <typename T, typename StateT>
class ChunkGDNV2InverseKernel;

// ============================================================================
// Launcher function
// ============================================================================
template <typename T, typename StateT>
void kernel_launcher_v2(
    sycl::queue& queue,
    T* core_attn_out,
    const T* q,
    const T* k,
    const T* v,
    const float* a_raw,       // [num_v_heads, total_virtual_seqlen] raw gate param
    const float* A_log,       // [num_v_heads]
    const T* dt_bias,         // [num_v_heads]
    float* alpha_buf,         // [num_v_heads, total_virtual_seqlen] output of prepare
    const float* beta,
    StateT* ssm_state,
    const int ssm_state_stride_0,
    const int* query_start_loc,
    const int* cache_indices,
    const bool* has_initial_state,
    const int* token_indx,
    T* qk_buf,
    InverseType* l_buf,
    T* t_buf,
    T* sk_buf,
    T* newv_buf,
    float* cumsum_log_buf,
    float* cumprod_buf,
    const int batch_size,
    const int total_seqlen,
    const int total_virtual_seqlen,
    const int num_k_heads,
    const int head_k_dim,
    const int num_v_heads,
    const int head_v_dim) {
  using Kernel = chunk_gated_delta_rule_v2_kernel<T, StateT>;

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;



  // -------------------------------------------------------------------------
  // Step 1: Launch V2 prepare kernel (design §2.9)
  //   T is already resolved, no dtype dispatch needed.
  // -------------------------------------------------------------------------
  {
    syclex::properties kernel_props_prepare{
      syclex::sub_group_size<cute::detail::subgroup_size>,
      intelex::grf_size<128>};
    int sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    sycl::range<3> local_prepare(1, 1, MaxThreadsPerSM);
    sycl::range<3> global_prepare(1, sm_count, 1);

    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<ChunkPrepareV2Kernel<T, StateT>>(
          sycl::nd_range<3>{global_prepare * local_prepare, local_prepare},
          kernel_props_prepare,
          [=](auto) {
            chunk_prepare_v2_kernel<T, StateT>(
                alpha_buf, a_raw, A_log, dt_bias,
                query_start_loc, total_virtual_seqlen, batch_size, num_v_heads);
          });
    });
  }

  // -------------------------------------------------------------------------
  // Step 2: Launch main V2 kernel
  // -------------------------------------------------------------------------

  Kernel kernel_obj{
      core_attn_out,
      q, k, v,
      alpha_buf, beta,
      ssm_state,
      ssm_state_stride_0,
      query_start_loc,
      cache_indices,
      has_initial_state,
      token_indx,
      qk_buf, l_buf, t_buf, sk_buf, newv_buf,
      cumsum_log_buf, cumprod_buf,
      batch_size,
      total_seqlen,
      total_virtual_seqlen,
      num_k_heads,
      head_k_dim,
      num_v_heads,
      head_v_dim};

  // -------------------------------------------------------------------------
  // Step 1b: Launch chunk-parallel front kernel (KK -> L build -> l_buf).
  // Runs after prepare (needs alpha_buf); 128 threads (8 sub-groups) for the
  // DPAS KK build. Reuses the same Kernel struct/params; only the grid and the
  // entry point (run_front) differ.
  // -------------------------------------------------------------------------
  {
    syclex::properties kernel_props_front{
        syclex::sub_group_size<cute::detail::subgroup_size>,
        intelex::grf_size<128>};
    auto front_range = Kernel::get_front_nd_range(num_v_heads);
    int front_slm_size = Kernel::get_front_slm_size();
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> local_mem(sycl::range<1>(front_slm_size), cgh);
      cgh.parallel_for<ChunkGDNV2FrontKernel<T, StateT>>(
          front_range,
          kernel_props_front,
          [=](sycl::nd_item<3> item) {
            float* slm_ptr = static_cast<float*>(
                local_mem.template get_multi_ptr<sycl::access::decorated::no>().get());
            kernel_obj.run_front(item, slm_ptr);
          });
    });
  }

  // -------------------------------------------------------------------------
  // Step 1c: Launch chunk-parallel inverse kernel (l_buf -> T = L^{-1}·diag(β)
  // -> t_buf). 1 sub-group (16 threads) per work-group, NO work-group barriers,
  // so it is embarrassingly parallel across chunks (mirrors V1's
  // ChunkInverseOptKernel). Runs after the front kernel (needs l_buf) and
  // before the main kernel (reads t_buf in Stage 5). No SLM needed.
  // -------------------------------------------------------------------------
  {
    syclex::properties kernel_props_inverse{
        syclex::sub_group_size<cute::detail::subgroup_size>,
        intelex::grf_size<128>};
    auto inverse_range = Kernel::get_inverse_nd_range(num_v_heads);
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<ChunkGDNV2InverseKernel<T, StateT>>(
          inverse_range,
          kernel_props_inverse,
          [=](sycl::nd_item<3> item) {
            kernel_obj.run_inverse(item);
          });
    });
  }

  {
    syclex::properties kernel_props{
      syclex::sub_group_size<cute::detail::subgroup_size>,
      intelex::grf_size<128>};
    auto nd_range = Kernel::get_nd_range(batch_size, num_v_heads);
    int slm_size = Kernel::get_slm_size();
    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> local_mem(sycl::range<1>(slm_size), cgh);
      cgh.parallel_for<ChunkGDNV2Kernel<T, StateT>>(
          nd_range,
          kernel_props,
          [=](sycl::nd_item<3> item) {
            float* slm_ptr = static_cast<float*>(
                local_mem.template get_multi_ptr<sycl::access::decorated::no>().get());
            kernel_obj(item, slm_ptr);
          });
    });
  }
}

// ============================================================================
// Top-level implementation function (matches V1 interface pattern)
// ============================================================================
void chunk_gated_delta_rule_v2_impl_xe2(
    sycl::queue& queue,
    torch::Tensor& core_attn_out,  // [total_seqlen, num_v_heads, head_v_dim]
    const torch::Tensor& q,  // [total_virtual_seqlen, num_k_heads, head_k_dim]
    const torch::Tensor& k,  // [total_virtual_seqlen, num_k_heads, head_k_dim]
    const torch::Tensor& v,  // [total_virtual_seqlen, num_v_heads, head_v_dim]
    const torch::Tensor& b,  // [num_v_heads, total_virtual_seqlen] (beta)
    const torch::Tensor& a,  // [num_v_heads, total_virtual_seqlen] (alpha, linear space from prepare)
    const torch::Tensor& A_log,    // [num_v_heads]
    const torch::Tensor& dt_bias,  // [num_v_heads]
    torch::Tensor&
        ssm_state,  // [cache_batch_size, num_v_heads, head_v_dim, head_k_dim]
    const torch::Tensor& query_start_loc,  // [batch_size + 1]
    const torch::Tensor& cache_indices,    // [batch_size]
    const std::optional<torch::Tensor>&
        has_initial_state,  // [batch_size] or None
    const int num_prefills,
    const int num_decodes,
    const int* token_indx) {
  if (num_prefills == 0 && num_decodes == 0) {
    return;
  }

  int batch_size = query_start_loc.size(0) - 1;
  if (num_prefills == 0 && num_decodes > 0) {
    batch_size = num_decodes;
  }
  const int total_seqlen = core_attn_out.size(0);
  const int total_virtual_seqlen = q.size(0);
  const int num_k_heads = q.size(1);
  const int head_k_dim = q.size(2);
  const int num_v_heads = v.size(1);
  const int head_v_dim = v.size(2);
  const int ssm_state_stride_0 = ssm_state.stride(0);

  TORCH_CHECK(num_v_heads % num_k_heads == 0);
  TORCH_CHECK(
      A_log.scalar_type() == at::kFloat,
      "A_log dtype must be float32, but got ", A_log.scalar_type());
  TORCH_CHECK(
      dt_bias.scalar_type() == core_attn_out.scalar_type(),
      "dt_bias dtype must match core_attn_out dtype");

  auto dtype = core_attn_out.dtype();
  auto device = core_attn_out.device();

  int padding_size = batch_size * (chunk_size - 1);

  // Allocate alpha buffer for prepare kernel output
  torch::Tensor alpha_buf = torch::empty(
      {num_v_heads, total_virtual_seqlen},
      torch::dtype(torch::kFloat32).device(device).requires_grad(false));

  // Alpha-channel scratch (design: Option A). The front kernel computes
  // cumsum_log / cumprod once per chunk×head and persists them here so the
  // recurrent back kernel reloads instead of recomputing log2/scan/exp2.
  torch::Tensor cumsum_log_buf = torch::empty(
      {num_v_heads, total_virtual_seqlen},
      torch::dtype(torch::kFloat32).device(device).requires_grad(false));
  torch::Tensor cumprod_buf = torch::empty(
      {num_v_heads, total_virtual_seqlen},
      torch::dtype(torch::kFloat32).device(device).requires_grad(false));

  // Allocate intermediate buffers (design §2.6)
  torch::Tensor qk_buf = torch::zeros(
      {num_v_heads, total_seqlen + padding_size, chunk_size},
      torch::dtype(dtype).device(device).requires_grad(false));
  torch::Tensor t_buf = torch::zeros(
      {num_v_heads, total_seqlen + padding_size, chunk_size},
      torch::dtype(dtype).device(device).requires_grad(false));
  torch::Tensor l_buf = torch::zeros(
      {num_v_heads, total_seqlen + padding_size, chunk_size},
        torch::dtype(
            std::is_same_v<InverseType, tfloat32_t>
                ? torch::kFloat32
                : torch::kFloat16)
          .device(device)
          .requires_grad(false));
  torch::Tensor sk_buf = torch::zeros(
      {num_v_heads, total_seqlen + padding_size, head_v_dim},
      torch::dtype(dtype).device(device).requires_grad(false));
  torch::Tensor newv_buf = torch::zeros(
      {num_v_heads, total_seqlen + padding_size, head_v_dim},
      torch::dtype(dtype).device(device).requires_grad(false));

  // kernel_launcher_v2 handles both prepare kernel and main kernel launch.
  //   prepare: raw (a, dt_bias, A_log) → alpha_buf (linear space, per-token)
  //   main:    uses alpha_buf + beta(=b) for the fused computation

#define V2_KERNEL_LAUNCHER(scalar_t, state_scalar_t)                   \
  kernel_launcher_v2<scalar_t, state_scalar_t>(                        \
      queue,                                                           \
      reinterpret_cast<scalar_t*>(core_attn_out.data_ptr()),           \
      reinterpret_cast<const scalar_t*>(q.data_ptr()),                 \
      reinterpret_cast<const scalar_t*>(k.data_ptr()),                 \
      reinterpret_cast<const scalar_t*>(v.data_ptr()),                 \
      reinterpret_cast<const float*>(a.data_ptr()),                    \
      reinterpret_cast<const float*>(A_log.data_ptr()),                \
      reinterpret_cast<const scalar_t*>(dt_bias.data_ptr()),           \
      reinterpret_cast<float*>(alpha_buf.data_ptr()),                  \
      reinterpret_cast<const float*>(b.data_ptr()),                    \
      reinterpret_cast<state_scalar_t*>(ssm_state.data_ptr()),         \
      ssm_state_stride_0,                                              \
      reinterpret_cast<const int*>(query_start_loc.data_ptr()),        \
      reinterpret_cast<const int*>(cache_indices.data_ptr()),          \
      has_initial_state.has_value()                                     \
          ? reinterpret_cast<const bool*>(has_initial_state->data_ptr()) \
          : nullptr,                                                   \
      token_indx,                                                      \
      reinterpret_cast<scalar_t*>(qk_buf.data_ptr()),                  \
      reinterpret_cast<InverseType*>(l_buf.data_ptr()),                \
      reinterpret_cast<scalar_t*>(t_buf.data_ptr()),                   \
      reinterpret_cast<scalar_t*>(sk_buf.data_ptr()),                  \
      reinterpret_cast<scalar_t*>(newv_buf.data_ptr()),                \
      reinterpret_cast<float*>(cumsum_log_buf.data_ptr()),             \
      reinterpret_cast<float*>(cumprod_buf.data_ptr()),                \
      batch_size,                                                      \
      total_seqlen,                                                    \
      total_virtual_seqlen,                                            \
      num_k_heads,                                                     \
      head_k_dim,                                                      \
      num_v_heads,                                                     \
      head_v_dim);

#define V2_DISPATCH_STATE_DTYPE(scalar_t)                                    \
  do {                                                                       \
    if (ssm_state.scalar_type() == at::kFloat) {                             \
      using state_scalar_t = float;                                          \
      V2_KERNEL_LAUNCHER(scalar_t, state_scalar_t)                           \
    } else if (ssm_state.scalar_type() == at::kBFloat16) {                   \
      using state_scalar_t = bfloat16_t;                                     \
      V2_KERNEL_LAUNCHER(scalar_t, state_scalar_t)                           \
    } else if (ssm_state.scalar_type() == at::kHalf) {                       \
      using state_scalar_t = half_t;                                         \
      V2_KERNEL_LAUNCHER(scalar_t, state_scalar_t)                           \
    } else {                                                                 \
      TORCH_CHECK(                                                           \
          false,                                                             \
          "ssm_state dtype must be float32/float16/bfloat16, but got ",      \
          ssm_state.scalar_type());                                          \
    }                                                                        \
  } while (0)

  if (core_attn_out.scalar_type() == at::kBFloat16) {
    using scalar_t = bfloat16_t;
    V2_DISPATCH_STATE_DTYPE(scalar_t);
  } else if (core_attn_out.scalar_type() == at::kHalf) {
    using scalar_t = half_t;
    V2_DISPATCH_STATE_DTYPE(scalar_t);
  } else {
    TORCH_CHECK(
        false,
        "core_attn_out dtype must be float16/bfloat16, but got ",
        core_attn_out.scalar_type());
  }

#undef V2_DISPATCH_STATE_DTYPE
#undef V2_KERNEL_LAUNCHER
}

}  // namespace v2
}  // namespace gdn
