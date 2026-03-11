#include <limits>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <torch/all.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/platform/platform.h"

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/tensor.hpp"

#include "csrc/utils.h"
#include "csrc/xpu/grouped_gemm/xe_2/gemm_xe2_policy.hpp"
#include "mqa_logits_xe2.h"


using namespace cute;

class mma_policy_base {
 public:
  using WGTile = Shape<_256, _256, _32>;
  using SGLayout = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;

  // Copy can be turned for better performance
  using GmemTiledCopyA = void;  // same as make_block_2d_copy_A
  using GmemTiledCopyB = void;  // same as make_block_2d_copy_B
  using GmemTiledCopyD = void;  // same as make_block_2d_copy_D
};


class w8a8_policy_m_32 : public mma_policy_base {
 public:
  using WGTile = Shape<_32, _64, _32>;
  using SGLayout = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
};



torch::Tensor fp8_mqa_logits_xe2(
    const torch::Tensor& q, // (seq_len, num_heads, head_dim), dtype float8
    const torch::Tensor& kv, // (seq_len_kv, head_dim), dtype float8
    const torch::Tensor& kv_scales, // (seq_len_kv), dtype float32
    const torch::Tensor& weights, // (seq_len, num_heads), dtype float32
    const torch::Tensor& cu_seqlen_ks, // (seq_len), dtype int32
    const torch::Tensor& cu_seqlen_ke, // (seq_len), dtype int32
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    int64_t seq_len_kv) {
    const float neg_inf = -std::numeric_limits<float>::infinity();
    auto logits = torch::full(
      {seq_len, seq_len_kv},
      neg_inf,
      torch::dtype(torch::kFloat).device(q.device()).requires_grad(false));

  const auto* q_ptr = reinterpret_cast<const float_e4m3_t*>(q.data_ptr());
  const auto* kv_ptr = reinterpret_cast<const float_e4m3_t*>(kv.data_ptr());
  const float* scales_ptr = kv_scales.data_ptr<float>();
  const float* weights_ptr = weights.data_ptr<float>();
  const int32_t* ks_ptr = cu_seqlen_ks.data_ptr<int32_t>();
  const int32_t* ke_ptr = cu_seqlen_ke.data_ptr<int32_t>();
  float* out_ptr = logits.data_ptr<float>();

  auto q_tensor = make_tensor(q_ptr, make_layout(make_shape(seq_len, num_heads, head_dim), make_stride(num_heads * head_dim, head_dim, _1{})));
  auto kv_tensor = make_tensor(kv_ptr, make_layout(make_shape(seq_len_kv, head_dim), make_stride(head_dim, _1{})));
  auto scales_tensor = make_tensor(scales_ptr, make_layout(make_shape(seq_len_kv), make_stride(_1{})));
  auto weights_tensor = make_tensor(weights_ptr, make_layout(make_shape(seq_len, num_heads), make_stride(num_heads, _1{})));
  auto ks_tensor = make_tensor(ks_ptr, make_layout(make_shape(seq_len), make_stride(_1{})));
  auto ke_tensor = make_tensor(ke_ptr, make_layout(make_shape(seq_len), make_stride(_1{})));
  auto out_tensor = make_tensor(out_ptr, make_layout(make_shape(seq_len, seq_len_kv), make_stride(seq_len_kv, _1{})));

  auto& queue = vllm::xpu::vllmGetQueue();
  // get the proper dpas mma atom, m=8 is fixed on xe2.
  // the mma atom shape is m8n16k16
  auto op = XE_DPAS_TT<8, float, bfloat16_t>{};

  using MqaPolicy = w8a8_policy_m_32;
  using WGTile = typename MqaPolicy::WGTile; // m32n64k32
  using SGLayout = typename MqaPolicy::SGLayout; // m1n4k1

  // Work group tile shape: m32n64k32
  // TiledMMA Subgroup layout: 1x4x1​
  // MMA atom shape: m8n16k16​
  // Totally MMA atom calls per subgroup: 4 * 1 * 2 = 8​
  // - Repeat on M dim: 32 / (8*1) = 4 times​
  // - Repeat on N dim: 64 / (16*4) = 1 times​
  // - Repeat on K dim: 32 / (1*16) = 2 times
  using MMA = typename TiledMMAHelper<
      MMA_Atom<decltype(op)>,
      Layout<WGTile>,
      SGLayout>::TiledMMA;
  auto mma = MMA{};

  // ------
  // the output tile sg tiling look like:
  // | sg0: m0~7 n0~15   | sg1: m0~7 n16~31   | sg2: m0~7 n32~47   | sg3: m0~7 n48~63   |, iter 0
  // | sg0: m8~15 n0~15  | sg1: m8~15 n16~31  | sg2: m8~15 n32~47  | sg3: m8~15 n48~63  |, iter 1
  // | sg0: m16~23 n0~15 | sg1: m16~23 n16~31 | sg2: m16~23 n32~47 | sg3: m16~23 n48~63 |, iter 2
  // | sg0: m24~31 n0~15 | sg1: m24~31 n16~31 | sg2: m24~31 n32~47 | sg3: m24~31 n48~63 |, iter 3
  // inside each sg, the work-items tiling look like (take sg0 as example):
  // | wi0: m0~7 n0 | wi1: m0~7 n1 | wi2: m0~7 n2 | ... | wi15: m0~7 n15 |
  // ------

  constexpr int64_t kBlockHeads = get<0>(typename MqaPolicy::WGTile{}); // 32
  const int64_t kBlockKV = get<1>(typename MqaPolicy::WGTile{}); // 64
  const int64_t mma_k_tile = get<2>(typename MqaPolicy::WGTile{}); // 32
  const int64_t threads_per_wg = size(mma); // 4x16=64

  // each wg compute one output tile of shape (1, kBlockKV) in the following way:
  // 1. load a q tile of shape (1, kBlockHeads, head_dim) from global memory to registers
  // 2. load a kv tile of shape (kBlockKV, head_dim) from global memory to registers
  // 3. compute q@kv^T to get a tile of shape (1, kBlockHeads, kBlockKV)
  // 4. scale the output tile with kv_scales
  // 5. apply relu
  // 6. broadcast multiply the output tile with weights of shape (1, kBlockHeads)
  // 7. reduce the output tile along the head_dim dimension to get a tile of shape (1, kBlockKV)

  const sycl::range<2> local_range(1, threads_per_wg);
  const sycl::range<2> global_range(
    seq_len, 
    cute::ceil_div(seq_len_kv, static_cast<int64_t>(kBlockKV)) * threads_per_wg
  );

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props {
    syclex::sub_group_size<16>,
    intelex::grf_size<256>
  };

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<2>(global_range, local_range), kernel_props,
        [=](sycl::nd_item<2> item) {

      // Get workgroup and local IDs
      auto q_token_idx = int(item.get_group(0));
      auto kv_block_idx = int(item.get_group(1));
      auto local_id = int(item.get_local_id(1));

      // get tensor slice for current query sequence
      Tensor curr_q_tensor = q_tensor(q_token_idx, _, _); // (num_heads, head_dim)
      Tensor curr_weights_tensor = weights_tensor(q_token_idx, _); // (num_heads)
      Tensor curr_out_tensor = out_tensor(q_token_idx, _); // (seq_len_kv)
      int32_t ks = ks_tensor(q_token_idx);
      int32_t ke = ke_tensor(q_token_idx);

      // Create proxy coordinate tensors for each global tensor
      Tensor cQ = make_identity_tensor(curr_q_tensor.shape());   // (num_heads, head_dim)
      Tensor cKV = make_identity_tensor(kv_tensor.shape());   // (seq_len_kv, head_dim)
      Tensor cScales = make_identity_tensor(scales_tensor.shape()); // (seq_len_kv)
      Tensor cWeights = make_identity_tensor(curr_weights_tensor.shape());   // (num_heads)
      Tensor cOut = make_identity_tensor(curr_out_tensor.shape());   // (seq_len_kv)

      // Split GEMM into workgroup tiles, and identify our workgroup's tile (wg_coord)
      auto wg_tile = mma.tile_mnk(); // (m,n,k) = (32,64,32)
      auto wg_coord = make_coord(q_token_idx, kv_block_idx, 0);

      // Local tiles for the current workgroup
      Tensor gQ = local_tile(cQ, select<0,2>(wg_tile), make_coord(_, _));  // (kBlockHeads, mma_k_tile, h, k)
      Tensor gKV = local_tile(cKV, select<1,2>(wg_tile), make_coord(kv_block_idx,_));  // (kBlockKV, mma_k_tile, k)
      Tensor gScales = local_tile(cScales, select<1>(wg_tile), make_coord(kv_block_idx)); // (kBlockKV)
      Tensor gWeights = local_tile(cWeights, select<0>(wg_tile), make_coord(_)); // (kBlockHeads, h)
      Tensor gOut = local_tile(cOut, select<1>(wg_tile), make_coord(kv_block_idx)); // (kBlockKV)

      // Create block 2D TiledCopies
      auto copy_q = make_block_2d_copy_A(mma, curr_q_tensor);
      auto copy_kv = make_block_2d_copy_B(mma, kv_tensor);

      // Slice TiledCopy/TiledMMA operations to thread (work-item) level
      auto thr_mma    =    mma.get_slice(local_id);
      auto thr_copy_q = copy_q.get_slice(local_id);
      auto thr_copy_kv = copy_kv.get_slice(local_id);

      // Register fragments for MMA
      auto tCrA = thr_mma.partition_sg_fragment_A(gQ(_,_,0,0));
      auto tCrB = thr_mma.partition_sg_fragment_B(gKV(_,_,0));

      // Register fragments for copies
      auto tArA = thr_copy_q.partition_sg_fragment_D(gQ(_,_,0,0));
      auto tBrB = thr_copy_kv.partition_sg_fragment_D(gKV(_,_,0));

      // Partition global tensor (proxies) for copies
      Tensor tAgA = thr_copy_q.partition_S(gQ); 
      Tensor tBgB = thr_copy_kv.partition_S(gKV);

      // Partition C
      Tensor tCrC = partition_fragment_C(mma, select<0,1>(wg_tile)/*m32n64*/);

      // Create prefetch TiledCopy instances
      auto prefetch_q = make_block_2d_prefetch(copy_q);
      auto prefetch_kv = make_block_2d_prefetch(copy_kv);

      auto thr_prefetch_q = prefetch_q.get_slice(local_id);
      auto thr_prefetch_kv = prefetch_kv.get_slice(local_id);

      // Partition global tensor (proxies) for prefetch
      // TODO: why the prefetch partitioned shape is different from the copy?
      auto pAgA = thr_prefetch_q.partition_S(gQ);
      auto pBgB = thr_prefetch_kv.partition_S(gKV);

      // Prefetch distance, in units of k tiles
      const int prefetch_dist = 3;

      int head_tile_count = ceil_div(num_heads, kBlockHeads);
      int k_tile_count = ceil_div(head_dim, mma_k_tile);
      // ------
      // Kernel
      // ------

      constexpr int barrier_scope = 2; // workgroup scope barrier

      float output = 0;

      // for loop along num_heads dimension with step size of kBlockHeads to cover the whole num_heads dimension
      for (int64_t head_tile = 0; head_tile < head_tile_count; head_tile++) {
        int k_tile_prefetch = 0;
        clear(tCrC);

        // Warm up loops with prefetch first k tile to L1
        CUTE_UNROLL
        for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
          prefetch(prefetch_q, pAgA(_,_,_,head_tile,k_tile_prefetch));
          prefetch(prefetch_kv, pBgB(_,_,_,k_tile_prefetch));
        }

        // for loop along head_dim dimension with step size of mma_k_tile to cover the whole head_dim dimension
        for (int64_t k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
          // Split barrier keeping threads loosely together 
          barrier_arrive(barrier_scope);

          // load q and kv tiles from global memory to registers
          copy(copy_q, tAgA(_,_,_,head_tile,k_tile), tArA);
          copy(copy_kv, tBgB(_,_,_,k_tile), tBrB);

          // prefetch next k tiles to L1
          if (k_tile_prefetch < k_tile_count) {
            prefetch(prefetch_q, pAgA(_,_,_,head_tile,k_tile_prefetch));
            prefetch(prefetch_kv, pBgB(_,_,_,k_tile_prefetch));
          }

          // Shuffle data from copy fragments to MMA fragments
          // and convert from fp8 to bf16
          reorder(tArA, tCrA);
          reorder(tBrB, tCrB);

          // Accumulate C += A * B
          gemm(mma, tCrA, tCrB, tCrC);
          barrier_wait(barrier_scope);
        }

        // load and mul scales
        // Note: now each wi own one column of the output tile, so we need to
        // read the corresponding scale for that column and mul it with the
        // output tile column
        int64_t kv_index = kv_block_idx * kBlockKV + local_id;
        float scale = kv_index < seq_len_kv ? scales_tensor(kv_index) : 0.0f;
        CUTE_UNROLL
        for (int i = 0; i < size(tCrC); i++) {
          tCrC[i] = tCrC[i] * scale;
        }

        // relu
        CUTE_UNROLL
        for (int i = 0; i < size(tCrC); i++) {
          tCrC[i] = tCrC[i] > 0 ? tCrC[i] : 0;
        }

        // load and mul weights
        float weight[kBlockHeads];
        CUTE_UNROLL
        for (int i = 0; i < kBlockHeads; i++) {
          weight[i] = curr_weights_tensor(head_tile * kBlockHeads + i);
        }
        CUTE_UNROLL
        for (int i = 0; i < size<0>(tCrC); i++) {
          for (int j = 0; j < size<1>(tCrC); j++) {
            int wei_index = j*size<0>(tCrC) + i;
            tCrC(i,j,0) = tCrC(i,j,0) * weight[wei_index];
          }
        }

        // reduce and accumlate along head_dim
        CUTE_UNROLL
        for (int i = 0; i < size(tCrC); i++) {
          output += tCrC[i];
        }
      }

      // write the output back to global memory
      int64_t kv_index = kv_block_idx * kBlockKV + local_id;
      if (kv_index >= seq_len_kv) {
        return;
      }

      curr_out_tensor(kv_index) =
          (kv_index >= ks && kv_index < ke) ? output : neg_inf;
    });
  });

  return logits;
}

torch::Tensor fp8_paged_mqa_logits_xe2(
    const torch::Tensor& q, // (batch_size, next_n, heads, index_dim), dtype float8
    const torch::Tensor& kv, // (num_blocks, block_size, 1, index_dim), dtype float8, only contiguous in each block
    const torch::Tensor& kv_scales, // (num_blocks, block_size, 1, 1), dtype float32, only contiguous in each block
    const torch::Tensor& weights, // (batch_size * next_n, heads), dtype float32
    const torch::Tensor& context_lens, // (batch_size), dtype int32
    const torch::Tensor& block_tables, // (batch_size, max_blocks), dtype int32
    int64_t batch_size,
    int64_t next_n,
    int64_t heads,
    int64_t index_dim,
    int64_t num_blocks,
    int64_t block_size,
    int64_t max_blocks,
    int64_t max_model_len) {
  const float neg_inf = -std::numeric_limits<float>::infinity();

  auto logits = torch::full(
      {batch_size * next_n, max_model_len},
      neg_inf,
      torch::dtype(torch::kFloat).device(q.device()).requires_grad(false));

  const auto* q_ptr = reinterpret_cast<const float_e4m3_t*>(q.data_ptr());
  const auto* kv_ptr = reinterpret_cast<const float_e4m3_t*>(kv.data_ptr());
  const float* scale_ptr = kv_scales.data_ptr<float>();
  const float* weights_ptr = weights.data_ptr<float>();
  const int32_t* context_ptr = context_lens.data_ptr<int32_t>();
  const int32_t* block_tables_ptr = block_tables.data_ptr<int32_t>();
  float* out_ptr = logits.data_ptr<float>();

  // kv and kv_scales are non-contiguous, so we need to use the strides to
  // access them correctly
  const int32_t kv_stride0 = kv.stride(0); // block_size * (index_dim + 4)
  const int32_t kv_stride1 = kv.stride(1); // index_dim
  const int32_t kv_stride3 = kv.stride(3); // 1
  const int32_t scale_stride0 = kv_scales.stride(0); // block_size * (index_dim + 4)
  const int32_t scale_stride1 = kv_scales.stride(1); // 1

  TORCH_CHECK(kv_stride1 == index_dim, "kv index_dim stride mismatch");
  TORCH_CHECK(kv_stride3 == 1, "kv last dim stride mismatch");
  TORCH_CHECK(scale_stride1 == 1, "kv_scales last dim stride mismatch");

  auto q_tensor = make_tensor(
      q_ptr,
      make_layout(
          make_shape(batch_size, next_n, heads, index_dim),
          make_stride(next_n * heads * index_dim, heads * index_dim, index_dim, _1{})));
  auto kv_tensor = make_tensor(
      kv_ptr,
      make_layout(
          make_shape(num_blocks, block_size, index_dim),
          make_stride(kv_stride0, index_dim, _1{})));
  auto scales_tensor = make_tensor(
      scale_ptr,
      make_layout(
          make_shape(num_blocks, block_size),
          make_stride(scale_stride0, _1{})));
  auto weights_tensor = make_tensor(
      weights_ptr,
      make_layout(
          make_shape(batch_size, next_n, heads),
          make_stride(next_n * heads, heads, _1{})));
  auto context_tensor = make_tensor(
      context_ptr,
      make_layout(make_shape(batch_size), make_stride(_1{})));
  auto block_tables_tensor = make_tensor(
      block_tables_ptr,
      make_layout(make_shape(batch_size, max_blocks), make_stride(max_blocks, _1{})));
  auto out_tensor = make_tensor(
      out_ptr,
      make_layout(
          make_shape(batch_size, next_n, max_model_len),
          make_stride(next_n * max_model_len, max_model_len, _1{})));

  auto& queue = vllm::xpu::vllmGetQueue();

  auto op = XE_DPAS_TT<8, float, bfloat16_t>{};
  using MqaPolicy = w8a8_policy_m_32;
  using WGTile = typename MqaPolicy::WGTile;
  using SGLayout = typename MqaPolicy::SGLayout;
  using MMA = typename TiledMMAHelper<
      MMA_Atom<decltype(op)>,
      Layout<WGTile>,
      SGLayout>::TiledMMA;
  auto mma = MMA{};

  constexpr int64_t kBlockHeads = get<0>(typename MqaPolicy::WGTile{}); // 32
  constexpr int64_t kBlockKV = get<1>(typename MqaPolicy::WGTile{}); // 64
  constexpr int64_t mma_k_tile = get<2>(typename MqaPolicy::WGTile{}); // 32
  constexpr int64_t threads_per_wg = size(mma);

  TORCH_CHECK(
      block_size == kBlockKV,
      "fp8_paged_mqa_logits_xe2 currently only supports block_size == ",
      kBlockKV,
      ", but got ",
      block_size);

  const int64_t block_count = cute::ceil_div(max_model_len, block_size);
  const sycl::range<2> local_range(1, threads_per_wg);
  const sycl::range<2> global_range(batch_size, block_count * threads_per_wg);

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props {
    syclex::sub_group_size<16>,
    intelex::grf_size<256>
  };

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<2>(global_range, local_range), kernel_props,
        [=](sycl::nd_item<2> item) {
      auto batch_idx = int(item.get_group(0));
      auto logical_block_idx = int(item.get_group(1));
      auto local_id = int(item.get_local_id(1));

      if (logical_block_idx >= max_blocks) {
        return;
      }

      int32_t context_len = context_tensor(batch_idx);
      int64_t block_start = static_cast<int64_t>(logical_block_idx) * block_size;
      if (block_start >= context_len) {
        return;
      }

      int32_t physical_block = block_tables_tensor(batch_idx, logical_block_idx);
      if (physical_block < 0 || physical_block >= num_blocks) {
        return;
      }

      int64_t block_end = cute::min(block_start + block_size, context_len);
      int64_t actual_block_size = block_end - block_start;

      Tensor curr_kv_tensor = kv_tensor(physical_block, _, _);           // (block_size, index_dim)
      Tensor curr_scales_tensor = scales_tensor(physical_block, _);       // (block_size)

      Tensor cKV = make_identity_tensor(curr_kv_tensor.shape()); // (block_size, index_dim)

      auto wg_tile = mma.tile_mnk(); // m32n64k32
      Tensor gKV = local_tile(cKV, select<1,2>(wg_tile), make_coord(0, _)); // (kBlockKV, mma_k_tile, k)

      auto copy_kv = make_block_2d_copy_B(mma, curr_kv_tensor);
      auto prefetch_kv = make_block_2d_prefetch(copy_kv);

      auto thr_mma = mma.get_slice(local_id);
      auto thr_copy_kv = copy_kv.get_slice(local_id);
      auto thr_prefetch_kv = prefetch_kv.get_slice(local_id);

      auto tCrB = thr_mma.partition_sg_fragment_B(gKV(_, _, 0));
      auto tBrB = thr_copy_kv.partition_sg_fragment_D(gKV(_, _, 0));
      Tensor tBgB = thr_copy_kv.partition_S(gKV);
      auto pBgB = thr_prefetch_kv.partition_S(gKV);

      int head_tile_count = ceil_div(heads, kBlockHeads);
      int k_tile_count = ceil_div(index_dim, mma_k_tile);
      const int prefetch_dist = 3;
      constexpr int barrier_scope = 2;

      for (int64_t q_token_id = 0; q_token_id < next_n; q_token_id++) {
        int64_t q_offset = static_cast<int64_t>(context_len) - next_n + q_token_id;
        if (q_offset < block_start) {
          continue;
        }

        Tensor curr_q_tensor = q_tensor(batch_idx, q_token_id, _, _);     // (heads, index_dim)
        Tensor curr_weights_tensor = weights_tensor(batch_idx, q_token_id, _); // (heads)
        Tensor curr_out_tensor = out_tensor(batch_idx, q_token_id, _);     // (max_model_len)

        Tensor cQ = make_identity_tensor(curr_q_tensor.shape());
        Tensor gQ = local_tile(cQ, select<0,2>(wg_tile), make_coord(_, _));

        auto copy_q = make_block_2d_copy_A(mma, curr_q_tensor);
        auto prefetch_q = make_block_2d_prefetch(copy_q);

        auto thr_copy_q = copy_q.get_slice(local_id);
        auto thr_prefetch_q = prefetch_q.get_slice(local_id);

        auto tCrA = thr_mma.partition_sg_fragment_A(gQ(_, _, 0, 0));
        auto tArA = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0, 0));
        Tensor tAgA = thr_copy_q.partition_S(gQ);
        auto pAgA = thr_prefetch_q.partition_S(gQ);

        Tensor tCrC = partition_fragment_C(mma, select<0,1>(wg_tile));

        float output = 0.0f;

        for (int64_t head_tile = 0; head_tile < head_tile_count; head_tile++) {
          int k_tile_prefetch = 0;
          clear(tCrC);

          CUTE_UNROLL
          for (; k_tile_prefetch < prefetch_dist && k_tile_prefetch < k_tile_count;
               k_tile_prefetch++) {
            prefetch(prefetch_q, pAgA(_, _, _, head_tile, k_tile_prefetch));
            prefetch(prefetch_kv, pBgB(_, _, _, k_tile_prefetch));
          }

          for (int64_t k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
            barrier_arrive(barrier_scope);

            copy(copy_q, tAgA(_, _, _, head_tile, k_tile), tArA);
            copy(copy_kv, tBgB(_, _, _, k_tile), tBrB);

            if (k_tile_prefetch < k_tile_count) {
              prefetch(prefetch_q, pAgA(_, _, _, head_tile, k_tile_prefetch));
              prefetch(prefetch_kv, pBgB(_, _, _, k_tile_prefetch));
            }

            reorder(tArA, tCrA);
            reorder(tBrB, tCrB);

            gemm(mma, tCrA, tCrB, tCrC);
            barrier_wait(barrier_scope);
          }

          int64_t block_offset = local_id;
          float scale = block_offset < actual_block_size ? curr_scales_tensor(block_offset) : 0.0f;

          CUTE_UNROLL
          for (int i = 0; i < size(tCrC); i++) {
            tCrC[i] = tCrC[i] * scale;
          }

          CUTE_UNROLL
          for (int i = 0; i < size(tCrC); i++) {
            tCrC[i] = tCrC[i] > 0 ? tCrC[i] : 0;
          }

          float weight[kBlockHeads];
          CUTE_UNROLL
          for (int i = 0; i < kBlockHeads; i++) {
            int64_t head_idx = head_tile * kBlockHeads + i;
            weight[i] = head_idx < heads ? curr_weights_tensor(head_idx) : 0.0f;
          }

          CUTE_UNROLL
          for (int i = 0; i < size<0>(tCrC); i++) {
            for (int j = 0; j < size<1>(tCrC); j++) {
              int wei_index = j * size<0>(tCrC) + i;
              tCrC(i, j, 0) = tCrC(i, j, 0) * weight[wei_index];
            }
          }

          CUTE_UNROLL
          for (int i = 0; i < size(tCrC); i++) {
            output += tCrC[i];
          }
        }

        int64_t kv_index = block_start + local_id;
        if (local_id >= actual_block_size || kv_index >= max_model_len) {
          continue;
        }
        if (kv_index <= q_offset) {
          curr_out_tensor(kv_index) = output;
        }
      }
    
    });
  });

  return logits;
}
