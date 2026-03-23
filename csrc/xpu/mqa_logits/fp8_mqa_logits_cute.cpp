#include "csrc/utils.h"
#include "xpu/ops.h"

#include <ATen/ATen.h>
#include <cute/tensor.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <sycl/sycl.hpp>

#include <limits>

using namespace cute;

namespace {

constexpr int kBlockHeads = 64;
constexpr int kBlockKv = 64;
constexpr int kHeadDim = 128;
constexpr int kBlockK = 32;

template <typename T, size_t = 0>
struct is_complete : std::false_type {};

template <typename T>
struct is_complete<T, 0 * sizeof(T)> : std::true_type {};

template <typename T>
static constexpr bool is_complete_v = is_complete<T>::value;

template <typename TA, typename TB, typename TC>
auto choose_mma_op() {
  if constexpr (is_complete_v<XE_DPAS_TT<8, TC, TA, TB>>) {
    return XE_DPAS_TT<8, TC, TA, TB>{};
  } else {
    return XE_DPAS_TT<8, float, half_t>{};
  }
}

auto choose_tiled_mma() {
  using ElementA = cutlass::float_e4m3_t;
  using ElementB = cutlass::float_e4m3_t;
  using ElementAccumulator = float;

  auto op = choose_mma_op<ElementA, ElementB, ElementAccumulator>();

  using WGTile = Shape<_64, _64, _32>;
  using SGLayout = Layout<Shape<_2, _2, _1>, Stride<_2, _1, _0>>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<decltype(op)>,
      Layout<WGTile>,
      SGLayout>::TiledMMA;

  return TiledMma{};
}

template <class TiledMma>
CUTE_DEVICE void mqa_logits_tile_device(
    const cutlass::float_e4m3_t* q_ptr,
    const cutlass::float_e4m3_t* kv_ptr,
    const float* kv_scale_ptr,
    const float* weights_ptr,
    const int32_t* cu_seqlen_ks_ptr,
    const int32_t* cu_seqlen_ke_ptr,
    float* logits_ptr,
    int64_t num_heads,
    int64_t seq_len_kv,
    int64_t seq_len_kv_padded,
    TiledMma const& mma) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
  int q_token_idx = int(item.get_group(1));
  int kv_block_idx = int(item.get_group(0));
  int local_id = int(item.get_local_id(0));

  int kv_start = kv_block_idx * kBlockKv;

  int32_t ks = cu_seqlen_ks_ptr[q_token_idx];
  int32_t ke = cu_seqlen_ke_ptr[q_token_idx];

  if (kv_start >= ke || kv_start + kBlockKv <= ks) {
    for (int col = local_id; col < kBlockKv; col += item.get_local_range(0)) {
      int64_t kv_idx = kv_start + col;
      if (kv_idx < seq_len_kv_padded) {
        logits_ptr[q_token_idx * seq_len_kv_padded + kv_idx] =
            -std::numeric_limits<float>::infinity();
      }
    }
    return;
  }

  for (int col = local_id; col < kBlockKv; col += item.get_local_range(0)) {
    int64_t kv_idx = kv_start + col;
    if (kv_idx < seq_len_kv_padded) {
      bool kv_in_seq = kv_idx < seq_len_kv;
      bool kv_in_mask = kv_idx >= ks && kv_idx < ke;
      logits_ptr[q_token_idx * seq_len_kv_padded + kv_idx] =
          (kv_in_seq && kv_in_mask) ? 0.0f
                                   : -std::numeric_limits<float>::infinity();
    }
  }
  item.barrier(sycl::access::fence_space::global_and_local);

  const cutlass::float_e4m3_t* q_base = q_ptr + q_token_idx * num_heads * kHeadDim;
  const cutlass::float_e4m3_t* kv_base = kv_ptr + kv_start * kHeadDim;

  Tensor A = make_tensor(
      make_gmem_ptr(q_base),
      make_shape(Int<kBlockHeads>{}, Int<kHeadDim>{}),
      make_stride(Int<kHeadDim>{}, Int<1>{}));
  Tensor B = make_tensor(
      make_gmem_ptr(kv_base),
      make_shape(Int<kBlockKv>{}, Int<kHeadDim>{}),
      make_stride(Int<kHeadDim>{}, Int<1>{}));

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());
  Tensor cC = make_identity_tensor(make_shape(Int<kBlockHeads>{}, Int<kBlockKv>{}));

  auto wg_tile = mma.tile_mnk();

  Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(_0{}, _));
  Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(_0{}, _));
  Tensor gC = cC;

  auto copy_a = make_block_2d_copy_A(mma, A);
  auto copy_b = make_block_2d_copy_B(mma, B);

  auto thr_mma = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);

  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  Tensor tCrC = partition_fragment_C(mma, make_shape(Int<kBlockHeads>{}, Int<kBlockKv>{}));
  Tensor tCcC = thr_mma.partition_C(gC);

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_a = prefetch_a.get_slice(local_id);
  auto thr_prefetch_b = prefetch_b.get_slice(local_id);

  auto pAgA = thr_prefetch_a.partition_S(gA);
  auto pBgB = thr_prefetch_b.partition_S(gB);

  constexpr int barrier_scope = 2;
  constexpr int prefetch_dist = 2;

  int k_tile_count = ceil_div(kHeadDim, int(get<2>(wg_tile)));
  int k_tile_prefetch = 0;

  clear(tCrC);

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; ++k_tile_prefetch) {
    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
  }

  for (int k_tile = 0; k_tile < k_tile_count; ++k_tile, ++k_tile_prefetch) {
    barrier_arrive(barrier_scope);

    copy(copy_a, tAgA(_, _, _, k_tile), tArA);
    copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

    prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));

    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    gemm(mma, tCrA, tCrB, tCrC);

    barrier_wait(barrier_scope);
  }

  for (int i = 0; i < size(tCrC); ++i) {
    auto coord = tCcC(i);
    int h = int(get<0>(coord));
    int n = int(get<1>(coord));

    if (h >= num_heads || n >= kBlockKv) {
      continue;
    }

    int64_t kv_idx = kv_start + n;
    bool kv_in_seq = kv_idx < seq_len_kv;
    bool kv_in_mask = kv_idx >= ks && kv_idx < ke;
    if (!kv_in_seq || !kv_in_mask) {
      continue;
    }

    float acc = float(tCrC(i));
    float scaled = acc * kv_scale_ptr[kv_idx];
    float relu = sycl::fmax(scaled, 0.0f);
    float weighted = relu * weights_ptr[q_token_idx * num_heads + h];

    auto& dst = logits_ptr[q_token_idx * seq_len_kv_padded + kv_idx];
    sycl::atomic_ref<
        float,
        sycl::memory_order::relaxed,
        sycl::memory_scope::device,
        sycl::access::address_space::global_space>
        out_atomic(dst);
    out_atomic.fetch_add(weighted);
  }
}

template <class TiledMma>
class Fp8MqaLogitsKernel;

template <class TiledMma>
struct Fp8MqaLogitsLaunchFunctor {
  const cutlass::float_e4m3_t* q_ptr;
  const cutlass::float_e4m3_t* kv_ptr;
  const float* kv_scale_ptr;
  const float* weights_ptr;
  const int32_t* ks_ptr;
  const int32_t* ke_ptr;
  float* logits_ptr;
  int64_t num_heads;
  int64_t seq_len_kv;
  int64_t seq_len_kv_padded;
  TiledMma mma;

  void operator()(sycl::nd_item<2>) const {
    mqa_logits_tile_device(
        q_ptr,
        kv_ptr,
        kv_scale_ptr,
        weights_ptr,
        ks_ptr,
        ke_ptr,
        logits_ptr,
        num_heads,
        seq_len_kv,
        seq_len_kv_padded,
        mma);
  }

  static constexpr auto get(sycl::ext::oneapi::experimental::properties_tag) {
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;
    return syclex::properties{syclex::sub_group_size<16>, intelex::grf_size<256>};
  }
};

}  // namespace

torch::Tensor fp8_mqa_logits_cute(
    torch::Tensor q,
    torch::Tensor kv,
    torch::Tensor kv_scale,
    torch::Tensor weights,
    torch::Tensor cu_seqlen_ks,
    torch::Tensor cu_seqlen_ke) {
  CHECK_DEVICE(q);
  CHECK_DEVICE(kv);
  CHECK_DEVICE(kv_scale);
  CHECK_DEVICE(weights);
  CHECK_DEVICE(cu_seqlen_ks);
  CHECK_DEVICE(cu_seqlen_ke);

  CHECK_CONTIGUOUS(q);
  CHECK_CONTIGUOUS(kv);
  CHECK_CONTIGUOUS(kv_scale);
  CHECK_CONTIGUOUS(weights);
  CHECK_CONTIGUOUS(cu_seqlen_ks);
  CHECK_CONTIGUOUS(cu_seqlen_ke);

  TORCH_CHECK(q.dim() == 3, "q must be [seq_len, num_heads, head_dim]");
  TORCH_CHECK(kv.dim() == 2, "kv must be [seq_len_kv, head_dim]");
  TORCH_CHECK(weights.dim() == 2, "weights must be [seq_len, num_heads]");
  TORCH_CHECK(cu_seqlen_ks.dim() == 1, "cu_seqlen_ks must be [seq_len]");
  TORCH_CHECK(cu_seqlen_ke.dim() == 1, "cu_seqlen_ke must be [seq_len]");

  TORCH_CHECK(
      q.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "q must be torch.float8_e4m3fn");
  TORCH_CHECK(
      kv.scalar_type() == at::ScalarType::Float8_e4m3fn,
      "kv must be torch.float8_e4m3fn");
  TORCH_CHECK(
      kv_scale.scalar_type() == at::ScalarType::Float,
      "kv_scale must be torch.float32");
  TORCH_CHECK(
      weights.scalar_type() == at::ScalarType::Float,
      "weights must be torch.float32");
  TORCH_CHECK(
      cu_seqlen_ks.scalar_type() == at::ScalarType::Int,
      "cu_seqlen_ks must be torch.int32");
  TORCH_CHECK(
      cu_seqlen_ke.scalar_type() == at::ScalarType::Int,
      "cu_seqlen_ke must be torch.int32");

  int64_t seq_len = q.size(0);
  int64_t num_heads = q.size(1);
  int64_t head_dim = q.size(2);
  int64_t seq_len_kv = kv.size(0);

  TORCH_CHECK(weights.size(0) == seq_len, "weights.shape[0] must equal seq_len");
  TORCH_CHECK(weights.size(1) == num_heads, "weights.shape[1] must equal num_heads");
  TORCH_CHECK(kv.size(1) == head_dim, "kv.shape[1] must equal head_dim");
  TORCH_CHECK(kv_scale.size(0) == seq_len_kv, "kv_scale.shape[0] must equal seq_len_kv");
  TORCH_CHECK(cu_seqlen_ks.size(0) == seq_len, "cu_seqlen_ks.shape[0] must equal seq_len");
  TORCH_CHECK(cu_seqlen_ke.size(0) == seq_len, "cu_seqlen_ke.shape[0] must equal seq_len");

  TORCH_CHECK(
      num_heads == kBlockHeads,
      "Only num_heads == ",
      kBlockHeads,
      " is currently supported, but got ",
      num_heads);
  TORCH_CHECK(
      head_dim == kHeadDim,
      "Only head_dim == ",
      kHeadDim,
      " is currently supported, but got ",
      head_dim);

  int64_t seq_len_kv_padded = ((seq_len_kv + kBlockKv - 1) / kBlockKv) * kBlockKv;
  torch::Tensor kv_padded = kv;
  torch::Tensor kv_scale_padded = kv_scale;

  if (seq_len_kv_padded != seq_len_kv) {
    kv_padded = torch::zeros({seq_len_kv_padded, head_dim}, kv.options());
    kv_padded.narrow(0, 0, seq_len_kv).copy_(kv);

    kv_scale_padded = torch::ones({seq_len_kv_padded}, kv_scale.options());
    kv_scale_padded.narrow(0, 0, seq_len_kv).copy_(kv_scale);
  }

  auto logits_padded = torch::zeros(
      {seq_len, seq_len_kv_padded},
      q.options().dtype(torch::kFloat));

  auto* q_ptr = reinterpret_cast<const cutlass::float_e4m3_t*>(
      q.data_ptr<c10::Float8_e4m3fn>());
  auto* kv_ptr = reinterpret_cast<const cutlass::float_e4m3_t*>(
      kv_padded.data_ptr<c10::Float8_e4m3fn>());
  auto* kv_scale_ptr = kv_scale_padded.data_ptr<float>();
  auto* weights_ptr = weights.data_ptr<float>();
  auto* ks_ptr = cu_seqlen_ks.data_ptr<int32_t>();
  auto* ke_ptr = cu_seqlen_ke.data_ptr<int32_t>();
  auto* logits_ptr = logits_padded.data_ptr<float>();

  auto mma = choose_tiled_mma();

  sycl::range<2> local = {static_cast<size_t>(size(mma)), 1};
  sycl::range<2> global = {
      local[0] * static_cast<size_t>(seq_len_kv_padded / kBlockKv),
      local[1] * static_cast<size_t>(seq_len)};

  auto& queue = vllm::xpu::vllmGetQueue(q.get_device());

    namespace syclex = sycl::ext::oneapi::experimental;

    syclex::nd_launch<Fp8MqaLogitsKernel<decltype(mma)>>(
      queue,
      sycl::nd_range<2>(global, local),
      Fp8MqaLogitsLaunchFunctor<decltype(mma)>{
        q_ptr,
        kv_ptr,
        kv_scale_ptr,
        weights_ptr,
        ks_ptr,
        ke_ptr,
        logits_ptr,
        num_heads,
        seq_len_kv,
        seq_len_kv_padded,
        mma});

  queue.wait_and_throw();

  if (seq_len_kv_padded == seq_len_kv) {
    return logits_padded;
  }

  return logits_padded.narrow(1, 0, seq_len_kv).contiguous();
}
