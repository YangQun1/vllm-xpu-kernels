// DeepSeek V4 MHC pre-processing kernel using sycl-tla cute API with DPAS tf32.
//
// Three-path design with Stage 1 dispatch:
//   Stage 1 vector (launch_mhc_pre_stage1_vector): scalar dot-product loop
//     - Best for small M (≤ DISPATCH_THRESHOLD): vectorized K-loop with
//       workgroup reduction, BLOCK_M=2, 256 threads, SG_SIZE=16
//     - Produces rms_mixes[M,24] fp32 in global memory
//
//   Stage 1 matrix (inline CuTe DPAS code): cute GEMM + sqrsum + RMS-norm
//     - Best for large M: tiled GEMM via DPAS tf32
//     - Uses GemmPolicy<7>: BLK_M=64, BLK_N=32, BLK_K=32, 4x2 SGs, pf=3
//     - Produces rms_mixes[M,24] fp32 in global memory
//
//   Stage 2 (launch_mhc_pre_stage2): post-processing (1 token per WG)
//     - Reads rms_mixes[M,24] + residual[M,HC,H]
//     - SG0 computes sigmoid (pre_mix, post_mix), Sinkhorn (comb_mix)
//     - All 256 threads compute layer_input = sum(pre_mix * residual, dim=HC)
//
// Data flow:
//   residual[M,HC,H] bf16 --+-- Stage1 (vector|matrix) --> rms_mixes[M,24] fp32
//   fn[24,K] fp32 ----------+                                  |
//                                                              v
//   residual[M,HC,H] bf16 ---- Stage2 (post-processing) --> post_mix, comb_mix, layer_input
//
// residual DRAM reads: 2x (same as original fused kernel)

#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "../../utils.h"

#include <cute/tensor.hpp>
#include <cute/algorithm/subgroup_algorithms.hpp>
#include <cute/numeric/arithmetic_tuple.hpp>

#if defined(__clang__)
  #pragma clang diagnostic ignored "-Wpass-failed"
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
  #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

using namespace cute;

namespace {

struct GemmRmsPolicy {
    using WGTile   = Shape<_64, _32, _32>;
    using SGLayout = Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>;
    static constexpr int prefetch_dist = 3;
};

template <int PrefetchDist,
          class ATensor, class BTensor, class CTensor,
          class TiledMMA>
void mhc_pre_stage1_device(
    ATensor const& A,
    BTensor const& B,
    CTensor& C,
    TiledMMA const& mma,
    float rms_eps) {
    auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
    auto wg_m = int(item.get_group(1));
    auto wg_n = int(item.get_group(0));
    auto local_id = int(item.get_local_id(0));

    Tensor cA = make_identity_tensor(A.shape());
    Tensor cB = make_identity_tensor(B.shape());
    Tensor cC = make_identity_tensor(C.shape());

    auto wg_tile = mma.tile_mnk();
    auto wg_coord = make_coord(wg_m, wg_n, 0);

    Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(wg_m, _));
    Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(wg_n, _));
    Tensor gC = local_tile(cC, wg_tile, wg_coord, Step<_1, _1, X>{});

    auto copy_a = make_block_2d_copy_A(mma, A);
    auto copy_b = make_block_2d_copy_B(mma, B);
    auto copy_c = make_block_2d_copy_D(mma, C);

    auto thr_mma = mma.get_slice(local_id);
    auto thr_copy_a = copy_a.get_slice(local_id);
    auto thr_copy_b = copy_b.get_slice(local_id);

    auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));

    auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_, _, 0));
    auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_, _, 0));

    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tBgB = thr_copy_b.partition_S(gB);

    Tensor cTileC = make_identity_tensor(select<0, 1>(wg_tile));
    auto tCrC = thr_mma.partition_sg_fragment_C(cTileC);
    Tensor tCgC = thr_mma.partition_C(gC);

    auto prefetch_a = make_block_2d_prefetch(copy_a);
    auto prefetch_b = make_block_2d_prefetch(copy_b);

    auto thr_prefetch_A = prefetch_a.get_slice(local_id);
    auto thr_prefetch_B = prefetch_b.get_slice(local_id);

    auto pAgA = thr_prefetch_A.partition_S(gA);
    auto pBgB = thr_prefetch_B.partition_S(gB);

    constexpr int prefetch_dist = PrefetchDist;
    int K_int = static_cast<int>(shape<1>(A));

    auto local_sqr_sum = cute::make_subgroup_tensor(
        make_tensor<float>(tArA.layout()), tArA.tv_layout());

    constexpr int barrier_scope = 2;
    int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
    int k_tile_prefetch = 0;

    clear(tCrC);

    CUTE_UNROLL
    for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
        prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
        prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));
    }

    for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
        barrier_arrive(barrier_scope);

        copy(copy_a, tAgA(_, _, _, k_tile), tArA);
        copy(copy_b, tBgB(_, _, _, k_tile), tBrB);

        prefetch(prefetch_a, pAgA(_, _, _, k_tile_prefetch));
        prefetch(prefetch_b, pBgB(_, _, _, k_tile_prefetch));

        reorder(tArA, tCrA);
        reorder(tBrB, tCrB);

        CUTE_UNROLL
        for (int frag_idx = 0; frag_idx < tArA.size(); frag_idx++) {
            float val = static_cast<float>(tArA(frag_idx));
            local_sqr_sum(frag_idx) += val * val;
        }

        gemm(mma, tCrA, tCrB, tCrC);
        barrier_wait(barrier_scope);
    }

    constexpr auto tv_layout = local_sqr_sum.tv_layout();
    constexpr auto coshape = atuple_coshape(tv_layout);
    auto tmp = cute::make_subgroup_tensor(
        make_tensor<float>(local_sqr_sum.layout()),
        make_layout(coshape, make_stride(E<0>{}, E<1>{})));
    reorder(local_sqr_sum, tmp);
    auto reduced = cute::reduce<1>(tmp, sycl::plus<void>{});

    const float K_val = static_cast<float>(K_int);
    for (int i = 0; i < reduced.size(); i++) {
        reduced(i) = sycl::rsqrt(reduced(i) / K_val + rms_eps);
    }

    CUTE_UNROLL
    for (int i = 0; i < tCrC.size(); i++) {
        tCrC(i) *= broadcast<0>(reduced, tCrC, i);
    }

    copy(copy_c, tCrC, tCgC);
}

class MhcPreStage1;

using bf16 = sycl::ext::oneapi::bfloat16;

using vllm::xpu::aligned_vec;

class MhcPreStage1Vector;

static inline float sigmoid(float x) {
    return 1.f / (1.f + sycl::native::exp(-x));
}

class MhcPreStage2;

}  // namespace

sycl::event launch_mhc_pre_stage1_vector(
    sycl::queue& q,
    const bf16* residual,
    const float* fn,
    float* rms_mixes,
    int N,
    int H,
    float rms_eps) {
    constexpr int HC = 4;
    constexpr int BLOCK_M = 2;
    constexpr int BLOCK_N = 12;
    constexpr int VEC_SIZE = 4;
    constexpr int SG_SIZE = 16;
    constexpr int WG_SIZE = 256;
    constexpr int NUM_SG = WG_SIZE / SG_SIZE;
    constexpr int HC_MULT3 = HC * (2 + HC);
    static_assert(HC_MULT3 % BLOCK_N == 0);
    constexpr int NUM_N_BLOCKS = HC_MULT3 / BLOCK_N;
    constexpr int NUM_REDUCE_VALUES = BLOCK_N + 1;
    const int k_size = HC * H;

    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    syclex::properties kernel_props{
        syclex::sub_group_size<SG_SIZE>,
        intelex::grf_size<128>};

    sycl::range<1> global(static_cast<size_t>((N + BLOCK_M - 1) / BLOCK_M) * WG_SIZE);
    sycl::range<1> local(WG_SIZE);

    return q.submit([&](sycl::handler& h) {
        using vec_bf16_t = aligned_vec<bf16, VEC_SIZE>;
        using vec_f32_t = aligned_vec<float, VEC_SIZE>;
        sycl::local_accessor<float, 1> red_scratch(
            NUM_REDUCE_VALUES * NUM_SG * BLOCK_M, h);

        h.parallel_for<MhcPreStage1Vector>(
            sycl::nd_range<1>(global, local), kernel_props,
            [=](sycl::nd_item<1> item) {
                const int wg_id = item.get_group(0);
                const int tid = item.get_local_id(0);
                auto sg = item.get_sub_group();
                const int sg_id = sg.get_group_id()[0];
                const int sg_lane_id = sg.get_local_id()[0];
                const int token_base = wg_id * BLOCK_M;

                #pragma unroll
                for (int block_idx = 0; block_idx < NUM_N_BLOCKS; ++block_idx) {
                    float local_mixes[BLOCK_M][BLOCK_N];
                    float local_sqrsum[BLOCK_M];
                    #pragma unroll
                    for (int t = 0; t < BLOCK_M; ++t) {
                        local_sqrsum[t] = 0.f;
                        #pragma unroll
                        for (int j = 0; j < BLOCK_N; ++j)
                            local_mixes[t][j] = 0.f;
                    }

                    for (int k_idx = tid * VEC_SIZE; k_idx < k_size; k_idx += WG_SIZE * VEC_SIZE) {
                        vec_f32_t fn_tile[BLOCK_N];
                        #pragma unroll
                        for (int j = 0; j < BLOCK_N; ++j) {
                            fn_tile[j] = *reinterpret_cast<const vec_f32_t*>(
                                fn + (block_idx * BLOCK_N + j) * k_size + k_idx);
                        }

                        #pragma unroll
                        for (int t = 0; t < BLOCK_M; ++t) {
                            int token_idx = token_base + t;
                            if (token_idx >= N)
                                break;

                            auto residual_vec = *reinterpret_cast<const vec_bf16_t*>(
                                residual + token_idx * k_size + k_idx);
                            float x_vec[VEC_SIZE];
                            #pragma unroll
                            for (int v = 0; v < VEC_SIZE; ++v)
                                x_vec[v] = float(residual_vec.val[v]);

                            #pragma unroll
                            for (int v = 0; v < VEC_SIZE; ++v)
                                local_sqrsum[t] += x_vec[v] * x_vec[v];

                            #pragma unroll
                            for (int j = 0; j < BLOCK_N; ++j) {
                                float acc = 0.f;
                                #pragma unroll
                                for (int v = 0; v < VEC_SIZE; ++v)
                                    acc += x_vec[v] * fn_tile[j].val[v];
                                local_mixes[t][j] += acc;
                            }
                        }
                    }

                    #pragma unroll
                    for (int t = 0; t < BLOCK_M; ++t) {
                        int token_idx = token_base + t;
                        if (token_idx >= N)
                            break;

                        float* token_scratch = &red_scratch[t * NUM_REDUCE_VALUES * NUM_SG];
                        float sg_sum = sycl::reduce_over_group(sg, local_sqrsum[t], sycl::plus<float>());
                        if (sg_lane_id == 0)
                            token_scratch[sg_id * NUM_REDUCE_VALUES] = sg_sum;

                        #pragma unroll
                        for (int j = 0; j < BLOCK_N; ++j) {
                            float sg_mix = sycl::reduce_over_group(sg, local_mixes[t][j], sycl::plus<float>());
                            if (sg_lane_id == 0)
                                token_scratch[sg_id * NUM_REDUCE_VALUES + (j + 1)] = sg_mix;
                        }
                        sycl::group_barrier(item.get_group());

                        if (sg_id == 0 && sg_lane_id == 0) {
                            #pragma unroll
                            for (int j = 0; j < NUM_REDUCE_VALUES; ++j) {
                                float s = 0.f;
                                #pragma unroll
                                for (int sg_idx = 0; sg_idx < NUM_SG; ++sg_idx)
                                    s += token_scratch[sg_idx * NUM_REDUCE_VALUES + j];
                                token_scratch[j] = s;
                            }
                        }
                        sycl::group_barrier(item.get_group());

                        float inv_rms = sycl::native::rsqrt(token_scratch[0] / k_size + rms_eps);
                        if (tid < BLOCK_N)
                            rms_mixes[token_idx * HC_MULT3 + block_idx * BLOCK_N + tid] =
                                token_scratch[tid + 1] * inv_rms;
                    }
                }
            });
    });
}

sycl::event launch_mhc_pre_stage2(
    sycl::queue& q,
    const float* rms_mixes,
    const bf16* residual,
    const float* hc_scale,
    const float* hc_base,
    float* post_mix,
    float* comb_mix,
    bf16* layer_input,
    int num_tokens,
    int hidden_size,
    float hc_pre_eps,
    float hc_sinkhorn_eps,
    float hc_post_mult_value,
    int sinkhorn_repeat) {
    static constexpr int SG_SIZE = 16;
    static constexpr int WG_THREADS = 256;
    static constexpr int VEC = 8;
    static constexpr int HC = 4;
    constexpr int HC3 = HC * (2 + HC);
    constexpr int COMB_LANES = HC * HC;
    using vec_bf16_t = aligned_vec<bf16, VEC>;

    sycl::range<1> global(static_cast<size_t>(num_tokens) * WG_THREADS);
    sycl::range<1> local(WG_THREADS);

    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> mixes_slm(HC3, h);

        h.parallel_for<MhcPreStage2>(
            sycl::nd_range<1>(global, local),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const int tok = item.get_group(0);
                const int tid = item.get_local_id(0);
                auto sg = item.get_sub_group();
                int sg_id = sg.get_group_id();

                if (tid < HC3)
                    mixes_slm[tid] = rms_mixes[tok * HC3 + tid];
                sycl::group_barrier(item.get_group());

                if (sg_id == 0) {
                    const int lane = sg.get_local_id()[0];
                    if (lane < HC) {
                        float pre_logits = mixes_slm[lane] * hc_scale[0] + hc_base[lane];
                        float pre_mix_val = sigmoid(pre_logits) + hc_pre_eps;
                        mixes_slm[lane] = pre_mix_val;

                        float post_logits = mixes_slm[lane + HC] * hc_scale[1] + hc_base[lane + HC];
                        post_mix[tok * HC + lane] = sigmoid(post_logits) * hc_post_mult_value;
                    }

                    int comb_idx = lane % COMB_LANES;
                    float comb_logits = mixes_slm[comb_idx + 2 * HC] * hc_scale[2] +
                                        hc_base[comb_idx + 2 * HC];

                    float vmax = comb_logits;
                    #pragma unroll
                    for (int off = 1; off < HC; off <<= 1) {
                        float tmp = sycl::permute_group_by_xor(sg, vmax, off);
                        vmax = sycl::max(vmax, tmp);
                    }
                    float comb_val = sycl::native::exp(comb_logits - vmax);

                    float rsum = comb_val;
                    #pragma unroll
                    for (int off = 1; off < HC; off <<= 1)
                        rsum += sycl::permute_group_by_xor(sg, rsum, off);
                    comb_val *= sycl::native::recip(rsum);
                    comb_val += hc_sinkhorn_eps;

                    #pragma unroll
                    for (int it_sk = 0; it_sk < sinkhorn_repeat; ++it_sk) {
                        float col_sum = comb_val;
                        #pragma unroll
                        for (int off = HC; off < COMB_LANES; off <<= 1)
                            col_sum += sycl::permute_group_by_xor(sg, col_sum, off);
                        comb_val *= sycl::native::recip(col_sum + hc_sinkhorn_eps);

                        if (it_sk < sinkhorn_repeat - 1) {
                            float row_sum = comb_val;
                            #pragma unroll
                            for (int off = 1; off < HC; off <<= 1)
                                row_sum += sycl::permute_group_by_xor(sg, row_sum, off);
                            comb_val *= sycl::native::recip(row_sum + hc_sinkhorn_eps);
                        }
                    }

                    if (lane < COMB_LANES)
                        comb_mix[tok * COMB_LANES + lane] = comb_val;
                }

                sycl::group_barrier(item.get_group());

                float pre_mix[HC];
                #pragma unroll
                for (int m = 0; m < HC; ++m)
                    pre_mix[m] = mixes_slm[m];

                for (int k = tid * VEC; k < hidden_size; k += WG_THREADS * VEC) {
                    float acc[VEC];
                    #pragma unroll
                    for (int v = 0; v < VEC; ++v)
                        acc[v] = 0.f;

                    #pragma unroll
                    for (int m = 0; m < HC; ++m) {
                        auto rv = *reinterpret_cast<const vec_bf16_t*>(
                            residual + (tok * HC + m) * hidden_size + k);
                        #pragma unroll
                        for (int v = 0; v < VEC; ++v)
                            acc[v] += pre_mix[m] * float(rv.val[v]);
                    }

                    vec_bf16_t ov;
                    #pragma unroll
                    for (int v = 0; v < VEC; ++v)
                        ov.val[v] = static_cast<bf16>(acc[v]);
                    *reinterpret_cast<vec_bf16_t*>(layer_input + tok * hidden_size + k) = ov;
                }
            });
    });
}

void launch_mhc_pre_stage1_matrix(
    sycl::queue& queue,
    const bf16* residual,
    const float* fn,
    float* rms_mixes,
    int M,
    int K,
    int N_gemm,
    float rms_eps) {
    auto A_cute = make_tensor(
        make_gmem_ptr(reinterpret_cast<const bfloat16_t*>(residual)),
        make_layout(make_shape(M, K), make_stride(K, Int<1>{})));
    auto B_cute = make_tensor(
        make_gmem_ptr(fn),
        make_layout(make_shape(N_gemm, K), make_stride(K, Int<1>{})));
    auto C_cute = make_tensor(
        make_gmem_ptr(rms_mixes),
        make_layout(make_shape(M, N_gemm), make_stride(N_gemm, Int<1>{})));

    using Policy = GemmRmsPolicy;
    using MMAOp = XE_DPAS_TT<8, float, tfloat32_t>;
    using MMAAtom = MMA_Atom<MMAOp>;
    using TiledMMA_t = typename TiledMMAHelper<
        MMAAtom,
        Layout<typename Policy::WGTile>,
        typename Policy::SGLayout>::TiledMMA;
    TiledMMA_t mma;

    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    syclex::properties kernel_props{
        syclex::sub_group_size<16>,
        intelex::grf_size<256>};

    sycl::range<2> local = {static_cast<size_t>(size(mma)), 1};
    sycl::range<2> global = {
        local[0] * ceil_div(shape<0>(B_cute), get<1>(mma.tile_mnk())),
        local[1] * ceil_div(shape<0>(A_cute), get<0>(mma.tile_mnk()))};

    queue.parallel_for<MhcPreStage1>(
        sycl::nd_range<2>(global, local), kernel_props, [=](auto) {
            mhc_pre_stage1_device<Policy::prefetch_dist>(
                A_cute, B_cute, C_cute, mma, rms_eps);
        });
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> mhc_pre(
    const at::Tensor& residual,
    const at::Tensor& fn,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    double rms_eps,
    double hc_pre_eps,
    double hc_sinkhorn_eps,
    double hc_post_mult_value,
    int64_t sinkhorn_repeat) {
    TORCH_CHECK(residual.is_xpu() && fn.is_xpu(), "mhc_pre: tensors must be on XPU");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be bfloat16");
    TORCH_CHECK(fn.scalar_type() == at::kFloat, "fn must be float32");
    TORCH_CHECK(hc_scale.scalar_type() == at::kFloat, "hc_scale must be float32");
    TORCH_CHECK(hc_base.scalar_type() == at::kFloat, "hc_base must be float32");

    auto residual_c = residual.contiguous();
    auto fn_c = fn.contiguous();

    const int64_t HC = residual_c.size(-2);
    const int64_t H = residual_c.size(-1);
    const int64_t HC2 = HC * HC;
    const int64_t HC3 = HC * 2 + HC2;

    TORCH_CHECK(HC == 4, "mhc_pre: only hc_mult=4 is supported");
    TORCH_CHECK(fn_c.size(0) == HC3 && fn_c.size(1) == HC * H, "fn shape mismatch");
    TORCH_CHECK(hc_scale.numel() == 3, "hc_scale must have 3 elements");
    TORCH_CHECK(hc_base.numel() == HC3, "hc_base must have HC3 elements");

    auto outer_shape = residual_c.sizes().slice(0, residual_c.dim() - 2).vec();
    auto residual_flat = residual_c.view({-1, HC, H});
    const int64_t N = residual_flat.size(0);

    auto opts_f32 = residual_c.options().dtype(at::kFloat);
    auto opts_bf16 = residual_c.options().dtype(at::kBFloat16);

    auto post_mix = at::empty({N, HC}, opts_f32);
    auto comb_mix = at::empty({N, HC2}, opts_f32);
    auto layer_input = at::empty({N, H}, opts_bf16);

    if (N == 0) {
        std::vector<int64_t> ps = outer_shape;
        ps.push_back(HC);
        ps.push_back(1);
        std::vector<int64_t> cs = outer_shape;
        cs.push_back(HC);
        cs.push_back(HC);
        std::vector<int64_t> ls = outer_shape;
        ls.push_back(H);
        return {post_mix.view(ps), comb_mix.view(cs), layer_input.view(ls)};
    }

    auto& queue = vllm::xpu::vllmGetQueue();
    const int M = static_cast<int>(N);
    const int K = static_cast<int>(HC * H);
    const int N_gemm = static_cast<int>(HC3);
    static constexpr int DISPATCH_THRESHOLD = 2048;

    auto rms_mixes = at::empty({M, N_gemm}, opts_f32);

    if (N < DISPATCH_THRESHOLD) {
        launch_mhc_pre_stage1_vector(
            queue,
            reinterpret_cast<const bf16*>(residual_flat.data_ptr()),
            fn_c.data_ptr<float>(),
            rms_mixes.data_ptr<float>(),
            M,
            static_cast<int>(H),
            static_cast<float>(rms_eps));
    } else {
        launch_mhc_pre_stage1_matrix(
            queue,
            reinterpret_cast<const bf16*>(residual_flat.data_ptr()),
            fn_c.data_ptr<float>(),
            rms_mixes.data_ptr<float>(),
            M,
            K,
            N_gemm,
            static_cast<float>(rms_eps));
    }

    launch_mhc_pre_stage2(
        queue,
        rms_mixes.data_ptr<float>(),
        reinterpret_cast<const bf16*>(residual_flat.data_ptr()),
        hc_scale.data_ptr<float>(),
        hc_base.data_ptr<float>(),
        post_mix.data_ptr<float>(),
        comb_mix.data_ptr<float>(),
        reinterpret_cast<bf16*>(layer_input.data_ptr()),
        static_cast<int>(N),
        static_cast<int>(H),
        static_cast<float>(hc_pre_eps),
        static_cast<float>(hc_sinkhorn_eps),
        static_cast<float>(hc_post_mult_value),
        static_cast<int>(sinkhorn_repeat));

    std::vector<int64_t> ps = outer_shape;
    ps.push_back(HC);
    ps.push_back(1);
    std::vector<int64_t> cs = outer_shape;
    cs.push_back(HC);
    cs.push_back(HC);
    std::vector<int64_t> ls = outer_shape;
    ls.push_back(H);
    return {post_mix.view(ps), comb_mix.view(cs), layer_input.view(ls)};
}
