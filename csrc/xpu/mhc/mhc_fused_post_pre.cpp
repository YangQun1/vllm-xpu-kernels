// DeepSeek V4 MHC fused post+pre kernel.
//
// Phase 0: composed — calls launch_mhc_post_opt, then
//   launch_mhc_pre_stage1_{vector,matrix} + launch_mhc_pre_stage2.
//   Future phases will fuse the small-N path into a single kernel.
//
// Data flow:
//   x[N,H] bf16 + residual[N,HC,H] bf16 + post_layer_mix[N,HC,1] fp32
//      + comb_res_mix[N,HC,HC] fp32
//             |
//             v
//      mhc_post  -->  residual_cur[N,HC,H] bf16
//             |
//             v
//      mhc_pre   -->  post_mix_cur[N,HC,1] fp32
//                      comb_mix_cur[N,HC,HC] fp32
//                      layer_input_cur[N,H] bf16

#include <ATen/ATen.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

#include "../../utils.h"

using bf16 = sycl::ext::oneapi::bfloat16;

static constexpr int HC = 4;
static constexpr int DISPATCH_THRESHOLD = 2048;

// ---- extern declarations for launch functions (defined in sibling TUs) ----

// from mhc_post.cpp
extern sycl::event launch_mhc_post_opt(
    sycl::queue& q,
    const float* comb_res_mix,
    const bf16* residual,
    const float* post_layer_mix,
    const bf16* x,
    bf16* out,
    int N,
    int H);

// from mhc_pre.cpp
extern sycl::event launch_mhc_pre_stage1_vector(
    sycl::queue& q,
    const bf16* residual,
    const float* fn,
    float* rms_mixes,
    int N,
    int H,
    float rms_eps);

extern void launch_mhc_pre_stage1_matrix(
    sycl::queue& queue,
    const bf16* residual,
    const float* fn,
    float* rms_mixes,
    int M,
    int K,
    int N_gemm,
    float rms_eps);

extern sycl::event launch_mhc_pre_stage2(
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
    int sinkhorn_repeat);

// ---- fused op implementation ----

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mhc_fused_post_pre(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post_layer_mix,
    const at::Tensor& comb_res_mix,
    const at::Tensor& fn,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    double rms_eps,
    double hc_pre_eps,
    double hc_sinkhorn_eps,
    double hc_post_mult_value,
    int64_t sinkhorn_repeat) {

    TORCH_CHECK(x.is_xpu() && residual.is_xpu() && fn.is_xpu(),
                "mhc_fused_post_pre: tensors must be on XPU");
    TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be bfloat16");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(post_layer_mix.scalar_type() == at::kFloat, "post_layer_mix must be float32");
    TORCH_CHECK(comb_res_mix.scalar_type() == at::kFloat, "comb_res_mix must be float32");
    TORCH_CHECK(fn.scalar_type() == at::kFloat, "fn must be float32");

    auto residual_c = residual.contiguous();
    auto x_c = x.contiguous();
    auto a_c = comb_res_mix.contiguous();
    auto c_c = post_layer_mix.squeeze(-1).contiguous();
    auto fn_c = fn.contiguous();

    TORCH_CHECK(residual_c.dim() == 3, "residual must be [N, HC, H]");
    TORCH_CHECK(residual_c.size(1) == HC, "residual dim-2 must be HC=4");

    const int64_t N = residual_c.size(0);
    const int64_t H = residual_c.size(2);
    const int64_t HC2 = HC * HC;
    const int64_t HC3 = HC * 2 + HC2;
    const int M = static_cast<int>(N);
    const int K = static_cast<int>(HC * H);
    const int N_gemm = static_cast<int>(HC3);

    auto opts_f32 = residual_c.options().dtype(at::kFloat);
    auto opts_bf16 = residual_c.options().dtype(at::kBFloat16);

    auto residual_cur = at::empty_like(residual_c);
    auto post_mix_cur = at::empty({N, HC}, opts_f32);
    auto comb_mix_cur = at::empty({N, HC2}, opts_f32);
    auto layer_input_cur = at::empty({N, H}, opts_bf16);

    if (N == 0) {
        return {residual_cur,
                post_mix_cur.view({N, HC, 1}),
                comb_mix_cur.view({N, HC, HC}),
                layer_input_cur};
    }

    auto& queue = vllm::xpu::vllmGetQueue();

    // ---- mhc_post: residual_cur = einsum(comb, residual) + post_mix * x ----
    launch_mhc_post_opt(
        queue,
        a_c.data_ptr<float>(),
        reinterpret_cast<const bf16*>(residual_c.data_ptr()),
        c_c.data_ptr<float>(),
        reinterpret_cast<const bf16*>(x_c.data_ptr()),
        reinterpret_cast<bf16*>(residual_cur.data_ptr()),
        M,
        static_cast<int>(H));

    // ---- mhc_pre stage 1: GEMM + RMS-norm → rms_mixes ----
    auto rms_mixes = at::empty({M, N_gemm}, opts_f32);

    if (N < DISPATCH_THRESHOLD) {
        launch_mhc_pre_stage1_vector(
            queue,
            reinterpret_cast<const bf16*>(residual_cur.data_ptr()),
            fn_c.data_ptr<float>(),
            rms_mixes.data_ptr<float>(),
            M,
            static_cast<int>(H),
            static_cast<float>(rms_eps));
    } else {
        launch_mhc_pre_stage1_matrix(
            queue,
            reinterpret_cast<const bf16*>(residual_cur.data_ptr()),
            fn_c.data_ptr<float>(),
            rms_mixes.data_ptr<float>(),
            M,
            K,
            N_gemm,
            static_cast<float>(rms_eps));
    }

    // ---- mhc_pre stage 2: sigmoid / Sinkhorn / weighted reduction ----
    launch_mhc_pre_stage2(
        queue,
        rms_mixes.data_ptr<float>(),
        reinterpret_cast<const bf16*>(residual_cur.data_ptr()),
        hc_scale.data_ptr<float>(),
        hc_base.data_ptr<float>(),
        post_mix_cur.data_ptr<float>(),
        comb_mix_cur.data_ptr<float>(),
        reinterpret_cast<bf16*>(layer_input_cur.data_ptr()),
        M,
        static_cast<int>(H),
        static_cast<float>(hc_pre_eps),
        static_cast<float>(hc_sinkhorn_eps),
        static_cast<float>(hc_post_mult_value),
        static_cast<int>(sinkhorn_repeat));

    return {residual_cur,
            post_mix_cur.view({N, HC, 1}),
            comb_mix_cur.view({N, HC, HC}),
            layer_input_cur};
}
