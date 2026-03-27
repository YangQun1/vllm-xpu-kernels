#include "xpu/causal_conv1d/causal_conv1d.hpp"

#include <cstdint>
#include <vector>

#include "utils.h"

namespace vllm::xpu::causal_conv1d {

namespace {

constexpr int kBlockM = 8;
constexpr int kBlockN = 128;

ActMode parse_act_mode(const std::string& activation) {
  if (activation.empty() || activation == "none") {
    return ActMode::none;
  }
  if (activation == "silu") {
    return ActMode::silu;
  }
  if (activation == "swish") {
    return ActMode::swish;
  }
  TORCH_CHECK(false, "Unsupported activation for causal_conv1d: ", activation);
}

std::pair<torch::Tensor, torch::Tensor> build_program_meta(
    const torch::Tensor& query_start_loc,
    int32_t pad_slot_id,
    int block_m,
    torch::Device device) {
  auto qsl_cpu = query_start_loc.to(torch::kCPU);
  auto qsl_ptr = qsl_cpu.data_ptr<int32_t>();
  const int batch = query_start_loc.size(0) - 1;

  std::vector<int32_t> batch_vec;
  std::vector<int32_t> chunk_vec;
  for (int seq = 0; seq < batch; ++seq) {
    const int seqlen = qsl_ptr[seq + 1] - qsl_ptr[seq];
    const int n_chunks = (seqlen + block_m - 1) / block_m;
    for (int c = 0; c < n_chunks; ++c) {
      batch_vec.push_back(seq);
      chunk_vec.push_back(c);
    }
  }
  if (batch_vec.empty()) {
    batch_vec.push_back(pad_slot_id);
    chunk_vec.push_back(0);
  }

  auto opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  auto batch_ptr = torch::from_blob(
                       batch_vec.data(),
                       {static_cast<int64_t>(batch_vec.size())},
                       opts)
                       .clone()
                       .to(device);
  auto token_chunk_offset_ptr = torch::from_blob(
                                    chunk_vec.data(),
                                    {static_cast<int64_t>(chunk_vec.size())},
                                    opts)
                                    .clone()
                                    .to(device);
  return {batch_ptr, token_chunk_offset_ptr};
}

template <typename scalar_t, int WIDTH>
void launch_fwd(
    sycl::queue& queue,
    torch::Tensor& out,
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias,
    torch::Tensor& conv_states,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& cache_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    const torch::Tensor& batch_ptr,
    const torch::Tensor& token_chunk_offset_ptr,
    int64_t pad_slot_id,
    ActMode act_mode) {
    const int dim = static_cast<int>(x.size(0));
    const int width = static_cast<int>(weight.size(1));
  const int state_len = width - 1;
    const int num_cache_lines = static_cast<int>(conv_states.size(0));
    const int sx_dim = static_cast<int>(x.stride(0));
    const int sx_tok = static_cast<int>(x.stride(1));
    const int sw_dim = static_cast<int>(weight.stride(0));
    const int sw_w = static_cast<int>(weight.stride(1));
    const int ss_seq = static_cast<int>(conv_states.stride(0));
    const int ss_dim = static_cast<int>(conv_states.stride(1));
    const int ss_tok = static_cast<int>(conv_states.stride(2));
    const int sci = static_cast<int>(cache_indices.stride(0));
    const int so_dim = static_cast<int>(out.stride(0));
    const int so_tok = static_cast<int>(out.stride(1));
    const int pad_slot_id_i32 = static_cast<int>(pad_slot_id);

    auto nd_range =
      causal_conv1d_fwd_kernel<scalar_t, kBlockM, kBlockN, WIDTH>::get_nd_range(
      static_cast<int>(batch_ptr.size(0)), dim);

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<scalar_t, 1> smem_x(
        (kBlockM + state_len) * kBlockN,
        cgh);
    causal_conv1d_fwd_kernel<scalar_t, kBlockM, kBlockN, WIDTH> task(
        reinterpret_cast<scalar_t*>(out.data_ptr()),
        reinterpret_cast<const scalar_t*>(x.data_ptr()),
        reinterpret_cast<const scalar_t*>(weight.data_ptr()),
        bias.has_value() ? reinterpret_cast<const scalar_t*>(bias->data_ptr())
                         : nullptr,
        reinterpret_cast<scalar_t*>(conv_states.data_ptr()),
        reinterpret_cast<const int32_t*>(query_start_loc.data_ptr()),
        reinterpret_cast<const int32_t*>(cache_indices.data_ptr()),
        has_initial_state.has_value()
            ? reinterpret_cast<const bool*>(has_initial_state->data_ptr())
            : nullptr,
        reinterpret_cast<const int32_t*>(batch_ptr.data_ptr()),
        reinterpret_cast<const int32_t*>(token_chunk_offset_ptr.data_ptr()),
        dim,
        width,
        state_len,
        num_cache_lines,
        sx_dim,
        sx_tok,
        sw_dim,
        sw_w,
        ss_seq,
        ss_dim,
        ss_tok,
        sci,
        so_dim,
        so_tok,
        pad_slot_id_i32,
        bias.has_value(),
        true,
        act_mode,
        smem_x);
    cgh.parallel_for(nd_range, task);
  });
}

template <typename scalar_t, int WIDTH>
void launch_fwd_channellast(
    sycl::queue& queue,
    torch::Tensor& out,
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias,
    torch::Tensor& conv_states,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& cache_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    const torch::Tensor& batch_ptr,
    const torch::Tensor& token_chunk_offset_ptr,
    int64_t pad_slot_id,
    ActMode act_mode) {
  const int dim = static_cast<int>(x.size(0));
  const int width = static_cast<int>(weight.size(1));
  const int state_len = width - 1;
  const int num_cache_lines = static_cast<int>(conv_states.size(0));
  const int sx_tok = static_cast<int>(x.stride(1));
  const int sw_dim = static_cast<int>(weight.stride(0));
  const int sw_w = static_cast<int>(weight.stride(1));
  const int ss_seq = static_cast<int>(conv_states.stride(0));
  const int ss_dim = static_cast<int>(conv_states.stride(1));
  const int ss_tok = static_cast<int>(conv_states.stride(2));
  const int sci = static_cast<int>(cache_indices.stride(0));
  const int so_tok = static_cast<int>(out.stride(1));
  const int pad_slot_id_i32 = static_cast<int>(pad_slot_id);

  auto nd_range =
      causal_conv1d_channellast_fwd_kernel<
          scalar_t,
          kBlockM,
          kBlockN,
          WIDTH>::get_nd_range(static_cast<int>(batch_ptr.size(0)), dim);

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<scalar_t, 1> smem_x(
        (kBlockM + state_len) * kBlockN,
        cgh);
    causal_conv1d_channellast_fwd_kernel<
        scalar_t,
        kBlockM,
        kBlockN,
        WIDTH>
        task(
            reinterpret_cast<scalar_t*>(out.data_ptr()),
            reinterpret_cast<const scalar_t*>(x.data_ptr()),
            reinterpret_cast<const scalar_t*>(weight.data_ptr()),
            bias.has_value() ? reinterpret_cast<const scalar_t*>(bias->data_ptr())
                             : nullptr,
            reinterpret_cast<scalar_t*>(conv_states.data_ptr()),
            reinterpret_cast<const int32_t*>(query_start_loc.data_ptr()),
            reinterpret_cast<const int32_t*>(cache_indices.data_ptr()),
            has_initial_state.has_value()
                ? reinterpret_cast<const bool*>(has_initial_state->data_ptr())
                : nullptr,
            reinterpret_cast<const int32_t*>(batch_ptr.data_ptr()),
            reinterpret_cast<const int32_t*>(token_chunk_offset_ptr.data_ptr()),
            dim,
            state_len,
            num_cache_lines,
            sx_tok,
            sw_dim,
            sw_w,
            ss_seq,
            ss_dim,
            ss_tok,
            sci,
            so_tok,
            pad_slot_id_i32,
            bias.has_value(),
            true,
            act_mode,
            smem_x);
    cgh.parallel_for(nd_range, task);
  });
}

template <typename scalar_t, int WIDTH>
void launch_update(
    sycl::queue& queue,
    torch::Tensor& out,
    const torch::Tensor& x,
    torch::Tensor& conv_state,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias,
    const std::string& activation,
    const torch::Tensor& conv_state_indices,
    const std::optional<torch::Tensor>& num_accepted_tokens,
    const std::optional<torch::Tensor>& query_start_loc,
    int64_t max_query_len,
    int64_t pad_slot_id) {
  int batch = 0;
  int dim = 0;
  int seqlen = 0;
  int stride_x_seq = 0;
  int stride_x_dim = 0;
  int stride_x_token = 0;
  int stride_o_seq = 0;
  int stride_o_dim = 0;
  int stride_o_token = 0;

  const bool is_varlen = query_start_loc.has_value();
  if (is_varlen) {
    batch = static_cast<int>(conv_state_indices.size(0));
    dim = static_cast<int>(x.size(1));
    seqlen = static_cast<int>(max_query_len);
    stride_x_seq = 0;
    stride_x_token = static_cast<int>(x.stride(0));
    stride_x_dim = static_cast<int>(x.stride(1));
    stride_o_seq = 0;
    stride_o_token = static_cast<int>(out.stride(0));
    stride_o_dim = static_cast<int>(out.stride(1));
  } else {
    batch = static_cast<int>(x.size(0));
    dim = static_cast<int>(x.size(1));
    seqlen = static_cast<int>(x.size(2));
    stride_x_seq = static_cast<int>(x.stride(0));
    stride_x_dim = static_cast<int>(x.stride(1));
    stride_x_token = static_cast<int>(x.stride(2));
    stride_o_seq = static_cast<int>(out.stride(0));
    stride_o_dim = static_cast<int>(out.stride(1));
    stride_o_token = static_cast<int>(out.stride(2));
  }

  const int width = static_cast<int>(weight.size(1));
  int state_len = width - 1;
  if (num_accepted_tokens.has_value()) {
    state_len = width - 1 + (seqlen - 1);
  }

  const int sw_dim = static_cast<int>(weight.stride(0));
  const int sw_w = static_cast<int>(weight.stride(1));
  const int ss_seq = static_cast<int>(conv_state.stride(0));
  const int ss_dim = static_cast<int>(conv_state.stride(1));
  const int ss_tok = static_cast<int>(conv_state.stride(2));
  const int ssi = static_cast<int>(conv_state_indices.stride(0));
  const int n_cache = static_cast<int>(conv_state.size(0));
  const int pad_slot_id_i32 = static_cast<int>(pad_slot_id);

    auto nd_range =
      causal_conv1d_update_kernel<scalar_t, kBlockN, WIDTH>::get_nd_range(
        batch, dim);
  const ActMode act_mode = parse_act_mode(activation);

  queue.submit([&](sycl::handler& cgh) {
    causal_conv1d_update_kernel<scalar_t, kBlockN, WIDTH> task(
        reinterpret_cast<scalar_t*>(out.data_ptr()),
        reinterpret_cast<const scalar_t*>(x.data_ptr()),
        reinterpret_cast<const scalar_t*>(weight.data_ptr()),
        bias.has_value() ? reinterpret_cast<const scalar_t*>(bias->data_ptr())
                         : nullptr,
        reinterpret_cast<scalar_t*>(conv_state.data_ptr()),
        reinterpret_cast<const int32_t*>(conv_state_indices.data_ptr()),
        num_accepted_tokens.has_value()
            ? reinterpret_cast<const int32_t*>(num_accepted_tokens->data_ptr())
            : nullptr,
        query_start_loc.has_value()
            ? reinterpret_cast<const int32_t*>(query_start_loc->data_ptr())
            : nullptr,
        batch,
        dim,
        seqlen,
        width,
        state_len,
        n_cache,
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        sw_dim,
        sw_w,
        ss_seq,
        ss_dim,
        ss_tok,
        ssi,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        pad_slot_id_i32,
        bias.has_value(),
        is_varlen,
        num_accepted_tokens.has_value(),
        act_mode);
    cgh.parallel_for(nd_range, task);
  });
}

}  // namespace

torch::Tensor causal_conv1d_fwd(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias,
    torch::Tensor& conv_states,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& cache_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    const std::string& activation,
    int64_t pad_slot_id,
    bool validate_data) {
  CHECK_DEVICE(x);
  CHECK_DEVICE(weight);
  CHECK_DEVICE(conv_states);
  CHECK_DEVICE(query_start_loc);
  CHECK_DEVICE(cache_indices);

  TORCH_CHECK(x.dim() == 2, "x must be 2D [dim, cu_seqlen]");
  TORCH_CHECK(weight.dim() == 2, "weight must be 2D [dim, width]");
  TORCH_CHECK(query_start_loc.dim() == 1, "query_start_loc must be 1D");
  TORCH_CHECK(cache_indices.dim() == 1, "cache_indices must be 1D");
  TORCH_CHECK(conv_states.dim() == 3, "conv_states must be 3D");

  if (validate_data) {
    TORCH_CHECK(weight.stride(1) == 1, "weight must be contiguous on width axis");
    TORCH_CHECK(
        conv_states.stride(1) == 1 || conv_states.stride(2) == 1,
        "conv_states must have contiguous dim or state axis");
  }

  auto act_mode = parse_act_mode(activation);
  const int dim = static_cast<int>(x.size(0));
  const int width = static_cast<int>(weight.size(1));
  const bool is_channel_last = (x.stride(0) == 1) && (x.stride(1) > 1);

  auto original_dtype = x.scalar_type();
  torch::Tensor x_cast = x.to(conv_states.scalar_type());
  torch::Tensor out = torch::empty_like(x_cast);
  torch::Tensor query_start_loc_i32 = query_start_loc.to(torch::kInt32);
  torch::Tensor cache_indices_i32 = cache_indices.to(torch::kInt32);

  constexpr int kVec = 4;
  const int elem_size = static_cast<int>(x_cast.element_size());
  const int vec_bytes = kVec * elem_size;
  auto is_ptr_aligned = [&](const torch::Tensor& tensor) {
    const auto addr = reinterpret_cast<uintptr_t>(tensor.data_ptr());
    return (addr % static_cast<uintptr_t>(vec_bytes)) == 0;
  };

  const bool can_vec_x = (x_cast.stride(1) % kVec == 0) && is_ptr_aligned(x_cast);
  const bool can_vec_o = (out.stride(1) % kVec == 0) && is_ptr_aligned(out);
  const bool can_vec_state =
      (conv_states.stride(0) % kVec == 0) &&
      (conv_states.stride(1) == 1) &&
      (conv_states.stride(2) % kVec == 0) &&
      is_ptr_aligned(conv_states);

  const bool weight_stride_ok =
      (weight.stride(1) == 1) && ((width < 4) || (weight.stride(0) % kVec == 0));
    const bool can_vec_weight =
      weight_stride_ok && ((width < 4) || is_ptr_aligned(weight));

  const bool use_channellast_vec_path =
      is_channel_last &&
      (out.stride(0) == 1) &&
      (dim % kVec == 0) &&
      can_vec_x &&
      can_vec_o &&
      can_vec_state &&
      can_vec_weight;

  auto [batch_ptr, token_chunk_offset_ptr] =
      build_program_meta(
        query_start_loc_i32,
        static_cast<int32_t>(pad_slot_id),
        kBlockM,
        x.device());

  auto& queue = vllm::xpu::vllmGetQueue();
#define LAUNCH_FWD_CALL(scalar_t_, W_)                                        \
  do {                                                                         \
    if (use_channellast_vec_path) {                                            \
      launch_fwd_channellast<scalar_t_, W_>(                                   \
          queue,                                                                \
          out,                                                                  \
          x_cast,                                                               \
          weight,                                                               \
          bias,                                                                 \
          conv_states,                                                          \
          query_start_loc_i32,                                                  \
          cache_indices_i32,                                                    \
          has_initial_state,                                                    \
          batch_ptr,                                                            \
          token_chunk_offset_ptr,                                               \
          pad_slot_id,                                                          \
          act_mode);                                                            \
    } else {                                                                    \
      launch_fwd<scalar_t_, W_>(                                               \
          queue,                                                                \
          out,                                                                  \
          x_cast,                                                               \
          weight,                                                               \
          bias,                                                                 \
          conv_states,                                                          \
          query_start_loc_i32,                                                  \
          cache_indices_i32,                                                    \
          has_initial_state,                                                    \
          batch_ptr,                                                            \
          token_chunk_offset_ptr,                                               \
          pad_slot_id,                                                          \
          act_mode);                                                            \
    }                                                                           \
  } while (0)

#define DISPATCH_FWD_WIDTH(scalar_t_)                                         \
  switch (width) {                                                            \
    case 2:                                                                   \
      LAUNCH_FWD_CALL(scalar_t_, 2);                                          \
      break;                                                                   \
    case 3:                                                                   \
      LAUNCH_FWD_CALL(scalar_t_, 3);                                          \
      break;                                                                   \
    case 4:                                                                   \
      LAUNCH_FWD_CALL(scalar_t_, 4);                                          \
      break;                                                                   \
    case 5:                                                                   \
      LAUNCH_FWD_CALL(scalar_t_, 5);                                          \
      break;                                                                   \
    default:                                                                  \
      TORCH_CHECK(false, "causal_conv1d_fwd only supports width in {2,3,4,5}, got ", width); \
  }

  if (x_cast.scalar_type() == at::kBFloat16) {
    DISPATCH_FWD_WIDTH(sycl::ext::oneapi::bfloat16)
  } else if (x_cast.scalar_type() == at::kHalf) {
    DISPATCH_FWD_WIDTH(sycl::half)
  } else {
    DISPATCH_FWD_WIDTH(float)
  }
#undef DISPATCH_FWD_WIDTH
#undef LAUNCH_FWD_CALL

  return out.to(original_dtype);
}

torch::Tensor causal_conv1d_update(
    const torch::Tensor& x,
    torch::Tensor& conv_state,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias,
    const std::string& activation,
    const torch::Tensor& conv_state_indices,
    const std::optional<torch::Tensor>& num_accepted_tokens,
    const std::optional<torch::Tensor>& query_start_loc,
    int64_t max_query_len,
    int64_t pad_slot_id,
    bool validate_data) {
  CHECK_DEVICE(x);
  CHECK_DEVICE(conv_state);
  CHECK_DEVICE(weight);
  CHECK_DEVICE(conv_state_indices);
  if (query_start_loc.has_value()) {
    CHECK_DEVICE(query_start_loc.value());
  }
  if (num_accepted_tokens.has_value()) {
    CHECK_DEVICE(num_accepted_tokens.value());
  }

  TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
  TORCH_CHECK(conv_state.dim() == 3, "conv_state must be 3D");
  TORCH_CHECK(conv_state_indices.dim() == 1, "conv_state_indices must be 1D");

  auto original_dtype = x.scalar_type();
  torch::Tensor x_cast = x.to(conv_state.scalar_type());
  torch::Tensor conv_state_indices_i32 = conv_state_indices.to(torch::kInt32);
  std::optional<torch::Tensor> num_accepted_tokens_i32 = std::nullopt;
  if (num_accepted_tokens.has_value()) {
    num_accepted_tokens_i32 = num_accepted_tokens->to(torch::kInt32);
  }
  std::optional<torch::Tensor> query_start_loc_i32 = std::nullopt;
  if (query_start_loc.has_value()) {
    query_start_loc_i32 = query_start_loc->to(torch::kInt32);
  }

  bool unsqueeze = !query_start_loc.has_value() && x_cast.dim() == 2;
  if (unsqueeze) {
    x_cast = x_cast.unsqueeze(-1);
  }

  torch::Tensor out = torch::empty_like(x_cast);
  auto& queue = vllm::xpu::vllmGetQueue();

  if (validate_data) {
    TORCH_CHECK(weight.stride(1) == 1, "weight must be contiguous on width axis");
    TORCH_CHECK(
        conv_state.stride(1) == 1 || conv_state.stride(2) == 1,
        "conv_state must have contiguous dim or state axis");
  }

  const int width = static_cast<int>(weight.size(1));

#define LAUNCH_UPDATE_CALL(scalar_t_, W_)                                     \
  launch_update<scalar_t_, W_>(                                               \
      queue,                                                                   \
      out,                                                                     \
      x_cast,                                                                  \
      conv_state,                                                              \
      weight,                                                                  \
      bias,                                                                    \
      activation,                                                              \
      conv_state_indices_i32,                                                  \
      num_accepted_tokens_i32,                                                 \
      query_start_loc_i32,                                                     \
      max_query_len,                                                           \
      pad_slot_id)

#define DISPATCH_UPDATE_WIDTH(scalar_t_)                                      \
  switch (width) {                                                             \
    case 2:                                                                    \
      LAUNCH_UPDATE_CALL(scalar_t_, 2);                                        \
      break;                                                                    \
    case 3:                                                                    \
      LAUNCH_UPDATE_CALL(scalar_t_, 3);                                        \
      break;                                                                    \
    case 4:                                                                    \
      LAUNCH_UPDATE_CALL(scalar_t_, 4);                                        \
      break;                                                                    \
    case 5:                                                                    \
      LAUNCH_UPDATE_CALL(scalar_t_, 5);                                        \
      break;                                                                    \
    default:                                                                   \
      TORCH_CHECK(false, "causal_conv1d_update only supports width in {2,3,4,5}, got ", width); \
  }

  if (x_cast.scalar_type() == at::kBFloat16) {
    DISPATCH_UPDATE_WIDTH(sycl::ext::oneapi::bfloat16)
  } else if (x_cast.scalar_type() == at::kHalf) {
    DISPATCH_UPDATE_WIDTH(sycl::half)
  } else {
    DISPATCH_UPDATE_WIDTH(float)
  }
#undef DISPATCH_UPDATE_WIDTH
#undef LAUNCH_UPDATE_CALL

  if (unsqueeze) {
    out = out.squeeze(-1);
  }
  return out.to(original_dtype);
}

}  // namespace vllm::xpu::causal_conv1d

torch::Tensor causal_conv1d_fwd(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias,
    torch::Tensor& conv_states,
    const torch::Tensor& query_start_loc,
    const torch::Tensor& cache_indices,
    const std::optional<torch::Tensor>& has_initial_state,
    const std::string& activation,
    int64_t pad_slot_id,
    bool validate_data) {
  return vllm::xpu::causal_conv1d::causal_conv1d_fwd(
      x,
      weight,
      bias,
      conv_states,
      query_start_loc,
      cache_indices,
      has_initial_state,
      activation,
      pad_slot_id,
      validate_data);
}

torch::Tensor causal_conv1d_update(
    const torch::Tensor& x,
    torch::Tensor& conv_state,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias,
    const std::string& activation,
    const torch::Tensor& conv_state_indices,
    const std::optional<torch::Tensor>& num_accepted_tokens,
    const std::optional<torch::Tensor>& query_start_loc,
    int64_t max_query_len,
    int64_t pad_slot_id,
    bool validate_data) {
  return vllm::xpu::causal_conv1d::causal_conv1d_update(
      x,
      conv_state,
      weight,
      bias,
      activation,
      conv_state_indices,
      num_accepted_tokens,
      query_start_loc,
      max_query_len,
      pad_slot_id,
      validate_data);
}
