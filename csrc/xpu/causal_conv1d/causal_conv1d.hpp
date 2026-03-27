#pragma once

#include <sycl/sycl.hpp>
#include <torch/all.h>

namespace vllm::xpu::causal_conv1d {

enum class ActMode {
  none = 0,
  silu = 1,
  swish = 2,
};

template <typename T, int BLOCK_M, int BLOCK_N, int WIDTH>
struct causal_conv1d_fwd_kernel {
 public:
  static constexpr int sub_group_size = 32;

  causal_conv1d_fwd_kernel(
      T* out,
      const T* x,
      const T* weight,
      const T* bias,
      T* conv_states,
      const int32_t* query_start_loc,
      const int32_t* cache_indices,
      const bool* has_initial_state,
      const int32_t* batch_ptr,
      const int32_t* token_chunk_offset_ptr,
      const int dim,
      const int width,
      const int state_len,
      const int num_cache_lines,
      const int stride_x_dim,
      const int stride_x_token,
      const int stride_w_dim,
      const int stride_w_width,
      const int stride_state_seq,
      const int stride_state_dim,
      const int stride_state_token,
      const int stride_cache_indices,
      const int stride_o_dim,
      const int stride_o_token,
      const int pad_slot_id,
      const bool has_bias,
      const bool use_pad_slot,
      const ActMode act_mode,
      sycl::local_accessor<float, 1> smem_x)
      : out(out),
        x(x),
        weight(weight),
        bias(bias),
        conv_states(conv_states),
        query_start_loc(query_start_loc),
        cache_indices(cache_indices),
        has_initial_state(has_initial_state),
        batch_ptr(batch_ptr),
        token_chunk_offset_ptr(token_chunk_offset_ptr),
        dim(dim),
        width(width),
        state_len(state_len),
        num_cache_lines(num_cache_lines),
        stride_x_dim(stride_x_dim),
        stride_x_token(stride_x_token),
        stride_w_dim(stride_w_dim),
        stride_w_width(stride_w_width),
        stride_state_seq(stride_state_seq),
        stride_state_dim(stride_state_dim),
        stride_state_token(stride_state_token),
        stride_cache_indices(stride_cache_indices),
        stride_o_dim(stride_o_dim),
        stride_o_token(stride_o_token),
        pad_slot_id(pad_slot_id),
        has_bias(has_bias),
        use_pad_slot(use_pad_slot),
        act_mode(act_mode),
        smem_x(smem_x) {}

  static inline sycl::nd_range<2>
  get_nd_range(int num_programs, int dim) {
    const int feat_groups = (dim + BLOCK_N - 1) / BLOCK_N;
    sycl::range<2> local(1, BLOCK_N);
    sycl::range<2> global(num_programs, feat_groups);
    return sycl::nd_range<2>(global * local, local);
  }

  static inline float silu(float x) {
    return x / (1.0f + sycl::exp(-x));
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<2> item) const {
    const int program_id = item.get_group(0);
    const int feat_group = item.get_group(1);
    const int local_id = item.get_local_linear_id();
    const int feat = feat_group * BLOCK_N + local_id;

    const int seq_idx = batch_ptr[program_id];
    const int chunk_offset = token_chunk_offset_ptr[program_id];
    if (seq_idx == pad_slot_id) {
      return;
    }

    const int sequence_start_index = query_start_loc[seq_idx];
    const int sequence_end_index = query_start_loc[seq_idx + 1];
    const int seqlen = sequence_end_index - sequence_start_index;
    if (seqlen <= 0) {
      return;
    }

    const int token_offset = BLOCK_M * chunk_offset;
    const int segment_len = sycl::max(0, sycl::min(BLOCK_M, seqlen - token_offset));
    if (segment_len <= 0) {
      return;
    }

    int cache_line = 0;
    if (cache_indices != nullptr) {
      cache_line = cache_indices[seq_idx * stride_cache_indices];
      if (use_pad_slot && cache_line == pad_slot_id) {
        return;
      }
    }
    const bool load_init_state =
        (has_initial_state == nullptr) ? true : has_initial_state[seq_idx];

    const T* x_seq_base =
        x + sequence_start_index * stride_x_token + feat * stride_x_dim;
    const T* w_base = weight + feat * stride_w_dim;
    T* o_seq_base = out + sequence_start_index * stride_o_token + feat * stride_o_dim;
    const T* state_base =
        conv_states + cache_line * stride_state_seq + feat * stride_state_dim;

    const int window_len = state_len + segment_len;
    if (feat < dim) {
      for (int pos = 0; pos < window_len; ++pos) {
        const int src_token = token_offset - state_len + pos;
        float xv = 0.0f;
        if (src_token >= 0 && src_token < seqlen) {
          xv = static_cast<float>(x_seq_base[src_token * stride_x_token]);
        } else if (load_init_state && state_len > 0) {
          const int state_pos = state_len + src_token;
          if (state_pos >= 0 && state_pos < state_len &&
              cache_line < num_cache_lines) {
            xv = static_cast<float>(state_base[state_pos * stride_state_token]);
          }
        }
        smem_x[pos * BLOCK_N + local_id] = xv;
      }
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (feat >= dim) {
      return;
    }

    const float bias_val =
        (has_bias && bias != nullptr) ? static_cast<float>(bias[feat]) : 0.0f;

    float w_reg[WIDTH];
#pragma unroll
    for (int k = 0; k < WIDTH; ++k) {
      w_reg[k] = static_cast<float>(w_base[k * stride_w_width]);
    }

    for (int t = 0; t < segment_len; ++t) {
      float acc = bias_val;
#pragma unroll
      for (int k = 0; k < WIDTH; ++k) {
        const float xv = smem_x[(t + k) * BLOCK_N + local_id];
        acc += xv * w_reg[k];
      }

      if (act_mode == ActMode::silu || act_mode == ActMode::swish) {
        acc = silu(acc);
      }
      o_seq_base[(token_offset + t) * stride_o_token] = static_cast<T>(acc);
    }

    if (chunk_offset == 0 && state_len > 0) {
      for (int s = 0; s < state_len; ++s) {
        const int src_token = seqlen - state_len + s;
        float new_state = 0.0f;
        if (src_token >= 0) {
          new_state = static_cast<float>(x_seq_base[src_token * stride_x_token]);
        } else if (load_init_state) {
          const int prev_pos = state_len + src_token;
          if (prev_pos >= 0 && prev_pos < state_len &&
              cache_line < num_cache_lines) {
            new_state = static_cast<float>(state_base[prev_pos * stride_state_token]);
          }
        }
        const_cast<T*>(state_base)[s * stride_state_token] = static_cast<T>(new_state);
      }
    }
  }

 private:
  T* out;
  const T* x;
  const T* weight;
  const T* bias;
  T* conv_states;
  const int32_t* query_start_loc;
  const int32_t* cache_indices;
  const bool* has_initial_state;
  const int32_t* batch_ptr;
  const int32_t* token_chunk_offset_ptr;
  const int dim;
  const int width;
  const int state_len;
  const int num_cache_lines;
  const int stride_x_dim;
  const int stride_x_token;
  const int stride_w_dim;
  const int stride_w_width;
  const int stride_state_seq;
  const int stride_state_dim;
  const int stride_state_token;
  const int stride_cache_indices;
  const int stride_o_dim;
  const int stride_o_token;
  const int pad_slot_id;
  const bool has_bias;
  const bool use_pad_slot;
  const ActMode act_mode;
  sycl::local_accessor<float, 1> smem_x;
};

template <typename T, int BLOCK_N, int WIDTH>
struct causal_conv1d_update_kernel {
 public:
  static constexpr int sub_group_size = 32;

  causal_conv1d_update_kernel(
      T* out,
      const T* x,
      const T* weight,
      const T* bias,
      T* conv_state,
      const int32_t* conv_state_indices,
      const int32_t* num_accepted_tokens,
      const int32_t* query_start_loc,
      const int batch,
      const int dim,
      const int seqlen,
      const int width,
      const int state_len,
      const int num_cache_lines,
      const int stride_x_seq,
      const int stride_x_dim,
      const int stride_x_token,
      const int stride_w_dim,
      const int stride_w_width,
      const int stride_state_seq,
      const int stride_state_dim,
      const int stride_state_tok,
      const int stride_state_indices,
      const int stride_o_seq,
      const int stride_o_dim,
      const int stride_o_token,
      const int pad_slot_id,
      const bool has_bias,
      const bool is_varlen,
      const bool is_spec_decoding,
      const ActMode act_mode)
      : out(out),
        x(x),
        weight(weight),
        bias(bias),
        conv_state(conv_state),
        conv_state_indices(conv_state_indices),
        num_accepted_tokens(num_accepted_tokens),
        query_start_loc(query_start_loc),
        batch(batch),
        dim(dim),
        seqlen(seqlen),
        width(width),
        state_len(state_len),
        num_cache_lines(num_cache_lines),
        stride_x_seq(stride_x_seq),
        stride_x_dim(stride_x_dim),
        stride_x_token(stride_x_token),
        stride_w_dim(stride_w_dim),
        stride_w_width(stride_w_width),
        stride_state_seq(stride_state_seq),
        stride_state_dim(stride_state_dim),
        stride_state_tok(stride_state_tok),
        stride_state_indices(stride_state_indices),
        stride_o_seq(stride_o_seq),
        stride_o_dim(stride_o_dim),
        stride_o_token(stride_o_token),
        pad_slot_id(pad_slot_id),
        has_bias(has_bias),
        is_varlen(is_varlen),
        is_spec_decoding(is_spec_decoding),
        act_mode(act_mode) {}

  static inline sycl::nd_range<2> get_nd_range(int batch, int dim) {
    const int feat_groups = (dim + BLOCK_N - 1) / BLOCK_N;
    sycl::range<2> local(1, BLOCK_N);
    sycl::range<2> global(batch, feat_groups);
    return sycl::nd_range<2>(global * local, local);
  }

  static inline float silu(float x) {
    return x / (1.0f + sycl::exp(-x));
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<2> item) const {
    const int seq_idx = item.get_group(0);
    const int feat_group = item.get_group(1);
    const int local_id = item.get_local_linear_id();
    const int feat = feat_group * BLOCK_N + local_id;
    if (seq_idx >= batch || feat >= dim) {
      return;
    }

    int q_start = 0;
    int q_end = 0;
    int x_offset = 0;
    int o_offset = 0;
    int seq_len = seqlen;
    if (is_varlen) {
      q_start = query_start_loc[seq_idx];
      q_end = query_start_loc[seq_idx + 1];
      seq_len = q_end - q_start;
      x_offset = q_start * stride_x_token;
      o_offset = q_start * stride_o_token;
    } else {
      q_start = seq_idx * seqlen;
      q_end = q_start + seqlen;
      x_offset = seq_idx * stride_x_seq;
      o_offset = seq_idx * stride_o_seq;
    }
    if (q_start == q_end) {
      return;
    }

    const int state_row =
        conv_state_indices[seq_idx * stride_state_indices];
    if (state_row == pad_slot_id || state_row >= num_cache_lines) {
      return;
    }

    int conv_state_token_offset = 0;
    if (is_spec_decoding) {
      conv_state_token_offset = num_accepted_tokens[seq_idx] - 1;
      if (conv_state_token_offset < 0) {
        conv_state_token_offset = 0;
      }
    }

    const T* x_base = x + x_offset + feat * stride_x_dim;
    T* o_base = out + o_offset + feat * stride_o_dim;
    const T* w_base = weight + feat * stride_w_dim;
    T* state_base =
        conv_state + state_row * stride_state_seq + feat * stride_state_dim;

    const float bias_val =
        (has_bias && bias != nullptr) ? static_cast<float>(bias[feat]) : 0.0f;

    float w_reg[WIDTH];
#pragma unroll
    for (int k = 0; k < WIDTH; ++k) {
      w_reg[k] = static_cast<float>(w_base[k * stride_w_width]);
    }

    auto load_prev = [&](int src_token) -> float {
      if (src_token >= 0) {
        return static_cast<float>(x_base[src_token * stride_x_token]);
      }
      const int state_pos = state_len + src_token + conv_state_token_offset;
      if (state_pos < 0 || state_pos >= (state_len + conv_state_token_offset)) {
        return 0.0f;
      }
      return static_cast<float>(state_base[state_pos * stride_state_tok]);
    };

    for (int t = 0; t < seq_len; ++t) {
      float acc = bias_val;
#pragma unroll
      for (int k = 0; k < WIDTH; ++k) {
        const int src_token = t - (WIDTH - 1 - k);
        const float xv = load_prev(src_token);
        acc += xv * w_reg[k];
      }
      if (act_mode == ActMode::silu || act_mode == ActMode::swish) {
        acc = silu(acc);
      }
      o_base[t * stride_o_token] = static_cast<T>(acc);
    }

    if (state_len > 0) {
      for (int s = 0; s < state_len; ++s) {
        const int src_token = seq_len - state_len + s;
        float sv = 0.0f;
        if (src_token >= 0) {
          sv = static_cast<float>(x_base[src_token * stride_x_token]);
        } else {
          const int prev_pos = state_len + src_token + conv_state_token_offset;
          if (prev_pos >= 0 && prev_pos < (state_len + conv_state_token_offset)) {
            sv = static_cast<float>(state_base[prev_pos * stride_state_tok]);
          }
        }
        state_base[s * stride_state_tok] = static_cast<T>(sv);
      }
    }
  }

 private:
  T* out;
  const T* x;
  const T* weight;
  const T* bias;
  T* conv_state;
  const int32_t* conv_state_indices;
  const int32_t* num_accepted_tokens;
  const int32_t* query_start_loc;
  const int batch;
  const int dim;
  const int seqlen;
  const int width;
  const int state_len;
  const int num_cache_lines;
  const int stride_x_seq;
  const int stride_x_dim;
  const int stride_x_token;
  const int stride_w_dim;
  const int stride_w_width;
  const int stride_state_seq;
  const int stride_state_dim;
  const int stride_state_tok;
  const int stride_state_indices;
  const int stride_o_seq;
  const int stride_o_dim;
  const int stride_o_token;
  const int pad_slot_id;
  const bool has_bias;
  const bool is_varlen;
  const bool is_spec_decoding;
  const ActMode act_mode;
};

}  // namespace vllm::xpu::causal_conv1d
