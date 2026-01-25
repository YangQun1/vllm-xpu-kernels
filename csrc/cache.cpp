#include <sycl/sycl.hpp>

#include <algorithm>
#include <iostream>
#include <string>

#include "dispatch_utils.h"
#include "quantization/fp8/quant_utils.h"
#include "utils.h"

namespace vllm {

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
class reshape_and_cache_kernel {
 public:
  reshape_and_cache_kernel(
      const scalar_t* __restrict__ key,
      const scalar_t* __restrict__ value,
      cache_t* __restrict__ key_cache,
      cache_t* __restrict__ value_cache,
      const int64_t* __restrict__ slot_mapping,
      int key_stride,
      int value_stride,
      int num_heads,
      int head_size,
      int block_size,
      int x,
      const float* k_scale,
      const float* v_scale)
      : key_(key),
        value_(value),
        key_cache_(key_cache),
        value_cache_(value_cache),
        slot_mapping_(slot_mapping),
        key_stride_(key_stride),
        value_stride_(value_stride),
        num_heads_(num_heads),
        head_size_(head_size),
        block_size_(block_size),
        x_(x),
        k_scale_(k_scale),
        v_scale_(v_scale) {}

  void operator()(const sycl::nd_item<1>& item) const {
    int group_idx = item.get_group(0);
    int local_idx = item.get_local_id(0);
    int local_range = item.get_local_range(0);
    int slot_idx = slot_mapping_[group_idx];
    if (slot_idx < 0) return;

    const int block_idx = slot_idx / block_size_;
    const int block_offset = slot_idx % block_size_;
    const int n = num_heads_ * head_size_;
    for (int i = local_idx; i < n; i += local_range) {
      const int src_key_idx = group_idx * key_stride_ + i;
      const int src_value_idx = group_idx * value_stride_ + i;
      const int head_idx = i / head_size_;
      const int head_offset = i % head_size_;
      const int x_idx = head_offset / x_;
      const int x_offset = head_offset % x_;
      const int dst_key_idx =
          block_idx * num_heads_ * (head_size_ / x_) * block_size_ * x_ +
          head_idx * (head_size_ / x_) * block_size_ * x_ +
          x_idx * block_size_ * x_ + block_offset * x_ + x_offset;
      const int dst_value_idx = block_idx * n * block_size_ +
                                head_idx * head_size_ * block_size_ +
                                head_offset * block_size_ + block_offset;
      scalar_t tgt_key = key_[src_key_idx];
      scalar_t tgt_value = value_[src_value_idx];
      if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
        key_cache_[dst_key_idx] =
            static_cast<at::Float8_e5m2>(tgt_key * (*k_scale_));
        value_cache_[dst_value_idx] =
            static_cast<at::Float8_e5m2>(tgt_value * (*v_scale_));
      } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
        key_cache_[dst_key_idx] =
            static_cast<at::Float8_e4m3fn>(tgt_key * (*k_scale_));
        value_cache_[dst_value_idx] =
            static_cast<at::Float8_e4m3fn>(tgt_value * (*v_scale_));
      } else {  // kv_dt == Fp8KVCacheDataType::kAuto
        key_cache_[dst_key_idx] = tgt_key;
        value_cache_[dst_value_idx] = tgt_value;
      }
    }
  }

 private:
  const scalar_t* __restrict__ key_;    // [num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ value_;  // [num_tokens, num_heads, head_size]
  cache_t* __restrict__ key_cache_;     // [num_blocks, num_heads, head_size/x,
                                        // block_size, x]
  cache_t* __restrict__ value_cache_;   // [num_blocks, num_heads, head_size,
                                        // block_size]
  const int64_t* __restrict__ slot_mapping_;  // [num_tokens]
  const int key_stride_;
  const int value_stride_;
  const int num_heads_;
  const int head_size_;
  const int block_size_;
  const int x_;
  const float* k_scale_;
  const float* v_scale_;
};

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
class reshape_and_cache_flash_kernel {
 public:
  reshape_and_cache_flash_kernel(
      const scalar_t* __restrict__ key,
      const scalar_t* __restrict__ value,
      cache_t* __restrict__ key_cache,
      cache_t* __restrict__ value_cache,
      const int64_t* __restrict__ slot_mapping,
      const int64_t block_stride,
      const int64_t page_stride,
      const int64_t head_stride,
      const int64_t key_stride,
      const int64_t value_stride,
      const int num_heads,
      const int head_size,
      const int block_size,
      const float* k_scale,
      const float* v_scale)
      : key_(key),
        value_(value),
        key_cache_(key_cache),
        value_cache_(value_cache),
        slot_mapping_(slot_mapping),
        block_stride_(block_stride),
        page_stride_(page_stride),
        head_stride_(head_stride),
        key_stride_(key_stride),
        value_stride_(value_stride),
        num_heads_(num_heads),
        head_size_(head_size),
        block_size_(block_size),
        k_scale_(k_scale),
        v_scale_(v_scale) {}

  void operator()(const sycl::nd_item<1>& item) const {
    int64_t group_idx = item.get_group(0);
    int64_t local_idx = item.get_local_id(0); // the wi id in a wg
    int local_range = item.get_local_range(0); // the num of wi in a wg
    int64_t slot_idx = slot_mapping_[group_idx];
    if (slot_idx < 0) return;

    const int64_t block_idx = slot_idx / block_size_;
    const int64_t block_offset = slot_idx % block_size_;
    const int n = num_heads_ * head_size_;

    // pointers to the beginning of the source row for this token.
    const scalar_t* __restrict__ key_src = key_ + group_idx * key_stride_;
    const scalar_t* __restrict__ value_src = value_ + group_idx * value_stride_;

    // find the start position inside the kv-cache for this token.
    cache_t* __restrict__ key_dst =
        key_cache_ + block_idx * block_stride_ + block_offset * page_stride_;
    cache_t* __restrict__ value_dst =
        value_cache_ + block_idx * block_stride_ + block_offset * page_stride_;

    float k_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *k_scale_;
    float v_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *v_scale_;

    fp8::CopyWithScaleOp<cache_t, scalar_t, kv_dt> k_op{k_scale_val};
    fp8::CopyWithScaleOp<cache_t, scalar_t, kv_dt> v_op{v_scale_val};
    fp8::scaled_convert_vec(key_src, key_dst, n, local_idx, local_range, k_op);
    fp8::scaled_convert_vec(
        value_src, value_dst, n, local_idx, local_range, v_op);
  }

 private:
  const scalar_t* __restrict__ key_;    // [num_tokens, num_heads, head_size]
  const scalar_t* __restrict__ value_;  // [num_tokens, num_heads, head_size]
  cache_t* __restrict__ key_cache_;     // [num_blocks, block_size, num_heads,
                                        // head_size]
  cache_t* __restrict__ value_cache_;   // [num_blocks, block_size, num_heads,
                                        // head_size]
  const int64_t* __restrict__ slot_mapping_;  // [num_tokens]
  const int64_t block_stride_;
  const int64_t page_stride_;
  const int64_t head_stride_;
  const int64_t key_stride_;
  const int64_t value_stride_;
  const int num_heads_;
  const int head_size_;
  const int block_size_;
  const float* k_scale_;
  const float* v_scale_;
};

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
class concat_and_cache_mla_kernel {
 public:
  concat_and_cache_mla_kernel(
      const scalar_t* __restrict__ kv_c,
      const scalar_t* __restrict__ k_pe,
      cache_t* __restrict__ kv_cache,
      const int64_t* __restrict__ slot_mapping,
      const int block_stride,
      const int entry_stride,
      const int kv_c_stride,
      const int k_pe_stride,
      const int kv_lora_rank,
      const int pe_dim,
      const int block_size,
      const float* scale)
      : kv_c(kv_c),
        k_pe(k_pe),
        kv_cache(kv_cache),
        slot_mapping(slot_mapping),
        block_stride(block_stride),
        entry_stride(entry_stride),
        kv_c_stride(kv_c_stride),
        k_pe_stride(k_pe_stride),
        kv_lora_rank(kv_lora_rank),
        pe_dim(pe_dim),
        block_size(block_size),
        scale(scale) {}

  void operator()(const sycl::nd_item<1> item_id) const {
    const int64_t token_idx = item_id.get_group(0);
    const int64_t slot_idx = slot_mapping[token_idx];
    // NOTE: slot_idx can be -1 if the token is padded
    if (slot_idx < 0) {
      return;
    }
    const int block_idx = slot_idx / block_size;
    const int block_offset = slot_idx % block_size;

    auto copy = [&](const scalar_t* __restrict__ src,
                    cache_t* dst,
                    int src_stride,
                    int dst_stride,
                    int size,
                    int offset) {
      for (int i = item_id.get_local_id(0); i < size;
           i += item_id.get_local_range(0)) {
        const int src_idx = token_idx * src_stride + i;
        const int dst_idx =
            block_idx * block_stride + block_offset * entry_stride + i + offset;
        if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
          dst[dst_idx] = static_cast<at::Float8_e4m3fn>(src[src_idx] * *scale);
        } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
          dst[dst_idx] = static_cast<at::Float8_e5m2>(src[src_idx] * *scale);
        } else {
          dst[dst_idx] = src[src_idx];
        }
      }
    };

    copy(kv_c, kv_cache, kv_c_stride, block_stride, kv_lora_rank, 0);
    copy(k_pe, kv_cache, k_pe_stride, block_stride, pe_dim, kv_lora_rank);
  }

 private:
  const scalar_t* __restrict__ kv_c;  // [num_tokens, kv_lora_rank]
  const scalar_t* __restrict__ k_pe;  // [num_tokens, pe_dim]
  cache_t* __restrict__ kv_cache;  // [num_blocks, block_size, (kv_lora_rank +
                                   // pe_dim)]
  const int64_t* __restrict__ slot_mapping;  // [num_tokens]
  const int block_stride;                    //
  const int entry_stride;                    //
  const int kv_c_stride;                     //
  const int k_pe_stride;                     //
  const int kv_lora_rank;                    //
  const int pe_dim;                          //
  const int block_size;                      //
  const float* scale;                        //
};

// grid is launched with dimensions (batch, num_splits)
template <typename scalar_t>
class gather_cache_kernel {
 public:
  gather_cache_kernel(
      const scalar_t* __restrict__ src_cache,
      scalar_t* __restrict__ dst,
      const int32_t* __restrict__ block_table,
      const int32_t* __restrict__ cu_seq_lens,
      const int32_t block_size,
      const int32_t entry_size,
      const int64_t block_table_stride,
      const int64_t cache_block_stride,
      const int64_t cache_entry_stride,
      const int64_t dst_entry_stride,
      const int32_t* __restrict__ seq_starts)
      : src_cache(src_cache),
        dst(dst),
        block_table(block_table),
        cu_seq_lens(cu_seq_lens),
        block_size(block_size),
        entry_size(entry_size),
        block_table_stride(block_table_stride),
        cache_block_stride(cache_block_stride),
        cache_entry_stride(cache_entry_stride),
        dst_entry_stride(dst_entry_stride),
        seq_starts(seq_starts) {}

  void operator()(const sycl::nd_item<2> item_id) const {
    const int64_t bid = item_id.get_group(1);  // Batch ID
    const int32_t num_splits = item_id.get_global_range(0);
    const int32_t split = item_id.get_group(0);
    const int32_t seq_start = cu_seq_lens[bid];
    const int32_t seq_end = cu_seq_lens[bid + 1];
    const int32_t seq_len = seq_end - seq_start;
    const int32_t tot_blocks = (seq_len + block_size - 1) / block_size;
    const int32_t split_blocks = (tot_blocks + num_splits - 1) / num_splits;

    const int32_t split_start = split * split_blocks;
    const int32_t split_end = std::min((split + 1) * split_blocks, tot_blocks);

    const bool is_active_split = (split_start < tot_blocks);
    const bool is_last_split = (split_end == tot_blocks);

    if (!is_active_split) return;

    int32_t full_blocks_end = split_end;
    int32_t partial_block_size = 0;

    // Adjust the pointer for the block_table for this batch.
    // If seq_starts is provided, compute an offset based on (seq_starts[bid] /
    // page_size)
    const int32_t batch_offset = bid * block_table_stride;
    int32_t offset = 0;
    if (seq_starts != nullptr) {
      offset = seq_starts[bid] / block_size;
    }
    const int32_t* batch_block_table = block_table + batch_offset + offset;

    // Adjust dst pointer based on the cumulative sequence lengths.
    scalar_t* dst_ptr = dst + seq_start * dst_entry_stride;

    if (is_last_split) {
      partial_block_size = seq_len % block_size;
      if (partial_block_size) full_blocks_end -= 1;
    }

    auto copy_entry = [&](const scalar_t* __restrict__ _src,
                          scalar_t* __restrict__ _dst) {
      for (int i = item_id.get_local_id(1); i < entry_size;
           i += item_id.get_local_range(1))
        _dst[i] = _src[i];
    };

    for (int pid = split_start; pid < full_blocks_end; ++pid) {
      auto block_id = batch_block_table[pid];
      auto block_start_ptr = src_cache + block_id * cache_block_stride;
      auto block_dst_ptr = dst_ptr + pid * block_size * dst_entry_stride;
      for (int eid = 0; eid < block_size; ++eid) {
        copy_entry(
            block_start_ptr + eid * cache_entry_stride,
            block_dst_ptr + eid * dst_entry_stride);
      }
    }

    if (partial_block_size) {
      auto block_id = batch_block_table[full_blocks_end];
      auto block_start_ptr = src_cache + block_id * cache_block_stride;
      auto block_dst_ptr =
          dst_ptr + full_blocks_end * block_size * dst_entry_stride;
      for (int eid = 0; eid < partial_block_size; ++eid) {
        copy_entry(
            block_start_ptr + eid * cache_entry_stride,
            block_dst_ptr + eid * dst_entry_stride);
      }
    }
  }

 private:
  const scalar_t* __restrict__ src_cache;   // [NUM_BLOCKS, BLOCK_SIZE,
                                            // ENTRIES...]
  scalar_t* __restrict__ dst;               // [TOT_TOKENS, ENTRIES...]
  const int32_t* __restrict__ block_table;  // [BATCH, BLOCK_INDICES]
  const int32_t* __restrict__ cu_seq_lens;  // [BATCH+1]
  const int32_t block_size;
  const int32_t entry_size;
  const int64_t block_table_stride;
  const int64_t cache_block_stride;
  const int64_t cache_entry_stride;
  const int64_t dst_entry_stride;
  const int32_t* __restrict__ seq_starts;  // Optional: starting offsets per
};


template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
class indexer_k_quant_and_cache_kernel {
  public:
  indexer_k_quant_and_cache_kernel(
      const scalar_t* __restrict__ k,  // [num_tokens, head_dim]
      cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, cache_stride]
      const int64_t* __restrict__ slot_mapping,  // [num_tokens]
      const int head_dim,                        // dimension of each head
      const int quant_block_size,                // quantization block size
      const int cache_block_size,                // cache block size
      const int cache_stride,                    // stride for each token in kv_cache
      const bool use_ue8m0                       // use ue8m0 scale format
  ) : k_(k), kv_cache_(kv_cache), slot_mapping_(slot_mapping),
      head_dim_(head_dim), quant_block_size_(quant_block_size),
      cache_block_size_(cache_block_size), cache_stride_(cache_stride),
      use_ue8m0_(use_ue8m0) {}

  void operator()(const sycl::nd_item<2>& item) const {
    static_assert(sizeof(scalar_t) == 2, "Only 16 bits dtype is supported for k tensor.");

    constexpr int VEC_SIZE = 4;

    auto blockIdx_x = item.get_global_id(0);
    auto blockIdx_y = item.get_global_id(1);
    auto blockDim_x = item.get_local_range(0);
    auto blockDim_y = item.get_local_range(1);
    auto threadIdx_x = item.get_local_id(0);
    auto threadIdx_y = item.get_local_id(1);

    const int64_t token_idx = item.get_global_id(0);
    const int64_t head_dim_idx = (blockIdx_y * blockDim_y * blockDim_x +
                                threadIdx_y * blockDim_x + threadIdx_x) *
                               VEC_SIZE;
    const int64_t slot_idx = slot_mapping_[token_idx];
    const int64_t block_idx = slot_idx / cache_block_size_;
    const int64_t block_offset = slot_idx % cache_block_size_;

    // NOTE: slot_idx can be -1 if the token is padded
    if (slot_idx < 0 || (head_dim_idx >= head_dim_)) {
      return;
    }

    using srcx4_t = vec4_t<scalar_t>;

    srcx4_t k_val = (reinterpret_cast<srcx4_t const*>(k_))[(token_idx * head_dim_ + head_dim_idx) / VEC_SIZE];
    scalar_t* k_val_ptr = reinterpret_cast<scalar_t*>(&k_val);
    float amax = 0.0f;
    for (int i = 0; i < VEC_SIZE; i++) {
      amax = sycl::max(amax, sycl::fabs(float(k_val_ptr[i])));
    }

    // Reduced amax
    for (int mask = 16; mask > 0; mask /= 2) {
      amax = sycl::max(amax, sycl::sub_group::reduce_over_group(item.get_sub_group(), amax));
    }

    float scale = sycl::max(amax, 1e-4f) / 448.0f;
    if (use_ue8m0_) {
      scale = sycl::exp2(sycl::ceil(sycl::log2(scale)));
    }

    fp8::CopyWithScaleOp<cache_t, scalar_t, kv_dt> k_op{scale};

    const int64_t dst_offset = block_idx * cache_block_size_ * cache_stride_ +
                                block_offset * head_dim_ + head_dim_idx;
    for (int i = 0; i < VEC_SIZE; i++) {
      k_op(kv_cache_[dst_offset + i], k_val_ptr[i]);
    }
    if (threadIdx_x == 0) {
      const int64_t dst_scale_idx =
          block_idx * cache_block_size_ * cache_stride_ +
          cache_block_size_ * head_dim_ +
          (block_offset * head_dim_ + head_dim_idx) * 4 / quant_block_size_;
      reinterpret_cast<float*>(kv_cache_)[dst_scale_idx / 4] = scale;
    }
  }

  private:
  const scalar_t* __restrict__ k_;  // [num_tokens, head_dim]
  cache_t* __restrict__ kv_cache_;   // [num_blocks, block_size, cache_stride]
  const int64_t* __restrict__ slot_mapping_;  // [num_tokens]
  const int head_dim_;
  const int quant_block_size_;
  const int cache_block_size_;
  const int cache_stride_;
  const bool use_ue8m0_;
};



template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
class concat_and_cache_ds_mla_kernel {
 public:
  concat_and_cache_ds_mla_kernel(
      const scalar_t* __restrict__ kv_c,
      const scalar_t* __restrict__ k_pe,
      cache_t* __restrict__ kv_cache,
      const int64_t* __restrict__ slot_mapping,
      const int block_stride,
      const int entry_stride,
      const int kv_c_stride,
      const int k_pe_stride,
      const int kv_lora_rank,
      const int pe_dim,
      const int block_size,
      const float* scale)
      : kv_c(kv_c),
        k_pe(k_pe),
        kv_cache(kv_cache),
        slot_mapping(slot_mapping),
        block_stride(block_stride),
        entry_stride(entry_stride),
        kv_c_stride(kv_c_stride),
        k_pe_stride(k_pe_stride),
        kv_lora_rank(kv_lora_rank),
        pe_dim(pe_dim),
        block_size(block_size),
        scale(scale) {}

  void operator()(const sycl::nd_item<1> item) const {
    auto blockIdx_x = item.get_global_id(0);
    auto blockIdx_y = item.get_global_id(1);
    auto blockDim_x = item.get_local_range(0);
    auto blockDim_y = item.get_local_range(1);
    auto threadIdx_x = item.get_local_id(0);
    auto threadIdx_y = item.get_local_id(1);

    const int64_t token_idx = blockIdx_x;
    const int64_t slot_idx = slot_mapping[token_idx];
    // NOTE: slot_idx can be -1 if the token is padded
    if (slot_idx < 0) {
      return;
    }
    const int64_t block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;
    const int64_t dst_idx_start =
        block_idx * block_stride + block_offset * entry_stride;

    // For the NoPE part, each tile of 128 elements is handled by half of one warp
    // (16 threads). There are 4 total tiles, so 2 warps (64 threads).
    // Lanes 0 and 16 of each warp write the scale values for that warp's tiles.
    // The RoPE part (last 64 elements) is handled by another 1 warp (32 threads).
    // So in total, we use 3 warps (96 threads) per block.

    // Cast kv_cache to 16_bit for RoPE values
    scalar_t* kv_cache_16bit =
        reinterpret_cast<scalar_t*>(&kv_cache[dst_idx_start]);

    // The last warp handles the RoPE part
    if (threadIdx_x >= 64) {
      // Each thread handles two elements of RoPE
      const int8_t pe_idx_start = (threadIdx_x - 64) * 2;
      const int64_t src_idx = token_idx * k_pe_stride + pe_idx_start;
      // Vectorized load of two 16-bit values, performed as one 32-bit load
      const int32_t vals = *reinterpret_cast<const int32_t*>(&k_pe[src_idx]);
      // RoPE values start after the packed 8-bit NoPE values and the
      // 32-bit scales
      const int64_t dst_idx = kv_lora_rank / 2 + 8 + pe_idx_start;
      // Vectorized store of two 16-bit values, performed as one 32-bit store
      *reinterpret_cast<int32_t*>(&kv_cache_16bit[dst_idx]) = vals;
      return;
    }

    // The first two warps handle the NoPE part
    const int8_t warp_idx = threadIdx_x >> 5;
    const int8_t lane_idx = threadIdx_x & 31;
    const int8_t tile_idx = warp_idx * 2 + (lane_idx >> 4);

    // Each thread handles 8 elements of NoPE
    // Load the NoPE elements for this thread into registers
    const int64_t src_idx_start = token_idx * kv_c_stride + (threadIdx_x * 8);
    // Vectorized load of eight 16-bit values, performed as an int4 load
    using int4 = vec4_t<int32_t>;
    const int4 vals_i4 = *reinterpret_cast<const int4*>(&kv_c[src_idx_start]);
    const scalar_t* vals = reinterpret_cast<const scalar_t*>(&vals_i4);

    // Max absolute value of this thread's elements
    float max_abs = fmaxf(fmaxf(fmaxf(fabsf(vals[0]), fabsf(vals[1])),
                                fmaxf(fabsf(vals[2]), fabsf(vals[3]))),
                          fmaxf(fmaxf(fabsf(vals[4]), fabsf(vals[5])),
                                fmaxf(fabsf(vals[6]), fabsf(vals[7]))));

    // Warp-level reduction to find the max absolute value in each half-warp
  #pragma unroll
    for (int offset = 8; offset > 0; offset /= 2) {
      max_abs = fmaxf(max_abs, VLLM_SHFL_XOR_SYNC_WIDTH(max_abs, offset, 16));
    }

    // Compute the scale for the tile
    float tile_scale = max_abs / 448.f;
    tile_scale = fmaxf(tile_scale, FLT_MIN);

    // The first lane of each half-warp writes the scale to kv_cache
    if ((lane_idx == 0) || (lane_idx == 16)) {
      float* kv_cache_32bit = reinterpret_cast<float*>(&kv_cache[dst_idx_start]);
      const uint64_t dst_idx = kv_lora_rank / 4 + tile_idx;
      kv_cache_32bit[dst_idx] = tile_scale;
    }

    // Now all threads in the block scale and write their elements
    // NoPE data is packed in the first kv_lora_rank/2 bytes (first 256 bytes)
    const int64_t dst_idx_base = dst_idx_start + (threadIdx_x * 8);

    fp8::CopyWithScaleOp<uint8_t, scalar_t, kv_dt> val_op{tile_scale};

    uint8_t result[8];
  #pragma unroll
    for (int i = 0; i < 8; i++) {
      val_op(result[i], vals[i]);
    }

    // Store as aligned 64-bit writes
    *reinterpret_cast<uint64_t*>(&kv_cache[dst_idx_base]) =
        *reinterpret_cast<const uint64_t*>(result);
  }

 private:
  const scalar_t* __restrict__ k_pe;  // [num_tokens, pe_dim]
  cache_t* __restrict__ kv_cache;  // [num_blocks, block_size, (kv_lora_rank +
                                   // pe_dim)]
  const int64_t* __restrict__ slot_mapping;  // [num_tokens]
  const int block_stride;                    //
  const int entry_stride;                    //
  const int kv_c_stride;                     //
  const int k_pe_stride;                     //
  const int kv_lora_rank;                    //
  const int pe_dim;                          //
  const int block_size;                      //
  const float* scale;                        //
};

}  // namespace vllm

// KV_T is the stored data type of kv-cache.
// CACHE_T is the data type of key and value tensors.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_RESHAPE_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)           \
  queue.submit([&](sycl::handler& cgh) {                          \
    cgh.parallel_for(                                             \
        sycl::nd_range<1>(grid * block, block),                   \
        vllm::reshape_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE>(  \
            reinterpret_cast<KV_T*>(key.data_ptr()),              \
            reinterpret_cast<KV_T*>(value.data_ptr()),            \
            reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),     \
            reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),   \
            slot_mapping.data_ptr<int64_t>(),                     \
            key_stride,                                           \
            value_stride,                                         \
            num_heads,                                            \
            head_size,                                            \
            block_size,                                           \
            x,                                                    \
            reinterpret_cast<const float*>(k_scale.data_ptr()),   \
            reinterpret_cast<const float*>(v_scale.data_ptr()))); \
  });

void reshape_and_cache(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    torch::Tensor& k_scale,
    torch::Tensor& v_scale) {
  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);

  sycl::range<1> grid(num_tokens);
  sycl::range<1> block(std::min(num_heads * head_size, 1024));
  const at::DeviceGuard device_guard(key.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  DISPATCH_BY_KV_CACHE_DTYPE(
      key.scalar_type(), kv_cache_dtype, CALL_RESHAPE_AND_CACHE);
}

// KV_T is the stored data type of kv-cache.
// CACHE_T is the data type of key and value tensors.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_RESHAPE_AND_CACHE_FLASH(KV_T, CACHE_T, KV_DTYPE)          \
  queue.submit([&](sycl::handler& cgh) {                               \
    cgh.parallel_for(                                                  \
        sycl::nd_range<1>(grid * block, block),                        \
        vllm::reshape_and_cache_flash_kernel<KV_T, CACHE_T, KV_DTYPE>( \
            reinterpret_cast<KV_T*>(key.data_ptr()),                   \
            reinterpret_cast<KV_T*>(value.data_ptr()),                 \
            reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),          \
            reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),        \
            slot_mapping.data_ptr<int64_t>(),                          \
            block_stride,                                              \
            page_stride,                                               \
            head_stride,                                               \
            key_stride,                                                \
            value_stride,                                              \
            num_heads,                                                 \
            head_size,                                                 \
            block_size,                                                \
            reinterpret_cast<const float*>(k_scale.data_ptr()),        \
            reinterpret_cast<const float*>(v_scale.data_ptr())));      \
  });

void reshape_and_cache_flash(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    torch::Tensor& k_scale,
    torch::Tensor& v_scale) {
  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(1);

  int64_t key_stride = key.stride(0);
  int64_t value_stride = value.stride(0);
  int64_t block_stride = key_cache.stride(0);
  int64_t page_stride = key_cache.stride(1);
  int64_t head_stride = key_cache.stride(2);
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

  sycl::range<1> grid(num_tokens);
  sycl::range<1> block(std::min(num_heads * head_size, 1024));
  const at::DeviceGuard device_guard(key.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  DISPATCH_BY_KV_CACHE_DTYPE(
      key.scalar_type(), kv_cache_dtype, CALL_RESHAPE_AND_CACHE_FLASH);
}

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_CONCAT_AND_CACHE_MLA(KV_T, CACHE_T, KV_DTYPE)          \
  queue.submit([&](sycl::handler& cgh) {                            \
    cgh.parallel_for(                                               \
        sycl::nd_range<1>(grid * block, block),                     \
        vllm::concat_and_cache_mla_kernel<KV_T, CACHE_T, KV_DTYPE>( \
            reinterpret_cast<KV_T*>(kv_c.data_ptr()),               \
            reinterpret_cast<KV_T*>(k_pe.data_ptr()),               \
            reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),        \
            slot_mapping.data_ptr<int64_t>(),                       \
            block_stride,                                           \
            entry_stride,                                           \
            kv_c_stride,                                            \
            k_pe_stride,                                            \
            kv_lora_rank,                                           \
            pe_dim,                                                 \
            block_size,                                             \
            reinterpret_cast<const float*>(scale.data_ptr())));     \
  });

#define CALL_CONCAT_AND_CACHE_DS_MLA(KV_T, CACHE_T, KV_DTYPE)          \
  queue.submit([&](sycl::handler& cgh) {                            \
    cgh.parallel_for(                                               \
        sycl::nd_range<1>(grid * block, block),                     \
        vllm::concat_and_cache_ds_mla_kernel<KV_T, CACHE_T, KV_DTYPE>( \
            reinterpret_cast<KV_T*>(kv_c.data_ptr()),               \
            reinterpret_cast<KV_T*>(k_pe.data_ptr()),               \
            reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),        \
            slot_mapping.data_ptr<int64_t>(),                       \
            block_stride,                                           \
            entry_stride,                                           \
            kv_c_stride,                                            \
            k_pe_stride,                                            \
            kv_lora_rank,                                           \
            pe_dim,                                                 \
            block_size,                                             \
            reinterpret_cast<const float*>(scale.data_ptr())));     \
  });

void concat_and_cache_mla(
    torch::Tensor& kv_c,          // [num_tokens, kv_lora_rank]
    torch::Tensor& k_pe,          // [num_tokens, pe_dim]
    torch::Tensor& kv_cache,      // [num_blocks, block_size, (kv_lora_rank +
                                  // pe_dim)]
    torch::Tensor& slot_mapping,  // [num_tokens] or [num_actual_tokens]
    const std::string& kv_cache_dtype,
    torch::Tensor& scale) {
  // NOTE(woosuk): In vLLM V1, key.size(0) can be different from
  // slot_mapping.size(0) because of padding for CUDA graphs.
  // In vLLM V0, key.size(0) is always equal to slot_mapping.size(0) because
  // both include padding.
  // In vLLM V1, however, key.size(0) can be larger than slot_mapping.size(0)
  // since key includes padding for CUDA graphs, while slot_mapping does not.
  // In this case, slot_mapping.size(0) represents the actual number of tokens
  // before padding.
  // For compatibility with both cases, we use slot_mapping.size(0) as the
  // number of tokens.
  int num_tokens = slot_mapping.size(0);
  int kv_lora_rank = kv_c.size(1);
  int pe_dim = k_pe.size(1);
  int block_size = kv_cache.size(1);

  TORCH_CHECK(kv_cache.size(2) == kv_lora_rank + pe_dim);

  int kv_c_stride = kv_c.stride(0);
  int k_pe_stride = k_pe.stride(0);
  int block_stride = kv_cache.stride(0);
  int entry_stride = kv_cache.stride(1);

  const at::DeviceGuard device_guard(kv_c.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  if (kv_cache_dtype == "fp8_ds_mla") {
    sycl::range<1> grid(num_tokens);
    // For the NoPE part, each tile of 128 elements is handled by half of one
    // warp (16 threads). There are 4 total tiles, so 2 warps (64 threads).
    // Lanes 0 and 16 of each warp write the scale values for that warp's tiles.
    // The RoPE part (last 64 elements) is handled by another 1 warp (32
    // threads). So in total, we use 3 warps (96 threads) per block.
    sycl::range<1> block(96);
    DISPATCH_BY_KV_CACHE_DTYPE(kv_c.scalar_type(), kv_cache_dtype,
                               CALL_CONCAT_AND_CACHE_DS_MLA);
  } else {
    sycl::range<1> grid(num_tokens);
    sycl::range<1> block(std::min(kv_lora_rank, 1024));
    DISPATCH_BY_KV_CACHE_DTYPE(
        kv_c.scalar_type(), kv_cache_dtype, CALL_CONCAT_AND_CACHE_MLA);
  }
}

// Macro to dispatch the kernel based on the data type.
#define CALL_GATHER_CACHE(CPY_DTYPE)                            \
  queue.submit([&](sycl::handler& cgh) {                        \
    cgh.parallel_for(                                           \
        sycl::nd_range<2>(grid * block, block),                 \
        vllm::gather_cache_kernel<CPY_DTYPE>(                   \
            reinterpret_cast<CPY_DTYPE*>(src_cache.data_ptr()), \
            reinterpret_cast<CPY_DTYPE*>(dst.data_ptr()),       \
            block_table.data_ptr<int32_t>(),                    \
            cu_seq_lens.data_ptr<int32_t>(),                    \
            block_size,                                         \
            entry_size,                                         \
            block_table_stride,                                 \
            cache_block_stride,                                 \
            cache_entry_stride,                                 \
            dst_entry_stride,                                   \
            seq_starts_ptr));                                   \
  });

// Gather sequences from the cache into the destination tensor.
//  - cu_seq_lens contains the cumulative sequence lengths for each batch
//  - block_table contains the cache block indices for each sequence
//  - Optionally, seq_starts (if provided) offsets the starting block index by
//  (seq_starts[bid] / page_size)
void gather_cache(
    torch::Tensor const& src_cache,    // [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    torch::Tensor const& dst,          // [TOT_TOKENS, ENTRIES...]
    torch::Tensor const& block_table,  // [BATCH, BLOCK_INDICES]
    torch::Tensor const& cu_seq_lens,  // [BATCH+1]
    int64_t batch_size,
    std::optional<torch::Tensor> seq_starts = std::nullopt) {
  const at::DeviceGuard device_guard(src_cache.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  int32_t block_size = src_cache.size(1);
  int32_t entry_size = src_cache.flatten(2, -1).size(2);

  TORCH_CHECK(block_table.dtype() == at::kInt, "block_table must be int32");
  TORCH_CHECK(cu_seq_lens.dtype() == at::kInt, "cu_seq_lens must be int32");
  if (seq_starts.has_value()) {
    TORCH_CHECK(
        seq_starts.value().dtype() == at::kInt, "seq_starts must be int32");
  }

  TORCH_CHECK(
      src_cache.device() == dst.device(),
      "src_cache and dst must be on the same device");
  TORCH_CHECK(
      src_cache.device() == block_table.device(),
      "src_cache and block_table must be on the same device");
  TORCH_CHECK(
      src_cache.device() == cu_seq_lens.device(),
      "src_cache and cu_seq_lens must be on the same device");
  if (seq_starts.has_value()) {
    TORCH_CHECK(
        src_cache.device() == seq_starts.value().device(),
        "src_cache and seq_starts must be on the same device");
  }

  int64_t block_table_stride = block_table.stride(0);
  int64_t cache_block_stride = src_cache.stride(0);
  int64_t cache_entry_stride = src_cache.stride(1);
  int64_t dst_entry_stride = dst.stride(0);

  // Decide on the number of splits based on the batch size.
  int num_splits = batch_size > 128 ? 2 : batch_size > 64 ? 4 : 16;
  sycl::range<2> grid(num_splits, batch_size);
  sycl::range<2> block(1, 1024);

  TORCH_CHECK(
      src_cache.dtype() == dst.dtype(),
      "src_cache and dst must have the same dtype");

  const int dtype_bits = src_cache.element_size() * 8;
  const int32_t* seq_starts_ptr =
      seq_starts.has_value() ? seq_starts.value().data_ptr<int32_t>() : nullptr;

  if (dtype_bits == 32) {
    CALL_GATHER_CACHE(uint32_t);
  } else if (dtype_bits == 16) {
    CALL_GATHER_CACHE(uint16_t);
  } else if (dtype_bits == 8) {
    CALL_GATHER_CACHE(uint8_t);
  } else {
    TORCH_CHECK(false, "Unsupported data type width: ", dtype_bits);
  }
}

#define CALL_INDEXER_K_QUANT_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)        \
  queue.submit([&](sycl::handler& cgh) {                                 \
    cgh.parallel_for(                                                    \
        sycl::nd_range<2>(grid * block, block),                         \
        vllm::indexer_k_quant_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE>( \
            reinterpret_cast<KV_T*>(k.data_ptr()),                      \
            reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),            \
            slot_mapping.data_ptr<int64_t>(),                            \
            head_dim,                                                   \
            quant_block_size,                                           \
            cache_block_size,                                           \
            cache_stride,                                              \
            use_ue8m0));                                               \
  });


void indexer_k_quant_and_cache(
    torch::Tensor& k,             // [num_tokens, head_dim]
    torch::Tensor& kv_cache,      // [num_blocks, block_size, cache_stride]
    torch::Tensor& slot_mapping,  // [num_tokens]
    int64_t quant_block_size,     // quantization block size
    const std::string& scale_fmt) {
  int num_tokens = k.size(0);
  int head_dim = k.size(1);
  int cache_block_size = kv_cache.size(1);
  int cache_stride = kv_cache.size(2);
  bool use_ue8m0 = scale_fmt == "ue8m0";

  TORCH_CHECK(k.device() == kv_cache.device(),
              "k and kv_cache must be on the same device");
  TORCH_CHECK(k.device() == slot_mapping.device(),
              "k and slot_mapping must be on the same device");
  TORCH_CHECK(head_dim % quant_block_size == 0,
              "head_dim must be divisible by quant_block_size");

  constexpr int vec_size = 4;
  // 纵向，每个token分别由不同的block处理
  // 横向，每个token的head_dim维度被quant_block_size * vec_size个block处理
  // 这样，每个block就会处理一个token的quant_block_size * vec_size部分的数据
  sycl::range<2> grid(num_tokens, (head_dim + quant_block_size * vec_size - 1) /
                            (quant_block_size * vec_size));
  sycl::range<2> block(32, vec_size);
  const at::DeviceGuard device_guard(src_cache.device());
  auto& queue = vllm::xpu::vllmGetQueue();

  DISPATCH_BY_KV_CACHE_DTYPE(k.scalar_type(), "fp8_e4m3",
                             CALL_INDEXER_K_QUANT_AND_CACHE);
}
