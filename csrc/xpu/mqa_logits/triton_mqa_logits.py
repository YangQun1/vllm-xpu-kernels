from typing import Optional, final

import torch
import triton
import triton.language as tl


@triton.jit
def triton_fp8_mqa_logits_kernel(
    # Pointers to tensors
    q_ptr,
    kv_ptr,
    kv_scales_ptr,
    weights_ptr,
    cu_seqlen_ks_ptr,
    cu_seqlen_ke_ptr,
    logits_ptr,
    # Shapes
    seq_len,
    num_heads,
    head_dim,
    seq_len_kv,
    # Meta-parameters
    BLOCK_HEADS: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # Get the strides
    stride_q_s = num_heads * head_dim
    stride_q_h = head_dim
    stride_q_d = 1

    stride_kv_s = head_dim
    stride_kv_d = 1

    stride_kv_scales_s = 1

    stride_weights_s = num_heads
    stride_weights_h = 1

    stride_logits_s = seq_len_kv
    stride_logits_s_kv = 1

    q_token_idx = tl.program_id(0)
    kv_block_idx = tl.program_id(1)
    
    kv_start = kv_block_idx * BLOCK_KV
    
    n_offsets = kv_start + tl.arange(0, BLOCK_KV)
    
    n_mask = n_offsets < seq_len_kv # [BLOCK_KV]
    
    ks_vals = tl.load(cu_seqlen_ks_ptr + q_token_idx)
    ke_vals = tl.load(cu_seqlen_ke_ptr + q_token_idx)
    
    seq_mask = (n_offsets >= ks_vals) & \
               (n_offsets < ke_vals) # [BLOCK_KV]
    
    seq_mask = seq_mask & n_mask  # [BLOCK_KV]
    
    # if the block is masked out, directly write -inf and return
    if tl.sum(seq_mask) == 0:
        out_offsets = q_token_idx * stride_logits_s + n_offsets * stride_logits_s_kv
        tl.store(
            logits_ptr + out_offsets,
            tl.full((BLOCK_KV,), float('-inf'), dtype=tl.float32),
            mask=n_mask
        )
        return

    logits_acc = tl.zeros((BLOCK_KV,), dtype=tl.float32)

    for head_start in tl.range(0, num_heads, BLOCK_HEADS):
        # load q
        q_offsets = (
            q_token_idx * stride_q_s + 
            (head_start + tl.arange(0, BLOCK_HEADS)[:, None]) * stride_q_h + 
            tl.arange(0, HEAD_DIM)[None, :] * stride_q_d
        )
        
        q_block_fp8 = tl.load(
            q_ptr + q_offsets,
        )
        q_block = q_block_fp8.to(tl.float16) # [BLOCK_HEADS, HEAD_DIM]
        
        # load k
        kv_offsets = n_offsets[:, None] * stride_kv_s + tl.arange(0, HEAD_DIM)[None, :] * stride_kv_d
        kv_block_fp8 = tl.load(
            kv_ptr + kv_offsets,
            mask=n_mask[:, None],
            other=0.0
        )
        kv_block = kv_block_fp8.to(tl.float16) # [BLOCK_KV, HEAD_DIM]
        
        head_logits = tl.dot(q_block, kv_block.T).to(tl.float32) # [BLOCK_HEADS, BLOCK_KV]
        
        # load and mul scale
        scale_vals = tl.load(
            kv_scales_ptr + n_offsets * stride_kv_scales_s, 
            mask=n_mask, 
            other=1.0
        ) # [BLOCK_KV]
        head_logits = head_logits * scale_vals[None, :]
        
        # relu
        head_logits = tl.maximum(head_logits, 0.0)  # [BLOCK_HEADS, BLOCK_KV]
        
        # load and mul weight
        weights_vals = tl.load(
            weights_ptr + q_token_idx * stride_weights_s + 
            (head_start + tl.arange(0, BLOCK_HEADS)) * stride_weights_h,
        ) # [BLOCK_HEADS]

        head_logits = head_logits * weights_vals[:, None]  # [BLOCK_HEADS, BLOCK_KV]

        # reduce along num_head
        logits_acc += tl.sum(head_logits, axis=0)  # [BLOCK_KV]
    
    # apply mask
    logits_acc = tl.where(seq_mask, logits_acc, float('-inf'))
    
    out_offsets = q_token_idx * stride_logits_s + n_offsets * stride_logits_s_kv
    tl.store(
        logits_ptr + out_offsets,
        logits_acc,
        mask=n_mask
    )


def triton_fp8_mqa_logits(
    q: torch.Tensor, # [seq_len, num_heads, head_dim]
    kv: tuple[torch.Tensor, torch.Tensor], # [seq_len_kv, head_dim], [seq_len_kv]
    weights: torch.Tensor, # [seq_len, num_heads]
    cu_seqlen_ks: torch.Tensor, # [seq_len]
    cu_seqlen_ke: torch.Tensor, # [seq_len]
) -> torch.Tensor: # [seq_len, seq_len_kv]
    kv, scale = kv

    if hasattr(torch.ops, "_xpu_C") and hasattr(
        torch.ops._xpu_C, "fp8_mqa_logits_cute"
    ):
        return torch.ops._xpu_C.fp8_mqa_logits_cute(
            q,
            kv,
            scale,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
        )
    
    seq_len = q.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    seq_len_kv = kv.shape[0]

    logits = torch.empty((seq_len, seq_len_kv), device=q.device, dtype=torch.float32)

    BLOCK_HEADS = 32
    BLOCK_KV = 64

    grid = (
        seq_len,
        triton.cdiv(seq_len_kv, BLOCK_KV)
    )

    triton_fp8_mqa_logits_kernel[grid](
        q_ptr=q,
        kv_ptr=kv,
        kv_scales_ptr=scale,
        weights_ptr=weights,
        cu_seqlen_ks_ptr=cu_seqlen_ks,
        cu_seqlen_ke_ptr=cu_seqlen_ke,
        logits_ptr=logits,
        seq_len=seq_len,
        seq_len_kv=seq_len_kv,
        num_heads=num_heads,
        head_dim=head_dim,
        BLOCK_HEADS=BLOCK_HEADS,
        BLOCK_KV=BLOCK_KV,
        HEAD_DIM=head_dim,
    )

    return logits


@triton.jit
def triton_fp8_paged_mqa_logits_kernel(
    # Pointers to tensors
    q_fp8_ptr,
    kv_cache_fp8_ptr,
    kv_cache_fp32_ptr,
    weights_ptr,
    context_lens_ptr,
    block_tables_ptr,
    logits_ptr,
    # Shapes
    batch_size,
    next_n,
    heads,
    index_dim,
    num_blocks,
    block_size,
    max_blocks,
    max_model_len,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):    
    curr_batch = tl.program_id(0)
    curr_block = tl.program_id(1)
    
    if curr_batch >= batch_size:
        return
    
    context_len = tl.load(context_lens_ptr + curr_batch)
    
    block_start = curr_block * BLOCK_N
    
    if block_start >= context_len:
        return
    
    block_end = tl.minimum(block_start + BLOCK_N, context_len)
    actual_block_size = block_end - block_start
    
    block_table_idx = curr_batch * max_blocks + curr_block
    physical_block_idx = tl.load(block_tables_ptr + block_table_idx)
    
    if physical_block_idx < 0 or physical_block_idx >= num_blocks:
        return
    
    q_start = context_len - next_n
    
    k_offsets = block_start + tl.arange(0, BLOCK_N)
    
    mask_k_valid = k_offsets < context_len
        
    cache_block_stride = block_size * (index_dim + 4)
    cache_slot_value_stride = index_dim

    cache_block_value_base_ptr = kv_cache_fp8_ptr + physical_block_idx * cache_block_stride

    cache_block_scale_offset = block_size * cache_slot_value_stride
    cache_slot_scale_stride = 4
    cache_block_scale_base_ptr = kv_cache_fp32_ptr + \
        (physical_block_idx * cache_block_stride + cache_block_scale_offset) // cache_slot_scale_stride

    scale_ptrs = cache_block_scale_base_ptr + tl.arange(0, BLOCK_N)
    scales = tl.load(scale_ptrs, mask=mask_k_valid, other=1.0) # [BLOCK_N]
    
    q_batch_stride = next_n * heads * index_dim
    q_seq_stride = heads * index_dim
    q_head_stride = index_dim
    
    q_curr_batch_ptr = q_fp8_ptr + curr_batch * q_batch_stride

    weights_curr_batch_ptr = weights_ptr + curr_batch * next_n * heads

    for q_token_id in tl.range(next_n):
        logits_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        q_curr_token_ptr = q_curr_batch_ptr + q_token_id * q_seq_stride
        weights_curr_token_ptr = weights_curr_batch_ptr + q_token_id * heads

        for curr_head_block_offset in tl.range(0, heads, BLOCK_M):
            q_head_ptr = q_curr_token_ptr + (curr_head_block_offset + tl.arange(0, BLOCK_M)[:, None]) * q_head_stride
            q_ptrs = q_head_ptr + tl.arange(0, BLOCK_D)[None, :]
            q_fp8 = tl.load(q_ptrs)  # [BLOCK_M, BLOCK_D]
            
            k_ptrs = cache_block_value_base_ptr + tl.arange(0, BLOCK_N)[:, None] * cache_slot_value_stride + \
                        tl.arange(0, BLOCK_D)[None, :]
            k_fp8 = tl.load(k_ptrs, mask=tl.arange(0, BLOCK_N)[:, None] < actual_block_size) # [BLOCK_N, BLOCK_D]

            head_block_logits = tl.dot(q_fp8.to(tl.float16), k_fp8.to(tl.float16).T, out_dtype=tl.float32) # [BLOCK_M, BLOCK_N]

            head_block_logits = head_block_logits * scales[None, :] # [BLOCK_M, BLOCK_N]
            
            head_block_logits = tl.maximum(head_block_logits, 0.0) # [BLOCK_M, BLOCK_N]
            
            weight = tl.load(weights_curr_token_ptr + curr_head_block_offset + tl.arange(0, BLOCK_M))  # [BLOCK_M]

            head_block_logits = head_block_logits * weight[:, None] # [BLOCK_M, BLOCK_N]

            logits_acc += tl.sum(head_block_logits, axis=0) # [BLOCK_N]
        
        q_offsets = q_start + q_token_id
        mask_kq = (k_offsets <= q_offsets) & mask_k_valid # [BLOCK_N]

        logits_masked = tl.where(mask_kq, logits_acc, float('-inf'))

        row_offsets = q_token_id * max_model_len
        col_offsets = block_start + tl.arange(0, BLOCK_N)

        out_offsets = curr_batch * next_n * max_model_len + row_offsets + col_offsets
        valid_key_mask = tl.arange(0, BLOCK_N) < actual_block_size
        tl.store(logits_ptr + out_offsets, logits_masked, mask=valid_key_mask)
    

def triton_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor, # (batch_size, next_n, heads, index_dim)
    kv_cache_fp8: torch.Tensor, # (num_blocks, blocksize, 1, index_dim+4)
    weights: torch.Tensor, # (batch_size * next_n, heads)
    context_lens: torch.Tensor, # (batch_size,)
    block_tables: torch.Tensor, # (batch_size, max_blocks)
    schedule_metadata,
    max_model_len: int,
):
    """Compute FP8 MQA logits using paged KV-cache.

    Args:
        q_fp8: Query tensor of shape [B, next_n, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv_cache_fp8: Paged KV-cache in packed FP8+scale layout with shape
            [num_blocks, block_size, 1, D+4], dtype `torch.uint8`. The last
            4 bytes per (block,pos) store the `float` dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context length
            for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs. Currently unused.
        max_model_len: Maximum sequence length used to size the logits output.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype
        `torch.float32`.
    """
    batch_size, next_n, heads, index_dim = q_fp8.size()
    num_blocks, block_size, _, _ = kv_cache_fp8.size()
    max_blocks = block_tables.size(1)

    BLOCK_M = min(16, heads)
    BLOCK_N = block_size
    BLOCK_D = index_dim

    logits = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=q_fp8.device,
        dtype=torch.float32,
    )

    grid = (
        batch_size,
        triton.cdiv(max_model_len, BLOCK_N),
    )

    kv_cache_fp8 = kv_cache_fp8.view(torch.float8_e4m3fn)
    kv_cache_fp32 = kv_cache_fp8.view(torch.float32)

    triton_fp8_paged_mqa_logits_kernel[grid](
        q_fp8_ptr=q_fp8,
        kv_cache_fp8_ptr=kv_cache_fp8,
        kv_cache_fp32_ptr=kv_cache_fp32,
        weights_ptr=weights,
        context_lens_ptr=context_lens,
        block_tables_ptr=block_tables,
        logits_ptr=logits,
        batch_size=batch_size,
        next_n=next_n,
        heads=heads,
        index_dim=index_dim,
        num_blocks=num_blocks,
        block_size=block_size,
        max_blocks=max_blocks,
        max_model_len=max_model_len,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    return logits
