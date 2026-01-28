import random
import pytest
import pdb

import torch
import triton
import triton.language as tl

DEVICES = [
    f"xpu:{i}" for i in range(1 if torch.xpu.device_count() == 1 else 2)
]

def kv_cache_cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
    # x: (num_blocks, block_size, 1, head_dim)
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4)),
        device=x.device,
        dtype=torch.uint8,
    )
    x_fp8[:, : block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim
    ).view(dtype=torch.uint8)
    x_fp8[:, block_size * head_dim :] = sf.view(num_blocks, block_size).view(
        dtype=torch.uint8
    )
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4)

def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)

def calc_diff(x: torch.Tensor, y: torch.Tensor):
    """Return a global difference metric for unit tests.

    DeepGEMM kernels on Blackwell/B200 currently exhibit noticeable per-element
    error, causing `torch.testing.assert_close` to fail.  Instead of checking
    every element, we compute a cosine-style similarity over the whole tensor
    and report `1 - sim`.  Once kernel accuracy improves this helper can be
    removed.
    """

    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim

def _ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))

def per_custom_dims_cast_to_fp8(
    x: torch.Tensor, dims: tuple, use_ue8m0: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    excluded_dims = tuple([i for i in range(x.dim()) if i not in set(dims)])
    x_amax = x.abs().float().amax(dim=excluded_dims, keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    sf = _ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled, sf.squeeze()


def _generate_cp_test_data(seq_len: int, seq_len_kv: int):
    assert seq_len_kv % seq_len == 0 and seq_len % 2 == 0
    chunk_size = seq_len // 2
    cp_size = seq_len_kv // seq_len
    cp_id = cp_size // 3
    ks = torch.zeros(seq_len, dtype=torch.int32, device="xpu:0")
    ke = torch.zeros(seq_len, dtype=torch.int32, device="xpu:0")
    for i in range(chunk_size):
        ke[i] = cp_id * chunk_size + i
        ke[i + chunk_size] = (cp_size * 2 - 1 - cp_id) * chunk_size + i
    return ks, ke


def fp8_mqa_logits_torch(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Compute FP8 MQA logits for a single sequence without KV paging.

    Args:
        q: Query tensor of shape [M, H, D]. Casted to
            `torch.float8_e4m3fn` by caller.
        kv: Tuple `(k_fp8, k_scales)` where `k_fp8` has shape [N, D] with
            dtype `torch.float8_e4m3fn` and `k_scales` has shape [N] (or
            [N, 1]) with dtype `torch.float32`.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query position,
            shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query position,
            shape [M], dtype int32.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    kv, scale = kv
    seq_len_kv = kv.shape[0]
    k = kv.to(torch.bfloat16)
    q = q.to(torch.bfloat16)

    mask_lo = (
        torch.arange(0, seq_len_kv, device="xpu:0")[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device="xpu:0")[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi

    score = torch.einsum("mhd,nd->hmn", q, k).float() * scale
    wei_t = weights.unsqueeze(-1).transpose(0, 1)
    logits = (score.relu() * wei_t).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))

    return logits

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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0) 
    pid_n = tl.program_id(1)
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    
    m_mask = m_offsets < seq_len
    n_mask = n_offsets < seq_len_kv
    
    logits_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    ks_vals = tl.load(cu_seqlen_ks_ptr + m_offsets, mask=m_mask, other=0)
    ke_vals = tl.load(cu_seqlen_ke_ptr + m_offsets, mask=m_mask, other=0)
    

    ks_vals_expanded = ks_vals[:, None]  # [BLOCK_M, 1]
    ke_vals_expanded = ke_vals[:, None]  # [BLOCK_M, 1]
    n_offsets_expanded = n_offsets[None, :]  # [1, BLOCK_N]
    
    seq_mask = (n_offsets_expanded >= ks_vals_expanded) & \
               (n_offsets_expanded < ke_vals_expanded) # [BLOCK_M, BLOCK_N]
    
    seq_mask = seq_mask & m_mask[:, None] & n_mask[None, :]
    
    # if the block is masked out, directly write -inf and return
    if tl.sum(seq_mask) == 0:
        out_offsets = m_offsets[:, None] * seq_len_kv + n_offsets[None, :]
        INF_NEG = float('-inf')
        tl.store(
            logits_ptr + out_offsets,
            tl.full((BLOCK_M, BLOCK_N), INF_NEG, dtype=tl.float32),
            mask=m_mask[:, None] & n_mask[None, :]
        )
        return
    
    for h in range(num_heads):
        # load q
        q_offsets = m_offsets[:, None] * num_heads * head_dim + h * head_dim
        q_block_fp8 = tl.load(
            q_ptr + q_offsets + tl.arange(0, BLOCK_D)[None, :],
            mask=m_mask[:, None] & (tl.arange(0, BLOCK_D)[None, :] < head_dim),
            other=0.0
        )
        q_block = q_block_fp8.to(tl.float16)
        
        # load k
        kv_offsets = n_offsets[:, None] * head_dim
        kv_block_fp8 = tl.load(
            kv_ptr + kv_offsets + tl.arange(0, BLOCK_D)[None, :],
            mask=n_mask[:, None] & (tl.arange(0, BLOCK_D)[None, :] < head_dim),
            other=0.0
        )
        kv_block = kv_block_fp8.to(tl.float16)
        
        # [BLOCK_M, head_dim] @ [head_dim, BLOCK_N] = [BLOCK_M, BLOCK_N]
        head_logits = tl.dot(q_block, kv_block.T).to(tl.float32)
        
        # load and mul scale
        scale_vals = tl.load(kv_scales_ptr + n_offsets, mask=n_mask, other=1.0)
        head_logits = head_logits * scale_vals[None, :]
        
        # relu
        head_logits = tl.maximum(head_logits, 0.0)
        
        # mul weight
        weights_vals = tl.load(
            weights_ptr + m_offsets * num_heads + h,
            mask=m_mask,
            other=0.0
        )

        # reduce along num_head
        logits_acc += head_logits * weights_vals[:, None]
    
    # apply mask
    INF_NEG = float('-inf')
    logits_acc = tl.where(seq_mask, logits_acc, INF_NEG)
    
    out_offsets = m_offsets[:, None] * seq_len_kv + n_offsets[None, :]
    tl.store(
        logits_ptr + out_offsets,
        logits_acc,
        mask=m_mask[:, None] & n_mask[None, :]
    )


def triton_fp8_mqa_logits(
    q: torch.Tensor, # [seq_len, num_heads, head_dim]
    kv: tuple[torch.Tensor, torch.Tensor], # [seq_len_kv, head_dim], [seq_len_kv]
    weights: torch.Tensor, # [seq_len, num_heads]
    cu_seqlen_ks: torch.Tensor, # [seq_len]
    cu_seqlen_ke: torch.Tensor, # [seq_len]
) -> torch.Tensor: # [seq_len, seq_len_kv]
    kv, scale = kv
    
    seq_len = q.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    seq_len_kv = kv.shape[0]

    BLOCK_M = 32
    BLOCK_N = 32
    BLOCK_D = head_dim

    logits = torch.empty((seq_len, seq_len_kv), device=q.device, dtype=torch.float32)

    grid = (
        triton.cdiv(seq_len, BLOCK_M), 
        triton.cdiv(seq_len_kv, BLOCK_N)
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
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    return logits


@pytest.mark.parametrize("seq_len", [512,])
@pytest.mark.parametrize("seq_len_kv", [1024,])
@pytest.mark.parametrize("disable_cp", [True, False])
@pytest.mark.parametrize("device", DEVICES)
def test_triton_fp8_mqa_logits(seq_len, seq_len_kv, disable_cp, device):
    torch.manual_seed(0)
    random.seed(0)
    num_heads, head_dim = 32, 128

    q = torch.randn(
        seq_len,
        num_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    kv = torch.randn(
        seq_len_kv, head_dim, device=device, dtype=torch.bfloat16
    )
    weights = torch.randn(
        seq_len, num_heads, device=device, dtype=torch.float32
    )

    if disable_cp:
        ks = torch.zeros(seq_len, dtype=torch.int32, device=device)
        ke = torch.arange(seq_len, dtype=torch.int32, device=device) + (
            seq_len_kv - seq_len
        )
    else:
        ks, ke = _generate_cp_test_data(seq_len, seq_len_kv)

    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fp8 = per_custom_dims_cast_to_fp8(kv, (0,), False)

    ref_logits = fp8_mqa_logits_torch(
        q=q_fp8,
        kv=kv_fp8,
        weights=weights,
        cu_seqlen_ks=ks,
        cu_seqlen_ke=ke,
    )

    logits = triton_fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke)

    ref_neginf_mask = ref_logits == float("-inf")
    neginf_mask = logits == float("-inf")

    assert torch.equal(neginf_mask, ref_neginf_mask)

    ref_logits = ref_logits.masked_fill(ref_neginf_mask, 0)
    logits = logits.masked_fill(neginf_mask, 0)
    diff = calc_diff(logits, ref_logits)
    assert diff < 1e-3, f"{diff=}"

    # simple benchmark: run multiple iterations for both torch impl and triton
    # impl, and comare the host time
    import time
    num_iterations = 1000
    start_time_torch = time.time()
    for _ in range(num_iterations):
        ref_logits = fp8_mqa_logits_torch(
            q=q_fp8,
            kv=kv_fp8,
            weights=weights,
            cu_seqlen_ks=ks,
            cu_seqlen_ke=ke,
        )
    torch.xpu.synchronize()
    end_time_torch = time.time()
    torch_time = end_time_torch - start_time_torch

    start_time_triton = time.time()
    for _ in range(num_iterations):
        logits = triton_fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke)
    torch.xpu.synchronize()
    end_time_triton = time.time()
    triton_time = end_time_triton - start_time_triton

    print(f"Torch implementation total time for {num_iterations} iterations: {torch_time:.6f} seconds")
    print(f"Triton implementation total time for {num_iterations} iterations: {triton_time:.6f} seconds")
    assert triton_time < torch_time, "Expected Triton implementation to be faster than Torch implementation"


def fp8_paged_mqa_logits_torch(
    q: torch.Tensor, # (batch_size, next_n, heads, index_dim)
    kv_cache: torch.Tensor, # (num_blocks, blocksize, 1, index_dim+4)
    weights: torch.Tensor, # (batch_size * next_n, heads)
    context_lens: torch.Tensor, # (batch_size,)
    block_tables: torch.Tensor, # (batch_size, max_blocks)
    max_model_len: int,
):
    fp8_dtype = torch.float8_e4m3fn # current_platform.fp8_dtype()
    batch_size, next_n, _, dim = q.size()
    num_blocks, block_size, _, _ = kv_cache.size()

    kv_cache = kv_cache.view(num_blocks, -1)
    kv_cache_value = kv_cache[:, : block_size * dim].view(num_blocks, block_size, 1, dim)
    kv_cache_scale = kv_cache[:, block_size * dim :].view(num_blocks, block_size, 1, 4).view(torch.float32)

    q = q.float()
    kv_cache_value = kv_cache_value.view(fp8_dtype).float() * kv_cache_scale

    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    context_lens = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device="xpu:0")
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        for block_rk in range(cdiv(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache_value[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size, (block_rk + 1) * block_size, device="xpu:0"
            )
            mask = (k_offsets[None, :] < context_len) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(k_offsets[None, :] <= q_offsets[:, None], s, float("-inf"))
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
    q_offsets = q_start + tl.arange(0, BLOCK_M)
    
    k_offsets = block_start + tl.arange(0, BLOCK_N)
    
    mask_k_valid = k_offsets < context_len
    mask_kq = (k_offsets[None, :] <= q_offsets[:, None]) & mask_k_valid[None, :]
    
    weight_start_idx = curr_batch * next_n
    
    logits_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    cache_block_stride = block_size * (index_dim + 4)
    cache_slot_value_stride = index_dim

    cache_block_value_base_ptr = kv_cache_fp8_ptr + physical_block_idx * cache_block_stride

    cache_block_scale_offset = block_size * cache_slot_value_stride
    cache_slot_scale_stride = 4
    cache_block_scale_base_ptr = kv_cache_fp32_ptr + \
        (physical_block_idx * cache_block_stride + cache_block_scale_offset) // cache_slot_scale_stride

    scale_ptrs = cache_block_scale_base_ptr + tl.arange(0, BLOCK_N)
    scales = tl.load(scale_ptrs, mask=mask_k_valid, other=1.0)
    
    q_batch_stride = next_n * heads * index_dim
    q_seq_stride = heads * index_dim
    q_head_stride = index_dim
    
    q_batch_batch_ptr = q_fp8_ptr + curr_batch * q_batch_stride

    for h in range(heads):
        q_head_ptr = q_batch_batch_ptr + h * q_head_stride
        q_ptrs = q_head_ptr + tl.arange(0, BLOCK_M)[:, None] * q_seq_stride + \
                    tl.arange(0, BLOCK_D)[None, :]
        q_fp8 = tl.load(q_ptrs) 
        
        k_ptrs = cache_block_value_base_ptr + tl.arange(0, BLOCK_N)[:, None] * cache_slot_value_stride + \
                    tl.arange(0, BLOCK_D)[None, :]
        k_fp8 = tl.load(k_ptrs, mask=tl.arange(0, BLOCK_N)[:, None] < actual_block_size)

        head_logits = tl.dot(q_fp8.to(tl.float16), k_fp8.to(tl.float16).T, out_dtype=tl.float32)

        head_logits = head_logits * scales[None, :]
        
        head_logits = tl.maximum(head_logits, 0.0)
        
        weight = tl.load(weights_ptr + weight_start_idx * heads + tl.arange(0, BLOCK_M) * heads + h)

        logits_acc += head_logits * weight[:, None]
    
    logits_masked = tl.where(mask_kq, logits_acc, float('-inf'))

    row_offsets = tl.arange(0, BLOCK_M) * max_model_len
    col_offsets = block_start + tl.arange(0, BLOCK_N)

    out_offsets = curr_batch * next_n * max_model_len + row_offsets[:, None] + col_offsets[None, :]
    valid_key_mask = tl.arange(0, BLOCK_N)[None, :] < actual_block_size
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

    BLOCK_M = next_n
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

@pytest.mark.parametrize("device", DEVICES)
def test_triton_fp8_paged_mqa_logits(device):
    torch.manual_seed(0)
    random.seed(0)

    max_model_len = 4096

    for batch_size, next_n in [(4, 1), (2, 2)]:
        for heads, index_dim in [(32, 128)]:
            for avg_kv in (2048,):
                num_blocks, blocksize = max_model_len * 2, 64

                q = torch.randn(
                    (batch_size, next_n, heads, index_dim),
                    device=device,
                    dtype=torch.bfloat16,
                )
                kv_cache = torch.randn(
                    (num_blocks, blocksize, 1, index_dim),
                    device=device,
                    dtype=torch.bfloat16,
                )
                weights = torch.randn(
                    (batch_size * next_n, heads),
                    device=device,
                    dtype=torch.float32,
                )

                context_lens = (
                    torch.randint(int(0.8 * avg_kv), int(1.2 * avg_kv), (batch_size,))
                    .xpu()
                    .to(torch.int32)
                )
                max_block_len = (
                    (context_lens.max().item() + blocksize - 1) // blocksize * blocksize
                )
                block_tables = torch.zeros(
                    (batch_size, max_block_len),
                    device=device,
                    dtype=torch.int32,
                )

                counter = 0
                block_idx_pool = list(range(num_blocks))
                random.shuffle(block_idx_pool)
                for i in range(batch_size):
                    ctx_len = int(context_lens[i].item())
                    for j in range((ctx_len + blocksize - 1) // blocksize):
                        block_tables[i][j] = block_idx_pool[counter]
                        counter += 1

                q_fp8 = q.to(torch.float8_e4m3fn)
                kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache)

                schedule_metadata = None
                
                ref_logits = fp8_paged_mqa_logits_torch(
                    q_fp8,
                    kv_cache_fp8,
                    weights,
                    context_lens,
                    block_tables,
                    max_model_len,
                )

                logits = triton_fp8_paged_mqa_logits(
                    q_fp8,
                    kv_cache_fp8,
                    weights,
                    context_lens,
                    block_tables,
                    schedule_metadata,
                    max_model_len,
                )

                ref_neginf_mask = ref_logits == float("-inf")
                neginf_mask = logits == float("-inf")

                assert torch.equal(neginf_mask, ref_neginf_mask)

                positions = (
                    torch.arange(max_model_len, device=device)
                    .unsqueeze(0)
                    .expand(batch_size * next_n, -1)
                )
                row_indices = torch.arange(batch_size * next_n, device=device) // next_n
                next_n_offset = (
                    torch.arange(batch_size * next_n, device=device) % next_n
                )
                mask = positions <= (
                    context_lens[row_indices] - next_n + next_n_offset
                ).unsqueeze(1)

                logits = logits.masked_fill(~mask, 0)
                ref_logits = ref_logits.masked_fill(~mask, 0)
                diff = calc_diff(logits, ref_logits)
                assert diff < 1e-3, f"{diff=}"

                torch.xpu.synchronize()

                # simple benchmark: run multiple iterations for both torch impl and triton
                # impl, and comare the host time
                import time
                num_iterations = 300
                start_time_torch = time.time()
                for _ in range(num_iterations):
                    ref_logits = fp8_paged_mqa_logits_torch(
                        q_fp8,
                        kv_cache_fp8,
                        weights,
                        context_lens,
                        block_tables,
                        max_model_len,
                    )
                torch.xpu.synchronize()
                end_time_torch = time.time()
                torch_time = end_time_torch - start_time_torch

                start_time_triton = time.time()
                for _ in range(num_iterations):
                    logits = triton_fp8_paged_mqa_logits(
                        q_fp8,
                        kv_cache_fp8,
                        weights,
                        context_lens,
                        block_tables,
                        schedule_metadata,
                        max_model_len,
                    )
                torch.xpu.synchronize()
                end_time_triton = time.time()
                triton_time = end_time_triton - start_time_triton

                print(f"Torch implementation total time for {num_iterations} iterations: {torch_time:.6f} seconds")
                print(f"Triton implementation total time for {num_iterations} iterations: {triton_time:.6f} seconds")
                assert triton_time < torch_time, "Expected Triton implementation to be faster than Torch implementation"
