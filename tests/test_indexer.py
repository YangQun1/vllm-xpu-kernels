import random
import pytest

import torch
import triton
import triton.language as tl

DEVICES = [
    f"xpu:{i}" for i in range(1 if torch.xpu.device_count() == 1 else 2)
]


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

import pdb

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
    
    torch.testing.assert_close(logits, ref_logits, atol=0.3, rtol=0.3)

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

