import random
import pytest

import torch
import vllm_xpu_kernels._xpu_C  # noqa: F401

# just for profiling
from csrc.xpu.mqa_logits.triton_mqa_logits import (
    triton_fp8_mqa_logits,
    triton_fp8_paged_mqa_logits
)

DEVICES = ["xpu:0"]

# Picked from deepgemm tests
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


def _generate_cp_test_data(seq_len: int, seq_len_kv: int, device):
    assert seq_len_kv % seq_len == 0 and seq_len % 2 == 0
    chunk_size = seq_len // 2
    cp_size = seq_len_kv // seq_len
    cp_id = cp_size // 3
    ks = torch.zeros(seq_len, dtype=torch.int32, device=device)
    ke = torch.zeros(seq_len, dtype=torch.int32, device=device)
    for i in range(chunk_size):
        ke[i] = cp_id * chunk_size + i
        ke[i + chunk_size] = (cp_size * 2 - 1 - cp_id) * chunk_size + i
    return ks, ke


def _pytorch_mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor, 
    scale: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    seq_len_kv = kv.shape[0]
    k = kv.to(torch.bfloat16)
    q = q.to(torch.bfloat16)

    mask_lo = (
        torch.arange(0, seq_len_kv, device=q.device)[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device=q.device)[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi

    score = torch.einsum("mhd,nd->hmn", q, k).float() * scale
    logits = (score.relu() * weights.unsqueeze(-1).transpose(0, 1)).sum(dim=0)
    logits = logits.masked_fill(~mask, float("-inf"))
    return logits


# Taken from vllm/attention/ops/rocm_aiter_mla_sparse.py::fp8_paged_mqa_logits_torch, 
# but modify the kv cache layout to store the scales at end of each block instead of 
# each slot, to align with CUDA
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
        q_offsets = torch.arange(context_len - next_n, context_len, device=q.device)
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )
        for block_rk in range(cdiv(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache_value[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size, (block_rk + 1) * block_size, device=q.device
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


@pytest.mark.parametrize("seq_len_qkv", [(512, 1024), (24, 24)])
@pytest.mark.parametrize("disable_cp", [True, False])
@pytest.mark.parametrize("device", DEVICES)
def test_fp8_mqa_logits_xpu(seq_len_qkv, disable_cp, device):
    torch.manual_seed(0)
    random.seed(0)
    num_heads, head_dim = 64, 128

    seq_len, seq_len_kv = seq_len_qkv

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
        ks, ke = _generate_cp_test_data(seq_len, seq_len_kv, device=device)

    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fp8, kv_scales = per_custom_dims_cast_to_fp8(kv, (0,), False)

    ref_logits = _pytorch_mqa_logits(
        q=q_fp8,
        kv=kv_fp8,
        scale=kv_scales,
        weights=weights,
        cu_seqlen_ks=ks,
        cu_seqlen_ke=ke,
    )
    logits = torch.ops._xpu_C.fp8_mqa_logits(q_fp8, kv_fp8, kv_scales, weights, ks, ke)

    ref_neginf_mask = ref_logits == float("-inf")
    neginf_mask = logits == float("-inf")
    assert torch.equal(neginf_mask, ref_neginf_mask)

    ref_logits = ref_logits.masked_fill(ref_neginf_mask, 0)
    logits = logits.masked_fill(neginf_mask, 0)
    diff = calc_diff(logits, ref_logits)
    assert diff < 1e-3, f"{diff=}"

@pytest.mark.parametrize("bs_nextn", [(4, 1), (4, 2), (2, 2)])
@pytest.mark.parametrize("device", DEVICES)
def test_fp8_paged_mqa_logits_xpu(bs_nextn, device):
    torch.manual_seed(0)
    random.seed(0)

    batch_size, next_n = bs_nextn

    max_model_len = 4096

    for heads, index_dim in [(64, 128)]:
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

            logits = torch.ops._xpu_C.fp8_paged_mqa_logits(
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


def test_fp8_mqa_logits_profiling():
    torch.manual_seed(0)
    random.seed(0)
    device = "xpu:0"

    seq_len, seq_len_kv = 512, 1024
    num_heads, head_dim = 64, 128

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

    ks = torch.zeros(seq_len, dtype=torch.int32, device=device)
    ke = torch.arange(seq_len, dtype=torch.int32, device=device) + (
        seq_len_kv - seq_len
    )

    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fp8, kv_scales = per_custom_dims_cast_to_fp8(kv, (0,), False)

    NUM_WARMUP = 10
    for _ in range(NUM_WARMUP):
        torch.ops._xpu_C.fp8_mqa_logits(q_fp8, kv_fp8, kv_scales, weights, ks, ke)
        triton_fp8_mqa_logits(q_fp8, (kv_fp8, kv_scales), weights, ks, ke)
    torch.xpu.synchronize()

    NUM_ITER = 100
    import time
    print("\nProfiling fp8_mqa_logits kernels...")

    # profiling the sycl/sycl-tla kernel
    time_start = time.time()
    for _ in range(NUM_ITER):
        sycl_logits = torch.ops._xpu_C.fp8_mqa_logits(q_fp8, kv_fp8, kv_scales, weights, ks, ke)
    torch.xpu.synchronize()
    time_end = time.time()
    sycl_time = time_end - time_start
    print(f"Sycl kernel total time for {NUM_ITER} iterations: {sycl_time} seconds")

    # profiling the Triton kernel
    time_start = time.time()
    for _ in range(NUM_ITER):
        triton_logits = triton_fp8_mqa_logits(q_fp8, (kv_fp8, kv_scales), weights, ks, ke)
    torch.xpu.synchronize()
    time_end = time.time()
    triton_time = time_end - time_start
    print(f"Triton kernel total time for {NUM_ITER} iterations: {triton_time} seconds")

    assert sycl_time < triton_time, "Well-tuned Sycl kernel should be faster than Triton kernel"

    assert torch.allclose(sycl_logits, triton_logits, rtol=1e-2, atol=1e-2), "Sycl and Triton kernels should produce equivalent results"


def test_fp8_paged_mqa_logits_profiling():
    torch.manual_seed(0)
    random.seed(0)
    device = "xpu:0"

    batch_size, next_n = 4, 2
    heads, index_dim = 64, 128
    max_model_len = 4096
    avg_kv = 2048
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

    NUM_WARMUP = 10
    NUM_ITER = 1000

    for _ in range(NUM_WARMUP):
        torch.ops._xpu_C.fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8,
            weights,
            context_lens,
            block_tables,
            schedule_metadata,
            max_model_len,
        )
        triton_fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8,
            weights,
            context_lens,
            block_tables,
            schedule_metadata,
            max_model_len,
        )
    torch.xpu.synchronize()

    import time
    print("\nProfiling fp8_paged_mqa_logits kernels...")

    time_start = time.time()
    for _ in range(NUM_ITER):
        sycl_logits = torch.ops._xpu_C.fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8,
            weights,
            context_lens,
            block_tables,
            schedule_metadata,
            max_model_len,
        )
    torch.xpu.synchronize()
    time_end = time.time()
    sycl_time = time_end - time_start
    print(f"Sycl paged kernel total time for {NUM_ITER} iterations: {sycl_time} seconds")

    time_start = time.time()
    for _ in range(NUM_ITER):
        triton_logits = triton_fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8,
            weights,
            context_lens,
            block_tables,
            schedule_metadata,
            max_model_len,
        )
    torch.xpu.synchronize()
    time_end = time.time()
    triton_time = time_end - time_start
    print(f"Triton paged kernel total time for {NUM_ITER} iterations: {triton_time} seconds")

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

    sycl_logits = sycl_logits.masked_fill(~mask, 0)
    triton_logits = triton_logits.masked_fill(~mask, 0)
    diff = calc_diff(sycl_logits, triton_logits)

    assert diff < 1e-3, f"{diff=}"
    assert sycl_time < triton_time, "Well-tuned paged Sycl kernel should be faster than Triton kernel"

