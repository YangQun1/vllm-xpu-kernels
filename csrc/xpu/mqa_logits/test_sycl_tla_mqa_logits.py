import random

import pytest
import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

from sycl_tla_mqa_logits import sycl_tla_fp8_mqa_logits


DEVICES = ["xpu:0"]


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def _ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def per_custom_dims_cast_to_fp8(
    x: torch.Tensor,
    dims: tuple,
    use_ue8m0: bool,
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


@pytest.mark.skipif(
    not (hasattr(torch.ops, "_xpu_C") and hasattr(torch.ops._xpu_C, "fp8_mqa_logits_cute")),
    reason="torch.ops._xpu_C.fp8_mqa_logits_cute not found",
)
@pytest.mark.parametrize("seq_len_qkv", [(512, 1024), (24, 24)])
@pytest.mark.parametrize("disable_cp", [True, False])
@pytest.mark.parametrize("device", DEVICES)
def test_sycl_tla_fp8_mqa_logits(seq_len_qkv, disable_cp, device):
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
        seq_len_kv,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    weights = torch.randn(
        seq_len,
        num_heads,
        device=device,
        dtype=torch.float32,
    )

    if disable_cp:
        ks = torch.zeros(seq_len, dtype=torch.int32, device=device)
        ke = torch.arange(seq_len, dtype=torch.int32, device=device) + (
            seq_len_kv - seq_len
        )
    else:
        ks, ke = _generate_cp_test_data(seq_len, seq_len_kv, device=device)

    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fp8 = per_custom_dims_cast_to_fp8(kv, (0,), False)

    ref_logits = _pytorch_mqa_logits(
        q=q_fp8,
        kv=kv_fp8[0],
        scale=kv_fp8[1],
        weights=weights,
        cu_seqlen_ks=ks,
        cu_seqlen_ke=ke,
    )

    logits = sycl_tla_fp8_mqa_logits(q_fp8, kv_fp8, weights, ks, ke)

    ref_neginf_mask = ref_logits == float("-inf")
    neginf_mask = logits == float("-inf")
    assert torch.equal(neginf_mask, ref_neginf_mask)

    ref_logits = ref_logits.masked_fill(ref_neginf_mask, 0)
    logits = logits.masked_fill(neginf_mask, 0)

    diff = calc_diff(logits, ref_logits)
    assert diff < 1e-3, f"{diff=}"
