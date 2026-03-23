import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401


def sycl_tla_fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    kv_value, kv_scale = kv

    if not (hasattr(torch.ops, "_xpu_C") and hasattr(torch.ops._xpu_C, "fp8_mqa_logits_cute")):
      raise RuntimeError("torch.ops._xpu_C.fp8_mqa_logits_cute is not available")

    return torch.ops._xpu_C.fp8_mqa_logits_cute(
        q,
        kv_value,
        kv_scale,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )
