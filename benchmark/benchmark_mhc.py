# SPDX-License-Identifier: Apache-2.0

import argparse

import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

HC = 4
BENCH_CASES = [
    (1, 4096),
    (33, 4096),
    (128, 4096),
    (256, 4096),
    (1024, 4096),
    (2048, 4096),
    (4096, 4096),
    (8192, 4096),
    (16384, 4096),
    (1, 7168),
    (33, 7168),
    (128, 7168),
    (256, 7168),
    (1024, 7168),
    (2048, 7168),
    (4096, 7168),
    (8192, 7168),
    (16384, 7168),
]


def benchmark_op(fn, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()

    start_event = torch.xpu.Event(enable_timing=True)
    end_event = torch.xpu.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        fn()
    end_event.record()
    end_event.synchronize()

    return start_event.elapsed_time(end_event) * 1e3 / iters


def run_mhc_pre(num_tokens: int, hidden_size: int, warmup: int, iters: int):
    hc3 = HC * 2 + HC * HC
    residual = torch.randn((num_tokens, HC, hidden_size), dtype=torch.bfloat16, device="xpu")
    fn = torch.randn((hc3, HC * hidden_size), dtype=torch.float32, device="xpu")
    hc_scale = torch.randn((3,), dtype=torch.float32, device="xpu")
    hc_base = torch.randn((hc3,), dtype=torch.float32, device="xpu")

    rms_eps = 1e-6
    hc_pre_eps = 1e-3
    hc_sinkhorn_eps = 1e-3
    hc_post_mult_value = 1.0
    sinkhorn_repeat = 20

    def run():
        torch.ops._xpu_C.mhc_pre(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
        )

    avg_s = benchmark_op(run, warmup, iters)

    # FLOPs model for `mhc_pre`
    # - Stage 1 dominant cost is one dense matmul:
    #     [num_tokens, HC * hidden_size] x [HC * hidden_size, hc3]
    # - We count 2 FLOPs per MAC.
    # - Stage 2 contains sigmoid / Sinkhorn / weighted reduction, but those are
    #   intentionally omitted here because the matmul dominates and we want a
    #   simple roofline-style estimate.
    flops = 2.0 * num_tokens * hc3 * (HC * hidden_size)

    # Bytes moved model for `mhc_pre` (ideal / theoretical minimum)
    # Only counts external inputs read once and final outputs written once.
    # Internal intermediates (rms_mixes) are assumed to stay in registers/SLM.
    #
    # Inputs (read once):
    #   residual[N,HC,H] bf16            num_tokens * HC * hidden_size * 2
    #   fn[hc3,HC*H] fp32               hc3 * HC * hidden_size * 4
    #   hc_scale[3] fp32                 3 * 4
    #   hc_base[hc3] fp32               hc3 * 4
    #
    # Outputs (written once):
    #   post_mix[N,HC] fp32              num_tokens * HC * 4
    #   comb_mix[N,HC,HC] fp32           num_tokens * HC * HC * 4
    #   layer_input[N,H] bf16            num_tokens * hidden_size * 2
    #
    # Not counted (ideal fusion eliminates these DRAM round-trips):
    #   - rms_mixes[N,hc3] intermediate between Stage 1 and Stage 2
    bytes_moved = (
        # inputs
        num_tokens * HC * hidden_size * 2        # residual
        + hc3 * HC * hidden_size * 4             # fn
        + 3 * 4                                  # hc_scale
        + hc3 * 4                                # hc_base
        # outputs
        + num_tokens * HC * 4                    # post_mix
        + num_tokens * HC * HC * 4               # comb_mix
        + num_tokens * hidden_size * 2           # layer_input
    )
    return avg_s, flops, bytes_moved


def run_mhc_post(num_tokens: int, hidden_size: int, warmup: int, iters: int):
    x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device="xpu")
    residual = torch.randn((num_tokens, HC, hidden_size), dtype=torch.bfloat16, device="xpu")
    post_mix = torch.randn((num_tokens, HC, 1), dtype=torch.float32, device="xpu")
    comb_mix = torch.randn((num_tokens, HC, HC), dtype=torch.float32, device="xpu")

    def run():
        torch.ops._xpu_C.mhc_post(x, residual, post_mix, comb_mix)

    avg_s = benchmark_op(run, warmup, iters)

    # FLOPs model for `mhc_post`
    # - The reference uses:
    #     mixed_residual = einsum("...ij,...ih->...jh", comb_res_mix, residual)
    #     out = mixed_residual + post_layer_mix * x
    # - For each output element `(tok, o, h)`:
    #     * einsum reduction over `i in [0, HC)`:
    #         HC muls + (HC - 1) adds
    #     * scaled-x term:
    #         1 mul
    #     * final accumulation into output:
    #         1 add
    # - Total per output element = HC + (HC - 1) + 1 + 1 = 2 * HC + 1 FLOPs.
    # - Number of output elements = num_tokens * HC * hidden_size.
    flops = num_tokens * HC * hidden_size * (2.0 * HC + 1.0)

    # Bytes moved model for `mhc_post`
    # - x input:                     [num_tokens, hidden_size] bf16
    # - residual input:              [num_tokens, HC, hidden_size] bf16
    # - post_mix input:              [num_tokens, HC] fp32
    # - comb_mix input:              [num_tokens, HC, HC] fp32
    # - out output:                  [num_tokens, HC, hidden_size] bf16
    # Notes:
    # - This assumes one logical read/write of each tensor.
    # - Register / local-memory reuse is not reflected in this estimate.
    bytes_moved = (
        num_tokens * hidden_size * 2
        + num_tokens * HC * hidden_size * 2
        + num_tokens * HC * 4
        + num_tokens * HC * HC * 4
        + num_tokens * HC * hidden_size * 2
    )
    return avg_s, flops, bytes_moved


def run_hc_head_fused(num_tokens: int, hidden_size: int, warmup: int, iters: int):
    hs_flat = torch.randn((num_tokens, HC, hidden_size), dtype=torch.bfloat16, device="xpu")
    fn = torch.randn((HC, HC * hidden_size), dtype=torch.float32, device="xpu")
    hc_scale = torch.randn((1,), dtype=torch.float32, device="xpu")
    hc_base = torch.randn((HC,), dtype=torch.float32, device="xpu")
    out = torch.empty((num_tokens, hidden_size), dtype=torch.bfloat16, device="xpu")

    rms_eps = 1e-6
    hc_eps = 1e-6

    def run():
        torch.ops._xpu_C.hc_head_fused(hs_flat, fn, hc_scale, hc_base, out, rms_eps, hc_eps)

    avg_s = benchmark_op(run, warmup, iters)

    # FLOPs model for `hc_head_fused`
    # - Dominant cost is one dense projection from flattened hidden states:
    #     [num_tokens, HC * hidden_size] x [HC * hidden_size, HC]
    # - Counted as 2 FLOPs per MAC.
    # - RMS normalization, sigmoid, and final HC-weighted reduction are omitted
    #   from the FLOPs estimate for simplicity.
    flops = 2.0 * num_tokens * HC * (HC * hidden_size)

    # Bytes moved model for `hc_head_fused`
    # - hs_flat input:               [num_tokens, HC, hidden_size] bf16
    # - fn weights:                  [HC, HC * hidden_size] fp32
    # - hc_scale:                    [1] fp32
    # - hc_base:                     [HC] fp32
    # - out output:                  [num_tokens, hidden_size] bf16
    # Notes:
    # - This is a logical tensor traffic estimate.
    # - Internal temporary values used for normalization / gating are ignored.
    bytes_moved = (
        num_tokens * HC * hidden_size * 2
        + HC * HC * hidden_size * 4
        + 4
        + HC * 4
        + num_tokens * hidden_size * 2
    )
    return avg_s, flops, bytes_moved


def run_mhc_fused_post_pre(num_tokens: int, hidden_size: int, warmup: int, iters: int):
    hc3 = HC * 2 + HC * HC

    x = torch.randn((num_tokens, hidden_size), dtype=torch.bfloat16, device="xpu")
    residual = torch.randn((num_tokens, HC, hidden_size), dtype=torch.bfloat16, device="xpu")
    post_mix = torch.randn((num_tokens, HC, 1), dtype=torch.float32, device="xpu")
    comb_mix = torch.randn((num_tokens, HC, HC), dtype=torch.float32, device="xpu")
    fn = torch.randn((hc3, HC * hidden_size), dtype=torch.float32, device="xpu")
    hc_scale = torch.randn((3,), dtype=torch.float32, device="xpu")
    hc_base = torch.randn((hc3,), dtype=torch.float32, device="xpu")

    rms_eps = 1e-6
    hc_pre_eps = 1e-3
    hc_sinkhorn_eps = 1e-3
    hc_post_mult_value = 1.0
    sinkhorn_repeat = 20

    def run():
        torch.ops._xpu_C.mhc_fused_post_pre(
            x, residual, post_mix, comb_mix,
            fn, hc_scale, hc_base,
            rms_eps, hc_pre_eps, hc_sinkhorn_eps,
            hc_post_mult_value, sinkhorn_repeat,
        )

    avg_s = benchmark_op(run, warmup, iters)

    # FLOPs model for `mhc_fused_post_pre` (Phase 0: composed)
    # This is the sum of mhc_post and mhc_pre FLOPs:
    #   mhc_post:  num_tokens * HC * hidden_size * (2 * HC + 1)
    #   mhc_pre:   2 * num_tokens * hc3 * (HC * hidden_size)
    flops_post = num_tokens * HC * hidden_size * (2.0 * HC + 1.0)
    flops_pre = 2.0 * num_tokens * hc3 * (HC * hidden_size)
    flops = flops_post + flops_pre

    # Bytes moved model for `mhc_fused_post_pre` (ideal / theoretical minimum)
    # The benchmark bytes model reflects the minimum DRAM traffic assuming
    # perfect fusion, independent of the actual kernel implementation.
    #
    # Inputs (read once):
    #   x[N,H] bf16                      num_tokens * hidden_size * 2
    #   residual[N,HC,H] bf16            num_tokens * HC * hidden_size * 2
    #   post_mix[N,HC] fp32              num_tokens * HC * 4
    #   comb_mix[N,HC,HC] fp32           num_tokens * HC * HC * 4
    #   fn[hc3,HC*H] fp32               hc3 * HC * hidden_size * 4
    #   hc_scale[3] fp32                 3 * 4
    #   hc_base[hc3] fp32               hc3 * 4
    #
    # Outputs (written once):
    #   residual_cur[N,HC,H] bf16        num_tokens * HC * hidden_size * 2
    #   post_mix_cur[N,HC] fp32          num_tokens * HC * 4
    #   comb_mix_cur[N,HC,HC] fp32       num_tokens * HC * HC * 4
    #   layer_input[N,H] bf16            num_tokens * hidden_size * 2
    #
    # Not counted (ideal fusion eliminates these DRAM round-trips):
    #   - residual_cur read by mhc_pre (stays in registers/SLM)
    #   - rms_mixes[N,hc3] intermediate between Stage 1 and Stage 2 of mhc_pre
    bytes_moved = (
        # inputs
        num_tokens * hidden_size * 2             # x
        + num_tokens * HC * hidden_size * 2      # residual
        + num_tokens * HC * 4                    # post_mix
        + num_tokens * HC * HC * 4               # comb_mix
        + hc3 * HC * hidden_size * 4             # fn
        + 3 * 4                                  # hc_scale
        + hc3 * 4                                # hc_base
        # outputs
        + num_tokens * HC * hidden_size * 2      # residual_cur
        + num_tokens * HC * 4                    # post_mix_cur
        + num_tokens * HC * HC * 4               # comb_mix_cur
        + num_tokens * hidden_size * 2           # layer_input
    )
    return avg_s, flops, bytes_moved


def print_metrics(name: str, num_tokens: int, hidden_size: int, latency_us: float, flops: float, bytes_moved: float):
    latency_s = latency_us / 1e6
    tflops = flops / latency_s / 1e12
    bandwidth = bytes_moved / latency_s / 1e9
    intensity = flops / bytes_moved if bytes_moved else 0.0
    print(
        f"{name:20s} shape=({num_tokens:5d}, {hidden_size:5d}) "
        f"latency={latency_us:9.3f}us tflops={tflops:8.3f} "
        f"bandwidth={bandwidth:8.3f}GB/s intensity={intensity:8.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark mHC kernels on XPU")
    parser.add_argument("--op", choices=["mhc_pre", "mhc_post", "hc_head_fused", "mhc_fused_post_pre", "all"], default="all")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=300)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_default_device("xpu")

    if args.op in ("mhc_pre", "all"):
        for num_tokens, hidden_size in BENCH_CASES:
            avg_s, flops, bytes_moved = run_mhc_pre(num_tokens, hidden_size, args.warmup, args.iters)
            print_metrics("mhc_pre", num_tokens, hidden_size, avg_s, flops, bytes_moved)

    if args.op in ("mhc_post", "all"):
        for num_tokens, hidden_size in BENCH_CASES:
            avg_s, flops, bytes_moved = run_mhc_post(num_tokens, hidden_size, args.warmup, args.iters)
            print_metrics("mhc_post", num_tokens, hidden_size, avg_s, flops, bytes_moved)

    if args.op in ("hc_head_fused", "all"):
        for num_tokens, hidden_size in BENCH_CASES:
            avg_s, flops, bytes_moved = run_hc_head_fused(num_tokens, hidden_size, args.warmup, args.iters)
            print_metrics("hc_head_fused", num_tokens, hidden_size, avg_s, flops, bytes_moved)

    if args.op in ("mhc_fused_post_pre", "all"):
        for num_tokens, hidden_size in BENCH_CASES:
            avg_s, flops, bytes_moved = run_mhc_fused_post_pre(num_tokens, hidden_size, args.warmup, args.iters)
            print_metrics("mhc_fused_post_pre", num_tokens, hidden_size, avg_s, flops, bytes_moved)
