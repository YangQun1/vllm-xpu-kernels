# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import random
import time

import torch

import vllm_xpu_kernels._xpu_C  # noqa: F401

STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
}


def make_fwd_input(dim: int, total: int, dtype: torch.dtype, device: str, layout: str):
    if layout == "channel_first":
        return torch.randn((dim, total), dtype=dtype, device=device)
    if layout == "channel_last":
        return torch.randn((total, dim), dtype=dtype, device=device).transpose(0, 1)
    raise ValueError(f"Unsupported layout: {layout}")


def make_update_input(
    batch: int,
    dim: int,
    seqlen: int,
    dtype: torch.dtype,
    device: str,
    layout: str,
):
    if layout == "channel_first":
        return torch.randn((batch, dim, seqlen), dtype=dtype, device=device)
    if layout == "channel_last":
        return torch.randn((batch, seqlen, dim), dtype=dtype, device=device).transpose(1, 2)
    raise ValueError(f"Unsupported layout: {layout}")


def make_varlen_input(total: int, dim: int, dtype: torch.dtype, device: str, layout: str):
    if layout == "channel_first":
        return torch.randn((dim, total), dtype=dtype, device=device).transpose(0, 1)
    if layout == "channel_last":
        return torch.randn((total, dim), dtype=dtype, device=device)
    raise ValueError(f"Unsupported layout: {layout}")


def make_conv_state(
    cache_batch: int,
    dim: int,
    state_len: int,
    dtype: torch.dtype,
    device: str,
    layout: str,
):
    if layout == "channel_first":
        return torch.randn((cache_batch, dim, state_len), dtype=dtype, device=device)
    if layout == "channel_last":
        return (
            torch.randn((cache_batch, state_len, dim), dtype=dtype, device=device)
            .transpose(1, 2)
        )
    raise ValueError(f"Unsupported layout: {layout}")


@torch.inference_mode()
def benchmark_forward(args):
    device = "xpu"
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    seqlens = torch.randint(
        low=1, high=args.max_seqlen + 1, size=(args.batch,), dtype=torch.int32
    )
    query_start_loc = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(seqlens, dim=0)]
    ).to(device)
    total_tokens = int(query_start_loc[-1].item())

    dim = args.dim
    width = args.width
    state_len = width - 1

    x = make_fwd_input(dim, total_tokens, args.dtype, device, args.layout)
    weight = torch.randn((dim, width), dtype=args.dtype, device=device)
    bias = torch.randn((dim,), dtype=args.dtype, device=device) if args.has_bias else None
    conv_states = make_conv_state(
        args.cache_batch_size, dim, state_len, args.dtype, device, args.layout
    )
    cache_indices = torch.tensor(
        random.sample(range(args.cache_batch_size), args.batch),
        dtype=torch.int32,
        device=device,
    )
    has_initial_state = torch.ones((args.batch,), dtype=torch.bool, device=device)

    def run_once():
        torch.ops._xpu_C.causal_conv1d_fwd(
            x,
            weight,
            bias,
            conv_states,
            query_start_loc,
            cache_indices,
            has_initial_state,
            args.activation,
            -1,
            False,
        )

    for _ in range(args.warmup):
        run_once()
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.iters):
        run_once()
    torch.xpu.synchronize()
    t1 = time.perf_counter()

    avg_s = (t1 - t0) / args.iters
    print(f"[forward][{args.layout}] avg latency: {avg_s * 1e6:.2f} us")
    print(f"[forward][{args.layout}] throughput: {total_tokens / avg_s:.2f} tokens/s")


@torch.inference_mode()
def benchmark_update(args):
    device = "xpu"
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    dim = args.dim
    width = args.width
    state_len = width - 1

    weight = torch.randn((dim, width), dtype=args.dtype, device=device)
    bias = torch.randn((dim,), dtype=args.dtype, device=device) if args.has_bias else None

    if args.varlen:
        seqlens = torch.randint(
            low=1, high=args.max_seqlen + 1, size=(args.batch,), dtype=torch.int32
        )
        query_start_loc = torch.cat(
            [torch.zeros(1, dtype=torch.int32), torch.cumsum(seqlens, dim=0)]
        ).to(device)
        total_tokens = int(query_start_loc[-1].item())
        x = make_varlen_input(total_tokens, dim, args.dtype, device, args.layout)
        max_query_len = int(seqlens.max().item())
    else:
        query_start_loc = None
        x = make_update_input(
            args.batch,
            dim,
            args.max_seqlen,
            args.dtype,
            device,
            args.layout,
        )
        max_query_len = -1

    conv_state = make_conv_state(
        args.cache_batch_size, dim, state_len, args.dtype, device, args.layout
    )
    conv_state_indices = torch.tensor(
        random.sample(range(args.cache_batch_size), args.batch),
        dtype=torch.int32,
        device=device,
    )

    def run_once():
        torch.ops._xpu_C.causal_conv1d_update(
            x,
            conv_state,
            weight,
            bias,
            args.activation,
            conv_state_indices,
            None,
            query_start_loc,
            max_query_len,
            -1,
            False,
        )

    for _ in range(args.warmup):
        run_once()
    torch.xpu.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.iters):
        run_once()
    torch.xpu.synchronize()
    t1 = time.perf_counter()

    avg_s = (t1 - t0) / args.iters
    print(f"[update][{args.layout}] avg latency: {avg_s * 1e6:.2f} us")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Benchmark causal_conv1d kernels")
    parser.add_argument("--mode", choices=["forward", "update"], default="forward")
    parser.add_argument("--dtype", choices=["half", "bfloat16"], default="bfloat16")
    parser.add_argument("--activation", default="silu")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--dim", type=int, default=8192)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--max-seqlen", type=int, default=2048)
    parser.add_argument("--cache-batch-size", type=int, default=1024)
    parser.add_argument(
        "--layout", choices=["channel_first", "channel_last"], default="channel_last"
    )
    parser.add_argument("--has-bias", action="store_true")
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.dtype = STR_DTYPE_TO_TORCH_DTYPE[args.dtype]

    torch.set_default_device("xpu")
    if args.mode == "forward":
        benchmark_forward(args)
    else:
        benchmark_update(args)
