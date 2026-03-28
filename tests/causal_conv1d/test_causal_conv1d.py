# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest
import torch
import torch.nn.functional as F

import vllm_xpu_kernels._xpu_C  # noqa: F401


def _make_fwd_input(dim: int, total: int, dtype: torch.dtype, device: str, layout: str):
    if layout == "channel_first":
        return torch.randn((dim, total), dtype=dtype, device=device)
    if layout == "channel_last":
        return torch.randn((total, dim), dtype=dtype, device=device).transpose(0, 1)
    raise ValueError(f"Unsupported layout: {layout}")


def _make_update_input(
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


def _make_varlen_input(
    total: int,
    dim: int,
    dtype: torch.dtype,
    device: str,
    layout: str,
):
    if layout == "channel_first":
        return torch.randn((dim, total), dtype=dtype, device=device).transpose(0, 1)
    if layout == "channel_last":
        return torch.randn((total, dim), dtype=dtype, device=device)
    raise ValueError(f"Unsupported layout: {layout}")


def _make_conv_state(
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


def _ref_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    activation: str,
):
    dim, _ = x.shape
    width = weight.shape[1]
    state_len = width - 1

    out = torch.empty_like(x)
    conv_states_ref = conv_states.clone()

    for b in range(query_start_loc.numel() - 1):
        start = int(query_start_loc[b].item())
        end = int(query_start_loc[b + 1].item())
        seq = x[:, start:end]
        state_idx = int(cache_indices[b].item())

        if bool(has_initial_state[b].item()):
            init = conv_states_ref[state_idx]
        else:
            init = torch.zeros((dim, state_len), dtype=x.dtype, device=x.device)

        inp = torch.cat([init, seq], dim=1).unsqueeze(0)
        y = F.conv1d(
            inp.to(torch.float32),
            weight.unsqueeze(1).to(torch.float32),
            bias.to(torch.float32) if bias is not None else None,
            groups=dim,
        )
        if activation in ("silu", "swish"):
            y = F.silu(y)
        y = y.to(dtype=x.dtype).squeeze(0)
        out[:, start:end] = y

        if state_len > 0:
            conv_states_ref[state_idx] = inp.squeeze(0)[:, -state_len:]

    return out, conv_states_ref


def _ref_update(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str,
    conv_state_indices: torch.Tensor,
    query_start_loc: torch.Tensor | None,
):
    width = weight.shape[1]
    state_len = width - 1
    dim = weight.shape[0]

    conv_state_ref = conv_state.clone()
    out = torch.empty_like(x)

    if query_start_loc is None:
        batch = x.shape[0]
        for b in range(batch):
            idx = int(conv_state_indices[b].item())
            seq = x[b]
            inp = torch.cat([conv_state_ref[idx], seq], dim=1).unsqueeze(0)
            y = F.conv1d(
                inp.to(torch.float32),
                weight.unsqueeze(1).to(torch.float32),
                bias.to(torch.float32) if bias is not None else None,
                groups=dim,
            )
            if activation in ("silu", "swish"):
                y = F.silu(y)
            out[b] = y.to(dtype=x.dtype).squeeze(0)
            if state_len > 0:
                conv_state_ref[idx] = inp.squeeze(0)[:, -state_len:]
    else:
        for b in range(query_start_loc.numel() - 1):
            start = int(query_start_loc[b].item())
            end = int(query_start_loc[b + 1].item())
            idx = int(conv_state_indices[b].item())
            seq = x[start:end].transpose(0, 1)
            inp = torch.cat([conv_state_ref[idx], seq], dim=1).unsqueeze(0)
            y = F.conv1d(
                inp.to(torch.float32),
                weight.unsqueeze(1).to(torch.float32),
                bias.to(torch.float32) if bias is not None else None,
                groups=dim,
            )
            if activation in ("silu", "swish"):
                y = F.silu(y)
            out[start:end] = y.to(dtype=x.dtype).squeeze(0).transpose(0, 1)
            if state_len > 0:
                conv_state_ref[idx] = inp.squeeze(0)[:, -state_len:]

    return out, conv_state_ref


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation", ["silu", "none"])
@pytest.mark.parametrize("layout", ["channel_first", "channel_last"])
@torch.inference_mode()
def test_causal_conv1d_fwd(dtype, activation, layout):
    if not torch.xpu.is_available():
        pytest.skip("XPU is not available")

    random.seed(0)
    torch.manual_seed(0)
    device = "xpu"

    dim = 128
    width = 4
    state_len = width - 1
    batch = 8
    cache_batch = 32

    seqlens = torch.randint(low=1, high=20, size=(batch,), dtype=torch.int32)
    query_start_loc = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(seqlens, dim=0)]
    ).to(device).to(torch.int32)
    total = int(query_start_loc[-1].item())

    x = _make_fwd_input(dim, total, dtype, device, layout)
    weight = torch.randn((dim, width), dtype=dtype, device=device)
    bias = torch.randn((dim,), dtype=dtype, device=device)
    conv_states = _make_conv_state(cache_batch, dim, state_len, dtype, device, layout)
    cache_indices = torch.tensor(
        random.sample(range(cache_batch), batch), dtype=torch.int32, device=device
    )
    has_initial_state = torch.randint(0, 2, (batch,), dtype=torch.bool, device=device)

    ref_out, ref_states = _ref_fwd(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation,
    )

    out = torch.ops._xpu_C.causal_conv1d_fwd(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices,
        has_initial_state,
        activation,
        -1,
        True,
    )

    atol = 1e-2 if dtype == torch.float16 else 2e-2
    rtol = 1e-2
    assert torch.allclose(out, ref_out, atol=atol, rtol=rtol)
    assert torch.allclose(conv_states, ref_states, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("activation", ["silu", "none"])
@pytest.mark.parametrize("layout", ["channel_first", "channel_last"])
@torch.inference_mode()
def test_causal_conv1d_update(dtype, activation, layout):
    if not torch.xpu.is_available():
        pytest.skip("XPU is not available")

    random.seed(1)
    torch.manual_seed(1)
    device = "xpu"

    dim = 128
    width = 4
    state_len = width - 1
    cache_batch = 32
    batch = 10

    weight = torch.randn((dim, width), dtype=dtype, device=device)
    bias = torch.randn((dim,), dtype=dtype, device=device)
    conv_state = _make_conv_state(cache_batch, dim, state_len, dtype, device, layout)
    conv_state_indices = torch.tensor(
        random.sample(range(cache_batch), batch), dtype=torch.int32, device=device
    )

    x = _make_update_input(batch, dim, 6, dtype, device, layout)

    ref_out, ref_state = _ref_update(
        x, conv_state, weight, bias, activation, conv_state_indices, query_start_loc=None
    )

    out = torch.ops._xpu_C.causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias,
        activation,
        conv_state_indices,
        None,
        None,
        -1,
        -1,
        True,
    )

    atol = 1e-2 if dtype == torch.float16 else 2e-2
    rtol = 1e-2
    assert torch.allclose(out, ref_out, atol=atol, rtol=rtol)
    assert torch.allclose(conv_state, ref_state, atol=atol, rtol=rtol)

    seqlens = torch.randint(low=1, high=8, size=(batch,), dtype=torch.int32)
    query_start_loc = torch.cat(
        [torch.zeros(1, dtype=torch.int32), torch.cumsum(seqlens, dim=0)]
    ).to(device).to(torch.int32)
    total = int(query_start_loc[-1].item())

    x_varlen = _make_varlen_input(total, dim, dtype, device, layout)
    conv_state_varlen = _make_conv_state(
        cache_batch, dim, state_len, dtype, device, layout
    )

    ref_out2, ref_state2 = _ref_update(
        x_varlen,
        conv_state_varlen,
        weight,
        bias,
        activation,
        conv_state_indices,
        query_start_loc=query_start_loc,
    )

    out2 = torch.ops._xpu_C.causal_conv1d_update(
        x_varlen,
        conv_state_varlen,
        weight,
        bias,
        activation,
        conv_state_indices,
        None,
        query_start_loc,
        int(seqlens.max().item()),
        -1,
        True,
    )

    assert torch.allclose(out2, ref_out2, atol=atol, rtol=rtol)
    assert torch.allclose(conv_state_varlen, ref_state2, atol=atol, rtol=rtol)
