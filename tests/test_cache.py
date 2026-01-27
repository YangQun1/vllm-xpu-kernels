# SPDX-License-Identifier: Apache-2.0

import random

import pytest
import torch

from tests import register_ops as ops
from tests.register_ops import reshape_and_cache, reshape_and_cache_flash, indexer_k_quant_and_cache
from tests.utils import (_convert_from_fp8, create_kv_caches_with_random,
                         create_kv_caches_with_random_flash, opcheck,
                         seed_everything)

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [42]  # Arbitrary values for testing
NUM_LAYERS = [1]  # Arbitrary values for testing
NUM_HEADS = [8]  # Arbitrary values for testing
HEAD_SIZES = [64, 80, 120, 256]
BLOCK_SIZES = [8, 16, 32]

# Parameters for MLA tests.
KV_LORA_RANKS = [512]
QK_ROPE_HEAD_DIMS = [64]
NUM_TOKENS_MLA = [42]
BLOCK_SIZES_MLA = [16]
NUM_BLOCKS_MLA = [8]

# Arbitrary values for testing
# don't make it too large. e.g. [1024, 36000] will OOM
NUM_BLOCKS = [1024]

SEEDS = [0]
DEVICES = [
    f"xpu:{i}" for i in range(1 if torch.xpu.device_count() == 1 else 2)
]

KV_CACHE_DTYPE = ["auto"]  # FIXME: will add "fp8" when accuracy is improved

#override pytest parameters when enable mini pytest
MINI_PYTEST_PARAMS = {
    "default": {
        "num_tokens": [1],
        "head_size": [8],
        "num_blocks": [4],
        "block_size": [8],
    },
    "test_concat_and_cache_mla": {
        "num_tokens": [1],
        "num_blocks": [4],
        "block_size": [8],
    },
    "test_gather_cache_mla": {
        "num_blocks": [4],
        "block_size": [8],
        "max_seq_len": [4],
    },
}


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@torch.inference_mode()
def test_reshape_and_cache(
    num_tokens: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    kv_cache_dtype: str,
) -> None:
    if kv_cache_dtype == "fp8" and head_size % 16:
        pytest.skip()

    # Note: torch.set_default_device("xpu:1") not works.
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)
    # Create a random slot mapping.
    num_slots = block_size * num_blocks
    slot_mapping_lst = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long)

    qkv = torch.randn(num_tokens, 3, num_heads, head_size, dtype=dtype)
    _, key, value = qkv.unbind(dim=1)

    # Create the KV caches.
    key_caches, value_caches = create_kv_caches_with_random(
        num_blocks,
        block_size,
        1,
        num_heads,
        head_size,
        kv_cache_dtype,
        dtype,
        seed=seed,
        device=device,
    )
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Using default kv_scale
    k_scale = (key.amax() / 64.0).to(torch.float32)
    v_scale = (value.amax() / 64.0).to(torch.float32)

    def permute_and_compact(x):
        return x.contiguous()

    key_cache_compact = permute_and_compact(key_cache)
    value_cache_compact = permute_and_compact(value_cache)

    if kv_cache_dtype in ["fp8", "fp8_e4m3", "fp8_e5m2"]:
        cloned_key_cache = _convert_from_fp8(key_cache_compact, k_scale.item())
        cloned_value_cache = _convert_from_fp8(value_cache_compact,
                                               v_scale.item())
    else:
        cloned_key_cache = key_cache_compact.clone()
        cloned_value_cache = value_cache_compact.clone()

    # Call the reshape_and_cache kernel.
    opcheck(
        torch.ops._C_cache_ops.reshape_and_cache,
        (
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        ),
        cond=(head_size == HEAD_SIZES[0]),
    )
    reshape_and_cache(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )
    key_cache_compact = permute_and_compact(key_cache)
    value_cache_compact = permute_and_compact(value_cache)

    reshaped_key = key.reshape(num_tokens, *key_cache[0, :, :, 0, :].shape)
    block_indices = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indices_lst = block_indices.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets_lst = block_offsets.cpu().tolist()
    for i in range(num_tokens):
        block_idx = block_indices_lst[i]
        block_offset = block_offsets_lst[i]
        cloned_key_cache[block_idx, :, :, block_offset, :] = reshaped_key[i]
        cloned_value_cache[block_idx, :, :, block_offset] = value[i]

    if kv_cache_dtype in ["fp8", "fp8_e4m3", "fp8_e5m2"]:
        result_key_cache = _convert_from_fp8(key_cache_compact, k_scale.item())
        result_value_cache = _convert_from_fp8(value_cache_compact,
                                               v_scale.item())
        torch.testing.assert_close(result_key_cache,
                                   cloned_key_cache,
                                   atol=0.1,
                                   rtol=0.1)
        torch.testing.assert_close(result_value_cache,
                                   cloned_value_cache,
                                   atol=0.1,
                                   rtol=0.1)
    else:
        torch.testing.assert_close(key_cache_compact, cloned_key_cache)
        torch.testing.assert_close(value_cache_compact, cloned_value_cache)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@torch.inference_mode()
def test_reshape_and_cache_flash(
    num_tokens: int,
    num_heads: int,
    num_layers: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    kv_cache_dtype: str,
) -> None:
    # Note: torch.set_default_device("xpu:1") not works.
    torch.set_default_device("xpu")
    torch.xpu.set_device(device)

    # Create a random slot mapping.
    num_slots = block_size * num_blocks
    slot_mapping_lst = random.sample(range(num_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst,
                                dtype=torch.long,
                                device=device)
    qkv = torch.randn(num_tokens,
                      3,
                      num_heads,
                      head_size,
                      dtype=dtype,
                      device=device)
    _, key, value = qkv.unbind(dim=1)

    # Create the KV caches.
    key_caches, value_caches = create_kv_caches_with_random_flash(
        num_blocks,
        block_size,
        1,  # num_layers
        num_heads,
        head_size,
        kv_cache_dtype,
        dtype,
        seed=seed,
        device=device,
    )
    key_cache, value_cache = key_caches[0], value_caches[0]

    k_scale = (key.amax() / 64.0).to(torch.float32)
    v_scale = (value.amax() / 64.0).to(torch.float32)

    def permute_and_compact(x):
        return x.contiguous()

    key_cache_compact = permute_and_compact(key_cache)
    value_cache_compact = permute_and_compact(value_cache)

    # Clone the KV caches.
    if kv_cache_dtype in ["fp8", "fp8_e4m3", "fp8_e5m2"]:
        cloned_key_cache = _convert_from_fp8(key_cache_compact, k_scale.item())
        cloned_value_cache = _convert_from_fp8(value_cache_compact,
                                               v_scale.item())
    else:
        cloned_key_cache = key_cache_compact.clone()
        cloned_value_cache = value_cache_compact.clone()

    # Call the reshape_and_cache kernel.
    opcheck(
        torch.ops._C_cache_ops.reshape_and_cache_flash,
        (
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        ),
    )
    reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )
    key_cache_compact = permute_and_compact(key_cache)
    value_cache_compact = permute_and_compact(value_cache)

    # Run the reference implementation.
    block_indicies = torch.div(slot_mapping, block_size, rounding_mode="floor")
    block_indicies_lst = block_indicies.cpu().tolist()
    block_offsets = slot_mapping % block_size
    block_offsets_lst = block_offsets.cpu().tolist()
    for i in range(num_tokens):
        block_idx = block_indicies_lst[i]
        block_offset = block_offsets_lst[i]
        cloned_key_cache[block_idx, block_offset, :, :] = key[i]
        cloned_value_cache[block_idx, block_offset, :, :] = value[i]
    if kv_cache_dtype in ["fp8", "fp8_e4m3", "fp8_e5m2"]:
        result_key_cache = _convert_from_fp8(key_cache_compact, k_scale.item())
        result_value_cache = _convert_from_fp8(value_cache_compact,
                                               v_scale.item())
        torch.testing.assert_close(result_key_cache,
                                   cloned_key_cache,
                                   atol=0.1,
                                   rtol=0.1)
        torch.testing.assert_close(result_value_cache,
                                   cloned_value_cache,
                                   atol=0.1,
                                   rtol=0.1)
    else:
        torch.testing.assert_close(key_cache_compact, cloned_key_cache)
        torch.testing.assert_close(value_cache_compact, cloned_value_cache)


def _create_mla_cache(
    num_blocks: int,
    block_size: int,
    entry_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    device: str,
) -> torch.Tensor:
    cache_dtype = torch.uint8 if kv_cache_dtype == "fp8" else dtype
    return torch.zeros(num_blocks,
                       block_size,
                       entry_size,
                       dtype=cache_dtype,
                       device=device)


def _fill_mla_cache(cache: torch.Tensor, kv_cache_dtype: str):
    rand_dtype = torch.float16 if kv_cache_dtype == "fp8" else cache.dtype

    vals = torch.randn(*cache.shape, device=cache.device, dtype=rand_dtype)
    if kv_cache_dtype == "fp8":
        temp = torch.zeros_like(cache)
        ops.convert_fp8(temp, vals, 1.0, kv_dtype=kv_cache_dtype)
        vals = temp
    cache.copy_(vals)


@pytest.mark.parametrize("kv_lora_rank", KV_LORA_RANKS)
@pytest.mark.parametrize("qk_rope_head_dim", QK_ROPE_HEAD_DIMS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS_MLA)
@pytest.mark.parametrize("block_size", BLOCK_SIZES_MLA)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS_MLA)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@torch.inference_mode()
def test_concat_and_cache_mla(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_tokens: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
    kv_cache_dtype: str,
) -> None:
    seed_everything(seed)
    torch.set_default_device(device)

    total_slots = num_blocks * block_size
    slot_mapping_lst = random.sample(range(total_slots), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst,
                                dtype=torch.long,
                                device=device)

    kv_c = torch.randn(num_tokens, kv_lora_rank, dtype=dtype, device=device)
    k_pe = torch.randn(num_tokens,
                       qk_rope_head_dim,
                       dtype=dtype,
                       device=device)
    entry_size = kv_lora_rank + qk_rope_head_dim

    scale = torch.tensor(0.1, dtype=torch.float32, device=device)
    kv_cache = _create_mla_cache(num_blocks, block_size, entry_size, dtype,
                                 kv_cache_dtype, device)
    ref_temp = torch.zeros(*kv_cache.shape, dtype=dtype, device=device)

    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        block_idx = slot // block_size
        block_offset = slot % block_size
        ref_temp[block_idx, block_offset, :kv_lora_rank] = kv_c[i]
        ref_temp[block_idx, block_offset, kv_lora_rank:] = k_pe[i]

    if kv_cache_dtype == "fp8":
        ref_kv_cache = torch.empty_like(ref_temp, dtype=kv_cache.dtype)
        ops.convert_fp8(ref_kv_cache,
                        ref_temp,
                        scale.item(),
                        kv_dtype=kv_cache_dtype)
    else:
        ref_kv_cache = ref_temp

    opcheck(
        torch.ops._C_cache_ops.concat_and_cache_mla,
        (kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale),
        # test_utils=DEFAULT_OPCHECK_TEST_UTILS,
    )

    ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping,
                             kv_cache_dtype, scale)

    if kv_cache_dtype == "fp8":
        result_temp = torch.empty_like(kv_cache, dtype=torch.float16)
        ops.convert_fp8(result_temp,
                        kv_cache.contiguous(),
                        scale.item(),
                        kv_dtype=kv_cache_dtype)
        expected_temp = torch.empty_like(ref_kv_cache, dtype=torch.float16)
        ops.convert_fp8(expected_temp,
                        ref_kv_cache,
                        scale.item(),
                        kv_dtype=kv_cache_dtype)
        torch.testing.assert_close(result_temp,
                                   expected_temp,
                                   atol=0.001,
                                   rtol=0.1)
    else:
        torch.testing.assert_close(kv_cache, ref_kv_cache)


@pytest.mark.parametrize("kv_lora_rank", [512])
@pytest.mark.parametrize("qk_rope_head_dim", [64])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_blocks", [1024])
@pytest.mark.parametrize("max_seq_len", [512])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("kv_cache_dtype",
                         ["auto"])  # You can also test "fp8" if needed.
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_gather_cache_mla(kv_lora_rank, qk_rope_head_dim, block_size,
                          num_blocks, max_seq_len, batch_size, dtype,
                          kv_cache_dtype, device):
    entry_size = kv_lora_rank + qk_rope_head_dim
    src_cache = _create_mla_cache(num_blocks, block_size, entry_size, dtype,
                                  kv_cache_dtype, device)
    _fill_mla_cache(src_cache, kv_cache_dtype=kv_cache_dtype)

    seq_len_tensor = torch.randint(0,
                                   max_seq_len + 1, (batch_size, ),
                                   device=device)

    total_tokens = seq_len_tensor.sum()
    cu_seq_lens = torch.empty((batch_size + 1),
                              dtype=torch.int32,
                              device=device)
    cu_seq_lens[0] = 0
    cu_seq_lens[1:] = seq_len_tensor.cumsum(dim=0).to(dtype=torch.int32)
    print("seq_len_tensor", seq_len_tensor)

    tot_blocks_tensor = (seq_len_tensor + block_size - 1) // block_size
    block_table = torch.empty((batch_size, num_blocks),
                              dtype=torch.int32,
                              device=device)

    for b in range(batch_size):
        perm = torch.randperm(num_blocks, device=device)
        block_table[b, :] = perm

    dst = torch.zeros((total_tokens, entry_size),
                      dtype=src_cache.dtype,
                      device=device)

    expected_batches = []
    for b in range(batch_size):
        s = seq_len_tensor[b]
        if s == 0:
            continue
        tot = tot_blocks_tensor[b]
        blocks = block_table[b, :tot].tolist()

        gathered_rows = []
        for i in range(tot - 1):
            gathered_rows.append(src_cache[blocks[i]])
        remaining = s - (tot - 1) * block_size
        gathered_rows.append(src_cache[blocks[-1], :remaining, :])

        batch_expected = torch.cat(gathered_rows, dim=0)
        expected_batches.append(batch_expected)
    expected = torch.cat(expected_batches, dim=0)

    opcheck(
        torch.ops._C_cache_ops.gather_cache,
        (src_cache, dst, block_table, cu_seq_lens, batch_size, None),
        # test_utils=DEFAULT_OPCHECK_TEST_UTILS,
    )

    ops.gather_cache(src_cache, dst, block_table, cu_seq_lens, batch_size)
    torch.testing.assert_close(dst, expected)

def _pytorch_group_quant(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    dtype: torch.dtype | None = None,
    column_major_scales: bool = False,
    out_q: torch.Tensor | None = None,
    use_ue8m0: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_ue8m0 is None:
        # Default fallback - could import is_deep_gemm_e8m0_used if needed
        use_ue8m0 = False

    if dtype is None:
        # dtype = current_platform.fp8_dtype()
        dtype = torch.float8_e4m3fn

    # Validate inputs
    assert x.shape[-1] % group_size == 0, (
        f"Last dimension {x.shape[-1]} must be divisible by group_size {group_size}"
    )
    assert x.stride(-1) == 1, "Input tensor groups must be contiguous"

    # Prepare output tensor
    if out_q is None:
        x_q = torch.empty_like(x, dtype=dtype)
    else:
        assert out_q.shape == x.shape
        x_q = out_q

    # Reshape input for group processing
    # Original shape: (..., last_dim)
    # Target shape: (..., num_groups, group_size)
    original_shape = x.shape
    num_groups = original_shape[-1] // group_size

    # Reshape to separate groups
    group_shape = original_shape[:-1] + (num_groups, group_size)
    x_grouped = x.view(group_shape)

    # Compute per-group absolute maximum values
    # Shape: (..., num_groups)
    abs_max = torch.amax(torch.abs(x_grouped), dim=-1, keepdim=False)
    abs_max = torch.maximum(abs_max, torch.tensor(eps, device=x.device, dtype=x.dtype))

    # Compute scales
    FP8_MAX = torch.finfo(dtype).max
    FP8_MIN = torch.finfo(dtype).min
    scale_raw = abs_max / FP8_MAX

    if use_ue8m0:
        # For UE8M0 format, scales must be powers of 2
        scales = torch.pow(2.0, torch.ceil(torch.log2(scale_raw)))
    else:
        scales = scale_raw

    # Expand scales for broadcasting with grouped data
    # Shape: (..., num_groups, 1)
    scales_expanded = scales.unsqueeze(-1)

    # Quantize the grouped data
    x_scaled = x_grouped / scales_expanded
    x_clamped = torch.clamp(x_scaled, FP8_MIN, FP8_MAX)
    x_quantized = x_clamped.to(dtype)

    # Reshape back to original shape
    x_q.copy_(x_quantized.view(original_shape))

    # Prepare scales tensor in requested format
    if column_major_scales:
        # Column-major: (num_groups,) + batch_dims
        # Transpose the scales to put group dimension first
        scales_shape = (num_groups,) + original_shape[:-1]
        x_s = scales.permute(-1, *range(len(original_shape) - 1))
        x_s = x_s.contiguous().view(scales_shape)
    else:
        # Row-major: batch_dims + (num_groups,)
        x_s = scales.contiguous()

    # Ensure scales are float32
    return x_q, x_s.float()


def _pytorch_indexer_k_quant_and_cache(
    k: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
) -> None:
    head_dim = k.shape[-1]
    k = k.view(-1, head_dim)  # [total_tokens, head_dim]

    k_fp8, k_scale = _pytorch_group_quant(
        k,
        group_size=quant_block_size,
        column_major_scales=False,
        use_ue8m0=(scale_fmt == "ue8m0"),
    )

    k_fp8_bytes = k_fp8.view(-1, head_dim).view(torch.uint8)
    scale_bytes = k_scale.view(torch.uint8).view(-1, 4)
    k = torch.cat([k_fp8_bytes, scale_bytes], dim=-1)  # [total_tokens, head_dim + 4]

    slot_mapping = slot_mapping.flatten()
    # kv_cache: [num_block, block_size, head_dim + 4]
    kv_cache.view(-1, kv_cache.shape[-1]).index_copy_(0, slot_mapping, k)

@pytest.mark.parametrize("quant_block_size", [128,])
@pytest.mark.parametrize("kv_cache_dtype", ["ue8m0",])
@pytest.mark.parametrize("num_blocks", [1,])
@pytest.mark.parametrize("num_tokens", [1,])
@pytest.mark.parametrize("device", DEVICES)
def test_indexer_k_quant_and_cache(quant_block_size, kv_cache_dtype, num_blocks, num_tokens, device):
    head_dim = 128
    cache_block_size = 128
    cache_stride = head_dim + head_dim // quant_block_size * 4

    # Create a random slot mapping.
    slot_mapping_lst = random.sample(range(cache_block_size), num_tokens)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.long, device=device)

    # Create random key tensor.
    k = torch.randn(num_tokens, head_dim, dtype=torch.bfloat16, device=device)

    # Create the KV cache.
    kv_cache = torch.zeros((num_blocks, cache_block_size, cache_stride), dtype=torch.uint8, device=device)

    # Quantize and cache the keys using the indexer kernel.
    opcheck(
        torch.ops._C_cache_ops.indexer_k_quant_and_cache,
        (k, kv_cache, slot_mapping, quant_block_size, kv_cache_dtype),
    )

    indexer_k_quant_and_cache(k, kv_cache, slot_mapping, quant_block_size, kv_cache_dtype)

    # Additional assertions can be added here to verify the correctness of the operation.
    kv_cache_ref = torch.zeros((num_blocks, cache_block_size, cache_stride), dtype=torch.uint8, device=device)
    _pytorch_indexer_k_quant_and_cache(k, kv_cache_ref, slot_mapping, quant_block_size, kv_cache_dtype)

    # TODO: fix the acc: the ref path seems not align with CUDA, to be confirmed

    print("slot_mapping: ", slot_mapping)
    
    print("res:")
    for i in range(kv_cache.view(-1, kv_cache.shape[-1]).shape[0]):
        if kv_cache.view(-1, kv_cache.shape[-1])[i].sum() > 0:
            print(i)
            print(kv_cache.view(-1, kv_cache.shape[-1])[i])

    print("ref:")
    for i in range(kv_cache_ref.view(-1, kv_cache.shape[-1]).shape[0]):
        if kv_cache_ref.view(-1, kv_cache_ref.shape[-1])[i].sum() > 0:
            print(i)
            print(kv_cache_ref.view(-1, kv_cache_ref.shape[-1])[i])

    torch.testing.assert_close(kv_cache, kv_cache_ref, atol=0.001, rtol=0.1)

