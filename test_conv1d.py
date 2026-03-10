import torch
import torch.nn.functional as F


def causal_conv1d_ref(
    x,
    weight,
    bias=None,
    initial_states=None,
    return_final_states=False,
    final_states_out=None,
    activation=None,
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(
            dtype_in
        )  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return out if not return_final_states else (out, final_states_out)

if __name__ == "__main__":
    import pdb; pdb.set_trace()
    
    # construct inputs and call the causal_conv1d_ref function for a quick test
    batch = 2
    dim = 4
    seqlen = 13
    width = 4
    x = torch.randn((batch, dim, seqlen))
    weight = torch.randn((dim, width))
    bias = torch.randn((dim,))
    initial_states = torch.randn((batch, dim, width - 1))
    out, final_states = causal_conv1d_ref(
        x,
        weight,
        bias=bias,
        initial_states=initial_states,
        return_final_states=True,
    )
    print("out:", out)
    print("final_states:", final_states)
