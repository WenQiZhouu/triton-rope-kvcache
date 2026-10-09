import pytest
import torch
from kernels import launch, torch_reference


def make_inputs(tokens, hq, hkv, d, rotary, dtype, inactive=False):
    torch.manual_seed(43 + tokens + d)
    device = "cuda"
    q = torch.randn(tokens, hq, d, device=device, dtype=dtype)
    k = torch.randn(tokens, hkv, d, device=device, dtype=dtype)
    freq = torch.randn(tokens, rotary // 2, device=device, dtype=torch.float32)
    cos, sin = torch.cos(freq).contiguous(), torch.sin(freq).contiguous()
    blocks = max(2, (tokens * 2 + 15) // 16)
    cache = torch.full((blocks, 16, hkv, d), -7., device=device, dtype=dtype)
    slots = torch.randperm(blocks * 16, device=device, dtype=torch.int64)[:tokens].to(torch.int32).contiguous()
    if inactive:
        slots[-1] = -1
    return q, k, cos, sin, slots, cache


CASES = [
    (1, 8, 2, 64, 64, torch.float16),
    (4, 8, 2, 128, 64, torch.float16),
    (7, 12, 3, 96, 64, torch.float16),
    (32, 16, 4, 128, 128, torch.float16),
    (64, 32, 8, 128, 64, torch.float16),
    (8, 8, 8, 64, 64, torch.float32),
    (16, 16, 4, 128, 128, torch.float32),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA GPU required")
@pytest.mark.parametrize("tokens,hq,hkv,d,rotary,dtype", CASES)
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("warps", [1, 4])
def test_rope_cache(tokens, hq, hkv, d, rotary, dtype, fused, warps):
    q, k, cos, sin, slots, cache = make_inputs(tokens,hq,hkv,d,rotary,dtype,inactive=True)
    q_expected, c_expected = torch_reference(q,k,cos,sin,slots,cache,rotary)
    qout = torch.empty_like(q)
    ktmp = torch.empty_like(k)
    launch(q,k,cos,sin,slots,cache,qout,ktmp,rotary, fused=fused,num_warps=warps)
    torch.cuda.synchronize()
    atol, rtol = (4e-3, 4e-3) if dtype == torch.float16 else (2e-5, 2e-5)
    torch.testing.assert_close(qout, q_expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(cache, c_expected, atol=atol, rtol=rtol)
