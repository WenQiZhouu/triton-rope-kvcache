"""Matched-output benchmark variant: also writes rotated K output.

RoPE (NeoX half-rotation) + paged KV-cache append, decode path.

Layout:
 q:       [tokens, query_heads, head_dim]
 k:       [tokens, kv_heads, head_dim]
 cos/sin: [tokens, rotary_dim//2]
 cache:   [physical_blocks, block_size, kv_heads, head_dim]
 slots:   [tokens] flattened physical slot (or -1 to skip cache write).

The fused kernel writes rotated Q and scattered rotated K in one launch.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(
    Q, K, COS, SIN, SLOTS, Q_OUT, K_TMP, CACHE,
    H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr,
    ROTARY: tl.constexpr, BLOCK_D: tl.constexpr, FUSED: tl.constexpr,
    V, V_CACHE, WITH_V: tl.constexpr, K_OUT, WITH_K_OUT: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    half = ROTARY // 2
    is_rot = d < ROTARY
    pair_d = tl.where(d < half, d + half, d - half)
    pair_d = tl.where(is_rot, pair_d, 0)
    angle_d = d % half
    c = tl.load(COS + t * half + angle_d, mask=is_rot, other=1).to(tl.float32)
    s = tl.load(SIN + t * half + angle_d, mask=is_rot, other=0).to(tl.float32)
    sign = tl.where(d < half, -1.0, 1.0)

    q_base = t * H_Q * D + h * D
    qv = tl.load(Q + q_base + d, mask=d < D, other=0).to(tl.float32)
    qp = tl.load(Q + q_base + pair_d, mask=is_rot, other=0).to(tl.float32)
    q_rot = tl.where(is_rot, qv * c + sign * qp * s, qv)
    tl.store(Q_OUT + q_base + d, q_rot, mask=d < D)

    if h < H_KV:
        k_base = t * H_KV * D + h * D
        kv = tl.load(K + k_base + d, mask=d < D, other=0).to(tl.float32)
        kp = tl.load(K + k_base + pair_d, mask=is_rot, other=0).to(tl.float32)
        k_rot = tl.where(is_rot, kv * c + sign * kp * s, kv)
        if WITH_K_OUT:
            tl.store(K_OUT + k_base + d, k_rot, mask=d < D)
        if FUSED:
            slot = tl.load(SLOTS + t)
            tl.store(CACHE + slot * H_KV * D + h * D + d,
                     k_rot, mask=(d < D) & (slot >= 0))
            if WITH_V:
                vv = tl.load(V + k_base + d, mask=d < D, other=0)
                tl.store(V_CACHE + slot * H_KV * D + h * D + d,
                         vv, mask=(d < D) & (slot >= 0))
        else:
            tl.store(K_TMP + k_base + d, k_rot, mask=d < D)


@triton.jit
def _scatter_k_kernel(K_TMP, SLOTS, CACHE, V, V_CACHE,
                      H_KV: tl.constexpr, D: tl.constexpr,
                      TOTAL: tl.constexpr, BLOCK: tl.constexpr,
                      WITH_V: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < TOTAL
    t = idx // (H_KV * D)
    head_d = idx % (H_KV * D)
    slot = tl.load(SLOTS + t, mask=mask, other=-1)
    val = tl.load(K_TMP + idx, mask=mask, other=0)
    tl.store(CACHE + slot * (H_KV * D) + head_d, val,
             mask=mask & (slot >= 0))
    if WITH_V:
        v = tl.load(V + idx, mask=mask, other=0)
        tl.store(V_CACHE + slot * (H_KV * D) + head_d, v,
                 mask=mask & (slot >= 0))


def launch(q, k, cos, sin, slots, cache, q_out, k_tmp=None,
           rotary_dim=None, fused=True, num_warps=4,
           v=None, v_cache=None, k_out=None):
    assert q.is_cuda and k.is_cuda and q.is_contiguous() and k.is_contiguous()
    tokens, hq, d = q.shape
    tk, hkv, dk = k.shape
    assert tokens == tk and d == dk and hq >= hkv
    rotary_dim = rotary_dim or d
    assert rotary_dim % 2 == 0 and 0 < rotary_dim <= d
    assert cos.shape == sin.shape == (tokens, rotary_dim // 2)
    assert cache.ndim == 4 and cache.shape[2:] == (hkv, d)
    assert slots.shape == (tokens,) and slots.dtype in (torch.int32, torch.int64)
    assert q.dtype == k.dtype == cache.dtype == q_out.dtype
    assert cos.is_contiguous() and sin.is_contiguous() and slots.is_contiguous()
    assert q.device == k.device == cos.device == sin.device == slots.device == cache.device
    with_v = v is not None
    if with_v:
        assert v_cache is not None and v.shape == k.shape
        assert v.dtype == k.dtype == v_cache.dtype and v.is_contiguous()
        assert v.device == k.device == v_cache.device
        assert v_cache.shape == cache.shape and v_cache.is_contiguous()
    if not fused:
        assert k_tmp is not None and k_tmp.shape == k.shape and k_tmp.is_contiguous()
    _rope_kernel[(tokens, hq)](
        q, k, cos, sin, slots, q_out,
        k if fused else k_tmp, cache,
        hq, hkv, d, rotary_dim, triton.next_power_of_2(d), fused,
        v if with_v else k, v_cache if with_v else cache, with_v,
        k_out if k_out is not None else k, k_out is not None,
        num_warps=num_warps,
    )
    if not fused:
        total = tokens * hkv * d
        _scatter_k_kernel[(triton.cdiv(total, 256),)](
            k_tmp, slots, cache, v if with_v else k,
            v_cache if with_v else cache,
            hkv, d, total, 256, with_v, num_warps=4
        )
    return q_out, cache


def torch_reference(q, k, cos, sin, slots, cache, rotary_dim):
    """Torch eager reference, uses float32 arithmetic for rotary math."""
    half = rotary_dim // 2
    bc = cos.float()[:, None, :]
    bs = sin.float()[:, None, :]

    def rotate(x):
        a = x[..., :half].float()
        b = x[..., half:rotary_dim].float()
        lo = a * bc - b * bs
        hi = b * bc + a * bs
        front = torch.cat((lo, hi), dim=-1).to(x.dtype)
        return torch.cat((front, x[..., rotary_dim:]), dim=-1) if rotary_dim < x.shape[-1] else front

    q_ref = rotate(q)
    k_ref = rotate(k)
    cache_ref = cache.clone()
    valid = slots >= 0
    cache_ref.view(-1, k.shape[1], k.shape[2])[slots[valid].long()] = k_ref[valid]
    return q_ref, cache_ref
