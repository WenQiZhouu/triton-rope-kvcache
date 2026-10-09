"""V2 experimental Head Tiling: one Triton program covers BH query heads.

Same RoPE math, paired GM loads, Q_rot + K_rot outputs and paged K/V cache
writes as matched_kernels.py. Specializations: BT=1, BD=next_power_of_two(D),
BH in {2,4,8}, num_warps in {1,2,4}. Cache uses Flash layout
[num_blocks, block_size, Hkv, D], slots flatten block/slot.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _head_tile_fused(
    Q, K, V, COS, SIN, SLOTS, Q_OUT, K_OUT, KEY_CACHE, VALUE_CACHE,
    H_Q:tl.constexpr, H_KV:tl.constexpr, D:tl.constexpr,
    ROTARY:tl.constexpr, BH:tl.constexpr, BD:tl.constexpr,
):
    t = tl.program_id(0)
    head_block = tl.program_id(1)
    h = head_block * BH + tl.arange(0, BH)
    d = tl.arange(0, BD)

    half = ROTARY // 2
    valid_d = d < D
    active_rot = d < ROTARY
    pair_d = tl.where(d < half, d + half, d - half)
    pair_d = tl.where(active_rot, pair_d, 0)
    angle_d = d % half

    # One logical cos/sin vector per token, broadcast across BH heads.
    cos = tl.load(COS + t * half + angle_d, mask=active_rot, other=1).to(tl.float32)
    sin = tl.load(SIN + t * half + angle_d, mask=active_rot, other=0).to(tl.float32)
    sign = tl.where(d < half, -1.0, 1.0)

    q_base = t * H_Q * D + h[:, None] * D
    qmask = (h[:, None] < H_Q) & valid_d[None, :]
    qpair_mask = (h[:, None] < H_Q) & active_rot[None, :]
    q = tl.load(Q + q_base + d[None, :], mask=qmask, other=0).to(tl.float32)
    paired_q = tl.load(Q + q_base + pair_d[None, :], mask=qpair_mask, other=0).to(tl.float32)
    q_rot = tl.where(active_rot[None, :], q * cos[None, :] +
                     sign[None, :] * paired_q * sin[None, :], q)
    tl.store(Q_OUT + q_base + d[None, :], q_rot, mask=qmask)

    # Only first Hkv query-head IDs also own their correspondingly indexed K/V head.
    # This is the same ownership policy as matched_kernels.py; each KV head written once.
    if head_block * BH < H_KV:
        kv_base = t * H_KV * D + h[:, None] * D
        kmask = (h[:, None] < H_KV) & valid_d[None, :]
        kpair_mask = (h[:, None] < H_KV) & active_rot[None, :]
        kval = tl.load(K + kv_base + d[None, :], mask=kmask, other=0).to(tl.float32)
        paired_k = tl.load(K + kv_base + pair_d[None, :], mask=kpair_mask, other=0).to(tl.float32)
        k_rot = tl.where(active_rot[None, :], kval * cos[None, :] +
                         sign[None, :] * paired_k * sin[None, :], kval)
        tl.store(K_OUT + kv_base + d[None, :], k_rot, mask=kmask)

        slot = tl.load(SLOTS + t)
        write_mask = kmask & (slot >= 0)
        dst = slot * H_KV * D + h[:, None] * D + d[None, :]
        tl.store(KEY_CACHE + dst, k_rot, mask=write_mask)
        v = tl.load(V + kv_base + d[None, :], mask=kmask, other=0)
        tl.store(VALUE_CACHE + dst, v, mask=write_mask)


def launch(q,k,v,cos,sin,slots,q_out,k_out,key_cache,value_cache,
           *,bh=2,num_warps=4,rotary_dim=None,return_compiled=False):
    t,hq,d=q.shape
    tk,hkv,dk=k.shape
    rotary_dim = rotary_dim or d
    assert q.is_cuda and q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert t==tk and d==dk and k.shape==v.shape and hq>=hkv
    assert bh in (2,4,8) and num_warps in (1,2,4)
    assert 0<rotary_dim<=d and rotary_dim%2==0
    assert cos.shape==sin.shape==(t,rotary_dim//2)
    assert q_out.shape==q.shape and k_out.shape==k.shape
    assert q_out.is_contiguous() and k_out.is_contiguous()
    assert key_cache.shape==value_cache.shape
    assert key_cache.ndim==4 and key_cache.shape[2:]==(hkv,d)
    assert key_cache.is_contiguous() and value_cache.is_contiguous()
    assert slots.shape==(t,) and slots.dtype in (torch.int32,torch.int64)
    kernel=_head_tile_fused[(t,triton.cdiv(hq,bh))](
        q,k,v,cos,sin,slots,q_out,k_out,key_cache,value_cache,
        hq,hkv,d,rotary_dim,bh,triton.next_power_of_2(d),
        num_warps=num_warps
    )
    return kernel if return_compiled else (q_out,key_cache,value_cache)
