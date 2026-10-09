"""Experimental V3 half-split NeoX RoPE fused with Paged KV cache.

For every frequency index i, load x[i] and x[i+D/2] into the same
logical Triton element; reuse both operands to compute the two outputs.
Theoretical logical Q/K GM input elements = D, rather than 2*D.
BT=1, BH=1/2/4/8, num_warps=1/2/4, full-dimensional NeoX RoPE only.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _half_split_fused(
    Q, K, V, COS, SIN, SLOTS, Q_OUT, K_OUT, KC, VC,
    H_Q:tl.constexpr, H_KV:tl.constexpr, D:tl.constexpr,
    BH:tl.constexpr, HALF_BLOCK:tl.constexpr, BD:tl.constexpr
):
    t = tl.program_id(0)
    group = tl.program_id(1)
    h = group * BH + tl.arange(0,BH)
    i = tl.arange(0,HALF_BLOCK)
    half:tl.constexpr = D // 2
    valid = i < half

    # Each logical lane uses the same pair (x[i], x[i+half]) twice.
    c=tl.load(COS+t*half+i,mask=valid,other=1).to(tl.float32)
    s=tl.load(SIN+t*half+i,mask=valid,other=0).to(tl.float32)

    qbase=t*H_Q*D + h[:,None]*D
    qmask=(h[:,None] < H_Q)&valid[None,:]
    qlo=tl.load(Q+qbase+i[None,:],mask=qmask,other=0).to(tl.float32)
    qhi=tl.load(Q+qbase+half+i[None,:],mask=qmask,other=0).to(tl.float32)
    olo=qlo*c[None,:]-qhi*s[None,:]
    ohi=qhi*c[None,:]+qlo*s[None,:]
    tl.store(Q_OUT+qbase+i[None,:],olo,mask=qmask)
    tl.store(Q_OUT+qbase+half+i[None,:],ohi,mask=qmask)

    if group*BH < H_KV:
        kbase=t*H_KV*D + h[:,None]*D
        kmask=(h[:,None]<H_KV)&valid[None,:]
        klo=tl.load(K+kbase+i[None,:],mask=kmask,other=0).to(tl.float32)
        khi=tl.load(K+kbase+half+i[None,:],mask=kmask,other=0).to(tl.float32)
        oklo=klo*c[None,:]-khi*s[None,:]
        okhi=khi*c[None,:]+klo*s[None,:]
        tl.store(K_OUT+kbase+i[None,:],oklo,mask=kmask)
        tl.store(K_OUT+kbase+half+i[None,:],okhi,mask=kmask)

        slot=tl.load(SLOTS+t)
        cmask=kmask&(slot>=0)
        dst=slot*H_KV*D+h[:,None]*D
        tl.store(KC+dst+i[None,:],oklo,mask=cmask)
        tl.store(KC+dst+half+i[None,:],okhi,mask=cmask)

        # Keep V path as in V2 (load one contiguous [BH,BD] tensor).
        d=tl.arange(0,BD)
        vmask=(h[:,None]<H_KV)&(d[None,:]<D)&(slot>=0)
        vval=tl.load(V+kbase+d[None,:],mask=(h[:,None]<H_KV)&(d[None,:]<D),other=0)
        tl.store(VC+dst+d[None,:],vval,mask=vmask)

def launch(q,k,v,cos,sin,slots,q_out,k_out,key_cache,value_cache,
           *,bh=1,num_warps=4,rotary_dim=None,return_compiled=False):
    t,hq,d=q.shape
    tk,hkv,dk=k.shape
    rotary_dim=rotary_dim or d
    assert rotary_dim==d and d%2==0, 'Experimental V3 requires full even-dimensional NeoX RoPE'
    assert bh in (1,2,4,8) and num_warps in (1,2,4)
    assert t==tk and d==dk and hq>=hkv and k.shape==v.shape
    assert q.is_cuda and all(x.is_contiguous() for x in (
       q,k,v,cos,sin,slots,q_out,k_out,key_cache,value_cache))
    assert q_out.shape==q.shape and k_out.shape==k.shape
    assert cos.shape==sin.shape==(t,d//2)
    assert slots.shape==(t,) and slots.dtype in (torch.int32,torch.int64)
    assert key_cache.shape==value_cache.shape
    assert key_cache.ndim==4 and key_cache.shape[2:]==(hkv,d)
    compiled=_half_split_fused[(t,triton.cdiv(hq,bh))](
        q,k,v,cos,sin,slots,q_out,k_out,key_cache,value_cache,
        hq,hkv,d,bh,triton.next_power_of_2(d//2),triton.next_power_of_2(d),
        num_warps=num_warps
    )
    return compiled if return_compiled else (q_out,key_cache,value_cache)
