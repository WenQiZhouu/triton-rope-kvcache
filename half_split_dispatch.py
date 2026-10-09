"""Conservative V3 dispatch for exact RTX3060, FP16, measured GQA shapes."""
import torch
from matched_kernels import launch as original_launch
from head_tiling import launch as head_tiled_launch
from half_split_rope import launch as half_split_launch

def select_config(q,k):
    T,Hq,D=q.shape
    Hkv=k.shape[1]
    if q.dtype==torch.float16 and D==128 and q.device.type=='cuda':
        if (T,Hq,Hkv)==(1,32,8):
            return ('half_split',1,4)
        if (T,Hq,Hkv)==(64,32,8):
            return ('half_split',4,4)
        if (T,Hq,Hkv)==(256,64,8):
            return ('v2_head_tile',2,1)
    return ('v1',1,4)

def launch(q,k,v,cos,sin,slots,q_out,k_out,kcache,vcache):
    family,bh,w=select_config(q,k)
    if family=='half_split':
        half_split_launch(q,k,v,cos,sin,slots,q_out,k_out,kcache,vcache,
                          bh=bh,num_warps=w)
    elif family=='v2_head_tile':
        head_tiled_launch(q,k,v,cos,sin,slots,q_out,k_out,kcache,vcache,
                          bh=bh,num_warps=w)
    else:
        original_launch(q,k,cos,sin,slots,kcache,q_out,None,q.shape[-1],
                        True,w,v=v,v_cache=vcache,k_out=k_out)
    return family,bh,w
