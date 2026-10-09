"""Conservative experimental dispatch from benchmarked GQA shapes.

Only exact tested FP16 shapes get BH>1. All untested shapes fall back to
the proven V1 matched-output path BH=1,num_warps=4.
This is not a general autotuner or a ready-to-deploy SGLang plugin.
"""
import torch
from matched_kernels import launch as launch_bh1
from head_tiling import launch as launch_head_tile

def select_config(q,k):
    t,hq,d=q.shape
    _,hkv,_=k.shape
    if q.dtype==torch.float16 and d==128:
        if (t,hq,hkv)==(64,32,8):
            return (4,4)
        if (t,hq,hkv)==(256,64,8):
            # BH=2,W1 and BH=4,W1 nearly tied in two independent runs.
            return (4,1)
    return (1,4)

def launch(q,k,v,cos,sin,slots,q_out,k_out,key_cache,value_cache):
    bh,w=select_config(q,k)
    if bh==1:
        launch_bh1(q,k,cos,sin,slots,key_cache,q_out,None,q.shape[-1],
                    True,w,v=v,v_cache=value_cache,k_out=k_out)
    else:
        launch_head_tile(q,k,v,cos,sin,slots,q_out,k_out,key_cache,value_cache,
                         bh=bh,num_warps=w)
    return bh,w
