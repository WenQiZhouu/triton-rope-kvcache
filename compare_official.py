"""Apples-to-apples RoPE Q/K + K/V paged cache comparison.

Upstream kernel source pinned to FlagGems-vLLM commit 4ea1b95...
The upstream RoPE op rotates full D, then reshape_and_cache_flash stores
rotated K and unmodified V. Our fused implementation performs exactly that.
No Python eager code included in speedup ratios.
"""
import csv
import sys
from pathlib import Path
from statistics import median
import torch
import triton
from kernels import launch, torch_reference
from test_kernels import make_inputs

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'official'))
from standalone_kernels import apply_rotary_pos_emb_kernel, reshape_and_cache_flash_kernel

CASES=[
    (1,8,2,64), (1,8,2,128), (4,8,2,128),
    (16,16,4,128), (64,16,4,128),
    (128,16,4,128), (256,32,8,128), (512,32,8,64),
]


def launch_upstream(q,k,v,cos,sin,slots,qout,ktmp,kcache,vcache):
    T,Hq,D=q.shape
    Hkv=k.shape[1]
    apply_rotary_pos_emb_kernel[(T,)](
        qout,ktmp,q,k,cos,sin,None,
        q.stride(0),q.stride(1),q.stride(2),
        k.stride(0),k.stride(1),k.stride(2),
        qout.stride(0),qout.stride(1),qout.stride(2),
        ktmp.stride(0),ktmp.stride(1),ktmp.stride(2),
        0,cos.stride(0),sin.stride(0),T,
        Hq,Hkv,D,triton.next_power_of_2(D),
        False,cos.shape[0],num_warps=4)
    reshape_and_cache_flash_kernel[(T,)](
        ktmp,v,kcache,vcache,slots,
        kcache.stride(0),ktmp.stride(0),v.stride(0),
        Hkv,D,kcache.shape[1],1.0,1.0,Hkv*D,num_warps=4)
    return qout,kcache,vcache


def graph_time(fn):
    # Side stream is required by triton.testing.do_bench_cudagraph on Windows.
    torch.cuda.synchronize()
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        trials=[triton.testing.do_bench_cudagraph(fn,rep=45) for _ in range(5)]
    torch.cuda.synchronize()
    return median(trials)*1000


def validate(q,k,v,cos,sin,slots,kcache,vcache,rotary):
    ref_q,ref_kcache=torch_reference(q,k,cos,sin,slots,kcache,rotary)
    ref_vcache=vcache.clone()
    good=slots>=0
    ref_vcache.view(-1,k.shape[1],k.shape[2])[slots[good].long()]=v[good]
    qout=torch.empty_like(q)
    ktmp=torch.empty_like(k)
    for method in ('upstream','fused4','fused2','split4'):
        ck=kcache.clone()
        cv=vcache.clone()
        if method=='upstream':
            launch_upstream(q,k,v,cos,sin,slots,qout,ktmp,ck,cv)
        elif method=='split4':
            launch(q,k,cos,sin,slots,ck,qout,ktmp,rotary,
                   fused=False,num_warps=4,v=v,v_cache=cv)
        else:
            launch(q,k,cos,sin,slots,ck,qout,ktmp,rotary,
                   fused=True,num_warps=(2 if method=='fused2' else 4),
                   v=v,v_cache=cv)
        torch.cuda.synchronize()
        tol=4e-3 if q.dtype==torch.float16 else 2e-5
        torch.testing.assert_close(qout,ref_q,atol=tol,rtol=tol)
        torch.testing.assert_close(ck,ref_kcache,atol=tol,rtol=tol)
        torch.testing.assert_close(cv,ref_vcache,atol=tol,rtol=tol)


def main():
    print('GPU:',torch.cuda.get_device_name(0),
          'torch:',torch.__version__,'triton:',triton.__version__,flush=True)
    records=[]
    for dtype in (torch.float16,torch.float32):
        for T,Hq,Hkv,D in CASES:
            q,k,cos,sin,slots,kcache=make_inputs(T,Hq,Hkv,D,D,dtype,inactive=True)
            v=torch.randn_like(k)
            vcache=torch.full_like(kcache,-7.0)
            validate(q,k,v,cos,sin,slots,kcache,vcache,D)
            qout=torch.empty_like(q)
            ktmp=torch.empty_like(k)
            # All methods reuse allocated buffers and update exactly Q,Kcache,Vcache.
            upstream=lambda:launch_upstream(q,k,v,cos,sin,slots,qout,ktmp,kcache,vcache)
            split=lambda:launch(q,k,cos,sin,slots,kcache,qout,ktmp,D,False,4,
                                v=v,v_cache=vcache)
            f4=lambda:launch(q,k,cos,sin,slots,kcache,qout,ktmp,D,True,4,
                             v=v,v_cache=vcache)
            u=graph_time(upstream)
            sp=graph_time(split)
            f=graph_time(f4)
            configs={4:f}
            for warps in (1,2,8):
                configs[warps]=graph_time(lambda w=warps:launch(
                    q,k,cos,sin,slots,kcache,qout,ktmp,D,True,w,
                    v=v,v_cache=vcache))
            bw=min(configs,key=configs.get)
            b=configs[bw]
            row=dict(dtype=str(dtype).replace('torch.',''),tokens=T,
                     q_heads=Hq,kv_heads=Hkv,head_dim=D,
                     official_us=round(u,3),our_split_us=round(sp,3),
                     our_fused4_us=round(f,3),our_tuned_warps=bw,
                     our_tuned_us=round(b,3),
                     vs_official_fused4=round(u/f,3),
                     vs_official_tuned=round(u/b,3),
                     vs_our_split=round(sp/f,3),
                     tuning_gain=round(f/b,3))
            records.append(row)
            print(row,flush=True)
    out=ROOT/'comparison_official.csv'
    with out.open('w',newline='',encoding='utf-8') as fp:
        writer=csv.DictWriter(fp,fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print('WROTE',str(out),flush=True)


if __name__=='__main__':
    main()
