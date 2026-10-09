"""Direct comparison to an existing *fused* official SGLang Triton kernel.

Runs standalone upstream source (saved snapshot) without installing SGLang.
Each implementation produces identical Q_rot, K_rot, Kcache, Vcache.
For apples-to-apples work, matched_kernels.py writes extra K_rot as SGLang does.
SGLang produces the rotated-Q output as flat [T,Hq*D], same underlying data.
"""
import sys
import csv
import argparse
from pathlib import Path
from statistics import median
import torch
import triton
from test_kernels import make_inputs
from kernels import torch_reference
from matched_kernels import launch as our_launch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'official'))
from sglang_rope_cache import fused_qk_rope_reshape_and_cache as sglang_fused

CASES=[
    (1,8,2,64), (1,8,2,128),
    (4,8,2,128), (16,16,4,128),
    (64,16,4,128), (128,16,4,128),
    (256,32,8,128), (512,32,8,64),
]


def run_sglang(q,k,v,cos_sin,positions,slots,kcache,vcache,qout,kout):
    return sglang_fused(
        q,k,v,kcache,vcache,slots,positions,cos_sin,
        k_scale=None,v_scale=None,is_neox=True,flash_layout=True,
        apply_scale=False,q_out=qout,k_out=kout,output_zeros=False
    )


def bench(fn,repeats=3,rep=35):
    torch.cuda.synchronize()
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        samples=[triton.testing.do_bench_cudagraph(fn,rep=rep) for _ in range(repeats)]
    torch.cuda.synchronize()
    return median(samples)*1000.0


def prepare(T,Hq,Hkv,D,dtype):
    q,k,cos,sin,slots,kcache=make_inputs(T,Hq,Hkv,D,D,dtype,inactive=False)
    v=torch.randn_like(k)
    vcache=torch.full_like(kcache,-7.)
    cs=torch.cat([cos,sin],dim=1).contiguous()
    positions=torch.arange(T,device='cuda',dtype=torch.int32)
    qo=torch.empty_like(q)
    ko=torch.empty_like(k)
    return q,k,v,cos,sin,cs,positions,slots,kcache,vcache,qo,ko


def check(data,dtype):
    q,k,v,cos,sin,cs,positions,slots,kcache,vcache,qo,ko=data
    T,Hq,D=q.shape
    ref_q,ref_kc=torch_reference(q,k,cos,sin,slots,kcache,D)
    tmp=torch.empty_like(k)
    ref_vcache=vcache.clone()
    ref_vcache.view(-1,k.shape[1],D)[slots.long()]=v
    # Avoid dependence on ordering; compare independent caches and outputs.
    for mode in ('sglang', 'ours4', 'ours2'):
        c=kcache.clone()
        vc=vcache.clone()
        out=torch.empty_like(q)
        kout=torch.empty_like(k)
        if mode=='sglang':
            run_sglang(q,k,v,cs,positions,slots,c,vc,out,kout)
        else:
            our_launch(q,k,cos,sin,slots,c,out,None,D,True,
                       4 if mode=='ours4' else 2,v=v,v_cache=vc,k_out=kout)
        torch.cuda.synchronize()
        tol=0.006 if dtype==torch.float16 else 0.00003
        try:
            torch.testing.assert_close(out,ref_q,atol=tol,rtol=tol)
            # Need Kout reference: independently compute using same RoPE
            _,ref_kout=torch_reference(k,k,cos,sin,slots,kcache,D)
            # Instead use rotate of k from independent reference: torch_reference returns qrot if q=k
            ref_krot,_=torch_reference(k,k,cos,sin,slots,kcache,D)
            torch.testing.assert_close(kout,ref_krot,atol=tol,rtol=tol)
            torch.testing.assert_close(c,ref_kc,atol=tol,rtol=tol)
            torch.testing.assert_close(vc,ref_vcache,atol=tol,rtol=tol)
        except Exception as e:
            raise AssertionError(f"Correctness failed mode={mode}, T={T}, dtype={dtype}: {e}") from e


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--output',default='comparison_sglang_fused.csv')
    args=parser.parse_args()
    print('GPU:',torch.cuda.get_device_name(0),
          'Torch:',torch.__version__,'Triton:',triton.__version__,
          'upstream SGLang fused vs own matched-output fused',flush=True)
    cases=CASES[:2] if args.quick else CASES
    records=[]
    for dtype in (torch.float16,torch.float32):
        for T,Hq,Hkv,D in cases:
            data=prepare(T,Hq,Hkv,D,dtype)
            check(data,dtype)
            q,k,v,cos,sin,cs,positions,slots,kc,vc,qo,ko=data
            base=lambda:run_sglang(q,k,v,cs,positions,slots,kc,vc,qo,ko)
            def ours(w):
                return our_launch(q,k,cos,sin,slots,kc,qo,None,D,True,w,
                                  v=v,v_cache=vc,k_out=ko)
            # Interleave variants to reduce drift in boost clocks.
            upstream_trials=[]
            our4_trials=[]
            for roundno in range(3):
                if roundno%2==0:
                    upstream_trials.append(bench(base,1,35))
                    our4_trials.append(bench(lambda:ours(4),1,35))
                else:
                    our4_trials.append(bench(lambda:ours(4),1,35))
                    upstream_trials.append(bench(base,1,35))
            up=median(upstream_trials)
            own4=median(our4_trials)
            row=dict(dtype=str(dtype).replace('torch.',''),tokens=T,
                    q_heads=Hq,kv_heads=Hkv,head_dim=D,
                    sglang_fused_us=round(up,3),ours_fused_4warps_us=round(own4,3),
                    ours_over_sglang_speedup=round(up/own4,3),
                    upstream_trials=';'.join(f'{t:.3f}' for t in upstream_trials),
                    ours_trials=';'.join(f'{t:.3f}' for t in our4_trials))
            print(row,flush=True)
            records.append(row)
    out=ROOT/args.output
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=records[0].keys())
        w.writeheader();w.writerows(records)
    print('WROTE',out,flush=True)

if __name__=='__main__':
    main()
