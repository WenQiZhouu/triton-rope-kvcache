"""BH=1/2/4/8 x num_warps=1/2/4, GQA small/medium/large.

Correctness: Qout, Kout, Kcache, Vcache against independent torch reference,
including invalid slot and non-divisible heads D edge cases. Performance:
FP16, CUDA Graph, compare existing BH=1 and all head-tiling variants.
BH=1 uses the existing matched-output fused kernel, not a newly rewritten baseline.
"""
import argparse
import csv
import itertools
import math
import statistics
from pathlib import Path
import torch
import triton
import triton.testing
from test_kernels import make_inputs
from kernels import torch_reference
from matched_kernels import launch as launch_bh1
from head_tiling import launch as launch_bh
from compare_sglang_fused import run_sglang

ROOT=Path(__file__).resolve().parent
CASES=[
  ('small_decode', 1,32,8,128),
  ('medium_prefill',64,32,8,128),
  ('large_prefill',256,64,8,128),
]
ALL=[(bh,warps) for bh in (1,2,4,8) for warps in (1,2,4)]

def inputs(T,Hq,Hkv,D,dtype,inactive=False):
    q,k,cos,sin,slots,kc=make_inputs(T,Hq,Hkv,D,D,dtype,inactive=inactive)
    v=torch.randn_like(k)
    vc=torch.full_like(kc,-7.)
    qo=torch.empty_like(q)
    ko=torch.empty_like(k)
    return q,k,v,cos,sin,slots,kc,vc,qo,ko

def run(bh,warps,data):
    q,k,v,cos,sin,slots,kc,vc,qo,ko=data
    if bh==1:
        launch_bh1(q,k,cos,sin,slots,kc,qo,None,q.shape[-1],True,warps,
                   v=v,v_cache=vc,k_out=ko)
        return None
    return launch_bh(q,k,v,cos,sin,slots,qo,ko,kc,vc,bh=bh,num_warps=warps,
                  return_compiled=True)

@torch.no_grad()
def verify(bh,warps,T,Hq,Hkv,D,dtype,inactive=False):
    data=inputs(T,Hq,Hkv,D,dtype,inactive=inactive)
    q,k,v,cos,sin,slots,kc,vc,qo,ko=data
    ref_q,ref_kc=torch_reference(q,k,cos,sin,slots,kc,D)
    # Use the same independent torch math for K_out and for V Cache.
    ref_ko,_=torch_reference(k,k,cos,sin,slots,kc,D)
    ref_vc=vc.clone()
    valid=slots>=0
    ref_vc.view(-1,Hkv,D)[slots[valid].long()] = v[valid]

    compiled=run(bh,warps,data)
    torch.cuda.synchronize()
    tol=6e-3 if dtype==torch.float16 else 3e-5
    for name,actual,expected in (
            ('Q_out',qo,ref_q),('K_out',ko,ref_ko),
            ('K_cache',kc,ref_kc),('V_cache',vc,ref_vc)):
        try:
            torch.testing.assert_close(actual,expected,rtol=tol,atol=tol)
        except AssertionError as ex:
            raise AssertionError(f'{name} FAIL BH={bh} warps={warps} '
              f'T={T} Hq={Hq} Hkv={Hkv} D={D} dtype={dtype} '
              f'inactive={inactive}: {ex}') from ex
    return compiled

def cuda_graph_us(fn,rep):
    # CUDA graph captures a single kernel invocation; timings exclude Python/launch overhead.
    return triton.testing.do_bench_cudagraph(fn,rep=rep)*1000

def bench_case(label,T,Hq,Hkv,D,*,rounds=5,rep=25):
    dtype=torch.float16
    data=inputs(T,Hq,Hkv,D,dtype,inactive=False)
    q,k,v,cos,sin,slots,kc,vc,qo,ko=data
    # First compile and validate all 12 configurations, including the existing BH=1.
    entries={}
    for bh,warps in ALL:
        kernel=verify(bh,warps,T,Hq,Hkv,D,dtype)
        # Validated separate buffers. Same original benchmark buffers for all candidates.
        fn=lambda bh=bh,warps=warps:run(bh,warps,data)
        fn()
        if bh==1:
            meta_kernel=__import__('matched_kernels')._rope_kernel
            grid=(T,Hq)
        else:
            grid=(T,triton.cdiv(Hq,bh))
        meta={'registers_per_thread':int(kernel.n_regs) if kernel is not None else None,
              'shared_bytes':int(kernel.metadata.shared) if kernel is not None else None}
        entries[(bh,warps)]=dict(fn=fn, grid_programs=grid[0]*grid[1],**meta,samples=[])

    # Baseline n_regs queried via compile handle from warmup cache to avoid changing source;
    # hardware counts available for BH>1; baseline metadata may be fetched separately.
    sg_pos=torch.arange(T,device='cuda',dtype=torch.int32)
    sg_cos_sin=torch.cat((cos,sin),dim=1).contiguous()
    official=lambda:run_sglang(q,k,v,sg_cos_sin,sg_pos,slots,kc,vc,qo,ko)
    official()
    torch.cuda.synchronize()
    sg_samples=[]
    names=list(ALL)
    print(f'PASS correctness: {label} T={T}, Hq={Hq}, Hkv={Hkv}, D={D}; '
          '12/12 configs; launching timed rounds',flush=True)
    for i in range(rounds):
        order=names[i%len(names):]+names[:i%len(names)]
        if i%2:order=order[::-1]
        if i==rounds-1:
            # After all variants' rounds, measure an official fused comparator.
            pass
        for key in order:
            entries[key]['samples'].append(cuda_graph_us(entries[key]['fn'],rep))
        sg_samples.append(cuda_graph_us(official,rep))
        print(f'  {label}: benchmark round {i+1}/{rounds} complete',flush=True)

    out=[]
    sg_us=statistics.median(sg_samples)
    bh1_4=statistics.median(entries[(1,4)]['samples'])
    bh1_best=min(statistics.median(entries[(1,w)]['samples']) for w in (1,2,4))
    for bh,warps in ALL:
        e=entries[(bh,warps)]
        us=statistics.median(e['samples'])
        row={
            'case':label,'dtype':'float16','tokens':T,'q_heads':Hq,'kv_heads':Hkv,
            'head_dim':D,'BH':bh,'warps':warps,'grid_programs':e['grid_programs'],
            'tile_elements':bh*D,'latency_us':round(us,4),
            'vs_bh1_w4':round(bh1_4/us,4),
            'vs_bh1_best':round(bh1_best/us,4),
            'sglang_fused_us':round(sg_us,4),
            'registers_per_thread':e['registers_per_thread'],
            'shared_bytes_per_program':e['shared_bytes'],
            'trials_us':';'.join(f'{x:.4f}' for x in e['samples']),
            'correctness':'PASS'
        }
        out.append(row)
        print(f"    BH={bh} W={warps} {us:.4f} us  "
              f"vs_BH1_4={bh1_4/us:.3f}x grid={e['grid_programs']}",flush=True)
    best=min(out,key=lambda x:x['latency_us'])
    print(f"BEST {label}: BH={best['BH']}, warps={best['warps']}, "
          f"{best['latency_us']:.4f}us; BH1(w4)={bh1_4:.4f}us "
          f"speedup={best['vs_bh1_w4']:.3f}x; SGLang={sg_us:.4f}us",flush=True)
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--rounds',type=int,default=5)
    parser.add_argument('--rep',type=int,default=25)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--output',default='head_tiling_benchmark.csv')
    a=parser.parse_args()
    print(f"GPU={torch.cuda.get_device_name(0)}, torch={torch.__version__},"
          f" Triton={triton.__version__}, "
          f"FP16 CUDA Graph, rounds={a.rounds}, rep={a.rep}ms",flush=True)
    print('Checking correctness of edge shapes and invalid slots...',flush=True)
    for dtype in (torch.float16,torch.float32):
        for T,Hq,Hkv,D,invalid in ((7,10,2,128,True),(4,32,8,96,True)):
            for bh,warps in ALL:
                verify(bh,warps,T,Hq,Hkv,D,dtype,inactive=invalid)
            print(f"PASS edge: T={T}, Hq={Hq}, Hkv={Hkv}, D={D}, "
                  f"{str(dtype)}, invalid_slot={invalid}, 12/12",flush=True)
    records=[]
    for case in CASES[:1] if a.quick else CASES:
        records.extend(bench_case(*case,rounds=a.rounds,rep=a.rep))
    path=ROOT/a.output
    with path.open('w',newline='',encoding='utf-8') as f:
        dw=csv.DictWriter(f,fieldnames=records[0].keys())
        dw.writeheader();dw.writerows(records)
    print('SAVED',path,'rows',len(records),flush=True)

if __name__=='__main__':main()
