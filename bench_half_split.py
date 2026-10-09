"""Controlled ablation V2 (paired GM loads) vs V3 (half-split register reuse).

On each exact matching (T,Hq,Hkv,D,BH,num_warps), test both implementations,
independently verify all four outputs against torch_ref, then CUDA graph
benchmark with interleaved order and collect PTX metadata.
"""
import argparse
import csv
import re
from statistics import median
from pathlib import Path
import torch
import triton
import triton.testing
from bench_head_tiling import CASES,ALL,inputs,run as run_v2
from half_split_rope import launch as launch_v3
from matched_kernels import _rope_kernel as baseline_kernel

ROOT=Path(__file__).resolve().parent

def compiled_v2_bh1(data,bw):
    q,k,v,cos,sin,slots,kc,vc,qo,ko=data
    t,hq,d=q.shape
    hkv=k.shape[1]
    return baseline_kernel[(t,hq)](
        q,k,cos,sin,slots,qo,k,kc,hq,hkv,d,d,triton.next_power_of_2(d),True,
        v,vc,True,ko,True,num_warps=bw
    )

def run_v3(data,bh,warps,compiled=False):
    q,k,v,cos,sin,slots,kc,vc,qo,ko=data
    return launch_v3(q,k,v,cos,sin,slots,qo,ko,kc,vc,bh=bh,
                     num_warps=warps,return_compiled=compiled)

def reference_outputs(data):
    from kernels import torch_reference
    q,k,v,cos,sin,slots,kc,vc,qo,ko=data
    D=q.shape[-1]
    rq,rkc=torch_reference(q,k,cos,sin,slots,kc,D)
    rk,_=torch_reference(k,k,cos,sin,slots,kc,D)
    rvc=vc.clone()
    valid=slots>=0
    rvc.view(-1,k.shape[1],D)[slots[valid].long()]=v[valid]
    return (rq,rk,rkc,rvc)

def validate(T,Hq,Hkv,D,dtype,inactive,bh,w):
    data=inputs(T,Hq,Hkv,D,dtype,inactive=inactive)
    ref=reference_outputs(data)
    _,_,_,_,_,_,kc,vc,qo,ko=data
    for kind in ('v2','v3'):
        if kind=='v2':
            run_v2(bh,w,data)
        else:
            run_v3(data,bh,w)
        torch.cuda.synchronize()
        tol=6e-3 if dtype==torch.float16 else 3e-5
        for name,val,exp in zip(('Qrot','Krot','Kcache','Vcache'),
                                (qo,ko,kc,vc),ref):
            try:
                torch.testing.assert_close(val,exp,atol=tol,rtol=tol)
            except Exception as e:
                raise AssertionError(f'{kind} BH={bh} warps={w} T={T} Hq={Hq} Hkv={Hkv} '
                                     f'D={D} dtype={dtype} invalid={inactive} {name}: {e}') from e
        # GPU result after previous kernel may contaminate stale cache: expected
        # refs for every active slot; invalid stays -7 after either kernel.
    return True

def ptx_stats(kernel):
    asm=kernel.asm['ptx']
    pats=('ld.global','st.global','shfl.sync','ld.shared','st.shared','bar.sync')
    out={k.replace('.','_'):len(re.findall(r'\b'+re.escape(k),asm)) for k in pats}
    out.update(registers=int(kernel.n_regs),
               shared_bytes=int(kernel.metadata.shared),
               ptx_lines=len(asm.splitlines()))
    return out

def bench(fn,rep):
    return triton.testing.do_bench_cudagraph(fn,rep=rep)*1000.0

def bench_shape(name,T,Hq,Hkv,D,repeats,rep):
    data=inputs(T,Hq,Hkv,D,torch.float16,inactive=False)
    entries={}
    for bh,w in ALL:
        validate(T,Hq,Hkv,D,torch.float16,False,bh,w)
        v2_fn=lambda bh=bh,w=w:run_v2(bh,w,data)
        v3_fn=lambda bh=bh,w=w:run_v3(data,bh,w)
        if bh==1:
            c2=compiled_v2_bh1(data,w)
        else:
            c2=run_v2(bh,w,data)
        c3=run_v3(data,bh,w,compiled=True)
        entries[(bh,w)]={'v2':v2_fn,'v3':v3_fn,
                         'ptx_v2':ptx_stats(c2),'ptx_v3':ptx_stats(c3),
                         'trials_v2':[],'trials_v3':[]}
        print('COMPILED',name,'BH',bh,'warp',w,'v2_regs',c2.n_regs,
              'v3_regs',c3.n_regs,'v2_shared',c2.metadata.shared,
              'v3_shared',c3.metadata.shared,flush=True)
    torch.cuda.synchronize()
    for round_no in range(repeats):
        order=ALL[round_no%len(ALL):]+ALL[:round_no%len(ALL)]
        if round_no%2:order=order[::-1]
        for key in order:
            e=entries[key]
            for mode in (('v2','v3') if round_no%2==0 else ('v3','v2')):
                e['trials_'+mode].append(bench(e[mode],rep))
        print('TIMED',name,'round',round_no+1,'/',repeats,flush=True)
    rows=[]
    for (bh,w),e in entries.items():
        v2=median(e['trials_v2'])
        v3=median(e['trials_v3'])
        row={
            'shape':name,'dtype':'float16','tokens':T,'q_heads':Hq,'kv_heads':Hkv,
            'head_dim':D,'BH':bh,'warps':w,
            'v2_us':round(v2,4),'v3_us':round(v3,4),
            'speedup_v3_over_v2':round(v2/v3,4),
            **{f'v2_{k}':v for k,v in e['ptx_v2'].items()},
            **{f'v3_{k}':v for k,v in e['ptx_v3'].items()},
            'v2_trials_us':';'.join(f'{z:.4f}' for z in e['trials_v2']),
            'v3_trials_us':';'.join(f'{z:.4f}' for z in e['trials_v3']),
            'correctness':'PASS'
        }
        print('RESULT',name,'BH',bh,'warps',w,
              'V2',f'{v2:.4f}','V3',f'{v3:.4f}',
              'speedup',f'{v2/v3:.3f}x',flush=True)
        rows.append(row)
    best2=min(rows,key=lambda x:x['v2_us'])
    best3=min(rows,key=lambda x:x['v3_us'])
    print('BEST',name,'V2:',best2['BH'],best2['warps'],best2['v2_us'],
          'V3:',best3['BH'],best3['warps'],best3['v3_us'],
          'speedup_best_over_best',round(best2['v2_us']/best3['v3_us'],3),flush=True)
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--rounds',type=int,default=5)
    p.add_argument('--rep',type=int,default=25)
    p.add_argument('--output',default='half_split_benchmark.csv')
    a=p.parse_args()
    print('ENV',torch.cuda.get_device_name(0),torch.__version__,triton.__version__,flush=True)
    # Explicit edge cases: BH tail, invalid cache slot, non-power-of-2 D,
    # FP32 correctness, head group tail. Perf remains FP16/D=128 only.
    edgecases=[
        (7,10,2,128,torch.float16,True),
        (4,32,8,96,torch.float16,True),
        (7,10,2,128,torch.float32,True),
        (4,32,8,96,torch.float32,True),
        (2,16,3,64,torch.float16,True),
    ]
    for args in edgecases:
        for bh,w in ALL:
            validate(*args,bh,w)
        print('EDGE_OK',args,'24 baseline/V3 runs',flush=True)
    shapes=CASES[:1] if a.smoke else CASES
    rows=[]
    for args in shapes:
        rows+=bench_shape(*args,a.rounds,a.rep)
    path=ROOT/a.output
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)
    print('WROTE',path,'ROWS',len(rows),flush=True)

if __name__=='__main__':
    main()
