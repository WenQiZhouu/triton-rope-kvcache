"""Isolated Krot-output deletion ablation: intact V3 vs cache-only V3.

Modes:
  with_kout: original V3, outputs Qrot/Krot/Paged-K/V.
  no_kout:   same math, skips ONLY Krot standalone stores, still writes Cache.
All accesses otherwise identical. Meaningfully changed output contract!
"""
import argparse,csv,statistics
from pathlib import Path
import torch
import triton.testing
from bench_head_tiling import inputs
from kernels import torch_reference
from half_split_rope import launch as v3
from half_split_cache_only import launch as v3_cache_only

CASES=[
 ('medium',64,32,8,128,4,4),
 ('large',256,64,8,128,2,1)
]
ROOT=Path(__file__).resolve().parent
def bench_one(name,T,Hq,Hkv,D,bh,w,rounds=9,rep=30):
    data=inputs(T,Hq,Hkv,D,torch.float16,inactive=False)
    q,k,v,c,s,slots,kc,vc,qo,ko=data
    refq,refkc=torch_reference(q,k,c,s,slots,kc,D)
    refk,_=torch_reference(k,k,c,s,slots,kc,D)
    refvc=vc.clone()
    refvc.view(-1,Hkv,D)[slots.long()]=v
    fn={
       'with_kout':lambda:v3(q,k,v,c,s,slots,qo,ko,kc,vc,bh=bh,num_warps=w,return_compiled=True),
       'no_kout':lambda:v3_cache_only(q,k,v,c,s,slots,qo,ko,kc,vc,
                    bh=bh,num_warps=w,write_k_out=False,return_compiled=True)
    }
    metadata={}
    for label,func in fn.items():
        ko.fill_(135)
        kernel=func()
        torch.cuda.synchronize()
        for x,y in ((qo,refq),(kc,refkc),(vc,refvc)):
            torch.testing.assert_close(x,y,atol=.006,rtol=.006)
        if label=='with_kout':
            torch.testing.assert_close(ko,refk,atol=.006,rtol=.006)
        else:
            assert torch.all(ko==135).item(),'K_OUT unexpectedly modified'
        metadata[label]=dict(regs=int(kernel.n_regs),shared_bytes=int(kernel.metadata.shared),
                             ptx_lines=len(kernel.asm['ptx'].splitlines()))
    times={label:[] for label in fn}
    for i in range(rounds):
        order=['with_kout','no_kout'] if i%2==0 else ['no_kout','with_kout']
        for label in order:
            times[label].append(triton.testing.do_bench_cudagraph(fn[label],rep=rep)*1000)
    one=statistics.median(times['with_kout'])
    two=statistics.median(times['no_kout'])
    removable=T*Hkv*D*2
    result={
        'shape':name,'T':T,'Hq':Hq,'Hkv':Hkv,'D':D,'BH':bh,'Warps':w,
        'original_v3_us':round(one,4),
        'cache_only_us':round(two,4),
        'speedup':round(one/two,4),
        'removed_logical_Krot_bytes':removable,
        'original_regs':metadata['with_kout']['regs'],
        'cache_only_regs':metadata['no_kout']['regs'],
        'original_shared_bytes':metadata['with_kout']['shared_bytes'],
        'cache_only_shared_bytes':metadata['no_kout']['shared_bytes'],
        'original_ptx_lines':metadata['with_kout']['ptx_lines'],
        'cache_only_ptx_lines':metadata['no_kout']['ptx_lines'],
        'trials_with_kout':';'.join(f'{v:.4f}' for v in times['with_kout']),
        'trials_cache_only':';'.join(f'{v:.4f}' for v in times['no_kout']),
        'correctness':'PASS_Q_and_KV_cache;KOUT_dropped_by_design'
    }
    print('KOUT_ABLATION',result,flush=True)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--rounds',type=int,default=9)
    ap.add_argument('--rep',type=int,default=30)
    args=ap.parse_args()
    rows=[bench_one(*case,args.rounds,args.rep) for case in CASES]
    dst=ROOT/'profiling'/'kout_optional_ablation.csv'
    with dst.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)
    print('SAVED',dst,flush=True)
