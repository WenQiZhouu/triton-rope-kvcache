"""Same-output, same-run fair microbenchmark of FINAL selected V2/V3 vs 2K and SGLang.

All modes compute the SAME four outputs: Qrot, Krot, Paged Kcache, Paged Vcache.
- flaggems_2k: pinned extracted FlagGems-vLLM RoPE then reshape_and_cache_flash
- our_2k: own Triton split RoPE + cache scatter, preserving Krot
- sglang_fused: saved official SGLang fused Triton source
- our_v1: matched-output fused baseline with BH1 and 4 warps
- selected: conservative final choice: small V3 BH1/W4, medium V3 BH4/W4,
            large V2 BH2/W1, all with independent Krot output
- v3_alt: V3 BH2/W1 on large (optional, same output for comparison)
No cache-only variant because it changes output contract.
CUDA Graph latency only; no model throughput / PyTorch launch overhead.
"""
import argparse,csv,math
from statistics import median
from pathlib import Path
import torch
import triton.testing
from compare_sglang_fused import prepare,run_sglang
from compare_official import launch_upstream
from kernels import torch_reference
from matched_kernels import launch as launch_v1
from head_tiling import launch as launch_v2
from half_split_rope import launch as launch_v3

ROOT=Path(__file__).resolve().parent
CASES=[
    ('small_gqa',1,32,8,128,'v3',1,4),
    ('medium_gqa',64,32,8,128,'v3',4,4),
    ('large_gqa',256,64,8,128,'v2',2,1)
]

def run_case(name,T,Hq,Hkv,D,family,bh,w,rounds,rep):
    torch.manual_seed(10000+T)
    data=prepare(T,Hq,Hkv,D,torch.float16)
    q,k,v,c,s,cs,pos,slots,kc,vc,qo,ko=data
    ktmp=torch.empty_like(k)
    refq,refkc=torch_reference(q,k,c,s,slots,kc,D)
    refk,_=torch_reference(k,k,c,s,slots,kc,D)
    refvc=vc.clone()
    valid=slots>=0
    refvc.view(-1,Hkv,D)[slots[valid].long()]=v[valid]

    def selected():
        if family=='v2':
            return launch_v2(q,k,v,c,s,slots,qo,ko,kc,vc,bh=bh,num_warps=w)
        return launch_v3(q,k,v,c,s,slots,qo,ko,kc,vc,bh=bh,num_warps=w)

    fns={
        'flaggems_2k':lambda: launch_upstream(q,k,v,c,s,slots,qo,ko,kc,vc),
        'our_2k':lambda:launch_v1(q,k,c,s,slots,kc,qo,ktmp,D,False,4,
                                 v=v,v_cache=vc,k_out=ko),
        'sglang_fused':lambda:run_sglang(q,k,v,cs,pos,slots,kc,vc,qo,ko),
        'our_v1_bh1_w4':lambda:launch_v1(q,k,c,s,slots,kc,qo,None,D,True,4,
                                     v=v,v_cache=vc,k_out=ko),
        'selected':selected,
    }
    if name=='large_gqa':
        fns['v3_alt_bh2_w1']=lambda:launch_v3(q,k,v,c,s,slots,qo,ko,kc,vc,
                                                bh=2,num_warps=1)
    for label,func in fns.items():
        qcopy=q.clone();kcopy=k.clone()
        # Check overwrites and mathematical contract for each implementation.
        qo.fill_(999);ko.fill_(999);kc.fill_(-7);vc.fill_(-7)
        func()
        torch.cuda.synchronize()
        for attr,actual,expected in [
            ('Qrot',qo,refq),('Krot',ko,refk),('Kcache',kc,refkc),('Vcache',vc,refvc)]:
            try:
                torch.testing.assert_close(actual,expected,atol=.006,rtol=.006)
            except AssertionError as ex:
                raise AssertionError(f'{name} {label} {attr} mismatch: {ex}') from ex
        assert torch.equal(q,qcopy) and torch.equal(k,kcopy), 'Input modified unexpectedly'
        print('PASS',name,label,'4 outputs',flush=True)

    # Warm up with the same workload and preallocated output buffers.
    for f in fns.values():
        for _ in range(4):f()
    torch.cuda.synchronize()
    names=list(fns)
    samples={name:[] for name in names}
    for roundno in range(rounds):
        rotated=names[roundno%len(names):]+names[:roundno%len(names)]
        if roundno%2:rotated=rotated[::-1]
        for label in rotated:
            us=triton.testing.do_bench_cudagraph(fns[label],rep=rep)*1000
            samples[label].append(us)
        print('ROUND',name,roundno+1,'/',rounds,'; '+'; '.join(
            f"{x}={samples[x][-1]:.4f}" for x in names),flush=True)
    results={k:median(v) for k,v in samples.items()}
    chosen=results['selected']
    records=[]
    for label in names:
        row=dict(shape=name,T=T,QH=Hq,KVH=Hkv,D=D,dtype='float16',
                 selected_family=family,selected_BH=bh,selected_warps=w,
                 baseline=label,latency_us=round(results[label],4),
                 vs_selected_speedup=round(results[label]/chosen,4),
                 trials_us=';'.join(f'{x:.4f}' for x in samples[label]),
                 correctness='Qrot/Krot/Kcache/Vcache PASS')
        records.append(row)
    print('SUMMARY',name,'selected',f'{chosen:.4f}',
        '2K_FlagGems_x',f'{results["flaggems_2k"]/chosen:.3f}',
        '2K_ours_x',f'{results["our_2k"]/chosen:.3f}',
        'SGLang_x',f'{results["sglang_fused"]/chosen:.3f}',flush=True)
    return records

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--rounds',type=int,default=7)
    ap.add_argument('--rep',type=int,default=25)
    ap.add_argument('--output',default='final_same_output_benchmark.csv')
    args=ap.parse_args()
    print('DEVICE',torch.cuda.get_device_name(0),'CUDA Graph replay',
          'rounds',args.rounds,'rep',args.rep,flush=True)
    results=[]
    for cfg in CASES:
        results.extend(run_case(*cfg,args.rounds,args.rep))
    p=ROOT/args.output
    with p.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader();writer.writerows(results)
    print('SAVED',p,'rows',len(results),flush=True)
    for label in ('flaggems_2k','our_2k','sglang_fused','our_v1_bh1_w4'):
        ratios=[r['vs_selected_speedup'] for r in results if r['baseline']==label]
        print('GEOMEAN',label,round(math.exp(sum(math.log(r) for r in ratios)/len(ratios)),3),
              'N',len(ratios),flush=True)

if __name__=='__main__':main()
