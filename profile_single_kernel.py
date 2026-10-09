"""A single reproducible V2 or V3 fused kernel workload for NCU/Nsight Systems.

Use NCU --kernel-name ... --launch-skip 5 --launch-count 1
or nsys profile --trace=cuda,nvtx -o profiling/file ...
"""
import argparse
from pathlib import Path
import torch
from bench_head_tiling import inputs
from head_tiling import launch as launch_v2
from half_split_rope import launch as launch_v3
from kernels import torch_reference

SHAPES={
    'medium':(64,32,8,128,4,4),
    'large':(256,64,8,128,2,1),
}
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--shape',choices=SHAPES,required=True)
    p.add_argument('--version',choices=['v2','v3'],required=True)
    p.add_argument('--iterations',type=int,default=10)
    p.add_argument('--dump-code',action='store_true')
    p.add_argument('--validate',action='store_true')
    a=p.parse_args()
    T,Hq,Hkv,D,bh,w=SHAPES[a.shape]
    data=inputs(T,Hq,Hkv,D,torch.float16,False)
    q,k,v,c,s,slots,kc,vc,qo,ko=data
    ref=None
    if a.validate:
        ref_q,ref_kc=torch_reference(q,k,c,s,slots,kc,D)
        ref_k,_=torch_reference(k,k,c,s,slots,kc,D)
        ref_v=vc.clone()
        ref_v.view(-1,Hkv,D)[slots.long()]=v
        ref=(ref_q,ref_k,ref_kc,ref_v)
    launcher=launch_v2 if a.version=='v2' else launch_v3
    def step():
        return launcher(q,k,v,c,s,slots,qo,ko,kc,vc,bh=bh,num_warps=w,return_compiled=True)
    compiled=step()
    torch.cuda.synchronize()
    if ref is not None:
        for x,y in zip((qo,ko,kc,vc),ref):
            torch.testing.assert_close(x,y,atol=6e-3,rtol=6e-3)
    if a.dump_code:
        folder=ROOT/'profiling'
        folder.mkdir(exist_ok=True)
        name=f'{a.shape}_{a.version}_BH{bh}_W{w}'
        (folder/(name+'.ptx')).write_text(compiled.asm['ptx'],encoding='utf-8')
        (folder/(name+'.cubin')).write_bytes(compiled.asm['cubin'])
        print('CODE',name,'REGS',compiled.n_regs,'SMEM',compiled.metadata.shared,
              'PTX',len(compiled.asm['ptx'].splitlines()),flush=True)
    for i in range(5):
        step()
    torch.cuda.synchronize()
    for i in range(a.iterations):
        torch.cuda.nvtx.range_push(f'{a.shape}_{a.version}_BH{bh}_W{w}_rep{i}')
        step()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print('PROFILE_WORKLOAD_SUCCESS',a.shape,a.version,'iter',a.iterations,flush=True)

if __name__=='__main__':
    main()
