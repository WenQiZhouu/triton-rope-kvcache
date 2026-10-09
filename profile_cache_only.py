"""Profile paired V3 Kout-on / Kout-off with the same Q/K/V/cache inputs."""
import argparse
import torch
from bench_head_tiling import inputs
from kernels import torch_reference
from half_split_rope import launch as v3
from half_split_cache_only import launch as cache_only

CASES={'medium':(64,32,8,128,4,4),'large':(256,64,8,128,2,1)}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--shape',choices=CASES,required=True)
    ap.add_argument('--mode',choices=['with_kout','no_kout'],required=True)
    ap.add_argument('--iterations',type=int,default=100)
    args=ap.parse_args()
    T,Hq,Hkv,D,BH,W=CASES[args.shape]
    q,k,v,c,s,slots,kc,vc,qo,ko=inputs(T,Hq,Hkv,D,torch.float16,False)
    rv, rkcache=torch_reference(q,k,c,s,slots,kc,D)
    _,koutref=torch_reference(k,k,c,s,slots,kc,D)
    fun=lambda: (v3(q,k,v,c,s,slots,qo,ko,kc,vc,bh=BH,num_warps=W) if args.mode=='with_kout'
        else cache_only(q,k,v,c,s,slots,qo,ko,kc,vc,bh=BH,num_warps=W,write_k_out=False))
    fun()
    torch.cuda.synchronize()
    torch.testing.assert_close(qo,rv,atol=.006,rtol=.006)
    torch.testing.assert_close(kc,rkcache,atol=.006,rtol=.006)
    for _ in range(5):fun()
    torch.cuda.synchronize()
    for _ in range(args.iterations):
        torch.cuda.nvtx.range_push(f'{args.shape}_{args.mode}')
        fun()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    print('PROFILE_CACHE_ONLY_SUCCESS',args.shape,args.mode,args.iterations,flush=True)
if __name__=='__main__':main()
