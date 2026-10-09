"""CUDA Graph replay GPU latency: compares split/fused Triton with identical protocol.

Running CUDA Graph helps reduce Python launch noise; these values must never
be mixed with non-graph do_bench numbers.
"""
import csv
from pathlib import Path
from statistics import median
import torch
import triton
from kernels import launch
from test_kernels import make_inputs


CASES=[
    (1,8,2,128,128),
    (4,8,2,128,128),
    (16,16,4,128,128),
    (64,16,4,128,128),
    (256,32,8,128,128),
    (512,32,8,64,64),
    (128,16,4,96,64),
]


def bench(fn):
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        result = median(triton.testing.do_bench_cudagraph(fn,rep=40) for _ in range(3))
    torch.cuda.synchronize()
    return result


def main():
    print("GPU:",torch.cuda.get_device_name(0), "mode: cudagraph",flush=True)
    records=[]
    for dtype in (torch.float16,torch.float32):
        for T,hq,hkv,d,rotary in CASES:
            q,k,c,s,slots,cache=make_inputs(T,hq,hkv,d,rotary,dtype)
            out=torch.empty_like(q)
            tmp=torch.empty_like(k)
            split=bench(lambda: launch(q,k,c,s,slots,cache,out,tmp,rotary,fused=False,num_warps=4))
            timings={}
            for warps in (1,2,4,8):
                timings[warps]=bench(lambda w=warps: launch(q,k,c,s,slots,cache,out,tmp,rotary,fused=True,num_warps=w))
            best=min(timings,key=timings.get)
            record={"dtype":str(dtype).split(".")[-1],"tokens":T,"q_heads":hq,
                    "kv_heads":hkv,"head_dim":d,"rotary_dim":rotary,
                    "split_us":round(split*1000,3),
                    "fused_warps4_us":round(timings[4]*1000,3),
                    "best_warps":best,
                    "fused_best_us":round(timings[best]*1000,3),
                    "fusion_speedup":round(split/timings[4],3),
                    "tuned_speedup":round(split/timings[best],3),
                    "tuning_gain":round(timings[4]/timings[best],3)}
            records.append(record)
            print(record,flush=True)
    with Path("benchmark_cudagraph.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print("WROTE benchmark_cudagraph.csv",flush=True)

if __name__=="__main__":
    main()
