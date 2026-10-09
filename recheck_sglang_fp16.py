"""Robustness check against SGLang fused RoPE+K/V Cache: FP16, 8 shapes."""
import csv
from pathlib import Path
from statistics import median, geometric_mean
import torch
from compare_sglang_fused import CASES,ROOT,prepare,check,run_sglang,bench,our_launch

rows=[]
for T,Hq,Hkv,D in CASES:
    q,k,v,c,s,cs,pos,slots,kc,vc,qo,ko=prepare(T,Hq,Hkv,D,torch.float16)
    check((q,k,v,c,s,cs,pos,slots,kc,vc,qo,ko),torch.float16)
    official=lambda:run_sglang(q,k,v,cs,pos,slots,kc,vc,qo,ko)
    ours=lambda:our_launch(q,k,c,s,slots,kc,qo,None,D,True,4,v=v,v_cache=vc,k_out=ko)
    results=[]
    for i in range(3):
        if i%2:
            t_ours=bench(ours,1,45)
            t_official=bench(official,1,45)
        else:
            t_official=bench(official,1,45)
            t_ours=bench(ours,1,45)
        results.append((t_official,t_ours))
    u=median(x[0] for x in results)
    f=median(x[1] for x in results)
    row=dict(tokens=T,q_heads=Hq,kv_heads=Hkv,head_dim=D,
             sglang_fused_us=round(u,3),ours_fused_4warps_us=round(f,3),
             speedup=round(u/f,3))
    rows.append(row)
    print(row,flush=True)

out=ROOT/'comparison_sglang_fp16_recheck.csv'
with out.open('w',newline='',encoding='utf-8') as w:
    dw=csv.DictWriter(w,fieldnames=rows[0].keys());dw.writeheader();dw.writerows(rows)
print('GMEAN',round(geometric_mean(float(r['speedup']) for r in rows),3),flush=True)
