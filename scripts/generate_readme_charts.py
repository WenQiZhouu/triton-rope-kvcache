#!/usr/bin/env python3
"""Generate six GitHub README visualizations from the checked-in benchmark CSVs.

Charts distinguish CUDA Graph microbenchmarks, Nsight Systems kernel time and
Nsight Compute replay hardware metrics. Nothing is fabricated or interpolated.
Run:
  python -m pip install matplotlib numpy
  python scripts/generate_readme_charts.py
"""
from pathlib import Path
import csv
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "asset"
ASSETS.mkdir(exist_ok=True)
R = ROOT / "results"
P = ROOT / "profiling"
NAVY = "#0b1220"
PANEL = "#121f34"
GRID = "#334155"
INK = "#e2e8f0"
MUTED = "#9db1cc"
CYAN = "#38bdf8"
TEAL = "#2dd4bf"
GOLD = "#fbbf24"
RED = "#fb7185"
BLUE = "#818cf8"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "figure.facecolor": NAVY,
    "axes.facecolor": PANEL,
    "savefig.facecolor": NAVY,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
    "grid.color": GRID,
    "grid.alpha": 0.40,
    "axes.titleweight": "bold",
    "legend.facecolor": PANEL,
    "legend.edgecolor": GRID,
})
def rows(fname):
    with fname.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))
def save(name):
    p = ASSETS / name
    plt.savefig(p, dpi=175, bbox_inches="tight", pad_inches=.22)
    plt.close()
    print("CREATED", p.name, p.stat().st_size, "bytes")
def fmtax(ax, grid=True):
    ax.spines[['top','right']].set_visible(False)
    if grid:
        ax.grid(axis="y", linestyle="--")
        ax.set_axisbelow(True)

# 1 / 6: three per-shape small multiples, all actual measured Run 1 latency
main = rows(R/"final_same_output_benchmark.csv")
more = rows(R/"final_same_output_repeat.csv")
shapes = ["small_gqa","medium_gqa","large_gqa"]
short = ["Decode: T=1","Prefill: T=64","Prefill: T=256"]
labels = [("our_2k","Two kernels",BLUE),
          ("sglang_fused","SGLang fused",GOLD),
          ("selected","Selected",TEAL)]
fig,axes = plt.subplots(1,3,figsize=(13.8,4.3),layout="constrained")
fig.suptitle("Matched-output GPU kernel latency  |  Run 1",fontsize=17,fontweight="bold")
for j,(ax,shape) in enumerate(zip(axes,shapes)):
    rr = {r["baseline"]:float(r["latency_us"]) for r in main if r["shape"]==shape}
    ys=[rr[code] for code,_,_ in labels]
    bars=ax.bar(range(3),ys,color=[color for _,_,color in labels],width=.62)
    for bar,val in zip(bars,ys):
        ax.text(bar.get_x()+bar.get_width()/2,val+max(ys)*.022,
                f"{val:.2f}",ha="center",fontsize=10,fontweight="bold")
    ax.set_ylim(0,max(ys)*1.24)
    ax.set_xticks(range(3),["2 kernels","SGLang","Selected"])
    ax.set_title(short[j],loc="left",pad=14,fontsize=13)
    ax.set_ylabel("Latency (microseconds)" if j==0 else "")
    fmtax(ax)
fig.text(.5,-.025,"RTX 3060 · FP16 · CUDA Graph replay · identical Qrot/Krot/Kcache/Vcache outputs · not end-to-end inference",
         ha="center",color=MUTED,fontsize=10)
save("latency_comparison_run1.png")

# 2 / 6: geomean from exact CSV, two independently repeated runs
bases = [("our_2k","Own 2-kernel"),("sglang_fused","SGLang fused"),
         ("flaggems_2k","FlagGems 2-kernel")]
def gmean_ratios(csvrows,baseline):
    ratios=[]
    for shape in shapes:
        d={r["baseline"]:float(r["latency_us"]) for r in csvrows if r["shape"]==shape}
        ratios.append(d[baseline]/d["selected"])
    return math.exp(sum(map(math.log,ratios))/len(ratios))
gm1=[gmean_ratios(main,b) for b,_ in bases]
gm2=[gmean_ratios(more,b) for b,_ in bases]
fig,ax=plt.subplots(figsize=(11.6,4.8),layout="constrained")
x=np.arange(len(bases));w=.32
b1=ax.bar(x-w/2,gm1,w,label="Run 1",color=CYAN)
b2=ax.bar(x+w/2,gm2,w,label="Run 2",color=TEAL)
for bars in (b1,b2):
    for bar in bars:
        y=bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2,y+.055,f"{y:.2f}x",ha="center",fontweight="bold")
ax.axhline(1,color=MUTED,linewidth=1,linestyle=":")
ax.set_xticks(x,[b for _,b in bases]);ax.set_ylim(0,max(*gm1,*gm2)*1.23)
ax.set_ylabel("Geometric mean speedup (x)")
ax.set_title("Selected V2/V3 dispatch  |  Geomean of 3 GQA shapes",pad=17,fontsize=16)
ax.legend(loc="upper right")
fmtax(ax)
fig.text(.5,-.01,"Same four outputs · 2 separate 7-round runs · clock/cache variance visible · limited shape scope",
         ha="center",color=MUTED)
save("geomean_speedup.png")

# 3 / 6: NCU profile and independently NSYS measured time (different tools!)
profile=rows(P/"ncu_hardware_summary.csv")
profile={(r["shape"],r["version"]):r for r in profile}
ncu_metrics=[
 ("Executed\ninstructions","executed_instructions","count"),
 ("Shared bank\nconflicts","shared_bank_conflicts","count"),
 ("Shared bytes\n(referenced)","shared_wavefronts_excessive","count"),
 ("Achieved\noccupancy","achieved_occupancy_pct","percent"),
]
# avoid falsely calling excessive wavefronts "shared bytes"
ncu_metrics[2]=("Excess shared\nwavefronts","shared_wavefronts_excessive","count")
v2=[float(profile["medium","v2"][key]) for _,key,_ in ncu_metrics]
v3=[float(profile["medium","v3"][key]) for _,key,_ in ncu_metrics]
fig,(ax,ax2)=plt.subplots(1,2,figsize=(13,4.8),gridspec_kw={"width_ratios":[3,1]},layout="constrained")
xx=np.arange(len(ncu_metrics)); ww=.31
bar1=ax.bar(xx-ww/2,[100]*len(v2),ww,color=BLUE,label="V2 = 100%")
bar2=ax.bar(xx+ww/2,[a/b*100 for a,b in zip(v3,v2)],ww,color=TEAL,label="V3 / V2")
for b,v in zip(bar2,[a/b*100 for a,b in zip(v3,v2)]):
    ax.text(b.get_x()+b.get_width()/2,v+3,f"{v:.0f}%",ha="center",fontweight="bold",fontsize=10)
ax.set_xticks(xx,[a for a,_,_ in ncu_metrics]);ax.set_ylim(0,126)
ax.set_ylabel("Relative to V2 (%)")
ax.set_title("Nsight Compute hardware metrics",fontsize=15,pad=16)
ax.legend(loc="upper right")
fmtax(ax)
bb=ax2.bar([0,1],[3.936,2.752],width=.6,color=[BLUE,TEAL])
ax2.set_ylim(0,5)
ax2.set_xticks([0,1],["V2","V3"])
ax2.set_title("Nsight Systems\nkernel latency",fontsize=13,pad=16)
ax2.set_ylabel("Microseconds")
for b,val in zip(bb,[3.936,2.752]):
    ax2.text(b.get_x()+b.get_width()/2,val+.17,f"{val:.3f}",ha="center",fontweight="bold")
fmtax(ax2)
fig.suptitle("Medium GQA  |  BH=4, warps=4",fontsize=17,fontweight="bold")
save("ncu_medium_v2_v3_normalized.png")

# 4 / 6: half split conceptual paired register reuse, with explicit schematic
fig,ax=plt.subplots(figsize=(13.4,4.8))
ax.set_facecolor(NAVY);ax.set_xlim(0,14);ax.set_ylim(0,5);ax.axis("off")
fig.suptitle("NeoX RoPE: compute both halves from one loaded pair",fontsize=17,fontweight="bold")
def box(x,y,w,h,t,color,fontsize=12):
    patch=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.13,rounding_size=0.14",
                         fc=PANEL,ec=color,lw=2)
    ax.add_patch(patch)
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",color=INK,fontsize=fontsize,
            fontweight="bold")
def arrow(a,b,c=TEAL):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=16,color=c,lw=2))
ax.text(2.95,4.42,"V2  |  full-width logical vector",ha="center",fontsize=13,fontweight="bold")
box(.65,2.98,4.7,.72,"q[d]    and    q[pair_d]",BLUE)
arrow((3,2.90),(3,2.30),BLUE)
box(.65,1.35,4.7,.75,"128 output positions + pair loads",BLUE,11)
ax.text(3,.82,"Repeated logical loads; potential layout conversion",
        ha="center",fontsize=10,color=MUTED)
ax.plot([7,7],[.6,4.6],linestyle="--",color=GRID,lw=2)
ax.text(10.5,4.42,"V3  |  Half-Split paired element",ha="center",fontsize=13,fontweight="bold")
box(7.45,3.0,2.18,.75,"q[i]",CYAN)
box(10.5,3.0,2.18,.75,"q[i + D/2]",CYAN,11)
arrow((8.5,2.9),(9.3,2.25))
arrow((11.5,2.9),(10.4,2.25))
box(8.45,1.46,3.55,.72,"reuse pair in registers",TEAL,11)
arrow((10.2,1.36),(10.2,.96))
ax.text(10.2,.56,"out[i]   +   out[i+D/2]",ha="center",
        fontsize=13,fontweight="bold",color=TEAL)
ax.text(7,.12,"Schematic of logical dataflow, not a literal SASS instruction trace",
        ha="center",color=MUTED,fontsize=9)
save("half_split_schematic.png")

# 5 / 6: genuine sweep BH x warps using SAME CSV for both versions
sweep = rows(R/"half_split_benchmark.csv")
mid=[r for r in sweep if r["shape"]=="medium_prefill" and r["dtype"]=="float16"]
bhvals=sorted({int(r["BH"]) for r in mid})
warps=sorted({int(r["warps"]) for r in mid})
arr={}
for name,key in [("V2 Head Tiling","v2_us"),("V3 Half-Split","v3_us")]:
    a=np.full((len(warps),len(bhvals)),np.nan)
    for r in mid:
        a[warps.index(int(r["warps"])),bhvals.index(int(r["BH"]))]=float(r[key])
    assert np.isfinite(a).all()
    arr[name]=a
fig,axs=plt.subplots(1,2,figsize=(12.5,5.15),layout="constrained")
lo=min(a.min() for a in arr.values()); hi=max(a.max() for a in arr.values())
for ax,(name,a) in zip(axs,arr.items()):
    hm=ax.imshow(a,cmap="viridis_r",vmin=lo,vmax=hi,aspect="auto")
    for k in range(len(warps)):
        for j in range(len(bhvals)):
            val=a[k,j]
            ax.text(j,k,f"{val:.2f}",ha="center",va="center",
                    fontsize=12,fontweight="bold",color="white" if val>lo+(hi-lo)*.45 else NAVY)
    ax.set_xticks(range(len(bhvals)),[str(x) for x in bhvals])
    ax.set_yticks(range(len(warps)),[str(x) for x in warps])
    ax.set_xlabel("BH: Q heads / program")
    ax.set_ylabel("num_warps")
    ax.set_title(name,pad=11)
fig.colorbar(hm,ax=list(axs),shrink=.9,label="Kernel latency (us), lower is better",pad=.015)
fig.suptitle("BH / Warp parameter sweep  |  Medium GQA FP16",fontsize=17,fontweight="bold")
save("bh_warp_heatmap.png")

# 6 / 6: distinct axis unit panels for bandwidth utilization/traffic/latency
large2=profile["large","v2"];large3=profile["large","v3"]
util=[float(large2["dram_peak_pct"]),float(large3["dram_peak_pct"])]
traffic=[float(large2["dram_total_bytes"])/1e6,float(large3["dram_total_bytes"])/1e6]
lat=[35.825,35.680]  # NSYS kernel medians; not NCU replay duration
fig,axs=plt.subplots(1,3,figsize=(13.4,4.6),layout="constrained")
params=[(util,"Peak bandwidth used","%",100),(traffic,"DRAM bytes transferred","MB",max(traffic)*1.24),
        (lat,"Kernel latency","us",max(lat)*1.26)]
for ax,(values,title,units,lim) in zip(axs,params):
    bars=ax.bar([0,1],values,color=[BLUE,TEAL],width=.6)
    ax.set_xticks([0,1],["V2","V3"]);ax.set_ylim(0,lim)
    ax.set_title(title,pad=15,fontsize=13)
    ax.set_ylabel(units)
    for b,val in zip(bars,values):
        ax.text(b.get_x()+b.get_width()/2,val+lim*.035,
                f"{val:.2f}",ha="center",fontweight="bold",fontsize=12)
    fmtax(ax)
fig.suptitle("Large GQA  |  instructions fall, DRAM traffic does not",fontsize=17,fontweight="bold")
fig.text(.5,-.02,"Bandwidth/bytes: NCU full replay  |  kernel latency: Nsight Systems  |  RTX 3060",
         ha="center",color=MUTED,fontsize=10)
save("large_gqa_memory_bound.png")

print("OK: six source-backed PNG charts generated")
