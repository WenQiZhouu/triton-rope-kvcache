"""Extract Nsight Compute hardware counter tables (full-set cache-control=none).

Raw CSV rows: first header, second units, third values. DRAM bytes are true
hardware counters (contrary to prior modeled logical I/O bytes). Report
profiler replay duration separately from Nsight Systems GPU kernel timeline.
"""
import csv,json,re
from pathlib import Path
P=Path(__file__).resolve().parent/'profiling'
METRICS={
    'dram_read_bytes':'dram__bytes_read.sum',
    'dram_write_bytes':'dram__bytes_write.sum',
    'l2_sectors_tex':'lts__t_sectors_srcunit_tex.sum',
    'l2_sectors_all':'lts__t_sectors.sum',
    'shared_bank_conflicts':'l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum',
    'shared_bank_conflicts_ld':'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
    'shared_bank_conflicts_st':'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
    'shared_wavefronts_excessive':'derived__memory_l1_wavefronts_shared_excessive',
    'warps_eligible_per_sched':'smsp__warps_eligible.avg.per_cycle_active',
    'long_scoreboard_stall_per_issue':'smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio',
    'barrier_stall_per_issue':'smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio',
    'mio_throttle_stall_per_issue':'smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio',
    'lg_throttle_stall_per_issue':'smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio',
    'pcsamp_long_scoreboard_not_issued':'smsp__pcsamp_warps_issue_stalled_long_scoreboard_not_issued',
}
HUMAN={
    'duration_us':('GPU Speed Of Light Throughput','Duration'),
    'dram_peak_pct':('GPU Speed Of Light Throughput','DRAM Throughput'),
    'memory_throughput_gbps':('Memory Workload Analysis','Memory Throughput'),
    'l2_hit_pct':('Memory Workload Analysis','L2 Hit Rate'),
    'l1_hit_pct':('Memory Workload Analysis','L1/TEX Hit Rate'),
    'achieved_occupancy_pct':('Occupancy','Achieved Occupancy'),
    'theoretical_occupancy_pct':('Occupancy','Theoretical Occupancy'),
    'eligible_per_scheduler':('Scheduler Statistics','Eligible Warps Per Scheduler'),
    'no_eligible_pct':('Scheduler Statistics','No Eligible'),
    'active_per_scheduler':('Scheduler Statistics','Active Warps Per Scheduler'),
    'executed_instructions':('Instruction Statistics','Executed Instructions'),
    'issued_instructions':('Instruction Statistics','Issued Instructions'),
    'branch_efficiency_pct':('Source Counters','Branch Efficiency'),
}
PREFIX={'Kbyte':1000,'Mbyte':1000000,'Gbyte':1000000000,'byte':1}
def parsefloat(x):
    if x is None or x=='':return None
    try:return float(x.replace(',',''))
    except ValueError:return None
def one(shape,version):
    stem=P/f'{shape}_{version}_ncu'
    rows=list(csv.DictReader((stem.with_suffix('.raw.csv')).open(encoding='utf-8-sig',newline='')))
    assert len(rows)>=2
    units,values=rows[0],rows[-1]
    out={'shape':shape,'version':version,'cache_mode':'none','ncu_set':'full'}
    for key,ref in METRICS.items():
        out[key]=parsefloat(values.get(ref))
        if key.endswith('bytes'):
            unit=units.get(ref)
            out[key]=out[key]*PREFIX[unit] if out[key] is not None else None
    details=list(csv.DictReader((stem.with_suffix('.details.csv')).open(encoding='utf-8-sig',newline='')))
    data={(r['Section Name'],r['Metric Name']):r['Metric Value'] for r in details}
    for key,ref in HUMAN.items():
        out[key]=parsefloat(data.get(ref))
    out['dram_total_bytes']=(out['dram_read_bytes'] or 0)+(out['dram_write_bytes'] or 0)
    return out
def main():
    rows=[one(s,v) for s in ('medium','large') for v in ('v2','v3')]
    path=P/'ncu_hardware_summary.csv'
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    (P/'ncu_hardware_summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in rows:
        print('NCU_HARDWARE',r,flush=True)
    print('WROTE',path,flush=True)
if __name__=='__main__':
    main()
