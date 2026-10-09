# V2 / V3 Nsight Compute 硬件计数器分析

2026-10-09 · NVIDIA RTX 3060 12GB / Ampere SM86 · PyTorch 2.4.0+cu121 · Triton-Windows 3.3.1

## 结论

**V3 对中 GQA 的显著加速（约 1.43×）来自更少的执行指令、Shared Memory 通信、Barrier 和 Bank Conflict，而非 Occupancy 提高，也非显著减少实际 DRAM Bytes。**

**大 GQA 的 V3 也消除了大量指令/Bank Conflict，但 DRAM 流量并未下降；实测 DRAM 吞吐达到硬件峰值约 83–85%，所以整体速度几乎不变。**

用 Nsight Systems 看实际执行耗时；用 Nsight Compute 看硬件微架构计数器。Nsight Compute 内核 Replay 40 passes、cache-control=none，会改变设备缓存/时钟环境，报告内的 NCU Duration 不等于正常的 CUDA Graph 或 Nsight Systems 延迟，不拿它计算优化加速比。

## 采集方法

- Nsight Systems 2024.6.2：记录 CUDA Kernel 时间线；每个对照配置采集约 106 次 Kernel Launch，使用中位数。
- Nsight Compute 2025.1.1：以 --set full 和 --cache-control none 采集 DRAM、Shared Memory 和 Warp 计数器，每组约 40 个 Replay Pass。
- CUDA 12.8 nvdisasm：对 SM86 编译产物反汇编，统计静态 SASS 指令位置。

NCU 多次重放可能改变 Cache 状态，因此 NCU Replay Duration 与正常 Kernel 时间不能直接比较。

## Nsight Systems 的正常 Kernel 执行时间

| 指标 | 中 GQA V2 | 中 GQA V3 | 大 GQA V2 | 大 GQA V3 |
|---|---:|---:|---:|---:|
| Tokens / QH / KVH | 64 / 32 / 8 | 64 / 32 / 8 | 256 / 64 / 8 | 256 / 64 / 8 |
| BH / Warps | 4 / 4 | 4 / 4 | 2 / 1 | 2 / 1 |
| Nsight Systems Median (μs) | 3.936 | 2.752 | 35.825 | 35.680 |

中 GQA 速度提升 1.430×；
大 GQA 仅有 1.004×，相当于没有显著收益。

## Nsight Compute 硬件指标对照（--set full, --cache-control none）

| 指标 | 中 GQA V2 | 中 GQA V3 | 大 GQA V2 | 大 GQA V3 |
|---|---:|---:|---:|---:|
| DRAM Read MB | 0.820 | 0.825 | 5.420 | 5.570 |
| DRAM Write MB | 0.692 | 0.742 | 5.644 | 5.687 |
| DRAM Total MB | 1.512 | 1.567 | 11.064 | 11.256 |
| DRAM Throughput (GB/s) | 206.34 | 220.61 | 289.81 | 297.85 |
| DRAM peak utilization (%) | 59.35 | 63.77 | 82.86 | 85.24 |
| L2 Hit Rate (%) | 99.19 | 58.27 | 64.11 | 62.17 |
| L1/TEX Hit Rate (%) | 45.28 | 29.70 | 26.36 | 4.96 |
| Executed Instructions | 242,688 | 127,488 | 1,265,664 | 656,384 |
| Shared Bank Conflicts | 10,407 | 0 | 218,155 | 4,161 |
| Extra Shared Wavefronts | 10,240 | 0 | 204,800 | 0 |
| Achieved Occupancy (%) | 84.40 | 70.60 | 30.45 | 29.88 |
| Theoretical Occupancy (%) | 100.00 | 100.00 | 33.33 | 33.33 |
| Eligible Warps / Scheduler | 1.11 | 0.53 | 0.29 | 0.15 |
| No Eligible (%) | 58.95 | 80.96 | 76.18 | 86.89 |
| NCU Replay Duration (μs；非正常执行耗时) | 7.33 | 7.10 | 38.18 | 37.79 |

### 计数器观察

- 中 GQA：V2→V3 的执行指令从 242,688 降到 127,488，减少 **47.5%**。
Bank Conflict 从 10,407 降为 0；但是 DRAM Total Bytes **增加 3.6%**，说明本次加速不是因为减少了 off-chip 流量。

- 大 GQA：执行指令减少 **48.1%**；
Bank Conflict 从 218,155 降到 4,161（减少 **98.1%**）。
DRAM Total Bytes 反而增加 **1.7%**。

- 大 GQA 的 DRAM Throughput 82.86%→85.24% 峰值，表明显存通道已经占用较高；省下的指令和 Shared 通信并不能显著减少必须发生的 Q/K/V 和 Cache 数据搬运。

### 为什么 V3 L2 Hit Rate 中规模反而更低？

- L2 命中率是对某一类请求的**命中次数比例**，V3 改变了访问/复用方式；命中率低不等同于 DRAM Bytes 高。
- Kernel Replay 运行不同硬件计数器时可能多次重放，--cache-control none 没有强制每轮缓存相同。Nsight 明确警告 uncontrolled GPU caches。
- 对比本次实际 DRAM Read Bytes，中规模 V2≈0.820 MB、V3≈0.825 MB，实际外部读量基本持平。
- 因此不从 L2 Hit Rate 99.19% vs 58.27% 单独推断缓存“更差”或“两者有 40% DRAM 流量差”；要把完整事务量放在一起看。

### Warp Stall 与 Occupancy

- 中 GQA V2 的理论 Occupancy 100%、实际 84.4%；V3 的理论 Occupancy 仍 100%、实际反而只有 70.6%，**并未通过提高 Occupancy 加速**。
- 大 GQA 的 Warp 数为 1/Program；硬件每 SM 最多可驻留的 CTA 数限制理论上只有 16 个 Warp/SM，占 Ampere 最大 48 Warp/SM 的 33.3%。V2/V3 都一样，降低寄存器无法解除这一 CTA 数限制。
- 大 GQA V3 的 Eligible Warps/Scheduler=0.15，No Eligible 达 86.89%，相较 V2 的 0.29、76.18% 更容易受剩余访存相关指令依赖影响。
- Long Scoreboard Stall per issue-active 是**每发射指令归一化的等待指标**；V3 总指令减少约一半，不能仅因归一化 stall 值更大就推断总等待时间变长。

## SASS 静态指令 vs NCU 动态统计（互相验证）

| SM86 SASS | 中 GQA V2 | 中 GQA V3 | 大 GQA V2 | 大 GQA V3 |
|---|---:|---:|---:|---:|
| Total SASS sites | 184 | 112 | 264 | 152 |
| LDG sites | 14 | 8 | 22 | 8 |
| LDS sites | 4 | 0 | 6 | 2 |
| STS sites | 10 | 0 | 18 | 2 |
| BAR sites | 7 | 0 | 7 | 3 |
| SHFL sites | 0 | 0 | 0 | 0 |

**解释：** 中 GQA 的 V3 将 LDS/STS/BAR 位点彻底去除，NCU 的共享内存 Bank Conflict 也归零。
V2→V3 两者的 SASS 中均没有 SHFL 位点，该计数不支持将收益归因于 Warp Shuffle 指令减少。
大 GQA 仍然有少量 Shared 指令，但冲突数显著减少。SASS 静态计数与 NCU 动态计数不是同一种单位。

## L2 容量和尺寸扫描的作用（辅助因果验证）

- RTX 3060 的 CUDA Driver API 报告 L2=2.25 MiB，28 SM，显存总线 192bit，显存时钟换算理论带宽≈360 GB/s。
- 固定 QH32/KVH8/BH4/W4，只增加 Tokens，V3 相对 V2 在 T64–80 约 1.39–1.41×，T128 仅 1.02–1.03×。
- 固定 QH64/KVH8/BH2/W1，只增加 Tokens，T32–48 约 1.30×，T96 仅 1.01×。
- 容量阈值与最低逻辑 I/O 工作集越过 L2 容量高度相关，支持缓存和带宽压力转变；
**但 L2 缓存不是严格整体可用容量判据，动态命中率也不简单按工作集大小单调变化**。
- NCU 大 GQA 实测 DRAM 带宽 290–298GB/s、峰值利用率 83–85%，让“大规模数据搬运瓶颈”的判断获得了直接硬件计数器支撑。

## 解释范围

固定 BH4/W4 的中 GQA 下，V3 减少动态指令、Shared Memory 访问与同步，Bank Conflict 在 NCU 计数中归零。实际 DRAM Bytes 并未降低，表明该组改进主要发生在片上计算和数据布局转换路径。仍不能将性能变化归因于单一硬件指标。

大 GQA 下也观察到指令数与 Shared Bank Conflict 下降，但带宽利用率达到持续峰值约 83–85%，实际外部流量未减少，Kernel 时间基本持平。尺寸扫描与 L2 容量相关，但缓存容量不能作为所有 Shape 的唯一判断条件。

## 数据文件

- [NCU 指标摘要 CSV](../profiling/ncu_hardware_summary.csv)、[JSON](../profiling/ncu_hardware_summary.json)。
- [profiling/](../profiling/) 下的四份 raw.csv、details.csv 为 NCU 导出计数器。
- [collect_ncu_hardware_counters.py](../collect_ncu_hardware_counters.py) 为计数器提取脚本。
- [profile_single_kernel.py](../profile_single_kernel.py) 为单配置测量入口。

精简版公开仓库不包含全部原始二进制 .ncu-rep / .nsys-rep、CUBIN 和 SASS 文件；硬件性能计数器采集需要对应权限。
