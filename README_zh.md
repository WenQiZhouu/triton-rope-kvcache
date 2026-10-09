# Triton 融合 RoPE + Paged KV Cache 算子优化

独立 AI Infra 工程项目。实现 Q/K NeoX RoPE 与 GQA 分页 KV Cache 更新，研究 Kernel Fusion、Head Tiling、Half-Split 寄存器复用与硬件性能归因。

**最新同输出测试，RTX 3060 / FP16 / 三组 GQA / CUDA Graph：** 当前按形状分派的版本，相比自研两个独立 Kernel，两轮几何平均 **1.57–1.64x**；相对 SGLang 官方 Triton 融合算子快照 **2.13–2.50x**。这是独立算子微基准，**不是**整个 SGLang 默认 CUDA 路径或大模型端到端吞吐。

技术路线：V1 把 RoPE 与 K/V Cache Scatter 融合，避免中间 K_rot 写后再读；V2 一次处理多个 Q Head；V3 使用前后半区配对 Half-Split，减少重复 Load 与 Shared Memory 数据交换；大 GQA 则根据 DRAM 带宽瓶颈保留 V2。

关键硬件证据：中 GQA V2->V3 Kernel 3.936->2.752 us（1.43x），动态指令 242,688->127,488，Bank Conflict 10,407->0；大 GQA DRAM 带宽利用率约 83–85%，V3 改善不明显。

项目有四路同输出 Q_rot、K_rot、K Cache、V Cache 真实性测试；Cache-only 不返回独立 K_rot 的版本改变了输出接口，**不参与上述同输出性能对比**。

## 可视化：从算子加速到硬件证据

六张图均来自真实保存的 Benchmark / NCU CSV；测试属于 **RTX 3060 的独立 CUDA Graph 算子级微基准**，不代表整个 SGLang 框架的端到端 Token/s。

### 1. 两 Kernel、SGLang 和当前最优版延迟

![三组 GQA 的同输出延迟对比](asset/latency_comparison_run1.png)

三个子图分别为 small、medium、large GQA 的**第一轮实测 Kernel 时间**。自研两 Kernel 基线是真正的两次 Kernel 调用；SGLang 官方 Triton 对照**本身已经融合**。三者均输出 Q_rot、K_rot、K Cache、V Cache，注意子图纵轴范围不同。

### 2. 两轮复测的几何平均加速比

![两轮独立几何平均加速比](asset/geomean_speedup.png)

仅统计明确测试的三组 FP16 GQA：相比自研两 Kernel 为 **1.57–1.64×**，相比保存的 SGLang 已融合 Triton 算子为 **2.13–2.50×**。两轮结果均展示，避免挑选单次最好成绩；微秒级延迟可能受 GPU 频率和缓存状态影响。

### 3. Half-Split：一次配对计算两个旋转输出

![Half-Split NeoX RoPE 数据流](asset/half_split_schematic.png)

NeoX RoPE 的旋转元素成对出现：x[i] 和 x[i+D/2]。V3 将一对元素映射到同一个 Triton 逻辑元素中，在寄存器里复用并计算两个输出，减少逻辑重复 Load。图中为**数据流示意**，不是逐 Lane 的 SASS 指令图，也不能据此声称 DRAM 字节数减半。

### 4. BH / Warp 调参热力图（原始实测）

![BH Warp 真实测试热力图](asset/bh_warp_heatmap.png)

热力图读取公开的 [Half-Split 原始 36 组数据](results/half_split_benchmark.csv)，比较 medium GQA 下 BH=1/2/4/8、num_warps=1/2/4 的 V2/V3 延迟（μs）。当前中规模选择的 V3 BH4/W4 来自**多轮重点复测和 Profile 证据**，不完全依赖带噪声的粗扫最小值。[原始 V2 Head Tiling 扫描](results/head_tiling_benchmark.csv) 也已经公开。

### 5. Nsight Compute：指令与 Bank Conflict

![V2 V3 中 GQA 的 NCU 对比](asset/ncu_medium_v2_v3_normalized.png)

同一中规模 BH4/W4：动态执行指令 **242,688→127,488（约 -47.5%）**，Shared Bank Conflict **10,407→0**；Nsight Systems Kernel 中位数 **3.936→2.752 μs（1.43×）**。**V3 的实际 Occupancy 反而下降**，说明收益主要来自数据流简化，而非 Occupancy 越高越快。NCU 与 Nsight Systems 分别采集，不能当成一次测量。

### 6. 大 GQA 为什么 V3 收益不明显？

![大 GQA 的 DRAM 带宽瓶颈图](asset/large_gqa_memory_bound.png)

NCU 测得 V2/V3 的 DRAM 带宽利用率为 **82.86%/85.24%**，**真实 DRAM 读写总量没有减少**；Nsight Systems 时间为 **35.825/35.680 μs**，几乎没有差异。片上计算减少不等于必须写回 Q_rot、K_rot 和 KV Cache 的数据减少，所以当前大 GQA 保留 V2。

### 复现所有图表

绘图只读取仓库中已保存的 CSV，**不需要 GPU**：

~~~bash
pip install -r requirements-viz.txt
python scripts/generate_readme_charts.py
~~~

图片位于 [asset/](asset/)，原始测试记录位于 [results/](results/)，Nsight 硬件数据位于 [profiling/](profiling/)。

---

## Windows + NVIDIA GPU 复现

1. Python 3.12，安装兼容 CUDA 的 PyTorch（验证环境：2.4.0+cu121）。
2. pip install -r requirements-windows.txt
3. python -m pytest -q test_kernels.py test_head_tiling.py test_half_split_dispatch.py test_cache_only_kout.py
4. python bench_final_same_output.py --rounds 7 --rep 25 --output my_results.csv

两轮实测数据见 [results/](results/)；V2/V3 的硬件指标分析见 [Nsight Compute 报告](docs/V2_V3_NCU_硬件计数器验证报告.md)。
第三方 SGLang 和 FlagGems 对照源码来自公开项目，许可证和出处见 NOTICE.md。
