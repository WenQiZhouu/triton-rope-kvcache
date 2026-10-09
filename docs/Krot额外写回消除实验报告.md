# Cache-only 输出契约消融

2026-10-09，RTX 3060 12GB，Triton-Windows 3.3.1，FP16，全维 NeoX RoPE。

## 目标与输出合同

原 V3 计算 Q/K RoPE 后，同时输出 Q_rot、**K_rot**、Paged K Cache 和 Paged V Cache。

本次可选模式 WRITE_K_OUT=False，仅删除 Krot 独立数组写回：Qrot、K Cache、V Cache 保持不变。
所有有效缓存槽位必须能保存所需 K，且 **Attention 下游仅从 K Cache 获取 K** 才能安全执行此模式；
对于 slot=-1 的 Token，Krot 不会出现在 Cache 中，下游不能依赖它。原 SGLang fused_qk_rope_reshape_and_cache 函数明确返回 k_out，所以这是**改变输出契约的实验，不是与官方同输出公平加速对比**。

## 数据流与理论上限

原内核最低逻辑 I/O = 2*Q_bytes + 5*KV_bytes；去除 Krot 输出后 = 2*Q_bytes + 4*KV_bytes。

| Shape | Krot 独立写入量 | 原最低逻辑 IO | 新最低逻辑 IO | 理想带宽限制加速上限 |
|---|---:|---:|---:|---:|
| 中 T64/QH32/KVH8/D128 | 0.125 MiB | 1.625 MiB | 1.500 MiB | 1.083× |
| 大 T256/QH64/KVH8/D128 | 0.500 MiB | 10.500 MiB | 10.000 MiB | 1.050× |

这些是基于最低逻辑 I/O 且完全受带宽约束的模型值，不是通用性能上限。编译器会改变寄存器和指令，Cache L1/L2 与内存写回会改变实际 DRAM Bytes。

## Nsight Systems Kernel 延迟

| Shape | V3 保留 Kout | Cache-only 不写 Kout | 加速 |
|---|---:|---:|---:|
| 中 GQA，BH4/W4 | 2.720 μs | 2.656 μs | 1.024× |
| 大 GQA，BH2/W1 | 35.360 μs | 33.633 μs | 1.051× |

Nsight Systems 独立采样核时，不代表端到端模型延迟。

## Nsight Compute DRAM 计数

| 项目 | 大 GQA 保留 Kout | 大 GQA Cache-only |
|---|---:|---:|
| DRAM Read Bytes | 5,422,336 | 5,397,120 |
| DRAM Write Bytes | 5,667,840 | 4,330,752 |
| DRAM 写入差值 | — | **减少 1,337,088 Bytes** |
| DRAM 峰值吞吐利用率 | 84.14% | 84.40% |

**重要注意：** 计数器是一次 Nsight Compute 内核重放、--cache-control none 的测量；
实际 DRAM Write Bytes 的下降高于 Krot 物理大小 524288 Bytes，
可能包含 L2 缓存竞争、脏数据驱逐、异步写回或不同 Replay 缓存状态影响。
这一测量观察到了 DRAM 写入减少，但额外差值可能受 L2 驱逐与 Replay 缓存状态影响，不能全部归因于取消 K_rot Store。

## 测量波动

交错 CUDA Graph 9 轮初测存在明显 GPU 频率/缓存波动，尤其微秒级中 GQA 有反向结果；原始逐轮时间保存在 profiling/kout_optional_ablation.csv。以 Nsight Systems 106 次中位数为较稳健的复核证据，但仍应在不同负载、时间段重复。

## Cache 布局与后续验证

当前物理 Cache 形状 [num_blocks, block_size, Hkv, D]，地址 dst=slot*Hkv*D + head*D + feature。
每 Warp 对 feature 的访存通常是连续的；随机 slot 导致 Token 跨 Program 写入离散 Cache Page，但不等于 Warp 内天然不合并。不能简单把 Cache 改成其他排列就声称提升。

未验证的方向：

1. 核对真实推理调用路径：Attention 是否消费单独 Krot；是否所有所需 K 都有有效 slot；是否存在 SWA、fp8 量化、变长逻辑。只有符合时才开启 Cache-only。
2. 如果确定不需要 Krot，配置 WRITE_K_OUT=False 并同时优化后续 Attention API，保证没有隐藏的 Krot materialization。
3. 保持 Attention 读侧不变，在 Nsight Compute 中对比 Cache 写入的 sectors/request、L2 sector 数与 DRAM write bytes，核对是否因为 token slot 随机导致事务放大。
4. 仅当 Attention 读侧支持时测试 [block,head,slot,D] 等不同布局，写入的 Coalescing 可能更好或更差，但通常要与读侧联合 Benchmark。
5. 可尝试 K/V cache FP8 量化降低写入和后续 Attention 读取的带宽，但需额外 scale、精度损失与模型任务正确性验证。
6. 进一步减少 Qout 的独立写入必须将 Q RoPE 融入 Attention 消费端，属于更大融合范围，与当前独立 Kernel 的公平比较不同。

## 公开代码与结果

- [half_split_cache_only.py](../half_split_cache_only.py)：编译期开关控制独立 K_rot Store。
- [bench_cache_only_kout.py](../bench_cache_only_kout.py)：Cache-only 正确性与延迟测试。
- [profile_cache_only.py](../profile_cache_only.py)：硬件 Profile 入口。
- [CUDA Graph CSV](../profiling/kout_optional_ablation.csv)：交错测量记录。
- [NCU CSV 导出](../profiling/)：启用与禁用 K_rot Store 的计数器。
- [测试](../test_cache_only_kout.py)：输出契约和 Cache 写入正确性验证。

精简公开仓库未包含全部 Nsight Systems 二进制报告。由于 Cache-only 改变输出契约，其性能结果不计入 SGLang 同输出比较。
