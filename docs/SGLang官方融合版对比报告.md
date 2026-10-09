# SGLang Triton 融合 Kernel 对照基准

测试日期：2026-10-08。GPU：NVIDIA GeForce RTX 3060 12 GB（Windows）。PyTorch 2.4.0+cu121，Triton-Windows 3.3.1。计时：CUDA Graph replay。

## 1. 对照对象与来源

此前 2× 左右数据是与 FlagGems-vLLM **两个独立** RoPE 和 Cache Kernel 的比较，不足以表明自研融合 Kernel 优于已有官方融合版本。此次改用 **SGLang 官方的 fused_qk_rope_reshape_and_cache** 做一对一单 Kernel 对照。

- 源码： https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/kvcache/rope_cache.py
- 本地源码：official/sglang_rope_cache.py（获取于 2026-10-08，SHA256：a4f191b832b6f81617c5e35e20c4fb02f77a65b5c008a111df84e01c0900e832）
- 使用 Triton 3.3.1 **直接运行保存的未修改官方源码**；没有使用为了兼容 Triton 3.1 而删除 tl.assume 的版本。
- 自研匹配输出实现：matched_kernels.py；原始独立项目算子：kernels.py（未修改）。

## 2. 相同工作负载与公平性

两边都只启动 **一个 Triton GPU Kernel**，都完成：
- 对 Q、K 做 NeoX 风格 **全维 RoPE**；输出旋转后的 Q 与 K（K_out）。
- 旋转后的 K 写入 Paged K Cache；原始 V 写入 Paged V Cache。
- 同样的 Q/KV Heads、Token 数量、Head Dim、FP16/FP32、Flash-layout Cache。
- 禁用额外量化、滑动窗口映射、零张量输出等可选功能。
- 每个测试先与 PyTorch 数学参考计算比较 Q_out、K_out、K Cache、V Cache。
- 都用预分配 Buffer + CUDA Graph replay 测量，轮次交错以降低频率漂移的影响。

SGLang 的 RoPE 参数是打包的 cos_sin 和显式 position ID；本项目使用分离的 cos/sin。这是各自 API 的原生布局，**不是逐指令、逐访存严格相同**。SGLang 默认 num_warps=1，自研版本在这组比较固定 num_warps=4。由于资源分配与算法路径不同，这个微基准结果并不能代表部署整个 SGLang 服务后的改善。

## 3. 第一轮 FP16 结果（μs）

| Tokens | Q heads | KV heads | D | SGLang 官方融合 | 自研融合（4 Warps） | SGLang / 自研 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 2 | 64 | 2.324 | 1.604 | 1.45× |
| 1 | 8 | 2 | 128 | 2.583 | 1.642 | 1.57× |
| 4 | 8 | 2 | 128 | 2.576 | 1.643 | 1.57× |
| 16 | 16 | 4 | 128 | 3.654 | 1.827 | 2.00× |
| 64 | 16 | 4 | 128 | 7.447 | 2.886 | 2.58× |
| 128 | 16 | 4 | 128 | 14.921 | 4.693 | 3.18× |
| 256 | 32 | 8 | 128 | 41.270 | 25.013 | 1.65× |
| 512 | 32 | 8 | 64 | 49.986 | 38.201 | 1.31× |

**FP16 8 组几何平均：1.83×**。

## 4. 第二轮 FP16 复测

数据文件：comparison_sglang_fp16_recheck.csv。8 组几何平均：**1.82×**；具体每形状测试在 CSV 中，均为正加速。

FP32 八组输入第一轮几何平均为 **1.49×**，由于缺少同条件独立复测，未纳入主要性能汇总。

## 5. 结果范围与限制

测试观察：
- 有官方同功能融合 Kernel，SGLang 的实现并非两 Kernel 拼接。
- 自研输出匹配版本与官方融合版本数值一致（测试过 Q_rot、K_rot、K Cache、V Cache）。
- 在 **RTX 3060 这 8 组 FP16 微基准**里，自研融合版两轮几何平均性能约为 SGLang 融合版的 **1.82–1.83 倍**。
- 本组结果比较的是两个已融合实现之间的性能差异。

未覆盖的因素：
- 为什么自研版更快的具体指令级根因，尚未逐形状用 Nsight Compute 对照。
- 是否适用于其他 GPU、调度条件、后端和更多边界模式；SGLang 支持更多可选功能和 Cache Layout。
- 没有证明比 SGLang 最新所有生产路径更快，也没有验证大模型端到端 Tokens/s。
- **Token × Head 映射并非本实现独有。** SGLang 同样采用 token × head 粒度；该基准中的映射差异包含 1D flattened Program ID 与 2D Grid。
- 此 SGLang 官方源码为下载当日的 main 快照，以 SHA256 标识；没有核实该文件的 Git 提交 SHA。今后测评可直接使用已保存的源码保证重现。

## 6. 数据与复现

在安装 CUDA PyTorch 与 Triton-Windows 的环境中，从仓库根目录运行：

    python compare_sglang_fused.py
    python recheck_sglang_fp16.py

原始结果：[第一轮](../results/comparison_sglang_fused.csv)、[FP16 复测](../results/comparison_sglang_fp16_recheck.csv)。

这组八 Shape 测试来自早期匹配四路输出的 V1，不等同于后续 V2/V3 三组 GQA 的结果。该对照基于保存的 SGLang Triton 融合源码，不代表 NVIDIA 默认生产路径或整个 SGLang 服务的吞吐。第三方源码及许可证见 [NOTICE](../NOTICE.md)。
