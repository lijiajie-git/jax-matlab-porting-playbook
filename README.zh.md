# MATLAB → JAX → TPU 移植知识库

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![JAX](https://img.shields.io/badge/JAX-0.4+-blue.svg)](https://jax.readthedocs.io/)
[![TPU](https://img.shields.io/badge/TPU-v5e-orange.svg)](https://cloud.google.com/tpu)

把 MATLAB 写的时间步进式仿真（agent-based / compartmental / microsim）移植到
JAX、并在 Google Cloud TPU 多主机集群上跑起来的实战手册。

提炼自一个真实端口：26.3 万 agent 的 mpox 传染病微观模型 —— 85 周 × 26 个干预
情景，从单线程 MATLAB 移植到 TPU v5e-64（16 个 host × 4 个 chip = 64 chip
多主机集群），单 chip-sample 约 52 毫秒，Monte Carlo 轴近线性 scaling。

四份文档结构上为编码助手 LLM 的持久化 context 设计（Claude Code、Cursor 等），
直接给人看也是流畅的散文。

---

## 内容

| 文件 | 主题 | 大约行数 |
|---|---|---|
| **`jax-patterns.md`** | MATLAB 习惯翻译成 JAX 原语 ——`find()` mask、函数式更新、静态 shape 抽样、transition 表、`lax.scan`、`vmap`、PRNG key | 640 |
| **`tpu-ops.md`** | 多主机 TPU 部署 ——`gcloud` 硬规则、`jax.distributed.initialize` 调用顺序、持久化编译缓存、`shard_map`、per-chip batch 调参、槽位规划 | 485 |
| **`validation.md`** | MATLAB ↔ JAX 对账方法 —— 三层对齐（cell / 分布 / 生产）、replay bundle bit-match、z-score 排序、exact-vs-correct 双路径、bisect 工作流 | 520 |

---

## 内文具体例子

**MATLAB 的 `find(mask)` 返回变长数组；JAX 下要求静态 shape。**

```python
# MATLAB
ids = find(alive==1 & pox_status==2);
state(ids, TREATMENT) = 1;        # 变长 scatter

# JAX
mask = (alive == 1) & (pox_status == 2)         # 永远 shape (N,)
treatment = jnp.where(mask, 1, treatment)       # 静态，可 jit
```

**新生儿要追加行，但 jit 下 shape 不能动态增长。**

用「cumsum 当排名」的技巧：把空槽标 `1`，前缀求和，每个空槽得到一个唯一序号
`1, 2, 3, …`，再用 `rank ≤ k` 过滤选前 `k` 个。无 sort、无 scatter，纯
稠密运算。

**两边 seed 都给 42，MATLAB 和 JAX 的随机序列依然不同。怎么 bit-match？**

不要试图共享 seed。把 MATLAB 的「实际决策」（每周哪些 agent id 被选中）dump
出来，通过 replay bundle 直接注入 JAX 状态更新流程，**完全跳过 JAX 自己的
RNG**。一旦注入决策能复现 MATLAB 的状态演化，整个确定性管线的逻辑就被验证了。

---

## 决策树（遇到问题查哪份）

| 症状 / 问题 | 读哪份 | 章节 |
|---|---|---|
| MATLAB 里 `find()` 返回变长索引，JAX 怎么写 | `jax-patterns.md` | §2 布尔 mask |
| `state_matrix(mask, col) = val` 在 `jit` 下报错 | `jax-patterns.md` | §2-3 函数式更新 |
| 想抽 `k` 个人但 `k` 是运行时才确定 | `jax-patterns.md` | §4 静态上界 + 动态有效数 |
| 无放回抽样（`randperm`）怎么 `jit` | `jax-patterns.md` | §5 `sample_up_to_k` |
| 新生儿要追加行但 `jit` 要静态 shape | `jax-patterns.md` | §6 空槽分配 |
| 规则表 / transition CSV 怎么翻成 JAX | `jax-patterns.md` | §8 `TransitionSpec` |
| `for t = 1:T` 的周循环怎么写 | `jax-patterns.md` | §9 `lax.scan` |
| MC 迭代怎么并行 | `jax-patterns.md` | §10 `vmap` |
| RNG 管理（MATLAB 隐式全局 → JAX 显式 key） | `jax-patterns.md` | §11 PRNG |
| `gcloud scp/ssh` 卡住或报 `FAILED_PRECONDITION` | `tpu-ops.md` | §1 硬规则 |
| `jax.distributed.initialize()` 顺序不对崩溃 | `tpu-ops.md` | §1 硬规则 |
| 每次 JIT 编译都几分钟 | `tpu-ops.md` | §4 编译缓存 |
| 分别测编译时间 vs TPU 计算时间 | `tpu-ops.md` | §5 `block_until_ready` |
| 多 chip 怎么切 MC 轴 | `tpu-ops.md` | §6 `shard_map` |
| per-chip batch 大小怎么选 | `tpu-ops.md` | §7 `per_chip_batch` |
| `N_max` 静态槽位开多大 | `tpu-ops.md` | §8 槽位规划 |
| 跑完不知道怎么和 MATLAB 比 | `validation.md` | §1 三个对账层次 |
| 想逐 cell 对齐 MATLAB | `validation.md` | §2 replay bundle |
| MC 噪声 vs 系统性偏差分不清 | `validation.md` | §3-4 z-score + 诊断 |
| MATLAB 原版有 bug 要不要复刻 | `validation.md` | §5 exact vs correct 双路径 |
| 不知道从哪开始 bisect | `validation.md` | §6 bisect 工作流 |
| 项目刚启动，整体对账流程怎么走 | `validation.md` | §7 三阶段工作流 |
| 两边 seed 都 42 了为啥还对不上 | `validation.md` | §7.1 复现的是结果不是随机数 |
| 每阶段什么时候算"过关" | `validation.md` | §7.2 退出标准 |
| Dump 先做多少周合适 | `validation.md` | §7.4 渐进扩展 |
| 两边差异是 bug 还是 encoding 问题 | `validation.md` | §7.5 encoding vs logic |

---

## 适用模型类型

这套手册适合：

- **Agent-based / compartmental / time-stepped 仿真** —— 每个步骤是固定的 update 序列
- **MATLAB 源码结构**：主循环 `for t = 1:T` + 若干 transition / 规则表 + 外层 MC 重复
- **目标环境**：JAX，单 GPU / 单 TPU / 多主机 TPU pod（v5e 或 v4）
- **规模**：~10⁴ 到 10⁶ 个 agent，能全部装进 HBM

如果你的模型本质不同（PDE / 连续时间 ODE / GNN / 带外部环境的 RL），里面的
通用模式（静态 shape、`scan`、`shard_map`）仍然适用，但具体例子需要自己翻译。

---

## 不覆盖的范围

- 训练神经网络（本项目无可学参数，纯 forward-only 仿真）
- GPU 专属优化（cuDNN、Triton kernel）
- JAX 的 AOT 编译 API（`jax.jit(f).lower().compile()`）
- 分布式训练 / `pjit` 的 optimizer state sharding
- 跨云 / 跨加速器部署（只用过 Google Cloud TPU v5e）

---

## 原项目结构参考（不包含在本 repo）

原 JAX 端口项目（私有代码，未上传）的典型布局，供结构参考。Appendix 章节里
的具体数字和代码片段来自这些文件的实测结果：

```
jax_port/
├── simulate.py             # 核心动力学 + scan + shard_map
├── driver.py               # 参数装配 + multi-host 初始化
├── prepare_inputs.py       # MATLAB Excel/CSV → JAX-friendly npz
├── launch_tpu.sh           # 四阶段 TPU 部署脚本
├── compare_to_matlab.py    # 统计对账脚本
└── debug_tools/
    ├── build_matlab_replay_bundle.py
    └── compare_dumps.py
```

---

## 协议

MIT —— 见 [LICENSE](LICENSE)。
