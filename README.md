# MATLAB → JAX → TPU 移植知识库

> 本目录沉淀了作者一个 agent-based 传染病模型从 MATLAB 移植到 JAX + TPU v5e-64 多主机集群的实战经验，供后续同类项目的 Claude Code / Cursor / 其他 LLM 会话快速上手。

## About

This is an LLM-assistable knowledge base distilled from a real
MATLAB → JAX → TPU port of an agent-based epidemic simulator (~260k
agents, 85 weeks, 26 intervention scenarios, running on a Google TPU
v5e-64 pod slice).

**The original project's source code is not included** (private research
codebase). What's here is the general-purpose porting playbook:
patterns, pitfalls, validation methodology. Portable to any MATLAB →
JAX port of a time-stepped simulation.

**Designed to be loaded as persistent context** into a coding-assistant
LLM session. Four files totaling ~1800 lines, organized for lazy
loading by topic. Humans can read it as plain prose too.

**Language**: Chinese (main text) + English (glossary / key terms).
Most headings, tables, and code examples are language-neutral.

---

---

## 新会话 AI：你是谁，你要做什么

**你**（后续会话的 Claude）正在帮用户把一个 MATLAB 写的传染病模拟器（或类似的 agent-based / compartmental / time-stepped 模型）移植到 JAX。最终可能要在 TPU 上跑。**用户之前做过一遍**，这些文档是他踩过坑之后沉淀的；**你不需要重新踩一遍**。

**你不要做**：
- 不要默认读 Appendix 的 mpox 案例，除非用户让你读或他问"上个项目是怎么做的"
- 不要假设新项目细节和 mpox 一样（人群规模、scenario 数量、tally 列都可能不同）
- 不要一次把 4 份文档全部读完塞进 context——**按需加载**

**你要做**：
1. 先读完本 README（你正在看的）
2. 按下面的"决策树"在用户提出具体问题时主动 `Read` 对应子文档
3. 引用原项目代码时直接用绝对路径（见文末"原项目位置"）

---

## 阅读顺序

| 时机 | 读什么 |
|---|---|
| 刚开始，还没写代码 | 本 README（整个）|
| 开始翻译 MATLAB 代码前 | `jax-patterns.md` |
| 准备上 TPU 前 | `tpu-ops.md` |
| 想验证 JAX 和 MATLAB 结果一致 | `validation.md` |
| 用户问"上个项目怎么做的" | 对应子文档的 Appendix |

---

## 决策树（遇到问题读哪份）

| 症状 / 问题 | 读哪份 | 定位 |
|---|---|---|
| MATLAB 里 `find()` 返回变长索引，不知道 JAX 怎么写 | `jax-patterns.md` | §2 布尔 mask |
| `state_matrix(mask, col) = val` 在 jit 下报错 | `jax-patterns.md` | §2-3 函数式更新 |
| 想抽 k 个人但 k 是运行时才确定 | `jax-patterns.md` | §4 静态上界+动态有效数 |
| 无放回抽样（`randperm`）不知道怎么 jit | `jax-patterns.md` | §5 sample_up_to_k |
| 新生儿要追加行但 jit 要静态 shape | `jax-patterns.md` | §6 空槽分配 |
| 规则表 / transition CSV 怎么翻成 JAX | `jax-patterns.md` | §8 TransitionSpec |
| `for t = 1:T` 的周循环 JAX 怎么写 | `jax-patterns.md` | §9 lax.scan |
| MC 迭代怎么并行 | `jax-patterns.md` | §10 vmap |
| RNG 怎么管理（MATLAB 是隐式全局，JAX 要显式 key）| `jax-patterns.md` | §11 PRNG |
| gcloud scp/ssh 卡住或报 FAILED_PRECONDITION | `tpu-ops.md` | §1 硬规则 |
| `jax.distributed.initialize()` 顺序不对导致崩 | `tpu-ops.md` | §1 硬规则 |
| 每次 JIT 编译都几分钟 | `tpu-ops.md` | §4 编译缓存 |
| 怎么分别测编译时间 vs TPU 计算时间 | `tpu-ops.md` | §5 block_until_ready |
| 多 chip 怎么切 MC 轴 | `tpu-ops.md` | §6 shard_map |
| per_chip batch 大小怎么选 | `tpu-ops.md` | §7 per_chip_batch |
| `N_max` 静态槽位开多大 | `tpu-ops.md` | §8 槽位规划 |
| 跑完不知道怎么和 MATLAB 比 | `validation.md` | §1 三个对账层次 |
| 想逐 cell 对齐 MATLAB | `validation.md` | §2 replay bundle |
| MC 噪声 vs 系统性偏差分不清 | `validation.md` | §3-4 z-score + 诊断 |
| MATLAB 原版有 bug 要不要复刻 | `validation.md` | §5 exact vs correct 双路径 |
| 不知道从哪开始 bisect | `validation.md` | §6 bisect 工作流 |
| 项目刚启动，不知道整体对账流程怎么走 | `validation.md` | §7 三阶段工作流（replay → distribution → production）|
| 两边 seed 都 42 了为啥还对不上 | `validation.md` | §7.1 复现的是结果不是随机数 |
| 每阶段什么时候算"过关" | `validation.md` | §7.2 退出标准 |
| Dump 先做多少周合适 | `validation.md` | §7.4 渐进扩展 |
| 两边差异是 bug 还是 encoding 问题 | `validation.md` | §7.5 encoding vs logic |

---

## 前提假设

本知识库假设你的新项目大致满足：

- **模型类型**：agent-based / compartmental / time-stepped 模拟（每步一个固定的 update 序列）
- **MATLAB 源码结构**：主循环 `for t = 1:T` + 若干 transition 表 / 规则表 + 外层 MC 重复
- **目标环境**：JAX，可能单 GPU / 单 TPU / 多主机 TPU（v5e 或 v4）
- **规模**：agent 数 ~10^4 到 10^6，可以全部放进 HBM

**如果你的模型本质不同**（比如 PDE / 连续时间 ODE / 图神经网络 / 需要外部 RL 环境），本知识库里"静态 shape / scan / shard_map"这几条通用手法仍然适用，但**具体例子可能对不上号，需要你自己翻译**。

---

## 写作约定（帮你快速扫读）

- **中文为主，关键术语英文括注**：`布尔掩码 (boolean mask)`, `扫描循环 (lax.scan)`
- **代码、函数名、JAX 原语一律英文**：`jnp.where`, `jax.random.split`, `_find_empty_slots`
- **每个概念三段式**：
  1. What（一句话说是什么）
  2. Why（为什么要这样，MATLAB 怎么写 → JAX 为什么不能直接翻）
  3. How（小数字具体例子走一遍）
- **Appendix 隔离**：每份文档最后一节 `Appendix: mpox 项目实战案例` 才出现具体的 infection9 / race shadow / compile 129s 这类数字，前面都是通用模式

---

## 原项目结构参考（不包含在本 repo）

原 JAX 端口项目（私有）的典型文件布局，供结构参考。这些文件**不在本 repo 里**，
Appendix 里的具体数字和代码片段来自这些文件的实测结果：

```
jax_port/                    # 原项目根目录（私有代码）
├── simulate.py              # 核心动力学 + scan + shard_map 实现
├── driver.py                # 参数装配 + multi-host 初始化 + 输出落盘
├── prepare_inputs.py        # MATLAB Excel/CSV → JAX-friendly npz
├── launch_tpu.sh            # 4 阶段 TPU 部署脚本（prep/sync/smoke/run）
├── compare_to_matlab.py     # z-score 对账脚本
├── BRIEFING.md              # 项目总览 + glossary + worked examples
├── ALIGNMENT_DEBUG_LOG.md   # 完整对齐调试史 + lessons learned
├── debug_tools/
│   ├── build_matlab_replay_bundle.py  # replay bundle 打包模式
│   └── compare_dumps.py     # MATLAB ↔ JAX 静态状态对比
└── logs/                    # 实测计时 + 对账输出（compile 129s / tpu_run 13.5s）
```

**在新项目中怎么用本知识库**：
- 遇到"MATLAB 这个模式 JAX 怎么写"→ 查 `jax-patterns.md` 对应章节
- 准备上 TPU 前 → 通读 `tpu-ops.md` §1-§3 的硬规则 + 陷阱
- 想证明 JAX 和 MATLAB 结果一致 → `validation.md` §1-§7 完整流程

---

## 已知的"不适用"情况

本知识库**不覆盖**：
- 训练神经网络（本项目无可学参数，全是 forward-only 模拟）
- GPU 专属优化（cuDNN、Triton kernel）
- JAX 的 AOT compile 细节（`jax.jit(f).lower().compile()` API 没深用）
- 分布式训练（没用过 `pjit` 的 optimizer state sharding）
- 跨云 / 跨加速器部署（只用过 Google Cloud TPU v5e）

如果用户问到这些，**明确告诉他**"这不在本知识库覆盖范围，需要现查 JAX 官方文档"。
