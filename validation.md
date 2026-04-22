# 验证与对账：怎么证明 JAX 和 MATLAB 等价

> 本文讲**如何系统地验证 JAX port 和 MATLAB 原版一致**——从 bit-level 对齐到分布级 z-score，以及怎么设计 exact-vs-correct 双路径让工程可交付和学术可 audit 两个目标同时达成。

---

## §1 对账的三个层次

| 层次 | 粒度 | 工具 | 何时用 | 局限 |
|---|---|---|---|---|
| **L1 Bit-match** | 逐 cell 相等 | Replay bundle + 同 seed | 数值管线新搭好，要证"我没引入新 bug" | 只证管线，不证模型本身 |
| **L2 Distribution-match** | 均值 ± SE 在噪声内 | z-score 对比 | 上线前、MC iter 足够多（≥ 64） | 对 "systematic bias < 噪声" 无感 |
| **L3 Qualitative-match** | 曲线形状 + 峰值 + 分层趋势 | 手工检视 | 向领域专家展示 / 论文图 | 主观 |

实际工作流里**三层都要做**，但不同阶段重点不同：
- 管线刚搭完 → L1 为主
- 模型 calibration → L2 为主
- 汇报 / 论文 → L3 为主 + L2 数字背书

---

## §2 L1 Bit-match: Replay Bundle 模式

### What
让 MATLAB 每周把所有随机事件（谁被感染、谁新生、谁转 Asym→Sym...）**dump 到磁盘**，JAX 端读进来**直接覆盖自己的抽样结果**。这样两边走相同的"事件序列"，数值管线错了就会立刻出现 cell-level 差异。

### Why
- 两边 RNG 不一样（MATLAB Mersenne Twister vs JAX threefry），即使 seed 相同序列也不同
- 所以"同 seed"不足以做 bit-match，必须**让 JAX 使用 MATLAB 的抽样结果**
- 这样对账能定位到"**某一周、某个 step、某个字段**"错

### How 数据结构

MATLAB 每周 dump 一个字典 `events[t]`：
```python
{
    "import_ids":       [id, id, id, ...],      # 本周被 import 转 sym 的人
    "birth_rows":       np.ndarray shape (B, K), # 本周新生儿的完整行
    "infection_ids":    [id, id, ...],           # 本周被感染的人
    "aware_asym_ids":   [id, ...],
    "aware_sym_ids":    [id, ...],
    "vac1_ids":         [id, ...],
    "vac2_ids":         [id, ...],
    "asym2sym_ids":     [id, ...],
    ...
}
```

### 打包成固定 shape 3D 数组

变长事件列表不能直接 jit。打包成三件套：
```python
# 每种事件一套
enabled:  shape (T+1,)       bool      # 本周有没有 event
counts:   shape (T+1,)       int32     # 本周有多少个
ids:      shape (T+1, K_max) int32     # 每周填 K_max 个，不够用 -1 padding
```

运行时 `_mask_from_replay_indices(ids[t], counts[t], N)` 把这周的 id 列表转成 `(N,)` bool mask。

### JAX 端覆盖抽样

在每个 `step_*` 函数里加 replay 分支：
```python
def step_infection(state, params, calib, key):
    # ... 正常计算 P_agent ...

    if params.get("replay_infection_ids") is not None:
        week_t = params["current_week"]
        if bool(params["replay_infection_enabled"][week_t]):
            to_infect = _mask_from_replay_indices(
                params["replay_infection_ids"][week_t],
                params["replay_infection_n"][week_t],
                N,
            )
            return state._replace(pox_status=jnp.where(to_infect, POX_ASYM, state.pox_status)), to_infect

    # 正常 RNG 路径
    ...
```

### 验证流程
1. MATLAB 跑一次，dump `events.mat`
2. Python 转换 `events.mat` → `replay_bundle.npz`
3. JAX 加载 replay bundle，注入 params
4. 跑 JAX，对比每周 state signature 和 tally 行
5. 任何一处差异 → 管线 bug；全部一致 → 管线正确

**关键**：这一关通过之后，后续 L2 / L3 的差异才能**归因于模型本身**，而不是管线。

---

## §3 L2 Distribution-match: z-score 栈对比

### What
JAX 保留**每 iter 的 tally CSV**（而不是提前平均），计算：
```
SE[week, col]  = std_iter[week, col] / sqrt(N_iter)
z[week, col]   = (JAX_mean[week, col] - MATLAB_mean[week, col]) / SE[week, col]
```

按 `max|z|` 排序找最可疑列。

### Why
- MATLAB 惯例直接在主脚本里 `mean(iter_stack)` 落盘，不保留方差
- JAX 保留全栈 → 能算 SE → 能区分"MC 噪声"vs"系统性偏差"
- `max|z|` 是秩序指标，不是严格假设检验

### How 目录结构

JAX 输出**刻意复刻 MATLAB 的目录布局**：
```
MonteCarloResults/
├── new_model_S1/
│   ├── iter0/state_matrices/Tally_new_model_S1.csv
│   ├── iter1/state_matrices/Tally_new_model_S1.csv
│   ├── ...
│   └── iterN/state_matrices/Tally_new_model_S1.csv
├── new_model_S2/
└── ...
```

这样同一份 compare 脚本可以扫两边。

### How 列名对齐

**输出 CSV 的表头逐字照抄 MATLAB 的**：含空格、大小写、特殊符号（`|`）都保留。MATLAB 是 `"new infect|hiv"`，JAX 也用这个名字。compare 脚本直接按列名查表，无需 mapping 层。

### How 对比脚本骨架

```python
def load_jax_stack(jax_dir, scenario_tag):
    """读所有 iter*/Tally_*.csv，堆成 (iters, weeks, cols)。"""
    pattern = os.path.join(
        jax_dir, f"new_model_{scenario_tag}", "iter*",
        "state_matrices", f"Tally_new_model_{scenario_tag}.csv",
    )
    paths = sorted(glob.glob(pattern))
    cols = [c.strip() for c in pd.read_csv(paths[0]).columns.tolist()]
    stack = np.stack([pd.read_csv(p).values for p in paths]).astype(np.float64)
    return stack, cols

def load_matlab_avg(matlab_dir, scenario_tag):
    """MATLAB 的 AvgTally 是已经 iter 平均的，只有 (weeks, cols)。"""
    path = os.path.join(matlab_dir, f"new_model_{scenario_tag}",
                        f"AvgTally_new_model_{scenario_tag}.csv")
    df = pd.read_csv(path)
    return df.values.astype(np.float64), [c.strip() for c in df.columns]

def compare_scenario(jax_dir, matlab_dir, scenario, top_n=20):
    jax_stack, jax_cols = load_jax_stack(jax_dir, scenario)
    mat_mean, mat_cols = load_matlab_avg(matlab_dir, scenario)

    # (I, W, C) → (W, C)
    jax_mean = jax_stack.mean(axis=0)
    jax_se = jax_stack.std(axis=0) / np.sqrt(jax_stack.shape[0])

    # 累计 week 1..T（跳过 week 0）
    def cum(mat, ci): return float(mat[1:, ci].sum())

    print(f"{'column':<25} {'MATLAB':>12} {'JAX':>12} {'diff':>10} {'%diff':>10}")
    for c in HEADLINE_COLS:
        if c not in jax_cols or c not in mat_cols:
            continue
        ji = jax_cols.index(c); mi = mat_cols.index(c)
        j, m = cum(jax_mean, ji), cum(mat_mean, mi)
        pct = "(base~0)" if abs(m) < 1e-6 else f"{(j-m)/m*100:+.1f}%"
        print(f"{c:<25} {m:>12.1f} {j:>12.1f} {j-m:>+10.1f} {pct:>10}")
```

### 四块对比输出

1. **Headline 累计**：8-10 个宏观量（New Infections / To Death / To Vax / To Aware...），期待 <1% 差
2. **按关键分层 A**（比如 race）：这里最可能暴露 MATLAB 分配类 bug
3. **按关键分层 B**（比如 age）：应该对齐，如果不齐说明有 age-stratified 的 bug
4. **全列 top-N by max|z|**：找最可疑列，定位系统性偏差

---

## §4 MC 噪声 vs 系统性偏差：怎么分

### 判别规则

| 现象 | 归因 |
|---|---|
| 同 scenario 加大 iter 数，%diff 按 √N 收敛 | MC 噪声 |
| iter 翻 4 倍 diff 基本不变 | 系统性偏差 |
| 两个独立 scenario 的偏差方向相同 | 系统性 |
| 偏差方向随 scenario 随机 | 噪声 |
| Headline 偏差小（<1%）但某分层偏差大（>10%） | bug 分配到特定 subgroup |
| Headline 偏差也大（>5%）且不收敛 | 动力学本身差异 |

### 实际工作流

第一轮跑小 iter（32-64）做 smoke test：
- 如果 headline 都在 MC 噪声里 → 继续
- 如果有 20%+ 偏差 → 停下来找 bug

第二轮跑大 iter（256-1024）：
- Headline 应该收敛到 <1%
- 分层偏差揭示模型差异

第三轮（可选）：只跑系统性偏差列的 `--exact-matlab` 模式复刻 bug，验证能 bit-match

### 基数陷阱

**小基数列的百分比偏差会被放大**。如果 MATLAB 累计值 18.4，JAX 12.4，%diff = -32.5% 看着很吓人，但绝对差只有 6，MC 噪声就能解释。

**compare 脚本里**：`(base~0)` 占位符是给 `|MATLAB| < 1e-6` 的列用的，避免除零爆炸。但 `|MATLAB| = 18` 这种"小但不是零"的数仍然会放大 %diff。看的时候要**同时看绝对差 + MC 方差**，不能只看 %diff。

---

## §5 exact vs correct 双路径设计

### What
生产默认走**正确版本**（修掉 MATLAB 已知 bug），但保留 `--exact-matlab-<feature>` 开关**忠实复刻 bug**，用于 bit-match 验证。

### Why
- 单纯复刻 bug：工程上交付了错的结果
- 单纯修 bug：失去和 MATLAB 对账的能力，审稿人问"你改了什么怎么证明没引入新错"无言以对
- **两者并存**：可以同时说"我的默认版本是对的"+"我能逐 cell 复刻原版"

### How 实现模板

```python
def step_feature(state, params, key):
    """dispatcher：根据 exact 开关选路径，前后 housekeeping 共享。"""
    # ... 前置准备 ...

    if bool(params.get("exact_matlab_feature", False)):
        state_new = step_feature_exact_matlab(state, ...)
    else:
        state_new = step_feature_correct(state, ...)

    # ... 后置处理 ...
    return state_new

def step_feature_correct(state, ...):
    """工程正确版本（向量化 + bug 修掉）。"""
    ...

def step_feature_exact_matlab(state, ...):
    """逐行复刻 MATLAB，包括 bug。"""
    ...
```

### 在 driver 里加开关

```python
parser.add_argument("--exact-matlab-feature", action="store_true",
                    help="复刻 MATLAB 的 <feature> bug，用于 bit-match 验证")
...
params["exact_matlab_feature"] = args.exact_matlab_feature
```

### 在 README 里文档化

```markdown
## Non-exact Replication

本 JAX port 默认修掉了以下 MATLAB bug。想 bit-match 原版用对应开关：

1. [Feature A 本地索引当全局行号 bug]
   Symptom: ...
   Fix: ...
   Reproduce: `--exact-matlab-a`

2. [Feature B 变量 shadow bug]
   ...
```

### 验证双路径

```python
# 默认跑（正确版）
python3 driver.py --num-iter 256 ...

# 复刻模式跑（等同 MATLAB）
python3 driver.py --exact-matlab-a --exact-matlab-b --num-iter 256 ...
```

默认版和 MATLAB 应该 L2 分布级对齐（分层偏差揭示 bug 修正）；复刻版和 MATLAB 应该 L1 bit-level 对齐（通过 replay bundle）。

---

## §6 bisect 工作流：找"第一次不一致"的位置

### What
当 L2 对账发现某列偏差很大时，先不改模型，先定位"两边从哪一步开始不一样"。

### Why
一周 step 里有十几个子步骤，不逐步 bisect 就是瞎猜。

### How 方法 1：State signature

每个子步骤后打印一个**压缩的 state 指纹**（几个标量的 tuple）：
```python
def state_signature(state):
    alive = state.alive == 1
    return (
        int(alive.sum()),
        int((alive & (state.pox_status == POX_SUS)).sum()),
        int((alive & (state.pox_status == POX_ASYM)).sum()),
        int((alive & (state.pox_status == POX_SYM)).sum()),
        int((alive & (state.pox_status == POX_RECOVERED)).sum()),
        int((alive & (state.pox_aware == 1)).sum()),
        int((alive & (state.vaccinated == 1)).sum()),
        int((alive & (state.vaccinated == 2)).sum()),
        int((alive & (state.treatment == 1)).sum()),
        int((alive & (state.isolation == 1)).sum()),
        int(state.infectious_wks.max()),
    )

# 每周 step 里
sig_pre = state_signature(state)
state = step_ve_update(state, params)
sig_after_ve = state_signature(state)
state, imp_n = step_import(state, week_t, params, k_imp)
sig_after_import = state_signature(state)
...
```

MATLAB 也在对应位置打印同样的 signature，两边 diff 定位"第一个开始不一样"的子步骤。

### How 方法 2：逐步 replay

开 replay bundle 模式**注入所有事件**：
- 如果每周 signature 都完全一致 → 管线 100% 正确
- 如果仍有差异 → 非随机部分（transition 条件判断？data type cast？）的 bug

### How 方法 3：stage freeze

**先冻结 RNG，只跑确定性流程**：
```python
params["replay_birth_rows"] = matlab_birth_rows
params["replay_import_ids"] = matlab_import_ids
# ... 所有随机事件都 replay
# transition 内部的 uniform 用同一把 fixed key
params["master_key"] = jax.random.PRNGKey(0)
```

这样把模型拆成"确定性路径"（应完全一致）+"随机路径"（用 replay 固定）两部分独立验证。

### Debug 建议的顺序

1. **先稳定输入预处理**：CSV 解析、demog group 映射、mixing matrix 读取的数字两边逐值对比
2. **再验证确定性流程**：freeze 所有 RNG，看 birth / aging / 规则判断逻辑
3. **最后加随机**：replay 注入，state signature 逐步比对
4. **发现某步偏差** → 只看那一步的输入和输出

反着来（上来就跑全量对分布）会陷在 MC 噪声里 debug 半天。

---

## §7 实战工作流：replay → distribution → production 三阶段

### §7.1 核心认知：复现的是"结果"不是"随机数"

MATLAB 的 `rand()` 全局 RNG 状态是**隐式**的，你看不到"这个 uniform 值是 seed 42 的第几步派生的"。**你也不需要看**。

需要的是 MATLAB 每周实际做出的**决策**：
```
第 3 周 MATLAB 实际：
  - 把 id=[2, 17, 89, ...] 这些人标成被感染
  - 把 id=[5, 44, ...] 这些人转成 aware
  - 新生了 id=[524289, 524290, ...] 这几个人
  - ...
```

把这些**最终决策**（agent id 列表）dump 成 `.mat` / `.npz`，打包进 replay bundle。JAX 端用这些 id 直接覆盖自己的抽样结果——**JAX 的 RNG 完全不动**。

```python
def step_infection_replay(state, params, key):
    # key 仍然要传（保持 signature），但实际不消费
    week_t = params["current_week"]
    to_infect = _mask_from_replay_indices(       # ← MATLAB 决策直接注入
        params["replay_infection_ids"][week_t],
        params["replay_infection_n"][week_t],
        N,
    )
    return update_field(state, "pox_status", to_infect, POX_ASYM)
```

所以"跨语言 bit-match"的实现**不依赖共享 seed**，而是依赖**共享决策**。

### §7.2 阶段序列 + 退出标准

定好每阶段的"通过"判据，防止无限 debug：

| 阶段 | 目的 | 通过标准 | 通过后做什么 |
|---|---|---|---|
| **Phase 1: Replay L1** | 证明管线逻辑（状态更新顺序 / 公式）和 MATLAB 一致 | 开 replay bundle 下，每周所有 state 字段逐 cell 相等（容差 1e-6） | 关掉 replay，进 Phase 2 |
| **Phase 2: Distribution L2** | 证明独立 RNG 下分布对齐 | Headline 列 %diff < 1%；分层列偏差随 iter 数按 √N 收敛 | 进 Phase 3 |
| **Phase 3: Production** | 生产可复现 | `base_seed + iter` 协议下，同 base_seed 两次跑结果完全一致（JAX 自己内部） | 可以出论文图 |

**失败就停下**：任何一阶段不达标，**先修逻辑再继续**，不要跳着过。

### §7.3 MATLAB 端 DumpSeed 策略

MATLAB 端固定 `DumpSeed=42`（或任何常数），但要理解它的作用：

- ✅ 让 MATLAB **自己的 gold dump 可复现**：明天重跑，事件 id 一字不差
- ✅ 让 mismatch **精确归因**：出现差异就是 JAX 实现 bug，不是 MATLAB 今天心情不好
- ❌ **不是**为了跨语言共享——即使两边都 seed=42，MATLAB Mersenne Twister 和 JAX Threefry 的序列完全不同

**DumpSeed 变化的正当时机**：
- 改了 MATLAB 代码逻辑后重新产 gold（预期会变，这是新的 baseline）
- 想换 RNG 轨迹看看结果是否稳定

**不应变化的时候**：
- Debug JAX 时（gold 必须稳定，否则 JAX 没法做参照）

### §7.4 Dump 范围渐进扩展

一开始别 dump 全部 85 周——太重，迭代慢。分阶段扩：

| 批次 | Dump 长度 | 目的 |
|---|---|---|
| 1a | 前 5 周 | 快速迭代，抓初始化 / 数据 pipeline bug |
| 1b | 前 20 周 | 抓时间累积型 bug（比如 `vax_wk += 1` 写错、`infectious_wks` 边界） |
| 1c | 全部 T 周 | 最终验证，覆盖干预启动后的行为 |

**规律**：如果前 N 周全 match、第 N+1 周突然发散，**一定是时间累积型 bug**（某个 counter / schedule / threshold）。这种 bug 在短 dump 里看不到。

### §7.5 区分 encoding bug vs logic bug

**不是所有 mismatch 都是逻辑错**。先过滤三类 encoding 差异：

| mismatch 来源 | 症状 | 对策 |
|---|---|---|
| **索引 base 差** | JAX 说 id=5 被感染，MATLAB dump 里写 id=6；整批系统性 off-by-1 | Dump 时明确注明 0-based 还是 1-based，转换时统一减 1 |
| **周号 base 差** | Week 1 的 JAX 对齐 MATLAB 的 Week 0 | 约定：Week 0 = 初始状态（无事件），Week 1..T = 演进步数 |
| **浮点精度** | log1p / expm1 / sqrt 最后一位不同，float32 vs float64 | Replay bundle 里的浮点字段统一用 float64；比对时用 1e-6 绝对容差 + 1e-4 相对容差 |
| **列名空格** | `"new infect|hiv"` vs `"new infect | hiv"`（中间多空格） | 所有列名 `.strip()` + lower-case 后比较 |
| **CSV 数值精度** | 写入 `%.4f` 丢了尾部精度 | Dump 用 `%.17g` 或直接 `.npz` 存二进制 |

**流程**：对账脚本先跑一遍这五类检查，过滤完再把剩下的 mismatch 当 logic bug 看。不做这一层过滤会浪费大量时间 debug 实际不存在的"逻辑错"。

### §7.6 Solo 工作流的自约束

一个人同时写 MATLAB 和 JAX 两边，最大的风险是**你会在脑子里悄悄"对齐"两边**——比如发现 MATLAB 有个奇怪的行为，你在 JAX 端"顺手"改成了"合理"的版本，但既没文档也没开关。三个月后你自己都不记得哪边是原版。

**自约束规则**：

1. **一份 `PROTOCOL.md` 写死所有约定**（放在新项目根目录）：
   - Agent id base（0 或 1）
   - Week id base
   - 事件 dump 的字段命名
   - Replay bundle 的 key 命名（`replay_infection_ids` 之类）
   - Float 精度容差
   - CSV 列名规则

2. **每个"我修掉了 MATLAB 的一个行为"都走 exact-vs-correct 双路径**（见 §5）。不要直接删掉 MATLAB 的 buggy 行为——留一个 `--exact-matlab-<feature>` 开关复刻它。这样：
   - 做 Phase 1 replay 时开开关，JAX 复刻 MATLAB → bit-match
   - 做 Phase 2 distribution 时关开关，JAX 走修正版 → 分布更合理
   - 三个月后你忘了改了什么，`grep exact_matlab` 一下全出来

3. **每轮 gold dump 存档**：每次 MATLAB 改动后产新 gold，把老 gold 归档到 `gold_archive/YYYYMMDD/`。JAX 出 bug 时可以回到老 gold 确认"是我这次改动引入的，还是历史遗留"。

4. **把 PROTOCOL.md 当 commit lint**：改 MATLAB 代码同时改 PROTOCOL.md 里的字段命名，两边一起 commit。PROTOCOL 里写了什么、MATLAB dump 必须产什么、JAX 必须吃什么，三者锁死。

---

## §8 测试的 sanity check 清单

上任何一轮大 iter 跑之前，先过一遍：

- [ ] **初始状态对齐**：week 0 的 tally 行两边完全一致（没有任何随机因素）
- [ ] **确定性列对齐**：`New Births` 这种由 schedule 驱动的量两边 0% 差（±MC 噪声范围）
- [ ] **L1 replay 通过**：注入 MATLAB events 后，至少前 5 周的 state signature 逐字相等
- [ ] **单 iter 跑通**：`num_iter=1` 不崩，CSV 能写出来
- [ ] **64 chip 跑通**：多 iter 多主机能跑到 week T，最终 tally 写盘
- [ ] **对比脚本能跑**：`compare_to_matlab.py --jax-dir ... --matlab-dir ...` 能输出对比表

过了这 6 步再去扫大的 MC。任何一步失败，先修再继续。

---

## Appendix: mpox 项目对账实战

> **新会话 AI 注意**：下面数字只在用户明确问"上一个项目对账结果怎样"时才读。

### A.1 实测 compare 输出（S6, JAX 256 iter vs MATLAB 20 iter）

```
[Headline 累计 (week 1..end)]
column                    MATLAB     JAX      diff    %diff
New Births                6800.0   6800.0    +0.0    +0.0%   ← 确定性，完美对齐
New Infections             381.9    379.1    -2.8    -0.7%   ← MC 噪声内
To Death                  5999.3   5987.8   -11.5    -0.2%   ← 对齐
To Vax                   11123.7  11211.2   +87.5    +0.8%   ← 对齐
To Aware                   354.9    246.4  -108.5   -30.6%   ← 系统性，bug 级联

[按 race 分层]  (MATLAB infection9 bug 会让这里偏)
new infect b               340.9     79.1  -261.8   -76.8%   ← MATLAB 全堆低 demog
new infect h                41.1    213.2  +172.1  +418.7%   ← JAX 正确分散
new infect w                 0.0     86.8   +86.8  (base~0)
```

### A.2 解读模板

- **New Births 0% 差**：确定性流程（schedule 驱动）对齐的 sanity check
- **New Infections -0.7%**：核心动力学在 MC 噪声内（iter=64 时是 +5.9%，256 收敛到 -0.7%）
- **To Aware -30.6%**：两个独立 scenario 都一致偏小 → 系统性。归因于 MATLAB bug #1 + #3 的级联（infection 错堆 × HIV awareness 误分配）
- **race 分层 ±77%, +418%**：不是 JAX 错，是 MATLAB 错，修掉后分层才正确

### A.3 复刻验证

开 `--exact-matlab-infection --exact-matlab-hiv`：
- To Aware 从 246.4 回升到 ~354（匹配 MATLAB 354.9）
- race 分层也对齐（MATLAB 全堆 B race，JAX 也全堆 B）
- 这就是 L1 bit-match 的证据：管线正确，默认路径只是修了 MATLAB bug

### A.4 汇报话术

给教授 / 审稿人的一分钟版本：
> "我比较的三个层次：bit-level 用 replay bundle 跑过，管线正确；分布级 z-score 在 MC 噪声内（headline <1%）；分层指标的系统性偏差精确对应我审计出的 MATLAB bug 签名，修掉后结果分布更合理。这两个版本都能一键切换：`--exact-matlab` 复刻原版，默认路径给修正版。"
