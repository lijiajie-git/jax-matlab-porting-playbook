# JAX 翻译手法：从 MATLAB 过程式到 JAX 函数式

> 本文讲**把 MATLAB 的动态 shape / 原地赋值 / 隐式 RNG 翻译成 JAX 的静态 shape / 函数式更新 / 显式 key 的全套手法**。每一节的模式都是跨项目通用的，具体 mpox 案例放在文末 Appendix。

---

## §1 JAX 的三条核心约束（必须先懂的前提）

任何 MATLAB 代码翻成 JAX 前先对照这三条看哪里要改：

### 1.1 静态 shape (static shape)

**约束**：`jax.jit` 编译的函数里，所有数组的 shape 必须在**编译时**确定。运行时才知道的 shape（比如 `find(mask)` 返回多少个索引）不能直接用。

**后果**：MATLAB 里"动态 append 行"、"变长索引数组"、"if 分支改变 shape"都必须改写成"固定 shape + mask 过滤"。

### 1.2 函数式更新 (functional / immutable update)

**约束**：JAX 数组不可变。`x[i] = v` 不存在，必须 `x_new = x.at[i].set(v)` 或 `x_new = jnp.where(mask, v, x)`，**返回新数组**。

**后果**：MATLAB 的 `state_matrix(mask, col) = val` 要翻译成返回新 state 的函数。整个代码风格从"改变量"变成"返回新值"。

### 1.3 显式 RNG key (explicit pseudo-random key)

**约束**：JAX 没有全局随机状态。每次抽样都要传一个 `key: PRNGKey`，用完立即 `split` 出下一批 key。不 split 直接重用会得到一样的抽样结果。

**后果**：MATLAB 的 `rand(N,1)` 隐式使用全局 seed，翻成 JAX 是 `jax.random.uniform(key, (N,))`。一个函数里有 14 处独立抽样就要 `jax.random.split(key, 14)` 一次分发 14 个子 key。

---

## §2 布尔掩码 (boolean mask) 替代 MATLAB `find()`

### What
MATLAB 的 `find(cond)` 返回满足条件的**变长索引数组**。JAX 里用 **shape 恒为 `(N,)` 的布尔向量** 替代，配合 `jnp.where` 做条件更新。

### Why
- `find()` 返回的数组长度依赖数据 → jit 编译不出来
- Boolean mask shape 静态 → jit 友好
- 代价是"每个位置都算一遍"，但 TPU/GPU 的并行度正好吃这种形状

### How

**MATLAB 写法**：
```matlab
ids = find(alive==1 & pox_status==2);     % ids 可能是 [] 也可能 1000 个
state_matrix(ids, TREATMENT_COL) = 1;
```

**JAX 写法**：
```python
mask = (state.alive == 1) & (state.pox_status == 2)
# mask shape = (N,)，永远这么长
state = state._replace(
    treatment=jnp.where(mask, 1, state.treatment)
)
```

### 工具函数模板
把 helpers 单独抽出来，全代码库复用：

```python
def mask_alive_and(state, *masks):
    """返回 alive=1 ∧ 所有 masks 都为 True 的 bool 向量。"""
    m = state.alive == 1
    for x in masks:
        m = m & x
    return m

def update_field(state, field: str, mask, new_val):
    """把 state.<field> 在 mask=True 的位置换成 new_val。
    new_val 可是标量或 (N,) 数组。自动 cast 到原 dtype。"""
    old = getattr(state, field)
    new = jnp.where(mask, jnp.asarray(new_val, dtype=old.dtype), old)
    return state._replace(**{field: new})
```

翻译模式一旦建立，几乎所有 MATLAB `state_matrix(mask, col) = val` 都变成一行 `update_field(state, "col", mask, val)`。

---

## §3 函数式 state 设计（SoA: struct-of-arrays）

### What
整个模型的"当前状态"用一个 `NamedTuple`，字段是长度 N 的数组（不是 N 个 struct）。每次更新返回**新的 NamedTuple**。

### Why
- SoA 对 SIMD/MXU 友好：同类型字段 stride=1 的连续 memory
- NamedTuple 是 JAX 原生 pytree，可以被 `vmap` / `scan` / `shard_map` 透明传递
- `state._replace(field=new_val)` 语法清晰

### How

```python
from typing import NamedTuple
from jax import Array
import jax.numpy as jnp

class State(NamedTuple):
    # 人口学
    age:      Array   # int16, shape (N_max,)
    race:     Array   # int8
    demog_id: Array   # int16
    # 疾病状态
    pox_status: Array  # int8, 0=Sus/1=Asym/2=Sym/3=Rec
    # 干预
    treatment:  Array  # int8
    # 槽位簿记
    alive:  Array   # int8
    used:   Array   # int8

def make_initial_state(raw: dict, N_max: int) -> State:
    """raw 是 numpy / python-scalar 输入，pad 到 N_max。"""
    def pad(x, dtype):
        a = jnp.zeros((N_max,), dtype=dtype)
        return a.at[:len(x)].set(jnp.asarray(x, dtype=dtype))
    return State(
        age=pad(raw["age"], jnp.int16),
        race=pad(raw["race"], jnp.int8),
        ...
    )
```

**注意**：如果 NamedTuple 里有 `str` 字段，要手动 `jax.tree_util.register_pytree_node` 把字符串归为 `aux_data`（否则 `shard_map` 会把字符串当 leaf 报错）。

---

## §4 静态上界 + 动态有效数（Static upper bound + dynamic effective count）

### What
很多 MATLAB 操作形如"抽 k 个"、"写 k 行"，k 是运行时才确定的。JAX 里把"上界 `k_upper`"写成 Python int（静态），"实际有效数 `k_eff`"留成 tracer（动态），配合 mask 来控制。

### Why
- `top_k`、`one_hot` 这些原语都要静态 k
- 动态 k 用 mask 过滤：`active = arange(k_upper) < k_eff`
- 三重 `jnp.minimum` 封顶，确保 `k_eff ≤ k_upper`：

```python
k_eff = jnp.minimum(jnp.minimum(k_desired, eligible_count), jnp.int32(k_upper))
```

`k_desired` 是你想抽的数量（tracer），`eligible_count` 是实际有多少人可抽（tracer），`k_upper` 是编译期上界（Python int）。

### How 实际选 k_upper

典型场景：
- 每周出生数：历史最大值 × 1.5
- 每周导入数：历史最大值 × 2
- 每周某类 transition 触发数：`alive_count` 的一个保守上界

**宁可 k_upper 大一点多浪费些计算**，不要太紧导致运行时 overflow。

---

## §5 无放回抽样 `_sample_up_to_k`（uniform + top_k + one_hot）

### What
从 `mask=True` 的位置里无放回地抽 `k_eff` 个（`k_eff ≤ k_upper`），返回 shape `(N,)` 的 bool mask。

### Why
MATLAB 用 `randsample` 或 `randperm(N, k)`，返回变长索引数组 → 不能 jit。JAX 下这个经典实现非常 TPU 友好：
1. 给每个候选位置抽一个 uniform 分数，不在 mask 里的位置分数设 -1
2. `top_k(scores, k_upper)` 取分数最高的 k_upper 个
3. `active` mask 只保留前 k_eff 个
4. `one_hot + sum` 把选中的索引翻译成 `(N,)` bool

**关键点**：全程**无 scatter、无动态索引**，都是稠密广播 + reduce，MXU 拉满。

### How

```python
def _sample_up_to_k(mask, k_upper: int, k_eff, key):
    """从 mask=True 里无放回抽 k_eff 个。k_upper 必须是 Python int。"""
    # 1. 评分
    scores = jnp.where(
        mask,
        jax.random.uniform(key, mask.shape, dtype=jnp.float32),
        jnp.full(mask.shape, -1.0, dtype=jnp.float32),
    )
    # 2. top_k
    top_vals, top_idx = jax.lax.top_k(scores, k_upper)
    # 3. active
    active = (jnp.arange(k_upper, dtype=jnp.int32) < k_eff) & (top_vals >= 0.0)
    # 4. one_hot + sum
    selected = (
        jax.nn.one_hot(top_idx, mask.shape[0], dtype=jnp.int32)
        * active.astype(jnp.int32)[:, None]
    ).sum(axis=0) > 0
    return selected
```

### 手算一遍 N=6, k_upper=3, k_eff=2

```
mask     = [T, F, T, T, F, T]     # 4 人 eligible
scores   ≈ [0.7, -1, 0.3, 0.9, -1, 0.4]
top_vals ≈ [0.9, 0.7, 0.4]
top_idx  = [3,   0,   5]

active = arange(3) < 2 = [T, T, F]

one_hot([3,0,5], 6) =
    [[0,0,0,1,0,0],
     [1,0,0,0,0,0],
     [0,0,0,0,0,1]]

× active[:,None]:
    [[0,0,0,1,0,0],
     [1,0,0,0,0,0],
     [0,0,0,0,0,0]]

sum(axis=0) = [1,0,0,1,0,0]   > 0  →  [T,F,F,T,F,F]
```

选中 idx 0 和 3。✅

### 为什么不直接 `selected.at[top_idx].set(True)`
那是 scatter 写入。TPU 下 scatter 比 dense compute 慢得多。`one_hot + sum` 把 scatter 改写成稠密矩阵形状，XLA 降级成 matmul 类 kernel，MXU 吃满。这是本港口里最反复出现的技巧。

---

## §6 空槽分配 `_find_empty_slots`（cumsum-as-rank 技巧）

### What
给定一个 shape `(N,)` 的 `used` 向量（`1=已占` / `0=空闲`），在不排序、不 scatter 的前提下，找出"最靠前的 k 个 `used==0` 位置"，返回 bool mask + 每个空槽的 1-based 序号。

### Why
新生儿、导入者要写入一个静态 shape 的 state，但"当前有多少空位"是运行时动态的。这个技巧是整个 JAX port 里**最漂亮的一招**：用前缀和给每个空槽分配一个唯一序号，然后 mask 过滤。

### How

```python
def _find_empty_slots(used, k):
    """返回 (slot_mask, rank)。
    slot_mask shape (N,) bool，标记前 k 个未使用的槽位。
    rank      shape (N,) int32，每个未使用槽位的序号 1..empty_count。"""
    empty = (used == 0).astype(jnp.int32)
    rank  = jnp.cumsum(empty) * empty       # 非空位置归零
    slot_mask = (rank > 0) & (rank <= k)
    return slot_mask, rank
```

### 手算一遍 N=10, k=3

```
used    : [1, 1, 0, 1, 0, 1, 0, 0, 0, 0]    # 4 个已占 + 6 个空
empty   : [0, 0, 1, 0, 1, 0, 1, 1, 1, 1]
cumsum  : [0, 0, 1, 1, 2, 2, 3, 4, 5, 6]
rank    : [0, 0, 1, 0, 2, 0, 3, 4, 5, 6]    # 非空位置 × 0 归零
                 ↑     ↑     ↑   ↑   ↑   ↑
                 空   空   空   空  空  空
                 #1  #2  #3  #4 #5 #6

slot_mask (k=3) = (rank>0) & (rank<=3)
                = [F, F, T, F, T, F, T, F, F, F]
                       ↑     ↑     ↑
                     idx 2  idx 4  idx 6
```

前 3 个空槽是 idx 2, 4, 6。全程无 sort、无 scatter，shape 永远 (N,)。

### 怎么把 rank 用作写入索引

`rank` 告诉你"这个空槽要从 entrants 列表的第几个取数据"。配合 `rank - 1` 作为 gather 索引：

```python
# entrants: 预先抽好的 k_upper 个候选人的属性
src_idx = jnp.clip(rank - 1, 0, entrants["age"].shape[0] - 1)
new_age = entrants["age"][src_idx]   # gather；slot_mask=F 位置的值无所谓

state = state._replace(
    age=jnp.where(slot_mask, new_age, state.age),
    # 其他字段同理
)
```

### 陷阱：死亡槽位不回收
如果你只在"死亡"时改 `alive=0` 不改 `used=0`，那么死亡槽位就是"僵尸槽"：`alive=0, used=1`，永远不会被新人覆盖。**N_max 需要覆盖 N₀ + 累计出生数**，不是"同时存活最大数"。具体见 `tpu-ops.md §8`。

---

## §7 group-by 用 `jax.ops.segment_sum`

### What
MATLAB 的 `groupsummary(data, group_id, @sum)` 或 `accumarray(group_id, data)` 对应 JAX 的 `jax.ops.segment_sum(data, segment_ids, num_segments)`。

### Why
- `num_segments` 是静态的，输出 shape 固定
- 比 for over groups + sum 快几个数量级

### How

```python
# demog_id ∈ [0, G)，统计每组活人数
N_g = jax.ops.segment_sum(
    alive.astype(jnp.float32),
    demog_id,
    num_segments=G,
)   # shape (G,)
```

### 陷阱：-1 哨兵值（sentinel）怎么处理
如果某些 agent 的 `demog_id = -1`（比如未分组的），直接传会越界。惯用做法：

```python
valid = demog_id >= 0
safe_id = jnp.maximum(demog_id, 0)   # 把 -1 暂时改成 0 避免越界
result = jax.ops.segment_sum(
    data * valid.astype(data.dtype),  # 用 mask 把无效数据归零
    safe_id,
    num_segments=G,
)
```

---

## §8 规则表 / transition 用 `TransitionSpec` + `lax.scan`

### What
MATLAB 里常见的"读一张 CSV，每行一条规则（条件 + 概率 + 目标状态），逐行判断"。JAX 里编译成一个 `TransitionSpec` NamedTuple，用 `lax.scan` 遍历规则。

### Why
- Python `for rule_idx in range(R)` 在 jit 下会**展开**成 R 份独立子图 → 编译慢 + 二进制巨大
- `lax.scan` body 只编译一次，R 改了也不会重编

### How

```python
class TransitionSpec(NamedTuple):
    target_field: str           # 要更新哪个 state 字段
    cond_cols:    tuple         # 每个条件列对应的 state 字段名
    cond_ops:     tuple         # "=", ">=", "<=", "ignore"
    conds:        Array         # (R, C) float32
    new_state:    Array         # (R,)
    probs:        Array         # (R,)

# 字符串字段（target_field / cond_cols / cond_ops）必须归为 aux_data，
# 否则 shard_map / jit 会把字符串当 array leaf 报 "not a valid JAX type"
def _ts_flatten(ts):
    return (ts.conds, ts.new_state, ts.probs), (ts.target_field, ts.cond_cols, ts.cond_ops)
def _ts_unflatten(aux, children):
    target_field, cond_cols, cond_ops = aux
    conds, new_state, probs = children
    return TransitionSpec(target_field, cond_cols, cond_ops, conds, new_state, probs)
jax.tree_util.register_pytree_node(TransitionSpec, _ts_flatten, _ts_unflatten)

def apply_transition(state, spec: TransitionSpec, calib: float, key):
    target_init = getattr(state, spec.target_field)
    R = spec.probs.shape[0]

    def body(carry, rule_idx):
        tgt, k_, changed = carry
        k_, sub = jax.random.split(k_)
        mask = _rule_mask(state, spec, rule_idx)   # 基于入口 state
        u = jax.random.uniform(sub, tgt.shape, dtype=jnp.float32)
        p = spec.probs[rule_idx] * calib
        hit = mask & (u < p)
        new_val = spec.new_state[rule_idx].astype(tgt.dtype)
        tgt_new = jnp.where(hit, new_val, tgt)
        return (tgt_new, k_, changed + hit.sum(jnp.int32)), None

    (tgt_final, _, total_changed), _ = jax.lax.scan(
        body, (target_init, key, jnp.int32(0)), jnp.arange(R)
    )
    return state._replace(**{spec.target_field: tgt_final}), total_changed
```

### 语义上要注意
- **条件判断基于入口 state**，不基于 tgt 的累积值（MATLAB transition5 就是这样）
- **target 列累积更新**：前一条规则的命中会改变后续规则里 target 字段的值
- 随机是对**全体 N**抽 uniform，再用 mask 过滤（不要"只抽 mask=True 位置"）

---

## §9 时间循环用 `lax.scan`（替代 Python for t）

### What
主循环 `for t = 1:T` 改写成 `lax.scan(body, state0, (weeks, keys))`。

### Why
- scan body 编译一次，XLA 不会展开成 T 份子图
- 自动把每步输出堆成 `(T, ...)` 数组
- 心智模型：有状态的 fold

### How

```python
def simulate(state0, params, key, T: int):
    keys = jax.random.split(key, T)

    def body(carry, xs):
        st = carry
        week_t, k = xs
        st_new, tally_row = weekly_step(st, week_t, params, k)
        return st_new, tally_row

    weeks = jnp.arange(1, T + 1)
    final_state, rows = jax.lax.scan(body, state0, (weeks, keys))
    return final_state, rows   # rows shape = (T, tally_cols)
```

### 心智模型（和 Python 等价形式对照）

```python
# JAX lax.scan
final_carry, outputs = lax.scan(body, init_carry, xs)

# 等价 Python
carry = init_carry
outputs = []
for x in xs:
    carry, out = body(carry, x)
    outputs.append(out)
outputs = stack(outputs)
```

---

## §10 MC 用 `vmap`，设备用 `shard_map`（三层并行堆叠）

### What
典型的"MC 多次重复 × 时间 T × 设备 D" 三层并行：

- **最内层** `lax.scan` 沿时间 T
- **中层** `vmap` 沿 MC 轴（chip 内本地并行）
- **外层** `shard_map` 把 MC 轴切分到多 chip

### Why
- MC 独立，**零通信**，是最便宜的并行维度，放外层最合理
- 时间必须串行（下一步依赖这一步），scan 最内层
- 顺序反过来（比如 vmap 外层 shard_map 内层）开销会爆炸

### How

```python
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map

def run_monte_carlo_sharded(state0, params, master_key, num_iter, T, num_devices):
    assert num_iter % num_devices == 0
    per_chip = num_iter // num_devices

    devices = np.array(jax.devices())[:num_devices]
    mesh = Mesh(devices, axis_names=("mc",))

    keys = jax.random.split(master_key, num_iter)   # (num_iter, 2)
    params_dyn, params_static = _split_params(params)   # 见 §12

    @partial(shard_map,
             mesh=mesh,
             in_specs=(P(), P(), P("mc")),   # state 和 params 复制，keys 沿 mc 切
             out_specs=P("mc"),
             check_rep=False)
    def fn(s0, p_dyn, k_shard):
        p = {**p_dyn, **params_static}
        simulate_v = jax.vmap(simulate, in_axes=(None, None, 0, None))
        _, tally = simulate_v(s0, p, k_shard, T)
        return tally

    return fn(state0, params_dyn, keys)
```

更多 TPU 细节（`_split_params`、`Mesh` 构造、`PartitionSpec` 语义）见 `tpu-ops.md §6`。

---

## §11 RNG 显式管理（PRNG key）

### What
MATLAB 有隐式全局 RNG（`rng(42)` 后 `rand(...)` 就用那个 seed）。JAX 没有全局状态，每次抽样都要显式传 key，用完立刻 split。

### Why
- 函数式：相同 key 输入 → 相同输出，可重复、可并行
- 手动 split 避免多次抽样用同一把 key 得到相关结果

### How

基础模式：
```python
# 顶层拿一个 master key
master_key = jax.random.PRNGKey(42)

# 多次独立抽样前一次 split 出来
keys = jax.random.split(master_key, 14)
(k_imp, k_birth, k_inf, k_aw1, ..., k_hiv) = keys

# 每个 key 消费一次
state, imp_n = step_import(state, week_t, params, k_imp)
state, birth_n = step_birth(state, params, k_birth)
state, to_infect, ... = step_infection(state, params, calib, k_inf)
```

### 和 MATLAB 对齐的 seed 策略

如果要**和 MATLAB 逐 cell 对齐**，需要两边用**相同的 seed 方案**。博士生如果用 `rng(42 + iter)`，JAX 这边对应：

```python
# iter i 的 master key
master_key_i = jax.random.PRNGKey(42 + i)
```

**但注意**：即使 seed 相同，两边的随机序列也不会一样（MATLAB 用 Mersenne Twister，JAX 用 threefry）。如果目标是 bit-level 对齐，**不能只靠 seed**，必须走 replay bundle 模式（见 `validation.md §2`）。

### 陷阱：链式 split vs 一次 split N 份

**不推荐**：
```python
k_a, k = jax.random.split(k)
# 用 k_a
k_b, k = jax.random.split(k)
# 用 k_b
```

**推荐**：
```python
k_a, k_b, k_c, ... = jax.random.split(k, N)
```

前者每次都会生成新 tracer，JIT graph 更复杂；后者一次性分发，更清晰。

---

## §12 `_split_params`：分离静态 / 动态参数

### What
`shard_map` 会把所有 pytree leaf 升成 tracer。Python `int` / `float` / `str` / `tuple` 一旦被 trace，在 `int(params["G"])` 或 shape 分支里会炸 `Abstract tracer used in static context`。

### Why
- Array 类参数 → 动态，走 `in_specs`
- Python scalar / tuple / str → 静态，走闭包捕获

### How

```python
def _split_params(params):
    """按值类型拆分: array → dynamic, python scalar/tuple/list/str → static。"""
    dyn, sta = {}, {}
    for k, v in params.items():
        if isinstance(v, (int, float, bool, tuple, list, str)) or v is None:
            sta[k] = v
        else:
            dyn[k] = v
    return dyn, sta

# 用法
params_dyn, params_static = _split_params(params)

@partial(shard_map, mesh=mesh,
         in_specs=(P(), P(), P("mc")), out_specs=P("mc"))
def fn(s0, p_dyn, k_shard):
    p = {**p_dyn, **params_static}   # 静态部分闭包捕获
    ...
```

这是所有 `shard_map` 代码的标配模板。

---

## Appendix: mpox 项目实战案例

> **新会话 AI 注意**：下面案例只在用户明确问"上一个项目是怎么做的"时才读。

### A.1 MATLAB 的"局部索引当全局行号" bug 复刻

mpox `infection9.m` 里一个典型 MATLAB 坑：

```matlab
for g = 1:G
    rows_in_g = find(demog_id == g);            % 全局行号 [2, 7, 15, ...]
    sus_idx_local = find(pox_status(rows_in_g) == 0);  % 局部 [1, 2, 3]
    ...
    infected = sus_idx_local(u < p);             % 还是局部
    state_matrix(infected, POX_COL) = 1;         % BUG：把局部当全局行号写
end
```

后果：所有感染错写到前几个 agent（idx 1, 2, 3...），低 demog_id 组被重复感染，race 分层完全走样。

**JAX 复刻 bug 的写法**（对账时验证 bit-match 用）：
```python
local_rank = jnp.cumsum(group_mask.astype(jnp.int32)) - 1   # 伪全局索引
buggy_global_idx = jnp.maximum(local_rank, 0)
infect_hits = jax.ops.segment_sum(
    infect_member.astype(jnp.int32),
    buggy_global_idx,
    num_segments=N,
)
```

**生产版本**直接 `jnp.where(sus & ..., POX_ASYM, state.pox_status)`，不走 bug 路径。详见 `jax_port/simulate.py` 里 `step_infection_exact_matlab` vs `step_infection` 的对照。

### A.2 MATLAB `for race=0:2` 变量 shadow

```matlab
race = state_matrix(:, RACE_COL);   % 向量
for race = 0:2                       % ← 把向量覆盖成标量！
    mask = race == race;             % 永远 true
    ...
end
```

**检测模式**：grep 所有 `for VAR = ...` 循环，看 VAR 是否和外层作用域的向量重名。

**JAX 端**：直接按 `(age, race)` 严格过滤即可：
```python
for race_idx in range(3):
    race_mask = state.race == race_idx
    eligible = alive & hiv_neg & age_mask & race_mask
    ...
```

### A.3 `update_ve` mode=0 的 logical-index vs `jnp.where` 语义差异

MATLAB 用逻辑索引赋值，**只写命中行**：
```matlab
plwh_vax1_id = hiv_status~=0 & vaccinated==1 & vax_wk>=2 & alive==1;
state_matrix(plwh_vax1_id, StateMatCols.ve) = vac1_plwh;
% 不命中的行 ve 保持原值
```

JAX 用 `jnp.where` **全量写**，未命中位置被覆盖为 0：
```python
ve = jnp.where(
    (vaccinated == 1) & (wk >= 2.0), vac1,
    jnp.where((vaccinated == 2) & (wk >= 6.0), vac2, 0.0)
)
# 未命中的人 ve 被显式设为 0
```

**教训**：翻译 MATLAB 的 logical-index 赋值时，要**显式想清楚"未命中位置该保持原值还是置默认值"**，二者在 MATLAB 里是同一个写法但语义不同。如果要保持原值，写成：
```python
ve_new = jnp.where(mask, vac1, state.ve)   # 保持原值
```

### A.4 输出 CSV 表头对齐

mpox JAX 的 `TALLY_OUTPUT_COLS` **逐字照抄** MATLAB 的 `tallyHeaders`（含空格、大小写、`|` 符号全保留：`"new infect|hiv"`, `"To Aware ls 40|hiv"`）。这样 `compare_to_matlab.py` 可以直接按列名查表，无需映射。

**通用原则**：**输出 schema 对齐是对账的前提**。如果你重新设计了列名，对账工具就得多写一层 mapping。直接照抄最省事。
