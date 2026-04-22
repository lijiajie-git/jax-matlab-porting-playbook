# TPU 多主机运维：从部署到性能调优

> 本文讲**JAX 代码上 TPU 的所有运维细节**——多主机初始化、scp/ssh 批处理、JIT 编译缓存、shard_map 切分、性能测量。踩过就知道、不踩不知道的点都写在这里。

---

## §1 多主机 TPU 的四条硬规则（违反就炸）

### 规则 1：`gcloud scp/ssh` 必须 `--batch-size=16`

**错误症状**：
```
ERROR: (gcloud.alpha.compute.tpus.tpu-vm.scp) FAILED_PRECONDITION:
  <worker-X>: slice not built yet
```

**原因**：默认 serial 模式会一个 worker 一个 worker 串行连接，慢且不稳定，常在 worker 4-15 某个位置报 `slice not built`。

**正确写法**：
```bash
SCP="gcloud alpha compute tpus tpu-vm scp \
     --project=$PROJECT --zone=$ZONE \
     --tunnel-through-iap --batch-size=16"
SSH="gcloud alpha compute tpus tpu-vm ssh $TPU_NAME \
     --project=$PROJECT --zone=$ZONE \
     --tunnel-through-iap --batch-size=16"
```

`--batch-size=16` 会让 gcloud 并行连接 16 个 worker，全起来。

### 规则 2：`jax.distributed.initialize()` 必须在任何 `jnp.*` 之前

**错误症状**：
```
RuntimeError: Backend 'tpu' failed to initialize: ...
  distributed client not yet connected
```

**原因**：一旦你在模块顶层写了 `jnp.array(...)` 或 `jax.devices()`，XLA backend 就会用**默认 single-host** 配置初始化。之后再 `distributed.initialize()` 切换为 multi-host 就晚了。

**正确写法**：
```python
# driver.py 顶部
import sys

def _is_multi_host() -> bool:
    """在 import 任何 JAX 东西前嗅探 --multi-host 标志。"""
    return "--multi-host" in sys.argv

if _is_multi_host():
    import jax
    jax.distributed.initialize()   # 必须在这里，第一件事

# 现在才能 import 其他 JAX 东西
import jax.numpy as jnp
from simulate import simulate
...
```

**检测清单**：
- [ ] 模块顶层没有 `jnp.array(...)`、`jnp.zeros(...)`、`jax.device_count()`、`jax.devices()`
- [ ] 所有 `import simulate`（如果 simulate.py 里有顶层 `jnp.*`）都在 `distributed.initialize()` 之后

### 规则 3：文件 I/O 只能 `process_index() == 0` 做

**错误症状**：16 个 host 同时写 `output.csv` → 内容混乱 / 文件损坏 / race condition。

**原因**：`shard_map` 里每个 host 都执行一份 Python 代码。文件 I/O 不受 shard 约束，会被所有 host 同时执行。

**正确写法**：
```python
is_leader = (jax.process_index() == 0)

if is_leader:
    os.makedirs(output_dir, exist_ok=True)
    np.save(output_path, tally_all)
    print(f"wrote {output_path}")
```

**什么时候所有 host 都要读**：每 host 本地的 npz 输入文件必须事先 scatter 到所有 worker，每个 host 本地 `load_inputs()` 读自己机器上的副本（见规则 4）。

### 规则 4：输入文件必须 scatter 到所有 16 个 worker

**错误症状**：只有 worker 0 能 load，其他 worker 在 `jnp.load(...)` 报 `FileNotFoundError`。

**原因**：TPU VM 每个 host 有独立文件系统，不共享。

**正确写法**（scatter 阶段）：
```bash
# 1. 预处理在 worker 0 上跑，结果写到 worker 0 的 /home/.../inputs.npz
$SSH --worker=0 --command="python3 prepare_inputs.py ..."

# 2. 从 worker 0 拉回本机，再 scatter 给 all
$SCP --worker=0 "$TPU:/home/.../inputs.npz" /tmp/inputs.npz
$SCP --worker=all /tmp/inputs.npz "$TPU:/home/.../inputs.npz"
```

这样每个 worker 本地都有一份 `inputs.npz`，`load_inputs()` 在所有 host 上都能跑。

---

## §2 四阶段部署脚本模板

通用的 `launch_tpu.sh` 骨架，改改 TPU 名字和路径直接用：

```bash
#!/usr/bin/env bash
set -eu

TPU_NAME="your-tpu-name"         # e.g. my-v5e-64-vm
PROJECT="your-gcp-project"
ZONE="your-zone"                  # e.g. europe-west4-b, us-central2-b
LOCAL_ROOT="/path/to/local"
REMOTE_ROOT="/home/$USER/project"

SCP="gcloud alpha compute tpus tpu-vm scp --project=$PROJECT --zone=$ZONE --tunnel-through-iap --batch-size=16"
SSH="gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --project=$PROJECT --zone=$ZONE --tunnel-through-iap --batch-size=16"

# 阶段 1: prep (首次) - 建目录 + 上传代码 + 预处理输入
cmd_prep() {
    $SSH --worker=all --command="mkdir -p $REMOTE_ROOT/code $REMOTE_ROOT/inputs"
    $SCP --worker=0 --recurse "$LOCAL_ROOT/code" "$TPU_NAME:$REMOTE_ROOT/"
    $SCP --worker=0 --recurse "$LOCAL_ROOT/inputs" "$TPU_NAME:$REMOTE_ROOT/"
    $SSH --worker=0 --command="cd $REMOTE_ROOT/code && python3 prepare_inputs.py ..."
    cmd_sync   # 预处理好的文件散发
}

# 阶段 2: sync - 只同步代码（快速迭代）+ 散发预处理好的 npz
cmd_sync() {
    tar czf /tmp/code.tgz -C "$LOCAL_ROOT" code
    $SCP --worker=all /tmp/code.tgz "$TPU_NAME:/tmp/code.tgz"
    $SSH --worker=all --command="tar xzf /tmp/code.tgz -C $REMOTE_ROOT"

    # scatter 预处理好的 npz
    $SCP --worker=0 "$TPU_NAME:$REMOTE_ROOT/inputs/main.npz" /tmp/main.npz
    $SCP --worker=all /tmp/main.npz "$TPU_NAME:$REMOTE_ROOT/inputs/main.npz"
}

# 阶段 3: smoke - 最小 iter 数 + tail -40 确认端到端通
cmd_smoke() {
    $SSH --worker=all --command="
      cd $REMOTE_ROOT/code && \
      export JAX_COMPILATION_CACHE_DIR=/home/$USER/.jax_cache && \
      mkdir -p \$JAX_COMPILATION_CACHE_DIR && \
      python3 driver.py --num-iter 64 --smoke --multi-host 2>&1 | tail -40
    "
}

# 阶段 4: run - 正式跑，tail 多一点留日志
cmd_run() {
    $SSH --worker=all --command="
      cd $REMOTE_ROOT/code && \
      export JAX_COMPILATION_CACHE_DIR=/home/$USER/.jax_cache && \
      mkdir -p \$JAX_COMPILATION_CACHE_DIR && \
      python3 driver.py --num-iter 256 --multi-host 2>&1 | tail -200
    "
}

case "${1:-}" in
    prep)  cmd_prep ;;
    sync)  cmd_sync ;;
    smoke) cmd_smoke ;;
    run)   cmd_run ;;
    *) echo "usage: $0 {prep|sync|smoke|run}" ; exit 1 ;;
esac
```

**工作流**：
1. 首次：`bash launch_tpu.sh prep`（完整流程）
2. 改代码：`bash launch_tpu.sh sync` → `bash launch_tpu.sh smoke` → `bash launch_tpu.sh run`

---

## §3 过滤 gcloud 噪音日志

每个 `gcloud ssh/scp` 都会打一堆 `WARNING: To increase the performance of the tunnel, consider installing NumPy...`，16 个 worker × 若干条 = 几十行 warning。过滤：

```bash
# 本机保留完整日志（便于调试），同时过滤 warning 到屏幕
bash launch_tpu.sh run 2>&1 | tee logs/run_$(date +%Y%m%d_%H%M).log | grep -v "NumPy\|tunnel"
```

或者在 `$SSH --command` 里把输出过滤：
```bash
$SSH --worker=all --command="... 2>&1 | grep -v 'NumPy\|tunnel'"
```

---

## §4 JAX 编译缓存 `JAX_COMPILATION_CACHE_DIR`

### 效果
- 首次编译：2+ 分钟（XLA lowering + optimization + kernel generation）
- 缓存命中：3-10 秒
- 跨 run 复用：第二天再跑也能命中（只要 JAX 版本、代码、输入 shape 都一致）

### 启用方法

```bash
export JAX_COMPILATION_CACHE_DIR=/home/$USER/.jax_cache
mkdir -p $JAX_COMPILATION_CACHE_DIR
python3 driver.py ...
```

把这两行写进 `launch_tpu.sh` 的每个 `cmd_*` 里，永远不会忘。

### 什么时候 cache miss

缓存 key 包含：JAX 版本 + 代码哈希 + 输入 shape/dtype + compilation flags。下面任何一个变就 miss：
- JAX / XLA 版本升级
- 代码改动（包括 docstring，XLA 对代码做 hash）
- 输入数组的 shape 或 dtype 改变
- 编译器 flag 改变

### 工程建议

**多 scenario 场景**：每个 scenario 的 constant arrays 如果 shape 不同（比如 scenario A 的 `foi_schedule` 长 85，scenario B 长 100）就会 re-compile。解决方法：
- 预先 pad 到同一 shape
- 或把 scenario_id 设为 `static_argnames` 让 JAX 一次编译所有 scenario（适合 scenario 少 + 逻辑同构的情况）

**陷阱**：即使 shape 和 dtype 都一致，如果 params dict 的 pytree 结构（键顺序 / 嵌套层级 / 是否含 `None`）略有不同，也会 cache miss。建议 params 构造路径完全一致，或做一层"标准化" pytree（固定键顺序、用 sentinel 替代 `None`）。

---

## §5 compile vs tpu_run 分离测时

### 原理

JAX dispatch 是**异步**的：
- Python 调用 `fn(...)` 返回时，**XLA compile + dispatch 完成**，但 TPU 还在算
- 返回的 `DeviceArray` 是一个 future（句柄），读它的值才会 block
- `tally.block_until_ready()` 强制等 TPU 算完

### 三段法模板

```python
import time

t0 = time.perf_counter()
tally = run_monte_carlo_sharded(state0, params, key, num_iter, T)
# 这里返回时：compile + dispatch 完成，TPU 还在跑
t_compile_dispatch = time.perf_counter() - t0

t1 = time.perf_counter()
tally.block_until_ready()                # 强制等 TPU
t_tpu_run = time.perf_counter() - t1

t2 = time.perf_counter()
tally_np = np.array(tally)               # gather 到 host + numpy 转换
np.save(output_path, tally_np)
t_post = time.perf_counter() - t2

print(f"compile+dispatch {t_compile_dispatch:.2f}s   "
      f"tpu_run {t_tpu_run:.2f}s   "
      f"gather+write {t_post:.2f}s   "
      f"total {t_compile_dispatch + t_tpu_run + t_post:.2f}s")
```

### 解读

- **`compile+dispatch` 很大**：编译瓶颈，考虑 `JAX_COMPILATION_CACHE_DIR` + reduce re-trace
- **`tpu_run` 很大**：计算瓶颈，考虑 per_chip_batch 加大、或模型有串行依赖
- **`gather+write` 很大**：I/O 瓶颈，考虑 tally 只传回 host 0、或写并行文件

---

## §6 `shard_map × vmap × scan` 三层并行

### Mesh + PartitionSpec 基础

```python
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
import numpy as np

devices = np.array(jax.devices())[:num_devices]
mesh = Mesh(devices, axis_names=("mc",))   # 一维 mesh，名字 "mc"

# PartitionSpec 语义：
#   P()      —— 不分片，复制到每 chip
#   P("mc")  —— 沿 "mc" 轴分片，每 chip 拿 1/num_devices
```

### 典型的三层堆叠

```python
@partial(shard_map,
         mesh=mesh,
         in_specs=(P(), P(), P("mc")),   # state/params 复制, keys 切
         out_specs=P("mc"),              # 输出沿 mc 拼接
         check_rep=False)
def fn(s0, p_dyn, k_shard):
    # 每 chip 本地拿到 per_chip = num_iter / num_devices 把 keys
    p = {**p_dyn, **params_static}
    simulate_v = jax.vmap(simulate, in_axes=(None, None, 0, None))
    # vmap 里跑 per_chip 条轨迹，每条轨迹的 simulate 内部是 lax.scan over T
    _, tally = simulate_v(s0, p, k_shard, T)
    return tally

# 调用
keys = jax.random.split(master_key, num_iter)  # shape (num_iter, 2)
tally = fn(state0, params_dyn, keys)           # shape (num_iter, T+1, C)
```

### 检查表

- [ ] `num_iter % num_devices == 0`
- [ ] `_split_params(params)` 已经把 python scalar 摘出来（见 `jax-patterns.md §12`）
- [ ] `check_rep=False`（否则 JAX 会检查输出复制一致性，某些情况下触发假警报）
- [ ] `in_specs` 和 `out_specs` 的维度和函数 signature 对得上

---

## §7 per_chip_batch 尺寸规划

### 一般流程

1. 固定 `num_devices`（比如 64 for v5e-64）
2. 从 `per_chip_batch = 1` 起跑，观察 `tpu_run` 时间
3. 翻倍加到 2、4、8... 每次测 `tpu_run`
4. 直到 tpu_run 不再近似线性增长（per_chip 翻倍 → tpu_run 也翻倍），说明 HBM 或计算饱和
5. 取**打满但还在饱和前**的那个值作为生产配置

### 实测判断

**线性 scaling**（per_chip=1 时 3.3s，per_chip=4 时 13.5s ≈ 4× 线性）说明：
- TPU MXU 还没打满
- HBM 还有余量
- 可以继续加 per_chip 提高总吞吐

**次线性**（per_chip 翻倍，tpu_run 涨 1.5× 以下）说明：
- 饱和了
- 继续加 per_chip 无益，反而会被内存限制

### `wk_steps_per_chip` 指标

一个直观的诊断量：
```
wk_steps_per_chip = per_chip_batch × T
```

log 里打出来，知道每 chip 本地跑了多少个 weekly step。配合 tpu_run 时间算单步开销：
```
single_step_ms = tpu_run_sec × 1000 / wk_steps_per_chip
```

TPU v5e 上 ~50ms/step 对典型 agent-based model（N~10^5）是很合理的数字。

---

## §8 `N_max` 静态槽位规划

### 原理

JAX jit 要静态 shape，所有 state 数组 pad 到 N_max。从 agent 数规划：

```
N_max ≥ N0 + T周 × 每周最大出生数 + 安全余量（30-50%）
```

取 **2 的幂**（`N_max = 2^k`）：TPU HBM 和 MXU 对 2 的幂形状更友好。

### 死亡不回收槽位 ⚠️

如果 "死亡" transition 只改 `alive = 0` 不改 `used = 0`：
- 死亡槽位变"僵尸"：`alive=0`, `used=1`
- 新生儿不会填进去（`_find_empty_slots` 只看 `used==0`）

所以 `N_max` 必须覆盖**累计出生数**，不是"同时存活最大数"。

### 计算例子

- N0 = 100,000（初始活人）
- T = 85 周
- 每周出生 ≈ 0.0003 × alive ≈ 30 人
- 累计出生 ≈ 85 × 30 = 2,550
- 峰值 used count ≈ 100,000 + 2,550 = 102,550
- 加 50% 余量 ≈ 154,000
- 取 2 的幂 → **N_max = 262,144 (2^18)**

### 如果要复用槽位（省内存）

需要在 "死亡" transition 里同时改 `used = 0`：
```python
state = update_field(state, "alive", death_mask, 0)
state = update_field(state, "used",  death_mask, 0)   # 允许复用
```

但这会让 "agent id 的稳定性" 消失（同一个 slot 可能先是 agent A 再是 agent B），影响 replay bundle 模式（因为 replay 按 slot index 对齐）。**除非内存真的紧张，保持僵尸槽更安全**。

---

## §9 常见 `shard_map` 和 pytree 的坑

### NamedTuple 里有字符串字段
默认 `shard_map` 把所有 leaf 升成 tracer，字符串会炸：
```
TypeError: not a valid JAX type: 'some_field_name'
```

解决：手动 `register_pytree_node`，字符串归为 `aux_data`（见 `jax-patterns.md §8`）。

### params 里有 python list of strings
比如 `params["scenario_names"] = ["S1", "S2", ...]`，shard_map 下也会炸。

解决：
- 能用 int 就别用 str（`scenario_id = 0..N-1`）
- 必须用 str 就走 `_split_params` 放静态那边

### `in_specs` 数量和函数 signature 对不上
```
ValueError: in_specs has N entries but function takes M arguments
```
仔细数 signature 里的参数，每个配一个 PartitionSpec。

---

## §10 调试命令速查

### 看单台 worker 的输出
```bash
$SSH --worker=0 --command="cd $REMOTE_ROOT/code && tail -50 /tmp/jax.log"
```

### 看所有 worker 的进程
```bash
$SSH --worker=all --command="pgrep -af python3"
```

### 杀掉所有 Python 进程（有时卡死）
```bash
$SSH --worker=all --command="pkill -9 python3 || true"
```

### 清编译缓存（cache 坏了时）
```bash
$SSH --worker=all --command="rm -rf /home/$USER/.jax_cache"
```

### 检查 TPU 是否健康
```bash
gcloud alpha compute tpus tpu-vm describe $TPU_NAME \
  --project=$PROJECT --zone=$ZONE
```

---

## Appendix: mpox 项目实测数字

> **新会话 AI 注意**：下面数字只在用户明确问"上一个项目实测多少"时才读。

### A.1 配置
- TPU: v5e-64 = 16 host × 4 chip = 64 chip
- N0 = 263,322, N_max = 524,288 (2^19)
- T = 85 weeks
- num_iter = 256, per_chip_batch = 4
- wk_steps_per_chip = 4 × 85 = 340

### A.2 per-scenario 计时
```
[dist] process 0/16  local_devices=4  global=64
[batch] path=shard_map  num_iter=256  global_chips=64  per_chip_batch=4
        T=85  N_max=524288  wk_steps_per_chip=340

=== Scenario S6 ===
  compile+dispatch 129.03s   tpu_run 13.52s   gather+write 1.77s   total 144.32s

=== Scenario S26 ===
  compile+dispatch 127.84s   tpu_run 12.54s   gather+write 1.43s   total 141.81s
```

### A.3 解读
- **compile 129s**：一次性开销，多 scenario 本应 cache hit 但因 shape 微差仍 re-trace
- **tpu_run 13.5s**：64 chip × 4 × 85 = 21,760 step，单步约 **~52ms / (chip·sample)**，MXU 接近饱和
- **gather+write 1.7s**：只 host 0 在写，瓶颈是 TPU→host 拷贝 `(256, 86, 98)` float32 ≈ 8MB
- **scaling 验证**：per_chip=1 时 tpu_run 3.3s，per_chip=4 时 13.5s，**近线性**，说明 4 还没到饱和

### A.4 优化路径（按 ROI 排序）
1. 消除 scenario 间 re-compile（预 pad constant arrays 到同 shape）→ 省 127s × (N_scenario - 1)
2. 加大 per_chip_batch（8 或 16）→ 如果仍线性，总吞吐再翻倍
3. host 0 gather 改异步 → 省 1-2s（ROI 低）

所以**主要优化目标应该是编译**，不是计算。
