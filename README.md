# MATLAB → JAX → TPU Porting Playbook

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![JAX](https://img.shields.io/badge/JAX-0.4+-blue.svg)](https://jax.readthedocs.io/)
[![TPU](https://img.shields.io/badge/TPU-v5e-orange.svg)](https://cloud.google.com/tpu)

A field-tested knowledge base for porting MATLAB time-stepped simulations
to JAX and running them on Google Cloud TPU pods.

Distilled from a real port: a 263,000-agent agent-based mpox transmission
model — 85 weeks × 26 intervention scenarios, single-threaded MATLAB to
JAX on TPU v5e-64 (64 chips across 16 hosts), achieving ~52 ms per
chip-sample with near-linear Monte Carlo scaling.

The four documents are structured for direct ingestion as persistent
context by a coding-assistant LLM (Claude Code, Cursor, etc.); they
also read fine as plain prose.

---

## What's inside

| File | Topic | ~Lines |
|---|---|---|
| **`jax-patterns.md`** | Translating MATLAB idioms into JAX primitives — `find()` masks, functional updates, static-shape sampling, transition tables, `lax.scan`, `vmap`, PRNG keys | 640 |
| **`tpu-ops.md`** | Multi-host TPU deployment — `gcloud` hard rules, `jax.distributed.initialize` ordering, persistent compile cache, `shard_map`, per-chip batch sizing, slot planning | 485 |
| **`validation.md`** | MATLAB ↔ JAX comparison methodology — three alignment layers (cell / distribution / production), replay-bundle bit-match, z-score ranking, exact-vs-correct dual paths, bisect workflow | 520 |

---

## Concrete examples from inside

**MATLAB `find(mask)` returns variable-length arrays. JAX requires static shapes.**

```python
# MATLAB
ids = find(alive==1 & pox_status==2);
state(ids, TREATMENT) = 1;          # variable-length scatter

# JAX
mask = (alive == 1) & (pox_status == 2)         # always shape (N,)
treatment = jnp.where(mask, 1, treatment)       # static, jit-able
```

**Births need to add new agents, but shapes can't grow under `jit`.**

Use the cumulative-sum-as-rank trick: mark empty slots with `1`s,
prefix-sum so each empty slot receives a unique rank `1, 2, 3, …`,
filter to `rank ≤ k` to pick the first `k`. No sort, no scatter,
purely dense compute.

**MATLAB and JAX produce different random sequences even with the
same seed. Bit-matching them requires a different approach.**

Don't share seeds. Dump MATLAB's actual *decisions* (which agent
IDs got selected each week), inject them into JAX directly via a
"replay bundle", bypassing JAX's RNG entirely. Once the injected
decisions reproduce MATLAB's state evolution, the deterministic
pipeline is validated.

---

## Decision tree

Find the right document and section by symptom.

| Symptom / question | Read | Section |
|---|---|---|
| MATLAB `find()` returns variable-length arrays — what's the JAX equivalent? | `jax-patterns.md` | §2 boolean mask |
| `state_matrix(mask, col) = val` errors under `jit` | `jax-patterns.md` | §2-3 functional update |
| Sample `k` items but `k` is only known at runtime | `jax-patterns.md` | §4 static upper bound + dynamic `k` |
| Sampling without replacement (`randperm`) under `jit` | `jax-patterns.md` | §5 `sample_up_to_k` |
| Births / imports need to add rows but shapes must be static | `jax-patterns.md` | §6 empty-slot allocation |
| Translating transition / rule tables (CSVs) into JAX | `jax-patterns.md` | §8 `TransitionSpec` |
| Writing the `for t = 1:T` weekly loop in JAX | `jax-patterns.md` | §9 `lax.scan` |
| Parallelizing Monte Carlo iterations | `jax-patterns.md` | §10 `vmap` |
| RNG management (MATLAB implicit global → JAX explicit keys) | `jax-patterns.md` | §11 PRNG |
| `gcloud scp/ssh` hangs or returns `FAILED_PRECONDITION` | `tpu-ops.md` | §1 hard rules |
| `jax.distributed.initialize()` order causes a crash | `tpu-ops.md` | §1 hard rules |
| Every JIT compile takes minutes | `tpu-ops.md` | §4 compile cache |
| Separating compile time from TPU runtime | `tpu-ops.md` | §5 `block_until_ready` |
| Sharding the MC axis across chips | `tpu-ops.md` | §6 `shard_map` |
| Choosing per-chip batch size | `tpu-ops.md` | §7 `per_chip_batch` |
| Sizing the static `N_max` slot pool | `tpu-ops.md` | §8 slot planning |
| Don't know how to compare with MATLAB after a run | `validation.md` | §1 three alignment layers |
| Want cell-level alignment with MATLAB | `validation.md` | §2 replay bundle |
| Can't tell Monte Carlo noise from systematic bias | `validation.md` | §3-4 z-score + diagnosis |
| Should the JAX port reproduce MATLAB bugs? | `validation.md` | §5 exact vs correct |
| Don't know where to start bisecting a discrepancy | `validation.md` | §6 bisect workflow |
| Just starting — what's the overall validation flow? | `validation.md` | §7 three-phase workflow |
| Both sides seed=42, why aren't results identical? | `validation.md` | §7.1 results vs random numbers |
| When is each phase "done"? | `validation.md` | §7.2 exit criteria |
| How many weeks to dump first? | `validation.md` | §7.4 progressive expansion |
| Is this discrepancy a bug or encoding mismatch? | `validation.md` | §7.5 encoding vs logic |

---

## Assumed model class

This playbook applies cleanly to:

- **Agent-based / compartmental / time-stepped simulations** — each step is a fixed update sequence
- **MATLAB source structured as** a main loop `for t = 1:T` + transition / rule tables + outer Monte Carlo replication
- **Targeting JAX on** single GPU / single TPU / multi-host TPU pod (v5e or v4)
- **Population sizes** ~10⁴ to 10⁶ that fit entirely in HBM

If your model is fundamentally different (PDE / continuous-time ODE / GNN / RL with external environment), the general patterns (static shapes, `scan`, `shard_map`) still apply, but specific examples will need translation.

---

## Out of scope

- Training neural networks (this is forward-only simulation, no learnable parameters)
- GPU-specific optimization (cuDNN, Triton kernels)
- AOT compilation API (`jax.jit(f).lower().compile()`)
- Distributed training with optimizer state sharding (`pjit`)
- Cross-cloud / cross-accelerator deployment (only Google Cloud TPU v5e)

---

## Reference: original project structure

The reference port (private codebase, not included here) had this layout. Appendix sections in each document reference numbers and code patterns from these files:

```
jax_port/
├── simulate.py             # Core dynamics + scan + shard_map
├── driver.py               # Parameter assembly + multi-host init
├── prepare_inputs.py       # MATLAB Excel/CSV → JAX-friendly npz
├── launch_tpu.sh           # Four-stage TPU deploy script
├── compare_to_matlab.py    # Statistical comparison
└── debug_tools/
    ├── build_matlab_replay_bundle.py
    └── compare_dumps.py
```

---

## License

MIT — see [LICENSE](LICENSE).
