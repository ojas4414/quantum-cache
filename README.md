# ⚡ QuantumCache

> **Profiling transformer inference, rebuilding the KV cache with a seq-first tensor layout, and measuring — carefully — whether it actually helps on CPU. At this model size, it does not.**

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-grey?logo=onnx&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Result](https://img.shields.io/badge/Result-no%20measurable%20change-lightgrey)

---

## 📌 Project Summary

This project investigates **memory access patterns** in a transformer model during CPU inference. Using `cProfile` and `torch.profiler` I looked at whether the default KV cache layout `(batch, heads, seq, d_head)` — which places consecutive tokens 256 bytes apart — is costing measurable time on the attention reads, then implemented a seq-first alternative and benchmarked the two against each other.

**The honest result: at this model size the layout change makes no measurable difference.** A paired, order-counterbalanced benchmark puts the two within ~3% of each other, with the seq-first version winning about half the trials — i.e. noise. The write-up below keeps the reasoning that motivated the change, the profiling that informed it, and the measurement that failed to support it.

> **What this repo previously claimed, and why it was wrong.** An earlier version
> of this README reported a **26% latency improvement**. That number was an
> artifact of two bugs, both now fixed:
>
> 1. `Transformer.forward` appended to `self.kv_cache` and never cleared it, so
>    across a 100-run benchmark the baseline accumulated hundreds of retained
>    tensors. It was being penalised for unbounded list growth, not for its
>    tensor layout. Fixing only this line reversed the sign of the result.
> 2. The optimized attention computed `weights @ K` instead of `weights @ V`, so
>    the two models were not computing the same function and were never
>    comparable in the first place.
>
> Both are fixed, both models now produce bit-identical output given identical
> weights, and the benchmark below is the re-measured result.

---

## 🧠 The Problem — Cache-Unfriendly KV Layout

### Default KV Cache Shape: `(batch, heads, seq, d_head)`

```
Memory Layout (default):
┌─────────────────────────────────────────────────────┐
│  Token 0  │  Token 0  │ ... │  Token 0  │           │
│  Head 0   │  Head 1   │     │  Head H   │           │
├─────────────────────────────────────────────────────┤
│  Token 1  │  Token 1  │ ... │  Token 1  │           │  ← 256 bytes away
│  Head 0   │  Head 1   │     │  Head H   │           │
└─────────────────────────────────────────────────────┘

CPU Cache Line = 64 bytes = 16 floats
Seq stride    = 64        → tokens are 256 bytes apart in memory
```

### Why This Destroys Performance

When attention computes `Q · Kᵀ`, it reads the K matrix **16 times per forward pass** across the sequence dimension. But in the default layout, consecutive tokens are **256 bytes apart** — 4× the size of a CPU cache line (64 bytes).

```
Every token access = cache miss = round trip to RAM
16 reads × 16 tokens = 256 cache misses per single inference
```

This means the CPU spends most of its time waiting for RAM instead of computing — a classic **memory bandwidth saturation** problem.

---

## 🔧 The Fix — Seq-First Tensor Layout

### Redesigned KV Cache Shape: `(batch, seq, heads, d_head)`

```
Memory Layout (optimized):
┌──────────────────────────────────────────────────────┐
│  Token 0  │  Token 1  │  Token 2  │  Token 3  │ ... │
│  All Heads│  All Heads│  All Heads│  All Heads│     │
└──────────────────────────────────────────────────────┘

Consecutive tokens now sit in adjacent cache lines ✅
```

### What Changed in the Code

| | Baseline (`transformer.py`) | Seq-first (`kv_cache_optimized.py`) |
|---|---|---|
| **Stored KV cache layout** | `(batch, heads, seq, d_head)` | `(batch, seq, heads, d_head)` |
| **Transpose at store time** | `.transpose(1, 2)` on Q, K, V before caching | none — K and V are cached as projected |
| **Transpose at compute time** | reuses the already-transposed tensors | `.transpose(1, 2)` re-applied per matmul |
| **Materialization** | `.transpose(1,2).contiguous()` then `.view()` | `.transpose(1,2)` then `.reshape()` |

Two corrections to how this was described previously:

- **The seq-first version does not remove transposes — it moves them.** It calls
  `.transpose(1, 2)` four times in `forward` (lines 38, 39, 44, 45) because
  attention still needs a head-major view to do the matmuls. What changes is
  *what gets stored* in the cache, not how many transposes execute.
- **It does not eliminate copies either.** `.reshape()` on line 46 follows a
  `.transpose()`, so its input is non-contiguous and it copies — the same work
  `.contiguous()` does explicitly in the baseline. `torch.profiler` shows
  `aten::clone` firing in both.

The intended benefit is narrower than "no transposes, no copies": tokens are
contiguous in the *stored* cache, so a later kernel reading the cache
sequentially would touch adjacent cache lines. That benefit is real in principle
and is why FlashAttention/PagedAttention care about layout — but at
`d_model=64, seq_len=16` it is far too small to measure, which is what the
benchmark below shows.

---

## 📊 Results

### PyTorch Benchmark

Measured with both models loaded with **identical weights**, 40 interleaved
trials of 100 runs each, alternating which model is timed first, single-threaded,
reporting the median and the paired per-trial delta. Two independent executions
of the same harness:

```
run 1   baseline median 0.5671 ms   seq-first median 0.5844 ms   ->  -3.0%
run 2   baseline median 0.5271 ms   seq-first median 0.5289 ms   ->  -0.3%
        (negative = seq-first slower)

paired per-trial delta   run 1: median -0.9%,  IQR [-14.8%, +11.3%]
                         run 2: median +3.6%,  IQR [ -9.5%, +13.2%]
seq-first won            run 1: 19/40 trials    run 2: 22/40 trials
```

**Verdict: no measurable difference.** The medians differ by less than the
run-to-run spread, the interquartile range straddles zero in both executions, and
the seq-first layout wins roughly half the trials. This is what a null result
looks like.

This harness **is** `export/benchmark.py` — run `python export/benchmark.py` to
reproduce it. It asserts numerical equivalence before timing anything and
refuses to report a result if the two models disagree, then prints the verdict
itself rather than leaving a raw percentage to be quoted out of context.

> ⚠️ **Why the median delta alone is not the answer.** Three consecutive runs of
> the harness on this machine produced median deltas of **−1.5%, +4.8%, and
> +22.3%** — with win rates of 21/40, 23/40, and 26/40 and an IQR straddling zero
> every time. The harness returned *no measurable difference* on all three. That
> spread is the whole point: a single number from this data, in either direction,
> would be noise.

### ONNX Runtime Benchmark

```
┌─────────────────────┬──────────────┐
│ Model               │ Latency      │
├─────────────────────┼──────────────┤
│ ONNX Baseline       │  0.3795 ms   │
│ ONNX Optimized      │  0.3846 ms   │
├─────────────────────┼──────────────┤
│ Delta               │  ~0% (equal) │
└─────────────────────┴──────────────┘
```

> ⚠️ **These ONNX numbers are stale and have not been re-measured.** They were
> captured before the `weights @ K` → `weights @ V` fix, so the exported
> "optimized" graph was computing a different function than the one in the repo
> today. Re-running `python export/export_onnx.py` regenerates both graphs and
> re-benchmarks them; the numbers above should be replaced with that output.
>
> The original interpretation — that ONNX Runtime's graph optimizer rewrites
> memory layouts and fuses transposes at compile time, so both models converge to
> the same internal representation — remains the plausible explanation for why
> the two ONNX timings match. But with the PyTorch comparison now showing no
> difference either, ONNX Runtime showing no difference is no longer evidence of
> anything in particular.

---

## 🔍 Profiling Methodology

Full captured output is in [`profiling/analysis_notes.md`](profiling/analysis_notes.md).
Everything below is copied from an actual run of `profiling/profile_baseline.py`.

### cProfile — Python-level

```
4426 function calls (3776 primitive calls) in 0.024 seconds

ncalls  tottime  cumtime  function
   120    0.006    0.006  torch._C._nn.linear
    40    0.003    0.003  torch.matmul
    40    0.000    0.002  torch.nn.functional.layer_norm
```

- **`torch.matmul`: 40 calls**, ~12.5% of `tottime` — 10 passes × 2 layers × 2
  matmuls (`Q·Kᵀ` and `weights·V`).
- **`linear`: 120 calls**, ~25% — projections plus the feed-forward layers.
- **`contiguous` and `transpose` do not appear in the top 15.** They are C-level
  calls that cProfile does not surface as separate Python frames; their cost
  shows up as `aten::clone` under torch.profiler instead.

### torch.profiler — ATen-level

```
Name             Self CPU %   CPU total %   CPU Mem    # Calls
aten::linear          3.20%        29.29%   1.41 MB       120
aten::matmul          4.35%        28.32%   1.41 MB       120
aten::clone           2.16%         5.35%  640.00 KB       80
```

- **1.41 MB** allocated per the `linear`/`matmul` path. (`linear` dispatches into
  `matmul`, so this is the same memory counted at two stack depths — not 2.82 MB.)
- **GEMMs dominate**: `linear` + `matmul` are ~29% of CPU total each. At
  `d_model=64` almost all real work is matrix multiply.
- **`aten::clone`: 80 calls, 5.35%** — the actual cost of the
  transpose-then-materialize pattern. Real, but modest, and present in *both*
  models.

### What the profiling did not show

It did **not** demonstrate that CPU cache misses were the bottleneck. That claim
needs hardware PMU counters (`perf stat -e cache-misses` or equivalent), and none
were collected — this is a Windows machine with no `perf` available. A previous
version of this README reported "256 cache misses per pass"; that figure was
arithmetic derived from the stride layout, not a measurement, and has been
removed along with the empty `run_perf.sh` that implied it had been run.

---

## 🗂️ Project Structure

```
quantum-cache/
├── src/
│   ├── transformer.py          # Baseline transformer (head-first KV cache)
│   └── kv_cache_optimized.py   # Seq-first KV cache variant
├── profiling/
│   ├── profile_baseline.py     # cProfile + torch.profiler runner
│   └── analysis_notes.md       # Captured profiling output + interpretation
├── export/
│   ├── export_onnx.py          # Exports both models to ONNX, then benchmarks them
│   └── benchmark.py            # Paired, order-counterbalanced A/B latency harness
├── requirements.txt
└── .gitignore
```

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/ojas4414/quantum-cache.git
cd quantum-cache

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run profiling
python profiling/profile_baseline.py

# Run the PyTorch benchmark
python export/benchmark.py

# Export both models to ONNX, then benchmark them under onnxruntime
python export/export_onnx.py
```

All four commands are run **from the repository root** — the scripts use
`sys.path.append('src')` and relative `export/*.onnx` paths.

> **Note:** the `.onnx` files are gitignored, so `export/export_onnx.py` builds
> them on first run. `export/benchmark.py` (PyTorch-only) does not depend on them
> and can be run first.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| ![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=pytorch&logoColor=white) | Model implementation & baseline |
| ![ONNX](https://img.shields.io/badge/-ONNX-grey?logo=onnx) | Model export & cross-runtime comparison |
| `onnxruntime` | Optimized inference runtime |
| `cProfile` | Function-level Python profiling |
| `torch.profiler` | Memory traffic & operator profiling |
| `perf` | (Attempted) Hardware PMU cache miss counters |

---

## 💡 Key Takeaways

- **Tensor stride arithmetic** — `(batch, heads, seq, d_head)` vs
  `(batch, seq, heads, d_head)` is not just a reshape; it changes the physical
  memory access pattern. Reasoning about which one puts consecutive tokens in
  adjacent cache lines is the core exercise here, and it holds up.
- **A layout change has to be measured, not assumed.** The reasoning above
  predicts a win. The measurement says there isn't one at this size. The
  reasoning being sound is not evidence that the effect is present.
- **Benchmark methodology dominates small effects.** Measuring A then B once
  each reported +26%; measuring them interleaved and order-counterbalanced with
  identical weights reported ~0%. At sub-millisecond scale the harness design
  mattered more than anything in the model.
- **Two bugs can look like a result.** An uncleared cache list and a `K`-for-`V`
  substitution combined into a clean, plausible, entirely fictitious 26%. Both
  were invisible in the output and only showed up under a gradient check and a
  controlled A/B.
- **Why FlashAttention and PagedAttention were invented** — they attack this
  memory problem at a scale where it genuinely dominates. This model
  (`d_model=64, seq_len=16`, ~0.5 ms/pass) is nowhere near that regime, which is
  the most likely reason the effect is unmeasurable here.

---

## 📈 The Core Insight

```
The layout reasoning was right. The measurement still said no.
A 26% "improvement" turned out to be an uncleared list and a typo'd tensor.
Measure the thing you claim, with a harness that can survive being wrong.
```

---

## 🚧 Known Gaps

- The ONNX latency numbers predate the `K`→`V` fix and need regenerating via
  `python export/export_onnx.py`.
- No hardware cache-miss counters were ever collected, so the cache-line
  hypothesis remains untested at the hardware level rather than disproven.
- The null result is specific to this model size. Whether the layout matters at
  realistic `seq_len` and `d_model` is untested and is the obvious next step.

---

*Built by [@ojas4414](https://github.com/ojas4414)*
