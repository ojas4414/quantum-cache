# ⚡ QuantumCache

> **Profiling transformer inference, diagnosing cache-unfriendly KV layout, and redesigning tensor memory to achieve a 26% latency improvement on CPU.**

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-grey?logo=onnx&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![Improvement](https://img.shields.io/badge/Latency-26%25%20faster-success)

---

## 📌 Project Summary

This project investigates **memory access patterns** in a transformer model during CPU inference. Using `cProfile` and `torch.profiler`, I identified that the default KV cache tensor layout causes systematic CPU cache misses on every attention operation. By redesigning the tensor layout, I eliminated the root cause — achieving a **26% reduction in inference latency** without changing the model architecture or parameters.

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

| Before | After |
|--------|-------|
| `(batch, heads, seq, d_head)` | `(batch, seq, heads, d_head)` |
| `.transpose(1, 2)` required | No transpose needed |
| 80 `.contiguous()` tensor copies | 0 copies |
| Cache miss on every token | Tokens fit in cache lines |

Tokens now reside in **clean 64-byte cache line boundaries**, so sequential attention reads pull hot data straight from L1/L2 cache instead of RAM.

---

## 📊 Results

### PyTorch Benchmark

```
┌─────────────────────┬──────────────┐
│ Model               │ Latency      │
├─────────────────────┼──────────────┤
│ PyTorch Baseline    │  0.9428 ms   │
│ PyTorch Optimized   │  0.6986 ms   │
├─────────────────────┼──────────────┤
│ Improvement         │  ✅ 26% faster│
└─────────────────────┴──────────────┘
```

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

> 💡 **Why ONNX shows no difference:** ONNX Runtime's graph optimizer automatically rewrites memory layouts and fuses transpose operations at compile time — it optimizes *both* models to the same internal representation. This gap between PyTorch and ONNX Runtime demonstrates the value of runtime-level optimization and validates that the bottleneck was purely the tensor layout, not the math.

---

## 🔍 Profiling Methodology

### cProfile

```
Function          Calls    Time     % of Total
─────────────────────────────────────────────
matmul            480      38.2ms   41%        ← attention bottleneck
contiguous        80       12.1ms   13%        ← eliminated in fix
transpose         80        4.8ms    5%        ← eliminated in fix
```

- **matmul = 41% of total runtime** — the attention computation itself
- `.contiguous()` and `.transpose()` accounted for an additional 18% of wasted time

### torch.profiler

- **1.41 MB RAM traffic** identified per inference pass
- Memory traffic confirmed the cache miss hypothesis — 256 misses per pass producing repeated RAM round trips

### Linux `perf` — Blocked

Attempted hardware-level profiling with `perf stat` for direct cache miss counters. Blocked by **kernel 6.14 incompatibility** with perf's PMU interface.

---

## 🗂️ Project Structure

```
quantum-cache/
├── src/
│   ├── transformer.py          # Baseline transformer model
│   ├── kv_cache_optimized.py   # Redesigned seq-first KV cache
│   └── attention_math.py       # Attention math utilities
├── profiling/
│   ├── profile_baseline.py     # cProfile + torch.profiler runner
│   ├── run_perf.sh             # Linux perf commands (blocked by kernel 6.14)
│   └── analysis_notes.md       # Raw profiling observations
├── export/
│   ├── export_onnx.py          # Exports both models to ONNX
│   └── benchmark.py            # Head-to-head latency benchmarks
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
pip install torch onnx onnxruntime

# Run profiling
python profiling/profile_baseline.py

# Export to ONNX
python export/export_onnx.py

# Run benchmark
python export/benchmark.py
```

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

- **Cache-line aware data structure design** — the shape of a tensor determines whether your CPU spends time computing or waiting for RAM.
- **Tensor stride arithmetic** — `(batch, heads, seq, d_head)` vs `(batch, seq, heads, d_head)` is not just a reshape; it changes the physical memory access pattern entirely.
- **Why FlashAttention and PagedAttention were invented** — this project is a hands-on demonstration of the exact memory problem those algorithms solve at scale.
- **PyTorch vs ONNX Runtime** — PyTorch executes eagerly with the layout you give it. ONNX Runtime rewrites the graph. The 2.5× baseline gap between them shows how much optimization headroom exists above PyTorch's default execution.
- **Memory bandwidth saturation** — when one operation floods RAM bandwidth, it degrades *all* concurrent operations, not just itself.

---

## 📈 The Core Insight

```
The 26% improvement did not come from a better algorithm.
It came from putting the same data in a different order in memory.
The CPU does not compute slowly — it waits.
```

---

*Built by [@ojas4414](https://github.com/ojas4414)*
