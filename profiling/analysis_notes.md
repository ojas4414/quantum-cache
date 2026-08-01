# Profiling notes

Raw observations from `python profiling/profile_baseline.py`.
Machine: Windows 11, Python 3.14.3, torch 2.11.0+cpu, CPU-only.
Model: `batch=2, seq_len=16, d_model=64, n_heads=4, n_layers=2` — 10 forward passes.

Every number below is copied from an actual run. Re-running will shift the
absolute timings by tens of percent (see "Measurement noise" at the bottom);
the call counts and memory figures are stable.

---

## torch.profiler — operator breakdown

Sorted by CPU memory usage, top 10:

```
---------------------------  ------------  ------------  ------------  ------------  ---------
                       Name    Self CPU %      Self CPU   CPU total %     CPU total    CPU Mem   # Calls
---------------------------  ------------  ------------  ------------  ------------  ---------
               aten::linear         3.20%       1.547ms        29.29%      14.155ms    1.41 MB      120
               aten::matmul         4.35%       2.103ms        28.32%      13.686ms    1.41 MB      120
                aten::empty         1.73%     837.600us         1.73%     837.600us  970.00 KB      200
                aten::addmm         6.98%       3.375ms         8.25%       3.987ms  800.00 KB       40
                   aten::mm         7.29%       3.525ms         7.39%       3.570ms  640.00 KB       80
                aten::clone         2.16%       1.045ms         5.35%       2.585ms  640.00 KB       80
           aten::empty_like         0.69%     335.100us         1.44%     693.600us  640.00 KB       80
                 aten::relu         1.09%     527.100us         1.92%     926.200us  640.00 KB       20
            aten::clamp_min         0.83%     399.100us         0.83%     399.100us  640.00 KB       20
              aten::reshape         2.61%       1.260ms         8.35%       4.034ms  480.00 KB      160
---------------------------  ------------  ------------  ------------  ------------  ---------
Self CPU time total: 48.332ms
```

Observations:

- **`aten::linear` and `aten::matmul` each report 1.41 MB.** These are the same
  allocations seen from two levels of the stack — `linear` dispatches into
  `matmul` — not 2.82 MB in total.
- **`aten::linear` 29.3% / `aten::matmul` 28.3% of CPU total.** The projections
  and the attention matmuls dominate, which is expected: at `d_model=64` this is
  a tiny model and nearly all real work is GEMM.
- **`aten::clone` — 80 calls, 640 KB.** This is the materialization cost of
  `.contiguous()` in `transformer.py`, plus the copies `.reshape()` triggers
  when its input is non-contiguous. At 5.35% of CPU total it is a real cost, but
  not a dominant one.

## cProfile — Python-level call counts

```
4426 function calls (3776 primitive calls) in 0.024 seconds
Ordered by: cumulative time

ncalls  tottime  cumtime  filename:lineno(function)
     1    0.002    0.023  profiling/profile_baseline.py:11(run)
    10    0.000    0.019  src/transformer.py:87(forward)     <- Transformer
    20    0.001    0.018  src/transformer.py:74(forward)     <- Transformer_block
    20    0.001    0.011  src/transformer.py:30(forward)     <- Multihead
   120    0.006    0.006  {built-in method torch._C._nn.linear}
    40    0.003    0.003  {built-in method torch.matmul}
    40    0.000    0.002  torch/nn/functional.py:2914(layer_norm)
```

Observations:

- **`torch.matmul`: 40 direct calls** — 10 passes × 2 layers × 2 matmuls
  (`Q·Kᵀ` and `weights·V`). The 120 `linear` calls are 10 × 2 × (3 projections +
  output projection + 2 feed-forward layers).
- **`matmul` is ~12.5% of `tottime`**, `linear` ~25%. These percentages disagree
  with the torch.profiler table above because cProfile attributes only Python
  frame time while torch.profiler attributes dispatched ATen work. Both are
  correct measurements of different things.
- **Neither `contiguous` nor `transpose` appears in the top 15 by cumulative
  time.** They are C-level calls that cProfile does not surface as separate
  Python frames; their real cost shows up as `aten::clone` in the table above.

## What this did and did not establish

The hypothesis was that storing the KV cache as `(batch, seq, heads, d_head)`
rather than `(batch, heads, seq, d_head)` puts consecutive tokens in adjacent
cache lines, and should therefore reduce stalls on the attention reads.

The profiling **did** establish that:

- attention and projection GEMMs dominate runtime, and
- the transpose-then-materialize pattern costs a real but modest ~5%
  (`aten::clone`).

The profiling **did not** establish that CPU cache misses were the bottleneck.
Showing that requires hardware PMU counters (`perf stat -e cache-misses` or
equivalent), which were never collected — this is a Windows machine with no
`perf` available. Any "N cache misses per pass" figure derived here would be
arithmetic from the stride layout, not a measurement.

## Measurement noise

At this model size one forward pass is ~0.5 ms, and run-to-run variance on this
machine is large: repeated 100-run measurements of the *same* model produced
medians ranging from 0.35 ms to 1.04 ms.

Any A/B comparison at this scale therefore needs interleaved,
order-counterbalanced trials and a paired statistic. A single sequential
"measure A, then measure B" run can easily show a 20-30% difference that is
entirely noise plus ordering effects. See the Results section of the README for
the paired measurement and its conclusion.
