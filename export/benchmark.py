"""
Paired A/B benchmark: head-first vs seq-first KV cache layout.

At this model size a forward pass is ~0.5 ms and run-to-run variance on a
typical laptop is tens of percent. Measuring A once, then B once -- the obvious
approach, and what this script used to do -- reports differences of +20% to +34%
purely from ordering and drift. That is how the 26% figure this project
originally claimed came about.

So this harness does four things instead:

  1. Copies the baseline's weights into the seq-first model, so the two are
     numerically identical and only the cache layout differs.
  2. Interleaves the trials, alternating which model is timed first, so any
     warmup or thermal drift hits both equally.
  3. Runs many trials and reports the MEDIAN, not a single mean.
  4. Reports the PAIRED per-trial delta with its interquartile range and a
     win count, so the spread is visible rather than hidden behind one number.

Run from the repository root:  python export/benchmark.py
"""
import sys
import time
import statistics as st

import torch

sys.path.append('src')

from transformer import Transformer, batch_size, Seq_len, D_model
from kv_cache_optimized import OptimizedTransformer

TRIALS = 40
RUNS_PER_TRIAL = 100
WARMUP = 20


def paired_models():
    """Two models, same weights, different KV cache layout.

    Without this the comparison is confounded: each model would be randomly
    initialized separately, so they would not even be computing the same
    function.
    """
    baseline = Transformer()
    seq_first = OptimizedTransformer()

    for b_block, s_block in zip(baseline.blocks, seq_first.blocks):
        s_block.attn.W_q.load_state_dict(b_block.atten.W_q.state_dict())
        s_block.attn.W_k.load_state_dict(b_block.atten.W_k.state_dict())
        s_block.attn.W_v.load_state_dict(b_block.atten.W_v.state_dict())
        s_block.attn.W_o.load_state_dict(b_block.atten.o.state_dict())
        s_block.ff[0].load_state_dict(b_block.ff.net[0].state_dict())
        s_block.ff[2].load_state_dict(b_block.ff.net[2].state_dict())
        s_block.ln1.load_state_dict(b_block.norm1.state_dict())
        s_block.ln2.load_state_dict(b_block.norm2.state_dict())

    baseline.eval()
    seq_first.eval()
    return baseline, seq_first


def assert_equivalent(baseline, seq_first, x):
    """Fail loudly if the two models do not compute the same function.

    A latency comparison between two models that disagree numerically is
    meaningless. This repo shipped exactly that bug once (the seq-first
    attention multiplied by K instead of V), so the check is enforced here
    rather than left to the reader.
    """
    with torch.no_grad():
        out_b = baseline(x)
        out_s = seq_first(x)
    max_diff = float((out_b - out_s).abs().max())
    if not torch.allclose(out_b, out_s, atol=1e-6):
        raise SystemExit(
            f"Models are not equivalent (max |diff| = {max_diff:.3e}). "
            "Benchmarking them against each other would be meaningless."
        )
    return max_diff


def time_model(model, x, runs=RUNS_PER_TRIAL):
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
        start = time.perf_counter()
        for _ in range(runs):
            model(x)
        end = time.perf_counter()
    return ((end - start) / runs) * 1000


def main():
    torch.manual_seed(0)
    # One thread: the model is small enough that thread-pool scheduling noise
    # is comparable to the effect being measured.
    torch.set_num_threads(1)

    x = torch.randn(batch_size, Seq_len, D_model)
    baseline, seq_first = paired_models()

    max_diff = assert_equivalent(baseline, seq_first, x)
    print(f"Equivalence check: max |baseline - seq_first| = {max_diff:.3e}  OK\n")

    print(f"Layouts under test")
    print(f"  baseline  K: {tuple(baseline.kv_cache[0][0].shape)}  "
          f"strides {baseline.kv_cache[0][0].stride()}")
    print(f"  seq-first K: {tuple(seq_first.kv_cache[0][0].shape)}  "
          f"strides {seq_first.kv_cache[0][0].stride()}\n")

    print(f"Running {TRIALS} interleaved trials x {RUNS_PER_TRIAL} runs "
          f"(alternating order)...\n")

    base_times, seq_times = [], []
    for trial in range(TRIALS):
        if trial % 2 == 0:
            base_times.append(time_model(baseline, x))
            seq_times.append(time_model(seq_first, x))
        else:
            seq_times.append(time_model(seq_first, x))
            base_times.append(time_model(baseline, x))

    med_base = st.median(base_times)
    med_seq = st.median(seq_times)

    # Paired per-trial deltas are robust to drift across the run in a way that
    # comparing the two medians is not.
    deltas = sorted((b - s) / b * 100 for b, s in zip(base_times, seq_times))
    q1 = deltas[len(deltas) // 4]
    q3 = deltas[3 * len(deltas) // 4]
    wins = sum(1 for d in deltas if d > 0)

    print(f"  baseline   median {med_base:.4f} ms   "
          f"min {min(base_times):.4f}   max {max(base_times):.4f}")
    print(f"  seq-first  median {med_seq:.4f} ms   "
          f"min {min(seq_times):.4f}   max {max(seq_times):.4f}")
    print(f"  median delta      {(med_base - med_seq) / med_base * 100:+.1f}%  "
          f"(positive = seq-first faster)")
    print(f"  paired delta      median {st.median(deltas):+.1f}%   "
          f"IQR [{q1:+.1f}%, {q3:+.1f}%]")
    print(f"  seq-first won     {wins}/{len(deltas)} trials\n")

    # Interpretation. The IQR straddling zero, or a win rate near half, both
    # mean the difference is not distinguishable from noise on this machine.
    if q1 < 0 < q3 or 0.35 <= wins / len(deltas) <= 0.65:
        print("  Verdict: NO MEASURABLE DIFFERENCE. The interquartile range "
              "straddles zero\n           and/or the win rate is near chance. "
              "Any single-number\n           'improvement' quoted from this "
              "data would be noise.")
    elif st.median(deltas) > 0:
        print(f"  Verdict: seq-first is faster by ~{st.median(deltas):.1f}% "
              "(median paired delta),\n           consistently across trials.")
    else:
        print(f"  Verdict: seq-first is SLOWER by ~{abs(st.median(deltas)):.1f}% "
              "(median paired delta),\n           consistently across trials.")


if __name__ == "__main__":
    main()
