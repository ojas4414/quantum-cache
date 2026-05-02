import torch
import time
import sys
sys.path.append('src')

from transformer import Transformer, batch_size, Seq_len, D_model
from kv_cache_optimized import OptimizedTransformer


def benchmark(model,x,runs=100):

    for _ in range(10):
        model(x)

    start=  time.perf_counter()

    for _ in range(runs):
        model(x)
    end = time.perf_counter()

    avg_ms = ((end-start)/runs)*1000
    return avg_ms
if __name__ == "__main__":
    x = torch.randn(batch_size, Seq_len, D_model)

    baseline  = Transformer()
    optimized = OptimizedTransformer()

    baseline_ms  = benchmark(baseline, x)
    optimized_ms = benchmark(optimized, x)

    improvement = ((baseline_ms - optimized_ms) / baseline_ms) * 100

    print(f"Baseline  latency: {baseline_ms:.4f} ms")
    print(f"Optimized latency: {optimized_ms:.4f} ms")
    print(f"Improvement:       {improvement:.1f}%")