import cProfile
import pstats
import torch
import sys
sys.path.append('src')

from transformer import Transformer,batch_size,Seq_len,D_model
from torch.profiler import profile, record_function, ProfilerActivity
from kv_cache_optimized import OptimizedTransformer

def run():
    model=Transformer()
    x =  torch.randn(batch_size,Seq_len,D_model)

    for _ in range(10):
        out= model(x)
    return out


def run_torch_profile():
    model = Transformer()
    x  = torch.randn(batch_size,Seq_len,D_model)

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True
    ) as prof:
        with record_function("model_inference"):
            for _ in range(10):
                out=model(x)
    
    print(prof.key_averages().table(
        sort_by="cpu_memory_usage",
        row_limit=10
    ))

if __name__ == "__main__":
    run_torch_profile()

    profiler = cProfile.Profile()
    profiler.enable()

    run()

    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(15)