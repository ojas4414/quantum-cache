import torch
import sys
sys.path.append('src')
import onnxruntime as ort
import numpy as np
import time


from transformer import Transformer,batch_size, Seq_len, D_model
from kv_cache_optimized import OptimizedTransformer


def export(model,filename):
    model.eval()
    x = torch.randn(batch_size, Seq_len, D_model)


    torch.onnx.export(
        model,
        x,
        filename,
        opset_version=18,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input':{0:'batch_size'},
            'output':{0:'batch_size'}
        }



    )

    print(f"Exported: {filename}")



def benchmark_onnx(filename, runs=100):
    session = ort.InferenceSession(filename)
    x = np.random.randn(batch_size, Seq_len, D_model).astype(np.float32)
    
    
    for _ in range(10):
        session.run(None, {'input': x})
    
    start = time.perf_counter()
    for _ in range(runs):
        session.run(None, {'input': x})
    end = time.perf_counter()
    
    return ((end - start) / runs) * 1000

if __name__ == "__main__":
    baseline  = Transformer()
    optimized = OptimizedTransformer()

    export(baseline,  "export/baseline.onnx")
    export(optimized, "export/optimized.onnx")

    print("Both models exported successfully")

    # Benchmarking has to happen after the export above -- the .onnx files are
    # gitignored, so on a fresh clone they do not exist until this script
    # writes them.
    baseline_ms  = benchmark_onnx("export/baseline.onnx")
    optimized_ms = benchmark_onnx("export/optimized.onnx")
    improvement  = ((baseline_ms - optimized_ms) / baseline_ms) * 100

    print(f"\n--- ONNX Inference ---")
    print(f"Baseline  latency: {baseline_ms:.4f} ms")
    print(f"Optimized latency: {optimized_ms:.4f} ms")
    print(f"Improvement:       {improvement:.1f}%")