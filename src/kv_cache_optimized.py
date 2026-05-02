import torch
import torch.nn as nn
import math
import sys
sys.path.append('src')

from transformer import batch_size, Seq_len, D_model, N_Heads, N_layers



class OptimizedAttention(nn.Module):
    def __init__(self,d_model,n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.d_model = d_model

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, S, D = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        
        Q = Q.view(B, S, self.n_heads, self.d_head)  
        K = K.view(B, S, self.n_heads, self.d_head)
        V = V.view(B, S, self.n_heads, self.d_head)  

        
        scale  = math.sqrt(self.d_head)
        scores = torch.matmul(
            Q.transpose(1, 2),
            K.transpose(1, 2).transpose(-2, -1)
        ) / scale                                     
        weights = torch.softmax(scores, dim=-1)

        
        out = torch.matmul(weights, K.transpose(1, 2))  
        out = out.transpose(1, 2)                       
        out = out.reshape(B, S, D)

        return self.W_o(out), K, V
class OptimizedTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.attn = OptimizedAttention(d_model, n_heads)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, K, V = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x, K, V


class OptimizedTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks   = nn.ModuleList([
            OptimizedTransformerBlock(D_model, N_Heads) for _ in range(N_layers)
        ])
        self.kv_cache = []

    def forward(self, x):
        self.kv_cache = []
        for block in self.blocks:
            x, K, V = block(x)
            self.kv_cache.append((K, V))
        return x

if __name__ == "__main__":
    model = OptimizedTransformer()
    x     = torch.randn(batch_size, Seq_len, D_model)
    out   = model(x)

    print("Output shape:", out.shape)
    print("KV cache layers:", len(model.kv_cache))
    print("K shape layer 0:", model.kv_cache[0][0].shape)
    print("K strides layer 0:", model.kv_cache[0][0].stride())