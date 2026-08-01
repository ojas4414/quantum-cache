import torch 
import torch.nn as nn
import math

batch_size=2
Seq_len=16
D_model=64
N_Heads=4
N_layers=2
D_head=D_model//N_Heads


class Multihead(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads=n_heads
        self.d_head=d_model//n_heads


        self.W_q=nn.Linear(d_model,d_model,bias=False)
        self.W_k=nn.Linear(d_model,d_model,bias=False)
        self.W_v=nn.Linear(d_model,d_model,bias=False)
        self.o=nn.Linear(d_model,d_model,bias=False)


    def forward(self,x):
        B,S,D= x.shape

        Q= self.W_q(x)
        K=self.W_k(x)
        V= self.W_v(x)


        Q = Q.view(B,S,self.n_heads, self.d_head).transpose(1,2)
        K = K.view(B,S,self.n_heads, self.d_head).transpose(1,2)
        V = V.view(B,S,self.n_heads, self.d_head).transpose(1,2)

        scale=math.sqrt(self.d_head)
        score=torch.matmul(Q,K.transpose(-2,-1)) / scale
        weights=torch.softmax(score, dim=-1)

        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, S, self.d_model)

        return self.o(out), K, V


class Feedforward(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.net= nn.Sequential(
            nn.Linear(d_model,d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4,d_model)
        )
    def forward(self,x):
        return self.net(x)
    
class Transformer_block(nn.Module):
    def __init__(self,d_model,n_heads):
        super().__init__()
        self.atten= Multihead(d_model,n_heads)
        self.ff=Feedforward(d_model)

        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)


    def forward(self,x):
        attn_out,K,V=self.atten(self.norm1(x))
        x= x + attn_out
        x=x +self.ff(self.norm2(x))
        return x,K,V
    
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            Transformer_block(D_model,N_Heads) for _ in range(N_layers)
        ])
        self.kv_cache = []
    def forward(self,x):
        # Reset per forward pass. Without this the cache grows without bound
        # across repeated calls, so a 100-run benchmark ends up timing
        # allocator pressure from hundreds of retained tensors rather than
        # the attention math itself.
        self.kv_cache = []
        for block in self.blocks:
            x,K,V=block(x)
            self.kv_cache.append((K,V))
        return x



if __name__ == "__main__":
    model = Transformer()
    x     = torch.randn(batch_size, Seq_len, D_model)
    out   = model(x)

    print("Output shape:", out.shape)
    print("KV cache layers:", len(model.kv_cache))
    print("K shape layer 0:", model.kv_cache[0][0].shape)
    print("K strides layer 0:", model.kv_cache[0][0].stride())