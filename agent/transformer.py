import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class AgentConfig:
    vocab_size:int=512
    d_model:int=192
    n_heads:int=6
    n_layers:int=6
    ffn_mult:int=4
    ctx_len:int=512
    n_tok_types:int=4
    latent_dim:int =128
    dropout:float=0.1
    bias:bool=False
    head_dim:int=0
    ffn_dim:int=0


    def __post_init__(self):
        assert self.d_model%self.n_heads==0
        self.head_dim=self.d_model//self.n_heads
        self.ffn_dim=self.ffn_mult*self.d_model

class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-6):
        super().__init__()
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        rms=x.pow(2).mean(-1,keepdim=True).add(self.eps).sqrt()
        return (x/rms)*self.weight

def precompute_rope_freqs(head_dim:int,ctx_len:int,base:float=10000.0)->torch.Tensor:
    i=torch.arange(0,head_dim,2).float()
    theta=1.0/(base**(i/head_dim))
    pos=torch.arange(ctx_len).float()
    freqs=torch.outer(pos,theta)
    return torch.polar(torch.ones_like(freqs),freqs)

def apply_rope(x:torch.Tensor,freqs:torch.Tensor)->torch.tensor:
    B,H,T,D=x.shape
    x_complex=torch.view_as_complex(x.float().reshape(B,H,T,D//2,2))
    freqs  = freqs[:T].unsqueeze(0).unsqueeze(0)
    x_rot  = x_complex * freqs
    return torch.view_as_real(x_rot).reshape(B, H, T, D).to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self,cfg:AgentConfig):
        super().__init__()
        self.n_heads=cfg.n_heads
        self.head_dim=cfg.head_dim
        self.d_model=cfg.d_model
        self.dropout=cfg.dropout
        self.qkv=nn.Linear(cfg.d_model,3*cfg.d_model,bias=cfg.bias)
        self.proj=nn.Linear(cfg.d_model,cfg.d_model,bias=cfg.bias)
        self.drop=nn.Dropout(cfg.dropout)

    def forward(self,x:torch.Tensor,freqs:torch.Tensor)->torch.Tensor:
        B,T,D=x.shape
        q,k,v=self.qkv(x).split(D,dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q=apply_rope(q,freqs)
        k=apply_rope(k,freqs)
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask  = None,
            dropout_p  = self.dropout if self.training else 0.0,
            is_causal  = True,
        )
        y=y.transpose(1,2).contiguous().view(B,T,D)
        return self.drop(self.proj(y))


class FeedForward(nn.Module):
    def __init__(self,cfg:AgentConfig):
        super().__init__()
        self.w1=nn.Linear(cfg.d_model,cfg.ffn_dim,bias=cfg.bias)
        self.w2   = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=cfg.bias)
        self.proj = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.drop(self.proj(F.silu(self.w1(x))*self.w2(x)))

class Block(nn.Module):
    def __init__(self,cfg:AgentConfig):
        super().__init__()
        self.norm1=RMSNorm(cfg.d_model)
        self.attn=CausalSelfAttention(cfg)
        self.norm2=RMSNorm(cfg.d_model)
        self.ffn=FeedForward(cfg)
    
    def forward(self,x:torch.Tensor,freqs:torch.Tensor)->torch.Tensor:
        x=x+self.attn(self.norm1(x),freqs)
        x=x+self.ffn(self.norm2(x))
        return x
    
class AgentTransformer(nn.Module):
    def __init__(self,cfg:AgentConfig):
        super().__init__()
        self.cfg=cfg
        self.tok_emb=nn.Embedding(cfg.vocab_size,cfg.d_model)
        self.type_emb=nn.Embedding(cfg.n_tok_types,cfg.d_model)
        self.drop=nn.Dropout(cfg.dropout)
        self.blocks=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm=RMSNorm(cfg.d_model)
        self.private=nn.Parameter(torch.zeros(cfg.latent_dim))
        self.latent_proj=nn.Linear(cfg.latent_dim,cfg.d_model,bias=False)
        self.lm_head=nn.Linear(cfg.latent_dim,cfg.d_model,bias=False)
        self.lm_head.weight=self.tok_emb.weight
        freqs=precompute_rope_freqs(cfg.head_dim,cfg.ctx_len)
        self.register_buffer("rope_freqs",freqs)
        self.apply(self._init_weights)
    
    def _init_weights(self,module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if isinstance(module,nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        for name,p in module.named_parameters():
            if name=='proj.weight':
                nn.init.normal_(p,mean=0.0,std=0.02/math.sqrt(2*self.cfg.n_layers))
    
    def forward(self,idx:torch.Tensor,tok_types:torch.Tensor)->torch.Tensor:
        B,T=idx.shape
        assert T <= self.cfg.ctx_len, \
            f"Sequence {T} exceeds context window {self.cfg.ctx_len}"
        x=self.tok_emb(idx)+self.type_emb(tok_types)
        s=self.drop(x)

        for block in self.blocks:
            x=block(x,self.rope_freqs)
        
        x=self.norm(x)
        latent=self.latent_proj(self.private)
        x=x+latent.unsqueeze(0).unsqueeze(0)
        return self.lm_head(x)
    
    def count_params(self)->int:
        return sum(p.numel() for p in self.parameters())
    
    @torch.no_grad()
    def generate(self,idx:torch.Tensor,max_new_tokens: int = 64,
                 temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_ctx   = idx[:, -self.cfg.ctx_len:]
            types_ctx = tok_types[:, -self.cfg.ctx_len:]
 
            logits    = self(idx_ctx, types_ctx)
            logits    = logits[:, -1, :] / temperature
            probs     = F.softmax(logits, dim=-1)
            next_tok  = torch.multinomial(probs, num_samples=1)
            next_type = torch.ones_like(next_tok)   # new tokens = action type
 
            idx       = torch.cat([idx,       next_tok],  dim=1)
            tok_types = torch.cat([tok_types, next_type], dim=1)
 
        return idx
def make_synthetic_trace(cfg: AgentConfig,
                         seq_len: int = 128) -> tuple:

    tokens = torch.zeros(seq_len, dtype=torch.long)
    types  = torch.zeros(seq_len, dtype=torch.long)
 
    n       = seq_len - 4       
    quarter = n // 4            

    tokens[0] = 0; types[0] = 3

    end_task = 1 + quarter
    tokens[1:end_task] = torch.randint(10, cfg.vocab_size, (quarter,))
    types[1:end_task]  = 0

    tokens[end_task] = 1; types[end_task] = 3

    start_mem = end_task + 1
    end_mem   = start_mem + quarter * 2
    tokens[start_mem:end_mem] = torch.randint(10, cfg.vocab_size, (quarter * 2,))
    types[start_mem:end_mem]  = 2

    tokens[end_mem] = 1; types[end_mem] = 3
 

    start_act = end_mem + 1
    end_act   = seq_len - 1
    act_len   = end_act - start_act
    tokens[start_act:end_act] = torch.randint(10, cfg.vocab_size, (act_len,))
    types[start_act:end_act]  = 1
 
  
    tokens[seq_len - 1] = 2; types[seq_len - 1] = 3
 
    return tokens, types

def make_batch(cfg:AgentConfig,batch_size:int=32,seq_len:int=128)->tuple:
    all_tokens=[];all_types=[]
    for _ in range(batch_size):
        t, ty = make_synthetic_trace(cfg, seq_len)
        all_tokens.append(t); all_types.append(ty)
 
    tokens = torch.stack(all_tokens)
    types  = torch.stack(all_types)
 
    return tokens[:, :-1], types[:, :-1], tokens[:, 1:]

def train(model:AgentTransformer,cfg:AgentConfig,steps:int=300,lr:float=3e-4,batch_size:int=32)->list:
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.1,betas=(0.9,0.95))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,T_max=steps,eta_min=lr*0.1
    )
    model.train()
    history=[]
    for step in range(steps):
        x_tok, x_typ, targets = make_batch(cfg, batch_size, seq_len=128)
 
        logits = model(x_tok, x_typ)
        loss   = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            targets.reshape(-1)
        )
 
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
 
        history.append((step, loss.item()))
        if step % 50 == 0:
            print(f"  step {step:4d}  loss={loss.item():.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")
 
    return history

def overfit_test(model:AgentTransformer,cfg:AgentConfig,n_traces:int=100,steps:int=300)->float:
    print(f"\nOverfit test: {n_traces} fixed traces, {steps} steps")
    all_tok = []; all_typ = []
    for _ in range(n_traces):
        t, ty = make_synthetic_trace(cfg, seq_len=128)
        all_tok.append(t); all_typ.append(ty)
 
    toks    = torch.stack(all_tok)
    types   = torch.stack(all_typ)
    x_tok   = toks[:, :-1]
    x_typ   = types[:, :-1]
    targets = toks[:, 1:]
 
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
 
    for step in range(steps):
        logits = model(x_tok, x_typ)
        loss   = F.cross_entropy(
            logits.reshape(-1, cfg.vocab_size),
            targets.reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
 
        if step % 100 == 0:
            print(f"  step {step:4d}  loss={loss.item():.4f}")
    print(f"  final loss={loss.item():.4f}")
    return loss.item()

if __name__ == "__main__":
    print("=" * 54)
    print("PANTHEON Week 2 — Agent Core Transformer")
    print("=" * 54)
 
    cfg   = AgentConfig()
    model = AgentTransformer(cfg)

    n_params = model.count_params()
    print(f"\nParam count : {n_params:,}  (~{n_params/1e6:.1f}M)")
    assert 2_000_000 < n_params < 5_000_000
    print(" param count in range (2M–5M)")
 
    print("\nForward pass timing (CPU, B=1, T=128)...")
    model.eval()
    d = torch.randint(0, cfg.vocab_size, (1, 128))
    dt = torch.zeros(1, 128, dtype=torch.long)
    with torch.no_grad():
        for _ in range(3): model(d, dt)   # warmup
    times = []
    with torch.no_grad():
        for _ in range(10):
            t0 = time.perf_counter()
            out = model(d, dt)
            times.append((time.perf_counter() - t0) * 1000)
    avg_ms = sum(times) / len(times)
    print(f"  avg : {avg_ms:.1f}ms   shape={out.shape}")
    assert avg_ms < 200
    print("forward pass < 200ms")

    assert out.shape == (1, 128, cfg.vocab_size)
    print("output shape (1, 128, 512)")

    print("\nTraining 300 steps on synthetic traces...")
    model = AgentTransformer(cfg)
    history = train(model, cfg, steps=300, batch_size=32)
    start_loss = history[0][1]
    final_loss = history[-1][1]
    print(f"\n  start={start_loss:.4f}  final={final_loss:.4f}")
    assert final_loss < start_loss, "Loss not decreasing"
    print("loss is decreasing")

    model_o = AgentTransformer(cfg)
    overfit_loss = overfit_test(model_o, cfg, n_traces=100, steps=300)
    assert overfit_loss < start_loss, "Overfit loss not dropping"
    print("overfit test: loss drops on fixed dataset")
 
    print("\nGeneration test (20 new tokens)...")
    seed_tok  = torch.randint(0, cfg.vocab_size, (1, 10))
    seed_typ  = torch.zeros(1, 10, dtype=torch.long)
    generated = model.generate(seed_tok, seed_typ, max_new_tokens=20)
    assert generated.shape == (1, 30)
    print(f"  shape={generated.shape}")
    print("generation works")

    a = AgentTransformer(cfg)
    b = AgentTransformer(cfg)
    with torch.no_grad():
        b.private.copy_(torch.randn(cfg.latent_dim))
        out_a = a(d, dt)
        out_b = b(d, dt)
    diff = (out_a - out_b).abs().mean().item()
    assert diff > 0
    print(f"private latent produces different outputs (diff={diff:.4f})")
 
    print("\n" + "=" * 54)
    print("WEEK 2 COMPLETE")
    print(f"  params       : {n_params:,}")
    print(f"  forward ms   : {avg_ms:.1f}")
    print(f"  start loss   : {start_loss:.4f}")
    print(f"  final loss   : {final_loss:.4f}")
    print(f"  overfit loss : {overfit_loss:.4f}")
    print("=" * 54)
 














