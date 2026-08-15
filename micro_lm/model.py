import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class ESP32LLMConfig:
    def __init__(
        self,
        vocab_size: int = 44,
        block_size: int = 64,
        n_layer: int = 4,
        n_head: int = 4,
        n_kv_head: int = None,
        n_embd: int = 64,
        dropout: float = 0.0,
        bias: bool = True
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias

    @classmethod
    def micro_lm_max_ctx(cls):
        """Max Context Architecture (~13.7M params, 2048 ctx, MQA)"""
        return cls(vocab_size=256, block_size=2048, n_layer=2, n_head=12, n_kv_head=1, n_embd=768)

    @classmethod
    def micro_lm_ultra(cls):
        """Balanced Architecture (~11.4M params, 1024 ctx, MQA)"""
        return cls(vocab_size=256, block_size=1024, n_layer=4, n_head=8, n_kv_head=1, n_embd=512)

    @classmethod
    def micro_lm_pro(cls):
        """OTA Safe Architecture (~3.1M params, 1024 ctx, MQA, RoPE)"""
        return cls(vocab_size=256, block_size=1024, n_layer=4, n_head=4, n_kv_head=1, n_embd=256)

    @classmethod
    def micro_lm_mega(cls):
        """Massive Architecture to hit PPL 5-6 (~42M params, 1024 ctx, MQA)"""
        return cls(vocab_size=256, block_size=1024, n_layer=6, n_head=12, n_kv_head=1, n_embd=768)

    @classmethod
    def micro_lm_s3_large(cls):
        """Massive Architecture for ESP32-S3 8MB PSRAM (~26.2M params, 1024 ctx, MQA)"""
        return cls(vocab_size=256, block_size=1024, n_layer=10, n_head=8, n_kv_head=1, n_embd=512)

    @classmethod
    def micro_lm_colossus(cls):
        """Absolute Max Architecture for ESP32-S3 8MB PSRAM & 16MB Flash (~48.3M params, 1024 ctx, MQA)"""
        return cls(vocab_size=256, block_size=1024, n_layer=8, n_head=8, n_kv_head=1, n_embd=768)

    @classmethod
    def micro_lm_pico(cls):
        """Standard ESP32 Architecture (~150K params, 128 ctx, MQA)"""
        return cls(vocab_size=256, block_size=128, n_layer=4, n_head=4, n_kv_head=1, n_embd=64)

def get_rotary_matrix(seq_len, dim, base=10000):
    theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    seq_idx = torch.arange(seq_len).float()
    idx_theta = torch.einsum('i,j->ij', seq_idx, theta)
    idx_theta = torch.cat([idx_theta, idx_theta], dim=-1) # (seq_len, dim)
    cos = idx_theta.cos().unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, dim)
    sin = idx_theta.sin().unsqueeze(0).unsqueeze(0)
    return cos, sin

def apply_rope(x, cos, sin):
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    x_rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + x_rot * sin

class CausalSelfAttention(nn.Module):
    def __init__(self, config: ESP32LLMConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_kv_head = getattr(config, "n_kv_head", config.n_head)
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert self.n_head % self.n_kv_head == 0
        
        # key, query, value projections for GQA/MQA
        self.q_proj = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # causal mask buffer
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
        
        # precompute RoPE cache
        cos, sin = get_rotary_matrix(config.block_size, self.head_dim)
        self.register_buffer("cos_cached", cos)
        self.register_buffer("sin_cached", sin)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2) # (B, nkv, T, hs)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2) # (B, nkv, T, hs)
        
        # apply RoPE
        cos = self.cos_cached[:, :, :T, :]
        sin = self.sin_cached[:, :, :T, :]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # repeat k and v for MQA/GQA
        num_kv_groups = self.n_head // self.n_kv_head
        if num_kv_groups > 1:
            k = torch.repeat_interleave(k, repeats=num_kv_groups, dim=1)
            v = torch.repeat_interleave(v, repeats=num_kv_groups, dim=1)

        # causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config: ESP32LLMConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config: ESP32LLMConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class ESP32LLM(nn.Module):
    def __init__(self, config: ESP32LLMConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share embedding weights with final head
        self.transformer.wte.weight = self.lm_head.weight

        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self):
        """Returns total parameter count excluding tied weights double-counting."""
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"
        
        tok_emb = self.transformer.wte(idx) # (b, t, n_embd)
        x = tok_emb
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference optimization: only compute logits for last token
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive generation loop with temperature & top-k filtering."""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
