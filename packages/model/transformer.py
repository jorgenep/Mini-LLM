from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


def rope_cache(seq_len: int, head_dim: int, theta: float, device: torch.device, dtype: torch.dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    cos = angles.cos().to(dtype)[None, None, :, :]
    sin = angles.sin().to(dtype)[None, None, :, :]
    return cos, sin


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos = cos[..., :x_even.shape[-1]]
    sin = sin[..., :x_even.shape[-1]]
    return torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.group_size = config.n_heads // config.n_kv_heads
        self.rope_theta = config.rope_theta
        self.q_proj = nn.Linear(config.hidden_size, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.hidden_size, bias=False)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = rope_cache(seq_len, self.head_dim, self.rope_theta, x.device, x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k = k[:, :, None, :, :].expand(batch, self.n_kv_heads, self.group_size, seq_len, self.head_dim).flatten(1, 2)
        v = v[:, :, None, :, :].expand(batch, self.n_kv_heads, self.group_size, seq_len, self.head_dim).flatten(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        causal = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal[None, None, :, :], torch.finfo(scores.dtype).min)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask[:, None, None, :] == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        output = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(output)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, ffn_dim: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, ffn_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = Attention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = SwiGLU(config.hidden_size, config.ffn_dim)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        x = x + self.attention(self.attention_norm(x), attention_mask)
        return x + self.ffn(self.ffn_norm(x))


class CodeTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("Sequence length exceeds configured context length")
        x = self.token_embeddings(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return self.lm_head(self.final_norm(x))

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def parameter_breakdown(self) -> dict[str, int]:
        return {name: sum(p.numel() for p in module.parameters())
                for name, module in self.named_children()}
