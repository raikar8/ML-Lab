"""A tiny decoder with a *static* KV cache, so one decode step has fixed shapes and can be captured in a CUDA graph.

Dynamic caches (torch.cat each step) change tensor shapes every step and allocate memory, which CUDA graphs can't replay.
Here the cache is preallocated to max_len and written in place at position `pos`; attention masks out positions > pos.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyDecoder(nn.Module):
    def __init__(self, vocab=1000, d=256, n_layers=4, n_heads=4, max_len=512):
        super().__init__()
        self.h, self.dh, self.max_len = n_heads, d // n_heads, max_len
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.qkv = nn.ModuleList([nn.Linear(d, 3 * d, bias=False) for _ in range(n_layers)])
        self.o = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n_layers)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)) for _ in range(n_layers)])
        self.n1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.n2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_layers)])
        self.head = nn.Linear(d, vocab, bias=False)

    def new_cache(self, batch, device=None, dtype=torch.float32):
        shape = (len(self.qkv), 2, batch, self.h, self.max_len, self.dh)
        return torch.zeros(shape, device=device, dtype=dtype)

    def forward_full(self, tokens):
        """Reference: causal forward over the whole sequence (no cache)."""
        B, T = tokens.shape
        x = self.emb(tokens) + self.pos(torch.arange(T, device=tokens.device))
        for i in range(len(self.qkv)):
            q, k, v = self.qkv[i](self.n1[i](x)).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + self.o[i](a.transpose(1, 2).reshape(B, T, -1))
            x = x + self.ff[i](self.n2[i](x))
        return self.head(x)

    def step(self, token, pos, cache):
        """One decode step. token (B,) long, pos () long tensor (same for the batch), cache from new_cache(). In-place."""
        B = token.shape[0]
        x = self.emb(token) + self.pos(pos)[None]
        keep = (torch.arange(self.max_len, device=token.device) <= pos)[None, None, None, :]
        for i in range(len(self.qkv)):
            q, k, v = self.qkv[i](self.n1[i](x)).view(B, 3, self.h, self.dh).unbind(1)
            cache[i, 0].index_copy_(2, pos.view(1), k[:, :, None])
            cache[i, 1].index_copy_(2, pos.view(1), v[:, :, None])
            s = (q[:, :, None] @ cache[i, 0].transpose(-1, -2)) / math.sqrt(self.dh)
            s = s.masked_fill(~keep, float("-inf"))
            a = (torch.softmax(s, -1) @ cache[i, 1])[:, :, 0]
            x = x + self.o[i](a.reshape(B, -1))
            x = x + self.ff[i](self.n2[i](x))
        return self.head(x)


def capture_step(model, batch, device="cuda", warmup=3):
    """Capture model.step in a CUDA graph. Returns (graph, static_token, static_pos, cache, static_logits).
    To run a step: copy inputs into static_token/static_pos, then graph.replay(); read static_logits."""
    cache = model.new_cache(batch, device)
    tok = torch.zeros(batch, dtype=torch.long, device=device)
    pos = torch.zeros((), dtype=torch.long, device=device)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(warmup):
            model.step(tok, pos, cache)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.inference_mode():
        logits = model.step(tok, pos, cache)
    cache.zero_()
    return g, tok, pos, cache, logits
