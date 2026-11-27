"""
title: Back-of-envelope LLM estimates in code
week: 12
day: 5
level: easy
time: 25
tags: estimation, inference, embedded
core: yes

* `dense_params(L, d, vocab, ffn_mult=4.0, n_kv_ratio=1.0, tied=True)` ≈ `L·(2d² + 2d²·n_kv_ratio + 2·ffn_mult·d²) + vocab·d·(1 if tied else 2)`
  (attention Q,O = 2d², K,V = 2d²·n_kv_ratio, FFN two matrices of d×ffn_mult·d; ignore norms).
* `train_flops(n_active, tokens)` = `6·N·D`. `chinchilla_tokens(n)` = `20·n`.
* `weight_bytes(n_params, bits)` = `n·bits/8`.
* `decode_tokens_per_s(n_active, bits, bandwidth_gbs, kv_bytes_per_token=0, context=0)` — memory-bound estimate:
  bandwidth / (active weight bytes + KV bytes read per token).
* `prefill_tokens_per_s(n_active, tops, utilisation)` = `tops·1e12·utilisation / (2·N)`.
* `train_memory_gb(n_params)` — mixed-precision Adam: 16 bytes/param, in GB (1e9).
"""


def dense_params(L, d, vocab, ffn_mult=4.0, n_kv_ratio=1.0, tied=True):
    per_layer = 2 * d * d + 2 * d * d * n_kv_ratio + 2 * ffn_mult * d * d
    return int(L * per_layer + vocab * d * (1 if tied else 2))


def train_flops(n_active, tokens):
    return 6 * n_active * tokens


def chinchilla_tokens(n):
    return 20 * n


def weight_bytes(n_params, bits):
    return n_params * bits / 8


def decode_tokens_per_s(n_active, bits, bandwidth_gbs, kv_bytes_per_token=0, context=0):
    per_tok = weight_bytes(n_active, bits) + kv_bytes_per_token * context
    return bandwidth_gbs * 1e9 / per_tok


def prefill_tokens_per_s(n_active, tops, utilisation):
    return tops * 1e12 * utilisation / (2 * n_active)


def train_memory_gb(n_params):
    return 16 * n_params / 1e9
