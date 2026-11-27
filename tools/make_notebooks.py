"""Build the GPU notebooks in notebooks/ from the cell lists below (keeps them reviewable as plain Python).

    python tools/make_notebooks.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "https://github.com/raikar8/ML-Lab"

SETUP = f"""# Colab: Runtime → Change runtime type → GPU (a T4 is enough except where noted). Kaggle: Settings → Accelerator.
import os, sys
if not os.path.exists("ML-Lab") and not os.path.exists("gpu"):
    !git clone --depth 1 {REPO}.git
if os.path.exists("ML-Lab"):
    %cd ML-Lab
sys.path.insert(0, ".")
import torch, triton
print(torch.__version__, triton.__version__, torch.cuda.get_device_name() if torch.cuda.is_available() else "NO GPU")
assert torch.cuda.is_available(), "Switch the runtime to a GPU"
"""

BENCH = """import triton.testing
def bench(fn):  # median milliseconds, with warm-up and L2 flushing handled by Triton
    return triton.testing.do_bench(fn, warmup=25, rep=100)
"""


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(keepends=True)}


def badge(name):
    return (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
            f"(https://colab.research.google.com/github/raikar8/ML-Lab/blob/main/notebooks/{name})")


NOTEBOOKS = {
"01_triton_fusion.ipynb": [
    md("""# 1 · Triton fundamentals: fused softmax and fused residual + RMSNorm
Week 15. Memory-bound ops are limited by bytes moved, not FLOPs. Here you'll measure that directly: a fused Triton softmax reads
and writes each element once, while the unfused PyTorch chain (max → subtract → exp → sum → divide) makes several passes over HBM.
The kernels are in `gpu/triton_kernels.py` (written from scratch; read them first)."""),
    code(SETUP), code(BENCH),
    md("""## Correctness first
Always check a kernel against a reference before timing it."""),
    code("""from gpu import triton_kernels as tk
import torch.nn.functional as F
x = torch.randn(4096, 2048, device="cuda", dtype=torch.float16)
torch.testing.assert_close(tk.softmax(x), torch.softmax(x.float(), -1).half(), atol=2e-3, rtol=2e-3)
w, r = torch.rand(2048, device="cuda", dtype=torch.float16), torch.randn_like(x)
y, res = tk.add_rmsnorm(x, w, r)
torch.testing.assert_close(y, F.rms_norm((x + r).float(), (2048,), w.float(), eps=1e-6).half(), atol=2e-2, rtol=2e-2)
print("ok")"""),
    md("""## Bandwidth: fused vs unfused
Achieved bandwidth = bytes that *must* move (read x once, write y once) ÷ time. Compare with your GPU's peak (T4 ≈ 320 GB/s,
L4 ≈ 300 GB/s, A100 ≈ 1.5–2 TB/s)."""),
    code("""def naive_softmax(x):
    m = x.max(-1, keepdim=True).values      # pass 1
    e = torch.exp(x - m)                     # pass 2 (+ write)
    return e / e.sum(-1, keepdim=True)       # passes 3–4

rows = []
for n in [256, 1024, 2048, 4096, 8192]:
    x = torch.randn(8192, n, device="cuda", dtype=torch.float16)
    gb = 2 * x.numel() * x.element_size() / 1e9
    t_tri, t_torch, t_naive = bench(lambda: tk.softmax(x)), bench(lambda: torch.softmax(x, -1)), bench(lambda: naive_softmax(x))
    rows.append((n, gb / t_tri * 1e3, gb / t_torch * 1e3, gb / t_naive * 1e3))
print(f"{'cols':>6} {'triton GB/s':>12} {'torch GB/s':>11} {'naive GB/s':>11}")
for r in rows:
    print(f"{r[0]:>6} {r[1]:>12.0f} {r[2]:>11.0f} {r[3]:>11.0f}")"""),
    code("""x = torch.randn(16384, 4096, device="cuda", dtype=torch.float16); r = torch.randn_like(x); w = torch.ones(4096, device="cuda", dtype=torch.float16)
eager = lambda: F.rms_norm(x + r, (4096,), w, eps=1e-6)
print(f"fused add+RMSNorm: {bench(lambda: tk.add_rmsnorm(x, w, r)):.3f} ms   eager: {bench(eager):.3f} ms   "
      f"compiled: {bench(torch.compile(eager)):.3f} ms")"""),
    md("""## Your turn
1. Change the softmax kernel to process **several rows per program** (loop inside the kernel). When does it help?
2. Rows longer than one block (e.g. 65,536 columns) don't fit: write a two-pass (or online) version that loops over column blocks.
3. Explain in two sentences why `torch.compile` gets close to the hand-written kernel here — and when it wouldn't."""),
],
"02_triton_flash_attention.ipynb": [
    md("""# 2 · FlashAttention forward in Triton
Weeks 14–15. `gpu/flash_attention.py` implements the online-softmax tiled attention you wrote in NumPy-style PyTorch in week 14,
now as a real GPU kernel. You'll compare it with a naive implementation that materialises the T×T matrix and with PyTorch's
`scaled_dot_product_attention` (which dispatches to FlashAttention/efficient kernels)."""),
    code(SETUP), code(BENCH),
    code("""from gpu.flash_attention import flash_attention
import torch.nn.functional as F
B, H, D = 4, 8, 64
for causal in (False, True):
    q, k, v = (torch.randn(B, H, 1000, D, device="cuda", dtype=torch.float16) for _ in range(3))
    torch.testing.assert_close(flash_attention(q, k, v, causal), F.scaled_dot_product_attention(q, k, v, is_causal=causal),
                               atol=2e-2, rtol=2e-2)
print("correct")"""),
    code("""def naive(q, k, v, causal):
    s = (q @ k.transpose(-1, -2)) / D ** 0.5
    if causal:
        s = s.masked_fill(torch.ones(s.shape[-2:], device=s.device, dtype=torch.bool).triu(1), float("-inf"))
    return torch.softmax(s, -1) @ v

print(f"{'T':>6} {'naive ms':>9} {'ours ms':>8} {'sdpa ms':>8} {'ours TFLOP/s':>13} {'naive extra MB':>15}")
for T in [512, 1024, 2048, 4096, 8192]:
    q, k, v = (torch.randn(B, H, T, D, device="cuda", dtype=torch.float16) for _ in range(3))
    flops = 4 * B * H * T * T * D / 2  # causal ≈ half
    t_naive = bench(lambda: naive(q, k, v, True)) if T <= 4096 else float("nan")
    t_ours, t_sdpa = bench(lambda: flash_attention(q, k, v, True)), bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
    print(f"{T:>6} {t_naive:>9.2f} {t_ours:>8.2f} {t_sdpa:>8.2f} {flops / t_ours / 1e9:>13.1f} {B * H * T * T * 2 / 2**20:>15.0f}")"""),
    md("""## Your turn
1. Sweep `BLOCK_M`, `BLOCK_N` ∈ {32, 64, 128} and `num_warps` (pass it in the launch) — find the best config for your GPU.
2. Return the log-sum-exp per row (as in the week-14 exercise) — the backward pass needs it.
3. Implement **FlashDecoding**: for a single query per sequence, split the KV sequence across programs, then combine the partial
   results with the log-sum-exp merge (week 14 `combine_partials`)."""),
],
"03_triton_matmul_int8.ipynb": [
    md("""# 3 · Tiled matmul and int8 weight-only matmul (why W8A16 speeds up decoding)
Weeks 15 and 23. A tiled GEMM reuses each loaded tile many times (compute-bound at large M). At decode time M is tiny (batch 1–16),
so the GEMM becomes a memory-bound matrix–vector product — halving weight bytes with int8 (dequantised in registers) should give up
to ~2× even though the arithmetic is unchanged."""),
    code(SETUP), code(BENCH),
    code("""from gpu import triton_kernels as tk
a, b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16), torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
torch.testing.assert_close(tk.matmul(a, b), a @ b, atol=0.5, rtol=1e-2)
for BM, BN, BK in [(32, 32, 32), (64, 64, 32), (128, 64, 32), (128, 128, 32), (64, 128, 64)]:
    t = bench(lambda: tk.matmul(a, b, BM, BN, BK))
    print(f"tile {BM}x{BN}x{BK}: {t:.3f} ms = {2 * 2048**3 / t / 1e9:.1f} TFLOP/s")
print(f"cuBLAS: {2 * 2048**3 / bench(lambda: a @ b) / 1e9:.1f} TFLOP/s")"""),
    code("""K, N = 4096, 4096
W = torch.randn(K, N, device="cuda", dtype=torch.float16) * 0.02
Wq, s = tk.quantize_w8(W)
print(f"{'M':>4} {'fp16 ms':>8} {'w8 ms':>7} {'speed-up':>9} {'rel. error':>11}")
for M in [1, 4, 16, 64, 256]:
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    t16, t8 = bench(lambda: x @ W), bench(lambda: tk.w8_matmul(x, Wq, s, BM=16 if M <= 16 else 32))
    err = ((tk.w8_matmul(x, Wq, s) - x @ W).float().norm() / (x @ W).float().norm()).item()
    print(f"{M:>4} {t16:>8.3f} {t8:>7.3f} {t16 / t8:>8.2f}x {err:>11.4f}")"""),
    md("""## Your turn
1. Add `@triton.autotune` with a few configs to `_matmul_kernel` and compare with cuBLAS.
2. Implement **int4** weights (two values per byte, unpack with shifts/masks in the kernel, group-wise scales of 64) — the
   W4A16 kernel behind GPTQ/AWQ deployment. Measure speed-up at M = 1.
3. Explain the M at which int8 stops helping, using the roofline from week 21."""),
],
"04_torch_compile_profiling.ipynb": [
    md("""# 4 · `torch.compile`, graph breaks and the profiler
Week 9. Measure eager vs compiled on a small decoder, read a profiler table, and see what a graph break costs."""),
    code(SETUP),
    code("""from gpu.decode import TinyDecoder
m = TinyDecoder(vocab=32000, d=1024, n_layers=8, n_heads=16, max_len=1024).cuda().half().eval()
x = torch.randint(0, 32000, (8, 512), device="cuda")
def timed(f, n=20):
    for _ in range(3): f()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
    for _ in range(n): f()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n
with torch.inference_mode():
    eager = timed(lambda: m.forward_full(x))
    cm = torch.compile(m.forward_full)
    compiled = timed(lambda: cm(x))
print(f"eager {eager:.2f} ms   compiled {compiled:.2f} ms   speed-up {eager / compiled:.2f}x")"""),
    code("""from torch.profiler import profile, ProfilerActivity
with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    m.forward_full(x); torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))"""),
    md("""## Graph breaks
`.item()` forces a GPU→CPU sync and Dynamo can't trace data-dependent Python control flow, so it splits the graph."""),
    code("""import torch._dynamo as dynamo
def f(x):
    y = torch.relu(x) * 2
    if y.sum().item() > 0:      # graph break: data-dependent Python branch
        y = y + 1
    return torch.tanh(y)
explanation = dynamo.explain(f)(torch.randn(1000, device="cuda"))
print("graphs:", explanation.graph_count, " breaks:", explanation.graph_break_count)
for b in explanation.break_reasons: print(" -", b.reason[:120])
g = lambda x: torch.tanh(torch.where((torch.relu(x) * 2).sum() > 0, torch.relu(x) * 2 + 1, torch.relu(x) * 2))
print("rewritten:", dynamo.explain(g)(torch.randn(1000, device="cuda")).graph_break_count, "breaks")"""),
    md("""## Your turn
Export a Perfetto trace (`prof.export_chrome_trace("trace.json")`), open it at ui.perfetto.dev, and find: the largest kernel, any
gaps where the GPU idles, and how many kernels a single decoder layer launches eager vs compiled."""),
],
"05_cuda_graphs_decode.ipynb": [
    md("""# 5 · CUDA graphs for small-batch decoding
Weeks 14 and 25. At batch 1 a decode step is dozens of tiny kernels; CPU launch overhead (~5–10 µs each) can dominate. A CUDA graph
records the whole step once and replays it with one launch. It needs **static shapes and addresses** — hence the preallocated KV
cache in `gpu/decode.py`, written in place at position `pos`."""),
    code(SETUP),
    code("""from gpu.decode import TinyDecoder, capture_step
torch.manual_seed(0)
m = TinyDecoder(vocab=32000, d=512, n_layers=8, n_heads=8, max_len=512).cuda().eval()
B, steps = 1, 200
def eager_decode():
    cache = m.new_cache(B, "cuda"); tok = torch.zeros(B, dtype=torch.long, device="cuda")
    with torch.inference_mode():
        for t in range(steps):
            tok = m.step(tok, torch.tensor(t, device="cuda"), cache).argmax(-1)
    return tok
g, s_tok, s_pos, cache, s_logits = capture_step(m, B)
def graph_decode():
    cache.zero_(); s_tok.zero_()
    for t in range(steps):
        s_pos.fill_(t); g.replay(); s_tok.copy_(s_logits.argmax(-1))
    return s_tok
assert torch.equal(eager_decode(), graph_decode()), "graph replay must give identical tokens"
import time
for name, f in [("eager", eager_decode), ("cuda graph", graph_decode)]:
    f(); torch.cuda.synchronize(); t0 = time.perf_counter(); f(); torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"{name:>10}: {steps / dt:7.0f} tokens/s  ({dt / steps * 1e3:.2f} ms/token)")"""),
    md("""## Your turn
1. Repeat with batch 32 and with `d=2048`. When do graphs stop mattering, and why (compute- vs launch-bound)?
2. `torch.compile(mode="reduce-overhead")` uses CUDA graphs automatically — compare it with the manual capture.
3. Why must the sampling step (argmax/top-p) also be inside the graph for best results? Try moving it in."""),
],
"06_ddp_vs_fsdp.ipynb": [
    md("""# 6 · DDP vs FSDP2 on two GPUs
Week 15. **Needs 2 GPUs** — Kaggle's free "GPU T4 × 2" accelerator works. `gpu/dist_train.py` trains the same model with DDP
(full replica per GPU, gradients all-reduced) or FSDP2 (`fully_shard` per block: parameters all-gathered just before use, gradients
reduce-scattered). Compare peak memory per GPU and step time."""),
    code(SETUP + '\nassert torch.cuda.device_count() >= 2, "needs two GPUs (Kaggle T4 x2)"'),
    code("""!torchrun --nproc_per_node=2 gpu/dist_train.py --mode ddp  --d 1024 --layers 12 --batch 4 --seq 256"""),
    code("""!torchrun --nproc_per_node=2 gpu/dist_train.py --mode fsdp --d 1024 --layers 12 --batch 4 --seq 256"""),
    code("""# Find the largest model each mode can train (doubling layers until OOM)
for mode in ["ddp", "fsdp"]:
    for layers in [16, 24, 32, 48]:
        r = !torchrun --nproc_per_node=2 gpu/dist_train.py --mode {mode} --d 1536 --layers {layers} --steps 6 --batch 2 --seq 256 2>&1 | tail -4
        ok = any("rank 0" in line for line in r)
        print(mode, layers, "OK" if ok else "OOM / failed")
        if not ok: break"""),
    md("""## Your turn
1. Explain the memory numbers with the 16 bytes/param rule and the world size.
2. Add activation checkpointing (`torch.utils.checkpoint`) per block: what does it do to memory and step time?
3. Why is FSDP's step slower at small sizes but able to train models DDP can't fit? Where does the extra communication come from?"""),
],
"07_low_precision.ipynb": [
    md("""# 7 · Low-precision matmuls: FP16, BF16, FP8 and the cost of each
Weeks 6, 18 and 23. Measure speed and error of matmuls in different formats. **FP8 needs an Ada/Hopper GPU** (Colab L4 or better;
T4 will skip that part). FP8 uses per-tensor scaling here; DeepSeek-V3's recipe uses finer 1×128 / 128×128 scaling (week 19)."""),
    code(SETUP),
    code("""def timed(f, n=50):
    for _ in range(5): f()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True); s.record()
    for _ in range(n): f()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / n
N = 4096
a32, b32 = torch.randn(N, N, device="cuda"), torch.randn(N, N, device="cuda")
ref = a32.double() @ b32.double()
def rel(c): return ((c.double() - ref).norm() / ref.norm()).item()
for dt in [torch.float32, torch.float16, torch.bfloat16]:
    a, b = a32.to(dt), b32.to(dt)
    t = timed(lambda: a @ b)
    print(f"{str(dt):>15}: {2 * N**3 / t / 1e9:7.1f} TFLOP/s   rel. error {rel(a @ b):.2e}")"""),
    code("""cap = torch.cuda.get_device_capability()
if cap >= (8, 9) and hasattr(torch, "_scaled_mm"):
    f8 = torch.float8_e4m3fn
    sa, sb = a32.abs().max() / 448, b32.abs().max() / 448
    a8 = (a32 / sa).to(f8)
    b8 = (b32 / sb).to(f8).t().contiguous().t()          # _scaled_mm wants column-major B
    one = lambda x: torch.tensor(x, device="cuda", dtype=torch.float32)
    mm = lambda: torch._scaled_mm(a8, b8, scale_a=one(sa.item()), scale_b=one(sb.item()), out_dtype=torch.bfloat16)
    t = timed(mm)
    print(f"FP8 e4m3 (per-tensor scale): {2 * N**3 / t / 1e9:.1f} TFLOP/s   rel. error {rel(mm()):.2e}")
else:
    print(f"GPU capability {cap}: no FP8 Tensor Cores here (needs ≥ 8.9, e.g. L4/H100) — skipped")"""),
    md("""## Your turn
1. Add a few outlier columns to `a32` (×100) and compare FP8 per-tensor error with the 1×128 tile scaling from your week-19
   exercise (quantise/dequantise per tile, then multiply in BF16 to simulate).
2. Why does BF16 have larger error than FP16 here, yet is preferred for training?
3. Estimate the memory and bandwidth savings of FP8 KV caches for a 32-layer, 8-KV-head model at 32K context."""),
],
}


def main():
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        cells = [md(badge(name))] + cells
        nb = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python",
              "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
        (out / name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print("wrote", name)


if __name__ == "__main__":
    main()
