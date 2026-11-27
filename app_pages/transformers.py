"""Transformer hub: how the architecture evolved, what a 2026 block looks like, and cheat sheets."""
import pandas as pd
import streamlit as st

from mlprep.ui import go_theory, lab_link, sources

# Primary sources per tab — exact titles from content/reading.md (the Papers page tracks them).
SOURCES = {
    "timeline": [
        'Attention Is All You Need',
        'BERT: Pre-training of Deep Bidirectional Transformers',
        'Language Models are Few-Shot Learners (GPT-3)',
        'Scaling Laws for Neural Language Models',
        'Switch Transformers',
        'RoFormer: Enhanced Transformer with Rotary Position Embedding',
        'Train Short, Test Long (ALiBi)',
        'Training Compute-Optimal Large Language Models (Chinchilla)',
        'FlashAttention',
        'LLaMA: Open and Efficient Foundation Language Models',
        'Mistral 7B',
        'Mixtral of Experts',
        'Mamba: Linear-Time Sequence Modeling with Selective State Spaces',
        'The Llama 3 Herd of Models',
        'DeepSeek-V2',
        'DeepSeek-V3 Technical Report',
        'DeepSeek-R1',
        'Kimi K2: Open Agentic Intelligence',
        'Qwen3 Technical Report',
        'Qwen3-Next: Towards Ultimate Training & Inference Efficiency',
        'DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models',
        'gpt-oss-120b & gpt-oss-20b Model Card',
        'GLM-5: from Vibe Coding to Agentic Engineering',
        'Kimi K2.5: Visual Agentic Intelligence',
        'mHC: Manifold-Constrained Hyper-Connections',
        'Attention Residuals',
        'DeepSeek-V4 report and model cards',
    ],
    "block": [
        'Attention Is All You Need',
        'GLU Variants Improve Transformer',
        'LLaMA: Open and Efficient Foundation Language Models',
        'GQA: Training Generalized Multi-Query Transformer Models',
        'DeepSeek-V2',
        'Hyper-Connections',
        'mHC: Manifold-Constrained Hyper-Connections',
        'Attention Residuals',
        'gpt-oss-120b & gpt-oss-20b Model Card',
    ],
    "families": [
        'BERT: Pre-training of Deep Bidirectional Transformers',
        'Language Models are Few-Shot Learners (GPT-3)',
        'Conformer: Convolution-augmented Transformer for Speech Recognition',
        'Learning Transferable Visual Models From Natural Language Supervision (CLIP)',
        'Mamba: Linear-Time Sequence Modeling with Selective State Spaces',
        'Kimi Linear: An Expressive, Efficient Attention Architecture',
        'MiniMax-01: Scaling Foundation Models with Lightning Attention',
    ],
    "attention": [
        'Fast Transformer Decoding: One Write-Head is All You Need (MQA)',
        'GQA: Training Generalized Multi-Query Transformer Models',
        'DeepSeek-V2',
        'Mistral 7B',
        'Efficient Streaming Language Models with Attention Sinks',
        'MiniMax-01: Scaling Foundation Models with Lightning Attention',
        'Parallelizing Linear Transformers with the Delta Rule over Sequence Length',
        'Gated Delta Networks',
        'Kimi Linear: An Expressive, Efficient Attention Architecture',
        'Native Sparse Attention',
        'DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models',
        'DeepSeek-V4 report and model cards',
    ],
    "positions": [
        'Attention Is All You Need',
        'RoFormer: Enhanced Transformer with Rotary Position Embedding',
        'Train Short, Test Long (ALiBi)',
        'YaRN: Efficient Context Window Extension of Large Language Models',
        'DeepSeek-V2',
    ],
    "moe": [
        'Switch Transformers',
        'Mixtral of Experts',
        'DeepSeekMoE',
        'Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts',
        'DeepSeek-V3 Technical Report',
        'Kimi K2: Open Agentic Intelligence',
        'Qwen3-Next: Towards Ultimate Training & Inference Efficiency',
        'LongCat-Flash Technical Report',
    ],
    "multi_gpu": [
        'Megatron-LM',
        'ZeRO: Memory Optimizations Toward Training Trillion Parameter Models',
        'GPipe',
        'Reducing Activation Recomputation in Large Transformer Models',
        'Ring Attention with Blockwise Transformers',
        'DeepSeek-V3 Technical Report',
        'Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations',
    ],
    "inference": [
        'FlashAttention',
        'Efficient Memory Management for LLM Serving with PagedAttention',
        'Fast Inference from Transformers via Speculative Decoding',
        'EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty',
        'Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving',
        'Step-3 is Large yet Affordable: Model-system Co-design for Cost-effective Decoding',
        'LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale',
        'AWQ: Activation-aware Weight Quantization',
    ],
}

st.title("Transformer hub")
st.markdown(
    "Everything about transformers in one place: how the architecture evolved from 2017 to 2026, what changed inside "
    "the block and why, and cheat sheets for attention variants, positional encodings, MoE, multi-GPU training and "
    "inference optimisation. The full theory is in **weeks 10–16** (links at the bottom)."
)

tabs = st.tabs(["Evolution timeline", "Block: 2017 vs 2026", "Families & variants", "Attention variants",
                "Positional encodings", "MoE", "Multi-GPU", "Inference", "Study path"])

with tabs[0]:
    rows = [
        (2017, "Transformer (Vaswani et al.)", "Self-attention replaces recurrence; enc–dec; post-LN; sinusoidal PE",
         "Parallel training, long-range dependencies"),
        (2018, "GPT-1 · BERT", "Decoder-only LM pre-training · bidirectional masked LM", "Pre-train then fine-tune"),
        (2019, "GPT-2 · T5 · MQA", "Pre-norm · relative position bias · one shared K/V head",
         "Stable depth · transfer as text-to-text · smaller KV cache"),
        (2020, "GPT-3 · Kaplan scaling laws · GShard/Switch", "175B dense; in-context learning · sparse MoE",
         "Scale predictably; more params per FLOP"),
        (2021, "RoPE · ALiBi", "Rotary / linear-bias positions", "Relative positions, extrapolation"),
        (2022, "Chinchilla · PaLM · FlashAttention", "20 tokens/param · SwiGLU, parallel blocks, z-loss · IO-aware exact attention",
         "Compute-optimal data · quality per FLOP · memory/speed"),
        (2023, "Llama 1/2 · Mistral 7B · Mixtral · Mamba", "The open recipe (pre-RMSNorm, RoPE, SwiGLU, GQA) · SWA · open MoE · selective SSM",
         "Strong open models; long context; sparse compute"),
        (2024, "DeepSeek-V2/V3 · Llama 3 · Gemma 2", "MLA + fine-grained MoE; aux-loss-free; MTP; FP8 training · 128K vocab, 15T tokens · local/global + soft-cap",
         "KV cache ↓ ~57×; frontier quality at low cost"),
        (2025, "DeepSeek-R1 · Kimi K2 · Qwen3 · Qwen3-Next · V3.2 · MiniMax M1/M2 · gpt-oss",
         "RL reasoning (GRPO) · MuonClip · QK-norm, hybrid thinking · Gated DeltaNet + gated attention · DSA · lightning hybrid → back to full · sinks/SWA",
         "Reasoning via RL; stability; linear/sparse attention for long context"),
        (2026, "GLM-5 · Qwen3.5 · Kimi K2.5/K3 · DeepSeek-V4 · AttnRes · mHC",
         "MLA+DSA · 3:1 DeltaNet hybrid, native multimodal · KDA + AttnRes, 16/896 experts, MXFP4 QAT · CSA/HCA, mHC, Muon, FP4 experts, 1M ctx",
         "Memory/bandwidth is the bottleneck: compress KV, sparsify everything, low-bit by design"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["Year", "Milestone", "What changed", "Why it mattered"]),
                 hide_index=True, width="stretch")
    st.markdown(
        "**The arc in one sentence:** compute went from dense to sparse (MoE), and the bottleneck moved from FLOPs to "
        "memory and bandwidth — so attention was redesigned to shrink the KV cache (GQA → MLA → linear/sparse/compressed) "
        "and weights went to FP8/FP4, while optimisers (Muon) and residual designs (mHC, AttnRes) made huge sparse models "
        "stable to train."
    )

    sources(SOURCES["timeline"])

with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**2017 encoder/decoder layer (post-norm)**")
        st.graphviz_chart("""
digraph { rankdir=BT; node [shape=box, style="rounded,filled", fillcolor="#eef4fc", fontname=Helvetica, fontsize=10];
 x [label="x + sinusoidal PE"]; a [label="Multi-head attention (MHA)"]; n1 [label="Add & LayerNorm"];
 f [label="FFN: ReLU, 4d"]; n2 [label="Add & LayerNorm"]; o [label="out"];
 x -> a -> n1 -> f -> n2 -> o; x -> n1 [style=dashed, label=" residual"]; n1 -> n2 [style=dashed]; }""")
    with c2:
        st.markdown("**2026 frontier decoder block (one common pattern)**")
        st.graphviz_chart("""
digraph { rankdir=BT; node [shape=box, style="rounded,filled", fillcolor="#fdf0e8", fontname=Helvetica, fontsize=10];
 x [label="n residual streams (mHC) / AttnRes"]; r1 [label="RMSNorm"];
 a [label="Token mixer: GQA/MLA + RoPE (partial)\\nor Gated DeltaNet / KDA (3 of 4 layers)\\nor sparse/compressed (DSA, CSA/HCA)\\n+ QK-norm, output gate"];
 r2 [label="RMSNorm"]; f [label="MoE FFN: router top-k of 256-896\\nfine-grained SwiGLU experts + shared expert"];
 o [label="out (+ MTP head at the top)"];
 x -> r1 -> a -> r2 -> f -> o; x -> r2 [style=dashed, label=" residual mix"]; r2 -> o [style=dashed]; }""")
    st.dataframe(pd.DataFrame([
        ("Normalisation", "Post-LayerNorm", "Pre-RMSNorm (+QK-norm)", "Stable deep training; cheaper"),
        ("Positions", "Sinusoidal, added to input", "RoPE on q,k (large base, partial); YaRN to extend", "Relative, cache-friendly, extendable"),
        ("Attention", "MHA", "GQA / MLA / hybrids with linear or sparse layers", "KV cache & long-context cost"),
        ("FFN", "ReLU, 4d", "SwiGLU; mostly MoE", "Quality per FLOP; params without compute"),
        ("Biases / dropout", "Yes / 0.1", "No / 0", "Simpler, stabler, data ≫ params"),
        ("Residual", "x + F(x)", "Multi-stream, constrained (mHC) or attention over depth (AttnRes)", "Avoid dilution; stability"),
        ("Output", "Next token", "Next token + MTP heads", "Denser signal; speculative decoding"),
        ("Precision", "FP32", "BF16/FP8 training, FP4/INT4 QAT weights", "Memory & throughput"),
        ("Optimiser", "Adam", "AdamW → Muon (Kimi, DeepSeek-V4)", "Token efficiency"),
    ], columns=["Component", "2017", "2026", "Why"]), hide_index=True, width="stretch")
    if st.button("Read week 11 — the evolved transformer"):
        go_theory(11)

    sources(SOURCES["block"])

with tabs[2]:
    st.dataframe(pd.DataFrame([
        ("Encoder-only", "Bidirectional", "Masked LM / masked prediction", "BERT, RoBERTa, DeBERTa, HuBERT, WavLM", "Classification, embeddings, speech representations"),
        ("Decoder-only", "Causal", "Next token", "GPT, Llama, Qwen, DeepSeek, Kimi, GLM", "Generation, general-purpose LLMs"),
        ("Encoder–decoder", "Bi + causal + cross", "Seq2seq / span corruption", "T5, BART, Whisper", "ASR, translation, summarisation"),
        ("Prefix LM", "Bi on prefix, causal after", "LM with bidirectional prompt", "UL2, some VLMs", "Conditioned generation"),
        ("ViT", "Bidirectional over patches", "Supervised / MAE / DINO", "ViT, DeiT, MoonViT", "Vision encoders"),
        ("Conformer", "Bi (or chunked) + conv module", "CTC / RNN-T / attention", "Conformer, Zipformer", "ASR encoders"),
        ("DiT", "Bidirectional over latent patches", "Diffusion / flow matching", "SD3, Flux, video models", "Image/video/audio generation"),
        ("Hybrid linear/full", "Causal, mostly recurrent layers", "Next token", "Qwen3.5, Kimi Linear/K3, MiniMax-01", "Long context with small state"),
        ("SSM", "Recurrent (selective scan)", "Next token", "Mamba, Mamba-2", "Streaming, long sequences"),
    ], columns=["Family", "Attention", "Objective", "Examples", "Good for"]), hide_index=True, width="stretch")

    sources(SOURCES["families"])

with tabs[3]:
    st.markdown("KV bytes per token per layer (BF16) assume head dim 128; `h` = query heads, `g` = KV heads.")
    st.dataframe(pd.DataFrame([
        ("MHA", "2·h·128·2 B", "O(T²) prefill, O(T) per decode step", "Original, BERT, GPT-3"),
        ("MQA", "2·1·128·2 B", "same", "PaLM, Falcon"),
        ("GQA", "2·g·128·2 B", "same", "Llama 2-70B/3, Qwen, Mistral, Gemma, MiniMax M2"),
        ("MLA", "(512+64)·2 B", "same, + up-projections (absorbed)", "DeepSeek V2–V4, Kimi K2/K3, GLM-5"),
        ("Sliding window", "capped at window w", "O(T·w)", "Mistral, Gemma 2/3 (local layers), gpt-oss"),
        ("Linear / Lightning", "constant state d_k·d_v", "O(T)", "MiniMax-01/M1 (7:1 hybrid)"),
        ("Gated DeltaNet", "constant state", "O(T), chunkwise parallel", "Qwen3-Next, Qwen3.5 (3:1 hybrid)"),
        ("KDA", "constant state", "O(T)", "Kimi Linear, Kimi K3"),
        ("NSA / MoBA", "full cache, sparse reads", "O(T·k)", "DeepSeek NSA research, Moonshot MoBA"),
        ("DSA", "full MLA cache + small indexer cache", "main O(T·k), indexer O(T²) tiny", "DeepSeek-V3.2, GLM-5"),
        ("CSA / HCA", "~4× / 128× compressed, FP8", "sparse over compressed / dense over very compressed", "DeepSeek-V4"),
    ], columns=["Variant", "KV cache / token / layer", "Compute", "Used by"]), hide_index=True, width="stretch")
    lab_link("kv_cache", "KV-cache calculator lab")
    lab_link("efficient_attention", "Efficient attention lab")

    sources(SOURCES["attention"])

with tabs[4]:
    st.dataframe(pd.DataFrame([
        ("Sinusoidal", "Added to input", "None", "Poor in practice", "Original Transformer"),
        ("Learned absolute", "Added to input", "Table of max length", "None beyond table", "BERT, GPT-2, ViT"),
        ("Relative (Shaw)", "Added to K/V", "Clipped offsets", "Moderate", "Early relative models"),
        ("T5 bias", "Bias on logits", "Per head per log-bucket", "Good", "T5"),
        ("ALiBi", "Linear penalty on logits", "None (fixed slopes)", "Good (recency bias)", "BLOOM, MPT"),
        ("RoPE", "Rotate q, k", "None (base θ)", "With PI / NTK / YaRN", "Almost all modern LLMs"),
        ("Partial RoPE", "Rotate a fraction of dims", "None", "Better", "GPT-NeoX, Qwen3.5 (25%)"),
        ("Decoupled RoPE", "Separate small rotary key", "None", "With YaRN", "DeepSeek MLA"),
        ("NoPE", "Nothing (causal mask only)", "None", "Mixed", "Some layers in hybrids"),
    ], columns=["Scheme", "How", "Params", "Length extrapolation", "Used by"]), hide_index=True, width="stretch")
    st.latex(r"\mathrm{RoPE:}\quad \mathrm{Re}\big[q e^{im\theta}\,\overline{k e^{in\theta}}\big]=\mathrm{Re}\big[q\bar k\,e^{i(m-n)\theta}\big],\quad \theta_i=b^{-2i/d}")
    lab_link("attention", "Attention & RoPE lab")

    sources(SOURCES["positions"])

with tabs[5]:
    st.dataframe(pd.DataFrame([
        ("Mixtral 8×7B", "8", "2", "0", "Aux loss", "47B / 13B"),
        ("DeepSeek-V3", "256", "8", "1", "Aux-loss-free bias (+ tiny seq loss), node-limited", "671B / 37B"),
        ("Kimi K2", "384", "8", "1", "DeepSeek-style", "1T / 32B"),
        ("Qwen3-Next 80B", "512", "10", "1", "Load-balancing loss", "80B / 3B"),
        ("GLM-5", "256", "8", "—", "—", "≈744B / ≈40B"),
        ("DeepSeek-V4-Flash", "256", "6", "1", "Hash routing in first layers", "284B / 13B"),
        ("Kimi K3", "896", "16", "2", "Stable LatentMoE", "2.8T / ~104B"),
    ], columns=["Model", "Routed experts", "Active", "Shared", "Balancing", "Total / active params"]),
        hide_index=True, width="stretch")
    st.markdown("Router: `s = softmax or sigmoid(W_r x)`, pick top-k, renormalise. Aux loss: "
                "`E · Σ f_i P_i`. Aux-loss-free: bias added **only for selection**, nudged by ±γ toward balance.")
    lab_link("moe", "MoE routing lab")

    sources(SOURCES["moe"])

with tabs[6]:
    st.dataframe(pd.DataFrame([
        ("Data parallel (DDP)", "Batch", "All-reduce grads", "Model fits on one GPU"),
        ("ZeRO-1/2/3, FSDP", "Optimiser state / grads / params", "Reduce-scatter + all-gather", "Model state too big"),
        ("Tensor parallel", "Matmuls (column → row)", "All-reduce per block", "Within a node (NVLink)"),
        ("Sequence parallel", "Norm/dropout activations along T", "Reduce-scatter/all-gather", "With TP, long sequences"),
        ("Pipeline parallel", "Layers into stages", "Point-to-point activations", "Across nodes; 1F1B, DualPipe"),
        ("Context parallel", "Sequence (ring attention)", "Ring K/V passing", "Very long context"),
        ("Expert parallel", "Experts across GPUs", "All-to-all dispatch/combine", "MoE"),
    ], columns=["Strategy", "What's split", "Communication", "When"]), hide_index=True, width="stretch")
    st.markdown("Ring all-reduce moves `2(N−1)/N · S` bytes per GPU. Mixed-precision Adam ≈ 16 bytes/param. "
                "Pipeline bubble ≈ `(p−1)/(m+p−1)`. Full theory: week 15.")
    if st.button("Read week 15 — multi-GPU & GPU programming"):
        go_theory(15)

    sources(SOURCES["multi_gpu"])

with tabs[7]:
    st.dataframe(pd.DataFrame([
        ("Weights", "Quantise (W8A8, W4A16, FP8/FP4), distil, prune, MoE", "Decode speed ∝ bytes/token"),
        ("KV cache", "GQA/MLA, FP8/INT4 KV, paging, prefix caching, eviction, sparse/linear attention", "Memory, batch size, long context"),
        ("Kernels", "FlashAttention, FlashDecoding, fused norms/activations, CUDA graphs", "HBM traffic & launch overhead"),
        ("Batching", "Continuous batching, chunked prefill", "Throughput without TPOT spikes"),
        ("Scheduling", "Prefill/decode disaggregation, wide EP for MoE decode", "Different bottlenecks per phase"),
        ("Decoding", "Speculative (draft model, Medusa, EAGLE, MTP), prompt lookup", "More tokens per memory-bound pass"),
        ("On device", "4-bit weights + int8 acts, mmap, NPU prefill/CPU decode, bounded KV", "RAM, bandwidth, thermals, energy"),
    ], columns=["Bottleneck", "Techniques", "Why"]), hide_index=True, width="stretch")
    st.latex(r"\text{decode tokens/s}\approx\frac{\text{memory bandwidth}}{\text{active weight bytes}+\text{KV bytes read}}\qquad \mathbb E[\text{tokens/verify}]=\frac{1-\alpha^{\gamma+1}}{1-\alpha}")
    if st.button("Read week 14 — inference optimisation"):
        go_theory(14)

    sources(SOURCES["inference"])

with tabs[8]:
    path = [(10, "Tokens, embeddings, attention, positional encodings"), (11, "How the block evolved 2017 → 2026"),
            (12, "Scaling laws & MoE"), (13, "Efficient & long-context attention"),
            (14, "Inference optimisation"), (15, "Multi-GPU training & GPU programming"),
            (16, "Generative & self-supervised models"), (18, "The DeepSeek lineage"),
            (19, "Qwen, Kimi, GLM, MiniMax")]
    for wk, title in path:
        c1, c2 = st.columns([0.8, 0.2], vertical_alignment="center")
        c1.markdown(f"**Week {wk}** · {title}")
        if c2.button("Open", key=f"hub-{wk}", width="stretch"):
            go_theory(wk)
