"""Frontier watch: ongoing topics worth tracking, and a routine for staying current."""
import pandas as pd
import streamlit as st

from mlprep.ui import sources

st.title("Frontier watch")
st.markdown(
    "Topics that were still moving fast in late 2026. Interviewers often end with *\"what's something recent you "
    "found interesting?\"* — pick two of these, read one primary source for each, and prepare a 2-minute summary with "
    "your own opinion. The table says why each matters for an **embedded / audio ML** engineer in particular."
)

st.dataframe(pd.DataFrame([
    ("Hybrid & linear attention", "Gated DeltaNet / KDA / Mamba layers mixed with a few full-attention layers",
     "Constant-size state → long context on device; RNN-like streaming", "W13"),
    ("Compressed / sparse attention", "DSA, CSA/HCA, NSA; KV in FP8", "Mostly server-side, but KV compression ideas transfer", "W13, W18"),
    ("Residual-stream redesign", "mHC, Attention Residuals, hyper-connections", "Stability of deep models; watch if it reaches small models", "W11"),
    ("New optimisers", "Muon and variants; second-order-ish methods", "Token-efficient training of your own small models", "W6"),
    ("Low-bit by design", "FP4/INT4/MXFP4 QAT at frontier scale; ternary BitNet", "Directly relevant: QAT recipes, micro-scaled formats in NPUs", "W22, W23"),
    ("RL for reasoning & agents", "GRPO family, verifiable rewards, async RL, rubric rewards", "Distilled reasoning in small on-device models", "W17"),
    ("On-policy distillation", "Student samples, teacher scores each token", "Best recipe for small deployable models", "W17, W24"),
    ("Speculative decoding on device", "MTP/EAGLE heads, NPU prefill + CPU decode", "Faster local LLMs within bandwidth limits", "W14, W25"),
    ("Native multimodal & omni models", "Joint text–image–video–audio pre-training; speech in/out", "Voice assistants in cars; speech LLMs", "W8, W16"),
    ("Speech LLMs & full-duplex voice", "Codec tokens + LLMs; streaming interruptible dialogue", "Core to in-car assistants; latency/AEC interplay", "W8, W26"),
    ("Generative speech enhancement", "Flow matching / few-step diffusion for SE", "Quality at low SNR vs latency trade-off", "W16"),
    ("Edge runtimes", "ExecuTorch 1.x, LiteRT-LM, vendor NPU SDKs", "Your deployment toolchain; operator coverage changes quickly", "W25"),
    ("Hardware", "Blackwell FP4, Arm Ethos-U85, new mobile/automotive NPUs", "Know what the target supports natively", "W21"),
], columns=["Topic", "What's happening", "Why you care", "Weeks"]), hide_index=True, width="stretch")

START_HERE = {
    "Hybrid & linear attention": ["Gated Delta Networks", "Kimi Linear: An Expressive, Efficient Attention Architecture",
                                  "Qwen3-Next: Towards Ultimate Training & Inference Efficiency"],
    "Compressed / sparse attention": ["Native Sparse Attention",
                                      "DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models"],
    "Residual-stream redesign": ["Hyper-Connections", "mHC: Manifold-Constrained Hyper-Connections", "Attention Residuals"],
    "New optimisers": ["Muon: An optimizer for hidden layers", "Muon is Scalable for LLM Training"],
    "Low-bit by design": ["Microscaling Data Formats for Deep Learning", "The Era of 1-bit LLMs (BitNet b1.58)",
                          "gpt-oss-120b & gpt-oss-20b Model Card"],
    "RL for reasoning & agents": ["DeepSeekMath (introduces GRPO)", "DAPO: An Open-Source LLM Reinforcement Learning System at Scale",
                                  "Group Sequence Policy Optimization"],
    "Speculative decoding on device": ["EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"],
    "Generative speech enhancement": ["Speech Enhancement and Dereverberation with Diffusion-based Generative Models",
                                      "Flow Matching for Generative Modeling"],
    "Edge runtimes": ["ExecuTorch documentation", "TensorFlow Lite Micro: Embedded Machine Learning on TinyML Systems"],
}
st.subheader("Where to start reading")
c1, c2 = st.columns(2)
for i, (topic, titles) in enumerate(START_HERE.items()):
    sources(titles, label=topic, where=c1 if i % 2 == 0 else c2)

st.subheader("A 30-minute weekly routine to stay current")
st.markdown("""
1. **Skim** release notes and tech reports of 2–3 labs you follow (DeepSeek, Qwen, Kimi, Google, Meta, Apple, Arm,
   Qualcomm) and the changelog of one runtime you'd deploy with.
2. **Pick one paper** — alternate architecture/LLM weeks with efficiency/edge/audio weeks.
3. **Write five lines** using the week-19 template: what changed, why, the evidence, the cost, one question.
4. **Add one card** to your notes with the key number (e.g. "V4 at 1M ctx: ~10% KV of V3.2").
5. **Every month**, re-implement one idea at toy scale (an exercise-sized PyTorch version).
""")
st.info("Before you quote a number in an interview, check the primary source — model cards and reports are "
        "frequently updated, and secondary summaries often get details wrong.")
