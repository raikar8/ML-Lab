"""Chinese model atlas: the major open models from Chinese labs and their signature techniques."""
import pandas as pd
import streamlit as st

from mlprep.ui import go_theory

st.title("Chinese model atlas")
st.markdown(
    "A quick-reference table of the major open-weight models from Chinese labs and the techniques each introduced or "
    "popularised. Use it to revise before an interview; the reasoning behind each technique is in weeks 17–19 "
    "(and 11–15 for the architecture pieces). Figures are from the labs' reports and model cards as of November 2026 "
    "(≈ = approximate / as reported)."
)

rows = [
    ("DeepSeek", "DeepSeek-V2", "2024-05", "236B / 21B", "MLA; DeepSeekMoE (2 shared + 160 routed, 6 active)", "KV cache −93% vs 67B dense; 5.8× max throughput"),
    ("DeepSeek", "DeepSeek-V3", "2024-12", "671B / 37B", "Aux-loss-free MoE (1+256, top-8), MTP, FP8 training, DualPipe, node-limited routing", "2.788M H800 GPU-hours"),
    ("DeepSeek", "DeepSeek-R1", "2025-01", "671B / 37B", "R1-Zero pure RL with GRPO; cold-start + multi-stage RL; distilled 1.5–70B models", "Open reasoning model"),
    ("DeepSeek", "DeepSeek-V3.2", "2025-09/12", "671B / 37B", "DSA sparse attention (lightning indexer, top-2048); scaled RL; agentic data synthesis", "Much cheaper long context"),
    ("DeepSeek", "DeepSeek-V4-Pro / Flash", "2026-04", "1.6T / 49B · 284B / 13B", "CSA + HCA hybrid attention, mHC (4 streams), Muon, FP4 QAT experts, hash routing, 1M ctx, on-policy distillation", "1M ctx at ~27% FLOPs / ~10% KV of V3.2"),
    ("Qwen (Alibaba)", "Qwen3", "2025-04", "0.6B–32B dense; 30B-A3B, 235B-A22B", "Hybrid thinking + thinking budget, QK-norm, strong-to-weak distillation, GSPO RL", "36T tokens, 119 languages"),
    ("Qwen (Alibaba)", "Qwen3-Next", "2025-09", "80B / 3B", "3:1 Gated DeltaNet : gated attention, 512 experts (10+1), MTP, output gating", "~10% of Qwen3-32B training cost"),
    ("Qwen (Alibaba)", "Qwen3.5", "2026-02", "0.8B–9B, 27B; 35B-A3B, 122B-A10B, 397B-A17B", "3:1 DeltaNet hybrid across the family, native multimodal, 262K native ctx", "Small hybrid models run on device"),
    ("Kimi (Moonshot)", "Kimi K2", "2025-07", "1T / 32B", "MuonClip (Muon + QK-clip), 384 experts (8+1), MLA with 64 heads, agentic data synthesis, rubric RL", "15.5T tokens, no loss spikes"),
    ("Kimi (Moonshot)", "Kimi K2 Thinking", "2025-11", "1T / 32B", "Long-horizon tool use; native INT4 weights via QAT", "~2× faster generation"),
    ("Kimi (Moonshot)", "Kimi Linear", "2025-10", "48B / 3B", "Kimi Delta Attention (channel-wise gated delta rule), 3:1 KDA : MLA", "Up to ~6× faster decoding at 1M"),
    ("Kimi (Moonshot)", "Kimi K2.5", "2026-01", "1T / 32B", "Native multimodal (MoonViT-3D, ~15T mixed tokens), Agent Swarm (PARL)", "Parallel sub-agents"),
    ("Kimi (Moonshot)", "Kimi K3", "2026-07", "2.8T / ≈104B", "69 KDA + 24 gated-MLA layers, AttnRes, 896 experts (16+2), MXFP4 weights / MXFP8 acts via QAT", "Largest open-weight model at release"),
    ("Zhipu / Z.ai", "GLM-4.5", "2025-07", "355B / 32B (Air 106B / 12B)", "Unified agentic/reasoning/coding, deeper-not-wider, slime RL framework", "Agentic ARC model"),
    ("Zhipu / Z.ai", "GLM-5 (5.1, 5.2)", "2026-02", "≈744B / ≈40B", "MLA + DSA, 256 experts (8 active), async RL infra; IndexShare in 5.2", "Top open model on AA index at release"),
    ("MiniMax", "MiniMax-01", "2025-01", "456B / 46B", "Lightning attention hybrid (7 linear : 1 softmax), 1–4M ctx", "Long context"),
    ("MiniMax", "MiniMax-M1", "2025-06", "456B / 46B", "Reasoning on the hybrid; CISPO RL", "Full RL ≈ 3 weeks on 512 H800s"),
    ("MiniMax", "MiniMax-M2 / M2.5 / M2.7", "2025-10 →", "230B / 10B", "Back to full attention (GQA, QK-norm, MTP); interleaved thinking; agentic SWE RL", "Efficient agent model"),
    ("ByteDance Seed", "Seed series · DAPO · HC", "2024–26", "various", "Hyper-connections (basis of mHC), DAPO RL (with Tsinghua), UltraMem", "Methods widely adopted"),
    ("Meituan", "LongCat-Flash", "2025-09", "560B / ~27B (dynamic)", "Zero-computation experts → variable compute per token; shortcut-connected MoE", "Dynamic compute"),
    ("StepFun", "Step-3", "2025-07", "321B / 38B", "Attention–FFN disaggregation for cheaper decoding", "Decode-cost co-design"),
    ("Baidu", "ERNIE 4.5", "2025-06", "up to 424B / 47B", "Heterogeneous multimodal MoE", "Open multimodal MoE"),
]
df = pd.DataFrame(rows, columns=["Lab", "Model", "Released", "Total / active params", "Signature techniques", "Notable"])

labs = st.multiselect("Labs", sorted(df.Lab.unique()), default=[])
q = st.text_input("Search techniques (e.g. MLA, GRPO, DeltaNet, FP4)")
view = df
if labs:
    view = view[view.Lab.isin(labs)]
if q:
    view = view[view.apply(lambda r: q.lower() in " ".join(map(str, r)).lower(), axis=1)]
st.dataframe(view, hide_index=True, width="stretch", height=min(800, 38 * (len(view) + 1)))

st.subheader("Technique → where it came from → who adopted it")
st.dataframe(pd.DataFrame([
    ("MLA (latent KV compression)", "DeepSeek-V2", "DeepSeek V3/V4, Kimi K2/K3 (gated MLA), GLM-5", "W11, W18"),
    ("Fine-grained + shared experts", "DeepSeekMoE", "Qwen3/3.5, Kimi, GLM, most open MoEs", "W12"),
    ("Aux-loss-free balancing", "DeepSeek-V3", "Many DeepSeek-derived MoEs", "W12"),
    ("GRPO (critic-free RL)", "DeepSeekMath / R1", "Nearly every open reasoning pipeline; variants DAPO, GSPO, CISPO", "W17"),
    ("MTP heads", "DeepSeek-V3", "Qwen3-Next/3.5, MiniMax M2, GLM, V4", "W14, W18"),
    ("FP8 training (fine-grained scaling)", "DeepSeek-V3", "Industry-wide; MX formats in hardware", "W6, W23"),
    ("Sparse attention DSA", "DeepSeek-V3.2", "GLM-5 (+ IndexShare)", "W13"),
    ("Gated DeltaNet hybrids", "Qwen3-Next (from NVIDIA/MIT research)", "Qwen3.5 family", "W13"),
    ("KDA / AttnRes", "Moonshot", "Kimi Linear, K3", "W11, W13"),
    ("Muon / MuonClip", "Moonshot (Muon by Jordan et al.)", "Kimi K2/K3, DeepSeek-V4", "W6"),
    ("mHC", "DeepSeek (HC from ByteDance)", "DeepSeek-V4", "W11"),
    ("Low-bit QAT weights", "Kimi K2 Thinking (INT4), V4 (FP4), K3 (MXFP4)", "Trend for serving & on-device", "W22, W23"),
], columns=["Technique", "Origin", "Adopted by", "Weeks"]), hide_index=True, width="stretch")

c1, c2, c3 = st.columns(3)
if c1.button("Week 17 — post-training & GRPO"):
    go_theory(17)
if c2.button("Week 18 — DeepSeek lineage"):
    go_theory(18)
if c3.button("Week 19 — Qwen, Kimi, GLM, MiniMax"):
    go_theory(19)
