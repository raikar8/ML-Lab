import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.caption("Weeks 11 & 14 · Lab")
st.title("KV cache, GQA/MLA & decoding")
tab1, tab2 = st.tabs(["KV-cache & decode-speed calculator", "Sampling: temperature, top-k, top-p"])

with tab1:
    st.markdown("Compare KV-cache memory per sequence and the memory-bound decode speed for different attention "
                "designs. Decode tokens/s ≈ bandwidth ÷ (active weight bytes + KV bytes read per token).")
    c1, c2, c3, c4 = st.columns(4)
    L = c1.number_input("Layers", 1, 200, 32)
    h = c2.number_input("Query heads", 1, 256, 32)
    dh = c3.number_input("Head dim", 16, 512, 128)
    ctx = c4.number_input("Context tokens", 128, 1_048_576, 8192, step=1024)
    c1, c2, c3, c4 = st.columns(4)
    kv_bytes = c1.selectbox("KV dtype", {"BF16": 2, "FP8/INT8": 1, "INT4": 0.5}, format_func=lambda k: k)
    kvb = {"BF16": 2, "FP8/INT8": 1, "INT4": 0.5}[kv_bytes]
    params = c2.number_input("Active params (B)", 0.1, 200.0, 7.0)
    wbits = c3.selectbox("Weight bits", [16, 8, 4], index=2)
    bw = c4.number_input("Memory bandwidth (GB/s)", 10.0, 8000.0, 60.0, help="Phone ≈ 50–80, laptop ≈ 100–400, H100 ≈ 3350")
    g = st.slider("GQA KV heads", 1, int(h), min(8, int(h)))
    hybrid = st.slider("Hybrid: fraction of layers that are full attention (rest linear/DeltaNet)", 0.1, 1.0, 1.0, 0.05)
    variants = {
        "MHA": 2 * h * dh, "GQA": 2 * g * dh, "MQA": 2 * dh, "MLA (512+64)": 576,
        f"GQA hybrid ({hybrid:.0%} full)": 2 * g * dh * hybrid,
    }
    rows = []
    for name, per_layer in variants.items():
        per_tok = per_layer * L * kvb
        total = per_tok * ctx
        tps = bw * 1e9 / (params * 1e9 * wbits / 8 + total)
        rows.append({"Design": name, "KV / token": per_tok, "KV per sequence (GB)": total / 1e9, "Decode tok/s": tps})
    df = pd.DataFrame(rows)
    st.dataframe(df.style.format({"KV / token": lambda v: f"{v / 1024:.1f} KB", "KV per sequence (GB)": "{:.2f}",
                                  "Decode tok/s": "{:.1f}"}), hide_index=True, width="stretch")
    xs = np.linspace(1024, max(ctx, 2048), 50)
    fig = go.Figure()
    for name, per_layer in variants.items():
        fig.add_scatter(x=xs, y=bw * 1e9 / (params * 1e9 * wbits / 8 + per_layer * L * kvb * xs), name=name)
    fig.update_layout(height=320, xaxis_title="context tokens", yaxis_title="decode tokens/s (upper bound)",
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Linear-attention layers in hybrids keep a constant-size state (not counted here); the full-attention "
               "fraction is what grows with context.")

with tab2:
    st.markdown("See how temperature, top-k and top-p reshape a next-token distribution.")
    rng = np.random.default_rng(3)
    logits = np.sort(rng.normal(0, 2, 30))[::-1]
    c1, c2, c3 = st.columns(3)
    T = c1.slider("Temperature", 0.05, 2.0, 1.0, 0.05)
    k = c2.slider("Top-k", 1, 30, 30)
    p = c3.slider("Top-p", 0.05, 1.0, 1.0, 0.05)
    z = logits / T
    probs = np.exp(z - z.max()); probs /= probs.sum()
    keep = np.zeros_like(probs, bool); keep[:k] = True
    cs = np.cumsum(probs)
    keep &= (cs - probs) < p
    final = np.where(keep, probs, 0); final /= final.sum()
    fig = go.Figure()
    fig.add_bar(y=probs, name="after temperature", opacity=0.4)
    fig.add_bar(y=final, name="after top-k / top-p")
    fig.update_layout(barmode="overlay", height=320, xaxis_title="token rank", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    ent = -(final[final > 0] * np.log(final[final > 0])).sum()
    st.markdown(f"Tokens kept: **{int(keep.sum())}** · entropy **{ent:.2f} nats**. Top-p always keeps at least the top token.")
