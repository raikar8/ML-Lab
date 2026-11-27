import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.caption("Week 20 · Lab")
st.title("Roofline & fixed point")
tab1, tab2 = st.tabs(["Roofline", "Fixed point (Q15)"])

DEVICES = {
    "Cortex-M55 + Ethos-U55 (MCU)": (0.5, 1.0), "Audio DSP (HiFi-class)": (0.05, 2.0),
    "Phone NPU": (40.0, 60.0), "Laptop GPU": (100.0, 400.0), "Jetson Orin AGX": (170.0, 204.0),
    "H100 (BF16 dense)": (989.0, 3350.0),
}

with tab1:
    c1, c2 = st.columns(2)
    dev = c1.selectbox("Device preset", list(DEVICES))
    P, B = DEVICES[dev]
    P = c1.number_input("Peak compute (TOPS / TFLOPs)", 0.01, 5000.0, P)
    B = c2.number_input("Memory bandwidth (GB/s)", 0.1, 10000.0, B)
    bytes_per = c2.selectbox("Bytes per element", [1, 2, 4], index=0, help="int8 = 1, fp16/bf16 = 2")
    st.markdown("**Layers** — arithmetic intensity is computed from the shapes (ops ÷ bytes of weights + inputs + outputs).")
    M = st.slider("Tokens / batch rows (M) for linear layers", 1, 2048, 1)

    def lin(Mr, K, N):
        return 2 * Mr * K * N, (Mr * K + K * N + Mr * N) * bytes_per

    def conv(C, Co, k, H, W, groups=1):
        ops = 2 * Co * (C // groups) * k * k * H * W
        return ops, (C * H * W + Co * (C // groups) * k * k + Co * H * W) * bytes_per

    layers = {
        f"Linear 4096×4096, M={M}": lin(M, 4096, 4096),
        "Conv 3×3, 64→64, 56×56": conv(64, 64, 3, 56, 56),
        "Depthwise 3×3, 256 ch, 28×28": conv(256, 256, 3, 28, 28, groups=256),
        "Pointwise 1×1, 256→256, 28×28": conv(256, 256, 1, 28, 28),
        "Element-wise add, 1M elems": (1e6, 3e6 * bytes_per),
        "Attention decode, 4K ctx (read KV)": (2 * 2 * 4096 * 128, 2 * 4096 * 128 * bytes_per),
    }
    I = np.logspace(-1, 4, 200)
    perf = np.minimum(P * 1e12, I * B * 1e9) / 1e12
    fig = go.Figure()
    fig.add_scatter(x=I, y=perf, name="roofline", line=dict(width=3))
    rows = []
    for name, (ops, by) in layers.items():
        ai = ops / by
        att = min(P * 1e12, ai * B * 1e9)
        fig.add_scatter(x=[ai], y=[att / 1e12], mode="markers+text", text=[name.split(",")[0]], textposition="top center",
                        name=name)
        rows.append((name, f"{ai:.1f}", "compute" if ai > P * 1e3 / B else "memory", f"{ops / att * 1e6:.1f} µs"))
    fig.add_vline(x=P * 1e3 / B, line_dash="dot", annotation_text="ridge point")
    fig.update_layout(xaxis_type="log", yaxis_type="log", xaxis_title="ops per byte", yaxis_title="attainable T(FL)OP/s",
                      height=420, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.table({"Layer": [r[0] for r in rows], "Ops/byte": [r[1] for r in rows], "Bound": [r[2] for r in rows],
              "Best-case time": [r[3] for r in rows]})

with tab2:
    st.markdown("Multiply two numbers in **Q15** with rounding and saturation, and see the quantization error.")
    c1, c2 = st.columns(2)
    a = c1.slider("a", -1.0, 0.99997, 0.5, 0.0001)
    b = c2.slider("b", -1.0, 0.99997, -0.75, 0.0001)
    qa, qb = int(np.clip(round(a * 32768), -32768, 32767)), int(np.clip(round(b * 32768), -32768, 32767))
    prod = qa * qb
    q = (prod + (1 << 14)) >> 15
    q_sat = int(np.clip(q, -32768, 32767))
    st.code(f"a = {qa} (Q15)   b = {qb} (Q15)\nproduct = {prod} (Q30)\n(prod + 2^14) >> 15 = {q}  → saturated {q_sat}\n"
            f"result = {q_sat / 32768:.6f}   exact = {a * b:.6f}   error = {q_sat / 32768 - a * b:+.2e}", language="text")
    st.caption("Try a = b = −1: the exact result +1 isn't representable in Q15 and must saturate to 32767.")
    bits = st.slider("Bits", 4, 16, 8)
    st.markdown(f"Ideal SQNR of a full-scale sine at {bits} bits ≈ **{6.02 * bits + 1.76:.1f} dB**.")
