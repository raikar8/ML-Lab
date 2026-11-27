import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.caption("Week 10 · Lab")
st.title("Attention & RoPE")
tab1, tab2 = st.tabs(["Attention weights", "RoPE relative property"])
with tab1:
    c1, c2, c3 = st.columns(3)
    T = c1.slider("Tokens", 4, 32, 12); temp = c2.slider("Logit scale (1/√d is 'normal')", 0.1, 5.0, 1.0)
    mask = c3.radio("Mask", ["none", "causal", "sliding window (4)", "causal + sink"])
    rng = np.random.default_rng(0)
    q, k = rng.normal(size=(T, 16)), rng.normal(size=(T, 16))
    s = q @ k.T / 4 * temp
    i, j = np.arange(T)[:, None], np.arange(T)[None]
    allowed = {"none": np.ones((T, T), bool), "causal": j <= i, "sliding window (4)": (j <= i) & (j > i - 4),
               "causal + sink": (j <= i) & ((j > i - 4) | (j == 0))}[mask]
    s = np.where(allowed, s, -np.inf)
    p = np.exp(s - s.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    st.plotly_chart(go.Figure(go.Heatmap(z=p, colorscale="Blues")).update_layout(height=380, yaxis_autorange="reversed",
                    xaxis_title="key", yaxis_title="query", margin=dict(l=10, r=10, t=10, b=10)), width="stretch")
    ent = -(np.where(p > 0, p * np.log(p), 0)).sum(1).mean()
    st.caption(f"Mean attention entropy {ent:.2f} nats. Large logit scale → near one-hot (saturated softmax, tiny gradients).")
with tab2:
    d = 64; base = st.select_slider("RoPE base", [100, 1000, 10000, 500000, 1000000], 10000)
    rng = np.random.default_rng(1); q, k = rng.normal(size=d), rng.normal(size=d)
    inv = base ** (-np.arange(0, d, 2) / d)

    def rot(x, pos):
        a = pos * inv; x1, x2 = x[: d // 2], x[d // 2:]
        return np.r_[x1 * np.cos(a) - x2 * np.sin(a), x1 * np.sin(a) + x2 * np.cos(a)]
    offs = np.arange(0, 200)
    s1 = [rot(q, 50 + o) @ rot(k, 50) for o in offs]; s2 = [rot(q, 1000 + o) @ rot(k, 1000) for o in offs]
    fig = go.Figure(); fig.add_scatter(y=s1, name="key at 50"); fig.add_scatter(y=s2, name="key at 1000", line=dict(dash="dot"))
    fig.update_layout(height=320, xaxis_title="query − key offset", yaxis_title="score", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("The two curves coincide: the score depends only on the offset. A larger base makes low-frequency pairs "
               "rotate more slowly — the knob NTK/YaRN turn to extend context.")
