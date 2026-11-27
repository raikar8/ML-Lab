import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.caption("Week 13 · Lab")
st.title("Efficient attention")
tab1, tab2 = st.tabs(["Cost vs context", "Delta rule vs additive memory"])
with tab1:
    d = st.slider("Model width d", 512, 8192, 4096, 512)
    w = st.slider("Sliding window", 256, 8192, 4096, 256); k = st.slider("Sparse top-k", 256, 8192, 2048, 256)
    T = np.logspace(3, 6, 60)
    fig = go.Figure()
    fig.add_scatter(x=T, y=4 * T * T * d, name="full softmax O(T²d)")
    fig.add_scatter(x=T, y=4 * T * np.minimum(T, w) * d, name="sliding window O(T·w·d)")
    fig.add_scatter(x=T, y=4 * T * np.minimum(T, k) * d + 2 * T * T * 64, name="DSA-style top-k + tiny indexer")
    fig.add_scatter(x=T, y=4 * T * d * 128, name="linear / DeltaNet O(T·d·d_head)")
    fig.add_scatter(x=T, y=0.25 * 4 * T * T * d + 0.75 * 4 * T * d * 128, name="3:1 hybrid (Qwen3.5-style)")
    fig.update_layout(xaxis_type="log", yaxis_type="log", height=380, xaxis_title="context tokens T",
                      yaxis_title="attention FLOPs per layer (prefill)", margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
with tab2:
    st.markdown("Write N key→value pairs into a d×d memory, then **overwrite** half of them with new values, and read back. "
                "Additive linear attention accumulates both values; the delta rule (an NLMS step) replaces the old one.")
    D = st.slider("State size (d_k)", 8, 128, 32); N = st.slider("Pairs", 2, 128, 16); beta = st.slider("β", 0.1, 1.0, 1.0)
    rng = np.random.default_rng(0)
    K = rng.normal(size=(N, D)); K /= np.linalg.norm(K, axis=1, keepdims=True)
    V0, V1 = rng.normal(size=(N, D)), rng.normal(size=(N, D))
    seq = [(K[i], V0[i]) for i in range(N)] + [(K[i], V1[i]) for i in range(N // 2)]
    S_add, S_del = np.zeros((D, D)), np.zeros((D, D))
    for kk, vv in seq:
        S_add += np.outer(vv, kk)
        S_del += beta * np.outer(vv - S_del @ kk, kk)
    target = np.r_[V1[: N // 2], V0[N // 2:]]
    err = lambda S: np.mean(np.linalg.norm(K @ S.T - target, axis=1) / np.linalg.norm(target, axis=1))
    a, b = st.columns(2)
    a.metric("Additive recall error", f"{err(S_add):.2f}"); b.metric("Delta-rule recall error", f"{err(S_del):.2f}")
    st.caption("When N > d_k the memory is over capacity and both degrade — that's why hybrids keep a few full-attention layers.")
