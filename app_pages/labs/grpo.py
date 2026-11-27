import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.caption("Week 17 · Lab")
st.title("RL post-training: PPO vs GRPO vs CISPO")
tab1, tab2 = st.tabs(["Group advantages", "Clipping & gradient weight"])
with tab1:
    G = st.slider("Group size G", 2, 32, 8); p = st.slider("Probability a sample is correct", 0.0, 1.0, 0.3)
    rng = np.random.default_rng(0)
    r = (rng.random(G) < p).astype(float)
    std = r.std()
    adv = (r - r.mean()) / (std + 1e-6)
    st.bar_chart({"reward": r, "GRPO advantage": adv}, height=220)
    trials = (rng.random((20000, G)) < p).astype(float)
    useless = np.mean(trials.std(1) == 0)
    st.markdown(f"Fraction of prompts where **all G answers get the same reward** (zero advantage, no learning signal): "
                f"**{useless:.1%}** — why DAPO's *dynamic sampling* drops them and why prompt difficulty matters.")
with tab2:
    st.markdown("How much gradient a token gets (as a function of the importance ratio π_new/π_old) for a positive "
                "advantage.")
    eps = st.slider("ε (clip)", 0.05, 0.5, 0.2); eh = st.slider("ε_high (DAPO clip-higher)", 0.05, 1.0, 0.28)
    ratio = np.linspace(0.3, 2.0, 400)
    ppo = np.where(ratio <= 1 + eps, ratio, 0.0)
    dapo = np.where(ratio <= 1 + eh, ratio, 0.0)
    cispo = np.minimum(ratio, 1 + eh)
    fig = go.Figure()
    fig.add_scatter(x=ratio, y=ppo, name="PPO/GRPO (A>0): grad weight, zero once clipped")
    fig.add_scatter(x=ratio, y=dapo, name="DAPO clip-higher")
    fig.add_scatter(x=ratio, y=cispo, name="CISPO: IS weight clipped, gradient kept")
    fig.update_layout(height=360, xaxis_title="ratio", yaxis_title="effective gradient weight",
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")
    st.caption("Rare tokens whose probability jumps (e.g. 'wait') hit the clip and stop learning under PPO/GRPO; CISPO "
               "keeps their gradient with a bounded weight.")
