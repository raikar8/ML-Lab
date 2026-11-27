import ast

import pandas as pd
import streamlit as st

st.caption("Week 7 · Lab")
st.title("CNN cost calculator")
st.markdown("Compare a standard conv with depthwise-separable and inverted-residual blocks: parameters, MACs, activation "
            "memory and arithmetic intensity (int8).")
c1, c2, c3, c4 = st.columns(4)
C_in = c1.number_input("C_in", 1, 2048, 64); C_out = c2.number_input("C_out", 1, 2048, 64)
H = c3.number_input("H = W", 1, 512, 56); k = c4.selectbox("Kernel", [1, 3, 5, 7], index=1)
s = st.selectbox("Stride", [1, 2]); e = st.slider("Expansion (inverted residual)", 1, 8, 6)
Ho = (H + 2 * (k // 2) - k) // s + 1


def row(name, params, macs, act):
    return {"Block": name, "Params": params, "MACs (M)": macs / 1e6, "Activation KB (int8)": act / 1024,
            "Ops/byte": 2 * macs / (params + act)}


std = row("Standard conv", C_out * C_in * k * k, C_out * C_in * k * k * Ho * Ho, C_in * H * H + C_out * Ho * Ho)
dw = C_in * k * k; pw = C_in * C_out
dws = row("Depthwise + pointwise", dw + pw, dw * Ho * Ho + pw * Ho * Ho, C_in * H * H + C_in * Ho * Ho + C_out * Ho * Ho)
hid = C_in * e
ir_p = C_in * hid + hid * k * k + hid * C_out
ir = row(f"Inverted residual (×{e})", ir_p, C_in * hid * H * H + hid * k * k * Ho * Ho + hid * C_out * Ho * Ho,
         C_in * H * H + hid * H * H + hid * Ho * Ho + C_out * Ho * Ho)
dwo = row("Depthwise only", dw, dw * Ho * Ho, C_in * H * H + C_in * Ho * Ho)
df = pd.DataFrame([std, dws, ir, dwo])
st.dataframe(df.style.format({"Params": "{:,}", "MACs (M)": "{:.2f}", "Activation KB (int8)": "{:.1f}", "Ops/byte": "{:.1f}"}),
             hide_index=True, width="stretch")
st.markdown(f"Output {Ho}×{Ho}×{C_out}. Depthwise-separable saves **{std['MACs (M)'] / dws['MACs (M)']:.1f}×** MACs "
            f"(theory: 1/(1/C_out + 1/k²) = {1 / (1 / C_out + 1 / k ** 2):.1f}×), but the depthwise part has very low "
            "ops/byte — it's memory-bound, so real speed-ups are smaller. Inverted residuals also have large hidden "
            "activations: check peak SRAM on an MCU.")
rf = 1; j = 1
layers = st.text_input("Receptive field: list of (k, stride, dilation)", "(3,1,1),(3,2,1),(3,1,2),(3,1,4)")
try:
    spec = ast.literal_eval("[" + layers + "]")
except (ValueError, SyntaxError):
    spec = []
    st.warning("Use the form (k,stride,dilation),(k,stride,dilation),...")
for kk, ss, dd in spec:
    rf += (dd * (kk - 1)) * j; j *= ss
st.metric("Receptive field", rf)
