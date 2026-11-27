"""GPU notebooks: run on free Colab/Kaggle GPUs; kernels written from scratch in gpu/."""
import json

import streamlit as st

from mlprep.content import ROOT

st.title("GPU notebooks")
st.markdown(
    "Some interview-relevant skills can't be practised on a laptop CPU: Triton kernels, kernel fusion and bandwidth, "
    "FlashAttention, CUDA graphs, DDP vs FSDP, FP8. These notebooks run on **free Colab or Kaggle GPUs**. The kernels and "
    "scripts live in `gpu/` (written from scratch, unit-tested on CPU with Triton's interpreter: "
    "`TRITON_INTERPRET=1 pytest tests/test_gpu_kernels.py`), and the notebooks benchmark them against PyTorch."
)
st.info("The notebooks clone this repository. If the repo is private, either make it public, or in Colab upload the "
        "`gpu/` folder, or clone with a GitHub token.")
for f in sorted((ROOT / "notebooks").glob("*.ipynb")):
    nb = json.loads(f.read_text(encoding="utf-8"))
    title_md = "".join(nb["cells"][1]["source"])
    title = title_md.splitlines()[0].lstrip("# ").strip()
    blurb = " ".join(title_md.splitlines()[1:])[:400]
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(blurb)
        c1, c2 = st.columns(2)
        c1.link_button("Open in Colab", f"https://colab.research.google.com/github/raikar8/ML-Lab/blob/main/notebooks/{f.name}")
        c2.code(f"notebooks/{f.name}", language="text")
