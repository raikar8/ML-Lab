# ML Interview Lab

A 27-week, day-by-day study app for **ML Engineer / Deep Learning / Embedded AI** interviews — from Python and
classical ML to transformers, LLM post-training, the techniques behind DeepSeek, Qwen, Kimi, GLM and MiniMax, and
model optimisation for on-device deployment (quantization, pruning, distillation, runtimes, real-time audio).

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## What's inside

| Page | What it does |
|---|---|
| **Today** | Your tasks for today (theory section, coding exercise, interview questions; lab day; review day), catch-up list, streak |
| **6-month roadmap** | All 27 weeks in 6 phases with progress and card mastery per week |
| **Review** | Spaced-repetition deck of ~270 interview questions (★ = high-yield), unlocked as you go |
| **Mock interview** | Timed question rounds with follow-up probes and self-scoring; weak answers go back into review |
| **Theory by week** | 5 theory sections per week with equations, interview angles, questions and a mini-project |
| **Coding exercises** | 73 Python / NumPy / PyTorch stubs with pytest tests and reference solutions; edit & run in the app |
| **C/C++ track** | 15 embedded/real-time C++ exercises (fixed point, FIR/biquad, SPSC ring buffer, memory pool, int8 GEMM + requantization, streaming conv, im2col, tiled GEMM…) with tests; compile & run from the app or with CMake |
| **Transformer hub** | Evolution 2017→2026, block anatomy then vs now, attention/positional/MoE cheat sheets |
| **Chinese model atlas** | DeepSeek, Qwen, Kimi, GLM, MiniMax models and their signature techniques |
| **Frontier watch** | Ongoing topics worth tracking, and how to keep current |
| **Answer grader** | Type your answer; a model (local llama.cpp/Ollama server or the Claude API) grades it against the reference — covered, missed, wrong, vague, plus a follow-up. Offline key-concept fallback |
| **Error log** | Bugs you made and questions you fumbled become review cards |
| **System design bank** | 14 embedded/audio/edge designs with clarify → numbers → design → trade-offs → follow-ups, timer and notes |
| **GPU notebooks** | 7 Colab/Kaggle notebooks: Triton fused kernels, FlashAttention, tiled & int8 matmul, torch.compile/profiler, CUDA graphs, DDP vs FSDP2, FP8 |
| **Hardware capstone** | KWS from training → distillation → pruning → QAT → integer-only engine → device benchmark → C header for an MCU |
| **Interview in a week** | Enter your interview date and role → a 7-day crash plan with drills, top questions and checklists |
| **Labs** | 14 interactive labs: bias–variance, metrics, k-means/GMM, optimisers, CNN cost, attention/RoPE, KV cache & sampling, MoE routing, efficient attention, diffusion, GRPO/CISPO, roofline & Q15, quantization, pruning |

## The plan

| Phase | Weeks | Focus |
|---|---|---|
| 1 Foundations | 1–4 | Python for interviews, linear models & metrics, trees/boosting/SVM, unsupervised & messy data |
| 2 Deep learning & PyTorch | 5–9 | backprop & autograd, optimisers (incl. Muon) & mixed precision, CNNs, sequence/audio models, PyTorch engineering |
| 3 Transformers & LLMs | 10–16 | attention & positional encodings, the evolved block (GQA/MLA, mHC, AttnRes), MoE & scaling, efficient attention (DeltaNet, DSA, CSA/HCA), inference optimisation, multi-GPU & GPU kernels, generative models |
| 4 Post-training, Chinese playbook & LLM behaviour | 17–20 | SFT/LoRA/RLHF/DPO/GRPO family, the DeepSeek lineage, Qwen/Kimi/GLM/MiniMax, hallucination, tokenisation failures, tool use, LLM evaluation, on-device RAG & agents |
| 5 Embedded AI & optimisation | 21–25 | hardware & roofline & fixed point, quantization I & II (GPTQ, AWQ, SmoothQuant, rotations, MXFP4), pruning & distillation, deployment (ExecuTorch, LiteRT, TFLite Micro, on-device LLMs, streaming audio) |
| 6 System design & sprint | 26–27 | embedded ML system design, timed coding, debugging round, stories, full mock loops |

Each week: **days 1–5** theory + code + questions, **day 6** lab/mini-project, **day 7** review + timed mock + teach-back.

## Exercises

```bash
pytest tests/exercises/test_w10_mha.py      # test your implementation in exercises/
MLPREP_TARGET=solutions pytest tests/exercises   # check the reference solutions
pytest tests/test_app.py                   # every page renders
```

Progress is saved to `progress.json` (git-ignored). Set `MLPREP_PROGRESS=/path/progress.json` to keep it elsewhere.

Content reflects public information up to November 2026; model details come from the labs' technical reports and model cards.

## Repository layout

```
streamlit_app.py        entry point (navigation)
app_pages/              Today, roadmap, review, mock, theory, exercises, sprint, hub, atlas, frontier, settings
app_pages/labs/         the 14 interactive labs
content/weeks/wNN.md    theory (days 1–5), mini-project and question bank for each week (★ = high-yield)
exercises/              stubs you edit (generated from solutions/ by tools/make_stubs.py)
solutions/              reference solutions
tests/exercises/        pytest tests for every exercise; tests/test_app.py renders every page
content/reading.md      primary sources per week · content/system_design.md · content/electives/ (recommender systems)
cpp/                    C++ track: exercises/, solutions/, tests/ (dependency-free harness), CMakeLists.txt
gpu/                    from-scratch Triton kernels, FlashAttention, static-KV decoder, DDP/FSDP2 script (used by notebooks/)
notebooks/              GPU notebooks (generated by tools/make_notebooks.py)
capstone/               keyword-spotting pipeline: distil, prune, QAT, integer-only engine, C export
```

```bash
cmake -S cpp -B build && cmake --build build -j && ctest --test-dir build      # C++ track (your code)
TRITON_INTERPRET=1 pytest tests/test_gpu_kernels.py                            # Triton kernels on CPU
python -m capstone.pipeline --data synthetic --fast                            # capstone smoke test
```
