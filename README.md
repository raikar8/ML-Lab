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
| **Today** | Your tasks for today (theory section, coding exercise, interview coding problem, interview questions; lab day; review day), coding re-solves due, this week's reading, catch-up list, streak |
| **Diagnostic & Week 0** | A 24-question, 10-minute diagnostic (linear algebra, probability, calculus, Python, NumPy, algorithms) that tells you which days of the optional **Week 0 — Foundations** to do before week 1 |
| **6-month roadmap** | All 27 weeks in 6 phases with progress and card mastery per week |
| **Review** | Spaced-repetition deck of ~280 interview questions (★ = high-yield), unlocked as you go; optionally plus Week 0 and the two electives |
| **Mock interview** | Timed question rounds with follow-up probes and self-scoring; weak answers go back into review |
| **Theory by week** | 5 theory sections per week with equations, interview angles, questions and a mini-project |
| **Coding exercises** | 87 Python / NumPy / PyTorch implementation exercises (76 in the weekly plan, 5 in Week 0, 6 in the electives) with pytest tests and reference solutions; edit & run in the app |
| **Coding interview** | 59 classic interview problems across 19 patterns (hashing, two pointers, sliding window, stacks, binary search, linked lists, trees, tries, heaps, intervals, graphs, backtracking, DP, greedy, bits, matrix, design, randomised, ML from scratch). Each has my own problem statement, hints, follow-ups and a target complexity, plus tests that cross-check random inputs against a brute force. Two are scheduled each week; solve with a timer, and clean solves climb a re-solve ladder (3 → 10 → 30 → 90 days). Also: timed two-problem mocks, a pattern guide with templates, and the 31 ML coding questions interviewers ask most, with what each interviewer checks |
| **C/C++ track** | 15 embedded/real-time C++ exercises (fixed point, FIR/biquad, SPSC ring buffer, memory pool, int8 GEMM + requantization, streaming conv, im2col, tiled GEMM…) with tests; compile & run from the app or with CMake |
| **Papers** | The whole reading list (144 primary sources across all 27 weeks + the electives): search, filter by week/topic/source, tick off what you've read, write five-line notes (changed · why · evidence · cost · question), export them as Markdown. Every link goes to arXiv, a DOI or the official page |
| **Electives** | Optional sections in Theory, same format as the weeks. **E1 Recommender systems.** **E2 Python internals, OOP & software engineering**: the GIL and why threads don't speed up CPU-bound Python (and what to use instead: processes, asyncio, vectorisation, free-threaded 3.14t), the object model (descriptors, MRO, memory), OOP done well (composition, ABCs vs Protocols), SOLID and the patterns ML code uses (registry, strategy, callbacks, adapter), and testing/config/logging practice — 5 exercises, 21 questions |
| **Transformer hub** | Evolution 2017→2026, block anatomy then vs now, attention/positional/MoE cheat sheets, primary sources per tab |
| **Chinese model atlas** | DeepSeek, Qwen, Kimi, GLM, MiniMax models and their signature techniques, with the primary source for each model |
| **Frontier watch** | Ongoing topics worth tracking, where to start reading for each, and how to keep current |
| **Answer grader** | Type your answer; a model (local llama.cpp/Ollama server or the Claude API) grades it against the reference — covered, missed, wrong, vague, plus a follow-up. Offline key-concept fallback |
| **Error log** | Bugs you made and questions you fumbled become review cards |
| **System design bank** | 14 embedded/audio/edge designs with clarify → numbers → design → trade-offs → follow-ups, timer and notes |
| **GPU notebooks** | 7 Colab/Kaggle notebooks: Triton fused kernels, FlashAttention, tiled & int8 matmul, torch.compile/profiler, CUDA graphs, DDP vs FSDP2, FP8 |
| **Hardware capstone** | KWS from training → distillation → pruning → QAT → integer-only engine → device benchmark → C header for an MCU |
| **Interview in a week** | Enter your interview date and role → a 7-day crash plan with must-code drills, daily coding-round problems, top questions and checklists |
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

Each week: **days 1–5** theory + code + questions (days 2 and 4, and sometimes 5, also carry one interview coding problem), **day 6** lab/mini-project, **day 7** review + timed mock + teach-back.

**Week 0 (optional)** comes before week 1: linear algebra, probability & statistics, calculus & gradients, Python for interviews, NumPy & complexity, each with a from-scratch exercise. Take the diagnostic to see which days you need.

## Exercises

```bash
pytest tests/exercises/test_w10_mha.py      # test your implementation in exercises/
pytest tests/exercises/test_c25_min_window.py  # an interview coding problem
MLPREP_TARGET=solutions pytest tests/exercises   # check the reference solutions
pytest tests/test_app.py                   # every page renders
```

Progress is saved to `progress.json` (git-ignored). Set `MLPREP_PROGRESS=/path/progress.json` to keep it elsewhere.

Content reflects public information up to November 2026; model details come from the labs' technical reports and model cards.

## Repository layout

```
streamlit_app.py        entry point (navigation)
app_pages/              Today, diagnostic, roadmap, review, mock, theory, exercises, coding, papers, sprint, hub, atlas, frontier, settings
app_pages/labs/         the 14 interactive labs
content/weeks/wNN.md    theory (days 1–5), mini-project and question bank for each week (★ = high-yield)
content/foundations/    Week 0 (f0.md) · content/coding/patterns.md (interview method + pattern guide)
exercises/              stubs you edit (generated from solutions/ by tools/make_stubs.py): f0_* Week 0, wNN_* weekly, e1_*/e2_* electives, cNN_* interview problems
solutions/              reference solutions
tests/exercises/        pytest tests for every exercise; tests/test_app.py renders every page
content/reading.md      primary sources per week (## W<n>) and per elective (## E<n>) — parsed by mlprep/papers.py · content/system_design.md · content/electives/ (E1 recommender systems · E2 Python internals, OOP & software engineering)
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
