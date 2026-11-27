Primary sources per week. Read one per week, write five lines in your own words (week 19's template).

## W1
- [Hidden Technical Debt in Machine Learning Systems](https://proceedings.neurips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) — Sculley et al., NeurIPS 2015. Why production ML fails for reasons other than the model.
- [A Few Useful Things to Know About Machine Learning](https://doi.org/10.1145/2347736.2347755) — Domingos, CACM 2012. Generalisation, overfitting and intuition in 10 pages.

## W2
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) — Guo et al., ICML 2017. Temperature scaling and ECE.
- [The Relationship Between Precision-Recall and ROC Curves](https://doi.org/10.1145/1143844.1143874) — Davis & Goadrich, ICML 2006. When PR beats ROC.

## W3
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754) — Chen & Guestrin, KDD 2016. The second-order gain formula and systems tricks.
- [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815) — Grinsztajn et al., NeurIPS 2022. Evidence for the GBDT-vs-DL interview question.

## W4
- [k-means++: The Advantages of Careful Seeding](https://dl.acm.org/citation.cfm?id=1283494) — Arthur & Vassilvitskii, SODA 2007. The seeding you'll implement.
- [Finding Structure with Randomness (randomized SVD)](https://arxiv.org/abs/0909.4061) — Halko, Martinsson & Tropp, 2009. Low-rank approximation at scale.

## W5
- [Delving Deep into Rectifiers (He initialisation)](https://arxiv.org/abs/1502.01852) — He et al., ICCV 2015. Derivation of Var(w)=2/n.
- [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767) — Baydin et al., JMLR 2018. Forward vs reverse mode, properly.

## W6
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Kingma & Ba, 2014. Bias correction.
- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101) — Loshchilov & Hutter, ICLR 2019. Why L2 ≠ weight decay for Adam.
- [Muon: An optimizer for hidden layers](https://kellerjordan.github.io/posts/muon/) — Keller Jordan, 2024 (blog). Newton–Schulz orthogonalisation, from the author.
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) — Moonshot AI, 2025. Making Muon work at LLM scale.
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) — Micikevicius et al., ICLR 2018. Loss scaling and master weights.

## W7
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — He et al., CVPR 2016. Residuals.
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) — Sandler et al., CVPR 2018. The block you'll implement.
- [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164) — Ma et al., ECCV 2018. FLOPs ≠ latency.

## W8
- [Connectionist Temporal Classification](https://doi.org/10.1145/1143844.1143891) — Graves et al., ICML 2006. The original CTC paper.
- [Sequence Transduction with Recurrent Neural Networks (RNN-T)](https://arxiv.org/abs/1211.3711) — Graves, 2012. The transducer loss.
- [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100) — Gulati et al., Interspeech 2020. The ASR encoder workhorse.
- [wav2vec 2.0](https://arxiv.org/abs/2006.11477) — Baevski et al., NeurIPS 2020. Self-supervised speech.
- [DeepFilterNet](https://arxiv.org/abs/2110.05588) — Schröter et al., ICASSP 2022. Efficient real-time speech enhancement.

## W9
- [PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://arxiv.org/abs/1912.01703) — Paszke et al., NeurIPS 2019. Design of eager autograd.
- [PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation](https://doi.org/10.1145/3620665.3640366) — Ansel et al., ASPLOS 2024. How torch.compile works.

## W10
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., NeurIPS 2017. The original Transformer.
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — Devlin et al., NAACL 2019. Encoder-only masked LM.
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — Brown et al., NeurIPS 2020. In-context learning at scale.
- [Neural Machine Translation of Rare Words with Subword Units (BPE)](https://arxiv.org/abs/1508.07909) — Sennrich et al., ACL 2016. BPE for NLP.
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Su et al., 2021. RoPE.
- [Train Short, Test Long (ALiBi)](https://arxiv.org/abs/2108.12409) — Press et al., ICLR 2022. Linear biases.
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — Peng et al., ICLR 2024. RoPE scaling used by Qwen and DeepSeek.

## W11
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Touvron et al., 2023. The modern open recipe.
- [Fast Transformer Decoding: One Write-Head is All You Need (MQA)](https://arxiv.org/abs/1911.02150) — Shazeer, 2019. MQA.
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) — Ainslie et al., EMNLP 2023. GQA and uptraining.
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — Shazeer, 2020. SwiGLU.
- [Mistral 7B](https://arxiv.org/abs/2310.06825) — Jiang et al., 2023. Sliding-window attention + GQA in a small model.
- [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) — Llama Team (Meta), 2024. Dense recipe at 405B; 128K vocab.
- [gpt-oss-120b & gpt-oss-20b Model Card](https://arxiv.org/abs/2508.10925) — OpenAI, 2025. Open MoE with attention sinks, banded attention, MXFP4 experts.
- [Hyper-Connections](https://arxiv.org/abs/2409.19606) — Zhu et al. (ByteDance Seed), ICLR 2025. Learnable depth/width residual mixing; the basis of mHC.
- [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) — DeepSeek-AI, 2025. Doubly stochastic residual mixing.
- [Attention Residuals](https://arxiv.org/abs/2603.15031) — Moonshot AI (Kimi), 2026. Softmax over depth.

## W12
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Kaplan et al., 2020. Power laws.
- [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556) — Hoffmann et al., 2022. 20 tokens/param.
- [Switch Transformers](https://arxiv.org/abs/2101.03961) — Fedus et al., JMLR 2022. Top-1 MoE, capacity factor, aux loss.
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Jiang et al., 2024. Open top-2 sparse MoE.
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066) — Dai et al., 2024. Fine-grained and shared experts.
- [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — Wang et al. (DeepSeek), 2024. The bias-update method.

## W13
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — Gu & Dao, 2023. Selective SSMs.
- [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484) — Yang et al., NeurIPS 2024. DeltaNet training.
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464) — Yang, Kautz & Hatamizadeh, ICLR 2025. The layer in Qwen3-Next/3.5.
- [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) — Moonshot AI, 2025. Kimi Delta Attention in a 3:1 hybrid with MLA.
- [Native Sparse Attention](https://arxiv.org/abs/2502.11089) — Yuan et al. (DeepSeek), 2025. Hardware-aligned sparse attention.
- [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) — Xiao et al., ICLR 2024. StreamingLLM.
- [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654) — Hsieh et al., 2024. Long-context evaluation.

## W14
- [FlashAttention](https://arxiv.org/abs/2205.14135) — Dao et al., NeurIPS 2022. IO-aware exact attention.
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — Kwon et al., SOSP 2023. vLLM.
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — Leviathan et al., ICML 2023. The lossless proof.
- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — Li et al., ICML 2024. Feature-level drafting.
- [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079) — Qin et al. (Moonshot), FAST 2025. Kimi's serving system.

## W15
- [Megatron-LM](https://arxiv.org/abs/1909.08053) — Shoeybi et al., 2019. Tensor parallelism.
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) — Rajbhandari et al., SC 2020. Sharded data parallelism.
- [GPipe](https://arxiv.org/abs/1811.06965) — Huang et al., NeurIPS 2019. Pipeline parallelism and bubbles.
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) — Korthikanti et al., MLSys 2023. Sequence parallelism, selective checkpointing.
- [Ring Attention with Blockwise Transformers](https://arxiv.org/abs/2310.01889) — Liu et al., 2023. Context parallelism.
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://doi.org/10.1145/3315508.3329973) — Tillet et al., MAPL 2019. Block-level kernels.

## W16
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Kingma & Welling, 2013. VAEs.
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Ho et al., NeurIPS 2020. DDPM.
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — Song et al., ICLR 2021. DDIM.
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — Lipman et al., ICLR 2023. Flow matching.
- [Speech Enhancement and Dereverberation with Diffusion-based Generative Models](https://arxiv.org/abs/2208.05830) — Richter et al., 2023. SGMSE+.
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — Radford et al., 2021. Contrastive pre-training.

## W17
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Ouyang et al., 2022. RLHF pipeline.
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — Schulman et al., 2017. PPO.
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — Rafailov et al., NeurIPS 2023. DPO derivation.
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., ICLR 2022. LoRA.
- [QLoRA](https://arxiv.org/abs/2305.14314) — Dettmers et al., NeurIPS 2023. NF4 + LoRA.
- [DeepSeekMath (introduces GRPO)](https://arxiv.org/abs/2402.03300) — Shao et al., 2024. GRPO.
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476) — ByteDance Seed & Tsinghua, 2025. Clip-higher, dynamic sampling.
- [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071) — Qwen team, 2025. Sequence-level ratios for MoE RL.

## W18
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) — DeepSeek-AI, 2024. MLA.
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — DeepSeek-AI, 2024. FP8, MTP, DualPipe, aux-loss-free.
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — DeepSeek-AI, 2025. RL for reasoning, distillation.
- [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556) — DeepSeek-AI, 2025. DSA sparse attention, scaled RL, agentic task synthesis.
- [DeepSeek-V4 report and model cards](https://huggingface.co/deepseek-ai) — DeepSeek-AI, 2026. CSA/HCA, mHC, Muon, FP4 QAT.

## W19
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) — Qwen team, 2025. Hybrid thinking, strong-to-weak distillation.
- [Qwen3-Next: Towards Ultimate Training & Inference Efficiency](https://www.alibabacloud.com/blog/602580) — Qwen team, 2025 (blog). 3:1 Gated DeltaNet hybrid, 512 experts.
- [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534) — Moonshot AI, 2025. MuonClip, agentic data.
- [Kimi K2.5: Visual Agentic Intelligence](https://arxiv.org/abs/2602.02276) — Moonshot AI, 2026. Joint text–vision training, Agent Swarm (PARL).
- [MiniMax-01: Scaling Foundation Models with Lightning Attention](https://arxiv.org/abs/2501.08313) — MiniMax, 2025. Lightning attention hybrid.
- [MiniMax-M1 (CISPO)](https://arxiv.org/abs/2506.13585) — MiniMax, 2025. CISPO.
- [GLM-4.5: Agentic, Reasoning, and Coding Foundation Models](https://arxiv.org/abs/2508.06471) — Zhipu AI / Z.ai, 2025. slime RL, ARC model.
- [GLM-5: from Vibe Coding to Agentic Engineering](https://arxiv.org/abs/2602.15763) — Zhipu AI / Z.ai, 2026. MLA + DSA, async RL.
- [LongCat-Flash Technical Report](https://arxiv.org/abs/2509.01322) — Meituan, 2025. Zero-computation experts, shortcut-connected MoE.
- [Step-3 is Large yet Affordable: Model-system Co-design for Cost-effective Decoding](https://arxiv.org/abs/2507.19427) — StepFun, 2025. MFA attention, attention–FFN disaggregation.

## W20
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — Liu et al., TACL 2024. Position bias.
- [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895) — Olsson et al. (Anthropic), 2022. Mechanism behind ICL.
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Yao et al., ICLR 2023. Agent loop.
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Lewis et al., NeurIPS 2020. RAG.
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — Zheng et al., NeurIPS 2023. Judge biases.
- [Why Language Models Hallucinate](https://arxiv.org/abs/2509.04664) — Kalai et al. (OpenAI), 2025. Training and evaluation incentives for guessing.

## W21
- [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://doi.org/10.1145/1498765.1498785) — Williams, Waterman & Patterson, CACM 2009. The roofline model.
- [Hello Edge: Keyword Spotting on Microcontrollers](https://arxiv.org/abs/1711.07128) — Zhang et al., 2017. DS-CNN KWS on MCUs.
- [MCUNet: Tiny Deep Learning on IoT Devices](https://arxiv.org/abs/2007.10319) — Lin et al., NeurIPS 2020. TinyNAS + TinyEngine.

## W22
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877) — Jacob et al., CVPR 2018. The integer scheme you implement.
- [A White Paper on Neural Network Quantization](https://arxiv.org/abs/2106.08295) — Nagel et al., 2021. PTQ/QAT practice in one place.
- [Data-Free Quantization Through Weight Equalization and Bias Correction](https://arxiv.org/abs/1906.04721) — Nagel et al., ICCV 2019. CLE.
- [Up or Down? Adaptive Rounding for Post-Training Quantization (AdaRound)](https://arxiv.org/abs/2004.10568) — Nagel et al., ICML 2020. Learned rounding.
- [Learned Step Size Quantization](https://arxiv.org/abs/1902.08153) — Esser et al., ICLR 2020. LSQ.

## W23
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) — Dettmers et al., NeurIPS 2022. Outlier features.
- [GPTQ](https://arxiv.org/abs/2210.17323) — Frantar et al., ICLR 2023. Hessian-based weight quantization.
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978) — Lin et al., MLSys 2024. Salient channels.
- [SmoothQuant](https://arxiv.org/abs/2211.10438) — Xiao et al., ICML 2023. Migrating difficulty to weights.
- [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456) — Ashkboos et al., 2024. Hadamard rotations.
- [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537) — Rouhani et al., 2023. MX formats.
- [The Era of 1-bit LLMs (BitNet b1.58)](https://arxiv.org/abs/2402.17764) — Ma et al., 2024. Ternary weights.

## W24
- [Deep Compression](https://arxiv.org/abs/1510.00149) — Han, Mao & Dally, ICLR 2016. Prune + quantise + Huffman.
- [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635) — Frankle & Carbin, ICLR 2019. Sparse subnetworks.
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — Hinton, Vinyals & Dean, 2015. KD and the T² factor.
- [SparseGPT](https://arxiv.org/abs/2301.00774) — Frantar & Alistarh, ICML 2023. One-shot LLM pruning.
- [A Simple and Effective Pruning Approach for Large Language Models (Wanda)](https://arxiv.org/abs/2306.11695) — Sun et al., ICLR 2024. |W|·‖x‖.
- [Once-for-All](https://arxiv.org/abs/1908.09791) — Cai et al., ICLR 2020. Train once, specialise per device.

## W25
- [TensorFlow Lite Micro: Embedded Machine Learning on TinyML Systems](https://arxiv.org/abs/2010.08678) — David et al., MLSys 2021. How an MCU runtime is built.
- [MLPerf Tiny Benchmark](https://arxiv.org/abs/2106.07597) — Banbury et al., NeurIPS 2021. How to benchmark edge ML.
- [ExecuTorch documentation](https://pytorch.org/executorch/) — PyTorch, 2025–26. Export, delegates, memory planning.

## W26
- [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) — Zinkevich, Google. Production ML lessons for design rounds.
- [Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg)](https://arxiv.org/abs/1602.05629) — McMahan et al., AISTATS 2017. Federated learning.

## W27
- [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — Chip Huyen, O'Reilly 2022 (book). End-to-end ML systems thinking.
- **Your own week-27 project write-up** — The best preparation artefact is your own write-up.

## E1
- [Matrix Factorization Techniques for Recommender Systems](https://doi.org/10.1109/MC.2009.263) — Koren, Bell & Volinsky, IEEE Computer 2009. Biases, implicit feedback and time in MF.
- [Collaborative Filtering for Implicit Feedback Datasets](https://doi.org/10.1109/ICDM.2008.22) — Hu, Koren & Volinsky, ICDM 2008. Confidence-weighted ALS.
- [Deep Neural Networks for YouTube Recommendations](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/) — Covington, Adams & Sargin, RecSys 2016. Candidate generation → ranking funnel.
- [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://doi.org/10.1145/3298689.3346996) — Yi et al., RecSys 2019. Two-tower with logQ correction.
- [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792) — Cheng et al., DLRS 2016. Memorisation + generalisation.
- [Deep Learning Recommendation Model for Personalization and Recommendation Systems (DLRM)](https://arxiv.org/abs/1906.00091) — Naumov et al., 2019. Embedding tables + dot interactions.
- [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://doi.org/10.1145/3219819.3220007) — Ma et al., KDD 2018. MMoE.
