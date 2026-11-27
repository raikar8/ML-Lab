Primary sources per week. Read one per week, write five lines in your own words (week 19's template).

## W1
- [Hidden Technical Debt in Machine Learning Systems](https://scholar.google.com/scholar?q=Hidden+Technical+Debt+in+Machine+Learning+Systems) — Sculley et al., NeurIPS 2015. Why production ML fails for reasons other than the model.
- [A Few Useful Things to Know About Machine Learning](https://scholar.google.com/scholar?q=A+Few+Useful+Things+to+Know+About+Machine+Learning+Domingos) — Domingos, CACM 2012. Generalisation, overfitting and intuition in 10 pages.

## W2
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) — Guo et al., ICML 2017. Temperature scaling and ECE.
- [The Relationship Between Precision-Recall and ROC Curves](https://scholar.google.com/scholar?q=The+Relationship+Between+Precision-Recall+and+ROC+Curves) — Davis & Goadrich, ICML 2006. When PR beats ROC.

## W3
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754) — Chen & Guestrin, KDD 2016. The second-order gain formula and systems tricks.
- [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815) — Grinsztajn et al., NeurIPS 2022. Evidence for the GBDT-vs-DL interview question.

## W4
- [k-means++: The Advantages of Careful Seeding](https://scholar.google.com/scholar?q=k-means%2B%2B+The+Advantages+of+Careful+Seeding) — Arthur & Vassilvitskii, SODA 2007. The seeding you'll implement.
- [Finding Structure with Randomness (randomized SVD)](https://arxiv.org/abs/0909.4061) — Halko, Martinsson & Tropp, 2009. Low-rank approximation at scale.

## W5
- [Delving Deep into Rectifiers (He initialisation)](https://arxiv.org/abs/1502.01852) — He et al., ICCV 2015. Derivation of Var(w)=2/n.
- [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767) — Baydin et al., JMLR 2018. Forward vs reverse mode, properly.

## W6
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — Kingma & Ba, 2014. Bias correction.
- [Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101) — Loshchilov & Hutter, ICLR 2019. Why L2 ≠ weight decay for Adam.
- [Muon: An optimizer for hidden layers](https://kellerjordan.github.io/posts/muon/) — Keller Jordan, 2024 (blog). Newton–Schulz orthogonalisation, from the author.
- [Muon is Scalable for LLM Training](https://scholar.google.com/scholar?q=Muon+is+Scalable+for+LLM+Training) — Moonshot AI, 2025. Making Muon work at LLM scale.
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) — Micikevicius et al., ICLR 2018. Loss scaling and master weights.

## W7
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — He et al., CVPR 2016. Residuals.
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) — Sandler et al., CVPR 2018. The block you'll implement.
- [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164) — Ma et al., ECCV 2018. FLOPs ≠ latency.

## W8
- [Connectionist Temporal Classification](https://scholar.google.com/scholar?q=Connectionist+Temporal+Classification+labelling+unsegmented+sequence+data) — Graves et al., ICML 2006. The original CTC paper.
- [Sequence Transduction with Recurrent Neural Networks (RNN-T)](https://arxiv.org/abs/1211.3711) — Graves, 2012. The transducer loss.
- [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100) — Gulati et al., Interspeech 2020. The ASR encoder workhorse.
- [wav2vec 2.0](https://arxiv.org/abs/2006.11477) — Baevski et al., NeurIPS 2020. Self-supervised speech.
- [DeepFilterNet](https://scholar.google.com/scholar?q=DeepFilterNet) — Schröter et al., ICASSP 2022. Efficient real-time speech enhancement.

## W9
- [PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://arxiv.org/abs/1912.01703) — Paszke et al., NeurIPS 2019. Design of eager autograd.
- [PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation](https://scholar.google.com/scholar?q=PyTorch+2+Faster+Machine+Learning+Through+Dynamic+Python+Bytecode+Transformation+and+Graph+Compilation) — Ansel et al., ASPLOS 2024. How torch.compile works.

## W10
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., NeurIPS 2017. The original Transformer.
- [Neural Machine Translation of Rare Words with Subword Units (BPE)](https://arxiv.org/abs/1508.07909) — Sennrich et al., ACL 2016. BPE for NLP.
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Su et al., 2021. RoPE.
- [Train Short, Test Long (ALiBi)](https://arxiv.org/abs/2108.12409) — Press et al., ICLR 2022. Linear biases.
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — Peng et al., ICLR 2024. RoPE scaling used by Qwen and DeepSeek.

## W11
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Touvron et al., 2023. The modern open recipe.
- [Fast Transformer Decoding: One Write-Head is All You Need (MQA)](https://arxiv.org/abs/1911.02150) — Shazeer, 2019. MQA.
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) — Ainslie et al., EMNLP 2023. GQA and uptraining.
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — Shazeer, 2020. SwiGLU.
- [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) — DeepSeek-AI, 2025. Doubly stochastic residual mixing.
- [Attention Residuals](https://scholar.google.com/scholar?q=Attention+Residuals+Moonshot+Kimi) — Moonshot AI (Kimi), 2026. Softmax over depth.

## W12
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Kaplan et al., 2020. Power laws.
- [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556) — Hoffmann et al., 2022. 20 tokens/param.
- [Switch Transformers](https://arxiv.org/abs/2101.03961) — Fedus et al., JMLR 2022. Top-1 MoE, capacity factor, aux loss.
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066) — Dai et al., 2024. Fine-grained and shared experts.
- [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://scholar.google.com/scholar?q=Auxiliary-Loss-Free+Load+Balancing+Strategy+for+Mixture-of-Experts) — Wang et al. (DeepSeek), 2024. The bias-update method.

## W13
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — Gu & Dao, 2023. Selective SSMs.
- [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://scholar.google.com/scholar?q=Parallelizing+Linear+Transformers+with+the+Delta+Rule+over+Sequence+Length) — Yang et al., NeurIPS 2024. DeltaNet training.
- [Gated Delta Networks](https://scholar.google.com/scholar?q=Gated+Delta+Networks) — Yang, Kautz & Hatamizadeh, ICLR 2025. The layer in Qwen3-Next/3.5.
- [Native Sparse Attention](https://scholar.google.com/scholar?q=Native+Sparse+Attention) — Yuan et al. (DeepSeek), 2025. Hardware-aligned sparse attention.
- [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) — Xiao et al., ICLR 2024. StreamingLLM.
- [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://scholar.google.com/scholar?q=RULER%3A+What%27s+the+Real+Context+Size+of+Your+Long-Context+Language+Models%3F) — Hsieh et al., 2024. Long-context evaluation.

## W14
- [FlashAttention](https://arxiv.org/abs/2205.14135) — Dao et al., NeurIPS 2022. IO-aware exact attention.
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — Kwon et al., SOSP 2023. vLLM.
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — Leviathan et al., ICML 2023. The lossless proof.
- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://scholar.google.com/scholar?q=EAGLE%3A+Speculative+Sampling+Requires+Rethinking+Feature+Uncertainty) — Li et al., ICML 2024. Feature-level drafting.
- [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://scholar.google.com/scholar?q=Mooncake%3A+A+KVCache-centric+Disaggregated+Architecture+for+LLM+Serving) — Qin et al. (Moonshot), FAST 2025. Kimi's serving system.

## W15
- [Megatron-LM](https://arxiv.org/abs/1909.08053) — Shoeybi et al., 2019. Tensor parallelism.
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) — Rajbhandari et al., SC 2020. Sharded data parallelism.
- [GPipe](https://arxiv.org/abs/1811.06965) — Huang et al., NeurIPS 2019. Pipeline parallelism and bubbles.
- [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) — Korthikanti et al., MLSys 2023. Sequence parallelism, selective checkpointing.
- [Ring Attention with Blockwise Transformers](https://arxiv.org/abs/2310.01889) — Liu et al., 2023. Context parallelism.
- [Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations](https://scholar.google.com/scholar?q=Triton+An+Intermediate+Language+and+Compiler+for+Tiled+Neural+Network+Computations) — Tillet et al., MAPL 2019. Block-level kernels.

## W16
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Kingma & Welling, 2013. VAEs.
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Ho et al., NeurIPS 2020. DDPM.
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — Song et al., ICLR 2021. DDIM.
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — Lipman et al., ICLR 2023. Flow matching.
- [Speech Enhancement and Dereverberation with Diffusion-based Generative Models](https://scholar.google.com/scholar?q=Speech+Enhancement+and+Dereverberation+with+Diffusion-based+Generative+Models) — Richter et al., 2023. SGMSE+.
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — Radford et al., 2021. Contrastive pre-training.

## W17
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — Ouyang et al., 2022. RLHF pipeline.
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — Schulman et al., 2017. PPO.
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — Rafailov et al., NeurIPS 2023. DPO derivation.
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., ICLR 2022. LoRA.
- [QLoRA](https://arxiv.org/abs/2305.14314) — Dettmers et al., NeurIPS 2023. NF4 + LoRA.
- [DeepSeekMath (introduces GRPO)](https://arxiv.org/abs/2402.03300) — Shao et al., 2024. GRPO.
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://scholar.google.com/scholar?q=DAPO+An+Open-Source+LLM+Reinforcement+Learning+System+at+Scale) — ByteDance Seed & Tsinghua, 2025. Clip-higher, dynamic sampling.
- [Group Sequence Policy Optimization](https://scholar.google.com/scholar?q=Group+Sequence+Policy+Optimization+GSPO) — Qwen team, 2025. Sequence-level ratios for MoE RL.

## W18
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) — DeepSeek-AI, 2024. MLA.
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — DeepSeek-AI, 2024. FP8, MTP, DualPipe, aux-loss-free.
- [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — DeepSeek-AI, 2025. RL for reasoning, distillation.
- [DeepSeek-V3.2 and V4 reports and model cards](https://huggingface.co/deepseek-ai) — DeepSeek-AI, 2025–26. DSA, CSA/HCA, mHC, Muon, FP4 QAT.

## W19
- [Qwen3 Technical Report](https://scholar.google.com/scholar?q=Qwen3+Technical+Report) — Qwen team, 2025. Hybrid thinking, strong-to-weak distillation.
- [Kimi K2: Open Agentic Intelligence](https://scholar.google.com/scholar?q=Kimi+K2%3A+Open+Agentic+Intelligence) — Moonshot AI, 2025. MuonClip, agentic data.
- [MiniMax-01: Scaling Foundation Models with Lightning Attention](https://scholar.google.com/scholar?q=MiniMax-01%3A+Scaling+Foundation+Models+with+Lightning+Attention) — MiniMax, 2025. Lightning attention hybrid.
- [MiniMax-M1 (CISPO)](https://scholar.google.com/scholar?q=MiniMax-M1+Scaling+Test-Time+Compute+Efficiently+with+Lightning+Attention) — MiniMax, 2025. CISPO.
- [GLM-4.5: Agentic, Reasoning, and Coding Foundation Models](https://scholar.google.com/scholar?q=GLM-4.5+Agentic+Reasoning+and+Coding+ARC+Foundation+Models) — Zhipu AI / Z.ai, 2025. slime RL, ARC model.

## W20
- [Lost in the Middle: How Language Models Use Long Contexts](https://scholar.google.com/scholar?q=Lost+in+the+Middle%3A+How+Language+Models+Use+Long+Contexts) — Liu et al., TACL 2024. Position bias.
- [In-context Learning and Induction Heads](https://scholar.google.com/scholar?q=In-context+Learning+and+Induction+Heads) — Olsson et al. (Anthropic), 2022. Mechanism behind ICL.
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — Yao et al., ICLR 2023. Agent loop.
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Lewis et al., NeurIPS 2020. RAG.
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — Zheng et al., NeurIPS 2023. Judge biases.
- [Why Language Models Hallucinate](https://scholar.google.com/scholar?q=Why+Language+Models+Hallucinate+Kalai) — Kalai et al. (OpenAI), 2025. Training and evaluation incentives for guessing.

## W21
- [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://scholar.google.com/scholar?q=Roofline+An+Insightful+Visual+Performance+Model+for+Multicore+Architectures) — Williams, Waterman & Patterson, CACM 2009. The roofline model.
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
- [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://scholar.google.com/scholar?q=QuaRot%3A+Outlier-Free+4-Bit+Inference+in+Rotated+LLMs) — Ashkboos et al., 2024. Hadamard rotations.
- [Microscaling Data Formats for Deep Learning](https://scholar.google.com/scholar?q=Microscaling+Data+Formats+for+Deep+Learning) — Rouhani et al., 2023. MX formats.
- [The Era of 1-bit LLMs (BitNet b1.58)](https://scholar.google.com/scholar?q=The+Era+of+1-bit+LLMs+%28BitNet+b1.58%29) — Ma et al., 2024. Ternary weights.

## W24
- [Deep Compression](https://arxiv.org/abs/1510.00149) — Han, Mao & Dally, ICLR 2016. Prune + quantise + Huffman.
- [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635) — Frankle & Carbin, ICLR 2019. Sparse subnetworks.
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — Hinton, Vinyals & Dean, 2015. KD and the T² factor.
- [SparseGPT](https://arxiv.org/abs/2301.00774) — Frantar & Alistarh, ICML 2023. One-shot LLM pruning.
- [A Simple and Effective Pruning Approach for Large Language Models (Wanda)](https://arxiv.org/abs/2306.11695) — Sun et al., ICLR 2024. |W|·‖x‖.
- [Once-for-All](https://arxiv.org/abs/1908.09791) — Cai et al., ICLR 2020. Train once, specialise per device.

## W25
- [TensorFlow Lite Micro: Embedded Machine Learning on TinyML Systems](https://scholar.google.com/scholar?q=TensorFlow+Lite+Micro%3A+Embedded+Machine+Learning+on+TinyML+Systems) — David et al., MLSys 2021. How an MCU runtime is built.
- [MLPerf Tiny Benchmark](https://scholar.google.com/scholar?q=MLPerf+Tiny+Benchmark) — Banbury et al., NeurIPS 2021. How to benchmark edge ML.
- [ExecuTorch documentation](https://pytorch.org/executorch/) — PyTorch, 2025–26. Export, delegates, memory planning.

## W26
- [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) — Zinkevich, Google. Production ML lessons for design rounds.
- [Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg)](https://arxiv.org/abs/1602.05629) — McMahan et al., AISTATS 2017. Federated learning.

## W27
- [Designing Machine Learning Systems](https://scholar.google.com/scholar?q=Designing+Machine+Learning+Systems+Chip+Huyen) — Chip Huyen, O'Reilly 2022 (book). End-to-end ML systems thinking.
- **Your own week-27 project write-up** — The best preparation artefact is your own write-up.
