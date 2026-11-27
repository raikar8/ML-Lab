Each design follows the week-26 framework: clarify → numbers → design → trade-offs → monitoring. Do it out loud in 45 minutes
before revealing anything; the "Key numbers" are the budgets a strong candidate states unprompted.

## speaker-verification | On-device speaker verification for driver profiles | audio, edge, privacy

### Prompt
When someone says "Hey <brand>", the car should recognise which registered household member is speaking and load their
profile (seat, mirrors, playlists, contacts). Everything must run on the car; voice data must not leave the vehicle.

### Clarify first
How many enrolled speakers per car (≤ 8)? Text-dependent (the wake word) or text-independent? Security level — convenience
(seat position) or authorisation (payments, unlocking)? Which seat is speaking (multi-zone)? Latency target after the wake
word? Enrolment flow and re-enrolment? Children's voices change.

### Key numbers
Decision within ~300 ms of wake-word end; embedding model ≤ 5 MB int8, ≤ 50 M MACs per utterance on the cockpit DSP/NPU;
target EER ≤ 2–3% text-dependent in cabin noise; false-accept rate set by use (convenience ≈ 1%, authorisation ≪ 0.1% — then
voice alone is not enough).

### Reference design
Front end: AEC + beamforming toward the active zone (shared with the wake-word pipeline), VAD to trim the segment. Model: a small
ECAPA-TDNN or ResNet speaker-embedding network (trained on VoxCeleb-style data, fine-tuned on in-cabin augmented data with car
RIRs, road noise, HVAC, music bleed), output 192-d L2-normalised embedding. Enrolment: 3–5 utterances of the wake word →
average embedding (store only embeddings, encrypted, never audio). Scoring: cosine similarity with per-speaker adaptive
s-norm/AS-norm using a cohort of impostor embeddings shipped with the model; threshold per car calibrated on enrolment data.
Unknown speaker → guest profile. Continual adaptation: update a speaker's centroid with high-confidence accepted utterances
(bounded, with drift checks).

### Trade-offs & pitfalls
Text-dependent (wake word only) is far more accurate but can't verify mid-conversation. Short utterances (< 1 s) are the main
error source. Replay/spoofing attacks matter only for authorisation — add liveness or a second factor. Family members with similar
voices; colds; kids. Domain mismatch between phone-recorded enrolment and in-car audio — enrol in the car.

### Follow-ups
How do you evaluate without shipping user audio? (Consented in-house collection across car models + simulated cabins; on-device
aggregate metrics such as re-identification corrections.) What if two people speak at once? How would you quantise the model and
check embedding quality didn't drift (cosine similarity of float vs int8 embeddings, EER on a test set)?

## zone-separation | Multi-zone speech separation in the cabin | audio, edge, real-time

### Prompt
Four occupants. The assistant must respond to the person who addressed it and suppress the others, and hands-free calls should
transmit only the driver (or the chosen zone). Design the audio pipeline.

### Clarify first
Microphone count and placement (one per seat? an array in the headliner?), whether loudspeaker playback must be cancelled,
latency budget for calls (telephony) vs ASR, compute on the audio DSP vs application processor, how many simultaneous talkers to
handle.

### Key numbers
Call path ≤ ~20–30 ms added latency (ITU-T hands-free requirements cover delay, echo loss, double-talk); ASR path can tolerate
~100–200 ms lookahead; DSP budget a few hundred MIPS; zone leakage suppression target ≥ 20–30 dB; 16 kHz wideband (or 32/48 kHz
super-wideband for calls).

### Reference design
Per-zone AEC with the playback reference. Spatial stage: fixed or adaptive beamformers per zone (MVDR with steering vectors from
measured cabin transfer functions), producing one signal per zone plus a zone-activity estimate (energy ratios, coherence, a small
classifier). Neural stage: a multi-channel causal mask estimator (e.g. a compact CRN taking all mics + beam outputs) predicting a
mask for the target zone, conditioned on the zone index — trained on simulated mixtures convolved with measured multi-mic cabin
IRs from each seat. Post-filter for residual echo/noise. The wake word runs on all zones; the triggering zone becomes the target.

### Trade-offs & pitfalls
Classic beamforming is predictable, cheap and certifiable; neural separation is better in overlap but harder to guarantee for
call quality (artefacts, speech distortion). Measured IRs per car model are essential — sim-to-real gap otherwise. Seat
occupancy and head position change the acoustics. Loud music: AEC must converge before separation can work.

### Follow-ups
How do you test? (Per-car measurement campaign, ITU-T P.1100/P.1110 tests for the call path, WER per zone with competing talkers.)
How do you share compute between AEC, separation and wake word on one DSP?

## siren-detection | Emergency-vehicle (siren) detection for ADAS | audio, safety, edge

### Prompt
Detect approaching emergency-vehicle sirens from external microphones, estimate direction, and alert the driver (and the ADAS
stack) early enough to react.

### Clarify first
Which siren types/regions (wail, yelp, hi-lo, EU two-tone)? Required detection range/time before the vehicle is visible?
Direction of arrival needed? False-alarm tolerance? Microphone placement (exterior, exposed to wind/rain)? Safety level?

### Key numbers
Detect within ~1–2 s of the siren becoming audible above road noise; false alarms ≤ 1 per several hours of driving; DOA accuracy
±30° is useful; model ≤ 1 M params on a safety-island MCU/DSP; wind noise can exceed 90 dB SPL at speed.

### Reference design
Exterior mic array (≥ 3 mics for 2-D DOA) with wind-noise-robust housings. Front end: log-mel + harmonic features (sirens are
strongly tonal and frequency-modulated). Model: a small CRNN on 1–2 s windows with causal GRU smoothing → siren probability per
type; temporal hysteresis (trigger after N of M frames). DOA: GCC-PHAT between mic pairs on time–frequency bins selected by the
detector's mask. Data: siren recordings across countries, synthetic mixes with road/wind/rain noise at varying SNR, Doppler
simulation, hard negatives (car alarms, music with sirens, birds, reversing beepers).

### Trade-offs & pitfalls
Regional siren variety — design the label set per market. Wind and rain dominate low SNR. In-cabin music containing sirens is a
false-alarm source if interior mics are used. Safety: the alert is advisory, with graceful degradation and self-diagnostics
(blocked or failed mics).

### Follow-ups
How do you validate the false-alarm rate (thousands of hours of negative driving audio)? How would you fuse with cameras?

## ota-rollout | Over-the-air model rollout across a vehicle fleet | mlops, edge, automotive

### Prompt
You have a new wake-word model. Design how it gets from the training cluster to 2 million cars safely, and how you know it's better.

### Clarify first
Heterogeneous hardware/software versions in the fleet? Connectivity and update windows? Regulatory constraints on software
updates (UNECE R156 software update management)? What telemetry is allowed (privacy, consent)? Rollback requirements?

### Key numbers
Model artefact size (KB–MB; delta updates for big models); staged rollout 0.1% → 1% → 10% → 100% over days/weeks; guardrail
metrics: false accepts/hour, wake success rate (proxy: user retries), latency p99, CPU/DSP load, crash rate.

### Reference design
Artefact: model + metadata (version, training data hash, quantisation config, target runtime/SoC, minimum app version) signed and
versioned in a model registry. Offline gates: accuracy on curated slices, per-SoC latency/memory on a device farm, bit-exact or
tolerance parity tests of the exported model. **Shadow mode** first: the new model runs alongside the old one without acting;
devices report privacy-preserving aggregates (trigger agreement, score distributions). Then staged rollout with an A/B holdout,
automated health checks, and a kill switch. The device keeps the previous model for instant rollback; updates are atomic (A/B
partitions or two model slots).

### Trade-offs & pitfalls
Telemetry for wake word is tricky: false accepts are rarely labelled — use proxies (immediate cancellation, no follow-up speech).
Compatibility matrix explosion — pin model versions to runtime versions. Regional differences (languages, accents) mean a
global average can hide regressions; slice metrics by market and car model.

### Follow-ups
How would you detect a regression that only affects one car model with a different microphone? What do you do if shadow-mode
agreement is 99.9% — is that good?

## wearable-kws | Always-on keyword spotting on a battery-powered earbud | tinyml, audio, power

### Prompt
An earbud should respond to "Hey <brand>" all day on a tiny battery. Design the always-on detection.

### Clarify first
Battery capacity and target life (e.g. 6–8 h with always-on)? Available silicon: analog/low-power VAD, MCU class (Cortex-M4/M55),
NPU? Bone-conduction sensor available (own-voice detection)? False-accept budget? Languages?

### Key numbers
Always-on budget ≈ 100–500 µW total; a Cortex-M class core running continuously at even a few mW is too expensive, so the
cascade must keep the big core asleep most of the time. KWS model ≤ 50 k params, ≤ 1–5 M MACs/s, int8; false accepts ≤ 1 per
~24 h of normal audio.

### Reference design
Stage 0: analog/ultra-low-power VAD (energy/zero-crossing) wakes the digital path only when there is speech. Stage 1: tiny DS-CNN
or a small GRU on 40-band features at 10–20 ms hop, running on a low-power DSP/NPU, high recall. Stage 2: on stage-1 trigger, a
larger verifier (on the earbud's main core or on the phone via Bluetooth) confirms. Own-voice gating with the bone-conduction
sensor or inter-mic features prevents triggers from nearby talkers. Train with lots of hard negatives; quantise with QAT;
profile energy per inference on the target.

### Trade-offs & pitfalls
Every stage trades power for false accepts. Moving stage 2 to the phone saves earbud power but adds Bluetooth latency and
depends on connection. Front-end features must be computed in fixed point identically in training and deployment.

### Follow-ups
How do you measure energy per inference? How would you personalise the wake word to the user without storing audio?

## offline-translation | Offline speech-to-speech translation on a phone | asr, llm, edge

### Prompt
Two people speaking different languages hold a conversation through a phone, with no network. Design it.

### Clarify first
Language pairs (and how many)? Latency expectations (turn-based vs simultaneous)? Device class and RAM? Model download size
limits per language pair? Quality target vs cloud?

### Key numbers
End-of-utterance → translated speech start ≤ ~1–1.5 s for turn-based; memory for models ≤ ~1–2 GB total; streaming ASR RTF ≤ 0.3
on the NPU; download per language pair ideally ≤ 300–500 MB.

### Reference design
Cascade: streaming multilingual ASR (RNN-T/Conformer or a quantised Whisper-style model with chunked decoding) with language ID →
text translation (a compact multilingual encoder–decoder or a 1–3B LLM fine-tuned for translation, int4 weights) → on-device TTS
(fast non-autoregressive acoustic model + GAN vocoder, int8). Endpointing via VAD + ASR end-of-utterance probability. Share one
multilingual model across languages where possible to limit downloads. Display text alongside audio for verification.

### Trade-offs & pitfalls
Cascade is debuggable and modular; end-to-end speech-to-speech models preserve prosody but are bigger and harder to fit and
evaluate. Errors compound (ASR error → wrong translation). Named entities and numbers need special care (copy/transliterate).

### Follow-ups
How do you evaluate (BLEU/COMET on references, WER, human ratings, latency per stage)? How would you handle code-switching?

## ml-aec | ML-assisted acoustic echo cancellation for hands-free calls | audio, real-time, dsp

### Prompt
Hands-free calls in the car suffer residual echo during double-talk and loud music. Design an ML-augmented AEC that keeps
ITU-T compliance.

### Clarify first
Existing linear AEC (NLMS/FDAF) and residual echo suppressor? Compute budget on the audio DSP? Required echo return loss
enhancement, double-talk performance and delay per the relevant ITU-T hands-free specs? Narrowband vs wideband?

### Key numbers
Added algorithmic latency ≤ ~10 ms (frame + lookahead); total send-path delay budget per the hands-free spec; model ≤ ~1–2 M
params, int8/int16 on the DSP; targets: higher ERLE during double-talk, minimal near-end speech attenuation.

### Reference design
Keep the linear adaptive filter (it handles the linear echo path efficiently and adapts fast to path changes) and replace/augment
the residual echo suppressor with a small causal neural network: inputs = microphone STFT, linear-AEC output, and the far-end
reference (or the linear echo estimate); output = a gain mask for the AEC output (per ERB band) — a hybrid DSP+DNN design like
recent AEC challenge winners. Train on synthetic data: far-end speech/music through simulated + measured car echo paths with
non-linear loudspeaker distortion, near-end speech, noise, path changes and double-talk. Fall back to the classic suppressor if the
DNN exceeds its time budget.

### Trade-offs & pitfalls
End-to-end neural AEC is attractive but struggles with fast echo path changes and is hard to certify; the hybrid keeps
guarantees. Loss: combine echo suppression and near-end preservation terms explicitly; otherwise the network learns to suppress
everything during double-talk. Measure with ITU-T test methods plus AECMOS-style metrics and listening tests.

### Follow-ups
How do you detect double-talk? How would the mask network behave if the reference signal is delayed or clipped?

## asr-commands | Streaming on-device ASR for in-car commands, multilingual | asr, edge, streaming

### Prompt
Design on-device speech recognition for in-car commands ("set temperature to 21", "navigate home", "call Mum") in 5 languages,
working offline.

### Clarify first
Command grammar vs open vocabulary? Personal entities (contacts, saved places) — how many? Languages simultaneously or per car?
Latency target? Compute: NPU TOPS, RAM budget? Fallback to cloud when connected?

### Key numbers
First partial result ≤ 300 ms, final ≤ ~200 ms after end of speech; WER on commands ≤ 5%; model ≤ 100–150 MB per language (or one
multilingual ≤ 300 MB); RTF ≤ 0.2 on the NPU.

### Reference design
Streaming Conformer/Zipformer-transducer (RNN-T) with chunked attention (e.g. 320 ms chunks, small right context), int8.
Contextual biasing for personal entities (contacts, places) via a biasing list/trie with shallow fusion or a neural biasing
module, updated on device when contacts change. Endpointing from VAD + model. NLU: a small intent/slot model, or a 1–3B LLM with
constrained function calling for flexible phrasing. Per-language models selected by the car's language setting, or a multilingual
model with language ID.

### Trade-offs & pitfalls
Personal names are the hardest part — biasing is essential and must be privacy-preserving (on device only). Accents and code-
switching. Training data must include cabin conditions. Grammar-based systems are robust but rigid; LLM NLU is flexible but must be
validated and constrained.

### Follow-ups
How do you evaluate contact-name recognition without real users' contacts? How do you keep the model up to date with new commands?

## tinyml-vision | Person detection on a microcontroller camera | tinyml, vision, power

### Prompt
A battery-powered doorbell camera must detect people (not cars, animals, swaying trees) and only then wake the Wi-Fi radio.

### Clarify first
MCU and memory (e.g. Cortex-M55 + Ethos-U55, 512 KB SRAM, 2 MB flash)? Frame rate needed? Battery life target? Day/night (IR)?
Privacy (no frames uploaded unless a person is detected)?

### Key numbers
Model ≤ ~300 KB flash int8; peak activation memory ≤ ~250 KB; ≤ 50–100 ms per frame; 1–5 fps; energy per inference in the mJ range;
false wakes cost far more energy (radio) than inferences.

### Reference design
Low-power motion trigger (PIR or pixel-difference on a low-res stream) → person classifier/detector at 96×96–160×160 grayscale or
RGB: a MobileNetV2-0.35 / MCUNet-style network found with hardware-aware NAS for the memory budget, int8 QAT, patch-based inference
for early layers to fit SRAM. Temporal smoothing over 3–5 frames. Only on confirmed detection, wake the main SoC for a bigger
model and upload.

### Trade-offs & pitfalls
Resolution vs memory vs accuracy; detection (boxes) vs classification (cheaper). Night IR images need their own data. Measure real
latency and peak SRAM with the deployment runtime (TFLite Micro / ExecuTorch + CMSIS-NN); FLOPs mislead.

### Follow-ups
How would you reduce false wakes from trees? How do you collect data responsibly for this product?

## acoustic-anomaly | Acoustic anomaly detection for machines at the edge | audio, anomaly, edge

### Prompt
Detect early faults in motors/pumps from microphones and accelerometers on an edge gateway, with very few labelled failures.

### Clarify first
How many machine types and operating modes? Sampling rates (audio 16 kHz; vibration up to tens of kHz)? Labels available? Alert
budget per site per week? Connectivity?

### Key numbers
Alert precision matters more than recall at first (≤ a few false alerts per site per month); lead time before failure (days);
gateway CPU-class compute, models ≤ a few MB; per-machine drift over months.

### Reference design
Features: log-mel / order-tracked spectra (normalise by rotation speed), spectral kurtosis, harmonics. Unsupervised model per
machine type and operating mode: autoencoder or normalising-flow density model trained on "normal" data, or k-NN distance in a
pretrained audio embedding space; anomaly score with per-machine calibration (z-score against that machine's recent baseline).
Operating-mode conditioning (speed/load) is critical. Alerts with persistence rules; human feedback labels alerts, which train a
supervised classifier over time (data flywheel).

### Trade-offs & pitfalls
Normal variation (load changes, maintenance, new sensor placement) causes false alarms — condition on operating state and reset
baselines after maintenance. Evaluation with few failures: inject synthetic faults, use public benchmarks (e.g. DCASE machine
anomaly tasks) for development, and precision-at-alert-budget on historical incidents.

### Follow-ups
How would you share learning across factories without centralising raw audio? How do you set thresholds for a new machine with
no history?

## edge-compiler | A platform to ship any PyTorch model to five different SoCs | mlops, deployment, edge

### Prompt
Research teams train models in PyTorch; product teams need them on five SoCs (Qualcomm, NXP, TI, an Arm MCU, an NVIDIA Orin).
Design the internal deployment platform.

### Clarify first
Model types (CNNs, RNNs, small transformers)? Who owns quantisation? Turnaround target from "trained" to "running on device"?
Device farm available? Accuracy/latency acceptance criteria?

### Key numbers
Turnaround ≤ 1 day for supported architectures; per-target operator coverage report; parity tolerance (e.g. cosine ≥ 0.999 on
float, task metric drop ≤ 0.5% for int8); latency p50/p99 and memory reported per target.

### Reference design
A pipeline: `torch.export` → graph checks (unsupported ops per backend, dynamic shapes) → PT2E quantisation with per-backend
quantizers (calibration data versioned with the model) → lowering per target (ExecuTorch delegates, TensorRT, vendor SDKs) →
automated device-farm benchmark (latency distribution, memory, energy) and accuracy parity on a golden dataset → signed artefacts
in a registry with a model card. Early feedback for researchers: an "operator coverage linter" and latency predictor run in CI
on every model change, so incompatible layers are caught at design time.

### Trade-offs & pitfalls
One IR (export) vs per-vendor paths; vendor toolchains change versions — pin and test. Quantisation differences per backend
(per-tensor vs per-channel, symmetric vs asymmetric) cause accuracy surprises — make the quantizer backend-aware from the start.

### Follow-ups
How do you handle a custom op that only one SoC supports? How do you make accuracy regressions traceable?

## keyboard-fl | Next-word prediction with federated learning | llm, privacy, federated

### Prompt
Improve a phone keyboard's next-word prediction using users' typing without uploading their text.

### Clarify first
Model size budget and latency per keystroke? Languages? Privacy guarantees required (differential privacy level)? Device
participation conditions (charging, Wi-Fi, idle)?

### Key numbers
Prediction latency ≤ ~10–20 ms per keystroke; model ≤ ~10–50 MB; federated rounds with hundreds to thousands of devices each;
client-level DP with a privacy budget agreed with privacy/legal.

### Reference design
A small causal LM (or LSTM/transformer of a few tens of millions of params), int8 on device. Federated averaging: the server sends
the current model; eligible devices fine-tune for a few local steps on recent typing, clip their update, and send it through
**secure aggregation**; the server adds calibrated noise (DP-FedAvg) and averages. Evaluate on held-out devices (federated
evaluation) and with proxy public data; ship improved models via staged rollout.

### Trade-offs & pitfalls
Non-IID data, device heterogeneity and stragglers; DP noise vs utility; memorisation risk of rare sequences (DP + clipping address
it); evaluation without seeing data.

### Follow-ups
How would you personalise per user on top of the global model? How do you detect a poisoned update?

## sound-events | Privacy-preserving sound event detection for a smart home | audio, privacy, edge

### Prompt
A smart speaker should detect smoke alarms, glass breaking and a baby crying, and notify the user — without streaming audio to the
cloud.

### Clarify first
Event list and which are safety-critical (smoke alarm)? Acceptable false alarms per week? Hardware (low-power DSP + application
processor)? Notification path? User controls?

### Key numbers
Detection within 1–3 s; false alarms ≤ ~1 per week per home for intrusive alerts; always-on budget on a DSP; model ≤ ~1–2 MB.

### Reference design
Always-on low-cost event detector (small CNN on log-mel, multi-label sigmoid outputs) with per-class thresholds and temporal
smoothing; safety-critical classes (smoke/CO alarm) use additional DSP features (alarm tones have well-defined frequency
patterns) for robustness. Only event labels and timestamps leave the device. Training data: curated event recordings + home
background noise (TV, music, cooking), mixing at varied SNRs and room IRs; hard negatives (TV sounds of glass breaking!).

### Trade-offs & pitfalls
TV/media audio is the main false-alarm source — use AEC with the speaker's own playback and media-detection classifiers. Per-class
thresholds should be tuned on precision targets, not one global threshold.

### Follow-ups
How do you evaluate without collecting private home audio at scale? How would you add a new event type with few examples?

## personal-enhancement | Personalised speech enhancement with on-device adaptation | audio, edge, personalisation

### Prompt
Make a call-quality noise suppressor work better for each user (their voice, their typical environments) by adapting on the
device, without uploading audio.

### Clarify first
Device (phone, earbud, car)? What can adapt (a speaker embedding, a few layers, thresholds)? Compute/energy budget for
adaptation? How to know it's better without labels?

### Key numbers
Adaptation must be cheap (minutes of compute, during charging) or zero-shot (an enrolment embedding); inference cost unchanged;
no regression on other speakers (e.g. passengers on a car call).

### Reference design
Speaker-conditioned enhancement: a target-speaker embedding (from a short enrolment, computed on device) conditions a causal
enhancement network via FiLM layers (personalised speech enhancement / target speech extraction). Optional environment
adaptation: adjust a few normalisation statistics or a small adapter (LoRA-style) using self-supervised objectives on clean-
looking segments. Safeguards: fall back to the generic model if personalised output deviates, and A/B on device using
non-intrusive quality estimates (DNSMOS-style).

### Trade-offs & pitfalls
Personalisation can suppress other legitimate talkers (a passenger joining a call) — expose a mode switch. Embedding drift when the
voice changes; enrolment in noisy conditions; evaluation without references.

### Follow-ups
How would you evaluate personalised vs generic models fairly? What's the memory cost of storing several users' embeddings?
