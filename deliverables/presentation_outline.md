# DeepCue — Presentation Outline (Deep Learning Focus)

**Scope:** 14 slides, deep learning only. Full-stack/web components excluded (mention only where architecture demands it, e.g., "runs on CPU via ONNX").
**Suggested length:** 15–20 minutes.

---

## Slide 1 — Title
- **DeepCue: Real-Time Multimodal Emotion Recognition for Hebrew-Speaking Interview Candidates**
- Rachel Brodsky · Solo Project · Deep Learning Final Project
- One-line hook: *"What a candidate says, how they say it, and what their face shows — fused by a Transformer."*
- Visual: project logo / hero screenshot of the live emotion panel.

## Slide 2 — The Problem: Emotion Is Multimodal
- Human emotion is expressed through **three partially independent channels**: facial micro-expressions, vocal prosody (pitch, energy, tempo), and word semantics.
- Any single channel is ambiguous: a calm voice can mask an anxious face; positive words can carry sarcastic prosody.
- Interview setting raises the stakes: candidates actively regulate their expression → subtle, conflicting cues.
- Gap: almost no emotion-recognition work targets **Hebrew** speech semantics.
- Visual: 3-panel example of conflicting cues (same sentence, different face/voice).

## Slide 3 — The Solution: DeepCue Architecture (Mixture of Experts)
- Three independently trained deep "experts," one per modality, each producing class logits / a score.
- A **Cross-Modal Transformer** fuses the concatenated 17-dim expert output (video 8 + audio 8 + text 1) into a unified 8-class emotion.
- 8 classes: neutral, confident, anxious, happy, sad, angry, surprised, uncertain.
- Design principle: **decouple the experts** — each can be retrained/upgraded independently without touching the others.
- Visual: full architecture diagram (the [Architecture Diagram] block — experts → 17-dim vector → fusion → emotion).

## Slide 4 — Design Constraints That Shaped the DL Choices
- **CPU-only inference** on a weak Windows machine, no GPU → every model must export to ONNX and run fast.
- **End-to-end latency budget: < 10 seconds** from signal to fused prediction.
- **Quality gate:** macro F1 ≥ 0.50 per classifier.
- Consequence: compact backbones (EfficientNet-**B0**, not B7), 3-second audio chunks, landmark-based video input instead of raw pixels over the wire.

## Slide 5 — The Data
- **RAVDESS** (video + audio experts): 24 professional actors, 8 emotion categories, audio-visual recordings.
  - Key methodological point: **actor-disjoint splits** — no actor appears in both train and validation (prevents identity leakage; an early leaked split gave a misleading ~0.99 F1).
- **Hebrew Sentiment corpus** (`omilab/hebrew_sentiment`) for the text expert — adopted after CMU-MOSI availability issues; also makes the text expert natively Hebrew.
- Class-mapping caveat to state honestly: RAVDESS has no "confident" category → that class has zero support in video/audio evaluation.
- Visual: dataset sample grid + split diagram.

## Slide 6 — Video Expert: EfficientNet-B0 + LSTM
- Frame sampling from clips → EfficientNet-B0 extracts per-frame spatial features → LSTM models temporal dynamics → **mean pooling over timesteps** (outperformed last-timestep) → 8-class head.
- Training details: horizontal-flip augmentation, actor-disjoint validation, transfer learning from ImageNet weights.
- Why this pair: EfficientNet-B0 = best accuracy/FLOPs trade-off for CPU deployment; LSTM captures micro-expression *dynamics* a single frame misses.
- Result preview: **macro F1 ≈ 0.80**.

## Slide 7 — Audio Expert: Fine-Tuned wav2vec 2.0
- Self-supervised speech Transformer pre-trained on raw waveforms → fine-tuned for 8-class emotion on RAVDESS audio.
- **Two-stage fine-tuning:** Stage 1 trains the classification head with the backbone frozen; Stage 2 unfreezes for full fine-tuning (best val F1 ≈ 0.73 on the audio-only set).
- Paralinguistics, not words: the model reads *how* something is said — prosody, energy, voice quality.
- Result preview & honesty point: evaluation on a different source (audio tracks of AV MP4s) drops F1 to ≈ 0.45 → domain-shift lesson, revisited in Results.

## Slide 8 — Text Expert: XLM-RoBERTa for Hebrew Sentiment
- Multilingual masked-LM (100 languages) → fine-tuned as a **regression head** predicting continuous sentiment from Hebrew transcripts.
- Why regression: sentiment intensity is a graded signal that feeds fusion better than a hard label; evaluated with **MAE**.
- Why XLM-R: strongest open multilingual encoder for a low-resource-in-sentiment language like Hebrew.
- Result preview: **MAE ≈ 0.046** on normalized scale (~5% average error).
- Future work note: upgrade to an 8-class head to match sibling experts.

## Slide 9 — The Fusion Model: Cross-Modal Transformer
- Input: **17-dim vector** = video logits (8) ⊕ audio logits (8) ⊕ text sentiment (1).
- Cross-modal attention lets each modality's evidence re-weight the others (e.g., a confident voice can override an ambiguous face) → MLP head → 8-class output.
- Why a Transformer over simple averaging/voting: learned, *context-dependent* weighting instead of fixed weights — exactly what conflicting-cue cases require.

## Slide 10 — Fusion Training: The Architectural Viability Simulation
- Core challenge: **no real paired multimodal dataset exists** for Hebrew interview emotion — you'd need the same moment labeled across all three channels.
- Deliberate strategy: construct a **synthetic dataset simulating the empirical output distributions** of the three trained experts (intensity-modulated class means, realistic noise/overlap).
- Purpose: prove the cross-modal attention mechanism can learn the 17-dim feature space and that the full pipeline runs end-to-end — an *architectural proof-of-concept* that de-risks the system before real paired data arrives.
- Result: **macro F1 ≈ 0.98** on the simulation — states clearly: this validates the architecture's capacity, and real-world fusion performance will be established once deployed sessions generate paired data.
- Bonus methodological anecdote: the first simulation (overlapping class means) scored 0.41 — diagnosing that as a *data* problem, not a model problem, is itself a DL lesson.

## Slide 11 — Results & Evaluation
- Table (build as chart — see graphs script):
  - Video: **macro F1 = 0.80** ✅ (gate ≥ 0.50)
  - Audio: **macro F1 = 0.45** ⚠️ (just under gate; domain-shift explanation)
  - Text: **MAE = 0.046** ✅
  - Fusion (simulation): **macro F1 = 0.98** ✅
- Explain the two metrics in one breath each: macro F1 = per-class fairness under imbalance; MAE = average distance for regression.
- Visual: bar chart of F1 scores + MAE gauge (from `generate_evaluation_graphs.py`).

## Slide 12 — Challenges & Engineering Decisions
- **Quantization dropped for ONNX stability:** INT8 dynamic quantization shrank the video model 74% (33.3 → 8.7 MB) but introduced accuracy/stability issues in the ONNX inference path → shipped **full-precision ONNX**, still within the CPU latency budget. Lesson: compression is only free on paper.
- **Actor leakage:** first video result (~0.99) was too good to be true — actor-disjoint splits revealed the honest number.
- **Domain shift in audio:** train/eval source mismatch (WAV-only vs. AV-extracted audio) is the leading suspect for the 0.45.
- **Metric-model matching:** text expert initially gated with F1 despite being a regressor — corrected to MAE.

## Slide 13 — Future Work
- Collect **real paired multimodal data** from deployed interview sessions → retrain fusion on real distributions.
- Audio fine-tuning pass on AV-sourced audio to close the domain gap.
- Text expert: regression → 8-class classification for architectural symmetry.
- Revisit quantization with QAT (quantization-aware training) instead of post-training INT8.

## Slide 14 — Conclusions & Q&A
- Three take-home messages:
  1. Modality decoupling + late Transformer fusion is a practical, upgradeable recipe for multimodal emotion recognition.
  2. Rigorous evaluation hygiene (disjoint splits, right metric per task) changed every headline number in this project.
  3. Synthetic architectural validation is a legitimate way to de-risk fusion before paired data exists.
- Repo: github.com/RachelBSade/DeepCue
- Thank you / questions.

---

### Speaker notes — anticipated questions
- *"Isn't 0.98 fusion F1 meaningless?"* → It measures architectural capacity on a controlled distribution, not deployment performance — and we say so on the slide. The 0.41 → 0.98 progression also shows the metric responds to data quality as expected.
- *"Why not end-to-end training of all modalities jointly?"* → No paired data; also decoupling enables independent upgrades and fits the CPU latency budget.
- *"Why LSTM and not a temporal Transformer for video?"* → Sequence lengths are short, LSTM is cheaper on CPU, and the latency budget is hard.
