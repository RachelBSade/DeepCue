# DeepCue: Transformer-Based Multimodal Emotion Recognition for Hebrew-Speaking Interview Candidates

**Author:** Rachel Brodsky (Solo Project) · B.Sc. Computer Science · Deep Learning Final Project

[Author Photo]

---

## Overview

When a job candidate answers a question, three channels of information arrive at once: the words they choose, the way their voice carries them, and the expressions that cross their face. Each channel alone is ambiguous — steady speech can mask an anxious face, and positive words can ride on strained prosody. **DeepCue** is a real-time system that reads all three channels simultaneously and fuses them, using Transformer-based cross-modal attention, into a single evolving picture of a candidate's emotional state across eight categories: neutral, confident, anxious, happy, sad, angry, surprised, and uncertain.

The project's distinctive contribution is its focus on **Hebrew-speaking candidates**. While emotion recognition research is dominated by English-language resources, DeepCue pairs universal non-verbal signals (facial dynamics, vocal prosody) with a linguistically localized text model fine-tuned on Hebrew sentiment data — bringing affective computing to a language largely underserved by the field.

## Architecture

[Architecture Diagram]

DeepCue adopts a **mixture-of-experts** design in which three independently trained deep networks each specialize in one modality. Facial expression is handled by an **EfficientNet-B0 + LSTM** pipeline that extracts spatial features from sampled video frames and models their temporal dynamics, capturing the micro-expressions a single frame would miss. The audio stream is processed along two complementary paths: a fine-tuned **wav2vec 2.0** — a self-supervised speech Transformer — hears *how* something is said, while **Whisper** transcribes *what* is said into Hebrew text. Speech semantics are then handled by **XLM-RoBERTa**, a multilingual Transformer fine-tuned as a sentiment regressor on Hebrew text, operating directly on Whisper's transcript.

The three experts emit a combined 17-dimensional feature vector (eight emotion logits from video, eight from audio, and one continuous sentiment score from text), which a **Cross-Modal Transformer** fuses through learned attention: each modality's evidence can dynamically re-weight the others, so a confident voice can override an ambiguous facial read in one moment while the face dominates in the next. This decoupled design also means any expert can be retrained or replaced without disturbing the rest of the system.

The entire inference path runs on **CPU-only hardware** via ONNX export, meeting a sub-10-second end-to-end latency budget with no GPU — a deliberate engineering constraint that shaped every model choice.

## Results

The video expert achieved a **macro F1 of 0.80** across eight emotion classes on RAVDESS under strict actor-disjoint evaluation, and the Hebrew text expert achieved a **mean absolute error of ~0.046** on a normalized sentiment scale — predictions landing within roughly 5% of ground truth on average. The audio expert reached a macro F1 of 0.45 under a deliberately challenging cross-source evaluation, identifying domain shift between training and evaluation audio as a documented avenue for refinement.

Because no paired multimodal dataset yet exists for Hebrew interview emotion, the fusion stage was validated through an **architectural viability simulation**: a carefully constructed synthetic dataset modeling the empirical output distributions of the three trained experts. On this benchmark the Cross-Modal Transformer achieved a **macro F1 of 0.98**, demonstrating that the attention mechanism reliably learns the 17-dimensional fused feature space and that the full pipeline operates end-to-end — a deliberate proof-of-concept that prepares the architecture for real paired data as deployed interview sessions begin generating it.

## Significance

DeepCue demonstrates that rigorous multimodal deep learning — transfer learning across vision, speech, and language Transformers, disciplined evaluation methodology, and attention-based fusion — can be engineered into a single real-time system by one developer on consumer hardware, and extended to a language the field has largely overlooked. Beyond interview practice, the same architecture generalizes to telehealth, education, and any domain where understanding *how* people feel matters as much as what they say.

**Repository:** [github.com/RachelBSade/DeepCue](https://github.com/RachelBSade/DeepCue)
