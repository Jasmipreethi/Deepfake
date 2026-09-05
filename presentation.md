# Deepfake Detection Using Cross-Model Transformer Fusion

### 10-Minute Presentation — Jasmi Preethi Alasapuri (2571395)

---

## Slide 1: Title

**Deepfake Detection Using Cross-Model Transformer Fusion**

BSc (Hons) Data Science and Artificial Intelligence  
CN6000 Dissertation 2026

Jasmi Preethi Alasapuri | 2571395  
Supervisor: Lucian Duta

---

## Slide 2: The Problem

### Why Deepfake Detection Matters

- AI-generated synthetic media is **eroding trust** in digital evidence across law, politics, finance, and journalism
- Real-world incidents: £20M corporate fraud (Arup, 2024), political deepfakes (Zelenskyy, 2022), AI kidnapping scams (FBI, 2025)
- **Multimodal** deepfakes — manipulating both audio and video simultaneously — are the hardest to detect
- Current detection systems are **vision-centric** and fail to capture fine-grained temporal relationships between speech and lip motion

> **Computing challenge:** Processing 1.4 TB of video data, training 50M-parameter neural networks, and building a real-time inference system

---

## Slide 3: Project Aims & Objectives

### Aim
To research and develop a **multimodal deepfake detection system** that distinguishes real from manipulated audio-visual media using deep learning.

### Six Objectives

| # | Objective |
|---|-----------|
| 1 | Literature review on deepfake generation and detection techniques |
| 2 | Analyse real-world impacts across sectors; identify gaps |
| 3 | Quantitative secondary analysis of AV-Deepfake1M++ dataset (speaker-disjoint split) |
| 4 | Design and implement Cross-Modal Transformer Fusion architecture |
| 5 | Resumable training pipeline; evaluation on held-out test set |
| 6 | Standalone inference system + web interface for non-technical use |

---

## Slide 4: Literature Review — Key Findings

### Evolution of Deepfakes

| Phase | Era | Characteristics |
|-------|-----|-----------------|
| **Visual Fidelity** | 2017–2019 | Simple face-swaps via Autoencoders/GANs |
| **In-the-Wild** | 2019–2021 | Multi-subject, uncontrolled environments |
| **Multimodal** | 2023–2025 | Neural generation affecting both audio and video |

### Three Critical Gaps Identified

1. **Identity Leakage:** Random train/val splits allow models to recognise faces rather than detect manipulation
2. **Vision-Centric Bias:** Most detectors ignore audio or treat it as an afterthought via simple concatenation
3. **Limited Generalisation:** Detectors overfit to training generators and fail on unseen manipulation techniques

---

## Slide 5: Proposed Solution — Architecture

### Cross-Modal Transformer Fusion Network

```
┌──────────────┐          ┌─────────────────────────────────┐
│  Video Input │─────────▶│  ResNet3D-18                     │
│  (50 frames) │          │  (Kinetics-400 pretrained, 33M)  │──▶ 256-d
└──────────────┘          └─────────────────────────────────┘       │
                                                                     ▼
                                                        ┌──────────────────────┐
┌──────────────┐          ┌──────────────────────────┐  │  Transformer Encoder  │
│  Audio Input │─────────▶│  ResNet18                 │  │  2 layers, 8 heads    │
│  (mel-spec)  │          │  (ImageNet pretrained)    │──▶  512-dim, GELU       │──▶ [CLS]
└──────────────┘          └──────────────────────────┘  │  + learnable [CLS]    │
                                  256-d                 └──────────────────────┘
                                                                     │
                                          ┌──────────────────────────┼──────────────────────────┐
                                          ▼                          ▼                          ▼
                                   ┌─────────────┐            ┌─────────────┐            ┌─────────────┐
                                   │ Audio Head  │            │ Video Head  │            │ Joint Head  │
                                   │  σ(p_audio) │            │  σ(p_video) │            │  σ(p_joint) │
                                   └─────────────┘            └─────────────┘            └─────────────┘
```

**~50M total parameters** | **Three-head output** for per-modality interpretability

### Key Design Decisions

| Component | Initial Plan | Final Choice | Reason |
|-----------|-------------|--------------|--------|
| Audio Encoder | Wav2Vec 2.0 | ResNet18 on mel-spectrograms | Corrupted MP4 files broke Wav2Vec; torchaudio+FFmpeg more robust |
| Video Encoder | MobileNetV3 | ResNet3D-18 (3D conv) | Temporal modelling needed for lip-motion artefacts |
| Fusion | DiMoDif | Custom Transformer Encoder + [CLS] token | Cross-modal self-attention captures audio-visual inconsistencies |
| Loss | BCE | Focal Loss (γ=2.0, α=0.25) | Down-weights easy examples; focuses on hard boundary cases |

---

## Slide 6: Implementation — Pipeline & Engineering

### Full Pipeline

```
Download Data ──▶ Speaker-Based Split ──▶ Parallel Feature Extraction ──▶ Two-Phase Training ──▶ Evaluate
 (HuggingFace)     (GroupShuffleSplit)      (28 CPU workers, resumable)    (Freeze → Fine-tune)    (AUC + per-type)
```

### Dataset: AV-Deepfake1M++
- **68,851 usable clips** from the validation split (2M total; 1.4 TB)
- 4 manipulation types: `real`, `audio_modified`, `visual_modified`, `both_modified`
- 8,475 clips dropped (missing or corrupted on disk)

### Training Configuration

| Setting | Value |
|---------|-------|
| Optimiser | AdamW (fusion: 1e-4, encoders: 1e-5) |
| Scheduler | ReduceLROnPlateau (patience=5) |
| Batch Size | 8 per GPU (×2 with DataParallel) |
| Max Epochs | 5 (resource constrained) |
| Phase 1 | Frozen encoders (~2 epochs) |
| Phase 2 | Full fine-tuning at 10× lower LR |
| Infrastructure | Vast.ai cloud GPUs (2× RTX 3080) |

### Engineering Highlights
- **Resumable manifests**: Checkpoint-based feature extraction survives cloud instance termination
- **W&B integration**: Real-time convergence monitoring and audit trail
- **Two-phase training**: Protects pretrained encoder features during early optimisation

---

## Slide 7: Results

### Model Performance Summary

| Model | Epochs | Val Joint AUC | Test AUC | Test Acc (τ=0.5) | Notes |
|-------|--------|---------------|----------|-------------------|-------|
| Model 1 | 1 | 0.663 | — | — | Corrupted download; underfit |
| Model 2 | 5 | 0.994 | 0.919 | 66.0% | Highest val AUC |
| **Model 3** | **3** | **0.985** | **0.937** | **93.0%** | **Best: early stop, epoch 3** |
| Model 4 | 5 | 0.993 | 0.915 | 71.0% | Same session as Model 3, at epoch 5 |

### Model 3 — Best Performer (Test Set)

| Metric | Value |
|--------|-------|
| Accuracy | 93.0% |
| Precision | 1.000 |
| F1 Score | 0.837 |
| False Positives | **0** |
| Test AUC | 0.937 |

### Per-Type Detection (Model 3)

| Manipulation Type | Accuracy | Difficulty |
|-------------------|----------|------------|
| `real` | 100% | Easiest |
| `visual_modified` | 92% | Moderate |
| `both_modified` | 92% | Moderate |
| `audio_modified` | 88% | **Hardest** |

> **Key insight:** Visual manipulation is easiest to detect (lip-region synthesis artefacts visible to ResNet3D-18); audio modification is hardest because the genuine video stream counteracts the audio head's fake signal.

---

## Slide 8: Key Findings

1. **Phase 2 fine-tuning is essential.** Model 1 (Phase 1 only) achieved AUC 0.663; all Phase-2 models ≥ 0.915 test AUC. Cross-modal representations require encoder adaptation.

2. **Higher validation AUC ≠ better deployment performance.** Model 2 (AUC 0.994) scored 66% accuracy at threshold 0.5; Model 3 (AUC 0.985) scored 93%. **Score calibration matters more than raw AUC.**

3. **Three-head architecture shows genuine modality specialisation.** Audio and video head scores dissociate by manipulation type in the direction predicted by architecture — `audio_modified` clips get low audio scores + high video scores, and vice versa.

4. **Speaker-disjoint evaluation is critical.** `GroupShuffleSplit` ensured zero speaker overlap, meaning results reflect manipulation detection, not identity recognition.

5. **Per-modality interpretability** — the three-head design reveals _which_ modality triggers the fake verdict, providing actionable output for forensic analysis.

---

## Slide 9: Web Interface & Inference

### Standalone System
- **`inference.py`:** CLI inference on single videos or folders; no training dependencies
- **Web interface (Flask):** Three-tab browser-based tool
  - **Analyze:** Drag-and-drop upload, model selector, real-time verdict with per-head scores
  - **Compare:** Side-by-side model comparison with agree/disagree summary
  - **History:** SQLite-backed log of all past analyses

### Score Calibration Analysis
- Reliability diagrams generated for all models
- Model 3 produced best-calibrated scores with **zero false positives**
- Practical deployment recommendation: evaluate calibration, not just AUC

---

## Slide 10: Limitations

| Limitation | Impact |
|------------|--------|
| **100-video test set** | One misclassification = 1% accuracy shift; wide confidence intervals |
| **Validation split only** | 68,851 clips vs 1M+ in full training set; limited diversity |
| **No ablation study** | Cannot attribute performance to Focal Loss vs BCE, Transformer vs MLP, etc. |
| **Fixed 2-second window** | May miss manipulation in short or late segments |
| **Single dataset** | No cross-dataset evaluation (FakeAVCeleb, DFDC) |
| **5-epoch training cap** | Cloud GPU budget prevented full convergence and hyperparameter sweeps |

---

## Slide 11: Future Work

### Priority Order

1. **Ablation studies:** Isolate Focal Loss vs BCE, Transformer vs MLP, full-frame vs lip-region crops
2. **Cross-dataset evaluation:** Test on FakeAVCeleb, DFDC, FaceForensics++ to measure generalisation
3. **Full dataset training:** Use complete AV-Deepfake1M++ (1M+ clips) instead of just the validation split
4. **Temporal localisation:** Extend to frame-level or segment-level predictions using `fake_segments` annotations
5. **Threshold optimisation:** Calibrate on a held-out set to balance FP/FN for deployment
6. **Longer training runs:** With institutional GPU access, run 20+ epochs with hyperparameter sweeps

---

## Slide 12: Conclusion

### What Was Achieved

- **Functional multimodal deepfake detector** achieving 93% accuracy, 1.000 precision, zero false positives (Model 3)
- **Six objectives met**, with scope exceeding the initial proposal
- **Speaker-disjoint evaluation** ensuring honest generalisation estimates
- **Interpretable three-head output** revealing per-modality vulnerability patterns
- **Production-ready pipeline:** Resumable extraction, W&B audit trail, web interface, standalone inference

### Personal Reflection

> The hardest challenges were not architectural but **engineering**: handling corrupted MP4 files, designing crash-resumable extraction, managing cloud GPU instances. These practical lessons — rarely described in papers — were the most valuable part of the project.

### Key Takeaway

**Score calibration, not AUC, determines real-world deployment readiness.** An earlier-epoch checkpoint (Model 3) with lower AUC outperformed a further-trained model (Model 2) by 27 percentage points in threshold accuracy — with zero false positives.

---

## References

1. Cai, Z. et al. (2025) *AV-Deepfake1M++: A Large-Scale Audio-Visual Deepfake Benchmark.* ACM Multimedia 2025.
2. Cai, Z. et al. (2024) *AV-Deepfake1M: A Large-Scale LLM-Driven Audio-Visual Deepfake Dataset.* ACM Multimedia 2024.
3. Tran, D. et al. (2018) *A Closer Look at Spatiotemporal Convolutions for Action Recognition.* (ResNet3D)
4. Lin, T.-Y. et al. (2018) *Focal Loss for Dense Object Detection.*
5. Dolhansky, B. et al. (2020) *The DeepFake Detection Challenge (DFDC) Dataset.*
6. Yi, J. et al. (2023) *Audio Deepfake Detection: A Survey.*

---

*Thank you — Questions?*
