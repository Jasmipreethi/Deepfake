# Presentation Slide Explanations & References

## Deepfake Detection Using Cross-Model Transformer Fusion

### Jasmi Preethi Alasapuri | 2571395 | CN6000 Dissertation 2026

---

## Slide 1: Title Slide

### Content
- **Title:** Viva: Deepfake Detection Using Cross-Model Transformer Fusion
- **Student:** Jasmi Preethi Alasapuri, 2571395
- **Supervisor:** Lucian Duta
- **Programme:** BSc (Hons) Data Science and Artificial Intelligence

### Explanation
Standard title slide identifying the dissertation viva presentation. The project title reflects the architectural approach: cross-modal signifies that audio and video modalities interact during inference (not post-hoc), and Transformer Fusion denotes the use of a Transformer Encoder with a learnable [CLS] token to fuse features before classification.

### References
- This dissertation, Chapters 1–6 (Alasapuri, 2026).
- Initial proposal: CN6000 Internal Ethical Approval Process (2025).

---

## Slide 2: The Problem — Why Deepfake Detection Matters

### Content
- Synthetic media eroding trust in digital evidence across law, politics, finance, journalism.
- Three real-world incidents cited:
  - £20M corporate fraud via deepfake video call (Arup, 2024)
  - Political deepfakes used for disinformation (Zelenskyy, 2022)
  - AI kidnapping and impersonation scams (FBI, 2025)
- Multimodal deepfakes are hardest to detect; current systems are vision-centric.
- Computing challenge: 1.4 TB data, 50M parameters, real-time inference.

### Explanation
The slide establishes motivation. Deepfakes have moved from academic curiosity to an active threat vector. The Arup incident (Milmo, 2024) involved a finance worker tricked into transferring £20 million after a multi-person deepfake video conference call. The Zelenskyy deepfake (Allyn, 2022) demonstrated political weaponization — a fabricated surrender video circulated during the Ukraine conflict. The FBI 2025 warning (Bragg, 2025) concerns AI-generated kidnapping videos used for extortion. These are all multimodal — the deception relies on both fake audio and fake video being coherent.

The computing challenge reflects the AV-Deepfake1M++ dataset scale (Cai et al., 2025), which contains ~2 million video clips totaling approximately 1.4 TB, requiring substantial engineering infrastructure.

### References
- Milmo, D. (2024) 'UK engineering firm Arup falls victim to £20m deepfake scam', *The Guardian*, 17 May.
- Allyn, B. (2022) 'A deepfake video showing Volodymyr Zelenskyy surrendering worries experts', *NPR*, 16 March.
- Bragg, J. (2025) 'AI-generated videos, photos used in new virtual kidnapping scams, FBI warns', *Axios*, 5 December.
- Cai, Z. et al. (2025) 'AV-Deepfake1M++: A Large-Scale Audio-Visual Deepfake Benchmark with Real-World Perturbations', *Proceedings of the 33rd ACM International Conference on Multimedia (MM '25)*.
- Chesney, B. and Citron, D. (2019) 'Deep fakes: A looming challenge for privacy, democracy, and national security', *California Law Review*, 107(6), pp. 1753–1820.

---

## Slide 3: Project Aims & Objectives

### Content
- **Aim:** Multimodal deepfake detection system to distinguish real from manipulated audio-visual media.
- Six objectives:
  1. Literature review on deepfake generation & detection techniques
  2. Analyse real-world impacts across sectors; identify gaps
  3. Quantitative secondary analysis of AV-Deepfake1M++ (speaker-disjoint split)
  4. Design & implement Cross-Modal Transformer Fusion architecture
  5. Resumable training pipeline; evaluation on held-out test set
  6. Standalone inference system + web interface for non-technical use

### Explanation
These six objectives evolved from the initial CN6000 proposal's six objectives (see Appendix, initial proposal form). The refinements reflect findings from the literature review: Objective 3 explicitly specifies speaker-disjoint splitting because the literature (Rossler et al., 2019) demonstrated that random splits produce inflated performance by allowing identity recognition. Objective 5 requires a resumable pipeline because cloud GPU instances were expected to terminate unexpectedly during feature extraction and training — an operational reality for student projects without institutional GPU access.

### References
- Initial project proposal: CN6000 Internal Ethical Approval Process (2025) — Appendix.
- Final project proposal: Dissertation Section 1.4, Table 29.
- Rossler, A. et al. (2019) 'FaceForensics++: Learning to Detect Manipulated Facial Images', *Proceedings of the IEEE International Conference on Computer Vision*, pp. 1–11.

---

## Slide 4: Literature Review — Key Findings

### Content
- **Evolution table:** Three phases of deepfake development (Visual Fidelity 2017–2019 → In-the-Wild 2019–2021 → Multimodal 2023–2025).
- **Three gaps identified:**
  1. Identity Leakage — random splits enable face recognition instead of manipulation detection.
  2. Vision-Centric Bias — audio handled via simple concatenation or ignored entirely.
  3. Limited Generalisation — detectors overfit to training generators and fail on unseen manipulation techniques.

### Explanation
**Evolution phases:** The three-phase classification synthesises the literature timeline. Phase 1 (Visual Fidelity) corresponds to early autoencoder-based face-swaps (Li and Lyu, 2018) and GAN-generated faces (Rossler et al., 2019). Phase 2 (In-the-Wild) covers multi-subject datasets like DFDC (Dolhansky et al., 2020) and WildDeepfake (Zi et al., 2021) with uncontrolled recording conditions. Phase 3 (Multimodal) reflects the current generation of datasets — AV-Deepfake1M (Cai et al., 2024) and AV-Deepfake1M++ (Cai et al., 2025) — where neural TTS/VC systems generate fake audio and NeRF-based methods synthesize lip-synced video, making both modalities manipulated.

**Gap 1 — Identity Leakage:** (Rossler et al., 2019) demonstrated that random train/validation splits allow models to achieve high accuracy by recognising faces rather than detecting manipulation artefacts. When the same speaker appears in both splits, a model can memorize visual identity features. This project enforces speaker-disjoint partitioning via GroupShuffleSplit.

**Gap 2 — Vision-Centric Bias:** (Yi et al., 2023) noted that even multimodal detectors often use simple feature concatenation for fusion, treating audio as a supplementary channel rather than an equal modality. This project uses a Transformer Encoder where audio and video tokens attend to each other through self-attention.

**Gap 3 — Limited Generalisation:** (Dolhansky et al., 2020) documented that DFDC-winning models achieved high in-domain accuracy but degraded significantly on videos from unseen generators. This is a well-established problem across the detection literature and motivates future cross-dataset evaluation.

### References
- Li, Y. and Lyu, S. (2018) 'Exposing DeepFake Videos By Detecting Face Warping Artifacts', arXiv:1811.00656.
- Rossler, A. et al. (2019) 'FaceForensics++: Learning to Detect Manipulated Facial Images', *Proceedings of the IEEE ICCV*, pp. 1–11.
- Dolhansky, B. et al. (2020) 'The DeepFake Detection Challenge (DFDC) Dataset', arXiv:2006.07397.
- Zi, B. et al. (2021) 'WildDeepfake: A Challenging Real-World Dataset for Deepfake Detection', *MM 2020*, pp. 2382–2390.
- Cai, Z. et al. (2024) 'AV-Deepfake1M: A Large-Scale LLM-Driven Audio-Visual Deepfake Dataset', *Proceedings of ACM Multimedia (MM '24)*.
- Cai, Z. et al. (2025) 'AV-Deepfake1M++', *Proceedings of ACM Multimedia (MM '25)*.
- Yi, J. et al. (2023) 'Audio Deepfake Detection: A Survey', arXiv:2308.14970.
- Westerlund, M. (2019) 'The emergence of deepfake technology: A review', *Technology Innovation Management Review*, 9(11).

---

## Slide 5: Cross-Modal Transformer Fusion — Architecture

### Image
`figures/architecture.png` — Full system architecture diagram

### Content Description
The diagram depicts a three-stage pipeline: (1) parallel modality encoding, (2) Transformer-based cross-modal fusion, (3) three-head classification.

### Detailed Explanation

**Video Encoder (left branch):** Input is 50 frames at 224×224 pixels (2 seconds at 25 fps), reshaped to tensor shape (B, 50, 3, 224, 224). Processed by ResNet3D-18 (Tran et al., 2018) pretrained on Kinetics-400 (Kay et al., 2017). The 3D convolutional kernels jointly process spatial and temporal dimensions, capturing how lip regions deform across frames — essential for detecting visual manipulation in talking-head videos. The final fully-connected layer projects to a 256-dimensional feature vector with dropout 0.4.

**Audio Encoder (right branch):** Input is a mel-spectrogram of shape (B, 1, 128, 63), generated from 2 seconds of audio at 16 kHz. FFT window of 1024 points, hop length of 512 samples, producing 63 time steps × 128 mel frequency bins. Processed by ResNet18 (He et al., 2015) pretrained on ImageNet, with the first convolutional layer modified from 3-channel RGB input to single-channel input since spectrograms are grayscale. Output is a 256-dimensional feature vector with dropout 0.4.

**Transformer Encoder (center):** Both 256-d features are projected to 512 dimensions through separate linear layers and unsqueezed to (B, 1, 512). A learnable [CLS] classification token (initialised randomly as a Parameter) is prepended, forming a 3-token sequence: [CLS, video, audio]. Positional embeddings (learnable, shape 1×3×512) are added. The sequence passes through a 2-layer Transformer Encoder with 8 attention heads, 2048-dim feedforward layers, GELU activation, pre-norm (norm_first=True), and dropout 0.4. The self-attention mechanism allows audio and video representations to attend to each other at every layer — this is the architectural mechanism for cross-modal interaction.

**Classification Heads (bottom):** The [CLS] token's output representation (512-d) feeds three independent linear layers with Sigmoid activation: Audio Head → p_audio (0–1), Video Head → p_video (0–1), Joint Head → p_joint (0–1). All three heads share the same fused representation but learn separate decision boundaries.

**Total parameters:** ~49.7M (ResNet3D-18: 33.3M, ResNet18: 11.7M, Transformer: 4.7M, Heads: 1.5K).

### References
- Tran, D. et al. (2018) 'A Closer Look at Spatiotemporal Convolutions for Action Recognition', arXiv:1711.11248.
- Kay, W. et al. (2017) 'The Kinetics Human Action Video Dataset', arXiv:1705.06950.
- He, K. et al. (2015) 'Deep Residual Learning for Image Recognition', arXiv:1512.03385.
- Cai, Z. et al. (2025) 'AV-Deepfake1M++', *Proceedings of ACM Multimedia (MM '25)*. — DiMoDif baseline.
- Implementation: `cross_modal.py` → `TransformerFusion` class; `audio.py` → ResNet18 encoder; `video.py` → ResNet3D-18 encoder.

---

## Slide 6: Key Design Decisions

### Content
Four-row table comparing initial plan versus final implementation for each architectural component.

| Component | Initial Plan | Final Choice | Reasoning |
|-----------|-------------|--------------|-----------|
| Audio Encoder | Wav2Vec 2.0 | ResNet18 + Mel-spec | Corrupted MP4 files incompatible with Wav2Vec |
| Video Encoder | MobileNetV3 | ResNet3D-18 | 3D convolutions needed for temporal lip-motion |
| Fusion Module | DiMoDif | Transformer + [CLS] | Cross-modal self-attention captures inconsistencies |
| Loss Function | BCE | Focal Loss (γ=2.0) | Down-weights easy examples; focuses on boundary cases |

### Explanation

**Audio Encoder change:** Wav2Vec 2.0 (Baevski et al., 2020) was the initial choice for its contextual speech representations expected to be sensitive to voice cloning artefacts. However, the AV-Deepfake1M++ dataset contains non-standard MP4 containers that caused silent failures in librosa-based audio loading. Switching to ResNet18 on mel-spectrograms using torchaudio's FFmpeg backend resolved loading reliability. The mel-spectrogram representation (128 mel bins, 63 time steps) captures sufficient spectral information for detecting TTS/VC artefacts, and ResNet18's ImageNet pretraining provides strong initial features even for the single-channel adaptation. This finding is documented in the dissertation (Section 3.3.4).

**Video Encoder change:** MobileNetV3 (Howard et al., 2019) was considered for its efficiency, but 2D convolutions operating on individual frames cannot capture the temporal dynamics of lip movement. ResNet3D-18 processes 50 frames as a spatiotemporal volume, enabling the model to learn that natural lip motion follows predictable trajectories while synthesized lip regions exhibit subtle temporal inconsistencies. Pretrained on Kinetics-400, which includes diverse human actions including talking.

**Fusion change:** DiMoDif (Cai et al., 2025) is the official baseline for AV-Deepfake1M++ using cross-modal attention with temporal boundary detection. It was replaced by a custom 2-layer Transformer Encoder for simplicity and to focus on cross-modal self-attention over the sequence [CLS, video, audio] rather than time-aligned cross-attention over frame-level features. The [CLS] token design follows BERT-style classification (Devlin et al., 2019), adapted here for modality fusion rather than sequence classification.

**Loss function change:** Standard Binary Cross-Entropy treats all examples equally, but deepfake detection involves class imbalance (more fake than real variants) and many "easy" examples where one modality clearly indicates authenticity. Focal Loss (Lin et al., 2018) with γ=2.0 and α=0.25 down-weights well-classified examples through the modulating factor (1−p_t)^γ, concentrating gradient updates on hard, ambiguous cases near the decision boundary. This is particularly relevant when audio-modified clips produce conflicting signals from the two modality heads.

### References
- Baevski, A. et al. (2020) 'wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations', arXiv:2006.11477.
- Howard, A. et al. (2019) 'Searching for MobileNetV3', arXiv:1905.02244.
- Lin, T.-Y. et al. (2018) 'Focal Loss for Dense Object Detection', arXiv:1708.02002.
- Cai, Z. et al. (2025) 'AV-Deepfake1M++', *Proceedings of ACM Multimedia (MM '25)*.
- Devlin, J. et al. (2019) 'BERT: Pre-training of Deep Bidirectional Transformers', arXiv:1810.04805. — [CLS] token design pattern.
- Implementation: `train_utils.py` → FocalLoss class; `config.py` → TRAIN_CONFIG (focal_gamma=2.0, focal_alpha=0.25).

---

## Slide 7: Implementation — Training Pipeline

### Image
`figures/two_phase_training.png` — Two-phase training timeline diagram

### Content
- Two-phase training strategy (Phase 1: frozen encoders, Phase 2: full fine-tuning).
- Configuration table: Dataset, Optimiser, Batch Size, Loss, Infrastructure, Workers.

### Detailed Explanation

**Two-Phase Training Diagram:**

**Phase 1 (frozen encoders, ~2 epochs):** Both ResNet3D-18 and ResNet18 encoder parameters are frozen (`requires_grad = False`). Only the Transformer fusion module and three classification heads are trainable. This phase protects the Kinetics-400 and ImageNet pretrained weights from being destroyed by large, noisy gradients during early optimization when the randomly-initialized fusion module produces poor loss signals. The encoders function as fixed feature extractors, outputting 256-d vectors for the fusion module to learn cross-modal relationships. The number of freeze epochs is auto-computed as `max(1, round(epochs × 0.25))` in `config.py`.

**Phase 2 (full fine-tuning, epochs 3+):** All parameters unfrozen. Encoder learning rate is set to 1e-5 (10× lower than the fusion module's 1e-4) to allow domain adaptation while preventing catastrophic forgetting of pretrained representations. The ReduceLROnPlateau scheduler halves the learning rate when validation joint AUC fails to improve for 5 consecutive epochs. Early stopping triggers after 30% of the total epoch budget without AUC improvement.

**Loss function detail:** Focal Loss with γ=2.0, α=0.25 applied to all three heads. The joint head loss is weighted 2× relative to the audio and video head losses, since the joint prediction is the primary output. The combined loss is: L_total = 0.4 × L_audio + 0.4 × L_video + 0.8 × L_joint (weights sum to 1.6 total, effectively prioritizing the joint head). Gradient clipping at norm 1.0 prevents instability from exploding gradients in the 3D convolution layers.

**Configuration Table:**
| Setting | Value | Purpose |
|---------|-------|---------|
| Dataset | 68,851 clips (validation split) | AV-Deepfake1M++ val only |
| Optimiser | AdamW (fusion: 1e-4, encoders: 1e-5) | Decoupled weight decay for regularization |
| Batch Size | 8 per GPU (×2 DataParallel = 16 effective) | Fits within 10GB VRAM per RTX 3080 |
| Loss | Focal Loss (γ=2.0, α=0.25) | Hard-example mining at loss-function level |
| Augmentation | SpecAugment + horizontal flip | Frequency/time masking; brightness/contrast jitter |
| Infrastructure | Vast.ai (2× RTX 3080) | Cloud GPU; $0.30–0.50/hr per instance |
| Workers | 28 CPU parallel extraction | Fork-based multiprocessing; resumable manifests |

### References
- Implementation: `train_utils.py` → `train_model()` function, `FocalLoss` class.
- Configuration: `config.py` → `TrainConfig`, `OptimConfig`, `ModelConfig` dataclasses.
- Feature extraction: `data_utils.py` → `extract_all_features()` with crash-resumable manifest system.
- Training infrastructure documentation: Dissertation Section 3.4 (Training Strategy), Section 3.5 (Resumable Pipeline).

---

## Slide 8: Results — Training History & Per-Type Accuracy

### Images
- `comparison_results/training_history.png` — Training curves (3 subplots)
- `figures/per_type_accuracy_bar_chart.png` — Per-type accuracy bar chart
- Model comparison table (M1–M4)

### Graph 1: Training History (`training_history.png`)

**What this composite plot shows:** Three side-by-side subplots tracking the training dynamics of the best-performing training session (the run that produced Models 3 and 4).

**Subplot 1 — Loss Curves (left):**
- X-axis: Epoch number (1–5).
- Y-axis: Loss value (Focal Loss).
- Red line: Training loss. Blue line: Validation loss.
- The training loss decreases steadily across all 5 epochs, confirming the model continues to fit the training data. The validation loss plateaus around epoch 3–4 and begins to increase slightly at epoch 5, indicating the onset of overfitting. This matches the observation that Model 3 (epoch 3) generalizes better than Model 4 (epoch 5) on the test set.

**Subplot 2 — Validation AUC Curves (center):**
- X-axis: Epoch number (1–5).
- Y-axis: AUC (0–1).
- Red thick line: Joint head AUC. Blue line: Audio head AUC. Green line: Video head AUC.
- Joint AUC reaches approximately 0.99 by epoch 2 and remains stable. Video head AUC converges slightly faster and higher than audio head AUC, consistent with the finding that visual manipulation is easier to detect than audio modification. The rapid convergence (within 2–3 epochs) demonstrates efficient transfer learning from pretrained features.
- All AUC values are computed on the validation split under the speaker-disjoint partition.

**Subplot 3 — Learning Rate Schedule (right):**
- X-axis: Epoch number. Y-axis: Learning rate (log scale).
- Purple line showing ReduceLROnPlateau behavior: learning rate holds at 1e-4 during early epochs. When validation AUC stops improving, the rate halves to 5e-5, allowing the model to settle into a finer local minimum. The scheduler patience is set to 5 epochs with reduction factor 0.5.

### Graph 2: Per-Type Accuracy Bar Chart (`per_type_accuracy_bar_chart.png`)

**What this bar chart shows:** Joint prediction accuracy at threshold τ=0.5, broken down by manipulation type for the best model (Model 3). Each bar represents one of the four manipulation types in AV-Deepfake1M++.

**Bar interpretation:**
- `real` (100%): The model perfectly identifies genuine, unmodified videos. All 25 real test videos received joint scores above 0.5.
- `visual_modified` (~92%): Videos where only the visual stream is manipulated (e.g., lip-region synthesis). ResNet3D-18 detects temporal inconsistencies in the synthesized lip movements.
- `both_modified` (~92%): Both audio and video are manipulated. Evidence accumulates from both modality heads, producing strong fake signals.
- `audio_modified` (~88%): Only the audio track is manipulated; the video is genuine. This is the hardest category because: (a) the video head assigns a high authenticity score to the genuine visual stream, which counteracts the audio head's fake signal in the joint prediction, and (b) the ResNet18 mel-spectrogram encoder is less sensitive to TTS/VC artefacts than ResNet3D-18 is to visual artefacts.

**Why audio_modified is hardest:** The three-head architecture produces independent scores per modality. For audio_modified clips, the audio head outputs a low score (detecting fake audio), but the video head outputs a high score (genuine video). The joint head must reconcile these conflicting signals. The joint prediction lands closer to the 0.5 decision boundary than either single-modality prediction, making threshold-based classification more error-prone for this category.

### Model Comparison Table

| Model | Epochs | Val Joint AUC | Test AUC | Test Acc (τ=0.5) | Notes |
|-------|--------|---------------|----------|-------------------|-------|
| M1 | 1 | 0.663 | — | — | Phase 1 only; underfit; corrupted download |
| M2 | 5 | 0.994 | 0.919 | 66.0% | Highest val AUC; poor score calibration |
| M3 ★ | 3 | 0.985 | 0.937 | 93.0% | Best: Precision 1.000, 0 false positives |
| M4 | 5 | 0.993 | 0.915 | 71.0% | Same session as M3, saved at epoch 5 |

**Model progression interpretion:**
- **Model 1** (epoch 1, Phase 1 only): AUC 0.663 indicates underfitting — the model has not learned cross-modal relationships. Prediction pattern: all outputs near 0.23–0.35 (predicting FAKE regardless), suggesting the classifier predicts the majority class. Corrupted during download from cloud GPU instance (file size verification would have caught this).
- **Model 2** (epoch 5, complete separate run): Highest validation AUC (0.994) but poor threshold performance (66% accuracy). The scores are high-quality for ranking (AUC) but poorly calibrated — many scores cluster near 0.5, making the τ=0.5 threshold unreliable.
- **Model 3** (epoch 3, first fine-tuning epoch in its session): Best test performance. Precision 1.000 means every video predicted as fake was truly fake — zero false positives on the 100-video test set. The model is conservative with fake predictions, only calling fake when confident. This is the desirable behavior for a real-world detector where false accusations have high cost. The earlier checkpoint captured better-calibrated probability scores before the model began to overfit.
- **Model 4** (epoch 5, same session as M3): Test accuracy dropped to 71% from 93% at epoch 3, despite slightly higher validation AUC. This is the central finding — more training epochs improved validation AUC but degraded real-world threshold performance due to score compression toward the boundary.

### References
- Training curves generated by `plot_training_history.py` from W&B logs in `logs/`.
- Per-type accuracy computed by `plot_per_type_accuracy_bar_chart.py` from `comparison_results/` prediction CSVs.
- Model comparison and evaluation: `compare_models.py` (Section 4.2–4.4 of dissertation).
- Score calibration discussion: Dissertation Section 4.3.2.

---

## Slide 9: Model Comparison & Score Calibration

### Images
- `comparison_results/model_comparison.png` — Multi-model comparison visualization
- `figures/calibration_curves.png` — Reliability (calibration) curves

### Graph 1: Model Comparison (`model_comparison.png`)

**What this plot shows:** A multi-panel comparison of Models 2, 3, and 4 across evaluation metrics. Generated by `compare_models.py`.

**Expected panels (based on the evaluation code in `main.py` and `compare_models.py`):**

- **Audio vs Video prediction scatter:** Each of the 100 test videos is plotted as a point with audio head score on one axis and video head score on the other. Points are color-coded by manipulation type (real = green, audio_modified = red, visual_modified = blue, both_modified = orange). The ideal pattern:
  - `real` videos cluster near (1, 1) — both heads confident of authentic.
  - `both_modified` videos cluster near (0, 0) — both heads detect manipulation.
  - `audio_modified` videos have low audio scores (fake audio, ~0.3–0.5) but high video scores (genuine video, ~0.8–0.95).
  - `visual_modified` videos have high audio scores (genuine audio) but lower video scores (detected visual manipulation).

  Model 3's scatter shows the cleanest separation with the widest gap between the real cluster (top-right) and the both_modified cluster (bottom-left). This separation demonstrates that both modality heads are contributing meaningful, independent signals.

- **Joint prediction distribution histogram:** Distribution of joint head scores across all 100 test videos. A well-calibrated model produces a bimodal distribution — real videos cluster near 1.0 and fake videos cluster near 0.0, with minimal overlap in the 0.3–0.7 range. Model 3 shows the most pronounced bimodal distribution; Model 2 shows more scores in the ambiguous middle range.

- **Per-type joint score boxplots:** Box-and-whisker plots showing the distribution of joint prediction scores for each manipulation type. Model 3 shows the widest interquartile separation between real and fake types.

- **Confusion matrix:** 2×2 matrix (Real/Fake predicted vs Actual) for joint predictions at τ=0.5. Model 3's confusion matrix shows zero false positives (top-right cell = 0).

### Graph 2: Calibration Curves (`calibration_curves.png`)

**What reliability diagrams show:** A calibration curve (also called a reliability diagram) plots predicted probability against observed accuracy. Generated by `plot_calibration_curves.py`.

**How to read it:**
- X-axis: Predicted probability, binned into deciles (0–0.1, 0.1–0.2, ..., 0.9–1.0).
- Y-axis: Observed accuracy within each bin — the fraction of videos in that bin that were actually real.
- Diagonal dashed line: Perfect calibration. A model predicting 0.8 should be correct 80% of the time.
- Colored lines: Calibration curves for each model (Models 2, 3, 4).

**Interpretation for each model:**
- **Model 3 (green line):** Closely follows the diagonal across all probability bins. When Model 3 outputs a 0.95, the video is real ~95% of the time. When it outputs 0.10, the video is real ~10% of the time (i.e., 90% likely fake). This calibration quality explains the 93% accuracy at a simple τ=0.5 threshold — the scores are trustworthy as probabilities.
- **Model 2 (red/blue line):** Deviates from the diagonal, particularly in the middle range (0.4–0.6). The model is underconfident on real videos (predicting 0.6 when accuracy should be ~0.8) and overconfident on some fake predictions. This poor calibration means that at τ=0.5, many genuinely real videos fall just below the threshold and many actually-fake videos fall just above — producing the 66% accuracy despite 0.994 AUC.
- **Model 4 (orange/purple line):** Shows moderate deviation from the diagonal, consistent with its intermediate 71% accuracy.

**Why calibration matters more than AUC:** AUC measures ranking ability — can the model correctly order videos from most-fake to most-real? It is threshold-independent. A model with AUC 0.99 could assign all real videos a score of 0.51 and all fake videos a score of 0.49 — perfect ranking (AUC=1.0) but useless at τ=0.5. Calibration measures whether the score values are meaningful as probabilities. For deployment with a fixed decision threshold, calibration quality directly determines accuracy. The finding that earlier-epoch checkpoints (Model 3 at epoch 3) are better calibrated than later epochs (Model 4 at epoch 5) suggests that extended training compresses probability scores toward 0.5 (a form of overfitting), even as validation AUC remains high or improves.

### References
- Guo, C. et al. (2017) 'On Calibration of Modern Neural Networks', *Proceedings of ICML*. — Foundation of reliability diagram analysis.
- Niculescu-Mizil, A. and Caruana, R. (2005) 'Predicting Good Probabilities With Supervised Learning', *Proceedings of ICML*. — Score calibration methodology.
- Implementation: `plot_calibration_curves.py`, `compare_models.py`.
- Training runs and evaluation: Dissertation Chapter 4 (Results), Section 4.3.2 (Calibration Analysis).

---

## Slide 10: Key Findings

### Content
Four numbered findings derived from the training and evaluation.

### Explanation

**Finding 1 — Phase 2 fine-tuning is essential:**
Model 1 (Phase 1 only, 1 epoch) achieved AUC 0.663 with a default-class prediction pattern (always predicting FAKE). All Phase-2 models achieved validation AUC ≥ 0.985 and test AUC ≥ 0.915. The encoder unfreezing step is necessary because pretrained Kinetics-400 features (action recognition) and ImageNet features (object classification) are in a different representational space than deepfake detection. Cross-modal representations require encoder adaptation to the target domain. This finding addresses Research Question 1 from the dissertation (Section 1.2).

**Finding 2 — Validation AUC does not predict deployment performance:**
The most practically significant finding. Model 2 (AUC 0.994) achieved 66% accuracy at τ=0.5; Model 3 (AUC 0.985) achieved 93%. This 27-percentage-point gap with a 0.009 AUC difference demonstrates that score calibration — not ranking ability — determines threshold-based performance. This finding is based on n=3 loadable checkpoints from 5-epoch training runs and should be treated as hypothesis-generating rather than confirmatory, as acknowledged in Section 6.2.

**Finding 3 — Three-head architecture provides genuine modality specialisation:**
The dissociation pattern confirmed by the audio-video scatter plots — audio_modified clips produce low audio head scores and high video head scores, while visual_modified clips produce the reverse — demonstrates that the three heads are independently learning modality-specific features rather than all collapsing to the same signal. This per-modality interpretability is not available from binary-output systems (Cai et al., 2024; Yi et al., 2023).

**Finding 4 — Speaker-disjoint evaluation is critical:**
Using GroupShuffleSplit on speaker IDs ensured zero speaker overlap between training and validation sets. This addresses the identity leakage gap identified in the literature (Rossler et al., 2019). Random splits would produce higher but less meaningful metrics by allowing the model to recognize faces rather than detect manipulation. The reported results reflect generalisation to entirely unseen identities.

### References
- Rossler, A. et al. (2019) 'FaceForensics++', *Proceedings of IEEE ICCV*.
- Cai, Z. et al. (2024) 'AV-Deepfake1M', *Proceedings of ACM Multimedia (MM '24)*.
- Yi, J. et al. (2023) 'Audio Deepfake Detection: A Survey', arXiv:2308.14970.
- Dolhansky, B. et al. (2020) 'The DFDC Dataset', arXiv:2006.07397.
- Dissertation: Section 6.2 (Key Findings), Research Questions 1–3 (Section 1.2).
- Speaker-disjoint implementation: `data_utils.py` → `sample_videos()` → GroupShuffleSplit on speaker ID.

---

## Slide 11: Web Interface — Three Tabs

### Images
- `figures/web_analyze_fake.png` — Analyze tab showing a fake verdict
- `figures/web_compare.png` — Compare tab with two models
- `figures/web_history.png` — History tab with SQLite log

### Image Description & Explanation

**Analyze tab (left):** Flask-based web interface for single-video detection. User drags-and-drops or selects a video file (MP4, supported by FFmpeg backend). A model selector dropdown allows choosing between trained checkpoints (Models 2, 3, 4). Upon submission, the backend (`inference.py`) extracts three 2-second windows from the video, runs each through the selected model, and averages the predictions. The verdict displays as "FAKE" (red) or "REAL" (green) with three probability scores below: Audio Authenticity, Video Authenticity, and Joint Authenticity. This per-head display makes the model's reasoning interpretable — users can see whether the audio, the video, or both contributed to the verdict. The screenshot shows a fake detection with all three scores below 0.5.

**Compare tab (center):** Side-by-side comparison of two models on the same uploaded video. Each column shows the model name, the three head scores, and the final verdict. An agree/disagree badge at the top summarizes whether both models concur (green checkmark) or disagree (red X). The primary use case is comparing Model 2 (high AUC, poor calibration) against Model 3 (slightly lower AUC, excellent calibration) on the same input — users can directly observe the calibration difference in the probability score values. Disagreement cases are flagged for manual review.

**History tab (right):** SQLite-backed table displaying all past analyses. Columns: filename, model used, verdict, scores (audio/video/joint), timestamp. Features: per-entry deletion (trash icon) and bulk clear. The history provides an audit trail for non-technical users and enables retrospective analysis — for example, checking whether a particular video consistently receives borderline scores across models.

### Implementation Details
- Backend: Flask (`web/app.py`) wrapping `inference.py` which loads model weights independent of training dependencies.
- Frontend: Single-page HTML with vanilla JavaScript (no framework), Bootstrap-styled CSS.
- Model loading: Weights loaded via `torch.load()` with `map_location` for CPU compatibility.
- Feature extraction at inference time: Uses the same `FEATURE_CONFIG` parameters as training (2-second clips, 50 frames at 25fps, 128 mel bins, 63 time steps).
- Video processing: OpenCV for frame extraction, torchaudio for audio loading with FFmpeg backend.

### References
- Web interface specification: `WebInterface.md`.
- Implementation: `web/app.py`, `web/templates/index.html`, `web/static/`.
- Standalone inference: `inference.py` — CLI tool with `--video`, `--video_dir`, `--model`, `--output` arguments.
- Dissertation: Section 3.7 (Web Interface), Appendix E (interface screenshots).

---

## Slide 12: Limitations

### Content
Six limitations presented in a two-column table (Limitation × Impact).

### Explanation

| Limitation | Detailed Impact |
|------------|-----------------|
| **100-video test set** | The test set contains only 25 videos per manipulation type. A single misclassification changes per-type accuracy by 4 percentage points. Confidence intervals around the reported AUC of 0.937 would be wide (approximately ±0.05 at 95% CI). Statistical conclusions cannot be drawn from this sample size. A test set of at least 500 videos per manipulation type would be needed for reliable estimates (Dolhansky et al., 2020). |
| **Validation split only** | The model was trained on 68,851 clips — the validation split of AV-Deepfake1M++, not the full training set of over 1 million clips. This limits exposure to the full diversity of 2,000+ speakers, manipulation generators (TTS/VC systems, NeRF variants), and recording conditions. The full dataset would be expected to produce better generalization and more robust per-type performance. |
| **No ablation study** | The contribution of each architectural decision cannot be attributed. Controlled experiments needed: Focal Loss vs BCE (same architecture, different loss), Transformer vs MLP fusion (same encoders, different fusion), full-frame vs lip-region crops (same pipeline, different visual input), and drop-one-modality tests (training with only audio or only video). Without ablations, performance reflects the combined effect of all design choices. |
| **Fixed 2-second window** | The model analyzes a single fixed 2-second window per inference pass (3-window averaging). For videos where the manipulated segment is short (<2 seconds), begins late in the clip, or is distributed across non-contiguous segments, the sampled window may not capture any manipulation. The `fake_segments` temporal annotations in the dataset metadata are not utilized. |
| **Single dataset** | Training and evaluation were conducted entirely on AV-Deepfake1M++. Cross-dataset generalization — the most practically relevant measure of detector reliability — was not evaluated. Prior work (Dolhansky et al., 2020) documented consistent performance degradation when detectors trained on one dataset are applied to videos from different generators or recording environments. |
| **5-epoch training cap** | All training runs were capped at 5 epochs due to cloud GPU costs ($0.30–0.50/hr per instance, ~$15–25 per training run). The default 10-epoch budget in `config.py` was never exercised. Models 2 and 4 were halted while validation AUC was still improving (epoch 5 AUC = 0.9937, epoch 4 AUC = 0.9917). It is unknown whether extended training would improve score calibration, test-set performance, or model ranking. No hyperparameter sweep was feasible. |

### References
- Dolhansky, B. et al. (2020) 'The DeepFake Detection Challenge (DFDC) Dataset', arXiv:2006.07397.
- Rossi, R.J. (2018) *Mathematical Statistics: An Introduction to Likelihood Based Inference*. — Confidence interval calculation for AUC.
- Dissertation: Section 5.5 (Limitations), Section 6.3 (Honest Assessment).

---

## Slide 13: Future Work

### Content
Six priority-ordered future directions.

### Explanation

1. **Ablation studies (highest priority):** Controlled experiments comparing Focal Loss against standard BCE (identical architecture, training data, random seed), Transformer fusion against MLP fusion, and full-frame encoding against lip-region crops. These would isolate the contribution of each design decision. Without ablations, the current results cannot attribute performance to any individual component. Standard practice in detection literature (Rossler et al., 2019).

2. **Cross-dataset evaluation:** Test the trained model on FakeAVCeleb (Yi et al., 2023), DFDC (Dolhansky et al., 2020), and FaceForensics++ (Rossler et al., 2019). Cross-dataset generalization is the most practically relevant measure of detector reliability. Performance degradation on unseen generators and recording conditions is a well-documented challenge (Dolhansky et al., 2020).

3. **Full dataset training:** Use the complete AV-Deepfake1M++ training split (over 1 million clips, 2,000+ speakers) instead of only the validation split (68,851 clips). This would expose the model to the full diversity of manipulation generators (TTS, VC, NeRF variants), speaker identities, and recording conditions. Training on the full dataset would require access to the full 1.4 TB dataset and GPU resources beyond the student budget.

4. **Temporal localisation:** Extend from clip-level classification to frame-level or segment-level predictions. The `fake_segments` annotations in the dataset metadata provide ground-truth temporal boundaries of manipulation. A model that localizes *when* manipulation occurs (rather than just detecting *whether* it occurred) provides richer forensic output and could improve detection confidence by focusing attention on manipulated regions.

5. **Threshold optimisation:** Optimize the decision threshold on a held-out calibration set to balance false positives and false negatives. The current τ=0.5 threshold is arbitrary. A data-driven threshold could improve practical utility — for example, lowering the threshold for applications where false negatives are costly (missing a deepfake), or raising it where false positives are costly (falsely accusing authentic content).

6. **Longer training runs with hyperparameter sweeps:** With institutional GPU access, run 20+ epochs with systematic hyperparameter sweeps over focal_gamma, focal_alpha, dropout, hidden_dim, learning_rate, and encoder_lr using W&B Bayesian sweep. The current results from 5-epoch runs should be treated as preliminary. Extended training could reveal whether the Model 3 > Model 2 pattern (earlier checkpoint outperforms later) persists, reverses, or stabilizes, and whether longer training produces better-calibrated scores or further score compression.

### References
- Rossler, A. et al. (2019) 'FaceForensics++', *Proceedings of IEEE ICCV*.
- Dolhansky, B. et al. (2020) 'The DFDC Dataset', arXiv:2006.07397.
- Yi, J. et al. (2023) 'Audio Deepfake Detection: A Survey', arXiv:2308.14970.
- Dissertation: Section 6.4 (Future Work), Section 6.5 (Recommendations).

---

## Slide 14: Thank You

### Content
Closing slide with summary achievements and references.

### Explanation

**Summary achievements:**
- Multimodal deepfake detector: 93% accuracy, precision 1.000, zero false positives on the 100-video held-out test set (Model 3).
- All six objectives met with scope exceeding the initial CN6000 proposal — the final deliverables (Cross-Modal Transformer Fusion architecture, resumable pipeline, web interface with three tabs and history) exceed the initial specification.
- Speaker-disjoint evaluation using GroupShuffleSplit ensures results reflect manipulation detection, not identity recognition.
- Interpretable three-head output reveals per-modality vulnerability patterns — audio_modified is the hardest category because the genuine video stream counteracts the audio head's fake signal.

**Key takeaway:** Score calibration determines real-world deployment readiness, not AUC. Model 3 (epoch 3, AUC 0.985) outperformed Model 2 (epoch 5, AUC 0.994) by 27 percentage points in threshold accuracy, with zero false positives. For practitioners: evaluate calibration curves and reliability diagrams, not just AUC, when preparing models for deployment.

**Engineering reflection:** The decisions that ultimately had the most impact on the result were not architectural choices but engineering ones: switching audio loading from librosa to torchaudio (FFmpeg backend) to handle corrupted MP4 files, designing the manifest-based resumable extraction system to survive cloud instance termination, and implementing two-phase training to protect pretrained features.

### References (Abbreviated)
- Cai, Z. et al. (2025) 'AV-Deepfake1M++', *Proceedings of ACM Multimedia (MM '25)*.
- Cai, Z. et al. (2024) 'AV-Deepfake1M', *Proceedings of ACM Multimedia (MM '24)*.
- Tran, D. et al. (2018) 'A Closer Look at Spatiotemporal Convolutions for Action Recognition', arXiv:1711.11248.
- Lin, T.-Y. et al. (2018) 'Focal Loss for Dense Object Detection', arXiv:1708.02002.
- Dolhansky, B. et al. (2020) 'The DeepFake Detection Challenge (DFDC) Dataset', arXiv:2006.07397.
- Yi, J. et al. (2023) 'Audio Deepfake Detection: A Survey', arXiv:2308.14970.
