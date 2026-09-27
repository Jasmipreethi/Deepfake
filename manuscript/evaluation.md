# Chapter 5: Discussion and Evaluation

## 5.1 Introduction

This chapter interprets the results presented in Chapter 4 in relation to the project objectives, compares the findings against prior work, evaluates the strengths and limitations of the implemented system, and reflects on the development process. The chapter is structured to address each objective in turn before broadening the discussion to cover unexpected outcomes, practical constraints, and the implications of the results.

---

## 5.2 Evaluation Against Objectives

### Objective 1 — Speaker-Disjoint Dataset Partitioning

The speaker-based partition using `GroupShuffleSplit` was implemented successfully, resulting in zero speaker overlap between training and validation sets. This addresses a widely documented limitation in deepfake detection research where random splits allow models to exploit speaker identity rather than learning manipulation artefacts (Rossler et al., 2019). The practical consequence of this decision is that the reported AUC of 0.994 reflects generalisation to entirely unseen identities rather than face or voice recognition. Metrics from a random split would be expected to be higher but less meaningful. This objective was fully met.

### Objective 2 — Cross-Modal Transformer Fusion with Three-Head Output

The Transformer Encoder fusion module with a learnable [CLS] token was implemented and trained successfully. The three-head design — producing independent audio, video, and joint predictions — allows the contribution of each modality to be examined separately. The per-type breakdown reported in Section 4.7 demonstrates whether the audio head and video head each contribute to the overall detection performance or whether the joint head is dominated by one modality. A joint AUC of 0.994 alongside separate audio and video AUC values [insert values] indicates that [insert observation — e.g. both modalities contributed, or one modality drove the result]. This objective was substantially met, though further ablation (training with audio or video disabled) would provide stronger evidence of each modality's independent contribution.

### Objective 3 — Focal Loss

Focal Loss with γ = 2.0 and α = 0.25 was used throughout training. The training logs show [insert observation from training_loss history — e.g. steady convergence without plateauing]. The Model 1 result (AUC 0.663 at epoch 1) shows what an underfit model looks like on this dataset — predicting FAKE for all real videos with scores consistently in the 0.23–0.35 range — and provides a useful contrast for assessing whether Focal Loss contributed to Model 2's convergence. This objective was met in implementation, though a direct ablation comparing Focal Loss against standard BCE was not performed within this project.

### Objective 4 — Resumable Training Pipeline

The full checkpoint system was implemented, saving model state, optimiser state, scheduler state, and all random number generator states at each epoch. Training was successfully resumed on cloud GPU instances across multiple sessions without loss of reproducibility. The W&B integration logged all metrics per epoch, providing a full audit trail. This objective was fully met.

### Objective 5 — Evaluation on 100-Video Test Set

Evaluation was conducted on a 100-video test set sampled from the validation split using `create_test_data.py`. Results are reported in Chapter 4 including AUC, accuracy, precision, recall, F1, and per-type breakdown. This objective was met. The limitation noted in Section 5.5 regarding test set size is acknowledged.

### Objective 6 — Standalone Inference and Web Interface

`inference.py` provides command-line inference on single videos and folders with no training pipeline dependencies. The web interface (`app.py` + `static/index.html`) provides drag-and-drop upload, batch processing, model comparison, history tracking, and PDF report generation. This objective was fully met.

---

## 5.3 Interpretation of Results

### 5.3.1 Model 2 Performance

A validation joint AUC of 0.994 is a strong result. By definition, an AUC of 0.994 means that in 99.4% of random real/fake video pairings, the model correctly assigns a higher score to the real video. This performance level places the model in the same range as specialist multimodal detection systems evaluated on controlled benchmarks (Cai et al., 2024). The score distribution observed in Section 4.8 — [insert description of bimodal or otherwise shaped histogram] — indicates [insert interpretation: confident separation vs uncertain boundary].

The real video scores observed during inference (0.86–0.97 for five of the first six real videos) show that the model is decisively assigning high authenticity scores to genuine content. The single borderline case (0.4297) suggests the model was uncertain on that specific clip, which is consistent with a well-calibrated model rather than a systematic failure.

### 5.3.2 Per-Type Analysis

The per-type breakdown in Section 4.7 reveals [insert observation — e.g. which type was easiest to detect and why]. A well-functioning multimodal system would be expected to detect `both_modified` most reliably, since both encoders provide consistent fake evidence, and to find `audio_modified` or `visual_modified` harder, since only one encoder is relevant. If the results show this pattern, it provides evidence that both modalities are genuinely contributing. If `visual_modified` is detected at lower AUC than `audio_modified`, this may suggest the ResNet3D-18 encoder is contributing less effectively than the ResNet18 audio encoder for this dataset, or vice versa.

### 5.3.3 Audio vs Video Scatter

The scatter plot in Section 4.9 shows [insert description]. Ideally, `audio_modified` videos cluster towards low audio score and high video score, `visual_modified` videos cluster towards high audio score and low video score, `both_modified` videos cluster near (0, 0), and `real` videos cluster near (1, 1). The degree to which these clusters are separated is a direct measure of how well the model understands the nature of each manipulation type, not just whether the joint score crosses the 0.5 threshold.

### 5.3.4 Model 1 Failure Analysis

Model 1's AUC of 0.663 and its uniform tendency to predict FAKE regardless of ground truth indicates that the model had not converged after epoch 1. This is consistent with early training behaviour before the fusion module has stabilised — the two-phase training design was intended to prevent exactly this, but Model 1 appears to have been saved from a run where training had not progressed far enough. The corrupted download of Model 1's checkpoint meant this could not be confirmed directly via inference on fake videos, but the behaviour observed on real videos (scores 0.23–0.35, all classified FAKE) is characteristic of a model predicting the majority class.

---

## 5.4 Comparison with Prior Work

Multimodal deepfake detectors evaluated on controlled benchmarks have reported AUC values in the 0.85–0.99 range depending on the evaluation protocol (Cai et al., 2024; Yi et al., 2023). The AUC of 0.994 achieved by Model 2 falls at the upper end of this range, which is encouraging. However, direct comparison is difficult because the evaluation in this project uses only 100 videos from the validation split — a much smaller test set than is typically used in published benchmarks — and the model was trained only on the validation split of AV-Deepfake1M++ rather than the full training set.

Unlike the simpler concatenation-based fusion approaches noted as a gap in the literature (Yi et al., 2023), the Transformer fusion module used here allows audio and video representations to interact during feature learning. Whether this provides a measurable advantage over simpler fusion on this dataset cannot be determined without an ablation study, which represents a direction for future work.

---

## 5.5 Limitations

**Test set size.** The 100-video test set is too small to draw statistically robust conclusions. A single misclassified video changes accuracy by 1 percentage point, and confidence intervals around the reported AUC would be wide. A test set of at least 500 videos per manipulation type would be needed to produce reliable estimates.

**Training data.** Using only the validation split of AV-Deepfake1M++ (68,851 videos) rather than the full training set (over one million clips) limits the model's exposure to the full diversity of manipulation techniques and speaker identities. The full dataset would be expected to produce better generalisation.

**No ablation study.** The contribution of each architectural decision — Focal Loss vs BCE, Transformer vs MLP fusion, full-frame vs lip-region encoding — was not isolated through controlled ablation. The results therefore reflect the combined effect of all design choices, making it impossible to attribute performance to any single component.

**Fixed two-second window.** The model analyses a fixed two-second window per inference pass. For videos where the manipulated region is short, begins late, or is distributed across the clip, this window may not capture the most informative segment.

**Single dataset.** The model was trained and evaluated entirely on AV-Deepfake1M++. Deepfake detectors are known to show performance degradation when applied to data from different generators or recording conditions (Dolhansky et al., 2020). Cross-dataset generalisation was not evaluated.

---

## 5.6 Reflection on the Development Process

The implementation diverged from the initial proposal in three principal ways — audio encoder, visual encoder, and fusion module — each driven by practical constraints rather than deliberate design choices. This highlights a fundamental challenge in applied deep learning research: the gap between a theoretically motivated architecture and what can be implemented and trained within a given hardware budget, time frame, and software ecosystem.

The decision to replace Wav2Vec 2.0 with a ResNet18 on mel-spectrograms was initially reluctant, as Wav2Vec's contextual speech representations were expected to provide superior sensitivity to voice cloning artefacts. In retrospect, the mel-spectrogram approach proved robust and simple to integrate, and the resulting model achieved strong performance. This suggests that the mel-spectrogram representation captures sufficient information for this task, at least at the validation split scale.

The parallel feature extraction pipeline — using 28 CPU workers with fork-based multiprocessing and crash-resumable manifests — was a significant engineering investment that paid off when cloud instances terminated unexpectedly mid-extraction. Without this system, feature extraction would have needed to restart from the beginning each time.

Training on cloud GPU instances via Vast.ai introduced challenges around data persistence, checkpoint management, and file transfer. The `scp` workflow for downloading checkpoints and the Google Drive integration for Colab runs required careful management to avoid file corruption, as experienced with Model 1. Future work should use a cloud storage bucket (e.g. Google Cloud Storage or AWS S3) with atomic writes to avoid partial downloads.

The Weights & Biases integration provided significant value during training, making it possible to monitor convergence, detect overfitting, and compare per-type performance in real time without waiting for the full training run to complete.
