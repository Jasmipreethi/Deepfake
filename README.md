<h1 align="center">🎭 AV-Deepfake1M++ Detection</h1>

<p align="center">
  <strong>Multimodal Audio-Video Deepfake Detection using Cross-Modal Transformer Fusion</strong>
</p>

<p align="center">
  <a href="https://huggingface.co/datasets/ControlNet/AV-Deepfake1M-PlusPlus">
    <img src="https://img.shields.io/badge/Dataset-HuggingFace-yellow?style=flat-square&logo=huggingface" alt="Dataset">
  </a>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Accelerate-AMP%20%2F%20Compile-blue?style=flat-square" alt="PyTorch AMP">
  <img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey?style=flat-square" alt="License">
</p>

---

## Overview & Publication Framework

This repository presents an empirical evaluation framework and PyTorch 2.0+ pipeline designed to benchmark audio-visual deepfake detection on the **AV-Deepfake1M++** dataset.

The system features a **Cross-Modal Transformer Fusion architecture**—integrating ResNet3D-18 video encoder and ResNet18 audio encoder with a two-layer multi-head attention fusion module. Trained under a speaker-disjoint partition using Focal Loss, the architecture yields multi-head predictions for audio authenticity, video authenticity, and joint verdict.

### Key Capabilities & PyTorch 2.0+ Features:
* **Native PyTorch 2.0+ Optimizations:** Automatic Mixed Precision (`bfloat16`/`float16`), `torch.compile()` graph optimization, and GPU-accelerated `torchaudio.transforms.MelSpectrogram`.
* **Model Calibration Suite:** Temperature scaling (`scripts/evaluate_calibration.py`) and Expected Calibration Error (ECE) calculation.
* **Cross-Dataset Zero-Shot Generalization:** Benchmark evaluator (`scripts/evaluate_cross_dataset.py`) for FakeAVCeleb, DFDC, and FaceForensics++.
* **Perturbation Robustness Suite:** Noise injection and H.264 video compression sweep (`scripts/test_robustness.py`).
* **Systemic Ablation Suite:** Automated comparison (`scripts/run_ablations.py`) across fusion types, loss functions, and modality streams.

---

## Directory Organization

```
AV-Deepfake1M/Try/
├── src/                        # PyTorch Source Code & Neural Architectures
│   ├── main.py                 # Training pipeline entry point
│   ├── cross_modal.py          # Transformer & Temperature Scaling Fusion
│   ├── train_utils.py          # Focal Loss, AMP & Optimization
│   ├── config.py               # Hyperparameter & Path config
│   ├── data_utils.py           # Datasets & Speaker-Disjoint Splitting
│   ├── checkpoint_utils.py     # Checkpoint save/load & recovery
│   ├── inference.py            # Standalone prediction pipeline
│   ├── audio.py                # Audio extraction & spectrograms
│   └── video.py                # Video frame extraction
│
├── scripts/                    # Research Evaluation & Analysis Suite
│   ├── evaluate_calibration.py # Temperature scaling & ECE metrics
│   ├── evaluate_cross_dataset.py # Zero-shot OOD evaluation
│   ├── test_robustness.py      # Noise & Compression perturbation sweep
│   ├── run_ablations.py        # Systemic ablation runner
│   ├── compare_models.py       # Multi-model evaluation tool
│   ├── evaluate_models.py      # Benchmark test set evaluator
│   ├── plot_calibration_curves.py
│   ├── plot_mel_spectrogram.py
│   ├── plot_per_type_accuracy.py
│   └── plot_training_history.py
│
├── manuscript/                 # Dissertation Drafts & Journal Paper Materials
│   ├── draft.md                # Full dissertation manuscript
│   ├── draft.pdf               # Rendered PDF paper draft
│   ├── Deepfake_detection_using_cross-model_transformer_fusion.pdf
│   └── references.bib          # BibTeX citations
│
├── viva_presentation/          # Slide Decks, Viva Q&A & Defense Assets
│   ├── presentation.pptx / .md / .html
│   └── viva_questions.md / .pdf
│
├── docs_admin/                 # Ethics Approval & Official Forms
├── notebooks/                  # Jupyter Notebook Prototypes
├── data_and_results/           # Output Figures, Results & Metadata
└── web/                        # Web Dashboard Interface
```

---

## Quick Start & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training Pipeline (PyTorch 2.0+)
```bash
python src/main.py --fusion transformer --epochs 5 --batch_size 32
```

### 3. Run Calibration & Evaluation Suite
```bash
# Model Calibration & Temperature Scaling
python scripts/evaluate_calibration.py

# Cross-Dataset Zero-Shot Benchmarking
python scripts/evaluate_cross_dataset.py --checkpoint checkpoints/model_3.pt

# Real-World Compression & Perturbation Testing
python scripts/test_robustness.py

# Automated Ablation Studies
python scripts/run_ablations.py
```

### 4. Run Standalone Inference
```bash
python src/inference.py --video sample.mp4 --checkpoint checkpoints/model_3.pt
```

---

## License & Citation

This codebase is licensed under **CC BY-NC 4.0**.
