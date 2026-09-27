# Unresolved Questions

The following questions remain unresolved and would require additional data collection, computation, or clarification to answer:

1. **What are the exact per-epoch AUC values for Model 2?** The checkpoint contains `ck['history']['val_auc_joint']` but these were not extracted and reported in the available documents. Running `torch.load('best_model.pth')` and printing the history is required.

2. **What are the complete 25 real video inference scores?** Only six scores (0.4297, 0.8895, 0.9756, 0.8587, 0.9115, 0.9719) were reported. The remaining 19 scores from `test/real/` need to be computed by running `inference.py` or `evaluate_models.py`.

3. **What are the complete 75 fake video inference scores?** The scores need to be computed from `test/fake/` to populate Table 4.3 with per-type statistics (mean, std dev, min, max, accuracy) for audio_modified, visual_modified, and both_modified categories.

4. **What are the overall test metrics (accuracy, precision, recall, F1) from evaluate_models.py?** These require running the evaluation script on the full 100-video test set and extracting values from the output.

5. **What is the scatter plot configuration?** The audio-vs-video score scatter plot needs to be generated from the evaluation output to determine whether the four manipulation types form distinct clusters.

6. **What is the score distribution histogram?** The bimodality of the joint score distribution needs to be confirmed by generating the histogram from evaluation output.

7. **What was the audio AUC and video AUC separately at each epoch?** These values in `ck['history']['val_auc_audio']` and `ck['history']['val_auc_video']` would show whether the two modalities contributed equally to the joint performance.

8. **What are the per-type AUC values from the test evaluation?** These would show which manipulation type is hardest to detect and whether both modalities contribute symmetrically.

9. **How does the model perform on cross-dataset evaluation?** The model has not been tested on FakeAVCeleb, DFDC, or FaceForensics++, so cross-dataset generalisation remains unknown.

10. **What is the threshold sensitivity analysis?** The 0.5 threshold was used without optimisation. Varying the threshold across the full [0, 1] range and plotting precision-recall curves would reveal whether a different operating point would improve practical performance.

11. **What would an ablation study show?** Without controlled experiments comparing Focal Loss vs BCE, Transformer vs MLP fusion, and full-frame vs lip-region encoding, the independent contribution of each design decision cannot be quantified.

12. **What is the full dataset training result?** Training on the complete AV-Deepfake1M++ training split (over one million clips) rather than the 68,851-video validation subset would likely improve model quality and is the most impactful single extension.

---

# List of Figures

The following figures are referenced in this dissertation and should be generated from the evaluation outputs:

1. **Figure 2.1: Multimodal deepfake detection pipeline overview.** A flowchart showing the complete pipeline from data download through to evaluation output, including metadata loading, speaker-based split, parallel feature extraction, model architecture, and metric reporting.

2. **Figure 3.1: Cross-Modal Transformer Fusion architecture.** A diagram showing the full model architecture: video frames (50×3×224×224) through ResNet3D-18 to 256-d, audio mel-spectrogram (1×128×63) through ResNet18 to 256-d, projection to 512-d, Transformer Encoder with [CLS] token, and three sigmoid classification heads.

3. **Figure 3.2: Two-phase training schedule.** A timeline showing Phase 1 (frozen encoders) covering the first 25% of epochs and Phase 2 (fine-tuning all parameters) covering the remaining 75%, with learning rate differences noted.

4. **Figure 4.1: Score distribution histogram.** A histogram of joint prediction scores on the 100-video test set showing the frequency of scores across the [0, 1] range. A bimodal distribution with peaks near 0 and near 1 indicates a confident, well-calibrated model.

5. **Figure 4.2: Audio vs video authenticity scatter plot.** A scatter plot with audio authenticity score on the x-axis and video authenticity score on the y-axis, with each point coloured by its manipulation type (real, audio_modified, visual_modified, both_modified). Ideally forming four distinct clusters at (1,1), (0,1), (1,0), and (0,0).

6. **Figure 4.3: Training loss and validation AUC curves.** A dual-axis plot showing training loss decreasing over epochs and validation joint AUC increasing, with the phase transition marked at the encoder unfreezing boundary.

7. **Figure 4.4: Per-type AUC bar chart.** A bar chart comparing AUC values across the four manipulation types (real, audio_modified, visual_modified, both_modified) for both the joint prediction and the individual audio and video heads.

8. **Figure 4.5: Confusion matrix for 100-video test set.** A normalised confusion matrix showing true positive, true negative, false positive, and false negative rates at the 0.5 decision threshold.

9. **Figure 4.6: ROC curve for joint prediction.** A receiver operating characteristic curve showing the true positive rate against the false positive rate across all thresholds, with AUC annotated.

10. **Figure 5.1: Speaker split verification diagram.** A Venn diagram or set representation showing zero overlap between training speakers, validation speakers, and test speakers as verified by `GroupShuffleSplit`.

11. **Figure 5.2: Per-window inference breakdown.** A table or heatmap showing the per-window scores (audio, video, joint) for each of the three windows extracted per test video, demonstrating the averaging mechanism.

12. **Figure 6.1: Web interface screenshot.** A screenshot of the DeepScan web interface showing the upload page, analysis results, and the score visualisation with audio, video, and joint authenticity bars.