What to write in Chapter 4: Results and Findings
Your two training runs produced the following which you should report:
Model 1 — epoch 1, AUC 0.663, all real videos scored below 0.35 and classified as FAKE. This is a failed/underfit model.
Model 2 — epoch 5, AUC 0.994, real video scores 0.43–0.97, mostly correctly classified.
Structure Chapter 4 as follows:
4.1 Training Results
Report both runs side by side. Include the AUC history per epoch from the checkpoint (ck['history']['val_auc_joint']). Note that Model 1's file was corrupted and could not be loaded for inference. Report the best AUC, the epoch it was achieved, and the training loss trajectory for Model 2.
4.2 Inference on Real Videos (Model 2)
Report the scores you observed: 0.4297, 0.8895, 0.9756, 0.8587, 0.9115, 0.9719 across the first six real videos. Calculate and report the accuracy across all 25 real videos once inference completes. Note the one borderline case (0.43) and interpret it — the model is uncertain on that clip, not systematically wrong.
4.3 Inference on Fake Videos (Model 2)
Once you run inference on test/fake/, report the scores. For a well-performing model (AUC 0.994) you should expect fake video scores consistently below 0.4. Report how many were correctly classified as FAKE and compute accuracy.
4.4 Overall Metrics
Report accuracy, AUC, precision, recall, and F1 from evaluate_models.py once you run it. Use the table format. Report per-type breakdown — how well the model detected audio_modified, visual_modified, and both_modified separately.
4.5 Score Distribution
Describe the histogram from evaluate_models.py. A bimodal distribution (peaks near 0 and near 1) indicates a confident, well-calibrated model. Include the plot as a figure.
4.6 Model 1 vs Model 2 Comparison
Even though Model 1 is corrupted, report what was known from the checkpoint — epoch 1, AUC 0.663, all predictions FAKE — and compare against Model 2. This illustrates the importance of sufficient training epochs and motivates the conclusion that further training would likely improve performance further.
One important note — because you only have 100 test videos (25 real, 75 fake) the numbers will have high variance. Be honest about this in the text: "The test set of 100 videos, while sufficient for indicative results, is too small to draw statistically robust conclusions. A larger evaluation set would be needed to confirm these findings." This is exactly what Chapter 5 Evaluation is for.