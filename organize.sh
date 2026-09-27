#!/bin/bash
# Organization Script for AV-Deepfake1M/Try

TARGET_DIR="/Users/jasmi/Desktop/AV-Deepfake1M/Try"
cd "$TARGET_DIR" || exit 1

echo "Creating folder structure..."
mkdir -p src scripts manuscript viva_presentation docs_admin notebooks data_and_results

echo "Organizing Core Source Code..."
mv audio.py video.py cross_modal.py config.py data_utils.py train_utils.py checkpoint_utils.py inference.py main.py src/ 2>/dev/null

echo "Organizing Scripts..."
mv analyze_data.py analyze_manifests.py compare_models.py evaluate_models.py create_test_data.py download_data.py cleanup.py plot_calibration_curves.py plot_mel_spectrogram.py plot_per_type_accuracy.py plot_training_history.py regenerate_comparison_plots.py regenerate_manifests.py replace_citations.py scripts/ 2>/dev/null

echo "Organizing Manuscript & Writing..."
mv draft.md draft.pdf draft.typ Deepfake_detection_using_cross-model_transformer_fusion.pdf literature_review.pdf methodology.pdf references.bib PipelineAnalysis.md calculations.md eda.md eda_values.md evaluation.md figures_and_unresolved.md runlogs.md Walkthrough.md manuscript/ 2>/dev/null

echo "Organizing Viva & Presentations..."
mv presentation.md presentation.pptx presentation.html presentation_notes.md presentation_notes.pdf "Presentation Template May Viva.pptx" Viva_presentation.pdf viva_questions.md viva_questions.html viva_questions.pdf viva_questions_files viva_gantt.png viva_gantt.jpeg build_presentation.py pptx_to_reveal.py viva_presentation/ 2>/dev/null

echo "Organizing Ethics & Administrative Docs..."
mv "CN6000 Internal Ethical Approval Process 2025.pdf" "Jasmi_EULA_signed_04:12:2025.pdf" "Participation Form for 1 Million Deepfakes Detection challenge at ACM Multimedia 2025_1.pdf" docs_admin/ 2>/dev/null

echo "Organizing Notebooks..."
mv Df_try.ipynb Mini_pipeline.ipynb notebooks/ 2>/dev/null

echo "Organizing Data & Results..."
mv val_metadata.json dummy.mp4 uel.svg data_and_results/ 2>/dev/null
[ -d comparison_results ] && mv comparison_results data_and_results/ 2>/dev/null
[ -d eval_results_m5 ] && mv eval_results_m5 data_and_results/ 2>/dev/null
[ -d figures ] && mv figures data_and_results/ 2>/dev/null

echo "Organizing Web Docs..."
mv WebInterface.md web_server.log web/ 2>/dev/null

echo "Done organizing!"
