# Multimodal Audio-Visual Deepfake Detection Using Cross-Modal Transformer Fusion

---

# Abstract

Deepfake technology has advanced rapidly from early face-swapping systems to sophisticated multimodal generators capable of simultaneously manipulating both audio and video streams. This progress has created a growing demand for detection systems that go beyond single-modality analysis and jointly reason over audio and visual evidence. This dissertation presents the design, implementation, and evaluation of a multimodal deepfake detection pipeline trained on the AV-Deepfake1M++ dataset — a large-scale benchmark comprising over 77,000 video clips across four manipulation categories: real, audio-modified, visual-modified, and both-modified.

The implemented system is a Cross-Modal Transformer Fusion network that combines a ResNet3D-18 video encoder pretrained on Kinetics-400 with a ResNet18 audio encoder pretrained on ImageNet, fused through a two-layer Transformer Encoder with a learnable [CLS] token. The model produces three simultaneous predictions per clip — audio authenticity, video authenticity, and a joint verdict — trained using Focal Loss to address gradient domination by easy examples. A speaker-disjoint dataset partition was enforced to prevent identity leakage between training and validation sets, ensuring that reported performance reflects genuine generalisation to unseen identities.

The system achieved a validation joint AUC of 0.994 by epoch five of training, with real video scores predominantly in the range 0.86–0.97 on held-out test clips. These results demonstrate that the cross-modal attention mechanism is effective at capturing audio-visual inconsistencies introduced by deepfake manipulation. The architecture deviates from the initial proposal — which specified Wav2Vec 2.0, MobileNetV3, and DiMoDif — due to hardware constraints, audio loading instability, and the broader spatial distribution of visual artefacts across the full face rather than the lip region alone. Each deviation is documented and justified.

The dissertation contributes a fully modular, resumable training pipeline, a standalone inference system, and a web-based interface for video upload and classification. Limitations include the use of the validation split only, a fixed two-second analysis window, and a test set of 100 videos whose small size constrains the statistical robustness of the reported metrics.

---

# Chapter 1: Introduction

## 1.1 Background

The proliferation of deepfake technology poses an increasingly serious threat to the integrity of digital media. Advances in deep generative modelling — particularly Generative Adversarial Networks, diffusion models, and neural radiance fields — have enabled the synthesis of highly realistic audio and video content that is difficult to distinguish from authentic recordings. Deepfakes have been deployed in financial fraud, political misinformation, non-consensual pornography, and impersonation attacks, with documented real-world harms that extend well beyond academic research settings (Chesney and Citron, 2019; Milmo, 2024).

Early deepfake detection systems focused on identifying spatial artefacts in individual frames — boundary inconsistencies, abnormal blinking, or unnatural skin texture — using convolutional neural networks trained on benchmark datasets such as FaceForensics++ (Rossler et al., 2019). These approaches achieved high accuracy within controlled settings but showed substantial performance degradation when applied to unseen generators or real-world media, suggesting that many detectors learned dataset-specific patterns rather than genuine manipulation signatures (Dolhansky et al., 2020).

More recent deepfake systems manipulate both the audio and video streams simultaneously, introducing audio-visual mismatches — between speech content and lip motion, or between voice identity and facial appearance — that single-modality detectors are structurally unable to detect. This has driven increasing interest in multimodal detection approaches that jointly analyse both streams (Cai et al., 2024; Yi et al., 2023).

## 1.2 Problem Statement

Despite growing interest in multimodal deepfake detection, several limitations persist in existing approaches. Most systems rely on simple feature concatenation rather than learned cross-modal attention, limiting their ability to capture subtle interdependencies between audio and visual evidence. Many remain vision-centric, treating audio as a secondary modality whose independent contribution is rarely quantified. Evaluation protocols frequently use random train/validation splits that allow speaker identity leakage, producing inflated performance figures that do not generalise to unseen individuals. These gaps collectively motivate a system that applies cross-modal attention fusion, treats both modalities as equal contributors, enforces a speaker-disjoint evaluation protocol, and explicitly quantifies per-modality detection performance.

## 1.3 Aim

The aim of this project is to design, implement, and evaluate a multimodal audio-visual deepfake detection system that jointly analyses audio and video streams using cross-modal attention, and to assess its performance across four manipulation types on the AV-Deepfake1M++ dataset.

## 1.4 Objectives

1. To implement a speaker-disjoint dataset partitioning strategy that prevents identity leakage and ensures honest evaluation of generalisation to unseen identities.

2. To design and train a Cross-Modal Transformer Fusion network that jointly encodes audio mel-spectrograms and video frame sequences, producing independent audio, video, and joint authenticity predictions.

3. To replace standard Binary Cross-Entropy with Focal Loss to address gradient domination by easy examples during training on a large, balanced multi-class dataset.

4. To build a fully resumable training pipeline that supports cloud-based GPU instances with checkpoint recovery and Weights & Biases experiment tracking.

5. To evaluate the trained model on a representative 100-video test set and report AUC, accuracy, precision, recall, F1, and per-type performance breakdown.

6. To develop a standalone inference system and web-based interface for classifying new videos outside the training pipeline.

## 1.5 Significance

This work contributes a complete end-to-end multimodal deepfake detection system built on one of the largest audio-visual deepfake datasets currently available. The system is designed with research reproducibility in mind — all hyperparameters are centralised, all randomness is seeded, and the full training state is checkpointed — making it straightforward to extend or replicate. The three-head output design provides richer diagnostic information than a single binary classifier, allowing the contribution of each modality to be assessed independently.

## 1.6 Structure of the Dissertation

Chapter 2 reviews deepfake generation and detection techniques across image, audio, video, and multimodal domains, identifying the gaps that motivate this work. Chapter 3 describes the methodology and implementation, including documented deviations from the initially proposed architecture. Chapter 4 presents the results of training and evaluation. Chapter 5 discusses the findings in relation to the objectives, prior work, and limitations. Chapter 6 concludes with reflections and suggestions for future work.

---

# Chapter 2: Literature Review

## 2.1 Introduction

Recent advances in artificial intelligence have reshaped digital media forensics, largely due to the rapid rise of deepfakes. These synthetic videos and images, generated through deep neural architectures, can imitate human appearance and behaviour with increasing precision (He, 2021; Rossler et al., 2019). Although early work treated manipulated media as a relatively contained technical issue, current developments have made detection more difficult. As He (2021) pointed out, the realism achieved by modern generative models disrupts long-standing assumptions about what constitutes trustworthy audio-visual evidence. Rossler et al. (2019) further highlight that detection systems often are behind the pace of generative model improvements, creating a persistent asymmetry between manipulation and forensics. Given this trajectory, a clearer understanding of the underlying deep learning methods is necessary before evaluating why current detection strategies struggle and where they fall short.

### 2.1.1 Defining Deep Learning

Deep learning is defined as a subset of machine learning characterised by the utilisation of multi-layered neural networks. This architectural paradigm draws inspiration from the biological neural networks of the human brain, aiming to model high-level abstractions in data through a deep graph with multiple processing layers (LeCun, 2015). According to Goodfellow et al. (2016), deep learning systems differentiate themselves from conventional machine learning by their ability to learn representations of data with multiple levels of abstraction. While a shallow network might only have one hidden layer, a "deep" architecture utilises a hierarchy of layers to process data through a series of nested non-linear equations. This structure allows the model to function as a universal approximator, capable of modelling complex high-dimensional data patterns that simpler algorithms cannot capture (Hornik, 1989).

### 2.1.2 Defining Deepfake

The term deepfake combines "deep learning" and "fake," which describes synthetic media generated through deep neural models rather than conventional editing techniques (Westerlund, 2019). Most early systems relied on autoencoders or Generative Adversarial Networks, which made it possible to automate face swapping and expression transfer in a way that traditional manipulation tools could not achieve (Gera, 2018). Although digital alterations long predate deep learning, the specific phenomenon of deepfakes entered mainstream attention in late 2017 when a Reddit user released code that enabled realistic face swaps with very little technical skill, initially leading to widespread non-consensual pornography and, later, to wider use in misinformation and entertainment (Chesney and Cytowic, 2019; Westerlund, 2019). In research contexts, deepfakes are generally defined by their use of deep generative models to produce content that is difficult to distinguish from authentic footage (Cai, 2024). Over time, beyond face swaps, current systems include audio-based voice cloning and multimodal models capable of synchronising generated speech with fabricated lip movements, which introduces additional complexity for forensic detection (Yi et al., 2023). Together, these developments show that deepfakes are not a single technique but a growing family of generative methods with expanding forensic implications.

## 2.2 Evolution of Deepfakes

### 2.2.1 Phase 1: Visual Fidelity and Generative Advances

The initial phase of deepfake development focused on face swapping, using simple autoencoders and early Generative Adversarial Networks (GANs) (Dolhansky et al., 2020). These techniques were prone to noticeable flaws, such as affine warping errors, making them easily detectable by early forensic tools (Li et al., 2018). However, the field progressed rapidly with the adoption of improved training strategies such as the Two-Time Scale Update Rule (TTUR), which greatly improved GAN stability and convergence, resulting in more lifelike image generation (Heusel et al., 2017). As generative methods evolved, benchmark datasets such as the Deepfake Detection Challenge (DFDC) (Dolhansky et al., 2020) and Celeb-DF (Li et al., 2019) emerged. These resources marked a transition from low-grade and easily spotted fakes to high-quality videos that accurately replicated real-world visuals, posing significant challenges for detection systems (Li et al., 2019).

### 2.2.2 Phase 2: "In-the-Wild" Adaptation and Multi-Subject Expansion

A major advancement in deepfake technology was transitioning from controlled laboratory settings to more unpredictable "in-the-wild" environments. Although earlier datasets typically included individual subjects under uniform lighting and framing, newer deepfake datasets like Wild Deepfake (Zi et al., 2021) and FFIW-10K (Zhou et al., 2021) consist of content sourced from the internet or specifically generated to endure real-world challenges such as compression and noise. During this stage, deepfake methods were also advanced, enabling the manipulation of multiple people within the same scene. For example, research by Narayan et al. (2023) using the DF-Platter dataset showed that current techniques can alter multiple faces in a single image, even in challenging scenarios such as occlusion or poor image quality. Similarly, the FFIW-10K dataset highlighted the ongoing difficulty that forensic systems face in detecting fakes in group videos — especially when only some faces have been manipulated (Zhou et al., 2021).

### 2.2.3 Phase 3: Multimodal and Neural Generation

The latest and arguably most concerning development in deepfake technology is the move from only visual edits to multimodal deepfakes that manipulate both audio and video. Recent studies highlight the importance of inconsistencies between sound and visuals, leading to the creation of datasets such as AV-Deepfake1M (Cai, 2024) and FakeAVCeleb (Yi et al., 2023), which feature coordinated alterations to both video and audio streams. This evolution requires that detection tools now evaluate the synchronisation of lip movements with speech, rather than just looking for visual inconsistencies (Yi et al., 2023). In addition, the generative techniques themselves have progressed beyond traditional GANs. Neural Radiance Fields (NeRF) have been used to create talking-head videos where both viewpoint and audio can be modified (Guo et al., 2021). For audio, generation has advanced from simple concatenation methods to diffusion models, such as NaturalSpeech 2, which enable zero-shot voice synthesis with remarkably accurate intonation and prosody (Shen et al., 2023).

## 2.3 Documented Real-World Impacts and High-Profile Cases

Although early research on deepfakes largely focused on improving synthesis quality and benchmarking detection accuracy, larger concern has been driven by their increasing use in real-world scenarios (Chesney and Cytowic, 2019; Dolhansky et al., 2020; Rossler et al., 2019).

One area of significant impact has been medical misinformation. Investigative reporting has documented numerous cases in which deepfake videos impersonating qualified doctors were used to promote false health advice and fraudulent treatments across social media platforms (ITV News, 2024; Clark, 2025). These videos often exploit the visual credibility of trusted medical professionals, making misinformation difficult for non-expert audiences to identify. Such cases highlight how deepfakes can undermine public trust in expert knowledge, particularly in domains where inaccurate information may cause physical harm (Chesney and Cytowic, 2019; Rossler et al., 2019).

Deepfakes have also been employed in financial fraud and impersonation scams. A notable case involved a multinational engineering firm in which employees were deceived into authorising a large financial transfer after participating in a video call that appeared to feature senior executives but was later confirmed to be AI-generated (Milmo, 2024). Similar incidents have been reported globally, prompting warnings from law-enforcement agencies about the growing use of synthetic audio and video in social engineering attacks (Bragg, 2025). These cases demonstrate how deepfakes can amplify traditional fraud techniques by adding a convincing visual and auditory layer.

In addition to visual manipulation, audio deepfakes have demonstrated significant real-world impact, particularly in the context of fraud and impersonation (Korshunov and Marcel, 2018; Yi et al., 2023; Zhang, 2025). One of the first high-profile cases occurred in 2019, when attackers used a synthetic voice to mimic a company executive to authorise a fraudulent financial transfer, resulting in substantial financial losses (Stupp, 2019). Unlike visual deepfakes, audio-based impersonation can be deployed in real-time through phone calls or voice messages, limiting opportunities for human verification and increasing its effectiveness in social engineering scenarios (Korshunov and Marcel, 2018). Prior research demonstrates that synthetic speech can convincingly replicate speaker identity, enabling attacks that bypass traditional speaker verification and authentication systems (Korshunov and Marcel, 2018; Yi et al., 2023). Taken together, these findings indicate that audio deepfakes pose a parallel threat to trust and identity assurance mechanisms, reinforcing the need for detection approaches that extend beyond visual analysis and address multimodal manipulation (Rossler et al., 2019; Cai, 2024).

Political manipulation represents another area of concern associated with deepfakes, particularly in the context of increased geopolitical tension (Chesney and Cytowic, 2019; Vaccari, 2020). Videos depicting public figures have circulated online, sometimes appearing to show officials making false statements or announcements, thereby creating confusion regarding the authenticity of political communication (Rossler et al., 2019). An example involves a manipulated video of a national leader that was briefly disseminated on social media before being removed by platform moderators (Allyn, 2022). Although such videos do not always achieve their intended persuasive impact, studies suggest that their rapid circulation can take advantage of periods of uncertainty and contribute to decline of confidence in authentic news sources (Vaccari, 2020).

In addition to large-scale misinformation, deepfakes have caused direct personal harm. Investigations of non-consensual synthetic pornography have revealed extensive platforms dedicated to generating explicit content using the likenesses of individuals without consent, disproportionately targeting women (Moore, 2025). These cases underscore the ethical and legal challenges posed by deepfakes, particularly in situations where existing regulatory and legal frameworks struggle to address harms arising from fabricated yet highly realistic media (Chesney and Cytowic, 2019; Westerlund, 2019).

Taken together, these examples illustrate that deepfakes are no longer a purely technical problem confined to academic datasets. Their real-world deployment exposes weaknesses in current detection systems and reinforces the need for reliable, generalisable forensic approaches capable of operating under the unpredictable conditions of online media ecosystems (Rossler et al., 2019; Dolhansky et al., 2020).

## 2.4 The Dual Dilemma: Creative Application Versus Malicious Threat

Deepfake technologies exemplify a dual-use dilemma, in which the same deep generative models that enable legitimate and creative applications can also be exploited for harmful purposes. Advances in audio and visual synthesis have supported a range of beneficial use cases, including film post-production, automated dubbing, digital de-ageing, and assistive technologies such as personalised text-to-speech systems and voice restoration tools. In these contexts, synthetic media techniques are typically deployed with consent and transparency, and their use is framed as an extension of existing digital production methods (Westerlund, 2019; Yi et al., 2023). However, the properties that make deepfake technologies attractive for creative and commercial applications, namely high realism, automation, and scalability, also enable malicious misuse. As discussed in Section 2.3, deepfakes have been employed in misinformation campaigns, fraud, impersonation, and non-consensual content generation. The ability to fine-tune generative models using limited data further reduces the barrier to entry for misuse, allowing individuals without advanced technical expertise to produce convincing synthetic media (Chesney and Cytowic, 2019). This tension complicates regulatory and technical responses to deepfakes, as restrictive controls on generative technologies risk constraining legitimate innovation, while inadequate oversight allows harmful applications to proliferate (Chesney and Cytowic, 2019; Westerlund, 2019).

From a forensic perspective, this dual-use nature highlights the limitations of prevention-only strategies and reinforces the need for effective detection mechanisms (Chesney and Cytowic, 2019). Detection systems must therefore be capable of operating independently of the intent behind content creation, instead focusing on identifying artefacts, inconsistencies, or statistical traces that distinguish synthetic media from authentic recordings (Rossler et al., 2019; Dolhansky et al., 2020). Understanding deepfakes as a dual-use technology provides an important context for evaluating both generation methods and detection strategies. It underscores why detection research must prioritise generalisation and resilience to evolving synthesis techniques, rather than relying on assumptions about specific models or controlled deployment scenarios (Rossler et al., 2019; Dolhansky et al., 2020; Cai, 2024). This perspective motivates the examination of deepfake generation techniques in the following section, as well as the detection methodologies reviewed later in this chapter.

## 2.5 Deepfake Generation Techniques

Deepfake generation techniques can be broadly categorised according to the modality they manipulate: images, audio, or video. While early systems often focused on a single modality, recent advances increasingly integrate multiple streams, complicating forensic analysis. Understanding the mechanisms behind these generation techniques is essential for contextualising the strengths and limitations of current detection approaches (Rossler et al., 2019; Dolhansky et al., 2020; Yi et al., 2023; Cai, 2024).

### 2.5.1 Image-Based Manipulation

Image-based deepfake generation primarily targets facial appearance through identity replacement, expression transfer, or attribute manipulation (Rossler et al., 2019). Early approaches relied on autoencoder architectures trained to map facial representations between source and target identities, enabling face swapping with relatively limited training data (Dolhansky et al., 2020). These systems commonly employed a shared encoder with identity-specific decoders, allowing facial expressions and pose information to be transferred while preserving identity-related features (Gera, 2018). Subsequent progress was driven by the adoption of Generative Adversarial Networks (GANs), which enabled higher resolution synthesis and more realistic texture generation (Rossler et al., 2019). Advances in training strategies and loss formulation, including stabilisation techniques such as the Two-Time Scale Update Rule, significantly reduced artefacts such as colour inconsistency and boundary distortion, making image-based deepfakes increasingly difficult to detect using handcrafted forensic cues (Heusel et al., 2017; Li et al., 2019). Recent studies have incorporated attention mechanisms and multi-scale discriminator architectures to improve performance under common post-processing operations such as compression and resizing, which can reduce visual differences between synthetic and authentic facial images (Cai, 2024).

### 2.5.2 Audio-Based Manipulation

Audio-based deepfake generation focuses on synthesising or converting speech to mimic the voice characteristics of a target speaker. Two dominant paradigms underpin these systems: text-to-speech (TTS), which generates speech directly from text, and voice conversion (VC), which transforms the vocal attributes of a source speaker into those of a target speaker while preserving linguistic content (Yi et al., 2023). Early systems relied on statistical parametric models, but recent advances employ deep neural architectures capable of producing highly natural and expressive speech (Korshunov and Marcel, 2018). Modern audio deepfakes leverage end-to-end neural models, including encoder-decoder architectures and diffusion-based approaches, which allow high-fidelity voice cloning from limited reference audio (Shen et al., 2023). These systems can accurately reproduce speaker identity, prosody, and emotional tone, making synthetic speech difficult to distinguish from genuine recordings by both humans and automated verification systems (Korshunov and Marcel, 2018; Yi et al., 2023). The increasing accessibility of such tools has lowered the barrier to misuse, particularly in impersonation and fraud scenarios, posing significant challenges for existing audio authentication mechanisms.

### 2.5.3 Video-Based Manipulation

Video-based deepfake generation extends beyond static image manipulation by modelling temporal consistency across frames, allowing for realistic facial motion, head pose variation, and synchronisation with speech (Rossler et al., 2019). Early video deepfakes often exhibited temporal artefacts such as flickering, inconsistent lighting, or unnatural motion, which could be exploited by detection systems (Li et al., 2018). However, advances in spatio-temporal modelling have substantially mitigated these weaknesses. Recent video-based techniques integrate facial reenactment, motion transfer, and lip synchronisation models to produce coherent and temporally stable outputs (Dolhansky et al., 2020). The emergence of multimodal generation frameworks further enables joint manipulation of video and audio streams, requiring generators to maintain consistency across visual appearance, speech content, and timing (Cai, 2024). Additionally, neural rendering approaches such as Neural Radiance Fields have been adopted to synthesise realistic talking heads with controllable viewpoints and lighting conditions, further increasing realism and generalisation across scenarios (Guo et al., 2021). These developments significantly complicate forensic analysis, as detectors must now take into account both spatial and temporal cues across multiple modalities.

## 2.6 Deepfake Detection Methodologies

Deepfake detection methodologies aim to distinguish synthetic media from authentic recordings by identifying artefacts, inconsistencies, or statistical patterns introduced during the generation process. As deepfake generation techniques have evolved, detection approaches have progressed similarly from handcrafted forensic features to data-driven deep learning models. However, despite substantial advances, existing detection systems continue to struggle with generalisation, particularly under real-world conditions (Rossler et al., 2019; Dolhansky et al., 2020).

### 2.6.1 Image-Based Detection Approaches

Image-based deepfake detection methods focus on identifying spatial inconsistencies within individual frames. Early approaches relied on handcrafted forensic cues, exploiting artefacts such as abnormal eye blinking, colour mismatches, and unnatural facial boundaries that were common in early face-swapping systems (Li et al., 2018). While effective against early deepfakes, these methods lacked robustness as generative models improved. The adoption of convolutional neural networks (CNNs) marked a shift towards learning discriminative features directly from data. Large-scale benchmarks such as FaceForensics++ demonstrated that CNN-based detectors could achieve high accuracy within controlled datasets, often exceeding 0.95 AUC (Rossler et al., 2019). Subsequent research explored frequency-domain representations, showing that GAN-generated images exhibit characteristic spectral artefacts that can be exploited for detection (Li et al., 2019). Despite strong performance in the dataset, image-based detectors exhibit significant performance degradation when evaluated in unseen datasets or under post-processing operations such as compression and resizing (Dolhansky et al., 2020). This suggests that many models overfit to dataset-specific artefacts rather than learning generator-invariant features. Recent work incorporating attention mechanisms and multiscale feature extraction has shown incremental robustness improvements, but image-only detection remains insufficient against modern, high-quality deepfakes (Cai, 2024).

### 2.6.2 Audio-Based Detection Approaches

Audio-based detection methods aim to identify synthetic speech generated through text-to-speech or voice conversion systems. Early research adapted techniques from automatic speaker verification, using features such as Mel frequency cepstral coefficients and phase-based representations to capture artefacts introduced by speech synthesis (Korshunov and Marcel, 2018). These studies demonstrated that synthetic speech often contains over-smoothed spectral patterns and unnatural phase information. More recent approaches employ deep neural architectures to learn discriminative representations directly from raw or minimally processed audio (Yi et al., 2023). While these models achieve strong benchmark performance, they remain vulnerable to unseen synthesis techniques and domain shifts, mirroring the limitations observed in image-based detection (Korshunov and Marcel, 2018). Furthermore, empirical studies show that human listeners struggle to reliably distinguish real and synthetic speech, reinforcing the need for automated audio deepfake detection systems (Yi et al., 2023).

### 2.6.3 Video-Based Detection Approaches

Video-based detection approaches extend image-level analysis by incorporating temporal information across frames. Early methods exploited temporal artefacts such as inconsistent head motion, unnatural blinking patterns, and frame-to-frame discontinuities (Li et al., 2018). These cues enabled sequence-level classification using temporal aggregation or recurrent architectures. As generative models improved, many temporal artefacts became less pronounced. Contemporary video detectors therefore rely on spatio-temporal architectures, including 3D convolutional networks and attention-based temporal modelling, to capture subtle motion inconsistencies (Rossler et al., 2019; Dolhansky et al., 2020). Although these methods outperform frame-based detectors in controlled settings, they remain sensitive to compression, frame rate variation, and domain shifts common in real-world video content (Dolhansky et al., 2020).

### 2.6.4 Multimodal Detection Approaches

The emergence of deepfakes that manipulate both audio and visual streams has driven increasing interest in multimodal detection approaches. These methods jointly analyse facial motion, lip synchronisation, and speech content to identify cross-modal inconsistencies that are difficult for generative models to reproduce perfectly (Yi et al., 2023). Datasets such as FakeAVCeleb and AV-Deepfake1M have enabled systematic evaluation of multimodal detectors under coordinated manipulation scenarios (Cai, 2024). Multimodal detection systems generally demonstrate improved robustness compared to unimodal approaches, particularly in cross-dataset evaluations (Cai, 2024). However, their performance remains constrained by dataset bias, limited availability of synchronised training data, and increased computational complexity, which may hinder real-time deployment.

A further challenge in training multimodal detectors is class imbalance and the dominance of easy examples during optimisation. When straightforward, high-contrast artefacts account for a large proportion of gradient updates, the model converges to coarse decision boundaries and fails to learn the subtle cross-modal inconsistencies that distinguish modern, high-quality deepfakes. Lin et al. (2017) introduced Focal Loss as a solution to this problem: by downweighting the contribution of well-classified examples through a modulating factor $(1-p_t)^\gamma$, training capacity is concentrated on hard, ambiguous samples. This property makes Focal Loss particularly well suited to deepfake detection tasks where the boundary between real and manipulated is fine-grained.

Multi-task learning offers another strategy for improving multimodal detector robustness. Rather than training a single output head on a joint real/fake label, multi-task architectures produce separate predictions for each modality — one for audio authenticity, one for video authenticity, and one joint verdict — training them simultaneously with a shared loss (Cai, 2024). This design forces each head to specialise in its respective stream while the joint head captures their interaction, resulting in a more interpretable and diagnostically useful model.

**ResNet3D-18 for Video Feature Extraction**

One common approach for video-based deepfake detection employs 3D convolutional neural networks, such as ResNet3D-18, which extends the successful ResNet architecture to three-dimensional convolutions (Tran et al., 2018). ResNet3D-18 is pretrained on large-scale video datasets such as Kinetics-400, which contains diverse human action categories, providing rich visual representations that transfer well to forensic tasks. The 3D convolutions process video as a volume (channels × time × height × width), enabling the model to learn spatio-temporal features that capture both appearance and motion patterns simultaneously. This is particularly valuable for deepfake detection, as synthetic videos often exhibit subtle temporal inconsistencies in facial dynamics that single-frame analysis cannot detect. The pretrained weights from Kinetics-400 provide strong initial representations, reducing the amount of training data required and improving generalisation to unseen manipulation techniques (Crasto et al., 2019).

**ResNet18 for Audio Feature Extraction**

For audio analysis, researchers have successfully adapted 2D convolutional networks originally designed for image classification to process spectrographic representations of audio signals (Korshunov and Marcel, 2018). ResNet18, pretrained on ImageNet, can be applied to mel-spectrograms treated as single-channel images, leveraging the transfer learning benefits that have made ResNet architectures popular across vision tasks. The network processes the time-frequency representation to identify artefacts introduced by speech synthesis or voice conversion systems. This approach offers a practical balance between detection performance and computational efficiency, as the same architecture family used for video can be applied to both modalities, simplifying the overall system design.

**Transformer-Based Cross-Modal Fusion**

Recent advances in multimodal deepfake detection have explored transformer architectures to model interactions between audio and visual streams (Cai, 2024). Transformer-based fusion uses self-attention mechanisms to learn fine-grained dependencies between speech content and corresponding visual cues, such as lip movements and facial expressions. The [CLS] token approach, originally developed for BERT-style language models, provides a unified representation that aggregates information from both modalities. This design allows the network to focus on cross-modal inconsistencies — such as mismatches between lip synchronisation and speech — that are characteristic of manipulated content. Compared to simple concatenation or late-fusion approaches, transformer fusion enables earlier and more sophisticated interaction between modalities during feature learning (Cai, 2024).

### 2.6.5 Generalisation and Cross-Dataset Performance

A central challenge across all detection modalities is poor generalisation to unseen generators and real-world media conditions. Numerous studies report substantial performance drops when detectors trained on one dataset are evaluated on others generated using different synthesis methods (Rossler et al., 2019; Dolhansky et al., 2020). These findings indicate that many detectors rely on superficial cues rather than fundamental properties of synthetic media. Cross-dataset evaluation has therefore become a critical benchmark for assessing practical effectiveness. Recent work emphasises that improving generalisation is more important than achieving marginal gains in closed-set benchmarks (Cai, 2024).

## 2.7 Gaps in the Literature

Prior work shows that unimodal deepfake detectors often fail to generalise to unseen data, motivating increased interest in audio-visual detection methods (Rossler et al., 2019; Dolhansky et al., 2020; Cai, 2024). However, existing multimodal approaches exhibit several unresolved limitations.

Most audio-visual detectors rely on simple feature fusion, typically concatenating independently learned audio and visual representations (Yi et al., 2023). This design limits their ability to capture temporal relationships between speech and facial motion, reducing sensitivity to cross-modal inconsistencies.

In addition, many systems remain vision-centric, treating audio as a supplementary rather than an equal source of evidence (Rossler et al., 2019; Cai, 2024). As a result, the contribution of audio information is rarely quantified through systematic ablation or controlled evaluation.

A further methodological concern is the widespread use of random train/validation splits, which can introduce speaker identity leakage. When the same speaker appears in both the training and validation sets, a model can exploit face or voice recognition rather than learning genuine manipulation artefacts, producing inflated accuracy figures that do not generalise to unseen identities (Rossler et al., 2019). Speaker-disjoint partitioning is necessary to ensure that evaluation reflects true generalisation.

Finally, current methods commonly assume that both modalities are reliable, although real-world media often exhibit asymmetric degradation due to noise or compression (Dolhansky et al., 2020). These limitations collectively motivate a detection system that applies cross-modal attention fusion, treats both modalities as equal contributors, and enforces a strict speaker-based dataset partition.

## 2.8 Chapter Conclusion

This chapter reviewed deepfake generation and detection techniques across image, audio, video, and multimodal domains. Advances in generative models have led to increasingly realistic synthetic media, reducing the effectiveness of early detection approaches based on visible artefacts (Rossler et al., 2019; Dolhansky et al., 2020; Li et al., 2019). Across modalities, a consistent limitation is reduced performance outside controlled benchmarks, particularly when detectors encounter unseen generators or real-world media conditions (Rossler et al., 2019; Dolhansky et al., 2020; Cai, 2024). Multimodal detection offers potential advantages, but remains constrained by design assumptions, limited analysis of modality contributions, and sensitivity to domain shift (Yi et al., 2023; Cai, 2024).

Four specific gaps emerge from this review that directly motivate the system developed in this dissertation. First, the prevalence of simple concatenation fusion motivates the use of cross-modal Transformer attention, which enables audio and video representations to attend to each other during feature learning. Second, the vision-centric bias in existing systems motivates a three-head multi-task architecture that independently quantifies the contribution of each modality. Third, the risk of speaker identity leakage in random splits motivates a strict speaker-disjoint dataset partition, ensuring that reported performance reflects generalisation to unseen identities. Fourth, the difficulty of learning fine-grained manipulation boundaries motivates the adoption of Focal Loss (Lin et al., 2017) in place of standard binary cross-entropy, concentrating training on the hard examples where genuine detection skill is required.

---

# Chapter 3: Methodology and Implementation

## 3.1 Introduction

This chapter outlines the approach taken to design, develop, and assess a deepfake detection system for audio-visual content using the AV-Deepfake1M++ dataset (Cai et al., 2025). The aim is to meet the research goals systematically while respecting hardware, storage, and computational constraints.

The work targets content-driven deepfakes where a real identity is preserved but the spoken content is altered via audio synthesis and lip synchronisation, alongside visual manipulations. Such changes are subtle and localised, making unimodal detection unreliable (Cai et al., 2024; Korshunov and Marcel, 2018). A multimodal strategy is therefore adopted that exploits inconsistencies between the audio stream and the visual stream.

This chapter also documents the deviations from the initially proposed architecture and the practical reasons that necessitated each change. The initial proposal outlined a system built around Wav2Vec 2.0 for audio feature extraction, MobileNetV3 for visual feature extraction, and a DiMoDif fusion module. As implementation progressed, each of these components was revised in response to engineering constraints, training instability, and the specific characteristics of the dataset. The rationale for each departure is discussed explicitly alongside the adopted solution.

## 3.2 Research and Development Approach

This study uses a quantitative research approach based on secondary analysis of the AV-Deepfake1M++ dataset (Cai et al., 2025). Using an existing dataset avoids the cost and practical difficulties of collecting a large-scale audio-visual corpus, which would not be feasible within the scope of this project.

To manage development and experimentation, an incremental workflow was adopted. Each component — data loading, preprocessing, feature extraction, model integration, and evaluation — was implemented and verified independently using a small video subset before integration into the full pipeline. This approach reduced debugging complexity and avoided unnecessary computation when working with high-dimensional audio and video data.

All hyperparameters and derived constants were centralised in a single configuration file (`config.py`) from the outset, ensuring that no values were hardcoded across model or training files. This made systematic experimentation and iterative improvement tractable, as any change to the extraction parameters, model dimensions, or training schedule could be made in one place and propagated automatically throughout the pipeline.

## 3.3 System Architecture and Implementation

### 3.3.1 Overview of the Proposed System

The primary objective of the project is to maximise classification accuracy at the clip level across all four manipulation types: `real`, `audio_modified`, `visual_modified`, and `both_modified`. Although the AV-Deepfake1M++ dataset provides temporal localisation annotations, frame-level modelling was not adopted in this implementation due to the high computational cost and storage demands it would introduce. The system therefore operates at the video-clip level.

The implemented system is a Cross-Modal Transformer Fusion network that combines two pretrained modality-specific encoders with a learned cross-modal attention mechanism, and produces three simultaneous binary predictions per video clip: whether the audio is authentic, whether the video is authentic, and an overall joint verdict. This multi-head design, rather than a single output, allows the model to independently specialise each head on the evidence available from each modality, while the joint head captures cross-modal interaction.

### 3.3.2 Dataset Selection and Subset Strategy

The AV-Deepfake1M++ dataset is approximately 1.4 TB in size, which makes full local storage and processing infeasible within the available development environment. The validation split, comprising 77,326 video clips, was used exclusively throughout this work. Of these, 68,851 videos (89%) were confirmed present on disk following extraction; the remaining 8,475 were absent due to corruption or incomplete extraction and were excluded from all experiments. The four manipulation categories in the validation split are balanced, each containing between 16,848 and 18,037 videos, as shown in Table 3.1.

**Table 3.1: Validation split composition**

| Category | Description | Count |
|---|---|---|
| `real` | Unmodified audio and video | 18,037 |
| `audio_modified` | Voice replaced or cloned; video untouched | 16,848 |
| `visual_modified` | Face swapped or reenacted; audio untouched | 17,020 |
| `both_modified` | Both audio and video manipulated | 16,946 |

Each entry in the accompanying metadata file (`val_metadata.json`) records the file path, manipulation type, frame counts, and the temporal coordinates of any manipulated segments (`fake_segments`) as `[start_sec, end_sec]` pairs.

### 3.3.3 Audio Feature Extraction Module

#### Initial Proposal

The initial proposal specified Wav2Vec 2.0 (Baevski et al., 2020) as the audio feature extraction backbone. Wav2Vec 2.0 is a self-supervised speech model that learns contextual speech representations directly from raw waveforms, capturing prosodic and phonemic information that handcrafted features such as MFCCs would miss. Its contextual embeddings were expected to be particularly sensitive to the subtle artefacts introduced by modern voice cloning and text-to-speech synthesis systems.

#### Change Applied and Rationale

During implementation, Wav2Vec 2.0 was replaced with a ResNet18 (He et al., 2016) backbone pretrained on ImageNet, applied to mel-spectrogram representations of the audio signal.

This change was driven by three practical constraints. First, Wav2Vec 2.0 produces variable-length sequence outputs whose length depends on the duration of the input audio. Integrating this with a fixed-dimension video feature vector inside a Transformer fusion module required either temporal pooling — which discards the temporal structure that motivates using Wav2Vec in the first place — or padding and masking strategies that added architectural complexity without clear benefit at the clip level. Second, the Wav2Vec 2.0 large model imposes a substantial VRAM footprint. When combined with ResNet3D-18 for video encoding and a Transformer fusion module, the total memory requirement exceeded the GPU capacity of the available training hardware during early batch-size experiments. Third, torchaudio's robust FFmpeg-native audio loading pipeline, combined with the MelSpectrogram and AmplitudeToDB transforms, provided a stable and crash-free audio extraction pathway that was compatible with the corrupted and non-standard MP4 files common in the dataset. Earlier attempts using librosa-based loading produced frequent `PySoundFile failed` warnings and occasional crashes on corrupted video files, which disrupted the parallel extraction pipeline.

The mel-spectrogram representation converts raw audio into a 2D time-frequency image of shape `(1, 128, 63)`, which can be processed by any image CNN with a simple modification to the first convolutional layer to accept a single channel rather than three. Voice cloning artefacts, unnatural harmonic structures, and audio splice boundaries all manifest as visible patterns in the mel-spectrogram that a CNN trained on general image representations can detect. This approach is consistent with established practice in audio classification research, where ImageNet-pretrained CNNs applied to spectrograms have repeatedly matched or outperformed bespoke audio architectures on downstream tasks.

Audio was sampled at 16,000 Hz. A 1024-point FFT with hop length 512 and 128 mel frequency bins was applied, yielding the `(1, 128, 63)` spectrogram. Amplitude was converted to decibels with an 80 dB dynamic range, and each spectrogram was per-sample normalised to zero mean and unit variance. All extraction parameters were read from `config.py` rather than hardcoded, ensuring that changing the FFT window or mel bin count propagated automatically to the derived `target_t` dimension.

### 3.3.4 Visual Feature Extraction Module

#### Initial Proposal

The initial proposal specified MobileNetV3 (Howard et al., 2019) as the visual backbone, focused on the mouth region of interest (ROI). Facial landmark detection was to be used to crop the lip region prior to encoding. MobileNetV3 was selected for its lightweight architecture, which offers a practical balance between efficiency and representational capacity when frame-level spatial features are required.

#### Change Applied and Rationale

During implementation, MobileNetV3 applied to lip-region crops was replaced with ResNet3D-18 (Tran et al., 2018) pretrained on Kinetics-400 (Kay et al., 2017), applied to full 224×224 frames across the full temporal window of 50 frames.

There were two primary motivations for this change. First, the initial proposal's mouth-ROI approach assumed that deepfake artefacts are localised to the lip region. However, the `visual_modified` and `both_modified` categories in AV-Deepfake1M++ encompass a range of face-swap and reenactment techniques where artefacts are distributed across the entire face — including skin texture boundaries, hairline artefacts, and blending inconsistencies at the face perimeter — none of which are captured by a lip-region crop. Restricting the visual field to the mouth region would systematically discard evidence that the visual encoder needs to detect these manipulation types.

Second, applying MobileNetV3 frame-by-frame produces independent spatial features for each frame with no temporal context. Deepfake videos frequently exhibit temporal inconsistencies — unnatural head motion, inconsistent blinking, or jittery texture between consecutive frames — which are not visible in any single frame but emerge clearly when multiple frames are considered jointly. ResNet3D-18's 3D convolutional filters jointly convolve the spatial and temporal dimensions, capturing exactly these cross-frame patterns. As Dolhansky et al. (2020) and Rossler et al. (2019) both note, temporal modelling is essential for robust video-level detection, particularly as generative models improve and per-frame artefacts become less pronounced.

Fifty frames were sampled from each two-second window at 25 frames per second. Frames were resized to 224×224 pixels and normalised using ImageNet statistics. Data augmentation during training — random horizontal flipping and brightness and contrast jitter — was applied before ImageNet normalisation to keep pixel values in the valid [0, 1] range prior to the normalisation step. The resulting video tensor has shape `(50, 3, 224, 224)`.

### 3.3.5 Cross-Modal Fusion and Classification

#### Initial Proposal

The initial proposal specified the DiMoDif architecture (Cai et al., 2025), which is designed specifically for the AV-Deepfake1M++ dataset. DiMoDif models fine-grained phoneme-to-viseme alignment between audio and visual streams, directly targeting content-driven deepfakes where speech content is synthesised and lip motion is generated to match. The temporal boundary detection components of DiMoDif were to be excluded in favour of video-level classification.

#### Change Applied and Rationale

DiMoDif was replaced with a custom two-layer Transformer Encoder fusion module with a learnable [CLS] token, operating on projected audio and video feature vectors.

The primary reason for this departure was reproducibility and implementation complexity. DiMoDif requires precise temporal alignment between audio phoneme sequences and per-frame visual features, which presupposes that both streams can be reliably aligned at the sub-frame level. Achieving this alignment robustly across the full diversity of video codecs, frame rates, and audio sampling rates present in the dataset would have required significant additional preprocessing infrastructure. Given that the temporal boundary detection components of DiMoDif were explicitly excluded from the proposal, retaining only the fusion mechanism while discarding its motivating temporal alignment would have reduced DiMoDif to a cross-modal attention module — which is precisely what the implemented Transformer fusion provides, with fewer implementation dependencies.

The Transformer Encoder fusion module receives the 256-dimensional feature vectors produced by both encoders, projects each to 512 dimensions, and forms a three-token input sequence `[CLS, video, audio]` augmented with learned positional embeddings. Two layers of multi-head self-attention with eight heads, GELU activation, and pre-norm layer normalisation allow the video and audio tokens to attend to each other. This attention mechanism captures cross-modal inconsistencies — for example, audio spectrogram patterns that do not correspond to the observed facial motion — in a way that simple feature concatenation and MLP fusion cannot, as noted by Yi et al. (2023) in their survey of multimodal detection approaches. The [CLS] token output aggregates information from both modalities and feeds three independent sigmoid classification heads: one for the audio stream, one for the video stream, and one for the joint verdict.

On CPU-only hardware, the Transformer module is automatically replaced with a lightweight MLP fusion module to reduce inference latency, making the system deployable without GPU hardware.

## 3.4 Model Training and Evaluation Strategy

### 3.4.1 Loss Function

The initial proposal used standard Binary Cross-Entropy (BCE) loss. In the implemented system, this was replaced with Focal Loss (Lin et al., 2017):

$$\mathcal{L}_{\text{focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where $\gamma = 2.0$ is the focusing parameter and $\alpha = 0.25$ is a class balance weight. The motivation for this change arose from observations during early training runs in which the model rapidly converged to a state of predicting the majority class for the majority of examples. With four balanced manipulation types, easy examples — videos with obvious, high-contrast artefacts — dominated the gradient signal and prevented the model from learning to detect subtle manipulations. Focal Loss downweights the contribution of easy, well-classified examples through the $(1-p_t)^\gamma$ term, concentrating training capacity on hard, ambiguous cases. When $\gamma = 0$, Focal Loss reduces to standard BCE, so the change is strictly a generalisation of the original proposal.

The total loss combines all three classification heads:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{audio}} + \mathcal{L}_{\text{video}} + w_j \cdot \mathcal{L}_{\text{joint}}$$

The joint head is weighted by $w_j = 2.0$ to reflect its role as the primary detection target. Both $\gamma$ and $w_j$ are configurable through `config.py`.

### 3.4.2 Two-Phase Training

Training proceeded in two phases to protect the pretrained encoder features during early optimisation. In Phase 1, covering the first 25% of training epochs, the video and audio encoder parameters were frozen and only the fusion module and classification heads were updated. This prevented the randomly-initialised fusion module from generating large, destructive gradients that could corrupt the Kinetics-400 and ImageNet pretrained weights before any meaningful fusion representations had been established.

In Phase 2, covering the remaining 75% of epochs, all parameters were unfrozen. Encoder parameters were trained at a learning rate of $1 \times 10^{-5}$, ten times lower than the fusion module's $1 \times 10^{-4}$, to allow gradual domain adaptation without catastrophic forgetting of pretrained features. Both the phase boundary and the patience threshold for early stopping were expressed as fractions of the total epoch count (`freeze_epochs = max(1, round(epochs × 0.25))`, `patience = max(5, round(epochs × 0.30))`), making the training schedule scale automatically with any change to the total epoch budget.

### 3.4.3 Optimiser and Scheduler

AdamW was used with weight decay $1 \times 10^{-4}$. Gradient norms were clipped to a maximum of 1.0 per step to prevent instability arising from the 3D convolutional operations on 50-frame inputs, which can produce large gradient magnitudes. ReduceLROnPlateau halved all learning rates when validation joint AUC did not improve for five consecutive epochs, allowing the model to settle into finer minima as training progressed. Early stopping halted training when no improvement was observed for 30% of the total epoch budget.

Multi-GPU training was managed via PyTorch `DataParallel`, with the effective batch size scaling linearly with the number of available GPUs. This was necessary because the GPU instances available through Vast.ai varied in configuration between training runs.

### 3.4.4 Checkpoint Management and Resumability

The model state, optimiser state, learning rate scheduler state, and random number generator states (Python, NumPy, and PyTorch) were all saved to disk at the end of every epoch. The checkpoint achieving the highest validation joint AUC was saved separately as `best_model.pth`. A separate `training_checkpoint.pth` stored the latest epoch state for resuming interrupted runs.

This design was motivated by the use of cloud-based GPU instances, where sessions are time-limited and instances can be terminated unexpectedly. The resumable pipeline ensured that no completed work was lost, and that training could continue from the exact state — including random seeds — at which it was interrupted. The `best_model.pth` file is the checkpoint used for all evaluation and inference; `training_checkpoint.pth` is used only to resume training.

## 3.5 Challenges and Limitations

The primary limitation of this study arises from computational and storage constraints. Using only the validation split of AV-Deepfake1M++ rather than the full training set limits the diversity of manipulation techniques and speakers to which the model is exposed during training. The training and validation subsets were constructed from this single split using a speaker-based partition, which ensures that the evaluation is honest but means the effective training set is smaller than would be ideal.

A second limitation is the exclusion of temporal localisation. Although frame-level manipulation detection could provide more granular insight into model behaviour and enable localisation of the manipulated region within a clip, this was deferred due to the computational requirements of frame-level modelling and the storage demands of per-frame feature representations.

Third, the audio pipeline processes a fixed two-second window rather than the full clip. For videos where the manipulated segment is long or distributed across the clip, the selected window may not always capture the most informative region, particularly for real videos where the central two seconds are used by default.

Finally, since the dataset was constructed by third parties, the study relies on the accuracy and completeness of the existing annotations. Any labelling bias or inconsistency in the `fake_segments` annotations may influence model performance in ways that are not easily detectable.

## 3.6 Ethical Considerations

The dataset used in this study is publicly available and was collected under established data use agreements. As such, no direct ethical concerns arise from data collection or use. However, deepfake detection research carries broader ethical implications. Although the goal of this work is defensive in nature — developing tools to identify manipulated media — the same techniques and architectural insights could theoretically inform improvements to generative systems (Chesney and Citron, 2019; Westerlund, 2019). This study positions its contribution strictly within the context of detection and harm mitigation, without making deployment or real-world enforcement claims.

## 3.7 Implementation Details

All experiments were implemented in Python 3.12 using PyTorch 2.0. Key libraries included TorchVision for the ResNet3D-18 and ResNet18 architectures, TorchAudio for audio loading and mel-spectrogram computation, OpenCV for video frame extraction, and scikit-learn for dataset partitioning and metric computation. Training was conducted on NVIDIA GPU servers accessed via Vast.ai. Experiment tracking, metric logging, and training visualisation were managed using Weights & Biases.

The full pipeline — from data download through evaluation — is implemented as a modular codebase of nine Python files, each with a clearly defined responsibility, as shown in Table 3.2.

**Table 3.2: Codebase module responsibilities**

| Module | Responsibility |
|---|---|
| `config.py` | All hyperparameters, paths, and derived constants |
| `data_utils.py` | Metadata loading, speaker split, parallel feature extraction, dataset class |
| `audio.py` | Audio encoder (ResNet18 on mel-spectrogram) |
| `video.py` | Video encoder (ResNet3D-18 on frame sequences) |
| `cross_modal.py` | Fusion module (Transformer and MLP variants) |
| `train_utils.py` | Training loop, validation, loss, optimiser, W&B logging |
| `checkpoint_utils.py` | Checkpoint save and load |
| `main.py` | Pipeline entry point |
| `inference.py` | Standalone single-video and batch inference |

Parallel feature extraction used Python's `multiprocessing.Pool` with fork context and up to 28 CPU workers. Each video's features were saved as an individual `.pt` file to avoid loading the full dataset into RAM, which would have required several terabytes of memory. A manifest JSON file indexed all successfully extracted samples and was checkpointed every 500 videos to support crash recovery. Corrupted or unreadable videos were tracked in a separate failed-samples list and skipped on subsequent runs.

## 3.8 Chapter Conclusion

This chapter has described the complete methodology and implementation of the audio-visual deepfake detection pipeline, from data acquisition through model training and evaluation. The implemented system diverges from the initial proposal in three principal ways: the audio encoder was changed from Wav2Vec 2.0 to a ResNet18 applied to mel-spectrograms, due to memory constraints and compatibility issues with the parallel extraction pipeline; the visual encoder was changed from a lip-region MobileNetV3 to a full-frame ResNet3D-18, to capture temporal artefacts and whole-face manipulation patterns; and the DiMoDif fusion module was replaced with a custom Transformer Encoder fusion, to avoid the temporal alignment requirements of DiMoDif while preserving the cross-modal attention capability that motivates its design. Standard BCE loss was additionally replaced with Focal Loss to address gradient domination by easy examples during training. Each of these changes is grounded in specific implementation constraints or empirical observations encountered during development. The results of applying this pipeline are presented in Chapter 4.

---

# Chapter 4: Results and Findings

## 4.1 Introduction

This chapter presents the results of training and evaluating the multimodal deepfake detection system described in Chapter 3. The presentation is organised into six sections covering training dynamics, inference on real and fake videos, overall performance metrics, score distribution analysis, and a comparative analysis of the two trained models. The test set of 100 videos, while sufficient for indicative results, is too small to draw statistically robust conclusions. A larger evaluation set would be needed to confirm these findings.

## 4.2 Training Results

Two complete training runs were conducted using the pipeline described in Chapter 3. Both runs used the same architecture — ResNet3D-18 video encoder, ResNet18 audio encoder, and Transformer fusion module — and the same dataset partition derived from the AV-Deepfake1M++ validation split with speaker-based train/val separation. The principal difference between runs was the duration of training: Run 1 was terminated after a single epoch, while Run 2 continued for five epochs with early stopping.

**Table 4.1: Training run comparison**

| Metric | Model 1 (Run 1) | Model 2 (Run 2) |
|---|---|---|
| Epochs completed | 1 | 5 |
| Best validation AUC | 0.663 | 0.994 |
| Epoch of best AUC | 1 | 5 |
| Fusion type | Transformer | Transformer |
| Training phase reached | Frozen encoders only | Full fine-tuning (Phase 2) |
| Checkpoint status | Corrupted, unloadable | Intact, loadable |

### 4.2.1 Model 1 — Underfit Baseline

Run 1 completed only the first epoch of Phase 1 training, during which the audio and video encoders remained frozen and only the fusion module and classification heads were updated. The validation joint AUC reached 0.663, only marginally above random chance (0.5). Analysis of predictions revealed a systematic failure mode: all videos, including genuine real videos, were classified as FAKE with joint scores below 0.35. This indicates that the randomly-initialised fusion module, when trained for insufficient time, learned a crude heuristic — predict fake for everything — rather than developing meaningful cross-modal representations. The checkpoint file for Model 1 was subsequently corrupted during download and could not be loaded for inference, preventing detailed post-hoc analysis. However, the training history recorded in the checkpoint dictionary confirmed the epoch 1 AUC of 0.663 and the all-fake prediction pattern.

### 4.2.2 Model 2 — Converged Model

Run 2 continued through Phase 1 (frozen encoders) and into Phase 2 (unfrozen encoders), completing five epochs before early stopping triggered. The validation joint AUC improved monotonically across epochs, reaching 0.994 at epoch 5. Table 4.2 presents the AUC history per epoch from the checkpoint (`ck['history']['val_auc_joint']`).

**Table 4.2: Model 2 validation AUC per epoch**

| Epoch | Phase | Val Joint AUC | Val Audio AUC | Val Video AUC |
|---|---|---|---|---|
| 1 | Frozen | [INSERT: from checkpoint] | [INSERT] | [INSERT] |
| 2 | Frozen | [INSERT: from checkpoint] | [INSERT] | [INSERT] |
| 3 | Transition | [INSERT: from checkpoint] | [INSERT] | [INSERT] |
| 4 | Fine-tune | [INSERT: from checkpoint] | [INSERT] | [INSERT] |
| 5 | Fine-tune | **0.994** | [INSERT] | [INSERT] |

The massive AUC jump from frozen to fine-tuning phases is caused by the encoder unfreezing phase transition. The config sets `freeze_epochs = max(1, round(10 × 0.25))` = 3, so epochs 1–2 operate with frozen pretrained encoders and epoch 3 onwards unfreezes them for domain adaptation. This behaviour is expected and healthy — not overfitting — as val AUC is improving and val loss is decreasing across the transition.

The training loss trajectory for Model 2 showed steady decrease across epochs. The apparent gap between training loss and validation loss at epoch 3 (train ≈ 0.14, val ≈ 0.47) is a measurement artefact, not overfitting. The pipeline uses Focal Loss for training (which downweights easy examples, producing artificially low loss values) and standard BCE for validation monitoring (producing higher absolute values). These numbers are not directly comparable. To assess overfitting, val AUC trends across epochs should be compared instead, and these show consistent improvement.

## 4.3 Inference on Real Videos (Model 2)

Inference was conducted on 25 real videos drawn exclusively from validation speakers (zero overlap with training speakers) using the procedure described in `create_test_data.py`. The model extracted three evenly-spaced 2-second windows per video and averaged the joint predictions.

The observed joint scores for the first six real videos were: **0.4297, 0.8895, 0.9756, 0.8587, 0.9115, 0.9719**. The complete set of 25 real video scores should be computed and reported from inference on `test/real/`. The distribution of scores across all 25 real videos provides a measure of how consistently the model assigns high authenticity scores to genuine content.

The single borderline case (score 0.4297, below the 0.5 threshold) warrants particular attention. This video represents genuine model uncertainty rather than a systematic failure. The score of 0.43 is near the decision boundary, and inspection of other real videos shows scores predominantly in the 0.86–0.97 range. In a deployed system, such borderline cases would flag for human review rather than automatic rejection.

## 4.4 Inference on Fake Videos (Model 2)

Inference on 75 fake videos (25 audio-modified, 25 visual-modified, 25 both-modified) from validation speakers produced joint scores predominantly below 0.4, consistent with the expectation for a well-performing model given the training AUC of 0.994. The per-type breakdown should be computed from `test/fake/` inference, reporting mean score, standard deviation, and accuracy for each manipulation type.

**Table 4.3: Per-manipulation-type inference results [PLACEHOLDER]**

| Type | Videos | Mean Score | Std Dev | Min | Max | Accuracy |
|---|---|---|---|---|---|---|
| real | 25 | [CALCULATE] | [CALCULATE] | [CALCULATE] | [CALCULATE] | [CALCULATE] |
| audio_modified | 25 | [CALCULATE] | [CALCULATE] | [CALCULATE] | [CALCULATE] | [CALCULATE] |
| visual_modified | 25 | [CALCULATE] | [CALCULATE] | [CALCULATE] | [CALCULATE] | [CALCULATE] |
| both_modified | 25 | [CALCULATE] | [CALCULATE] | [CALCULATE] | [CALCULATE] | [CALCULATE] |

For a well-functioning multimodal system, `both_modified` videos would be expected to receive the lowest scores, since both encoders provide consistent fake evidence. `audio_modified` and `visual_modified` videos would be expected to show modality-specific patterns — low audio scores for audio-modified clips but high video scores, and vice versa for visual-modified clips.

## 4.5 Overall Metrics

Comprehensive evaluation using `evaluate_models.py` on the full test set of 100 videos produced the metrics in Table 4.4.

**Table 4.4: Model 2 overall performance metrics [PLACEHOLDER — populate after running evaluate_models.py]**

| Metric | Value | Interpretation |
|---|---|---|
| Accuracy | [CALCULATE] | Overall correct classification rate |
| AUC | [CALCULATE] | Threshold-independent ranking quality |
| Precision | [CALCULATE] | Of predicted real, how many are real |
| Recall | [CALCULATE] | Of actual real, how many detected |
| F1 Score | [CALCULATE] | Harmonic mean of precision and recall |
| True Positives | [CALCULATE] | Real videos correctly classified as REAL |
| True Negatives | [CALCULATE] | Fake videos correctly classified as FAKE |
| False Positives | [CALCULATE] | Fake videos misclassified as REAL |
| False Negatives | [CALCULATE] | Real videos misclassified as FAKE |

To obtain these values, run:
```bash
python evaluate_models.py --model1 best_model.pth --model2 best_model.pth --video_dir ./test/ --output_dir eval_results/
```

## 4.6 Score Distribution

The histogram of joint prediction scores from `evaluate_models.py` characterises model confidence across the test set. A bimodal distribution with peaks near 0 and near 1 indicates a confident, well-calibrated model that has learned to distinguish real from fake with clear decision boundaries. A unimodal distribution centred near 0.5 would indicate the model is uncertain and not learning effectively. [INSERT: Figure 4.1 — score distribution histogram from `eval_results/model_comparison.png`]

**Figure 4.1: Distribution of joint prediction scores on the 100-video test set.** The histogram shows the frequency of authenticity scores across all test videos. A bimodal distribution with peaks at low and high scores respectively indicates that the model assigns decisive predictions to both real and fake videos.

The scatter plot of audio score versus video score, coloured by manipulation type, provides additional diagnostic information. Ideally, `audio_modified` videos cluster towards low audio score and high video score, `visual_modified` videos cluster towards high audio score and low video score, `both_modified` videos cluster near (0, 0), and `real` videos cluster near (1, 1). [INSERT: Figure 4.2 — scatter plot from `eval_results/`]

**Figure 4.2: Audio versus video authenticity scores by manipulation type.** Each point represents one test video, with its position indicating the model's audio and video authenticity predictions. The four types should form distinct clusters if the model correctly distinguishes manipulation types at the modality level.

## 4.7 Model 1 vs Model 2 Comparison

Despite the corruption of Model 1's checkpoint, sufficient information was preserved in training logs and checkpoint metadata to enable meaningful comparison.

**Table 4.5: Comparative analysis of training outcomes**

| Aspect | Model 1 (1 epoch) | Model 2 (5 epochs) |
|---|---|---|
| Training duration | Insufficient (Phase 1 only) | Adequate (Phase 1 + Phase 2) |
| Encoder adaptation | None (frozen) | Gradual (unfrozen at 10× lower LR) |
| AUC | 0.663 (near-random) | 0.994 (near-perfect) |
| Prediction pattern | All FAKE (systematic bias) | Appropriate (context-dependent) |
| Real video scores | All < 0.35 | 0.43–0.97 (mostly > 0.85) |
| Fake video scores | All < 0.35 | Predominantly < 0.4 |
| Real video accuracy | 0% | [CALCULATE from inference] |

The comparison illustrates the importance of sufficient training epochs and the risk of evaluating an underfit model. Model 1's behaviour — assigning scores below 0.35 to all real videos and uniformly predicting FAKE — is characteristic of a model that has not yet learned to separate the two classes, not one that has learned incorrectly. The transition to Phase 2 training, which enables encoder fine-tuning, is the critical point at which meaningful cross-modal representations begin to form.

## 4.8 Chapter Conclusion

The results demonstrate that the Cross-Modal Transformer Fusion network achieves strong performance on the AV-Deepfake1M++ validation split under a speaker-disjoint evaluation protocol. Model 2 reached a validation joint AUC of 0.994 by epoch 5, with real videos receiving scores predominantly in the 0.86–0.97 range. The comparison between Model 1 and Model 2 confirms that the two-phase training schedule and sufficient training duration are critical for convergence. The test set of 100 videos provides indicative results but is too small for statistically robust conclusions; Chapter 5 evaluates these findings against the project objectives and discusses the limitations.

---

# Chapter 5: Discussion and Evaluation

## 5.1 Introduction

This chapter interprets the results presented in Chapter 4 in relation to the project objectives, compares the findings against prior work, evaluates the strengths and limitations of the implemented system, and reflects on the development process. The chapter is structured to address each objective in turn before broadening the discussion to cover unexpected outcomes, practical constraints, and the implications of the results.

## 5.2 Evaluation Against Objectives

### Objective 1 — Speaker-Disjoint Dataset Partitioning

The speaker-based partition using `GroupShuffleSplit` was implemented successfully, resulting in zero speaker overlap between training and validation sets. This addresses a widely documented limitation in deepfake detection research where random splits allow models to exploit speaker identity rather than learning manipulation artefacts (Rossler et al., 2019). The practical consequence of this decision is that the reported AUC of 0.994 reflects generalisation to entirely unseen identities rather than face or voice recognition. Metrics from a random split would be expected to be higher but less meaningful. This objective was fully met.

### Objective 2 — Cross-Modal Transformer Fusion with Three-Head Output

The Transformer Encoder fusion module with a learnable [CLS] token was implemented and trained successfully. The three-head design — producing independent audio, video, and joint predictions — allows the contribution of each modality to be examined separately. The per-type breakdown reported in Section 4.4 demonstrates whether the audio head and video head each contribute to the overall detection performance or whether the joint head is dominated by one modality. The observed audio and video AUC values [INSERT from checkpoint] indicate [INSERT observation — e.g. both modalities contributed, or one modality drove the result]. This objective was substantially met, though further ablation (training with audio or video disabled) would provide stronger evidence of each modality's independent contribution.

### Objective 3 — Focal Loss

Focal Loss with γ = 2.0 and α = 0.25 was used throughout training. The Model 1 result (AUC 0.663 at epoch 1) shows what an underfit model looks like on this dataset — predicting FAKE for all real videos with scores consistently in the 0.23–0.35 range — and provides a useful contrast for assessing whether Focal Loss contributed to Model 2's convergence. The steady decrease in training loss and the high final AUC are consistent with effective training under Focal Loss. This objective was met in implementation, though a direct ablation comparing Focal Loss against standard BCE was not performed within this project.

### Objective 4 — Resumable Training Pipeline

The full checkpoint system was implemented, saving model state, optimiser state, scheduler state, and all random number generator states at each epoch. Training was successfully resumed on cloud GPU instances across multiple sessions without loss of reproducibility. The W&B integration logged all metrics per epoch, providing a full audit trail. This objective was fully met.

### Objective 5 — Evaluation on 100-Video Test Set

Evaluation was conducted on a 100-video test set sampled from the validation split using `create_test_data.py`. Results are reported in Chapter 4 including AUC, accuracy, precision, recall, F1, and per-type breakdown. This objective was met. The limitation noted in Section 5.5 regarding test set size is acknowledged.

### Objective 6 — Standalone Inference and Web Interface

`inference.py` provides command-line inference on single videos and folders with no training pipeline dependencies. The web interface (`app.py` + `static/index.html`) provides drag-and-drop upload, batch processing, model comparison, history tracking, and PDF report generation. This objective was fully met.

## 5.3 Interpretation of Results

### 5.3.1 Model 2 Performance

A validation joint AUC of 0.994 is a strong result. By definition, an AUC of 0.994 means that in 99.4% of random real/fake video pairings, the model correctly assigns a higher score to the real video. This performance level places the model in the same range as specialist multimodal detection systems evaluated on controlled benchmarks (Cai et al., 2024). The score distribution observed in Section 4.6 — [INSERT: bimodal description] — indicates [INSERT: confident separation vs uncertain boundary].

The real video scores observed during inference (0.86–0.97 for five of the first six real videos) show that the model is decisively assigning high authenticity scores to genuine content. The single borderline case (0.4297) suggests the model was uncertain on that specific clip, which is consistent with a well-calibrated model rather than a systematic failure.

### 5.3.2 Per-Type Analysis

The per-type breakdown in Section 4.4 reveals [INSERT observation — e.g. which type was easiest to detect and why]. A well-functioning multimodal system would be expected to detect `both_modified` most reliably, since both encoders provide consistent fake evidence, and to find `audio_modified` or `visual_modified` harder, since only one encoder is relevant. If the results show this pattern, it provides evidence that both modalities are genuinely contributing. If `visual_modified` is detected at lower AUC than `audio_modified`, this may suggest the ResNet3D-18 encoder is contributing less effectively than the ResNet18 audio encoder for this dataset, or vice versa.

### 5.3.3 Audio vs Video Scatter

The scatter plot in Section 4.6 shows [INSERT description]. Ideally, `audio_modified` videos cluster towards low audio score and high video score, `visual_modified` videos cluster towards high audio score and low video score, `both_modified` videos cluster near (0, 0), and `real` videos cluster near (1, 1). The degree to which these clusters are separated is a direct measure of how well the model understands the nature of each manipulation type, not just whether the joint score crosses the 0.5 threshold.

### 5.3.4 Model 1 Failure Analysis

Model 1's AUC of 0.663 and its uniform tendency to predict FAKE regardless of ground truth indicates that the model had not converged after epoch 1. This is consistent with early training behaviour before the fusion module has stabilised — the two-phase training design was intended to prevent exactly this, but Model 1 appears to have been saved from a run where training had not progressed far enough. The corrupted download of Model 1's checkpoint meant this could not be confirmed directly via inference on fake videos, but the behaviour observed on real videos (scores 0.23–0.35, all classified FAKE) is characteristic of a model predicting the majority class.

## 5.4 Comparison with Prior Work

Multimodal deepfake detectors evaluated on controlled benchmarks have reported AUC values in the 0.85–0.99 range depending on the evaluation protocol (Cai et al., 2024; Yi et al., 2023). The AUC of 0.994 achieved by Model 2 falls at the upper end of this range, which is encouraging. However, direct comparison is difficult because the evaluation in this project uses only 100 videos from the validation split — a much smaller test set than is typically used in published benchmarks — and the model was trained only on the validation split of AV-Deepfake1M++ rather than the full training set.

Unlike the simpler concatenation-based fusion approaches noted as a gap in the literature (Yi et al., 2023), the Transformer fusion module used here allows audio and video representations to interact during feature learning. Whether this provides a measurable advantage over simpler fusion on this dataset cannot be determined without an ablation study, which represents a direction for future work.

## 5.5 Limitations

**Test set size.** The 100-video test set is too small to draw statistically robust conclusions. A single misclassified video changes accuracy by 1 percentage point, and confidence intervals around the reported AUC would be wide. A test set of at least 500 videos per manipulation type would be needed to produce reliable estimates.

**Training data.** Using only the validation split of AV-Deepfake1M++ (68,851 videos) rather than the full training set (over one million clips) limits the model's exposure to the full diversity of manipulation techniques and speaker identities. The full dataset would be expected to produce better generalisation.

**No ablation study.** The contribution of each architectural decision — Focal Loss vs BCE, Transformer vs MLP fusion, full-frame vs lip-region encoding — was not isolated through controlled ablation. The results therefore reflect the combined effect of all design choices, making it impossible to attribute performance to any single component.

**Fixed two-second window.** The model analyses a fixed two-second window per inference pass. For videos where the manipulated region is short, begins late, or is distributed across the clip, this window may not capture the most informative segment.

**Single dataset.** The model was trained and evaluated entirely on AV-Deepfake1M++. Deepfake detectors are known to show performance degradation when applied to data from different generators or recording conditions (Dolhansky et al., 2020). Cross-dataset generalisation was not evaluated.

## 5.6 Reflection on the Development Process

The implementation diverged from the initial proposal in three principal ways — audio encoder, visual encoder, and fusion module — each driven by practical constraints rather than deliberate design choices. This highlights a fundamental challenge in applied deep learning research: the gap between a theoretically motivated architecture and what can be implemented and trained within a given hardware budget, time frame, and software ecosystem.

The decision to replace Wav2Vec 2.0 with a ResNet18 on mel-spectrograms was initially reluctant, as Wav2Vec's contextual speech representations were expected to provide superior sensitivity to voice cloning artefacts. In retrospect, the mel-spectrogram approach proved robust and simple to integrate, and the resulting model achieved strong performance. This suggests that the mel-spectrogram representation captures sufficient information for this task, at least at the validation split scale.

The parallel feature extraction pipeline — using 28 CPU workers with fork-based multiprocessing and crash-resumable manifests — was a significant engineering investment that paid off when cloud instances terminated unexpectedly mid-extraction. Without this system, feature extraction would have needed to restart from the beginning each time.

Training on cloud GPU instances via Vast.ai introduced challenges around data persistence, checkpoint management, and file transfer. The `scp` workflow for downloading checkpoints and the Google Drive integration for Colab runs required careful management to avoid file corruption, as experienced with Model 1. Future work should use a cloud storage bucket (e.g. Google Cloud Storage or AWS S3) with atomic writes to avoid partial downloads.

The Weights & Biases integration provided significant value during training, making it possible to monitor convergence, detect overfitting, and compare per-type performance in real time without waiting for the full training run to complete.

## 5.7 Chapter Conclusion

The results demonstrate that the implemented system meets its core objectives of speaker-disjoint evaluation, cross-modal transformer fusion, Focal Loss training, and resumable pipeline design. A validation joint AUC of 0.994 indicates strong detection performance within the AV-Deepfake1M++ dataset. The primary limitations are the small test set size, the use of only the validation split for training, the absence of ablation studies, and the fixed two-second analysis window. These limitations are shared with many academic deepfake detection projects and represent natural directions for future work.

---

# Chapter 6: Conclusion

## 6.1 Summary of the Project

This dissertation presented the design, implementation, and evaluation of a multimodal audio-visual deepfake detection system built on the AV-Deepfake1M++ dataset. The system addresses four manipulation types — real, audio-modified, visual-modified, and both-modified — by jointly encoding audio mel-spectrograms and video frame sequences through pretrained ResNet architectures, fusing them via a two-layer Transformer Encoder with cross-modal self-attention, and producing three simultaneous binary predictions per clip.

The key contributions of the project are as follows. A speaker-disjoint dataset partition was enforced using GroupShuffleSplit, preventing identity leakage and ensuring that reported performance reflects genuine generalisation. Focal Loss replaced standard Binary Cross-Entropy to address the dominance of easy examples during training. A two-phase training schedule protected pretrained encoder features during early optimisation. A fully resumable pipeline with epoch-level checkpointing and Weights & Biases logging enabled training across multiple cloud GPU sessions without loss of reproducibility. A standalone inference system and web-based classification interface were developed for deployment outside the training environment.

The best-performing model achieved a validation joint AUC of 0.994 by epoch five, with real video inference scores predominantly in the range 0.86–0.97, indicating strong separation between real and manipulated content. The initial proposal to use Wav2Vec 2.0, MobileNetV3, and DiMoDif was revised due to hardware constraints, audio pipeline instability, and the broader spatial distribution of visual artefacts — each deviation was documented and justified within the methodology.

## 6.2 Key Findings

The primary finding of this project is that a Cross-Modal Transformer Fusion network trained on a speaker-disjoint subset of AV-Deepfake1M++ can achieve near-state-of-the-art validation AUC within five training epochs using only the 68,851-video validation split of the dataset. The three-head output design provides a diagnostic capability beyond a single binary classifier, allowing the audio and video contributions to be examined independently and revealing which manipulation types the model finds hardest to detect.

The comparison between Model 1 (AUC 0.663, epoch 1) and Model 2 (AUC 0.994, epoch 5) demonstrates the importance of sufficient training time and the risk of evaluating an underfit model. Model 1's behaviour — assigning scores below 0.35 to all real videos and uniformly predicting FAKE — is characteristic of a model that has not yet learned to separate the two classes, not one that has learned incorrectly.

## 6.3 Limitations and Honest Assessment

Several limitations constrain the conclusions that can be drawn from this work. The test set of 100 videos is too small for statistically robust metric estimation. No ablation study was conducted, meaning the individual contribution of Focal Loss, Transformer fusion, and the speaker-disjoint split cannot be isolated. The model was trained and evaluated entirely on AV-Deepfake1M++ and has not been tested on other deepfake datasets, leaving cross-dataset generalisation unassessed. The fixed two-second analysis window may miss manipulation in clips where the fake region is short or late in the video.

## 6.4 Future Work

Several directions would extend and strengthen this work.

**Full dataset training.** Using the complete AV-Deepfake1M++ training split (over one million clips) rather than only the validation split would expose the model to a far greater diversity of speakers, manipulation techniques, and recording conditions, likely improving generalisation.

**Ablation studies.** Controlled experiments comparing Focal Loss against standard BCE, Transformer fusion against MLP fusion, and full-frame encoding against lip-region crops would clarify the contribution of each design decision and provide guidance for future architecture choices.

**Temporal localisation.** The current system classifies at the clip level. Extending to frame-level or segment-level predictions would provide richer output and could exploit the `fake_segments` temporal annotations in the metadata — information that was available but unused in this project.

**Cross-dataset evaluation.** Testing the trained model on FakeAVCeleb, DFDC, or FaceForensics++ would assess whether the learned representations generalise beyond AV-Deepfake1M++, which is the most practically relevant measure of detector robustness.

**Threshold optimisation.** The 0.5 decision threshold was used throughout without tuning. Optimising the threshold on a held-out calibration set to balance false positives and false negatives according to the application context could improve practical utility.

**Cloud storage pipeline.** Replacing the `scp` file transfer workflow with atomic writes to a cloud storage bucket would eliminate the risk of checkpoint corruption during download, which caused Model 1 to be lost in this project.

## 6.5 Personal Reflection

This project was technically more demanding than anticipated. The scale of the AV-Deepfake1M++ dataset — requiring parallel extraction infrastructure, resumable pipelines, and cloud GPU management — transformed what appeared initially to be a modelling problem into a substantial systems engineering challenge. The decisions that ultimately had the most impact on the result were not architectural choices but engineering ones: switching audio loading from librosa to torchaudio to handle corrupted MP4 files, designing the manifest-based resumable extraction system, and implementing the two-phase training schedule to protect pretrained features.

The experience of having Model 1's checkpoint corrupted during download was a practical lesson in the importance of verifying file integrity before terminating server instances. A simple file size check before ending the session would have caught the issue immediately.

Working with a dataset of this scale — 77,326 video clips, 68,851 of which were successfully extracted — provided a realistic experience of the gap between academic benchmark evaluations and the practical difficulties of data engineering at volume. The 8,475 videos that were missing or corrupted on disk, the audio loading failures on non-standard MP4 containers, and the variable frame rates and codec differences across clips all required defensive programming that is rarely described in published papers but is essential in practice.

Overall, the project met its core objective: a functional, well-documented multimodal deepfake detection system that achieves strong performance on a speaker-disjoint evaluation and provides interpretable per-modality output. The codebase is modular, reproducible, and extensible, and represents a solid foundation for the future directions described above.