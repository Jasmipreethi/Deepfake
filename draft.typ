// draft.typ
#let project(
  title: "Deepfake Detection Using Cross-Modal Transformer Fusion",
  author: "Jasmi Preethi Alasapuri",
  student_id: "2571395",
  degree: "Data Science and Artificial Intelligence",
  supervisor: "Lucian Duta",
  date: datetime.today(),
  abstract: [
    This dissertation presents a multimodal deepfake detection system trained on the validation subset of the AV-Deepfake1M++ dataset. The system uses a cross-modal transformer fusion network integrating ResNet3D-18 and ResNet18 encoders to produce simultaneous audio, video, and joint authenticity predictions under a speaker-disjoint evaluation protocol.

    While Model 2 reached a peak validation AUC of 0.994 after five training epochs, evaluation on a 100-video speaker-disjoint test set identified Model 3 (the epoch-3 checkpoint from the same run that produced Model 4) as achieving superior threshold-based performance, with 93% accuracy and zero false positives at a 0.5 threshold — demonstrating the importance of score calibration over raw AUC optimisation. The project contributes a modular, resumable training pipeline, a standalone inference system, and a web interface for video upload and classification. Results are indicative trends rather than definitive benchmarks: the 100-video test set (25 per category) means a single misclassification shifts category accuracy by 4 percentage points, with 95% confidence intervals of ±7–10%. Training was limited to five epochs due to student resource constraints, and no ablation studies were conducted.
  ],
  acknowledgments: [
    I would like to express my sincere gratitude to my supervisor, Lucian Duta, for his invaluable guidance, patience, and support throughout this project. His feedback and encouragement were instrumental in shaping this work.

    I am grateful to the University of East London and the course leader for providing the academic environment and resources that made this project possible.

    To my family, thank you for your unwavering love, patience, and belief in me throughout this journey.

    To my friends, especially Jayrup, for his constant support, guidance, and encouragement — thank you for being there through the late nights and challenging moments.

    I am also grateful to the developers of the AV-Deepfake1M++ dataset for providing the benchmark data that made this work possible.
  ],
  body,
) = {
  // 1. Page & Font Setup
  set document(author: author, title: title)
  set page(
    paper: "a4",
    margin: (left: 2.5cm, right: 2.5cm, top: 2.5cm, bottom: 2.5cm),
    numbering: none,
  )
  set text(lang: "en", size: 12pt)

  // 2. The Title Page
  align(center)[
    #image("uel.svg", width: 5cm)
    #v(2em)

    #text(size: 14pt)[SCHOOL OF ARCHITECTURE, COMPUTING AND ENGINEERING] \
    #v(0.5em)
    #text(size: 12pt)[Department of Engineering and Computing]

    #v(4fr)

    #text(weight: "bold", size: 22pt)[#title]

    #v(2fr)

    #text(weight: "bold", size: 14pt)[#author] \
    #text(size: 12pt)[#student_id]

    #v(2fr)

    A report submitted in part fulfilment of the degree of \
    BSc (Hons) in #text(style: "italic")[#degree]

    #v(1em)

    Supervisor: #supervisor \
    CN6000

    #v(2fr)

    #v(1em)
    #date.display("[day] [month repr:long] [year]")
    #v(2em)
  ]
  pagebreak()

  set page(numbering: "i")
  counter(page).update(1)

  // 3. Front Matter (Abstract & Acknowledgments)
  set heading(numbering: none)

  heading(level: 1, outlined: true)[Abstract]
  abstract
  pagebreak()

  heading(level: 1, outlined: true)[Acknowledgments]
  acknowledgments
  pagebreak()

  // 4. Table of Contents
  heading(level: 1, outlined: false)[Contents]
  outline(depth: 3, indent: 2em, title: none)
  pagebreak()

  // List of Figures
  heading(level: 1, outlined: true)[List of Figures]
  outline(target: figure.where(kind: image), depth: 1, indent: 2em, title: none)
  pagebreak()

  // List of Tables
  heading(level: 1, outlined: true)[List of Tables]
  outline(target: figure.where(kind: table), depth: 1, indent: 2em, title: none)
  pagebreak()

  // 4.1 Start numbering pages
  set page(numbering: "1")
  counter(page).update(1)

  // 5. Main Content Styling & Headers
  set page(
    numbering: "1",
    header: context {
      let current_page = here().page()
      let before = query(selector(heading.where(level: 1)).before(here()))
      let after = query(selector(heading.where(level: 1)).after(here()))
      let target_heading = none

      if after.len() > 0 and after.first().location().page() == current_page {
        target_heading = after.first()
      } else if before.len() > 0 {
        target_heading = before.last()
      }

      if target_heading != none {
        grid(
          columns: (1fr, 1fr),
          align(left)[
            #text(style: "italic", size: 10pt)[
              #if target_heading.numbering != none [
                #if "A" in target_heading.numbering [
                  #let app_val = counter(heading).get().first()
                  #if app_val != none and app_val >= 1 [
                    Appendix #counter(heading).display("A"):
                  ] else [
                    Appendix
                  ]
                ] else [
                  Chapter #counter(heading).display():
                ]
              ]
              #target_heading.body
            ]
          ],
          align(right)[
            #text(size: 10pt)[#author]
          ],
        )
        v(-0.8em)
        line(length: 100%, stroke: 0.5pt + gray)
      }
    },
  )

  counter(page).update(1)
  set heading(numbering: "1.1")

  // Custom rule to force "Chapter X: Title" formatting
  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    v(1em)
    if it.body == [References] or it.body == [Bibliography] {
      text(size: 16pt, weight: "bold")[#it.body]
    } else {
      text(size: 16pt, weight: "bold")[
        Chapter #counter(heading).display(): #it.body
      ]
    }
    v(1em)
  }

  // Paragraph styling
  set par(justify: true, leading: 0.8em)

  // Image styling — black border
  show image: it => box(stroke: 0.5pt + black, it)

  // Spacing after all figures (images and tables)
  show figure: it => {
    it
    v(1em)
  }

  // Table styling - bold headers, bottom border, left-aligned content
  show table.cell.where(y: 0): strong
  set table(
    stroke: (x, y) => if y == 0 {
      (bottom: 0.7pt + black)
    },
    align: left,
  )

  body
}

// Helper for Appendices
#let appendix(body) = {
  pagebreak()
  set heading(numbering: none)
  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    v(1em)
    text(size: 16pt, weight: "bold")[#it.body]
    v(1em)
  }
  body
}

#show: project

// Chapter 1
= Introduction

//1.1
== Background

The proliferation of deepfake technology poses an increasingly serious threat to the integrity of digital media. Advances in deep generative modelling, particularly GANs, diffusion models, and neural radiance fields, have enabled the synthesis of highly realistic audio and video content that is difficult to distinguish from authentic recordings. Deepfakes have been deployed in financial fraud, political misinformation, non-consensual pornography, and impersonation attacks, with documented real-world harms that extend well beyond academic research settings @Chesney2019 @Milmo2024. The evolution of these techniques and their real-world impacts are reviewed in Sections 2.2 and 2.3.

Early deepfake detection systems focused on identifying spatial artefacts in individual frames - boundary inconsistencies, abnormal blinking, or unnatural skin texture - using convolutional neural networks trained on benchmark datasets such as FaceForensics++ @Rossler2019. These approaches achieved high accuracy within controlled settings but showed substantial performance degradation when applied to unseen generators or real-world media, suggesting that many detectors learned dataset-specific patterns rather than genuine manipulation signatures @Dolhansky2020. Detection methodologies and their generalisation limitations are reviewed in Section 2.6.

More recent deepfake systems manipulate both the audio and video streams simultaneously, introducing audio-visual mismatches - between speech content and lip motion, or between voice identity and facial appearance - that single-modality detectors are structurally unable to detect. This has driven increasing interest in multimodal detection approaches that jointly analyse both streams @Cai2024 @yi2023audiodeepfakedetectionsurvey. Multimodal detection methods are reviewed in Section 2.6.4, and the gaps that remain are identified in Section 2.7.

//1.2
== Problem Statement

Despite growing interest in multimodal deepfake detection, existing approaches exhibit three persistent limitations: simple feature concatenation rather than learned cross-modal attention, vision-centric treatment of audio as a secondary modality, and evaluation protocols that allow speaker identity leakage through random splits. These gaps, examined in detail in Section 2.7, motivate a system that applies cross-modal attention fusion, treats both modalities as equal contributors through per-modality output heads, and enforces a speaker-disjoint evaluation protocol.

_Research Questions_

This dissertation addresses the following research questions:

*RQ1:* To what extent can a deep learning model trained on a speaker-disjoint subset of AV-Deepfake1M++ distinguish real from fake audio-visual media across four manipulation types? _(Evaluated via test-set AUC, accuracy, precision, recall, and F1 — Sections 4.3–4.4; discussed in Section 5.3.1)_

*RQ2:* How does a three-head multi-task architecture contribute to per-modality interpretability in audio-visual deepfake detection? _(Evaluated via per-type head score dissociation analysis — Section 4.4; discussed in Sections 5.3.2–5.3.3)_

*RQ3:* How can a speaker-disjoint partition mitigate identity leakage and support out-of-distribution evaluation in audio-visual deepfake detection? _(Implemented via GroupShuffleSplit with zero speaker overlap — Section 3.3.2; cross-referenced against random-split limitations documented in literature — Section 2.6.5; identified as a limitation requiring future direct comparison — Section 5.5)_

//1.3
== Aim

The initial proposal for this project (CN6000, 2025) set the aim of researching and creating demo software that distinguishes between real and deepfake media files using deep learning techniques. During literature review, this aim evolved into a more technically specific target: the design, implementation, and evaluation of a multimodal audio-visual deepfake detection system that jointly analyses audio and video streams using cross-modal attention (Section 3.3.1), and assesses its performance across four manipulation types on the AV-Deepfake1M++ dataset (Section 3.3.2). The shift from a general CNN-based approach to a cross-modal transformer architecture was driven by the gaps identified in Section 2.7 and by the specific characteristics of the AV-Deepfake1M++ dataset, as documented in Sections 3.3.3–3.3.5.


//1.4
== Objectives

The initial proposal (CN6000, 2025) established six objectives covering literature review, impact analysis, dataset research, implementation, evaluation, and reflection. These were refined during the project into technically specific targets. The six objectives of this dissertation are:

1. To explore the literature on deepfake generation techniques across image, audio, and video domains, and to identify the gaps that motivate a cross-modal approach. _(Chapter 2; gaps identified in Section 2.7)_

2. To research the documented real-world impacts of deepfakes across finance, healthcare, politics, and media, and to evaluate the limitations of current detection solutions. _(Sections 2.3, 2.6; evaluated in Section 5.3)_

3. To conduct quantitative secondary analysis on the AV-Deepfake1M++ dataset, including exploratory data analysis, data cleaning, and speaker-disjoint dataset partitioning. _(Sections 3.3.2–3.3.3; evaluated in Section 5.2.1)_

4. To design and implement a Cross-Modal Transformer Fusion network that jointly encodes audio mel-spectrograms and video frame sequences, producing independent audio, video, and joint authenticity predictions. _(Sections 3.3.3–3.3.5; evaluated in Section 5.2.2)_

5. To build a fully resumable training pipeline with checkpoint recovery, two-phase encoder fine-tuning, and experiment tracking, and to evaluate the resulting models on a 100-video test set. _(Sections 3.4.2–3.4.4; Chapter 4; evaluated in Section 5.2.4–5.2.5)_

6. To develop a standalone inference system and web-based interface for classifying new videos outside the training pipeline, addressing the initial proposal's objective of demo software for real/fake distinction. _(Section 3.7; evaluated in Section 5.2.6)_

//1.5
== Significance

This work contributes a complete end-to-end multimodal deepfake detection system built on one of the largest audio-visual deepfake datasets currently available (Section 3.3.2). The system is designed with research reproducibility in mind - all hyperparameters are centralised, all randomness is seeded, and the full training state is checkpointed (Section 3.4.4) - making it straightforward to extend or replicate. The three-head output design (Section 3.3.1) provides richer diagnostic information than a single binary classifier, allowing the contribution of each modality to be assessed independently. Beyond academic contribution, the project delivers a standalone command-line inference tool and a browser-based web interface with model comparison and history tracking, making the detection capability accessible to non-technical users without requiring any knowledge of the underlying training pipeline. The empirical observation that the earliest fine-tuned model (Model 3, epoch 3) achieved better score calibration and test-set accuracy than models trained to higher training AUC (Models 2 and 4, epoch 5) - observed across multiple independent training runs within the project's computational budget - is a practical finding of interest for practitioners deploying similar architectures on speaker-disjoint data, though extended training beyond 5 epochs would be needed to confirm whether this pattern persists or reverses with additional fine-tuning.

//1.6
== Structure of the Dissertation

Chapter 2 provides a comprehensive literature review, covering the evolution of deepfakes from early visual-only manipulation to modern multimodal generation, documented real-world harms across healthcare, finance, and politics, and an assessment of image, audio, video, and multimodal detection methodologies. The chapter identifies four specific gaps - simple concatenation fusion, vision-centric bias, speaker identity leakage, and easy-example domination - that directly motivate the design of the system.

Chapter 3 describes the methodology and implementation in full, covering dataset selection and cleaning, the three architectural choices (Wav2Vec 2.0 → ResNet18, MobileNetV3 → ResNet3D-18, DiMoDif → Transformer fusion), the training protocol (Focal Loss, two-phase schedule with frozen then unfrozen encoders), the resumable checkpoint system, and the infrastructure challenges that necessitated migration from Colab to Vast.ai.

Chapter 4 presents the results of five training runs conducted across three configurations and their evaluation on a 100-video held-out test set, including per-epoch metrics, overall accuracy, per-type breakdown, and a comparative analysis of why the earliest stopping model (Model 3) outperforms later models on the test set.

Chapter 5 discusses these findings against the six stated objectives, interprets the per-type behaviour and the score calibration phenomenon, compares results against prior work, and reflects on the development process and the gap between the initial proposal and the delivered system.

Chapter 6 concludes the dissertation with a summary of contributions, five key findings, an honest assessment of limitations, directions for future work, and a personal reflection on the project.

// -----------------------------------------------------------------------------
// LITERATURE REVIEW
// -----------------------------------------------------------------------------
// Chapter 2
= Literature Review

//2.1
== Introduction

Recent advances in artificial intelligence have reshaped digital media forensics, largely due to the rapid rise of deepfakes. These synthetic videos and images, generated through deep neural architectures, can imitate human appearance and behaviour with increasing precision @He2021 @Rossler2019. Although early work treated manipulated media as a relatively contained technical issue, current developments have made detection more difficult. As @He2021 pointed out, the realism achieved by modern generative models disrupts long-standing assumptions about what constitutes trustworthy audio-visual evidence. @Rossler2019 further highlight that detection systems often are behind the pace of generative model improvements, creating a persistent asymmetry between manipulation and forensics.

//2.1.1
=== Defining Deepfake

The term deepfake combines "deep learning" and "fake," which describes synthetic media generated through deep neural models rather than conventional editing techniques @Westerlund2019. Most early systems relied on autoencoders or Generative Adversarial Networks, which made it possible to automate face swapping and expression transfer in a way that traditional manipulation tools could not achieve @Gera2018. Although digital alterations long predate deep learning, the specific phenomenon of deepfakes entered mainstream attention in late 2017 when a Reddit user released code that enabled realistic face swaps with very little technical skill, initially leading to widespread non-consensual pornography and, later, to wider use in misinformation and entertainment @Chesney2019 @Westerlund2019. In research contexts, deepfakes are generally defined by their use of deep generative models to produce content that is difficult to distinguish from authentic footage @Cai2024. Over time, beyond face swaps, current systems include audio-based voice cloning and multimodal models capable of synchronising generated speech with fabricated lip movements, which introduces additional complexity for forensic detection @yi2023audiodeepfakedetectionsurvey. Together, these developments show that deepfakes are not a single technique but a growing family of generative methods with expanding forensic implications.

//2.2
== Evolution of Deepfakes

//2.2.1
=== Phase 1: Visual Fidelity and Generative Advances

The initial phase of deepfake development focused on face swapping, using simple autoencoders and early Generative Adversarial Networks (GANs) @Rossler2019. These techniques were prone to noticeable flaws, such as affine warping errors, making them easily detectable by early forensic tools @Li2018. However, the field progressed rapidly with the adoption of improved training strategies such as the Two Time-Scale Update Rule (TTUR), which greatly improved GAN stability and convergence, resulting in more lifelike image generation @Heusel2017. As generative methods evolved, benchmark datasets such as the Deepfake Detection Challenge (DFDC) @Dolhansky2020 and Celeb-DF @Li2019 emerged. These resources mark a transition from low-grade and easily spotted fakes to high-quality videos that accurately replicated real-world visuals, posing significant challenges for detection systems @Li2019.

//2.2.2
=== Phase 2: "In-the-Wild" Adaptation and Multi-Subject Expansion

A major advancement in deepfake technology was transitioning from controlled laboratory settings to more unpredictable "in-the-wild" environments. Although earlier datasets typically included individual subjects under uniform lighting and framing, newer deepfake datasets like Wild Deepfake @Zi2021 and FFIW-10K @Zhou2021 consist of content sourced from the internet or specifically generated to endure real-world challenges such as compression and noise. During this stage, deepfake methods were also advanced, enabling the manipulation of multiple people within the same scene. For example, research by @Narayan2023 using the DF-Platter dataset showed that current techniques can alter multiple faces in a single image, even in challenging scenarios such as occlusion or poor image quality. Similarly, the FFIW-10K dataset highlighted the ongoing difficulty that forensic systems face in detecting fakes in group videos - especially when only some faces have been manipulated @Zhou2021.

//2.2.3
=== Phase 3: Multimodal and Neural Generation

The most significant recent development in deepfake technology is the move from visual-only edits to multimodal deepfakes that manipulate both audio and video. Recent studies highlight the importance of inconsistencies between sound and visuals, leading to the creation of datasets such as AV-Deepfake1M @Cai2024 and FakeAVCeleb @yi2023audiodeepfakedetectionsurvey, which feature coordinated alterations to both video and audio streams. (AV-Deepfake1M [Cai et al., 2024] is the original benchmark; AV-Deepfake1M++ [Cai et al., 2025] is the extended version used in this dissertation.) This evolution requires that detection tools now evaluate the synchronisation of lip movements with speech, rather than just looking for visual inconsistencies @yi2023audiodeepfakedetectionsurvey.

// _Summary of key deepfake benchmark datasets_

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    [_Dataset_], [_Year_], [_Modality_], [_Scale_], [_Key Characteristics_],
    [FaceForensics++ @Rossler2019],
    [2019],
    [Video (visual)],
    [~4,000 videos],
    [Multiple manipulation methods; widely used baseline],

    [DFDC @Dolhansky2020], [2020], [Video (visual)], [100,000+ clips], [3,426 paid actors; multiple generation methods],
    [Celeb-DF @Li2019], [2020], [Video (visual)], [~5,600 videos], [High-quality face swaps of celebrities],
    [DF-Platter @Narayan2023],
    [2023],
    [Video (visual)],
    [Multi-face scenes],
    [Multi-subject manipulation; occlusion handling],

    [FakeAVCeleb @yi2023audiodeepfakedetectionsurvey],
    [2023],
    [Audio-visual],
    [~20,000 clips],
    [Multiple TTS/VC methods; balanced categories],

    [AV-Deepfake1M++ @Cai2025],
    [2025],
    [Audio-visual],
    [~2M clips],
    [2,000+ speakers; four manipulation types; real-world perturbations],
  ),
  caption: [Key deepfake benchmark datasets across evolutionary phases],
)

//2.3
== Documented Real-World Impacts and High-Profile Cases

Although early research on deepfakes largely focused on improving synthesis quality and benchmarking detection accuracy, larger concern has been driven by their increasing use in real-world scenarios @Chesney2019 @Dolhansky2020 @Rossler2019.

One area of significant impact has been medical misinformation. Investigative reporting has documented numerous cases in which deepfake videos impersonating qualified doctors were used to promote false health advice and fraudulent treatments across social media platforms @ITVNews2024 @Clark2025. These videos often exploit the visual credibility of trusted medical professionals, making misinformation difficult for non-expert audiences to identify. Such cases highlight how deepfakes can undermine public trust in expert knowledge, particularly in domains where inaccurate information may cause physical harm @Chesney2019 @Rossler2019.

Deepfakes have also been employed in financial fraud and impersonation scams. A notable case involved a multinational engineering firm in which employees were deceived into authorising a large financial transfer after participating in a video call that appeared to feature senior executives but was later confirmed to be AI-generated @Milmo2024. Similar incidents have been reported globally, prompting warnings from law-enforcement agencies about the growing use of synthetic audio and video in social engineering attacks @Bragg2025. These cases demonstrate how deepfakes can amplify traditional fraud techniques by adding a convincing visual and auditory layer.

In addition to visual manipulation, audio deepfakes have demonstrated significant real-world impact, particularly in the context of fraud and impersonation @Korshunov2018 @yi2023audiodeepfakedetectionsurvey @Zhang2025. One of the first high-profile cases occurred in 2019, when attackers used a synthetic voice to mimic a company executive to authorise a fraudulent financial transfer, resulting in substantial financial losses @Stupp2019. Unlike visual deepfakes, audio-based impersonation can be deployed in real-time through phone calls or voice messages, limiting opportunities for human verification and increasing its effectiveness in social engineering scenarios @Korshunov2018. Prior research demonstrates that synthetic speech can convincingly replicate speaker identity, enabling attacks that bypass traditional speaker verification and authentication systems @Korshunov2018 @yi2023audiodeepfakedetectionsurvey. Taken together, these findings indicate that audio deepfakes pose a parallel threat to trust and identity assurance mechanisms, reinforcing the need for detection approaches that extend beyond visual analysis and address multimodal manipulation @Rossler2019 @Cai2024.

Political manipulation represents another area of concern associated with deepfakes, particularly in the context of increased geopolitical tension @Chesney2019 @Vaccari2020. Videos depicting public figures have circulated online, sometimes appearing to show officials making false statements or announcements, thereby creating confusion regarding the authenticity of political communication @Rossler2019. An example involves a manipulated video of a national leader that was briefly disseminated on social media before being removed by platform moderators @Allyn2022. Although such videos do not always achieve their intended persuasive impact, studies suggest that their rapid circulation can take advantage of periods of uncertainty and contribute to the decline of confidence in authentic news sources @Vaccari2020.

In addition to large-scale misinformation, deepfakes have caused direct personal harm. Investigations of non-consensual synthetic pornography have revealed extensive platforms dedicated to generating explicit content using the likenesses of individuals without consent, disproportionately targeting women @Moore2025. These cases underscore the ethical and legal challenges posed by deepfakes, particularly in situations where existing regulatory and legal frameworks struggle to address harms arising from fabricated yet highly realistic media @Chesney2019 @Westerlund2019.

Taken together, these examples illustrate that deepfakes are no longer a purely technical problem confined to academic datasets. Their real-world deployment exposes weaknesses in current detection systems and reinforces the need for reliable, generalisable forensic approaches capable of operating under the unpredictable conditions of online media ecosystems @Rossler2019 @Dolhansky2020.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    [_Domain_], [_Example_], [_Mechanism_], [_Source_],
    [Medical misinformation],
    [Fake doctors promoting fraudulent treatments on social media],
    [Exploits visual credibility of medical professionals],
    [@ITVNews2024 @Clark2025],

    [Financial fraud],
    [Employees deceived into authorising transfer via AI-generated executives on video call],
    [Convincing visual and auditory impersonation],
    [@Milmo2024],

    [Audio-only impersonation],
    [Synthetic voice mimicking company executive to authorise fraudulent transfer (2019)],
    [Real-time phone-based social engineering],
    [@Stupp2019],

    [Political manipulation],
    [Manipulated video of national leader briefly circulated on social media],
    [Exploits periods of uncertainty; erodes trust in authentic news],
    [@Allyn2022 @Vaccari2020],

    [Personal harm],
    [Non-consensual synthetic pornography using individuals' likenesses without consent],
    [Disproportionately targets women; legal frameworks struggle to address],
    [@Moore2025],
  ),
  caption: [Documented real-world impacts of deepfakes across five domains],
)

//2.4
== The Dual Dilemma: Creative Application Versus Malicious Threat

Deepfake technologies exemplify a dual-use dilemma: the same deep generative models that enable legitimate applications - film post-production, automated dubbing, and assistive speech technologies - can also be exploited for harmful purposes, as documented in Section 2.3. The properties that make these technologies attractive (high realism, automation, and scalability) also reduce the barrier to misuse, allowing individuals without advanced technical expertise to produce convincing synthetic media @Chesney2019 @Westerlund2019.

This dual-use nature has a direct consequence for detection research: the same architectural insights that improve a detector can be applied to make generators more convincing, creating an adversarial loop where detector improvements incentivise better generators. Chesney and Citron (2019) describe this dynamic as a "race to the bottom" in which detection capabilities struggle to keep pace with generative improvements, particularly as open-source tools proliferate @Chesney2019. From a forensic perspective, this reinforces the need for detection mechanisms that are robust to unseen generators, not just those present in training data @Rossler2019 @Dolhansky2020.

//2.5
== Deepfake Generation Techniques

Deepfake generation techniques can be broadly categorised according to the modality they manipulate: images, audio, or video. While early systems often focused on a single modality, recent advances increasingly integrate multiple streams, complicating forensic analysis. Understanding the mechanisms behind these generation techniques is essential for contextualising the strengths and limitations of current detection approaches @Rossler2019 @Dolhansky2020 @yi2023audiodeepfakedetectionsurvey @Cai2024.

//2.5.1
=== Image-Based Manipulation

Image-based deepfake generation primarily targets facial appearance through identity replacement, expression transfer, or attribute manipulation @Rossler2019. Early approaches relied on autoencoder architectures trained to map facial representations between source and target identities, enabling face swapping with relatively limited training data @Dolhansky2020. These systems commonly employed a shared encoder with identity-specific decoders, allowing facial expressions and pose information to be transferred while preserving identity-related features @Gera2018. Subsequent progress was driven by the adoption of Generative Adversarial Networks (GANs), which enabled higher resolution synthesis and more realistic texture generation @Rossler2019. Advances in training strategies and loss formulation, including stabilisation techniques such as the Two Time-Scale Update Rule, significantly reduced artefacts such as colour inconsistency and boundary distortion, making image-based deepfakes increasingly difficult to detect using handcrafted forensic cues @Heusel2017 @Li2019. Recent studies have incorporated attention mechanisms and multi-scale discriminator architectures to improve performance under common post-processing operations such as compression and resizing, which can reduce visual differences between synthetic and authentic facial images @Cai2024.

//2.5.2
=== Audio-Based Manipulation

Audio-based deepfake generation focuses on synthesising or converting speech to mimic the voice characteristics of a target speaker. Two dominant paradigms underpin these systems: text-to-speech (TTS), which generates speech directly from text, and voice conversion (VC), which transforms the vocal attributes of a source speaker into those of a target speaker while preserving linguistic content @yi2023audiodeepfakedetectionsurvey. Early systems relied on statistical parametric models, but recent advances employ deep neural architectures capable of producing highly natural and expressive speech @Korshunov2018. Modern audio deepfakes leverage end-to-end neural models, including encoder-decoder architectures and diffusion-based approaches, which allow high-fidelity voice cloning from limited reference audio @shen2023naturalspeech2latentdiffusion. These systems can accurately reproduce speaker identity, prosody, and emotional tone, making synthetic speech difficult to distinguish from genuine recordings by both humans and automated verification systems @Korshunov2018 @yi2023audiodeepfakedetectionsurvey. The increasing accessibility of such tools has lowered the barrier to misuse, particularly in impersonation and fraud scenarios, posing significant challenges for existing audio authentication mechanisms.

//2.5.3
=== Video-Based Manipulation

Video-based deepfake generation extends beyond static image manipulation by modelling temporal consistency across frames, allowing for realistic facial motion, head pose variation, and synchronisation with speech @Rossler2019. Early video deepfakes often exhibited temporal artefacts such as flickering, inconsistent lighting, or unnatural motion between consecutive frames, which could be exploited by detection systems relying on frame-to-frame inconsistency @Li2018. However, advances in spatio-temporal modelling - particularly the integration of 3D convolutional architectures and attention-based temporal aggregation - have substantially reduced the visibility of these artefacts. Recent video-based techniques integrate facial reenactment, motion transfer, and lip synchronisation models to produce coherent and temporally stable outputs @Dolhansky2020, making individual frames nearly indistinguishable from authentic footage. The emergence of multimodal generation frameworks further requires generators to maintain consistency across visual appearance, speech content, and timing simultaneously @Cai2024; a mismatch in any one of these dimensions becomes a potential forensic signal. Additionally, neural rendering approaches such as Neural Radiance Fields have been adopted to synthesise realistic talking heads with controllable viewpoints and lighting conditions, further increasing realism and making post-hoc detection more challenging @Guo2021. These developments significantly complicate forensic analysis: as per-frame visual quality improves, detectors can no longer rely on spatial artefacts alone and must increasingly depend on temporal and cross-modal inconsistencies to distinguish real from synthetic video.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    [_Modality_], [_Target_], [_Key Architectures_], [_Advances_], [_Detection Challenge_],
    [Image / Visual],
    [Faces: identity swap, expression transfer, attribute manipulation],
    [Autoencoders, GANs, attention mechanisms, multi-scale discriminators],
    [TTUR stabilisation, high-resolution texture, post-processing resilience],
    [Per-frame quality near-indistinguishable; need temporal signals],

    [Audio],
    [Speech: voice cloning, text-to-speech, voice conversion],
    [Statistical parametric models → encoder-decoder → diffusion models (NaturalSpeech 2)],
    [Zero-shot voice synthesis, accurate prosody and intonation from limited audio],
    [Over-smoothed spectra, unnatural phase; humans perform near-chance],

    [Video],
    [Motion + appearance: reenactment, lip sync, talking heads],
    [3D CNNs, spatio-temporal attention, Neural Radiance Fields (AD-NeRF)],
    [Temporal consistency, controllable viewpoint and lighting, coherent lip sync],
    [Individual frames near-perfect; must exploit temporal and multimodal mismatches],
  ),
  caption: [Deepfake generation techniques by modality],
)

//2.6
== Deepfake Detection Methodologies

Deepfake detection methodologies aim to distinguish synthetic media from authentic recordings by identifying artefacts, inconsistencies, or statistical patterns introduced during the generation process. As deepfake generation techniques have evolved, detection approaches have progressed similarly from handcrafted forensic features to data-driven deep learning models. However, despite substantial advances, existing detection systems continue to struggle with generalisation, particularly under real-world conditions @Rossler2019 @Dolhansky2020.

//2.6.1
=== Image-Based Detection Approaches

Image-based deepfake detection methods focus on identifying spatial inconsistencies within individual frames. Early approaches relied on handcrafted forensic cues, exploiting artefacts such as abnormal eye blinking, colour mismatches, and unnatural facial boundaries that were common in early face-swapping systems @Li2018. While effective against early deepfakes, these methods lacked resilience as generative models improved. The adoption of convolutional neural networks (CNNs) marked a shift towards learning discriminative features directly from data. Models such as XceptionNet, pre-trained on ImageNet and fine-tuned for deepfake classification, demonstrated state-of-the-art frame-level detection performance on FaceForensics++ @Rossler2019. Large-scale benchmarks such as FaceForensics++ showed that CNN-based detectors could achieve high accuracy within controlled datasets, often exceeding 0.95 AUC @Rossler2019. Subsequent research explored frequency-domain representations, showing that GAN-generated images exhibit characteristic spectral artefacts in the low-to-mid frequency range that can be exploited for detection, with approaches such as spectrum-based classifiers reaching over 0.97 AUC on the Celeb-DF dataset @Li2019. Despite strong in-dataset performance, image-based detectors exhibit significant degradation when evaluated on unseen datasets or under post-processing operations such as compression and resizing - XceptionNet detectors achieving 0.99 AUC on FaceForensics++ dropped to below 0.70 on Celeb-DF, and further dropped under common video compression @Dolhansky2020. This suggests that many models overfit to dataset-specific artefacts rather than learning generator-invariant features. Recent work incorporating attention mechanisms and multiscale feature extraction has shown incremental reliability improvements, but image-only detection remains insufficient against modern, high-quality deepfakes @Cai2024. The generalisation challenge common to all detection modalities is examined further in Section 2.6.5.

//2.6.2
=== Audio-Based Detection Approaches

Audio-based detection methods aim to identify synthetic speech generated through text-to-speech (TTS) or voice conversion (VC) systems. Early research adapted techniques from automatic speaker verification, using handcrafted features such as Mel frequency cepstral coefficients (MFCCs), constant-Q cepstral coefficients, and short-term phase-based representations to capture artefacts introduced by speech synthesis @Korshunov2018. These studies demonstrated that synthetic speech often contains over-smoothed spectral patterns, unnatural phase information, and anomalous pitch contours that distinguish it from genuine recordings, with Gaussian mixture model-based classifiers achieving equal error rates below 5% in controlled settings @Korshunov2018. More recent approaches employ deep neural architectures - including RawNet2, which processes raw waveform samples through SincNet-style filters, and ResNet-based models applied to mel-spectrogram representations - to learn discriminative representations directly from minimally processed audio @yi2023audiodeepfakedetectionsurvey. The ASVspoof 2019 and 2021 challenges have provided standardised benchmarks, with top-performing systems achieving tandem detection cost function values below 0.05 under logical access scenarios yet degrading substantially under physical access conditions involving codec processing and channel variation @yi2023audiodeepfakedetectionsurvey. While these models achieve strong benchmark performance, they remain vulnerable to unseen synthesis techniques and domain shifts, mirroring the limitations observed in image-based detection. Recent work has explored self-supervised speech representations - including Wav2Vec 2.0 and HuBERT embeddings - to improve generalisation, though performance gaps remain when testing across TTS and VC systems not seen during training @yi2023audiodeepfakedetectionsurvey. Furthermore, empirical studies show that human listeners struggle to reliably distinguish real and synthetic speech, with accuracy rates only marginally above chance (approximately 60–65%) for high-quality voice clones, reinforcing the need for automated audio deepfake detection systems @yi2023audiodeepfakedetectionsurvey. The cross-dataset generalisation challenge shared across all detection modalities is examined in Section 2.6.5.

//2.6.3
=== Video-Based Detection Approaches

Video-based detection approaches extend image-level analysis by incorporating temporal information across frames. Early methods exploited temporal artefacts such as inconsistent head motion, unnatural blinking patterns, and frame-to-frame discontinuities, using optical flow analysis and recurrent architectures such as convolutional LSTM networks @Li2018. These cues enabled sequence-level classification through temporal aggregation - Li et al. (2018) demonstrated that detecting abnormal eye blinking across frame sequences could identify early deepfakes with over 0.95 AUC on the UADFV dataset @Li2018. As generative models improved, many temporal artefacts became less pronounced. Contemporary video detectors therefore rely on spatio-temporal architectures, including 3D convolutional networks such as I3D and ResNet3D pretrained on Kinetics, and attention-based temporal modelling, to capture subtle motion inconsistencies @Rossler2019 @Dolhansky2020. The DFDC top submissions demonstrated that ensemble approaches combining 3D CNNs with face-crop preprocessing and test-time augmentation could reach log-loss scores below 0.20, significantly outperforming frame-only baselines @Dolhansky2020. More recent work has explored Vision Transformer architectures and contrastive learning frameworks for video-level deepfake detection, with some approaches reporting cross-dataset AUC improvements of 0.05–0.10 compared to 3D CNN baselines @Cai2024. Although these methods outperform frame-based detectors in controlled settings, they remain sensitive to compression, frame rate variation, and domain shifts common in real-world video content @Dolhansky2020. The generalisation limitation common to all detection modalities is considered in Section 2.6.5.

//2.6.4
=== Multimodal Detection Approaches

The emergence of deepfakes that manipulate both audio and visual streams has driven increasing interest in multimodal detection approaches. These methods jointly analyse facial motion, lip synchronisation, and speech content to identify cross-modal inconsistencies that are difficult for generative models to reproduce perfectly @yi2023audiodeepfakedetectionsurvey. Datasets such as FakeAVCeleb and AV-Deepfake1M have enabled systematic evaluation of multimodal detectors under coordinated manipulation scenarios @Cai2024. Building on this trend, @Cai2025 introduced AV-Deepfake1M++, an extension containing approximately 2 million video clips across over 2,000 speakers, with four manipulation categories - `real`, `audio_modified`, `visual_modified`, and `both_modified` - reflecting independent manipulation of each stream rather than a single holistic transformation. This structure makes it suitable for evaluating detectors that decompose the detection problem into per-modality authenticity predictions, as each manipulation type corresponds to a known combination of stream-level tampering. The dataset includes diverse audio perturbations (codec compression, background noise) and video post-processing (H.264 transcoding, resolution scaling), simulating the real-world conditions under which deployed detectors must operate @Cai2025. Multimodal detection systems generally demonstrate improved generalisation compared to unimodal approaches, particularly in cross-dataset evaluations @Cai2024. However, their performance remains constrained by dataset bias, limited availability of synchronised training data, and increased computational complexity, which may hinder real-time deployment @Cai2024 @Cai2025.

A further challenge in training multimodal detectors is easy-example domination during optimisation. When straightforward, high-contrast artefacts account for a large proportion of gradient updates, the model converges to coarse decision boundaries and fails to learn the subtle cross-modal inconsistencies that distinguish modern, high-quality deepfakes. @lin2018focallossdenseobject introduced Focal Loss as a principled solution in the context of dense object detection: by downweighting well-classified examples through a modulating factor, training is concentrated on hard, ambiguous samples. Focal Loss has since been adopted in audio-visual deepfake detection to address easy-example domination, consistent with its original motivation of mitigating gradient dominance from well-classified samples @lin2018focallossdenseobject.

Multi-task learning offers another strategy for improving multimodal detector reliability. Rather than training a single output head on a joint real/fake label, multi-task architectures produce separate predictions for each modality - one for audio authenticity, one for video authenticity, and one joint verdict - training them simultaneously with a shared loss @Cai2024. @Cai2024 demonstrated that this design forces each head to specialise on its respective stream while the joint head captures their interaction, yielding a more interpretable and diagnostically useful model that independently quantifies the contribution of each modality. Such architectures are particularly well-suited to datasets where manipulation types affect each stream independently, as seen in the AV-Deepfake1M framework @Cai2025.

Several architectural choices commonly employed in prior multimodal detection work are worth reviewing. For video feature extraction, @tran2018closerlookspatiotemporalconvolutions demonstrated that 3D convolutional networks - specifically ResNet3D-18, pretrained on Kinetics-400 - jointly process spatial and temporal dimensions of video, capturing spatio-temporal artefacts that single-frame analysis cannot detect. This approach has been adopted in deepfake detection work targeting full-face video inputs, where temporal inconsistencies between frames provide additional forensic signals @Dolhansky2020. For audio feature extraction, ImageNet-pretrained 2D convolutional networks such as ResNet18 have been applied to mel-spectrogram representations of audio to identify spectral artefacts introduced by speech synthesis or voice conversion systems @Korshunov2018. This spectrogram-based approach is noted for its simplicity and robustness across diverse audio codecs and recording conditions. For cross-modal fusion, Transformer-based architectures have been explored in recent work, using self-attention to learn fine-grained dependencies between audio and visual streams, with a learnable [CLS] token aggregating both representations into a unified embedding @Cai2024. Compared to simple feature concatenation, this enables cross-modal inconsistencies - such as mismatches between lip synchronisation and speech content - to be captured during feature learning. Collectively, these architectural components - 3D CNNs for video, 2D CNNs on spectrograms for audio, and Transformer-based cross-modal fusion - form the basis for contemporary multimodal detectors and directly inform the system architecture developed in this dissertation (Chapter 3).

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    [_Modality_], [_Approach_], [_Named Architectures_], [_Best Reported Metric_], [_Key Limitation_],
    [Image],
    [CNN on frames; frequency-domain analysis],
    [XceptionNet, spectrum classifiers],
    [$>$ 0.95 AUC (in-dataset)],
    [Drops to $<$ 0.70 on unseen datasets; compression-sensitive],

    [Audio],
    [MFCC + GMM → deep neural → raw waveform → self-supervised],
    [RawNet2, ResNet on spectrograms, Wav2Vec 2.0, HuBERT],
    [t-DCF $<$ 0.05 (logical access)],
    [Degrades under physical access; generator-specific artefacts],

    [Video],
    [Optical flow + LSTM → 3D CNN → Vision Transformer],
    [I3D, ResNet3D, contrastive learning],
    [Log-loss $<$ 0.20 (DFDC)],
    [Compression, frame rate variation; +0.05–0.10 AUC with ViT],

    [Multimodal],
    [Feature concatenation → Transformer attention → multi-task heads],
    [Cross-modal fusion, DiMoDif, 3-head architectures],
    [0.85–0.99 AUC (benchmarks)],
    [Dataset bias, limited sync training data, computational cost],
  ),
  caption: [Deepfake detection approaches by modality],
)

//2.6.5
=== Generalisation and Cross-Dataset Performance

A central challenge across all detection modalities is poor generalisation to unseen generators and real-world media conditions. Numerous studies report substantial performance drops when detectors trained on one dataset are evaluated on others generated using different synthesis methods @Rossler2019 @Dolhansky2020. FaceForensics++ detectors achieve high AUC within that dataset but degrade sharply on DFDC or Celeb-DF, suggesting they learn dataset-specific artefacts rather than generalisable manipulation signatures @Dolhansky2020 @Li2019. Audio deepfake detectors similarly show performance drops when evaluated against unseen TTS or VC systems, indicating that spectral artefacts are generator-specific @Korshunov2018. This generalisation problem is particularly acute for deployed systems, which must handle media created by generators not present in any training set.

Several factors contribute to this fragility. First, many datasets used for training contain systematic biases - for example, specific face-swapping algorithms that leave detectable artefacts, or specific audio codec configurations that produce characteristic spectral patterns. Models trained on these datasets learn to detect the artefact rather than the underlying manipulation. Second, post-processing operations common in real-world media - transcoding, motion interpolation, noise reduction, audio resampling - can remove or alter the very artefacts that detectors rely on, further degrading performance outside the lab @Dolhansky2020. Third, identity leakage in evaluation protocols inflates performance figures: when the same speaker appears in both training and validation sets, the model may exploit speaker recognition rather than learning manipulation-specific features, producing results that do not generalise to truly unseen identities @Rossler2019.

Cross-dataset evaluation has therefore become a critical benchmark for assessing practical effectiveness. Studies that evaluate detectors across multiple datasets consistently find large AUC drops - often 0.10–0.20 - compared to within-dataset results @Cai2024 @yi2023audiodeepfakedetectionsurvey. AUC (Area Under the ROC Curve) is the standard primary metric in this literature because it is threshold-independent: it measures a detector's ability to rank samples by authenticity across all possible decision thresholds, rather than at a single arbitrarily chosen cutoff. This is particularly important under class imbalance, where accuracy at a fixed threshold of 0.5 can be misleading - a model that predicts FAKE for all inputs achieves 75% accuracy on a 3:1 fake-to-real split @Rossler2019 @Dolhansky2020. Supporting metrics such as precision, recall, and F1 score are commonly reported at the optimal threshold identified via AUC maximisation to provide a fuller picture of practical deployment performance @Cai2024. Recent work has attempted to address the generalisation problem through domain adaptation techniques, self-supervised pre-training on diverse data, and contrastive learning frameworks that encourage generator-agnostic representations @Cai2024. However, no current approach has fully solved the generalisation problem, and improving generalisation to unseen generators remains an open research challenge. This underscores the importance of the speaker-disjoint evaluation protocol adopted in this dissertation (Section 3.3.2) and the cross-dataset considerations identified as a limitation in Section 5.5.

//2.7
== Gaps in the Literature

Prior work shows that unimodal deepfake detectors often fail to generalise to unseen data, motivating increased interest in audio-visual detection methods @Rossler2019 @Dolhansky2020 @Cai2024. However, as reviewed in Sections 2.6.1 through 2.6.4, existing multimodal approaches exhibit several specific unresolved limitations that together define the gap this dissertation fills.

Gap 1 - Simple fusion without cross-modal attention: Most audio-visual detectors concatenate independently learned audio and visual representations rather than allowing the modalities to attend to each other during feature learning @yi2023audiodeepfakedetectionsurvey. This limits sensitivity to cross-modal inconsistencies such as mismatches between lip synchronisation and speech content.

Gap 2 - Vision-centric treatment of audio: Many multimodal systems treat audio as a secondary or supplementary modality, with the video encoder dominating the joint decision @Rossler2019 @Cai2024. As a result, the independent contribution of audio information is rarely quantified through controlled ablation or per-modality metric reporting.

Gap 3 - Speaker identity leakage in evaluation: The widespread use of random train/validation splits means that the same speaker can appear in both sets, allowing models to exploit face or voice recognition rather than learning genuine manipulation artefacts @Rossler2019. This produces inflated accuracy figures that do not reflect true generalisation to unseen identities.

Gap 4 - Easy-example domination: Standard loss functions allow the model to converge on coarse decision boundaries by exploiting high-confidence examples, without learning to distinguish the subtle manipulations that characterise modern high-quality deepfakes. While the dataset is balanced at the category level (Section 3.3.3), easy examples with obvious artefacts dominate gradient updates under standard loss functions like BCE, preventing learning of fine-grained manipulation boundaries.

These four gaps collectively motivate the design of the system developed in this dissertation: a Cross-Modal Transformer Fusion network with a three-head output architecture (addressing Gap 2), trained with Focal Loss on a speaker-disjoint partition (addressing Gaps 3 and 4), using self-attention cross-modal fusion (addressing Gap 1), as detailed in Chapter 3.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    [_Gap_], [_Problem_], [_Design Response_], [_Implementation_],
    [1: Simple concatenation fusion],
    [Cannot capture cross-modal inconsistencies such as lip-speech mismatches],
    [Transformer Encoder with self-attention],
    [2-layer Encoder, 8 heads, [CLS] token, GELU - Section 3.3.5],

    [2: Vision-centric bias],
    [Audio treated as secondary; per-modality contribution unquantified],
    [Three-head multi-task architecture],
    [Independent audio, video, and joint heads with sigmoid output - Section 3.3.1],

    [3: Speaker identity leakage],
    [Random splits allow same speaker in train and val; inflated metrics],
    [Speaker-disjoint partition],
    [GroupShuffleSplit on speaker ID - Section 3.3.2],

    [4: Easy-example domination],
    [Easy examples dominate gradients; subtle fakes missed],
    [Focal Loss replacing BCE],
    [γ = 2.0, α = 0.25 - Section 3.4.1],
  ),
  caption: [Literature gaps and corresponding design responses],
)

//2.8
== Conclusion

This chapter reviewed deepfake generation and detection techniques across image, audio, video, and multimodal domains. Across modalities, a consistent limitation is reduced performance outside controlled benchmarks, particularly when detectors encounter unseen generators or real-world media conditions @Rossler2019 @Dolhansky2020 @Cai2024. Four specific gaps were identified - simple concatenation fusion, vision-centric bias, speaker identity leakage, and easy-example domination - each motivating a corresponding design choice in the system developed in this dissertation, as detailed in Chapter 3.

// -----------------------------------------------------------------------------
// METHODOLOGY
// -----------------------------------------------------------------------------
// Chapter 3
= Methodology and Implementation

// 3.1
== Introduction

This chapter outlines the approach taken to design, develop, and assess a deepfake detection system for audio-visual content using the AV-Deepfake1M++ dataset @Cai2025. The aim is to meet the research goals systematically while respecting hardware, storage, and computational constraints.

The work targets content-driven deepfakes where a real identity is preserved but the spoken content is altered via audio synthesis and lip synchronisation, alongside visual manipulations. Such changes are subtle and localised, making unimodal detection unreliable @Cai2024 @Korshunov2018. A multimodal strategy is therefore adopted that exploits inconsistencies between the audio stream and the visual stream.

This chapter also documents the architectural decisions and the practical reasons that shaped each choice. The initial proposal outlined a system built around Wav2Vec 2.0 for audio feature extraction, MobileNetV3 for visual feature extraction, and a DiMoDif fusion module. During implementation, each of these components was evaluated against the specific characteristics of the AV-Deepfake1M++ dataset and the available training infrastructure. The rationale for the final architecture is discussed alongside each component.

// 3.2
== Research and Development Approach

//3.2.1
=== Research Methodology

This study adopts a quantitative research methodology based on secondary analysis of the AV-Deepfake1M++ dataset @Cai2025. Quantitative methods are appropriate here because the primary research questions concern measurable classification performance AUC, accuracy, precision, recall, and F1_score across a large, labelled corpus with well-defined ground truth. Using an existing benchmark dataset avoids the cost and ethical complexity of collecting a large-scale audio-visual corpus, which would not have been feasible within the scope or time frame of this project.

Development followed an agile, kanban-based incremental methodology, organised into iterative sprints targeting one functional layer of the pipeline at a time. A Kanban board tracked task status; project milestones were tracked using a Gantt chart (Appendix J). When infrastructure failures described in Section 3.5.1 caused schedule slippage during implementation phase I, the Gantt chart made it straightforward to identify which downstream tasks needed re-scoping.

//3.2.2
=== Implementation Methodology

Each component - data loading, preprocessing, feature extraction, model integration, and evaluation - was implemented and verified independently on a small video subset before integration into the full pipeline. All hyperparameters and derived constants were centralised in a single configuration file (`config.py`), ensuring that no values were hardcoded. Any change to extraction parameters, model dimensions, or training schedules could be made in one place and propagated automatically, serving as a live record of exact settings for each training run.

//3.3
== System Architecture and Implementation

//3.3.1
=== Overview of the Proposed System

The primary objective of the project is to maximise classification accuracy at the clip level across all four manipulation types: `real`, `audio_modified`, `visual_modified`, and `both_modified`. Although the AV-Deepfake1M++ dataset provides temporal localisation annotations, frame-level modelling was not adopted in this implementation due to the high computational cost and storage demands it would introduce. The system therefore operates at the video-clip level.

The implemented system is a Cross-Modal Transformer Fusion network that combines two pretrained modality-specific encoders with a learned cross-modal attention mechanism, and produces three simultaneous binary predictions per video clip: whether the audio is authentic, whether the video is authentic, and an overall joint verdict. This multi-head design, rather than a single output, allows the model to independently specialise each head on the evidence available from each modality, while the joint head captures cross-modal interaction.

_*Classification Framing: Binary Decomposition of a Four-Category Problem*_

A key design decision that requires explicit justification is the choice to frame the detection task as three concurrent binary classification problems rather than a single four-class classification problem. The AV-Deepfake1M++ dataset assigns one of four manipulation labels to each video clip - `real`, `audio_modified`, `visual_modified`, and `both_modified` - which might suggest that a four-class softmax output would be the natural choice. The system instead decomposes this into three independent binary questions, each answered by a dedicated sigmoid output head:

// - _Audio head:_ Is the audio stream authentic? _(binary: 1 = real, 0 = fake)_
// - _Video head:_ Is the video stream authentic? _(binary: 1 = real, 0 = fake)_
// - _Joint head:_ Is the clip overall authentic? _(binary: 1 = real, 0 = fake)_

// The four dataset categories map onto binary training labels for each head as follows:

#figure(
  table(
    columns: (auto, auto, auto, auto),
    [_Dataset Type_], [_Audio Label_], [_Video Label_], [_Joint Label_],
    [`real`], [1 (real)], [1 (real)], [1 (real)],
    [`audio_modified`], [0 (fake)], [1 (real)], [0 (fake)],
    [`visual_modified`], [1 (real)], [0 (fake)], [0 (fake)],
    [`both_modified`], [0 (fake)], [0 (fake)], [0 (fake)],
  ),
  caption: [Mapping of dataset manipulation types to per-head binary training labels],
)

The model never receives the category label string directly as a prediction target. It receives three integer labels per sample and is trained to minimise Focal Loss jointly across all three heads simultaneously (Section 3.4.1). The four-category structure is not predicted explicitly it _emerges implicitly_ as a by-product of the three binary heads: a clip assigned low audio score and high video score is effectively identified as `audio_modified`, and so on.

This framing is preferred over a four-class softmax for three reasons. First, it provides modality-level interpretability: the audio and video heads each produce an independent authenticity estimate, allowing the contribution of each modality to be quantified separately. A four-class softmax would conflate this information into a single opaque output. Second, the binary decomposition aligns with the generative structure of the dataset - each manipulation type is defined by a combination of independent manipulations applied to each stream, rather than by a distinct holistic transformation. Third, binary classification with Focal Loss is a well-understood and stable training objective; a four-class softmax would require careful per-class weight balancing and would not naturally provide the modality-specific diagnostic capability that the three-head design affords.

The primary evaluation metric for each head is AUC (Area Under the ROC Curve), which is calculated independently for each binary output. AUC is threshold-independent and is not adversely affected by the class imbalance present in the test set (25 real, 75 fake). Accuracy at the optimal decision threshold (determined by maximising F1), precision, recall, and F1 score are reported as supporting metrics. The test set imbalance (3:1 fake-to-real ratio) means that raw accuracy computed at a fixed 0.5 threshold would overstate true performance - a model that predicts FAKE for all inputs would achieve 75% accuracy - and AUC is therefore the more reliable primary measure. This is consistent with standard practice in deepfake detection evaluation @Rossler2019 @Cai2024.

The system constitutes a deep learning pipeline: ResNet3D-18 (pretrained on Kinetics-400) for video, ResNet18 (pretrained on ImageNet) for audio via mel-spectrograms, and a two-layer Transformer Encoder for cross-modal fusion - over thirty layers trained jointly with backpropagation and the AdamW optimiser. Transfer learning is applied to both encoders, with a two-phase fine-tuning schedule (Section 3.4.2) that progressively adapts pretrained weights to the deepfake detection task.

#figure(
  image("figures/architecture.png", width: 80%),
  caption: [Cross-Modal Transformer Fusion system architecture: ResNet3D-18 video encoder and ResNet18 audio encoder feed into a two-layer Transformer Encoder with [CLS] token, producing three independent sigmoid classification heads for audio, video, and joint authenticity predictions.],
)

//3.3.2
=== Dataset Selection and Subset Strategy

The AV-Deepfake1M++ dataset is distributed as approximately 1.4 TB of split compressed archives, which makes full local storage and processing infeasible within the available development environment. The extracted video files require substantially more disk space beyond the archive size. The validation split, comprising 77,326 video clips, was used exclusively throughout this work. Of these, 68,851 videos (89%) were confirmed present on disk following extraction; the remaining 8,264 were absent at the on-disk stage (in addition to 211 dropped at the metadata stage, for a total of 8,475 excluded across both cleaning stages) and were excluded from all experiments. The four manipulation categories in the validation split are balanced, each containing between 16,848 and 18,037 videos, as shown in the table below.

// _Post-cleaning dataset composition_

#figure(
  table(
    columns: (auto, auto, auto),
    [_Category_], [_Description_], [_Count (post-cleaning)_],
    [`real`], [Unmodified audio and video], [18,037],
    [`audio_modified`], [Voice replaced or cloned; video untouched], [16,848],
    [`visual_modified`], [Face swapped or reenacted; audio untouched], [17,020],
    [`both_modified`], [Both audio and video manipulated], [16,946],
  ),
  caption: [Post-cleaning dataset composition (68,851 videos). Pre-cleaning per-category distribution (77,326 videos) is reported in Section 3.3.3.],
)

Each entry in the accompanying metadata file (`val_metadata.json`) records the file path, manipulation type, frame counts, and the temporal coordinates of any manipulated segments (`fake_segments`) as `[start_sec, end_sec]` pairs.

A speaker-disjoint train/validation split was enforced using GroupShuffleSplit: all videos from a given speaker were assigned exclusively to either the training or validation set, ensuring zero speaker overlap between subsets. This prevents the model from exploiting face or voice recognition and ensures that reported metrics reflect generalisation to entirely unseen identities.

#figure(
  image("figures/speaker_split.png", width: 70%),
  caption: [Speaker-disjoint train/validation partition: 1,468 speakers (80%) assigned to training, 367 speakers (20%) held out for validation, with zero overlap. Implemented via GroupShuffleSplit on speaker ID to prevent identity leakage.],
)

//3.3.3
=== Exploratory Data Analysis and Data Cleaning

Before any feature extraction or model training, a dedicated exploratory data analysis (EDA) step was carried out using `analyze_data.py` against the full `val_metadata.json` metadata file. This standalone analysis - run separately from the main training pipeline - characterised the dataset composition, identified data quality issues, and informed the cleaning decisions documented below. All statistics reported in this section are derived directly from `val_metadata.json` (77,326 entries) and were verified by running the analysis script locally.

_*Dataset Composition*_

The metadata file catalogues 77,326 video clips across 1,835 unique speakers. The modification type distribution is near-uniform across all four categories (see the table and figure below), confirming that the dataset does not exhibit class imbalance at the category level.

#figure(
  table(
    columns: (auto, auto, auto),
    [_Modification Type_], [_Count_], [_Percentage_],
    [`real`], [20,220], [26.1%],
    [`visual_modified`], [19,099], [24.7%],
    [`both_modified`], [19,069], [24.7%],
    [`audio_modified`], [18,938], [24.5%],
  ),
  caption: [Metadata modification type distribution (pre-cleaning)],
)

#figure(
  image("analysis/modification_type_distribution.png", width: 80%),
  caption: [Modification type distribution — the table values above represent the authoritative counts derived directly from val_metadata.json.],
)

Speaker coverage is substantial but uneven: the most prolific speaker (`id02760`) contributes 1,417 clips, while the least represented (`id08916`) contributes only 1. The mean videos per speaker is 42.1, with a median of 20, indicating a right-skewed distribution that is common in datasets drawn from real-world video collections. Of the 1,835 speakers, 1,336 (72.8%) have clips across all four modification types, providing good cross-type coverage for the majority of identities.

_*Data Cleaning - Stage 1: Metadata-Level Filter*_

The first cleaning step discards videos whose metadata indicates they cannot yield valid features. The filter criterion applied in `data_utils.py` is:

```python
df_valid = df[(df['audio_frames'] > 0) & (df['video_frames'] > 0)].copy()
```

Running this filter against `val_metadata.json` produced the following counts:

#figure(
  table(
    columns: (auto, auto),
    [_Condition_], [_Count_],
    [Total metadata entries], [77,326],
    [Videos with zero audio frames], [211],
    [Videos with zero video frames], [0],
    [Total dropped (either condition)], [_211_],
    [Remaining after Stage 1], [_77,115_],
  ),
  caption: [Stage 1 cleaning - metadata-level filter],
)

The 211 videos with zero audio frames (0.27% of the dataset) represent clips where the audio stream was absent, empty, or could not be decoded to produce any samples by the metadata extraction step. Retaining these videos and attempting to extract audio features would yield zero tensors that carry no discriminative information, while simultaneously risking NaN propagation during normalisation. They were therefore dropped unconditionally at the metadata stage, before any file I/O was attempted.

No videos had zero video frames in the metadata; all 77,326 entries had at least one recorded video frame according to the metadata. All zero-frame exclusions are therefore attributable to audio quality alone at this stage.

_*Data Cleaning - Stage 2: On-Disk Availability Filter*_

The second cleaning step reflects the reality that not all videos catalogued in the metadata were successfully extracted from the compressed split archives onto the Vast.ai instance disk. Of the 77,115 metadata-valid videos, only _68,851_ were confirmed present on disk when the feature extraction pipeline was run. The remaining _8,264_ files were absent - either because the corresponding archive chunk was not transferred, the extraction was interrupted, or the video file was corrupted beyond recovery during decompression.

#figure(
  table(
    columns: (auto, auto),
    [_Condition_], [_Count_],
    [Metadata-valid videos (post Stage 1)], [77,115],
    [Confirmed present on disk], [_68,851_ (89.3%)],
    [Missing or corrupted on disk], [_8,264_ (10.7%)],
    [Final usable dataset], [_68,851_],
  ),
  caption: [Stage 2 cleaning - on-disk availability],
)

These missing files were tracked in a `*_failed.json` manifest by the extraction pipeline and were skipped on all subsequent runs. No attempt was made to impute or substitute features for absent videos. The 68,851 confirmed files represent the entirety of data used for training and evaluation in this project.

_*Runtime Fallbacks for Corrupted Files*_

During feature extraction, a subset of the 68,851 nominally present files could not be fully decoded at runtime, typically due to codec issues, truncated video containers, or MP4 files with broken audio tracks. The pipeline implements the following defensive fallbacks to prevent crashes:

#figure(
  table(
    columns: (auto, auto),
    [_Failure Scenario_], [_Fallback_],
    [Audio extraction fails entirely], [Returns zero tensor `(1, 128, 63)`],
    [Video frame read fails], [Skips the video; records in failed manifest],
    [Saved `.pt` file is corrupted], [Returns zero tensors with sentinel labels `[-1, −1]`],
    [Audio shape mismatch], [Pads or truncates to fixed shape `(1, 128, 63)`],
  ),
  caption: [Runtime fallbacks for corrupted files],
)

Samples loaded with sentinel labels `[-1, -1]` are excluded from loss computation during training via a masking operation, ensuring that corrupted files do not contribute incorrect gradient signals.

_*Key EDA Observations*_

Beyond data cleaning, the EDA revealed several structural properties of the dataset that informed design decisions:

1. _Short, localised fake segments._ The 57,106 fake videos contain a mean of 1.4 fake segments each (maximum 4), with a mean segment duration of 0.33 seconds (median 0.30 seconds) (Figure 3.2). This finding confirmed that a fixed two-second analysis window captures at least one, and typically all, fake segments in the majority of clips. The `fake_segments` annotations were used during evaluation to select informative windows in the multi-window inference pipeline; the training dataloader samples a central two-second window uniformly for all videos.

// Figure 2
#figure(
  image("analysis/fake_segment_analysis.png", width: 80%),
  caption: [Fake segment analysis],
)

2. _High speaker diversity._ With 1,835 speakers, a speaker-disjoint train/validation split (described further in §3.3.2 above and §3.4.2 of the pipeline) is both feasible and necessary - there are sufficient speakers to populate both subsets without identity overlap.

3. _Balanced class distribution._ Near-uniform distribution across all four modification types (24.5–26.1%) means that class imbalance is not a significant concern at the category level, though the real-vs-fake imbalance (26.1% vs 73.9%) motivated the use of the $α$ parameter in Focal Loss (Section 3.4.1) to provide a mild class balance correction.

// 3.3.4
=== Audio Feature Extraction Module

_*Initial Proposal*_

The initial proposal specified Wav2Vec 2.0 @baevski2020 as the audio feature extraction backbone. @baevski2020 is a self-supervised speech model that learns contextual speech representations directly from raw waveforms, capturing prosodic and phonemic information that handcrafted features such as MFCCs would miss. Its contextual embeddings were expected to be particularly sensitive to the subtle artefacts introduced by modern voice cloning and text-to-speech synthesis systems.

_*Change Applied and Rationale*_

During implementation, Wav2Vec 2.0 was replaced with a ResNet18 @he2015deepresiduallearningimage backbone pretrained on ImageNet, applied to mel-spectrogram representations of the audio signal.

This change was driven by three practical constraints. First, Wav2Vec 2.0 produces variable-length sequence outputs whose length depends on the duration of the input audio. Integrating this with a fixed-dimension video feature vector inside a Transformer fusion module required either temporal pooling - which discards the temporal structure that motivates using Wav2Vec in the first place - or padding and masking strategies that added architectural complexity without clear benefit at the clip level. Second, the Wav2Vec 2.0 large model imposes a substantial VRAM footprint. When combined with ResNet3D-18 for video encoding and a Transformer fusion module, the total memory requirement exceeded the GPU capacity of the available training hardware during early batch-size experiments. Third, torchaudio's robust FFmpeg-native audio loading pipeline, combined with the MelSpectrogram and AmplitudeToDB transforms, provided a stable and crash-free audio extraction pathway that was compatible with the corrupted and non-standard MP4 files common in the dataset. Earlier attempts using librosa-based loading produced frequent `PySoundFile failed` warnings and occasional crashes on corrupted video files, which disrupted the parallel extraction pipeline.

The mel-spectrogram representation converts raw audio into a 2D time-frequency image of shape `(1, 128, 63)`, which can be processed by any image CNN with a simple modification to the first convolutional layer to accept a single channel rather than three. Voice cloning artefacts, unnatural harmonic structures, and audio splice boundaries all manifest as visible patterns in the mel-spectrogram that a CNN trained on general image representations can detect. This approach is consistent with established practice in audio classification research, where ImageNet-pretrained CNNs applied to spectrograms have repeatedly matched or outperformed bespoke audio architectures on downstream tasks.

Audio was sampled at 16,000 Hz. A 1024-point FFT with hop length 512 and 128 mel frequency bins was applied, yielding the `(1, 128, 63)` spectrogram. Amplitude was converted to decibels with an 80 dB dynamic range, and each spectrogram was per-sample normalised to zero mean and unit variance. All extraction parameters were read from `config.py` rather than hardcoded, ensuring that changing the FFT window or mel bin count propagated automatically to the derived `target_t` dimension.

During training, SpecAugment was applied to mel-spectrogram tensors to improve robustness to audio degradation: frequency masking zeroes out a random band of mel bins (up to 20 bins), and time masking zeroes out a random band of time steps (up to 15 steps). This simulates partial frequency loss and temporal dropout common in compressed or noisy real-world audio.

#figure(
  image("figures/mel_spectrogram_comparison.png", width: 100%),
  caption: [Mel-spectrogram comparison: real audio (left) showing natural harmonic structure vs. synthetic audio (right) showing over-smoothed spectral patterns characteristic of voice cloning. Parameters: 16,000 Hz sample rate, 1024-point FFT, 128 mel bins, 63 time frames (2 seconds).],
)

// 3.3.5
=== Visual Feature Extraction Module

_*Initial Proposal*_

The initial proposal specified MobileNetV3 @Howard2019 as the visual backbone, focused on the mouth region of interest (ROI). Facial landmark detection was to be used to crop the lip region prior to encoding. MobileNetV3 was selected for its lightweight architecture, which offers a practical balance between efficiency and representational capacity when frame-level spatial features are required.

_*Change Applied and Rationale*_

During implementation, MobileNetV3 applied to lip-region crops was replaced with ResNet3D-18 @tran2018closerlookspatiotemporalconvolutions pretrained on Kinetics-400 @kay2017kineticshumanactionvideo, applied to full 224×224 frames across the full temporal window of 50 frames.

There were two primary motivations for this change. First, the initial proposal's mouth-ROI approach assumed that deepfake artefacts are localised to the lip region. However, the `visual_modified` and `both_modified` categories in AV-Deepfake1M++ encompass a range of face-swap and reenactment techniques where artefacts are distributed across the entire face - including skin texture boundaries, hairline artefacts, and blending inconsistencies at the face perimeter - none of which are captured by a lip-region crop. Restricting the visual field to the mouth region would systematically discard evidence that the visual encoder needs to detect these manipulation types.

Second, applying MobileNetV3 frame-by-frame produces independent spatial features for each frame with no temporal context. Deepfake videos frequently exhibit temporal inconsistencies - unnatural head motion, inconsistent blinking, or jittery texture between consecutive frames - which are not visible in any single frame but emerge clearly when multiple frames are considered jointly. ResNet3D-18's 3D convolutional filters jointly convolve the spatial and temporal dimensions, capturing exactly these cross-frame patterns. As @Dolhansky2020 and @Rossler2019 both note, temporal modelling is essential for robust video-level detection, particularly as generative models improve and per-frame artefacts become less pronounced.

Fifty frames were sampled from each two-second window at 25 frames per second. Frames were resized to 224×224 pixels and normalised using ImageNet statistics. Data augmentation during training - random horizontal flipping and brightness and contrast jitter - was applied before ImageNet normalisation to keep pixel values in the valid [0, 1] range prior to the normalisation step. The resulting video tensor has shape `(50, 3, 224, 224)`.

// 3.3.6
=== Cross-Modal Fusion and Classification

_*Initial Proposal*_

The initial proposal specified the DiMoDif architecture @Cai2025, which is designed specifically for the AV-Deepfake1M++ dataset. DiMoDif models fine-grained phoneme-to-viseme alignment between audio and visual streams, directly targeting content-driven deepfakes where speech content is synthesised and lip motion is generated to match. The temporal boundary detection components of DiMoDif were to be excluded in favour of video-level classification.

_*Change Applied and Rationale*_

DiMoDif was replaced with a custom two-layer Transformer Encoder fusion module with a learnable [CLS] token, operating on projected audio and video feature vectors.

The primary reason for this departure was reproducibility and implementation complexity. DiMoDif requires precise temporal alignment between audio phoneme sequences and per-frame visual features, which presupposes that both streams can be reliably aligned at the sub-frame level. Achieving this alignment robustly across the full diversity of video codecs, frame rates, and audio sampling rates present in the dataset would have required significant additional preprocessing infrastructure. Given that the temporal boundary detection components of DiMoDif were explicitly excluded from the proposal, retaining only the fusion mechanism while discarding its motivating temporal alignment would have reduced DiMoDif to a cross-modal attention module - which is precisely what the implemented Transformer fusion provides, with fewer implementation dependencies.

The Transformer Encoder fusion module receives the 256-dimensional feature vectors produced by both encoders, projects each to 512 dimensions, and forms a three-token input sequence `[CLS, video, audio]` augmented with learned positional embeddings. Two layers of multi-head self-attention with eight heads, GELU activation, and pre-norm layer normalisation allow the video and audio tokens to attend to each other. This attention mechanism captures cross-modal inconsistencies - for example, audio spectrogram patterns that do not correspond to the observed facial motion - in a way that simple feature concatenation and MLP fusion cannot, as identified as a key gap in Section 2.7 @yi2023audiodeepfakedetectionsurvey. The [CLS] token output aggregates information from both modalities and feeds three independent sigmoid classification heads: one for the audio stream, one for the video stream, and one for the joint verdict.

On CPU-only hardware, the Transformer module is automatically replaced with a lightweight MLP fusion module to reduce inference latency, making the system deployable without GPU hardware.

#figure(
  image("figures/transformer_fusion_module.png", width: 70%),
  caption: [Transformer Fusion module: 256-dim video and audio features projected to 512 dims, combined with a learnable [CLS] token and positional embeddings into a 3-token sequence. Two layers of multi-head self-attention (8 heads, GELU, pre-norm) capture cross-modal dependencies. The [CLS] token output feeds three independent sigmoid classification heads.],
)

A note on head independence: all three classification heads receive the same CLS token representation, which aggregates information from both audio and video modalities via self-attention. The audio and video heads therefore do not have exclusive access to their respective streams; the observed per-modality dissociation in Chapter 4 reflects learned specialisation within the fused representation rather than true independent per-stream access. A design variant in which the audio head reads directly from the audio token and the video head reads directly from the video token would provide stronger independent modality access and is a direction for future investigation.

// 3.4
== Model Training and Evaluation Strategy

// 3.4.1
=== Loss Function

The initial proposal used standard Binary Cross-Entropy (BCE) loss. In the implemented system, this was replaced with Focal Loss @lin2018focallossdenseobject :

$ cal(L)_"focal" = -alpha_t (1 - p_t)^gamma log(p_t) $

where $gamma = 2.0$ is the focusing parameter and $alpha = 0.25$ is a class balance weight. The motivation for this change arose from observations during early training runs in which the model rapidly converged to a state of predicting the majority class for the majority of examples. With four balanced manipulation types, easy examples - videos with obvious, high-contrast artefacts - dominated the gradient signal and prevented the model from learning to detect subtle manipulations. Focal Loss downweights the contribution of easy, well-classified examples through the $(1-p_t)^gamma$ term, concentrating training capacity on hard, ambiguous cases. When $gamma = 0$, Focal Loss reduces to standard BCE, so the change is strictly a generalisation of the original proposal.

The total loss combines all three classification heads:

$ cal(L)_"total" = cal(L)_"audio" + cal(L)_"video" + w_j dot cal(L)_"joint" $

The joint head is weighted by $w_j = 2.0$ to reflect its role as the primary detection target. The focal hyperparameters $gamma$ and $alpha$ are configurable through `config.py`; $w_j$ is fixed at 2.0 as no training runs required adjustment of this value.

// 3.4.2
=== Two-Phase Training

Training proceeded in two phases to protect the pretrained encoder features during early optimisation. In Phase 1, covering the first 25% of training epochs, the video and audio encoder parameters were frozen and only the fusion module and classification heads were updated. This prevented the randomly-initialised fusion module from generating large, destructive gradients that could corrupt the Kinetics-400 and ImageNet pretrained weights before any meaningful fusion representations had been established.

In Phase 2, covering the remaining 75% of epochs, all parameters were unfrozen. Encoder parameters were trained at a learning rate of $1 times 10^-5$, ten times lower than the fusion module's $1 times 10^-4$, to allow gradual domain adaptation without catastrophic forgetting of pretrained features. Both the phase boundary and the patience threshold for early stopping were expressed as fractions of the total epoch count (`freeze_epochs = max(1, round(epochs × 0.25))`, `patience = max(5, round(epochs × 0.30))`), making the training schedule scale automatically with any change to the total epoch budget.

#figure(
  image("figures/two_phase_training.png", width: 85%),
  caption: [Two-phase training schedule: Phase 1 (frozen encoders, 25% of epochs, LR 1×10⁻⁴) protects pretrained features; Phase 2 (unfrozen encoders, 75% of epochs, encoder LR 1×10⁻⁵) enables full-domain adaptation. ReduceLROnPlateau halves LR after 5 epochs without AUC improvement. Default budget: 10 epochs; runs capped at 5 due to student resource constraints.],
)

// 3.4.3
=== Optimiser and Scheduler

AdamW was used with weight decay $1 times 10^-4$. Gradient norms were clipped to a maximum of 1.0 per step to prevent instability arising from the 3D convolutional operations on 50-frame inputs, which can produce large gradient magnitudes. The observed learning rate drop at epoch 3 in the per-epoch tables reflects the two-phase fine-tuning schedule (encoder LR reduced from $1 times 10^-4$ to $1 times 10^-5$), not the ReduceLROnPlateau scheduler. ReduceLROnPlateau was configured to halve all learning rates when validation joint AUC did not improve for five consecutive epochs, but with only 5 total epochs and a 2-epoch freeze phase, the maximum fine-tuning duration was 3 epochs — fewer than the scheduler's 5-epoch patience threshold, meaning the scheduler likely never triggered. Early stopping halted training when no improvement was observed for 30% of the total epoch budget. The default epoch budget was set to 10 as a conservative upper bound; in practice, training runs were limited to 5 epochs due to student resource constraints.

Multi-GPU training was managed via PyTorch `DataParallel`, with the effective batch size scaling linearly with the number of available GPUs. This was necessary because the GPU instances available through Vast.ai varied in configuration between training runs.

// 3.4.4
=== Checkpoint Management and Resumability

The model state, optimiser state, learning rate scheduler state, and random number generator states (Python, NumPy, and PyTorch) were all saved to disk at the end of every epoch. The checkpoint achieving the highest validation joint AUC was saved separately as `best_model.pth`. A separate `training_checkpoint.pth` stored the latest epoch state for resuming interrupted runs.

This design was motivated by the use of cloud-based GPU instances, where sessions are time-limited and instances can be terminated unexpectedly. The resumable pipeline ensured that no completed work was lost, and that training could continue from the exact state - including random seeds - at which it was interrupted. The `best_model.pth` file is the checkpoint used for all evaluation and inference; `training_checkpoint.pth` is used only to resume training.

// 3.5
== Challenges and Limitations

// 3.5.1
=== Infrastructure and Platform Challenges

*_PyTorch GPU Incompatibility on macOS._*

Initial development was carried out on a MacBook Pro equipped with an Apple Silicon chip. PyTorch's CUDA-based GPU acceleration is not supported on macOS, and while Apple's Metal Performance Shaders (MPS) backend provides partial GPU support, it is incompatible with several operations used in this pipeline - particularly 3D convolutional operations in ResNet3D-18 and certain Transformer attention kernels. Attempting to run training on MPS produced unsupported operation errors that could not be resolved without fundamentally changing the model architecture. As a result, the MacBook Pro could only be used for CPU-based development and debugging on very small subsets; it was entirely unsuitable for full-scale training on the 68,851-video feature set.

*_Google Colab Storage and Access Limitations._*

The first attempted training environment was Google Colab, which provides free and paid access to GPU compute. However, Colab imposed several constraints that made it impractical for this project. Storage was the most significant bottleneck: the AV-Deepfake1M++ dataset is distributed as approximately 1.4 TB of split compressed archives, with the extracted video files requiring substantially more disk space on top of that. Both figures far exceed Colab's default and paid storage allocations. Temporary runtime storage is wiped at the end of every session, requiring the dataset to be re-transferred from Google Drive at the start of each run - a process that, given Drive's read bandwidth limits, took several hours and frequently failed partway through due to Google Drive API rate limits. Accessing large volumes of data from Drive multiple times in a single day triggered quota exhaustion errors, halting access for 24-hour periods. Furthermore, Colab's free tier imposes session time limits, disconnecting the runtime mid-training and losing unsaved progress. Even on paid tiers, the ephemeral nature of Colab's storage made managing a 68,851-video feature cache across multiple sessions infeasible. These compounding issues - storage quota, API rate limits, session timeouts, and bandwidth constraints - made Colab an unworkable environment for a project of this scale.

*_Migration to Vast.ai Cloud GPU Servers._*

To overcome these infrastructure barriers, all model training and large-scale feature extraction were migrated to dedicated GPU instances rented from Vast.ai, a marketplace for on-demand cloud GPU compute. Vast.ai provided persistent storage volumes that survived between sessions, NVIDIA CUDA-capable GPUs compatible with the full PyTorch stack, and configurable instance specifications that could be matched to the memory and compute requirements of each training run. Data was transferred to the instance once and retained across sessions, eliminating the re-transfer overhead that made Colab impractical. The resumable checkpoint system described in Section 3.4.4 was designed specifically to handle the case where a Vast.ai instance was terminated mid-training - either due to spot-instance preemption or session expiry - allowing training to resume from the last saved epoch without any loss of progress. Despite this, file transfer between the Vast.ai instance and the local machine remained a risk: as discussed in Section 4.2.1, the checkpoint for Model 1 was corrupted during download via `scp`, preventing post-hoc analysis of that training run.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    [_Criterion_], [_Google Colab_], [_Vast.ai Cloud GPU_], [_Local MacBook Pro_],
    [Storage],
    [Ephemeral; wiped per session; Drive API rate limits],
    [Persistent volumes; data retained across sessions],
    [Sufficient for small subsets only],

    [GPU],
    [Free tier: T4 (time-limited); paid: A100 (costly)],
    [On-demand RTX 3080/3090, A4000],
    [MPS backend incompatible with 3D convolutions],

    [Session persistence],
    [Disconnects mid-training; unsaved progress lost],
    [Resumable via checkpoint system],
    [Local; always available],

    [Cost],
    [Free tier unworkable; paid ~\$50+/month], [~\$10–15 per 5-epoch run (student-funded)],
    [N/A (existing hardware)],
    [Outcome],

    [Unworkable for 68K-video dataset], [Used for all training and extraction], [Used for CPU debugging only],
  ),
  caption: [Infrastructure platform comparison: Colab, Vast.ai, and local development environments],
)

Three Vast.ai GPU instances were rented over the course of the project. Their specifications are summarised below.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    [_GPU_], [_VRAM_], [_CPU / RAM_], [_Cost (per hour)_], [_Used for_],
    [2× RTX 3060], [24 GB total], [192 vCPU / 257.8 GB], [\$0.783/hr], [Training runs (higher-spec)],
    [2× RTX 3060], [24 GB total], [192 vCPU / 257.8 GB], [\$0.569/hr], [Training runs (spot pricing)],
    [1× RTX 3070], [8 GB], [24 vCPU / 64.4 GB], [\$0.028/hr], [Feature extraction, CPU debugging, small-batch experiments],
  ),
  caption: [Vast.ai GPU instances used for training and feature extraction],
)

Instances 1 and 2 shared identical hardware (dual RTX 3060 on a ProLiant DL580 Gen9 with Xeon E7-8890 v4 processors) but were acquired at different spot prices (\$0.783 and \$0.569 per hour). Instance 2's lower cost made extended training runs more economical. Instance 3 (single RTX 3070 on a X99 motherboard with Xeon E5-2678 v3) was used for smaller tasks where its 8 GB VRAM was sufficient, at a fraction of the cost.

#figure(
  image("figures/vastai_instance_1.jpg", width: 85%),
  caption: [Vast.ai Instance 1 — 2× RTX 3060 (24 GB), \$0.783/hr],
)

#figure(
  image("figures/vastai_instance_2.jpg", width: 85%),
  caption: [Vast.ai Instance 2 — 2× RTX 3060 (24 GB), \$0.569/hr],
)

#figure(
  image("figures/vastai_instance_3.jpg", width: 85%),
  caption: [Vast.ai Instance 3 — 1× RTX 3070 (8 GB), \$0.028/hr],
)

//3.5.2
=== Dataset and Modelling Limitations

The primary modelling limitation arises from computational and storage constraints. Using only the validation split of AV-Deepfake1M++ rather than the full training set limits the diversity of manipulation techniques and speakers to which the model is exposed during training. The training and validation subsets were constructed from this single split using a speaker-based partition, which ensures that the evaluation is honest but means the effective training set is smaller than would be ideal.

A second limitation is the exclusion of temporal localisation. Although frame-level manipulation detection could provide more granular insight into model behaviour and enable localisation of the manipulated region within a clip, this was deferred due to the computational requirements of frame-level modelling and the storage demands of per-frame feature representations.

Third, the audio pipeline processes a fixed two-second window rather than the full clip. For videos where the manipulated segment is long or distributed across the clip, the selected window may not always capture the most informative region, particularly for real videos where the central two seconds are used by default.

Finally, since the dataset was constructed by third parties, the study relies on the accuracy and completeness of the existing annotations. Any labelling bias or inconsistency in the `fake_segments` annotations may influence model performance in ways that are not easily detectable.

// 3.6
== Ethical Considerations

The dataset used in this study is publicly available and was collected under established data use agreements. As such, no direct ethical concerns arise from data collection or use. However, deepfake detection research carries broader ethical implications. Although the goal of this work is defensive in nature - developing tools to identify manipulated media - the same techniques and architectural insights could theoretically inform improvements to generative systems @Chesney2019 @Westerlund2019. This study positions its contribution strictly within the context of detection and harm mitigation, without making deployment or real-world enforcement claims.

// 3.7
== Implementation Details

The initial proposal specified Python, TensorFlow, SciPy, FFMPEG tools, and a Convolutional Neural Network (CNN) as the primary development stack. In practice, the implementation used Python 3.12 with PyTorch 2.0, reflecting a shift from TensorFlow to PyTorch for its superior compatibility with the ResNet3D-18 and Transformer architectures. TorchVision provided the ResNet3D-18 and ResNet18 architectures, TorchAudio handled audio loading and mel-spectrogram computation, OpenCV was used for video frame extraction, and scikit-learn supported dataset partitioning and metric computation. FFMPEG was used implicitly through torchaudio and OpenCV for video decoding. The CNN in the initial proposal evolved into a Cross-Modal Transformer Fusion architecture, as documented in Sections 3.3.3–3.3.5. Training was conducted on NVIDIA GPU servers accessed via Vast.ai. Experiment tracking, metric logging, and training visualisation were managed using Weights & Biases.

The full pipeline - from data download through evaluation, model comparison, and web-based inference - is implemented as a modular codebase of Python files, each with a clearly defined responsibility, as shown in the table below. Four additional utility scripts - `create_test_data.py` (test set generation), `compare_models.py` (multi-model metrics with plots), `download_data.py` (dataset acquisition), and `analyze_data.py` (standalone EDA) - handle preparation and analysis tasks invoked independently from the main training pipeline.

#figure(
  table(
    columns: (auto, auto),
    [_Module_], [_Responsibility_],
    [`config.py`], [All hyperparameters, paths, and derived constants],
    [`data_utils.py`], [Metadata loading, speaker split, parallel feature extraction, dataset class],
    [`audio.py`], [Audio encoder (ResNet18 on mel-spectrogram)],
    [`video.py`], [Video encoder (ResNet3D-18 on frame sequences)],
    [`cross_modal.py`], [Fusion module (Transformer and MLP variants)],
    [`train_utils.py`], [Training loop, validation, loss, optimiser, W&B logging],
    [`checkpoint_utils.py`], [Checkpoint save and load],
    [`main.py`], [Pipeline entry point],
    [`inference.py`], [Standalone single-video and batch inference],
  ),
  caption: [Codebase module responsibilities],
)

Parallel feature extraction used Python's `multiprocessing.Pool` with fork context and up to 28 CPU workers. Each video's features were saved as an individual `.pt` file to avoid loading the full dataset into RAM, which would have required several terabytes of memory. A manifest JSON file indexed all successfully extracted samples and was checkpointed every 500 videos to support crash recovery. Corrupted or unreadable videos were tracked in a separate failed-samples list and skipped on subsequent runs.

//3.7.1
=== Web Interface

A browser-based web application (`web/app.py`) was developed to make the detection capability accessible without command-line or ML expertise. Built on Flask with an HTML frontend and minimal static JavaScript and CSS, the interface provides three tabs:

- *Analyze:* Drag-and-drop video upload, model selector with per-model metadata (AUC, epochs, recommended use), real-time verdict and audio/video/joint score display, and reset for the next upload.
- *Compare:* Upload one video and run it through both configured models simultaneously, with side-by-side verdicts, three per-modality scores per model, and an agree/disagree summary.
- *History:* SQLite-backed table of all past analyses showing filename, verdict tag, joint score, and model used, with per-entry delete and bulk clear.

The backend imports `load_model` and `predict_video` directly from `inference.py`. `inference.py` redefines the model architecture classes inline, mirroring `audio.py`, `video.py`, and `cross_modal.py` so that it can run without depending on the training pipeline's module imports. An in-memory cache loads each model once and shares it across requests. Uploaded videos are saved to a temporary directory, processed by calling `predict_video`, and deleted immediately after. Model paths are hardcoded to `logs/logs_2/best_model.pth` (Model 2) and `logs/logs_3/best_model.pth` (Model 3), with environment variable overrides available. Multi-window inference (3 evenly-spaced 2-second windows from the start, centre, and end of the video, averaged) is applied for robustness. The full API specification, including all five endpoints (`/api/models`, `/api/analyze`, `/api/compare`, `/api/history`, `/api/history/<id>`), is documented in `WebInterface.md`.

//3.8
== Conclusion

This chapter described the complete methodology and implementation. The implemented system diverges from the initial proposal in three principal areas: Wav2Vec 2.0 → ResNet18 on mel-spectrograms (memory and codec compatibility), MobileNetV3 lip-region → ResNet3D-18 full-frame (temporal and whole-face coverage), and DiMoDif → Transformer Encoder fusion (reduced alignment dependency). BCE loss was replaced with Focal Loss to address easy-example domination. Each change is grounded in specific implementation constraints. The web interface (Section 3.7.1) provides browser-based access to the detection system. Results of applying this pipeline are presented in Chapter 4.

// Chapter 4
= Results and Findings

// 4.1
== Introduction

This chapter presents the results of all training runs and the evaluation of the best-performing model on a 100-video test set sampled from the validation split using `create_test_data.py`. Four training sessions were conducted, producing four checkpoint pairs (`best_model.pth` and `training_checkpoint.pth`) stored in `logs/logs_1` through `logs/logs_4`. Runs 3 and 4 originated from the same training session: Model 3 is the epoch-3 checkpoint, Model 4 is the epoch-5 result of the same run. Their training histories were extracted directly from checkpoint metadata using PyTorch and are reported below with full per-epoch detail. The 100-video test set provides indicative rather than statistically robust results; a larger evaluation set would be needed to confirm these findings.

== Training Results - All Four Runs

All four runs used the same architecture - ResNet3D-18 video encoder, ResNet18 audio encoder, and Transformer Encoder fusion module - and the same AV-Deepfake1M++ validation split with speaker-based train/validation partition. The training schedule applied two-phase learning (frozen encoders for the first 25% of epochs, unfrozen at 10× lower learning rate thereafter) with Focal Loss ($gamma = 2.0$, $alpha = 0.25$) and a joint loss weight of $w_j = 2.0$, as described in Sections 3.4.1 and 3.4.2. Model 3 is the epoch-3 checkpoint from the same training run that produced Model 4; epochs 1–3 are shared between them.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    [_Run_], [_Epochs_], [_Best Val Joint AUC_], [_Final Train Loss_], [_Final Val Loss_], [_Checkpoint Status_],
    [Run 1 (Model 1)], [1], [0.6626], [0.2621], [1.2895], [Corrupted on download],
    [Run 2 (Model 2)], [5], [_0.9937_], [0.0605], [0.2774], [Intact],
    [Run 3 (Model 3)], [3], [0.9851], [0.1418], [0.4673], [Intact (early stop)],
    [Run 4 (Model 4)], [5], [0.9925], [0.0609], [0.2854], [Intact],
  ),
  caption: [Summary of all four training runs],
)

Figure 4.1 shows the training history extracted from checkpoint metadata for all four runs: validation joint AUC progresses from near-chance levels during frozen-encoder training (epochs 1–2) to above 0.98 after encoder unfreezing (epochs 3–5). Models 2 and 4 follow near-identical trajectories, confirming pipeline reproducibility.

#figure(
  image("comparison_results/training_history.png", width: 100%),
  caption: [Training history across all four runs],
)

// 4.2.1
=== Model 1: Single-Epoch Baseline

Run 1 completed only one epoch of Phase 1 training, during which both encoders were frozen and only the fusion module and classification heads were updated. The epoch took approximately 8,091 seconds (∼2.25 hours) on the Vast.ai instance. The resulting validation joint AUC of 0.6626 is only marginally above random chance. Analysis of the checkpoint history confirmed a systematic prediction failure: all videos, regardless of true label, received joint scores below 0.35, indicating the fusion module had learned to uniformly predict fake rather than develop discriminative cross-modal representations. The checkpoint file was subsequently corrupted during `scp` download from the Vast.ai instance and cannot be loaded for inference. The training metadata extracted from the checkpoint confirms the epoch 1 statistics but precludes post-hoc evaluation on the test set.

//4.2.2
=== Model 2 Best Overall Model (5 Epochs)

Run 2 completed all five epochs, transitioning from Phase 1 (frozen encoders, learning rate 1×10⁻⁴) to Phase 2 (unfrozen, 1×10⁻⁵) at epoch 3. The table below presents the per-epoch metrics extracted directly from `logs/logs_2/best_model.pth`.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),
    [_Epoch_],
    [_Phase_],
    [_Train Loss_],
    [_Val Loss_],
    [_Val Joint AUC_],
    [_Val Audio AUC_],
    [_Val Video AUC_],
    [_LR_],
    [_Duration (s)_],

    [1], [Frozen], [0.2634], [1.3074], [0.6692], [0.6088], [0.5829], [1×10⁻⁴], [6,484],
    [2], [Frozen], [0.2544], [1.2704], [0.6924], [0.6564], [0.5846], [1×10⁻⁴], [6,491],
    [3], [Fine-tune], [0.1407], [0.4119], [_0.9879_], [0.9798], [0.9687], [1×10⁻⁵], [13,657],
    [4], [Fine-tune], [0.0801], [0.3465], [0.9917], [0.9830], [0.9856], [1×10⁻⁵], [13,755],
    [5], [Fine-tune], [0.0605], [0.2774], [_0.9937_], [0.9861], [0.9914], [1×10⁻⁵], [13,804],
  ),
  caption: [Model 2: per-epoch validation metrics],
)

The AUC jump from epoch 2 (0.692) to epoch 3 (0.988) reflects the encoder unfreezing behaviour described in Section 3.4.2: with both modality-specific encoders adapting to the deepfake detection task, the cross-modal representations rapidly specialise. The gap between training loss (0.14) and validation loss (0.41) at epoch 3 is not directly interpretable as overfitting, because Focal Loss (γ=2.0) was used for training while standard BCE was used for validation monitoring — the two loss functions produce values on different numerical scales. AUC, which is independent of the loss function used for monitoring, is the primary convergence indicator; the monotonic improvement in validation AUC through epoch 5 confirms that training was progressing correctly.

// 4.2.3
=== Models 3–4: Additional Runs

Runs 3–4 were generated from the same initial feature extraction as Runs 1–2 but were run with a revised epoch schedule or as continuation checkpoints.

_Run 3_ was saved at epoch 3 - the first fine-tuning epoch - as a checkpoint from the same training run that produced Run 4. Its best AUC of 0.9851 represents the performance achievable after a single fine-tuning epoch.

_Run 4_ completed 5 epochs from the same training session as Run 3, reaching a best AUC of 0.9925. The per-epoch metrics extracted from this run are shown below.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto),
    [_Epoch_],
    [_Phase_],
    [_Train Loss_],
    [_Val Loss_],
    [_Val Joint AUC_],
    [_Val Audio AUC_],
    [_Val Video AUC_],
    [_LR_],
    [_Duration (s)_],

    [1], [Frozen], [0.2631], [1.3117], [0.6829], [0.6410], [0.5866], [1×10⁻⁴], [5,507],
    [2], [Frozen], [0.2552], [1.2730], [0.6910], [0.6534], [0.5906], [1×10⁻⁴], [5,442],
    [3], [Fine-tune], [0.1418], [0.4673], [_0.9851_], [0.9812], [0.9597], [1×10⁻⁵], [13,721],
    [4], [Fine-tune], [0.0807], [0.3528], [0.9913], [0.9851], [0.9861], [1×10⁻⁵], [13,819],
    [5], [Fine-tune], [0.0609], [0.2854], [_0.9925_], [0.9860], [0.9874], [1×10⁻⁵], [13,837],
  ),
  caption: [Model 4: per-epoch validation metrics],
)

Model 3 is the epoch-3 checkpoint from this same run; Models 3 and 4 therefore share epochs 1–3. Comparison with Run 2's trajectory (0.9937 at epoch 5) reveals that the two independent runs produced nearly identical results, differing by less than 0.002 AUC at each epoch. This reproducibility confirms that the training pipeline is stable and that results are not an artefact of a lucky random initialisation.

#figure(
  image("figures/training_history_model4.png", width: 100%),
  caption: [Training history for Model 4: training and validation loss (Focal Loss, γ=2.0), validation AUC across all three heads, and learning rate schedule - showing the characteristic Phase 1→2 transition at epoch 3. Generated from checkpoint metadata stored in logs/logs_4/output.txt.],
)

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    [_Epoch_], [_Model 1_], [_Model 2_], [_Model 3_], [_Model 4_],
    [1], [0.6626], [0.6692], [0.6829], [0.6829],
    [2], [-], [0.6924], [0.6910], [0.6910],
    [3], [-], [_0.9879_], [_0.9851_], [0.9851],
    [4], [-], [0.9917], [-], [0.9913],
    [5], [-], [_0.9937_], [-], [0.9925],
  ),
  caption: [Per-epoch comparison across all four models (val joint AUC)],
)

All runs show the same characteristic pattern: low AUC during frozen-encoder training (epochs 1–2), a large jump at the first fine-tuning epoch (epoch 3), and steady improvement thereafter. The fine-tuning phase is therefore the essential driver of performance.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    [_Epoch_], [_Model 1_], [_Model 2_], [_Model 3_], [_Model 4_],
    [1], [0.2621], [0.2634], [0.2631], [0.2631],
    [2], [-], [0.2544], [0.2552], [0.2552],
    [3], [-], [0.1407], [0.1418], [0.1418],
    [4], [-], [0.0801], [-], [0.0807],
    [5], [-], [0.0605], [-], [0.0609],
  ),
  caption: [Training loss trajectory across all models],
)

The near-identical loss values across Models 2 and 4 further confirm that Runs 3–4 used the same data ordering and random seed as Run 2, producing highly reproducible results.

// 4.3
== Test Set Evaluation

Evaluation was conducted using `compare_models.py` on a 100-video test set (25 real, 75 fake) sampled from the validation split using `create_test_data.py`. The test set was held out from training — no video in the test set appears in the training partition. All 21 test speakers originate from the validation split, with zero overlap with the 105 training speakers, preserving the speaker-disjoint evaluation guarantee. All three loadable checkpoints (Models 2–4) were evaluated. Model 1's checkpoint was corrupted and could not be loaded. Each video was processed with one 2-second analysis window on CPU. All results below are measured directly from inference on the test set; no numbers are estimated.

// 4.3.1
=== Overall Metrics

#figure(
  table(
    columns: (auto, auto, auto, auto),
    [_Metric_], [_Model 2 (5ep)_], [_Model 3 (3ep)_], [_Model 4 (5ep)_],
    [_Test AUC_], [0.9189], [_0.9371_], [0.9152],
    [_Accuracy at threshold 0.5_], [66.0%], [_93.0%_], [71.0%],
    [_Best-threshold accuracy_], [87.0% (t=0.795)], [_93.0%_ (t=0.533)], [90.0% (t=0.979)],
    [_Precision_], [0.418], [_1.000_], [0.462],
    [_Recall_], [0.920], [0.720], [_0.960_],
    [_F1_], [0.575], [_0.837_], [0.623],
    [_Best-threshold F1_], [0.764], [_0.837_], [0.750],
    [TP (real → REAL)], [23], [18], [24],
    [TN (fake → FAKE)], [43], [_75_], [47],
    [FP (fake → REAL)], [32], [_0_], [28],
    [FN (real → FAKE)], [2], [7], [_1_],
    [Mean real score], [0.867], [0.676], [0.861],
    [Mean fake score], [0.355], [_0.146_], [0.323],
    [Train AUC (checkpoint)], [0.9937], [0.9851], [0.9925],
  ),
  caption: [Test set evaluation results - all three models (threshold = 0.50)],
)

Figure 4.7 presents a comprehensive comparison of all three models on the test set: bar charts of overall metrics (AUC, accuracy, precision, recall, F1), ROC curves, score distribution histograms, confusion matrix summaries, per-type AUC breakdowns, and audio-vs-video score scatter plots for each model. The scatter plots reveal the modality-specific behaviour of the three-head architecture - `audio_modified` clips cluster toward high video scores with low audio scores, while `visual_modified` clips show the reverse pattern.

#figure(
  image("comparison_results/model_comparison.png", width: 95%),
  caption: [Model comparison scatter plot and histograms],
)

_Model 3 wins on 5 of 7 head-to-head metrics_ - AUC, accuracy, F1, precision, and false positives - and is the best overall model on the test set despite having the lowest training AUC of the three. Model 4 achieves the lowest false negative count (1 missed real video) and highest recall at the cost of 28 false positives.

// 4.3.2
=== Interpretation: Why Model 3 Outperforms Model 2 on the Test Set

The finding that Model 3 (training AUC 0.9851, epoch 3) outperforms Model 2 (training AUC 0.9937, epoch 5) on the test set is counterintuitive and warrants analysis.

The key difference lies in _score calibration_. Model 2 produces real video scores predominantly in the 0.85–0.97 range and fake scores in the 0.30–0.45 range - a spread that is relatively narrow above the decision boundary. At a fixed threshold of 0.5, many fake videos score above the boundary. Model 3 produces a much lower mean fake score (0.146 vs 0.355) with all 75 fake videos falling below 0.5, achieving a clean true negative rate (TN = 75, FP = 0) at no threshold adjustment. Its real video scores are lower (mean 0.676), which causes 7 false negatives (28% of real videos misclassified as fake). Whether zero false positives or a lower false negative rate is preferable depends on the deployment context: in public-facing content moderation, false accusations carry reputational risk; in forensic authentication, missed deepfakes are the greater concern. The results illustrate that the choice between Model 2 and Model 3 involves an application-specific precision-recall tradeoff rather than a universally optimal model.

This behaviour is consistent with the known dynamics of encoder fine-tuning. Model 3 was saved at epoch 3 - the first fine-tuning epoch - when the model had just begun adapting the encoders. At this point the encoders are learning to suppress fake modality signals strongly, producing very low fake scores. Models 2 and 4 show evidence of mild score compression: the gap between real and fake scores narrows as the model continues to optimise the training objective. This is not overfitting in the classical sense (validation AUC continued to improve), but reflects the fact that maximum training AUC does not always correspond to maximum test-set accuracy at a fixed threshold.

_An important caveat:_ all runs were capped at 5 epochs due to student resource constraints on cloud GPU access (Section 3.4.3). It is unknown whether Models 2 and 4, if trained for the full 10-epoch budget, would have produced better-calibrated scores after undergoing a full learning rate schedule cycle with ReduceLROnPlateau. The score compression observed between epochs 3 and 5 may represent a transient phase rather than a permanent degradation - extended training with a reduced learning rate could potentially restore clean score separation. Model 3's results must therefore be interpreted as the best achieved within the resource limits of a student project, rather than as evidence that early stopping is universally preferable for this architecture. Further training would be needed to determine whether score separation improves, degrades, or stabilises with extended fine-tuning.

These results suggest that Model 3 at threshold 0.5 achieves the highest accuracy (93.0%) with zero false positives, while Model 2 at its best threshold of 0.795 offers a balanced precision/recall trade-off (F1 = 0.764, accuracy 87.0%). The optimal choice depends on the deployment context, as discussed in Section 5.3.1.

#figure(
  image("figures/calibration_curves.png", width: 100%),
  caption: [Reliability diagrams: predicted joint score probability vs. actual fraction of positives for Models 2, 3, and 4. Points on the diagonal indicate perfect calibration; points above the diagonal indicate the model is overconfident (predicting higher scores than justified). Model 3 shows the smallest mean calibration gap (0.19), confirming that earlier fine-tuning epochs produce better-calibrated scores for this architecture.],
)

// 4.4
== Per Type Breakdown

Each model was evaluated separately on the four manipulation types. This reveals modality-specific behaviour of the three-head architecture.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    [_Type_], [_Model 2 (5ep)_], [_Model 3 (3ep)_], [_Model 4 (5ep)_],
    [`real` (25 videos)], [92%], [72%], [_96%_],
    [`audio_modified` (25 videos)], [48%], [_100%_], [52%],
    [`visual_modified` (25 videos)], [64%], [_100%_], [72%],
    [`both_modified` (25 videos)], [60%], [_100%_], [64%],
  ),
  caption: [Per type test accuracy],
)

With a 100-video test set, a single misclassification shifts category accuracy by 4 percentage points. Wilson 95% confidence intervals for the key results are: overall accuracy Model 3 93.0% [86.3%, 96.6%]; Model 2 66.0% [56.3%, 74.5%]; Model 3 per-type fake categories 100% [86.7%, 100.0%] (n=25 each). The wide intervals on per-category results underscore that these accuracy figures represent indicative trends rather than statistically robust benchmarks. All results should be interpreted as evidence of system capability, not definitive performance claims.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    [_Type_], [_Model 2 (5ep)_], [_Model 3 (3ep)_], [_Model 4 (5ep)_],
    [`real`], [0.867], [0.676], [0.861],
    [`audio_modified`], [0.409], [_0.158_], [0.398],
    [`visual_modified`], [0.304], [_0.124_], [0.249],
    [`both_modified`], [0.353], [_0.156_], [0.323],
  ),
  caption: [Per type mean joint score],
)

#figure(
  image("figures/per_type_accuracy_bar_chart.png", width: 100%),
  caption: [Test set accuracy by manipulation type: Model 3 achieves 100% on all three fake categories (audio_modified, visual_modified, both_modified) while Model 2 and Model 4 show variable performance. Grouped bar chart comparing all three loadable models across four manipulation types.],
)


#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    [_Type_], [_M2 audio_], [_M2 video_], [_M3 audio_], [_M3 video_], [_M4 audio_], [_M4 video_],
    [`real`], [0.861], [0.930], [0.716], [0.712], [0.862], [0.906],
    [`audio_modified`], [0.425], [_0.874_], [0.375], [_0.294_], [0.427], [_0.849_],
    [`visual_modified`], [0.411], [0.702], [0.415], [0.242], [0.323], [0.772],
    [`both_modified`], [0.368], [0.851], [0.387], [0.277], [0.349], [0.820],
  ),
  caption: [Per type mean audio and video head scores],
)

Several important patterns emerge from these results:

_1. The audio head does not clearly dissociate on `audio_modified` clips._
For Models 2 and 4, the audio head score for `audio_modified` videos (mean ≈ 0.42–0.43) is not substantially lower than for `real` videos (0.86). The video head score, however, is anomalously high for `audio_modified` clips (0.87 and 0.85 respectively) - suggesting the video encoder is rating the video stream as authentic (which it is), thus reducing the joint fake signal. This is the expected modality dissociation, but the audio head's discriminative power is weaker than expected at this threshold.

_2. `visual_modified` clips produce the lowest joint scores for Models 2 and 4_, consistent with the expectation that visual manipulation is more detectable via the video head. The video head scores for `visual_modified` clips (0.70 and 0.77) are lower than for `audio_modified` clips (0.87 and 0.85), confirming modality-specific learning.

_3. Model 3 correctly identifies all fake types with 100% accuracy._ It shows uniformly low joint scores for all fake types (0.12–0.16) and suppresses both the audio and video head scores strongly. This is the idealised behaviour: all fake manipulation is clearly separated from real content regardless of type.

_4. The `real` video accuracy gap between models_ (72% for Model 3 vs 92–96% for Models 2 and 4) reflects the threshold effect described in §4.3.2. Model 3's lower calibration of real scores means 7 of 25 real videos score below 0.5, producing false negatives.

// 4.5
== Overall Performance Summary


#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    [_Use Case_], [_Recommended Model_], [_Threshold_], [_Expected Accuracy_], [_Expected F1_],
    [Maximum overall accuracy], [Model 3], [0.50], [_93.0%_], [_0.837_],
    [Maximum recall (miss no real videos)], [Model 4], [0.50], [71.0%], [0.623],
    [Balanced precision/recall], [Model 2], [0.795], [87.0%], [0.764],
    [Zero false positives], [Model 3], [0.50], [_93.0%_], [_0.837_],
  ),
  caption: [Overall performance summary],
)

// 4.6
== Four-Model Training vs Test AUC Comparison

A counterintuitive finding is that the ordering of models by training validation AUC does not match the ordering by test set AUC.


#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    [_Model_], [_Train Val AUC_], [_Test AUC_], [_Gap_], [_Test Accuracy_], [_Test F1_],
    [Model 1 (1ep)], [0.6626], [N/A (corrupted)], [-], [-], [-],
    [Model 2 (5ep)], [_0.9937_], [0.9189], [−0.075], [66.0%], [0.575],
    [Model 3 (3ep)], [0.9851], [_0.9371_], [−0.048], [_93.0%_], [_0.837_],
    [Model 4 (5ep)], [0.9925], [0.9152], [−0.077], [71.0%], [0.623],
  ),
  caption: [Training AUC vs test AUC - all models],
)

All models show a gap between training validation AUC and test set AUC. The gaps for Models 2 and 4 (∼0.075) are larger than for Model 3 (∼0.048). This is consistent with the score compression effect described in §4.3.2: continued fine-tuning reduces the AUC gap on the training distribution but narrows the score margin between classes, which disproportionately affects test set performance at a fixed threshold.


The training validation AUC measures ranking performance on the full 13,000-video validation set. The test AUC measures ranking performance on 100 videos. The small test set introduces sampling variance - the 0.022 AUC gap between Model 3 and Models 2/4 on the test set (0.9371 vs 0.9152) may partly reflect this variance, since the test set contains only 25 real videos. A definitive ranking would require evaluation on a substantially larger held-out set.

// 4.7
== Conclusion

Four training sessions were conducted, producing four checkpoints; three produced intact, loadable checkpoints (Models 2, 3, and 4). Model 3 is the epoch-3 checkpoint from the same run that produced Model 4 at epoch 5. All three were evaluated on a 100-video test set. Model 3 achieved the best test-set results: AUC 0.937, accuracy 93.0%, precision 1.000, with zero false positives. Model 2 achieved the highest training AUC (0.994) but delivered lower at-threshold accuracy (66.0%) due to score compression - a divergence discussed further in Chapter 5.

// -----------------------------------------------------------------------------
// Evaluation
// -----------------------------------------------------------------------------
= Discussion and Evaluation

// 5.1
== Introduction

This chapter interprets the results presented in Chapter 4 in relation to the project objectives, compares the findings against prior work, evaluates the strengths and limitations of the implemented system, and reflects on the development process. The chapter is structured to address each objective in turn before broadening the discussion to cover unexpected outcomes, practical constraints, and the implications of the results.

// 5.2
== Evaluation Against Objectives

// 5.2.1
=== Objective 1: Speaker-Disjoint Dataset Partitioning

The speaker-based partition described in Section 3.4 was implemented successfully, producing zero speaker overlap between training and validation sets. The practical consequence is that the reported validation AUC reflects generalisation to entirely unseen identities rather than face or voice recognition - addressing the identity leakage risk identified in Section 2.7. Metrics from a random split would be expected to be higher but less meaningful. This objective was fully met.

// 5.2.2
=== Objective 2: Cross-Modal Transformer Fusion with Three-Head Output

The Transformer Encoder fusion module described in Section 3.3.5 was implemented and trained successfully. The per-type breakdown in Section 4.4 (the per-type tables below) confirms that both modality heads contribute independently. For `audio_modified` clips, the video head produces elevated scores (mean 0.87 for Model 2) while the joint score is suppressed, confirming that the audio head is the primary driver of fake detection in those cases. For `visual_modified` clips, the video head score drops to 0.70 while the audio head remains elevated, demonstrating the reverse specialisation. Both heads contribute meaningfully and in a modality-appropriate direction, though the audio head's discriminative margin is weaker than the video head's for Models 2 and 4. This objective was substantially met. Full ablation (training with one modality disabled) would provide stronger evidence of each modality's independent contribution, and is identified as a limitation in Section 5.5.

// 5.2.3
=== Objective 3: Focal Loss

Focal Loss was used throughout training with the parameters specified in Section 3.4.1. The steady decrease in training loss and the monotonic improvement in validation AUC across epochs (the per-epoch metrics table) are consistent with effective convergence. The contrast between Phase-1-only training (Model 1, AUC 0.663) and all Phase-2-trained models (AUC ≥ 0.985, the training-vs-test comparison table) provides indirect evidence that sufficient training under Focal Loss produces strong results. However, a direct ablation against standard BCE was not performed, so the specific contribution of Focal Loss over BCE cannot be isolated from the combined effect of all design choices. This objective was met in implementation but not rigorously validated through controlled comparison.

// 5.2.4
=== Objective 4: Resumable Training Pipeline

The checkpoint system described in Section 3.4.4 was used across multiple cloud GPU sessions, with training resuming successfully without loss of reproducibility. The W&B integration provided a full audit trail of all per-epoch metrics. This objective was fully met.

// 5.2.5
=== Objective 5: Evaluation on 100-Video Test Set

Evaluation was conducted on a 100-video test set sampled from the validation split using `create_test_data.py`. Results are reported in Chapter 4 including AUC, accuracy, precision, recall, F1, and per-type breakdown. This objective was met. The limitation noted in Section 5.5 regarding test set size is acknowledged.

// 5.2.6
=== Objective 6: Standalone Inference and Web Interface

The initial proposal (CN6000, 2025, Objective 4) called for *"demo software that detects and differentiates deepfake media using deep learning."* This was delivered in two forms. `inference.py` provides command-line inference on single videos and folders with no training pipeline dependencies. The web interface (`web/app.py` + `templates/index.html`) provides drag-and-drop upload on a single video, model selection, real-time verdict and audio/video/joint score display, side-by-side model comparison, and a history of past analyses. Together, these components address the initial proposal's objective of demo software for distinguishing real from fake media, implemented in a form that exceeds the original scope by offering per-modality score dissociation and a model comparison facility. This objective was fully met.

#figure(
  table(
    columns: (auto, auto, auto),
    [\#], [_Objective_], [_Status_],

    [1], [Speaker-disjoint dataset partition using GroupShuffleSplit], [Fully met - zero speaker overlap],
    [2],
    [Cross-Modal Transformer Fusion with three-head output],
    [Substantially met - per-type dissociation confirmed; ablation deferred],

    [3], [Focal Loss ($γ$ = 2.0, $α$ = 0.25) training], [Met in implementation - BCE ablation not performed],
    [4],
    [Resumable training pipeline with checkpointing and W&B logging],
    [Fully met - resumed across multiple cloud sessions],

    [5],
    [Evaluation on 100-video held-out test set],
    [Met - AUC, accuracy, precision, recall, F1 reported; small test set acknowledged],

    [6],
    [Standalone inference system and web interface],
    [Fully met - inference.py CLI + Flask web app with comparison and history],
  ),
  caption: [Evaluation of the six dissertation objectives against implementation outcomes],
)

// 5.3
== Interpretation of Results

//  5.3.1
=== Model Performance on the Test Set

The validation joint AUC values reported in Chapter 4 are strong under a speaker-disjoint partition. However, as Section 4.3.2 showed, the model with the highest training AUC (Model 2, 0.994) produced the lowest at-threshold test accuracy (66.0%), while Model 3 - the earliest fine-tuned checkpoint - achieved 93.0% with zero false positives. The cause (score compression during extended fine-tuning) and the caveats around the 5-epoch budget were discussed in Section 4.3.2. At the best threshold for each model - 0.795 for Model 2 and 0.533 for Model 3 - accuracies are 87.0% and 93.0% respectively, demonstrating that threshold recalibration recovers much of Model 2's ranking quality.
// 5.3.2
=== Per Type Analysis

The per-type breakdown in the per-type breakdown tables reveals a consistent pattern across all models. Visual manipulation is the easiest to detect: `visual_modified` clips achieve the lowest mean joint scores across all models (0.124–0.304), and Model 3 achieves 100% while Model 4 achieves 72% accuracy on this type. Audio modification is the hardest: `audio_modified` clips produce the highest fake scores (mean joint 0.158–0.409 across models), yielding only 48–100% accuracy depending on the model. `both_modified` clips are intermediate.

This pattern is counterintuitive - one might expect `audio_modified` to be detected at lower scores since both the audio head and the joint head should respond. The explanation lies in the video head behaviour: for `audio_modified` clips, the video stream is genuine and the video head correctly assigns a high authenticity score (0.85–0.87 across models), which counteracts the lower audio head signal and raises the joint score. This is the correct modality-specific behaviour - the video head is not wrong to rate the video as authentic - but it means the joint score for `audio_modified` clips lands closer to the 0.5 boundary than for `visual_modified` clips, where the video head provides a strong fake signal.

`both_modified` clips are not the hardest: they produce lower mean joint scores than `audio_modified` clips in all models, because both encoders provide consistent fake evidence that reinforces the joint head. The ordering from hardest to easiest across Models 2 and 4 is: `audio_modified` > `both_modified` > `visual_modified`.

//  5.3.3
=== Audio vs Video Head Dissociation

The per-type head scores table provides direct evidence of per-modality score dissociation. For `audio_modified` clips, the video head scores (0.85–0.87 across models) are significantly higher than the audio head scores (0.37–0.43) and approach the level seen for real videos (0.86–0.93). This confirms that the video encoder correctly identifies the video stream as authentic while the audio encoder correctly flags the manipulated audio - exactly the dissociation expected from a modality-aware architecture.

For `visual_modified` clips, the reverse pattern holds: the video head scores (0.70–0.77) are lower than for real videos, while the audio head scores (0.32–0.41) remain somewhat elevated. The asymmetry is less sharp than for `audio_modified` clips, which may reflect the fact that visual manipulation in AV-Deepfake1M++ involves lip-region synthesis that ResNet3D-18 (operating on full frames at 224×224 rather than cropped lip regions) is less sensitive to than audio cloning artefacts in the mel-spectrogram representation.

For `real` videos, both heads assign high scores (audio: 0.72–0.86, video: 0.71–0.93), confirming that genuine content is not confused with fakes at the modality level.

The weaker audio head discriminative margin in Models 2 and 4 suggests that the vision-centric bias identified in Gap 2 (Section 2.7) is only partially resolved by the three-head architecture. While both heads produce directionally correct outputs, the video head provides substantially stronger per-type discrimination than the audio head in these models. Model 3's stronger audio suppression (mean audio head score 0.375 for `audio_modified` vs 0.716 for `real`) suggests that earlier checkpoints may provide more balanced modality specialisation, though this requires confirmation through larger-scale evaluation.

// 5.3.4
=== Model 1 Failure Analysis

The results reported in Section 4.2.1 indicate that Model 1 had not converged - the fusion module had not yet stabilised, producing default-class predictions regardless of input. The two-phase training design was intended to prevent this by protecting encoder features during early optimisation, but Model 1 was saved before that process could take effect. The corrupted download prevented further post-hoc analysis, but the prediction pattern is characteristic of an underfit classifier.
//  5.4
== Comparison with Prior Work

Multimodal deepfake detectors evaluated on comparable benchmarks have reported varying results depending on evaluation protocol and dataset scale. @Cai2024 demonstrated that multimodal detectors on AV-Deepfake1M can achieve AUC above 0.90 under speaker-disjoint evaluation, though specific figures vary by architecture and manipulation type. On FakeAVCeleb, audio-visual detectors have reported AUC values in the 0.85–0.95 range under controlled settings @yi2023audiodeepfakedetectionsurvey. The AV-Deepfake1M++ challenge at ACM Multimedia 2025 provided standardised benchmarks using the full 2M-clip dataset; baseline results and challenge submissions are documented in @Cai2025.

The validation AUC achieved in this project (0.985–0.994) and test AUC (0.915–0.937) are broadly within the range of published multimodal systems, but direct numerical comparison is not possible for two reasons. First, this project trained exclusively on the 68,851-video validation split rather than the full training set, limiting exposure to the dataset's full speaker and manipulation diversity. Second, published benchmarks typically report on fully held-out test sets with controlled data splits, whereas the test evaluation here uses 100 videos sampled from the same validation split. These differences mean the results presented in this chapter provide indicative evidence of system capability rather than competitive benchmark performance.

Unlike the simpler concatenation-based fusion approaches identified as a gap in Section 2.7 @yi2023audiodeepfakedetectionsurvey, the Transformer fusion module (Section 3.3.5) used here allows audio and video representations to interact during feature learning. Whether this provides a measurable advantage over simpler fusion on this dataset cannot be determined without an ablation study, which represents a direction for future work.

//  5.5
== Limitations

*_Test set size._*
The 100-video test set is too small to draw statistically reliable conclusions. A single misclassified video changes accuracy by 1 percentage point, and confidence intervals around the reported AUC would be wide. A test set of at least 500 videos per manipulation type would be needed to produce reliable estimates.

*_Training data._ *
Using only the validation split of AV-Deepfake1M++ (68,851 videos) rather than the full training set (over one million clips) limits the model's exposure to the full diversity of manipulation techniques and speaker identities. The full dataset would be expected to produce better generalisation.

*_No ablation study._*
The contribution of each architectural decision - Focal Loss vs BCE, Transformer vs MLP fusion, full-frame vs lip-region encoding - was not isolated through controlled ablation. The results therefore reflect the combined effect of all design choices, making it impossible to attribute performance to any single component.

*_Fixed two-second window._*
The model analyses a fixed two-second window per inference pass. For videos where the manipulated region is short, begins late, or is distributed across the clip, this window may not capture the most informative segment.

*_Single dataset._*
The model was trained and evaluated entirely on AV-Deepfake1M++. As discussed in Section 2.6.5, deepfake detectors consistently show performance degradation when applied to data from different generators or recording conditions @Dolhansky2020. Cross-dataset generalisation was not evaluated in this project.

*_Limited training budget._*
All training runs were capped at 5 epochs due to the financial and resource constraints of a student project (Section 3.4.3). The default 10-epoch budget in `config.py` was never exercised. As a result, Models 2 and 4 were halted while validation AUC was still improving (epoch 5 AUC 0.9937, epoch 4 AUC 0.9917), and it is unknown whether extended training would have yielded better score calibration, higher test-set performance, or different model ranking. The finding that Model 3 (early-stopped at epoch 3) outperforms later-epoch models must be interpreted within this resource constraint.

// 5.6
== Reflection on the Development Process

The three architectural choices documented in Chapter 3 - audio encoder, visual encoder, and fusion module - were each driven by practical constraints and dataset characteristics.

The decision to replace Wav2Vec 2.0 with a ResNet18 on mel-spectrograms was initially reluctant, as Wav2Vec's contextual speech representations were expected to provide superior sensitivity to voice cloning artefacts. In retrospect, the mel-spectrogram approach proved robust and simple to integrate, and the resulting model achieved strong performance. This suggests that the mel-spectrogram representation captures sufficient information for this task, at least at the validation split scale.

The parallel feature extraction pipeline - using 28 CPU workers with fork-based multiprocessing and crash-resumable manifests - was a significant engineering investment that paid off when cloud instances terminated unexpectedly mid-extraction. Without this system, feature extraction would have needed to restart from the beginning each time.

Training on cloud GPU instances via Vast.ai introduced challenges around data persistence, checkpoint management, and file transfer. The `scp` workflow for downloading checkpoints and the Google Drive integration for Colab runs required careful management to avoid file corruption, as experienced with Model 1. As a student project without institutional GPU access, the financial cost of cloud GPUs limited training runs to 5 epochs, and no hyperparameter sweep was feasible. Future work with access to institutional GPU resources could run longer training schedules, sweep over hyperparameters systematically, and evaluate the full training set rather than only the validation split.

The Weights & Biases integration provided significant value during training, making it possible to monitor convergence, detect overfitting, and compare per-type performance in real time without waiting for the full training run to complete.

The initial proposal specified Python, TensorFlow, SciPy, FFMPEG tools, and CNN as the development stack. In practice, TensorFlow was replaced with PyTorch for its compatibility with the ResNet3D-18 and Transformer architectures; the CNN backbone evolved into a Cross-Modal Transformer Fusion network; and the web interface expanded from a simple classifier to a three-tab interface with model comparison and history tracking. These changes were not anticipated in the initial proposal but proved necessary and beneficial - PyTorch's ecosystem handled the 3D convolutions and Transformer layers more naturally than TensorFlow would have, and the mel-spectrogram + ResNet18 audio approach proved more stable than the initially planned Wav2Vec 2.0 pipeline. The final deliverables exceed the initial proposal's scope in both technical depth and feature richness.

// 5.7
== Conclusion

The evaluation confirms that the implemented system meets its six stated objectives within the resource constraints of a student project. The limitations identified in Section 5.5 represent natural directions for future work.


// Chapter 6
= Conclusion

// 6.1
== Summary of the Project

This dissertation presented the design, implementation, and evaluation of a multimodal audio-visual deepfake detection system built on the AV-Deepfake1M++ dataset, addressing four manipulation types - real, audio-modified, visual-modified, and both-modified - through the Cross-Modal Transformer Fusion architecture described in Chapter 3.

The project's initial proposal (CN6000, 2025) set out to research and create demo software that distinguishes real from deepfake media using deep learning techniques. The implemented system meets and exceeds this aim: a speaker-disjoint evaluation protocol was enforced using GroupShuffleSplit; Focal Loss was adopted in place of standard Binary Cross-Entropy; the initial TensorFlow/CNN stack was replaced with PyTorch and a Cross-Modal Transformer Fusion network; and the demo software was delivered as both a command-line inference tool (`inference.py`) and a web-based interface with model selection, video upload, verdict display, model comparison, and history tracking. The key contributions - a speaker-disjoint evaluation protocol, Focal Loss training, a two-phase encoder fine-tuning schedule, a fully resumable cloud-compatible pipeline with W&B audit logging, and a standalone inference system with web interface - each trace back to the six objectives established in Section 1.4, which themselves evolved from the initial proposal's six objectives.

Four training sessions were conducted, producing four checkpoints (Models 2, 3, and 4; Model 3 is the epoch-3 checkpoint from the same session that produced Model 4 at epoch 5). Three loadable checkpoints were evaluated on a 100-video held-out test set. The best model on the test set was Model 3 (saved at epoch 3, the first fine-tuning epoch), which achieved test AUC 0.937, accuracy 93.0%, precision 1.000, F1 0.837, and zero false positives. While Model 2 achieved a higher peak validation joint AUC of 0.994 by epoch five, its test set accuracy at the 0.5 threshold was lower (66%) due to less optimal score calibration. This comparison, observed within the project's 5-epoch resource constraint, highlights that higher validation AUC does not always translate to better threshold-based performance on unseen speakers, and that earlier checkpoints may produce better-calibrated scores for practical deployment - though further training beyond 5 epochs would be needed to determine whether this pattern holds with a full learning rate schedule.

// 6.2
== Key Findings

The primary finding is that a Cross-Modal Transformer Fusion network trained on a speaker-disjoint subset of AV-Deepfake1M++ can achieve high-fidelity detection performance within three to five training epochs using only the 68,851-video validation split. Five specific findings are noted:

1. _Phase 2 fine-tuning is the essential driver of performance._ Model 1 (1 epoch, Phase 1 only) achieved AUC 0.663; all Phase-2-trained models achieved training AUC ≥ 0.985 and test AUC ≥ 0.915.
2. _Model 3 outperforms Model 2 on the test set despite a lower training AUC._ The model saved at the first fine-tuning epoch achieves better score calibration, zero false positives, 93.0% accuracy, and precision 1.000 - compared to 66.0% accuracy at the same threshold for the further-trained Model 2. This result must be interpreted within the project's resource constraints: all runs were capped at 5 epochs, and it is unknown whether further training would change the model ranking.
3. _Training AUC does not predict test-set accuracy rank under the 5-epoch budget._ Extended fine-tuning - within the epoch range tested - compressed fake scores toward the real distribution, degrading at-threshold accuracy without necessarily reducing ranking quality. Threshold recalibration recovered performance for Model 2 (87.0% at threshold 0.795). Whether this pattern would persist, reverse, or stabilise with longer training (10–20 epochs) is unknown due to resource constraints.
4. _Visual manipulation is most detectable; audio modification is hardest._ `visual_modified` clips produce the lowest mean joint scores; `audio_modified` clips land closest to the decision boundary because the genuine video stream earns a high video head score that counteracts the audio head's fake signal.
5. _The three-head architecture demonstrates genuine modality-specific specialisation_, with audio and video head scores dissociating by manipulation type in the direction predicted by the architecture.

// 6.3
== Limitations and Honest Assessment

The limitations discussed in Section 5.5 - small test set, single dataset, no ablation study, and fixed analysis window - constrain the conclusions that can be drawn. These are acknowledged as directions for future work in Section 6.4.

// 6.4
== Future Work

Several directions would extend and strengthen this work, listed in order of priority.

_Ablation studies._ The most urgent contribution: controlled experiments comparing Focal Loss against standard BCE, Transformer fusion against MLP fusion, and full-frame encoding against lip-region crops would clarify the contribution of each design decision. Without ablations, the current results cannot attribute performance to any individual architectural component.

_Cross-dataset evaluation._ Testing the trained model on FakeAVCeleb, DFDC, or FaceForensics++ would assess whether the learned representations generalise beyond AV-Deepfake1M++, which is the most practically relevant measure of detector reliability and a shared limitation across deepfake detection approaches (Section 2.6.5).

_Full dataset training._ Using the complete AV-Deepfake1M++ training split (over one million clips) rather than only the validation split would expose the model to far greater diversity and likely improve generalisation.

_Temporal localisation._ Extending to frame-level or segment-level predictions would provide richer output and could exploit the `fake_segments` temporal annotations.

_Threshold optimisation._ Optimising the decision threshold on a held-out calibration set to balance false positives and false negatives could improve practical utility.

An improved experiment management infrastructure (cloud storage for checkpoints, web interface enhancements) would reduce operational friction in future training campaigns.

//  6.5
== Personal Reflection

This project was technically more demanding than anticipated. The scale of the AV-Deepfake1M++ dataset - requiring parallel extraction infrastructure, resumable pipelines, and cloud GPU management - transformed what appeared initially to be a modelling problem into a substantial systems engineering challenge. The decisions that ultimately had the most impact on the result were not architectural choices but engineering ones: switching audio loading from librosa to torchaudio to handle corrupted MP4 files, designing the manifest-based resumable extraction system, and implementing the two-phase training schedule to protect pretrained features.

The experience of having Model 1's checkpoint corrupted during download was a practical lesson in the importance of verifying file integrity before terminating server instances. A simple file size check before ending the session would have caught the issue immediately.

Working with a dataset of this scale - 77,326 video clips, 68,851 of which were successfully extracted - provided a realistic experience of the gap between academic benchmark evaluations and the practical difficulties of data engineering at volume. The 8,475 videos that were missing or corrupted on disk, the audio loading failures on non-standard MP4 containers, and the variable frame rates and codec differences across clips all required defensive programming that is rarely described in published papers but is essential in practice.

Overall, the project met its core objective: a functional, well-documented multimodal deepfake detection system that achieves strong performance on a speaker-disjoint evaluation and provides interpretable per-modality output. The codebase is modular, reproducible, and extensible, and represents a solid foundation for the future directions described above.

#bibliography("references.bib", style: "harvard-cite-them-right", title: "References")

// Glossary
#appendix[
= Glossary

This glossary defines technical terminology used throughout this dissertation. Definitions are paraphrased from the cited sources or represent standard usage in the deep learning and digital forensics literature.

== A–D

*AUC (Area Under the ROC Curve).* A threshold-independent metric measuring a classifier's ability to rank samples by authenticity. An AUC of 0.99 means the model correctly ranks 99% of random real/fake pairs. Preferred over accuracy under class imbalance. @Rossler2019 @Dolhansky2020

*AdamW.* A variant of the Adam optimiser that decouples weight decay from gradient updates, providing better regularisation in deep network training.

*ASVspoof.* A challenge series providing standardised benchmarks for audio deepfake and spoofing detection under logical and physical access conditions. @yi2023audiodeepfakedetectionsurvey

*AV-Deepfake1M++.* A large-scale audio-visual deepfake benchmark containing ~2 million video clips, 2,000+ speakers, and four manipulation categories with real-world perturbations. @Cai2025

*BCE (Binary Cross-Entropy).* The standard loss function for binary classification, measuring divergence between predicted probabilities and true labels. Subsumed by Focal Loss when γ = 0.

*CLS token.* A learnable classification token prepended to input sequences in Transformer architectures. Its final representation aggregates information from all inputs and feeds the classification heads.

*Cross-modal attention.* A mechanism allowing representations from different modalities (audio, video) to attend to each other during feature learning, capturing inconsistencies that simple concatenation cannot. @Cai2024

*DataParallel.* A PyTorch wrapper splitting batches across multiple GPUs and synchronising gradients, scaling effective batch size linearly with GPU count.

*DFDC (Deepfake Detection Challenge).* A large-scale benchmark with 100,000+ clips from 3,426 actors, created for a Kaggle competition on deepfake detection. @Dolhansky2020

*Diffusion model.* A generative model that learns to reverse a gradual noising process, producing high-fidelity synthetic speech and images. @shen2023naturalspeech2latentdiffusion

== E–L

*FaceForensics++.* A widely-used face manipulation benchmark with ~4,000 videos and multiple manipulation methods. @Rossler2019

*Focal Loss.* A loss function extending BCE with a modulating factor (1−p_t)^γ that downweights well-classified examples, concentrating training on hard, ambiguous samples. γ = 2.0 in this project. @lin2018focallossdenseobject

*GAN (Generative Adversarial Network).* A framework where generator and discriminator networks compete, enabling realistic synthetic content generation. @Rossler2019

*Gradient clipping.* A technique capping gradient norms to prevent instability from exploding gradients, especially important with 3D convolutions.

*Identity leakage.* A bias where the same speaker appears in both training and validation sets, allowing models to recognise faces/voices rather than detect manipulation. @Rossler2019

*Kanban.* An agile methodology using visual boards to track task status across iterative sprints. @Ahmad2013

== M–R

*Mel-spectrogram.* A 2D time-frequency representation mapping audio frequencies to the mel scale (approximating human perception). Generated via FFT, mel filterbank, and decibel conversion.

*MFCC (Mel Frequency Cepstral Coefficients).* Handcrafted acoustic features representing the short-term power spectrum of speech, used in speaker verification. @Korshunov2018

*NeRF (Neural Radiance Field).* A neural rendering approach synthesising talking-head videos with controllable viewpoint and audio. @Guo2021

*RawNet2.* A deep architecture processing raw waveforms through SincNet filters for audio deepfake detection. @yi2023audiodeepfakedetectionsurvey

*ReduceLROnPlateau.* A scheduler halving the learning rate when a monitored metric stops improving, allowing models to settle into finer minima.

*ResNet3D.* A 3D convolutional network jointly processing spatial and temporal video dimensions, pretrained on Kinetics-400. @tran2018closerlookspatiotemporalconvolutions

== S–Z

*Score calibration.* The alignment between predicted probability scores and actual correctness. Poor calibration (fake scores near 0.5) degrades at-threshold accuracy without reducing AUC.

*Self-attention.* A mechanism allowing each element in a sequence to attend to all others, computing weighted representations. Multi-head runs several attention operations in parallel.

*Sigmoid activation.* A function mapping values to (0, 1), used as the output activation for binary classification.

*SpecAugment.* Data augmentation applying random frequency and time masking to spectrograms for robustness.

*Speaker-disjoint partition.* A dataset split where all videos from a given speaker are assigned exclusively to one subset, ensuring zero overlap. Implemented via `GroupShuffleSplit`.

*t-DCF (tandem Detection Cost Function).* A metric from ASVspoof measuring combined cost of missed detections and false alarms. @yi2023audiodeepfakedetectionsurvey

*Transformer Encoder.* A neural architecture using multi-head self-attention to model dependencies without recurrence. Adapted here for cross-modal fusion.

*Transfer learning.* Initialising a model with weights pretrained on a large dataset and fine-tuning on the target task. @tran2018closerlookspatiotemporalconvolutions

*TTUR (Two Time-Scale Update Rule).* A GAN stabilisation technique using different learning rates for generator and discriminator. @Heusel2017

*TTS (Text-to-Speech) / VC (Voice Conversion).* Deep learning systems for synthesising or converting speech, enabling realistic voice cloning. @yi2023audiodeepfakedetectionsurvey

*Two-phase fine-tuning.* Training where encoder parameters are frozen first (protecting pretrained features), then unfrozen for full-domain adaptation at a lower learning rate.

*XceptionNet.* A CNN using depthwise separable convolutions, achieving state-of-the-art frame-level deepfake detection on FaceForensics++. @Rossler2019
]

// Appendix
#appendix[
= Appendix

  // A
  == Initial Project Proposal (CN6000)

  The initial project proposal for this dissertation was submitted as part of the CN6000 Initial Proposal (2025) module. The proposal outlined the following scope, methodology, and objectives as originally conceived.

  _*Initial Proposal Form*_

  #figure(
    table(
      columns: (auto, auto),
      [_Programme_], [BSc (Hons) Data Science and Artificial Intelligence],
      [_Year_], [2025],
      [_Student Number_], [2571395],
      [_Proposed Title_], [Deepfake detection using deep learning model],
      [_Proposed Aim_], [To research on and create a demo software that will distinguish between real and deepfake media files using deep learning techniques.],
    ),
    caption: [],
  )

  _*Proposed Objectives*_

  By the end of this project, I will be able to:

  1. To explore the literature on the different types of deepfake technologies and how they are created.
  2. To research on the impact of deepfake across various sectors and current solutions.
  3. To conduct quantitative research by performing secondary data analysis on AV-Deepfake1M: A Large-Scale LLM-Driven Audio-Visual Deepfake Dataset.
  4. Design and implement a demo software that detects and differentiates deepfake media using deep learning.
  5. Evaluate the implementation and findings on the model accuracy on detecting audio-visual manipulation.
  6. Reflect on the final outcomes, challenges faced and suggest further development of the software.

  _*Draft of Rationale*_

  Easy access AI to tools lead evolving technologies and effortless media to manipulation, creating highly realistic synthetic media also known as deepfake. Deepfake is coined from both 'deep learning' technology using which it is created and 'fake' meaning a counterfeit.

  The frequent occurrence of such media is reducing the credibility of audio-visual media in areas like law, politics, finance and banking, etc, evolving a threat cybersecurity, public trust, and identity verification systems. Generative models like text-to-speech (TTS), voice conversion (VC) and many more create media that are is difficult to identify making them powerful tools for spreading misinformation and impersonation of political people.

  Many datasets available online provide less diverse or poor-quality data leading to poor performance as they are not trained to work in real world settings. Therefore, existing detection models often misclassify audio-visual deepfakes of higher quality as real. This project will help me to understand how deepfakes are created, what major industries they affect, and to develop an understanding of existing deepfake detection models. I will be able to identify techniques and methods that can increase the quality of accurately classifying data into real and fake.

  _*Facilities Required*_

  Python, TensorFlow, SciPy, FFMPEG tools, Convolutional Neural Network (CNN)

  _*Supervisor*_: Lucian Duta

  Three principal architectural deviations from the initial proposal were required during implementation, each driven by practical constraints encountered with the AV-Deepfake1M++ dataset and the available training infrastructure: the audio encoder changed from Wav2Vec 2.0 to ResNet18 (Section 3.3.4), the visual encoder changed from MobileNetV3 to ResNet3D-18 (Section 3.3.5), and the fusion module changed from DiMoDif to a custom Transformer Encoder (Section 3.3.6). These changes and their justifications are fully documented in Chapter 3.

  // B
  == Final Project Proposal

  The final project proposal reflected the evolved scope and methodology after the literature review and exploratory data analysis were completed. The refined objectives and deliverables are as follows.

  _*Final Proposal Details*_

  #figure(
    table(
      columns: (auto, auto),
      [_Field_], [_Content_],
      [Module], [CN6000 Dissertation (2026)],
      [Title], ["Deepfake Detection Using Cross-Modal Transformer Fusion"],
      [Dataset], [AV-Deepfake1M++ validation split (68,851 usable clips) @Cai2025],
      [Architecture],
      [Cross-Modal Transformer Fusion: ResNet3D-18 (video) + ResNet18 (audio) + 2-layer Transformer Encoder with three-head output],

      [Training Pipeline], [PyTorch, Focal Loss, two-phase fine-tuning, W&B tracking, checkpoint-resumable],
      [Deliverables],
      [Trained model checkpoints; standalone inference system; web-based detection interface with model comparison and history tracking],
    ),
    caption: [Final project proposal summary],
  )

  _*Refined Objectives*_

  The six refined objectives, as stated in Section 1.4, are:
  1. Literature review across generation and detection techniques; gap identification.
  2. Real-world deepfake impact analysis across finance, healthcare, politics, and media.
  3. Quantitative secondary analysis of AV-Deepfake1M++ including EDA and speaker-disjoint split.
  4. Design and implementation of the Cross-Modal Transformer Fusion architecture.
  5. Resumable training pipeline with evaluation on a 100-video test set.
  6. Standalone inference system and web interface for non-technical use.

  // C
  == Application for Approval of Research Activities

  This project uses a publicly available, pre-existing dataset (AV-Deepfake1M++) distributed under a research-only license. No primary data collection involving human participants was conducted. The dataset was collected by its creators under established data use agreements, and all recordings feature consenting individuals.

  The research activities fall under secondary data analysis and software development. No ethical approval was required for:
  - Secondary analysis of a publicly available benchmark dataset.
  - Development and evaluation of machine learning models.
  - Software engineering for the web interface.

  The project was conducted in accordance with the University's guidelines on research ethics and data protection. No personally identifiable information was processed or stored outside the dataset's original distribution. The dataset's consent framework and terms of use are documented in Appendix D.

  #figure(
    image("figures/ethics_approval_1.png", width: 90%),
    caption: [CN6000 Internal Ethical Approval Process (2025) - page 1],
  )

  #figure(
    image("figures/ethics_approval_2.png", width: 90%),
    caption: [CN6000 Internal Ethical Approval Process (2025) - page 2],
  )

  #figure(
    image("figures/ethics_approval_3.png", width: 90%),
    caption: [CN6000 Internal Ethical Approval Process (2025) - page 3],
  )

  // D
  == Dataset Usage and Consent

  The AV-Deepfake1M++ dataset @Cai2025 is distributed under a research-only license through Hugging Face and the official 1M-Deepfakes Detection Challenge platform. The dataset terms of use include the following provisions:

  #figure(
    table(
      columns: (auto, auto),
      [_Provision_], [_Compliance_],
      [Research-only use],
      [This dissertation constitutes academic research; no commercial application was developed or deployed.],

      [No redistribution],
      [The dataset was not redistributed. Only the validation split was downloaded and processed on cloud GPU instances. All extracted features and trained models remain on the author's local machine.],

      [Attribution],
      [The dataset is cited throughout this dissertation as @Cai2025. The official dataset paper and website are referenced in the bibliography.],

      [Consent of individuals],
      [The dataset creators obtained consent from all recorded individuals. All subjects agreed to participate and to have their likenesses modified during dataset construction @Cai2025.],

      [Non-consensual content prohibition],
      [The dataset explicitly prohibits the use of its data for generating non-consensual deepfake content. This project uses the dataset exclusively for detection research, which is aligned with harm mitigation.],
    ),
    caption: [Dataset usage terms and compliance statement],
  )

  The author confirms that all dataset usage complies with the terms set by the dataset creators and that no terms were violated during this research project.

  #figure(
    image("figures/participation_form_1.png", width: 90%),
    caption: [Participation Form for 1 Million Deepfakes Detection Challenge at ACM Multimedia 2025 - page 1],
  )

  #figure(
    image("figures/participation_form_2.png", width: 90%),
    caption: [Participation Form for 1 Million Deepfakes Detection Challenge at ACM Multimedia 2025 - page 2],
  )

  #figure(
    image("figures/eula.png", width: 90%),
    caption: [End User License Agreement - signed 04 December 2025],
  )

  // E
  == Hyperparameter Configuration

  All hyperparameters were centralised in `config.py`. The values used across all training runs are listed below.

  #figure(
    table(
      columns: (auto, auto, auto),
      [_Parameter_], [_Value_], [_Description_],
      [`feature_dim`], [`256`], [Encoder output dimension for both ResNet3D-18 and ResNet18],
      [`hidden_dim`], [`512`], [Fusion module hidden layer width],
      [`dropout`], [`0.4`], [Applied to encoder FC layers and fusion module],
      [`batch_size`], [`8`], [Per-GPU batch size],
      [`epochs`], [`10`], [Default epoch budget (capped at 5 in practice)],
      [`freeze_epochs`], [`max(1, round(epochs × 0.25))`], [Epochs with encoders frozen (auto-computed)],
      [`patience`], [`max(5, round(epochs × 0.30))`], [Epochs without improvement before early stop],
      [`focal_gamma`], [`2.0`], [Focal Loss focusing parameter],
      [`focal_alpha`], [`0.25`], [Focal Loss class balance weight],
      [`learning_rate`], [`$1 × 10^{-4}$`], [Fusion module learning rate],
      [`encoder_lr`], [`$1 × 10^{-5}$`], [Encoder fine-tuning learning rate (10× lower)],
      [`weight_decay`], [`$1 × 10^{-4}$`], [AdamW weight decay],
      [`grad_clip`], [`1.0`], [Maximum gradient norm before clipping],
      [`val_split`], [`0.2`], [Validation split fraction],
      [`scheduler_factor`], [`0.5`], [ReduceLROnPlateau LR reduction factor],
      [`scheduler_patience`], [`5`], [Epochs before LR reduction],
      [`seed`], [`42`], [Global random seed for reproducibility],
    ),
    caption: [Complete hyperparameter configuration],
  )

  The Focal Loss implementation (`train_utils.py`) downweights well-classified examples through a $(1-p_t)^gamma$ modulating factor, concentrating training on hard, ambiguous samples near the decision boundary:

  ```python
  class FocalLoss(nn.Module):
      def __init__(self, gamma=2.0, alpha=0.25):
          super().__init__()
          self.gamma = gamma
          self.alpha = alpha

      def forward(self, pred, target):
          pred = torch.clamp(pred, 1e-6, 1.0 - 1e-6)
          bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
          p_t = pred * target + (1 - pred) * (1 - target)
          focal_weight = (1 - p_t) ** self.gamma
          alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
          return (alpha_t * focal_weight * bce).mean()
  ```

  // F
  == Model Architecture Summary

  #figure(
    table(
      columns: (auto, auto, auto),
      [_Component_], [_Architecture_], [_Parameters (approx.)_],
      [Video Encoder], [ResNet3D-18 (Kinetics-400 pretrained)], [~33.3M],
      [Audio Encoder], [ResNet18 (ImageNet pretrained; 1-channel conv1)], [~11.7M],
      [Transformer Fusion], [2-layer Encoder, 8 heads, 512-dim, GELU], [~4.7M],
      [Classification Heads], [3 × Linear(512→1) + Sigmoid], [~1.5K],
      [_Total_], [-], [_~49.7M_],
    ),
    caption: [Model architecture component summary],
  )

  The core fusion module (`cross_modal.py`) uses a Transformer Encoder with a learnable [CLS] token to fuse 256-dim audio and video features into a 512-dim joint representation, then applies three independent sigmoid classification heads:

  ```python
  class TransformerFusion(nn.Module):
      def __init__(self, feature_dim=256, hidden_dim=512,
                   num_heads=8, num_layers=2, dropout=0.4):
          super().__init__()
          self.audio_proj  = nn.Linear(feature_dim, hidden_dim)
          self.video_proj  = nn.Linear(feature_dim, hidden_dim)
          self.cls_token   = nn.Parameter(torch.randn(1, 1, hidden_dim))
          self.pos_embedding = nn.Parameter(torch.randn(1, 3, hidden_dim))

          encoder_layer = nn.TransformerEncoderLayer(
              d_model=hidden_dim, nhead=num_heads,
              dim_feedforward=hidden_dim * 4, dropout=dropout,
              activation='gelu', batch_first=True, norm_first=True
          )
          self.transformer = nn.TransformerEncoder(
              encoder_layer, num_layers=num_layers,
              norm=nn.LayerNorm(hidden_dim)
          )
          self.audio_classifier = nn.Linear(hidden_dim, 1)
          self.video_classifier = nn.Linear(hidden_dim, 1)
          self.joint_classifier = nn.Linear(hidden_dim, 1)

      def forward(self, video_feat, audio_feat):
          B = video_feat.shape[0]
          v   = self.video_proj(video_feat).unsqueeze(1)
          a   = self.audio_proj(audio_feat).unsqueeze(1)
          cls = self.cls_token.expand(B, -1, -1)
          tokens = torch.cat([cls, v, a], dim=1) + self.pos_embedding
          fused  = self.transformer(tokens)
          cls_out = fused[:, 0, :]
          return {
              'audio_pred': torch.sigmoid(self.audio_classifier(cls_out)),
              'video_pred': torch.sigmoid(self.video_classifier(cls_out)),
              'joint_pred': torch.sigmoid(self.joint_classifier(cls_out)),
          }
  ```

  // G
  == Training Run Summary

  #figure(
    table(
      columns: (auto, auto, auto, auto, auto, auto),
      [_Run_], [_Epochs_], [_Best Val Joint AUC_], [_Test AUC_], [_Test Acc at 0.5_], [_Status_],
      [Model 1], [1], [0.6626], [N/A], [N/A], [Corrupted on download],
      [Model 2], [5], [0.9937], [0.9189], [66.0%], [Intact],
      [Model 3], [3], [0.9851], [0.9371], [93.0%], [Intact (early stop)],
      [Model 4], [5], [0.9925], [0.9152], [71.0%], [Intact],
    ),
    caption: [Summary of all training runs with validation and test metrics],
  )

  // H
  == Feature Extraction Parameters

  #figure(
    table(
      columns: (auto, auto),
      [_Parameter_], [_Value_],
      [Audio sample rate], [16,000 Hz],
      [FFT window], [1,024 points],
      [Hop length], [512 samples],
      [Mel frequency bins], [128],
      [Audio clip duration], [2.0 seconds],
      [Audio samples per clip], [32,000],
      [Spectrogram shape], [1 × 128 × 63],
      [Video FPS], [25],
      [Video frames per clip], [50],
      [Frame size], [224 × 224 pixels],
      [Video tensor shape], [50 × 3 × 224 × 224],
      [Augmentation (audio)], [SpecAugment: frequency masking (≤20 bins), time masking (≤15 steps)],
      [Augmentation (video)], [Random horizontal flip, brightness jitter (±0.2), contrast jitter (0.8–1.2)],
    ),
    caption: [Feature extraction parameters],
  )

  // I
  == Dataset Cleaning Summary

  #figure(
    table(
      columns: (auto, auto, auto, auto),
      [_Stage_], [_Criterion_], [_Remaining_], [_Dropped_],
      [Initial], [All val_metadata.json entries], [77,326], [-],
      [Stage 1], [audio_frames $>$ 0 and video_frames $>$ 0], [77,115], [211 (zero audio)],

      [Stage 2], [File confirmed present on disk], [68,851], [8,264 (missing/corrupted)],
      [_Final usable_], [-], [_68,851 (89%)_], [-],
    ),
    caption: [Dataset cleaning summary - from raw metadata to final usable dataset],
  )

  The speaker-disjoint partition (`data_utils.py`) extracts speaker IDs from file paths and uses `GroupShuffleSplit` to ensure zero speaker overlap between training and validation sets:

  ```python
  # Extract speaker IDs from file paths (e.g., "source/id00015/clip.mp4")
  df['speaker'] = df['file'].apply(
      lambda f: f.split('/')[1] if '/' in f else 'unknown'
  )
  # Split by speaker - all videos from one speaker stay in the same split
  gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
  train_idx, val_idx = next(gss.split(df, groups=df['speaker']))
  train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
  # Verify zero overlap
  overlap = set(train_df['speaker'].unique()) & set(val_df['speaker'].unique())
  assert len(overlap) == 0, f"Speaker overlap detected: {overlap}"
  ```

  // J
  == Test Set Composition

  #figure(
    table(
      columns: (auto, auto, auto),
      [_Manipulation Type_], [_Count_], [_Percentage_],
      [`real`], [25], [25.0%],
      [`audio_modified`], [25], [25.0%],
      [`visual_modified`], [25], [25.0%],
      [`both_modified`], [25], [25.0%],
      [_Total_], [_100_], [_100%_],
    ),
    caption: [100-video test set composition - 25 videos per manipulation type],
  )

  // K
  == Figure Attribution

  All figures in this dissertation were either generated programmatically from project code and results, or created using generative AI tools for conceptual diagrams. The attribution is as follows.

  #figure(
    table(
      columns: (26%, 46%, 8%, 20%),
      [_Figure_], [_Filename_], [_Section_], [_Source_],
      [System architecture], [`architecture.png`], [3.3.1], [AI-generated],
      [Speaker-disjoint split], [`speaker_split.png`], [3.3.2], [AI-generated],
      [Transformer fusion module], [`transformer_fusion_module.png`], [3.3.6], [AI-generated],
      [Two-phase training timeline], [`two_phase_training.png`], [3.4.2], [AI-generated],
      [Modification type distribution], [`modification_type_distribution.png`], [3.3.3], [\ `analyze_data.py`],
      [Fake segment analysis], [`fake_segment_analysis.png`], [3.3.3], [\ `analyze_data.py`],
      [Mel-spectrogram comparison], [`mel_spectrogram_comparison.png`], [3.3.4], [\ `plot_mel_spectrogram.py`],
      [Training history (all)], [`training_history.png`], [4.2], [\ `compare_models.py`],
      [Training history (Model 4)], [`training_history_model4.png`], [4.2.3], [\ `plot_training_history.py`],
      [Model comparison], [`model_comparison.png`], [4.3.1], [\ `compare_models.py`],
      [Per-type bar chart], [`per_type_accuracy_bar_chart.png`], [4.4], [\ `plot_per_type_accuracy.py`],
      [Calibration curves], [`calibration_curves.png`], [4.3.2], [\ `plot_calibration_curves.py`],
    ),
    caption: [Figure attribution - AI-generated conceptual diagrams vs. code-generated plots from project results],
  )

  All code-generated figures use data from the project's training runs (logs/), prediction outputs (comparison_results/), and dataset analysis (analysis/). The plotting scripts are available in the project repository and can be reproduced by running the corresponding Python files with the project's data.

  // L
  == Web Interface Screenshots

  The following screenshots demonstrate the browser-based web interface (`web/app.py`) described in Section 3.7.1. The interface was accessed at `http://localhost:5000` using a local browser. To reproduce: run `python web/app.py` with the model environment variables set, then open the URL in a browser.

  #figure(
    image("figures/web_analyze_empty.png", width: 90%),
    caption: [Analyze tab - initial state showing model selector dropdown, drag-and-drop upload area, and threshold slider.],
  )

  #figure(
    image("figures/web_analyze_real.png", width: 90%),
    caption: [Analyze tab - a real video classified with green "REAL" verdict, displaying audio, video, and joint authenticity scores.],
  )

  #figure(
    image("figures/web_analyze_fake.png", width: 90%),
    caption: [Analyze tab - a fake video classified with red "FAKE" verdict, showing per-modality score breakdown.],
  )

  #figure(
    image("figures/web_compare.png", width: 90%),
    caption: [Compare tab - the same video evaluated by two models side by side, showing both verdicts, three per-modality scores per model, and an agree/disagree summary.],
  )

  #figure(
    image("figures/web_history.png", width: 90%),
    caption: [History tab - SQLite-backed table of past analyses showing filename, verdict, joint score, model used, and timestamp, with per-entry delete and bulk clear options.],
  )

  // M
  == Project Gantt Chart

  #figure(
    image("figures/gantt_initial.png", width: 90%),
    caption: [Initial project Gantt chart — submitted as part of the CN6000 proposal (January 2026).],
  )

  #figure(
    image("figures/gantt_final.png", width: 90%),
    caption: [Final project Gantt chart — updated to reflect actual progress and infrastructure delays (May 2026).],
  )

]

