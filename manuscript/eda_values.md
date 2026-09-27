# Exploratory Data Analysis (EDA)

The AV-Deepfake1M++ dataset was analyzed to understand its scale, composition, and characteristics. Below are the key statistics derived from the validation set metadata.

---

## Dataset Scale & Composition

| Metric | Value |
|--------|-------|
| Total videos | 77,326 |
| Total speakers | 1,835 |
| Videos per speaker (mean) | 42.1 |
| Videos per speaker (median) | 20 |
| Videos per speaker (min) | 1 (speaker id08916) |
| Videos per speaker (max) | 1,417 (speaker id02760) |

**Modification Type Distribution:**

| Modification Type | Count | Percentage |
|-------------------|-------|------------|
| real | 20,220 | 26.1% |
| visual_modified | 19,099 | 24.7% |
| both_modified | 19,069 | 24.7% |
| audio_modified | 18,938 | 24.5% |

---

## Video Characteristics

| Metric | Value |
|--------|-------|
| Video frame count (mean) | 239 |
| Video frame count (median) | 192 |
| Video frame count (min) | 63 |
| Video frame count (max) | 3,810 |

---

## Audio Characteristics

| Metric | Value |
|--------|-------|
| Audio frame count (mean) | 184,984 |
| Audio frame count (median) | 126,528 |
| Audio frame count (min) | 0 |
| Audio frame count (max) | 6,719,488 |
| Videos with zero audio frames | 211 |

---

## Speaker Diversity

**Unique speakers per modification type:**

| Modification Type | Unique Speakers |
|-------------------|-----------------|
| real | 1,731 |
| visual_modified | 1,707 |
| both_modified | 1,522 |
| audio_modified | 1,498 |

**Speaker coverage:**

| Metric | Value |
|--------|-------|
| Speakers with all 4 modification types | 1,336 |
| Speakers with < 4 modification types | 499 |

---

## Fake Segment Analysis

For fake videos (excluding `real` modification type), the dataset contains:

| Metric | Value |
|--------|-------|
| Total fake videos analyzed | 57,106 |
| Total fake segments | 77,679 |
| Segments per video (mean) | 1.4 |
| Segments per video (max) | 4 |
| Segment duration (mean) | 0.33 seconds |
| Segment duration (median) | 0.30 seconds |
| Segment duration (min) | 0.02 seconds |
| Segment duration (max) | 8.10 seconds |

---

## Real vs. Fake Proportion

| Category | Count | Percentage |
|----------|-------|------------|
| Real videos | 20,220 | 26.1% |
| Fake videos (any modification) | 57,106 | 73.9% |

---

## Key Observations

1. **Balanced modification types**: The dataset exhibits near-uniform distribution across all four modification types (~24.5-26.1% each).

2. **High speaker diversity**: With 1,835 speakers and a mean of 42.1 videos per speaker, the dataset provides substantial per-speaker samples for learning individual speaking patterns.

3. **Sparse fake segments**: The mean of 1.4 segments per fake video with a maximum of 4 indicates that fake modifications are typically localized rather than spanning entire videos.

4. **Short fake segments**: The mean fake segment duration of 0.33 seconds (median 0.30s) suggests that most modifications are brief and targeted rather than covering long temporal spans.

5. **Audio quality concern**: 211 videos (0.27%) have zero audio frames, which may require filtering or special handling during training.

6. **Speaker coverage**: 72.8% of speakers (1,336 out of 1,835) have samples across all 4 modification types, enabling balanced comparative analysis.