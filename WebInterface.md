# Web Interface — AV Deepfake Detector

A browser-based tool for selecting a model, uploading a video, checking accuracy metrics, and comparing models. Wraps `inference.py` — no ML knowledge required.

---

## Overview

```
Select model → Upload video → Get verdict + scores
                ↓
           (Optional) Compare both models side-by-side
                ↓
           (Optional) Review history of past analyses
```

The interface wraps `inference.py` in a minimal Flask app. All model architecture and inference logic is imported directly from `inference.py` — no duplication.

---

## Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| **Frontend** | Single self-contained HTML | No build step, CSS inline in HTML |
| **Backend** | Flask | Lightweight, Python server |
| **Model serving** | PyTorch (CPU or GPU) | Same weights as training |
| **Inference** | Imported from `inference.py` | `load_model()`, `predict_video()` |
| **History** | SQLite | Lightweight, no separate service needed |
| **File handling** | Temp directory | Uploaded videos deleted after analysis |

---

## File Structure

```
web/
├── app.py              # Flask server — 5 API endpoints + SQLite history + model cache
├── templates/
│   └── index.html      # Single-page app with 3 tabs (Analyze/Compare/History), ~400 lines
├── static/             # Legacy static assets — not loaded by index.html
│   ├── app.js          # (orphaned)
│   └── style.css       # (orphaned)
├── history.db          # SQLite database created automatically at first run
└── .gitignore          # Ignores history.db
```

The frontend is entirely self-contained in `templates/index.html` (~400 lines). No external JS/CSS files are loaded.

---

## Pages

### 1. Analyze Tab (default)

**Layout:** Single centered card on dark background.

| Element | Description |
|---|---|
| **Dropzone** | Drag-and-drop or click to upload. Accepts `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.m4v` |
| **File info** | Shows filename and size after selection |
| **Model selector** | Dropdown — pick Model 1 or Model 2 (configured via `MODEL1_PATH` / `MODEL2_PATH` env vars) |
| **Analyze button** | Disabled until a file is selected |
| **Verdict + scores** | Shows after analysis: REAL/FAKE badge, confidence %, audio/video/joint scores |
| **Device badge** | Shows CPU or CUDA at bottom of card |
| **Reset button** | Returns to upload state |

**Result display:**
```
VERDICT: REAL         Confidence: 85%
Audio    ████████████  0.892
Video    ██████████░░  0.734
Joint    ████████████  0.892
```

### 2. Compare Tab

Upload a single video and run it through **both** models simultaneously for side-by-side comparison.

| Element | Description |
|---|---|
| **Dropzone** | Accepts one video file |
| **Compare button** | Disabled until file selected |
| **Side-by-side columns** | Model 1 results and Model 2 results, each with audio/video/joint scores and verdict |
| **Summary line** | Shows whether models agree or disagree on the verdict |

### 3. History Tab

Table of all past analyses stored in `history.db`.

| Element | Description |
|---|---|
| **Count** | Shows total number of stored analyses |
| **Table columns** | File, Verdict tag, Joint score, Model used, Delete button |
| **Clear All** | Wipes entire history after confirmation |
| **Per-row delete** | Click ✕ to delete individual entries |

History is saved automatically on every analysis. Each entry stores: `id`, `filename`, `model_key`, `joint_score`, `audio_score`, `video_score`, `confidence`, `threshold`, `verdict`, `timestamp`.

**Schema migration:** On startup, `init_db()` checks the actual column count of the `analyses` table against the expected number (10). If they differ — e.g., because `history.db` was created by an older version of the code — the old table is dropped and recreated with the current schema. The INSERT in `save_analysis()` also uses explicit column names rather than positional `VALUES`, making it resilient to future column reordering.

---

## API Endpoints

### `GET /api/models`

List available models and device info.

**Response:**
```json
{
  "model1": {
    "path": ".../logs/logs_2/best_model.pth",
    "available": true,
    "label": "Model 2 — Val AUC 0.994 (5 epochs)",
    "note": "Best validation AUC; use threshold 0.795+ for balanced accuracy"
  },
  "model2": {
    "path": ".../logs/logs_3/best_model.pth",
    "available": true,
    "label": "Model 3 — Test AUC 0.937 (3 epochs)",
    "note": "Best practical model: 93% accuracy, zero false positives"
  },
  "device": "cuda"
}
```

Model paths are hardcoded to `logs/logs_2/best_model.pth` and `logs/logs_3/best_model.pth` (the two best models from training runs). The `MODEL1_PATH` and `MODEL2_PATH` environment variables override these defaults.

### `POST /api/analyze`

Upload a video for analysis with a selected model.

**Request:** `multipart/form-data`
- `video` — video file
- `model` — `"model1"` or `"model2"` (default: `"model1"`)
- `threshold` — float (default: `0.5`)
- `n_windows` — int, number of windows to average (default: `3`)

**Response:**
```json
{
  "file": "video.mp4",
  "audio_score": 0.8921,
  "video_score": 0.7342,
  "joint_score": 0.8921,
  "verdict": "REAL",
  "confidence": 0.7842,
  "threshold": 0.5,
  "model_key": "model1",
  "id": "uuid-of-analysis"
}
```

### `POST /api/compare`

Run the same video through both models side-by-side.

**Request:** `multipart/form-data`
- `video` — video file
- `threshold` — float (default: `0.5`)
- `n_windows` — int (default: `3`)

**Response:**
```json
{
  "filename": "video.mp4",
  "threshold": 0.5,
  "results": {
    "model1": {
      "audio_score": 0.8921, "video_score": 0.7342,
      "joint_score": 0.8921, "verdict": "REAL", "confidence": 0.7842
    },
    "model2": {
      "audio_score": 0.1234, "video_score": 0.5678,
      "joint_score": 0.1234, "verdict": "FAKE", "confidence": 0.7532
    }
  }
}
```

### `GET /api/history`

Fetch past analyses.

**Response:**
```json
{
  "analyses": [
    {
      "id": "uuid", "filename": "video.mp4", "model_key": "model1",
      "joint_score": 0.8921, "audio_score": 0.8921, "video_score": 0.7342,
      "confidence": 0.7842, "threshold": 0.5, "verdict": "REAL",
      "timestamp": "2026-04-28T20:00:00"
    }
  ]
}
```

### `DELETE /api/history/<id>`

Delete a single history entry.

### `DELETE /api/history`

Clear all history.

---

## UI Flow

```
User opens page
      ↓
tab: Analyze ──────────────────────────────→ tab: Compare ──→ tab: History
      ↓                                         ↓                    ↓
drag/drop or click                            drag/drop           table of past
to upload video                               single video        analyses
      ↓                                         ↓                    ↓
pick model from dropdown                      click Compare       delete individual
click Analyze                                 Models button       or Clear All
      ↓                                         ↓                    ↓
show loader                                   run both models     (no re-analysis
      ↓                                         side-by-side       in history tab)
show VERDICT + scores                   show side-by-side
click "Analyze Another"                    agree/disagree
      ↓                                      summary
back to upload
```

---

## Design

| Aspect | Value |
|---|---|
| **Background** | `#0f172a` (deep navy) |
| **Surface** | `#1e293b` (card background) |
| **REAL color** | `#22c55e` (green) |
| **FAKE color** | `#ef4444` (red) |
| **Accent** | `#38bdf8` (sky blue) |
| **Audio score color** | `#a78bfa` (purple) |
| **Video score color** | `#f472b6` (pink) |
| **Joint score color** | `#38bdf8` (sky blue) |
| **Typography** | System UI / -apple-system / sans-serif |
| **Max file size** | 500MB (configurable via `MAX_UPLOAD_MB` env var) |
| **Responsive** | Single column, works on mobile |

---

## Running

```bash
# Model paths are hardcoded by default (logs/logs_2, logs/logs_3).
# Override with env vars if needed:
export MODEL1_PATH=/path/to/model1.pth
export MODEL2_PATH=/path/to/model2.pth

# Optional: set DB path and upload limits
export DB_PATH=web/history.db
export MAX_UPLOAD_MB=500

# Run
python web/app.py
# → http://localhost:5000
```

---

## What Was Removed (v1 → v2 simplification)

| Feature | Reason |
|---|---|
| PDF report generation | Not required for core use case |
| Per-window score breakdown UI | Kept multi-window inference on backend, simplified frontend |
| Mel spectrogram display | Added visual complexity, low value for simple metrics check |
| Explanation panel | Overhead for simple verdict display |
| Model architecture in `app.py` | Now imported from `inference.py` |
| `/api/health` endpoint | Redundant with basic fetch |
| Per-model epoch/AUC metadata in UI | Not critical for simple usage |

---

## Security Considerations

- Validate file extension server-side
- Limit upload size (default 500MB)
- Uploaded videos deleted immediately after processing (temp directory)
- No user data stored permanently beyond history (which can be cleared)