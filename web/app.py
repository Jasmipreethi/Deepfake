"""
AV Deepfake Detection — Web Application Backend

Flask server that exposes REST endpoints for the deepfake detection website.
Reuses all inference logic from inference.py — no duplication.

Features:
  1. Single + Batch video upload and classification
  3. Explanation panel (per-modality scores, window timeline, mel-spectrogram)
  4. Model comparison mode (two models on same video)
  5. History dashboard (all past analyses stored in SQLite)
  6. Confidence threshold slider (per-request)
  7. PDF report generation per video

Usage:
    # Install extras beyond inference.py requirements
    pip install flask flask-cors reportlab matplotlib

    # Set model paths in .env or environment
    export MODEL1_PATH=/content/drive/MyDrive/best_model_1.pth
    export MODEL2_PATH=/content/drive/MyDrive/best_model_2.pth

    python app.py
    # → http://localhost:5000
"""

import os
import sys
import json
import uuid
import sqlite3
import warnings
import tempfile
import datetime
import base64
import io
import threading
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Flask ──────────────────────────────────────────────────────────────────
from flask import Flask, request, jsonify, send_file, send_from_directory, abort

try:
    from flask_cors import CORS
except ImportError:
    CORS = None

# ── ML / inference ─────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torchaudio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Re-use architecture and extraction from inference.py
try:
    from config import FEATURE_CONFIG
except ImportError:
    FEATURE_CONFIG = {
        "sr": 16000,
        "fps": 25,
        "duration": 2.0,
        "num_frames": 50,
        "img_size": 224,
        "audio_samples": 32000,
        "n_fft": 1024,
        "hop_length": 512,
        "target_t": 63,
        "n_mels": 128,
    }

from torchvision.models import resnet18
from torchvision.models.video import r3d_18


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE (identical to inference.py)
# ─────────────────────────────────────────────────────────────────────────────


class PretrainedAudioEncoder(nn.Module):
    def __init__(self, feature_dim=256, dropout=0.4, intermediate_dim=512):
        super().__init__()
        b = resnet18(weights=None)
        b.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        b.fc = nn.Sequential(
            nn.Linear(b.fc.in_features, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, feature_dim),
        )
        self.backbone = b

    def forward(self, x):
        return self.backbone(
            F.interpolate(x, (224, 224), mode="bilinear", align_corners=False)
        )


class PretrainedVideoEncoder(nn.Module):
    def __init__(self, feature_dim=256, dropout=0.4, intermediate_dim=512):
        super().__init__()
        b = r3d_18(weights=None)
        b.fc = nn.Sequential(
            nn.Linear(b.fc.in_features, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, feature_dim),
        )
        self.backbone = b

    def forward(self, x):
        return self.backbone(x.permute(0, 2, 1, 3, 4))


class TransformerFusion(nn.Module):
    def __init__(
        self, feature_dim=256, hidden_dim=512, num_heads=8, num_layers=2, dropout=0.4
    ):
        super().__init__()
        self.audio_proj = nn.Linear(feature_dim, hidden_dim)
        self.video_proj = nn.Linear(feature_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 3, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(hidden_dim)
        )
        self.audio_cls = nn.Linear(hidden_dim, 1)
        self.video_cls = nn.Linear(hidden_dim, 1)
        self.joint_cls = nn.Linear(hidden_dim, 1)

    def forward(self, vf, af):
        B = vf.shape[0]
        tokens = (
            torch.cat(
                [
                    self.cls_token.expand(B, -1, -1),
                    self.video_proj(vf).unsqueeze(1),
                    self.audio_proj(af).unsqueeze(1),
                ],
                dim=1,
            )
            + self.pos_embedding
        )
        c = self.transformer(tokens)[:, 0, :]
        return {
            "audio_pred": torch.sigmoid(self.audio_cls(c)),
            "video_pred": torch.sigmoid(self.video_cls(c)),
            "joint_pred": torch.sigmoid(self.joint_cls(c)),
        }


class PretrainedFusion(nn.Module):
    def __init__(self, feature_dim=256, hidden_dim=512, dropout=0.4):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
        )
        self.audio_cls = nn.Linear(hidden_dim, 1)
        self.video_cls = nn.Linear(hidden_dim, 1)
        self.joint_cls = nn.Linear(hidden_dim, 1)

    def forward(self, vf, af):
        f = self.fusion(torch.cat([vf, af], dim=1))
        return {
            "audio_pred": torch.sigmoid(self.audio_cls(f)),
            "video_pred": torch.sigmoid(self.video_cls(f)),
            "joint_pred": torch.sigmoid(self.joint_cls(f)),
        }


class AVDeepfakeDetector(nn.Module):
    def __init__(
        self, fusion_type="transformer", feature_dim=256, hidden_dim=512, dropout=0.4
    ):
        super().__init__()
        self.video_encoder = PretrainedVideoEncoder(feature_dim, dropout)
        self.audio_encoder = PretrainedAudioEncoder(feature_dim, dropout)
        self.fusion_module = (
            TransformerFusion(feature_dim, hidden_dim, dropout=dropout)
            if fusion_type == "transformer"
            else PretrainedFusion(feature_dim, hidden_dim, dropout)
        )

    def forward(self, video, audio):
        return self.fusion_module(self.video_encoder(video), self.audio_encoder(audio))


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────


def _extract_at(path, start_sec, cfg=FEATURE_CONFIG):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, None, None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * cfg["fps"]))
    frames = []
    for _ in range(cfg["num_frames"]):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (cfg["img_size"], cfg["img_size"]))
        frames.append(frame / 255.0)
    cap.release()
    if not frames:
        return None, None, None
    while len(frames) < cfg["num_frames"]:
        frames.append(frames[-1])
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    video_tensor = torch.FloatTensor((np.array(frames) - mean) / std).permute(
        0, 3, 1, 2
    )

    mel_img = None
    try:
        waveform, orig_sr = torchaudio.load(path)
        if orig_sr != cfg["sr"]:
            waveform = torchaudio.transforms.Resample(orig_sr, cfg["sr"])(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        s = int(start_sec * cfg["sr"])
        waveform = waveform[:, s : s + cfg["audio_samples"]]
        if waveform.shape[1] < cfg["audio_samples"]:
            waveform = F.pad(waveform, (0, cfg["audio_samples"] - waveform.shape[1]))
        n_mels = cfg.get("n_mels", 128)
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg["sr"],
            n_mels=n_mels,
            n_fft=cfg["n_fft"],
            hop_length=cfg["hop_length"],
        )(waveform)
        audio_tensor = torchaudio.transforms.AmplitudeToDB(
            top_db=cfg.get("top_db", 80)
        )(mel)
        std_v = audio_tensor.std()
        if std_v > 0:
            audio_tensor = (audio_tensor - audio_tensor.mean()) / (std_v + 1e-6)
        t = audio_tensor.shape[2]
        target_t = cfg["target_t"]
        if t < target_t:
            audio_tensor = F.pad(audio_tensor, (0, target_t - t))
        elif t > target_t:
            audio_tensor = audio_tensor[:, :, :target_t]
        # Render mel-spectrogram as base64 PNG for explanation panel
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.imshow(
            audio_tensor[0].numpy(),
            aspect="auto",
            origin="lower",
            cmap="magma",
            interpolation="nearest",
        )
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=80)
        plt.close(fig)
        mel_img = base64.b64encode(buf.getvalue()).decode()
    except Exception:
        audio_tensor = torch.zeros(1, cfg.get("n_mels", 128), cfg["target_t"])

    return video_tensor, audio_tensor, mel_img


def predict_with_windows(model, path, device, n_windows=3, threshold=0.5):
    """Run multi-window inference and return detailed per-window breakdown."""
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 50
    fps = cap.get(cv2.CAP_PROP_FPS) or FEATURE_CONFIG["fps"]
    cap.release()
    total_sec = total_frames / fps
    max_start = max(0.0, total_sec - FEATURE_CONFIG["duration"])
    starts = (
        [max_start * i / (n_windows - 1) for i in range(n_windows)]
        if n_windows > 1 and max_start > 0
        else [0.0]
    )

    window_results = []
    mel_images = []
    preds = {"audio": [], "video": [], "joint": []}

    for start in starts:
        v, a, mel_img = _extract_at(path, start)
        if v is None:
            continue
        with torch.no_grad():
            out = model(v.unsqueeze(0).to(device), a.unsqueeze(0).to(device))
        ap = out["audio_pred"].item()
        vp = out["video_pred"].item()
        jp = out["joint_pred"].item()
        preds["audio"].append(ap)
        preds["video"].append(vp)
        preds["joint"].append(jp)
        window_results.append(
            {
                "start_sec": round(start, 2),
                "end_sec": round(start + FEATURE_CONFIG["duration"], 2),
                "audio_score": round(ap, 4),
                "video_score": round(vp, 4),
                "joint_score": round(jp, 4),
            }
        )
        if mel_img:
            mel_images.append(mel_img)

    if not window_results:
        return None

    audio_score = sum(preds["audio"]) / len(preds["audio"])
    video_score = sum(preds["video"]) / len(preds["video"])
    joint_score = sum(preds["joint"]) / len(preds["joint"])
    verdict = "REAL" if joint_score >= threshold else "FAKE"

    # Explanation: which modality drove the verdict
    if audio_score < threshold and video_score >= threshold:
        triggered_by = "audio"
        explanation = "Audio anomalies detected — voice or speech appears manipulated."
    elif video_score < threshold and audio_score >= threshold:
        triggered_by = "video"
        explanation = "Visual anomalies detected — face or motion appears manipulated."
    elif audio_score < threshold and video_score < threshold:
        triggered_by = "both"
        explanation = "Both audio and video show manipulation artifacts."
    elif joint_score < threshold:
        triggered_by = "mismatch"
        explanation = (
            "Audio-visual mismatch detected — audio and video may not belong together."
        )
    else:
        triggered_by = "none"
        explanation = (
            "No significant manipulation artifacts detected in audio or video."
        )

    return {
        "audio_score": round(audio_score, 4),
        "video_score": round(video_score, 4),
        "joint_score": round(joint_score, 4),
        "verdict": verdict,
        "confidence": round(abs(joint_score - threshold) * 2, 4),
        "triggered_by": triggered_by,
        "explanation": explanation,
        "windows": window_results,
        "mel_images": mel_images,
        "video_duration": round(total_sec, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

_model_cache = {}
_model_lock = threading.Lock()


def load_model(path, device):
    with _model_lock:
        if path in _model_cache:
            return _model_cache[path]
        if not os.path.exists(path):
            return None, None
        ck = torch.load(path, map_location=device, weights_only=False)
        state = ck.get("model_state_dict", ck)
        if any(k.startswith("module.") for k in state):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        fusion = (
            "transformer" if any("transformer" in k for k in state) else "pretrained"
        )
        model = AVDeepfakeDetector(fusion_type=fusion)
        model.load_state_dict(state)
        model.to(device).eval()
        info = {
            "epoch": ck.get("epoch", -1) + 1,
            "best_auc": round(ck.get("best_val_auc", 0), 4),
            "fusion": fusion,
            "path": path,
        }
        _model_cache[path] = (model, info)
        return model, info


# ─────────────────────────────────────────────────────────────────────────────
# SQLITE HISTORY
# ─────────────────────────────────────────────────────────────────────────────

DB_PATH = os.environ.get("DB_PATH", "history.db")


def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS analyses (
        id          TEXT PRIMARY KEY,
        filename    TEXT,
        model_path  TEXT,
        verdict     TEXT,
        joint_score REAL,
        audio_score REAL,
        video_score REAL,
        confidence  REAL,
        threshold   REAL,
        triggered_by TEXT,
        explanation TEXT,
        windows_json TEXT,
        timestamp   TEXT
    )""")
    con.commit()
    con.close()


def save_analysis(rec):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """INSERT INTO analyses VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            rec["id"],
            rec["filename"],
            rec["model_path"],
            rec["verdict"],
            rec["joint_score"],
            rec["audio_score"],
            rec["video_score"],
            rec["confidence"],
            rec["threshold"],
            rec["triggered_by"],
            rec["explanation"],
            json.dumps(rec["windows"]),
            rec["timestamp"],
        ),
    )
    con.commit()
    con.close()


def get_history(limit=50):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM analyses ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_analysis(aid):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM analyses WHERE id=?", (aid,)).fetchone()
    con.close()
    return dict(row) if row else None


# ─────────────────────────────────────────────────────────────────────────────
# PDF REPORT
# ─────────────────────────────────────────────────────────────────────────────


def generate_pdf(rec):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            HRFlowable,
        )
        from reportlab.lib.units import cm
    except ImportError:
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    story = []

    verdict_color = (
        colors.HexColor("#22c55e")
        if rec["verdict"] == "REAL"
        else colors.HexColor("#ef4444")
    )

    # Title
    title_style = ParagraphStyle(
        "title",
        fontSize=22,
        fontName="Helvetica-Bold",
        spaceAfter=6,
        textColor=colors.HexColor("#0f172a"),
    )
    story.append(Paragraph("AV Deepfake Detection Report", title_style))
    story.append(
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"))
    )
    story.append(Spacer(1, 0.4 * cm))

    # Meta
    meta_style = ParagraphStyle(
        "meta", fontSize=9, textColor=colors.HexColor("#64748b")
    )
    story.append(Paragraph(f"Generated: {rec['timestamp']}", meta_style))
    story.append(Paragraph(f"File: {rec['filename']}", meta_style))
    story.append(Paragraph(f"Model: {rec['model_path']}", meta_style))
    story.append(Spacer(1, 0.6 * cm))

    # Verdict
    verd_style = ParagraphStyle(
        "verdict",
        fontSize=28,
        fontName="Helvetica-Bold",
        textColor=verdict_color,
        spaceAfter=4,
    )
    story.append(Paragraph(f"Verdict: {rec['verdict']}", verd_style))
    conf_pct = int(rec["confidence"] * 100)
    story.append(
        Paragraph(
            f"Confidence: {conf_pct}%  |  Threshold: {rec['threshold']}", meta_style
        )
    )
    story.append(Spacer(1, 0.5 * cm))

    # Scores table
    score_data = [
        ["Modality", "Score", "Interpretation"],
        [
            "Joint (primary)",
            f"{rec['joint_score']:.4f}",
            "REAL" if rec["joint_score"] >= rec["threshold"] else "FAKE",
        ],
        [
            "Audio",
            f"{rec['audio_score']:.4f}",
            "Real audio" if rec["audio_score"] >= rec["threshold"] else "Fake audio",
        ],
        [
            "Video",
            f"{rec['video_score']:.4f}",
            "Real video" if rec["video_score"] >= rec["threshold"] else "Fake video",
        ],
    ]
    t = Table(score_data, colWidths=[5 * cm, 3 * cm, 8 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.HexColor("#f8fafc"), colors.white],
                ),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 0.5 * cm))

    # Explanation
    exp_style = ParagraphStyle(
        "exp",
        fontSize=11,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        borderPadding=(8, 8, 8, 8),
        borderColor=colors.HexColor("#e2e8f0"),
        borderWidth=1,
        backColor=colors.HexColor("#f8fafc"),
    )
    story.append(Paragraph(f"<b>Explanation:</b> {rec['explanation']}", exp_style))
    story.append(Spacer(1, 0.5 * cm))

    # Window breakdown
    windows = json.loads(rec.get("windows_json", "[]"))
    if windows:
        story.append(Paragraph("Window-by-Window Analysis", styles["Heading2"]))
        win_data = [["Window", "Time Range", "Audio", "Video", "Joint"]]
        for i, w in enumerate(windows):
            win_data.append(
                [
                    f"Window {i + 1}",
                    f"{w['start_sec']}s – {w['end_sec']}s",
                    f"{w['audio_score']:.4f}",
                    f"{w['video_score']:.4f}",
                    f"{w['joint_score']:.4f}",
                ]
            )
        wt = Table(win_data, colWidths=[3 * cm, 4 * cm, 3 * cm, 3 * cm, 3 * cm])
        wt.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.HexColor("#f8fafc"), colors.white],
                    ),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(wt)

    doc.build(story)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────────────────────

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", tempfile.gettempdir())
MODEL1_PATH = os.environ.get("MODEL1_PATH", "")
MODEL2_PATH = os.environ.get("MODEL2_PATH", "")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_MB = int(os.environ.get("MAX_UPLOAD_MB", 500))

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024
if CORS:
    CORS(app)

init_db()


# ── Helpers ───────────────────────────────────────────────────────────────


def allowed(filename):
    return Path(filename).suffix.lower() in {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
        ".m4v",
    }


# ── Static files ──────────────────────────────────────────────────────────


@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


# ── API: model info ───────────────────────────────────────────────────────


@app.route("/api/models")
def api_models():
    result = {}
    for key, path in [("model1", MODEL1_PATH), ("model2", MODEL2_PATH)]:
        if path and os.path.exists(path):
            _, info = load_model(path, DEVICE)
            result[key] = info
        else:
            result[key] = None
    result["device"] = str(DEVICE)
    return jsonify(result)


# ── API: analyse (single or batch) ────────────────────────────────────────


@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    model_key = request.form.get("model", "model1")
    threshold = float(request.form.get("threshold", 0.5))
    n_windows = int(request.form.get("n_windows", 3))
    model_path = MODEL1_PATH if model_key == "model1" else MODEL2_PATH

    if not model_path or not os.path.exists(model_path):
        return jsonify({"error": f"Model not found: {model_path}"}), 400

    model, info = load_model(model_path, DEVICE)
    if model is None:
        return jsonify({"error": "Failed to load model"}), 500

    files = request.files.getlist("videos")
    if not files:
        return jsonify({"error": "No videos uploaded"}), 400

    results = []
    for f in files:
        if not allowed(f.filename):
            continue
        tmp = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{f.filename}")
        try:
            f.save(tmp)
            r = predict_with_windows(model, tmp, DEVICE, n_windows, threshold)
            if r is None:
                results.append(
                    {"filename": f.filename, "error": "Could not read video"}
                )
                continue
            aid = str(uuid.uuid4())
            rec = {
                "id": aid,
                "filename": f.filename,
                "model_path": model_path,
                "threshold": threshold,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                **r,
            }
            save_analysis(rec)
            rec.pop("mel_images", None)  # don't store images in DB response
            results.append(rec)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    return jsonify({"results": results, "model_info": info})


# ── API: compare two models on same video ─────────────────────────────────


@app.route("/api/compare", methods=["POST"])
def api_compare():
    threshold = float(request.form.get("threshold", 0.5))
    n_windows = int(request.form.get("n_windows", 3))

    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    f = request.files["video"]
    if not allowed(f.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    tmp = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{f.filename}")
    try:
        f.save(tmp)
        out = {}
        for key, path in [("model1", MODEL1_PATH), ("model2", MODEL2_PATH)]:
            if not path or not os.path.exists(path):
                out[key] = {"error": f"Model not configured: {path}"}
                continue
            model, info = load_model(path, DEVICE)
            r = predict_with_windows(model, tmp, DEVICE, n_windows, threshold)
            if r is None:
                out[key] = {"error": "Could not process video"}
            else:
                out[key] = {**r, "model_info": info}
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    return jsonify({"filename": f.filename, "models": out, "threshold": threshold})


# ── API: history ──────────────────────────────────────────────────────────


@app.route("/api/history")
def api_history():
    limit = int(request.args.get("limit", 50))
    rows = get_history(limit)
    return jsonify({"analyses": rows, "total": len(rows)})


@app.route("/api/history/<aid>")
def api_history_item(aid):
    rec = get_analysis(aid)
    if not rec:
        abort(404)
    return jsonify(rec)


@app.route("/api/history/<aid>", methods=["DELETE"])
def api_history_delete(aid):
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM analyses WHERE id=?", (aid,))
    con.commit()
    con.close()
    return jsonify({"deleted": aid})


@app.route("/api/history", methods=["DELETE"])
def api_history_clear():
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM analyses")
    con.commit()
    con.close()
    return jsonify({"cleared": True})


# ── API: PDF report ───────────────────────────────────────────────────────


@app.route("/api/report/<aid>")
def api_report(aid):
    rec = get_analysis(aid)
    if not rec:
        abort(404)
    buf = generate_pdf(rec)
    if buf is None:
        return jsonify({"error": "reportlab not installed. pip install reportlab"}), 500
    safe = rec["filename"].replace(" ", "_").replace("/", "_")
    return send_file(
        buf,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"deepfake_report_{safe}.pdf",
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n{'=' * 55}")
    print(f"  AV Deepfake Detection — Web App")
    print(f"{'=' * 55}")
    print(f"  Device:  {DEVICE}")
    print(f"  Model 1: {MODEL1_PATH or '⚠ not set (export MODEL1_PATH=...)'}")
    print(f"  Model 2: {MODEL2_PATH or '⚠ not set (export MODEL2_PATH=...)'}")
    print(f"  URL:     http://localhost:{port}")
    print(f"{'=' * 55}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
