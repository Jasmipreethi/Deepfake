"""
AV Deepfake Detection — Simplified Web Backend
Flask server for model selection, video upload, accuracy metrics, model comparison, and history.
All ML/inference logic imported from inference.py.
"""

import os
import sys
import uuid
import json
import sqlite3
import tempfile
import datetime
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, send_from_directory, abort
import torch
from inference import load_model, predict_video

UPLOAD_DIR = os.environ.get("UPLOAD_DIR", tempfile.gettempdir())

# Default model paths — point to the two best models from the training runs.
# Model 2: logs/logs_2 — peak validation AUC 0.994, 5 epochs. Best for precision/recall balance.
# Model 3: logs/logs_3 — test AUC 0.937, 93% accuracy, F1 0.837, zero false positives.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL1_PATH = os.environ.get("MODEL1_PATH", os.path.join(_PROJECT_ROOT, "logs", "logs_2", "best_model.pth"))
MODEL2_PATH = os.environ.get("MODEL2_PATH", os.path.join(_PROJECT_ROOT, "logs", "logs_3", "best_model.pth"))
DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "history.db"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_MB = int(os.environ.get("MAX_UPLOAD_MB", 500))

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

_model_cache = {}

# ── Model loading ────────────────────────────────────────────────────────────

def get_model(key):
    path = MODEL1_PATH if key == "model1" else MODEL2_PATH
    if not path or not os.path.exists(path):
        return None, None
    if key in _model_cache:
        return _model_cache[key], {"path": path, "cached": True}
    model = load_model(path, DEVICE)
    _model_cache[key] = model
    return model, {"path": path, "cached": False}

def allowed(filename):
    return filename.lower().split(".")[-1] in {"mp4", "avi", "mov", "mkv", "webm", "m4v"}

# ── SQLite history ───────────────────────────────────────────────────────────

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS analyses (
        id TEXT PRIMARY KEY,
        filename TEXT,
        model_key TEXT,
        joint_score REAL,
        audio_score REAL,
        video_score REAL,
        confidence REAL,
        threshold REAL,
        verdict TEXT,
        timestamp TEXT
    )""")
    con.commit()
    con.close()

def save_analysis(rec):
    con = sqlite3.connect(DB_PATH)
    con.execute("""INSERT INTO analyses VALUES (?,?,?,?,?,?,?,?,?,?)""", (
        rec["id"], rec["filename"], rec["model_key"],
        rec["joint_score"], rec["audio_score"], rec["video_score"],
        rec["confidence"], rec["threshold"], rec["verdict"], rec["timestamp"]
    ))
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

def delete_analysis(aid):
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM analyses WHERE id=?", (aid,))
    con.commit()
    con.close()

def clear_history():
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM analyses")
    con.commit()
    con.close()

init_db()

# ── API: models ─────────────────────────────────────────────────────────────

@app.route("/api/models", methods=["GET"])
def api_models():
    m1_exists = os.path.exists(MODEL1_PATH) if MODEL1_PATH else False
    m2_exists = os.path.exists(MODEL2_PATH) if MODEL2_PATH else False
    return jsonify({
        "model1": {
            "path": MODEL1_PATH,
            "available": m1_exists,
            "label": "Model 2 — Val AUC 0.994 (5 epochs)",
            "note": "Best validation AUC; use threshold 0.795+ for balanced accuracy",
        },
        "model2": {
            "path": MODEL2_PATH,
            "available": m2_exists,
            "label": "Model 3 — Test AUC 0.937 (3 epochs)",
            "note": "Best practical model: 93% accuracy, zero false positives",
        },
        "device": str(DEVICE),
    })

# ── API: analyze single video ────────────────────────────────────────────────

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    model_key = request.form.get("model", "model1")
    threshold = float(request.form.get("threshold", 0.5))
    n_windows = int(request.form.get("n_windows", 3))

    video = request.files.get("video")
    if not video:
        return jsonify({"error": "No video file provided"}), 400
    if not allowed(video.filename):
        return jsonify({"error": "Unsupported video format"}), 400

    model, _ = get_model(model_key)
    if model is None:
        return jsonify({"error": f"Model '{model_key}' not found or not configured"}), 400

    tmp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{video.filename}")
    try:
        video.save(tmp_path)
        result = predict_video(model, tmp_path, DEVICE, n_windows=n_windows)
        if result is None:
            return jsonify({"error": "Could not read video file"}), 400

        result["threshold"] = threshold
        result["verdict"] = "REAL" if result["joint_score"] >= threshold else "FAKE"
        result["model_key"] = model_key

        # Save to history
        aid = str(uuid.uuid4())
        save_analysis({
            "id": aid,
            "filename": video.filename,
            "model_key": model_key,
            "joint_score": result["joint_score"],
            "audio_score": result["audio_score"],
            "video_score": result["video_score"],
            "confidence": result["confidence"],
            "threshold": threshold,
            "verdict": result["verdict"],
            "timestamp": datetime.datetime.utcnow().isoformat(),
        })
        result["id"] = aid
        return jsonify(result)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ── API: compare two models on same video ───────────────────────────────────

@app.route("/api/compare", methods=["POST"])
def api_compare():
    threshold = float(request.form.get("threshold", 0.5))
    n_windows = int(request.form.get("n_windows", 3))

    video = request.files.get("video")
    if not video:
        return jsonify({"error": "No video file provided"}), 400
    if not allowed(video.filename):
        return jsonify({"error": "Unsupported video format"}), 400

    tmp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{video.filename}")
    try:
        video.save(tmp_path)
        results = {}
        for key, model_path in [("model1", MODEL1_PATH), ("model2", MODEL2_PATH)]:
            if not model_path or not os.path.exists(model_path):
                results[key] = {"error": "Not configured"}
                continue
            model, _ = get_model(key)
            if model is None:
                results[key] = {"error": "Failed to load model"}
                continue
            r = predict_video(model, tmp_path, DEVICE, n_windows=n_windows)
            if r is None:
                results[key] = {"error": "Could not read video"}
                continue
            r["threshold"] = threshold
            r["verdict"] = "REAL" if r["joint_score"] >= threshold else "FAKE"
            r["model_key"] = key
            results[key] = r
        return jsonify({"filename": video.filename, "results": results, "threshold": threshold})
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ── API: history ─────────────────────────────────────────────────────────────

@app.route("/api/history", methods=["GET"])
def api_history():
    limit = int(request.args.get("limit", 50))
    return jsonify({"analyses": get_history(limit)})

@app.route("/api/history/<aid>", methods=["DELETE"])
def api_history_delete(aid):
    delete_analysis(aid)
    return jsonify({"deleted": aid})

@app.route("/api/history", methods=["DELETE"])
def api_history_clear():
    clear_history()
    return jsonify({"cleared": True})

# ── UI ───────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

# ── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n{'='*50}")
    print(f" AV Deepfake Detection — Web App")
    print(f"{'='*50}")
    print(f" Device  : {DEVICE}")
    print(f" Model 1 : {MODEL1_PATH or '(not set)'}")
    print(f" Model 2 : {MODEL2_PATH or '(not set)'}")
    print(f" URL     : http://localhost:{port}")
    print(f"{'='*50}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)