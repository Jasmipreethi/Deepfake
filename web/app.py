"""
AV Deepfake Detector — Web Interface

Flask server that wraps inference.py to provide a web UI
for uploading and analyzing videos.

Usage:
    python app.py
    python app.py --model /path/to/best_model.pth --port 5000
    python app.py --device cuda
"""

import os
import sys
import time
import uuid
import json
import argparse
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Add parent directory to path so we can import inference.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template, send_from_directory
import torch

# Import from inference.py
from inference import load_model, predict_video, _extract_at, FEATURE_CONFIG

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

# Config
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
UPLOAD_DIR = tempfile.mkdtemp(prefix='av_deepfake_')

# Global model state
model = None
device = None
model_info = {}


def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Upload and analyze a video."""
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'error': f'Unsupported format. Accepted: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

    # Save to temp file
    ext = Path(file.filename).suffix
    temp_name = f"{uuid.uuid4().hex}{ext}"
    temp_path = os.path.join(UPLOAD_DIR, temp_name)

    try:
        file.save(temp_path)
        file_size = os.path.getsize(temp_path)

        # Run inference
        start_time = time.time()
        n_windows = int(request.form.get('n_windows', 3))
        result = predict_video(model, temp_path, device, n_windows=n_windows)
        processing_time = time.time() - start_time

        if result is None:
            return jsonify({
                'error': 'Could not process video. Ensure it has both audio and video tracks.'
            }), 422

        # Build response
        response = {
            'verdict': result['verdict'],
            'confidence': result['confidence'],
            'scores': {
                'audio': result['audio_score'],
                'video': result['video_score'],
                'joint': result['joint_score'],
            },
            'file': file.filename,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'windows_analyzed': n_windows,
            'processing_time_sec': round(processing_time, 2),
            'interpretation': build_interpretation(result),
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

    finally:
        # Always clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/api/health')
def health():
    """Health check."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'device': str(device),
        'model_info': model_info,
    })


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_interpretation(result):
    """Generate a plain-English interpretation of the results."""
    audio = result['audio_score']
    video = result['video_score']
    joint = result['joint_score']
    verdict = result['verdict']

    audio_status = "authentic" if audio >= 0.5 else "manipulated"
    video_status = "authentic" if video >= 0.5 else "manipulated"

    if verdict == 'REAL':
        return (f"Both the audio (score: {audio:.2f}) and video (score: {video:.2f}) "
                f"appear authentic. The overall assessment is REAL with "
                f"{result['confidence']:.0%} confidence.")
    else:
        parts = []
        if audio < 0.5:
            parts.append(f"the audio appears manipulated (score: {audio:.2f})")
        if video < 0.5:
            parts.append(f"the video appears manipulated (score: {video:.2f})")
        if not parts:
            parts.append(f"the combined analysis indicates manipulation (joint score: {joint:.2f})")

        detail = " and ".join(parts)
        return (f"Analysis indicates this is likely a deepfake: {detail}. "
                f"Overall confidence: {result['confidence']:.0%}.")


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────

def create_app(model_path, device_name='auto'):
    """Initialize model and return configured app."""
    global model, device, model_info

    # Device
    if device_name == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)

    print(f"Device: {device}")
    print(f"Loading model: {model_path}")

    model = load_model(model_path, device)
    model_info = {
        'path': os.path.basename(model_path),
        'device': str(device),
    }

    print(f"Model loaded. Server ready.")
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AV Deepfake Detector Web Interface')
    parser.add_argument('--model', default='../best_model.pth',
                        help='Path to best_model.pth')
    parser.add_argument('--port', type=int, default=5000,
                        help='Server port (default: 5000)')
    parser.add_argument('--host', default='0.0.0.0',
                        help='Server host (default: 0.0.0.0)')
    parser.add_argument('--device', default='auto',
                        choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        print("Use --model to specify the path to your best_model.pth")
        sys.exit(1)

    create_app(args.model, args.device)
    app.run(host=args.host, port=args.port, debug=args.debug)
