"""
Download AV-Deepfake1M++ validation data from Hugging Face.

Usage:
    python download_data.py                   # Download to default path
    python download_data.py --data_dir /path  # Download to custom path

Requires:
    pip install huggingface_hub
    huggingface-cli login  (one-time, need to accept dataset terms first)
"""

import os
import sys
import glob
import subprocess
import argparse

try:
    from huggingface_hub import snapshot_download, login
except ImportError:
    print("Installing huggingface_hub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import snapshot_download, login


# Dataset info
REPO_ID = "ControlNet/AV-Deepfake1M-PlusPlus"
REPO_TYPE = "dataset"


def ensure_login():
    """Check HF login status and prompt if needed."""
    # Check for token from .env
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token and not hf_token.startswith("your_"):
        login(token=hf_token)
        print("✓ Logged in to Hugging Face using HF_TOKEN from .env")
        return
    
    try:
        from huggingface_hub import whoami
        user = whoami()
        print(f"✓ Logged in as: {user['name']}")
    except Exception:
        print("\nYou need to log in to Hugging Face to access this dataset.")
        print("Option 1: Set HF_TOKEN in your .env file")
        print("Option 2: Login interactively below")
        print("\nAccept dataset terms at: https://huggingface.co/datasets/ControlNet/AV-Deepfake1M-PlusPlus")
        login()


def download_val_data(data_dir):
    """Download all val zip volumes using snapshot_download."""
    print("\nDownloading val zip volumes...")
    
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=data_dir,
        allow_patterns="val/val.zip.*",
        local_dir_use_symlinks=False,
        resume_download=True
    )
    
    print(f"✓ Download complete: {path}")
    return path


def download_metadata(data_dir):
    """Download val_metadata.json."""
    print("\nDownloading val_metadata.json...")
    
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=data_dir,
        allow_patterns="val_metadata.json",
        local_dir_use_symlinks=False,
        resume_download=True
    )
    
    print(f"✓ Metadata download complete")
    return path


def extract_zip_files(data_dir):
    """Extract multi-volume zip files (val.zip.001, val.zip.002, etc.)."""
    # Look for zip parts inside val/ subdirectory
    zip_dir = os.path.join(data_dir, "val")
    first_part = os.path.join(zip_dir, "val.zip.001")
    
    if not os.path.exists(first_part):
        print(f"⚠ No zip files found at {first_part}")
        return None
    
    extract_dir = os.path.join(data_dir, "extracted_val")
    
    # Check if already extracted
    if os.path.exists(extract_dir) and os.listdir(extract_dir):
        print(f"✓ Already extracted at: {extract_dir}")
        user_input = input("  Re-extract? (y/n): ").strip().lower()
        if user_input != 'y':
            return extract_dir
    
    os.makedirs(extract_dir, exist_ok=True)
    
    print(f"\nExtracting to: {extract_dir}")
    print("  (this may take a while for large archives...)")
    
    # Try 7z first (handles multi-volume zips well)
    try:
        result = subprocess.run(
            ["7z", "x", first_part, f"-o{extract_dir}", "-aoa"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  ✓ Extracted with 7z")
            return extract_dir
    except FileNotFoundError:
        pass
    
    # Fallback: try unzip
    try:
        result = subprocess.run(
            ["unzip", "-o", first_part, "-d", extract_dir],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  ✓ Extracted with unzip")
            return extract_dir
    except FileNotFoundError:
        pass
    
    print("  ✗ Extraction failed — install 7z: apt-get install p7zip-full")
    return None


def download_and_extract(data_dir):
    """Full download + extract pipeline.
    
    Args:
        data_dir: root directory for downloads
        
    Returns:
        str: path to extracted val directory, or None if failed
    """
    print("=" * 60)
    print("DATASET DOWNLOAD (AV-Deepfake1M++ Validation)")
    print("=" * 60)
    
    ensure_login()
    
    # Download zip volumes
    print("\n" + "-" * 40)
    print("Step 1: Downloading from Hugging Face")
    print("-" * 40)
    download_val_data(data_dir)
    download_metadata(data_dir)
    
    # Extract
    print("\n" + "-" * 40)
    print("Step 2: Extracting zip archives")
    print("-" * 40)
    extract_dir = extract_zip_files(data_dir)
    
    if extract_dir and os.path.exists(extract_dir):
        file_count = sum(len(files) for _, _, files in os.walk(extract_dir))
        print(f"\n✓ Extraction complete: {file_count:,} files in {extract_dir}")
    
    # Check for metadata
    for path in [os.path.join(data_dir, "val_metadata.json"),
                 os.path.join(extract_dir, "val_metadata.json") if extract_dir else None]:
        if path and os.path.exists(path):
            print(f"✓ Metadata found: {path}")
            break
    else:
        print("⚠ val_metadata.json not found — may be inside the zip")
    
    return extract_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AV-Deepfake1M++ val data")
    parser.add_argument("--data_dir", type=str, 
                       default=os.environ.get("DATA_DIR", "/content/drive/MyDrive/val"),
                       help="Directory to download data to")
    args = parser.parse_args()
    
    extract_dir = download_and_extract(args.data_dir)
    
    if extract_dir:
        print(f"\n{'='*60}")
        print(f"Done! Set DATA_DIR in .env to: {os.path.dirname(extract_dir)}")
        print(f"{'='*60}")
