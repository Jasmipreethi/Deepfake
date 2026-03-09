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
    from huggingface_hub import hf_hub_download, list_repo_files, login
except ImportError:
    print("Installing huggingface_hub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import hf_hub_download, list_repo_files, login


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
        print("\nAccept dataset terms at: https://huggingface.co/datasets/ControlNet-XS/AV-Deepfake1M")
        login()


def list_val_files():
    """List all val-related files in the dataset repo."""
    print("\nFetching file list from Hugging Face...")
    all_files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    
    val_files = [f for f in all_files if f.startswith("val")]
    metadata_files = [f for f in all_files if "val_metadata" in f]
    
    # Combine and deduplicate
    download_files = sorted(set(val_files + metadata_files))
    
    print(f"Found {len(download_files)} val-related files:")
    for f in download_files:
        print(f"  • {f}")
    
    return download_files


def download_val_data(data_dir):
    """Download all val zip parts and metadata."""
    os.makedirs(data_dir, exist_ok=True)
    
    files_to_download = list_val_files()
    
    if not files_to_download:
        print("⚠ No val files found in repository!")
        return []
    
    downloaded = []
    for i, filename in enumerate(files_to_download):
        local_path = os.path.join(data_dir, filename)
        
        # Skip if already downloaded
        if os.path.exists(local_path):
            print(f"  [{i+1}/{len(files_to_download)}] Already exists: {filename}")
            downloaded.append(local_path)
            continue
        
        print(f"  [{i+1}/{len(files_to_download)}] Downloading: {filename}")
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                repo_type=REPO_TYPE,
                local_dir=data_dir
            )
            downloaded.append(path)
            print(f"    ✓ Done")
        except Exception as e:
            print(f"    ✗ Failed: {e}")
    
    return downloaded


def extract_zip_files(data_dir):
    """Extract multi-volume zip files (val.zip.001, val.zip.002, etc.)."""
    # Find the first part of the multi-volume zip
    first_parts = sorted(glob.glob(os.path.join(data_dir, "val.zip.001")))
    
    if not first_parts:
        # Try single zip
        single_zips = sorted(glob.glob(os.path.join(data_dir, "val.zip")))
        if single_zips:
            first_parts = single_zips
    
    if not first_parts:
        print("⚠ No val zip files found to extract.")
        return
    
    extract_dir = os.path.join(data_dir, "extracted_val")
    
    # Check if already extracted
    if os.path.exists(extract_dir) and os.listdir(extract_dir):
        print(f"✓ Already extracted at: {extract_dir}")
        user_input = input("  Re-extract? (y/n): ").strip().lower()
        if user_input != 'y':
            return extract_dir
    
    os.makedirs(extract_dir, exist_ok=True)
    
    for zip_path in first_parts:
        print(f"\nExtracting: {os.path.basename(zip_path)}")
        print("  (this may take a while for large archives...)")
        
        # Try 7z first (handles multi-volume zips well)
        try:
            result = subprocess.run(
                ["7z", "x", zip_path, f"-o{extract_dir}", "-aoa"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"  ✓ Extracted with 7z")
                continue
        except FileNotFoundError:
            pass
        
        # Fallback: try unzip
        try:
            result = subprocess.run(
                ["unzip", "-o", zip_path, "-d", extract_dir],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"  ✓ Extracted with unzip")
                continue
        except FileNotFoundError:
            pass
        
        # Fallback: Python zipfile (doesn't handle multi-volume)
        import zipfile
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
            print(f"  ✓ Extracted with Python zipfile")
        except Exception as e:
            print(f"  ✗ Extraction failed: {e}")
            print("  Try installing 7z: apt-get install p7zip-full")
    
    return extract_dir


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
    
    # Download
    print("\n" + "-" * 40)
    print("Step 1: Downloading from Hugging Face")
    print("-" * 40)
    downloaded = download_val_data(data_dir)
    
    if not downloaded:
        print("No files downloaded!")
        return None
    
    # Extract
    print("\n" + "-" * 40)
    print("Step 2: Extracting zip archives")
    print("-" * 40)
    extract_dir = extract_zip_files(data_dir)
    
    # Verify
    if extract_dir and os.path.exists(extract_dir):
        file_count = sum(1 for _ in os.walk(extract_dir) 
                        for _ in _[2])
        print(f"\n✓ Extraction complete: {file_count:,} files in {extract_dir}")
    
    # Check for metadata
    metadata_locations = [
        os.path.join(data_dir, "val_metadata.json"),
        os.path.join(extract_dir, "val_metadata.json") if extract_dir else None,
    ]
    
    for path in metadata_locations:
        if path and os.path.exists(path):
            print(f"✓ Metadata found: {path}")
            break
    else:
        print("⚠ val_metadata.json not found — may be inside the zip")
    
    return extract_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download AV-Deepfake1M++ val data")
    parser.add_argument("--data_dir", type=str, 
                       default="/content/drive/MyDrive/val",
                       help="Directory to download data to")
    args = parser.parse_args()
    
    extract_dir = download_and_extract(args.data_dir)
    
    if extract_dir:
        print(f"\n{'='*60}")
        print(f"Done! Set VAL_DIR in config.py to:")
        print(f"  VAL_DIR = '{extract_dir}'")
        print(f"{'='*60}")
