"""
Cell 1 — Environment installation & dependency check.
Run once before anything else.
"""

import subprocess
import sys


def pip_install(package: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", package])


PACKAGES = [
    "ultralytics>=8.0.0",
    "torch torchvision --index-url https://download.pytorch.org/whl/cu118",
    "timm>=0.9.0",
    "filterpy>=1.4.5",
    "lap>=0.4.0",
    "cython_bbox>=0.1.3",
    "opencv-python>=4.8.0",
    "scipy>=1.10.0",
    "matplotlib>=3.7.0",
]

if __name__ == "__main__":
    print("Installing dependencies...")
    for pkg in PACKAGES:
        print(f"  pip install {pkg}")
        try:
            pip_install(pkg)
        except subprocess.CalledProcessError as e:
            print(f"  WARNING: failed to install {pkg}: {e}")

    # Verify core imports
    import importlib
    for mod in ["torch", "ultralytics", "cv2", "timm", "filterpy"]:
        try:
            importlib.import_module(mod)
            print(f"  OK  {mod}")
        except ImportError:
            print(f"  MISSING  {mod}")

    # Check CUDA
    import torch
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Download MiDaS model weights
    print("\nDownloading MiDaS DPT-Large weights...")
    import urllib.request, os
    weights_dir = os.path.join(os.path.dirname(__file__), "..", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    midas_url = "https://github.com/isl-org/MiDaS/releases/download/v3/dpt_large_384.pt"
    dest = os.path.join(weights_dir, "dpt_large_384.pt")
    if not os.path.exists(dest):
        urllib.request.urlretrieve(midas_url, dest)
        print(f"  Saved to {dest}")
    else:
        print(f"  Already present: {dest}")

    print("\nSetup complete.")
