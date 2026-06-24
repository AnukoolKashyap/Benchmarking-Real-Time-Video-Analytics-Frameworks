"""
scripts/check_gpu.py — Verify your GPU setup before running the pipeline
Run this first to confirm everything is installed correctly.

Usage:
    python scripts/check_gpu.py
"""

import sys


def check(label, fn):
    try:
        result = fn()
        print(f"  ✓  {label}: {result}")
        return True
    except Exception as e:
        print(f"  ✗  {label}: FAILED — {e}")
        return False


print("\n" + "="*55)
print("  GPU Pipeline — Environment Check")
print("="*55 + "\n")

# Python version
check("Python version", lambda: sys.version.split()[0])

# CUDA via PyTorch
check("CUDA available (torch)",
      lambda: __import__("torch").cuda.get_device_name(0))

# TensorRT
check("TensorRT",
      lambda: __import__("tensorrt").__version__)

# PyCUDA
check("PyCUDA",
      lambda: __import__("pycuda.driver", fromlist=["driver"]) and "ok")

# PyAV (FFmpeg bindings)
def check_pyav():
    import av
    # Try opening a test CUDA context
    codecs = [c for c in av.codecs_available if "cuvid" in c]
    return f"av {av.__version__}  |  GPU codecs: {codecs}"
check("PyAV + GPU codecs", check_pyav)

# OpenCV
check("OpenCV",
      lambda: __import__("cv2").__version__)

# Ultralytics
check("Ultralytics",
      lambda: __import__("ultralytics").__version__)

print("\n" + "="*55)
print("  If any ✗ above — install the missing package")
print("  then re-run this check before running main.py")
print("="*55 + "\n")
