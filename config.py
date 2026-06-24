"""
config.py — All pipeline settings in one place.
Edit this file to match your camera and GPU setup.
"""

from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class Config:

    # ── Camera ───────────────────────────────────────────────────────────────
    # Your camera's RTSP URL — change IP, port, credentials to match your camera
    # Common formats:
    #   Hikvision : rtsp://admin:password@192.168.1.64:554/Streaming/Channels/1
    #   Dahua     : rtsp://admin:password@192.168.1.64:554/cam/realmonitor
    #   Generic   : rtsp://192.168.1.64:554/stream1
    rtsp_url:  str = os.getenv("RTSP_URL", "rtsp://YOUR_CAMERA_IP/stream")
    # ── GPU decode ───────────────────────────────────────────────────────────
    # codec for GPU decode — hevc_cuvid routes H265 to NVDEC chip on GPU
    # change to hevc_cuvid if your camera sends H265
    gpu_codec: str = "hevc_cuvid"

    # CUDA device index — 0 = first GPU
    cuda_device: int = 0

    # ── Decode resolution ────────────────────────────────────────────────────
    # Resolution to decode at — match your camera's native resolution
    decode_width: int  = 1280
    decode_height: int = 720

    # ── TensorRT model ───────────────────────────────────────────────────────
    # Path to your compiled .engine file
    engine_path: str = "models/yolov8n-face.engine"

    # Input size the YOLO model expects — must match what you used when building engine
    input_w: int = 640
    input_h: int = 640

    # ── Detection thresholds ─────────────────────────────────────────────────
    conf_threshold: float = 0.45
    iou_threshold:  float = 0.45

    # ── Performance ──────────────────────────────────────────────────────────
    # Run inference on every N-th frame (1 = every frame)
    # Increase to reduce GPU load — e.g. 2 = 12.5 FPS inference on 25 FPS stream
    process_every_n_frames: int = 1

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: str  = "captured_faces"
    save_annotated: bool = True      # draw boxes on saved images
    max_saves_per_minute: int = 60
