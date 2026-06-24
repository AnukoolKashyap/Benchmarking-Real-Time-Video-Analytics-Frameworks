"""
main.py — RTSP Face Detection Pipeline
FFmpeg GPU decode (NVDEC) → TensorRT inference (YOLOv8-face)

Pipeline:
    IP Camera (RTSP)
        → libavformat  : reads H264 packets over network
        → libavcodec   : NVDEC decodes H264 → NV12 frame in VRAM
        → GPU preprocess: NV12 → RGB, resize, normalize (stays in VRAM)
        → TensorRT     : yolov8n-face inference in VRAM
        → CPU          : bounding boxes saved to disk
"""

import sys
import logging
import time
from datetime import datetime
from pathlib import Path

from config import Config
from utils.ffmpeg_capture import FFmpegCapture
from utils.trt_inference import TRTFaceDetector
from utils.frame_saver import FrameSaver

# ── Logging ──────────────────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)
log = logging.getLogger("main")


def run(cfg: Config):
    log.info("=" * 60)
    log.info("  RTSP Face Detection — FFmpeg GPU + TensorRT")
    log.info("=" * 60)
    log.info(f"Camera  : {cfg.rtsp_url}")
    log.info(f"Codec   : {cfg.gpu_codec}  (GPU decode via NVDEC)")
    log.info(f"Engine  : {cfg.engine_path}")

    # ── Load TensorRT engine ─────────────────────────────────────────────────
    log.info("Loading TensorRT engine...")
    detector = TRTFaceDetector(
        engine_path=cfg.engine_path,
        conf_threshold=cfg.conf_threshold,
        iou_threshold=cfg.iou_threshold,
        input_size=(cfg.input_w, cfg.input_h),
    )

    # ── Frame saver ──────────────────────────────────────────────────────────
    saver = FrameSaver(
        output_dir=cfg.output_dir,
        max_saves_per_minute=cfg.max_saves_per_minute,
        save_annotated=cfg.save_annotated,
    )

    # ── Open RTSP stream via FFmpeg GPU decode ───────────────────────────────
    log.info("Connecting to camera...")
    capture = FFmpegCapture(
        rtsp_url=cfg.rtsp_url,
        gpu_codec=cfg.gpu_codec,
        cuda_device=cfg.cuda_device,
        width=cfg.decode_width,
        height=cfg.decode_height,
    )

    if not capture.open():
        log.error("Failed to connect to camera. Check your RTSP URL.")
        sys.exit(1)

    log.info("Camera connected. Pipeline running. Press Ctrl+C to stop.")

    # ── Main loop ────────────────────────────────────────────────────────────
    frame_count   = 0
    detect_count  = 0
    fps_frames    = 0
    fps_time      = time.time()

    try:
        while True:
            frame = capture.read()          # BGR numpy array from GPU decode
            if frame is None:
                log.warning("No frame received — stream may have dropped")
                time.sleep(0.1)
                continue

            frame_count += 1
            fps_frames  += 1

            # Skip frames if process_every_n_frames > 1
            if frame_count % cfg.process_every_n_frames != 0:
                continue

            # ── TensorRT inference ───────────────────────────────────────────
            detections = detector.detect(frame)
            detect_count += 1

            # ── FPS log every 5 seconds ──────────────────────────────────────
            elapsed = time.time() - fps_time
            if elapsed >= 5.0:
                fps = fps_frames / elapsed
                log.info(
                    f"FPS: {fps:.1f}  |  "
                    f"Frames: {frame_count}  |  "
                    f"Inferences: {detect_count}  |  "
                    f"Faces this frame: {len(detections)}"
                )
                fps_frames = 0
                fps_time   = time.time()

            # ── Save if faces detected ───────────────────────────────────────
            if len(detections) > 0:
                saver.save(frame, detections)

    except KeyboardInterrupt:
        log.info("\nStopped by user.")
    finally:
        capture.close()
        log.info(f"Total frames read     : {frame_count}")
        log.info(f"Total inferences run  : {detect_count}")
        log.info(f"Total frames saved    : {saver.save_count}")
        log.info("Pipeline shut down cleanly.")


if __name__ == "__main__":
    cfg = Config()
    run(cfg)
