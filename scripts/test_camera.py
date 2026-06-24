"""
scripts/test_camera.py — Test RTSP camera, three decode modes
Tries GPU first, falls back automatically

Usage:
    python scripts/test_camera.py --url "rtsp://..."
    python scripts/test_camera.py --url "rtsp://..." --mode cpu
    python scripts/test_camera.py --url "rtsp://..." --mode gpu
    python scripts/test_camera.py --url "rtsp://..." --mode auto
"""

import av
import cv2
import argparse
import time
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("test_camera")

RTSP_OPTIONS = {
    "rtsp_transport": "tcp",
    "stimeout":       "5000000",
    "fflags":         "nobuffer",
    "flags":          "low_delay",
}


def decode_loop(container, video_stream, label):
    """Shared decode + display loop."""
    frame_count = 0
    start_time  = time.time()
    log.info(f"Receiving frames [{label}] — press Q to stop")

    try:
        for packet in container.demux(video_stream):
            if packet.size == 0:
                continue
            try:
                for frame in packet.decode():
                    img = frame.to_ndarray(format="bgr24")
                    frame_count += 1
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    cv2.putText(img, f"FPS:{fps:.1f} [{label}]", (10, 36),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                    cv2.imshow("Camera Test — Q to quit", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        raise KeyboardInterrupt
                    if frame_count % 50 == 0:
                        log.info(f"Frames: {frame_count}  FPS: {fps:.1f}")
            except av.error.InvalidDataError:
                continue
    except KeyboardInterrupt:
        pass
    finally:
        container.close()
        cv2.destroyAllWindows()
        elapsed = time.time() - start_time
        if frame_count > 0 and elapsed > 0:
            log.info(f"Result — {frame_count} frames / {elapsed:.1f}s = {frame_count/elapsed:.1f} FPS")
        else:
            log.warning("No frames decoded — this mode may not be supported")
    return frame_count


def mode_cpu(url):
    """Pure CPU software decode — always works."""
    log.info("Mode: CPU software decode")
    try:
        c = av.open(url, options=RTSP_OPTIONS, timeout=15.0)
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        log.info(f"Connected — {s.codec_context.width}x{s.codec_context.height} @ {float(s.average_rate):.0f}fps codec:{s.codec_context.name}")
        decode_loop(c, s, "CPU")
    except Exception as e:
        log.error(f"CPU mode failed: {e}")


def mode_gpu_cuvid(url):
    """
    GPU decode — forces hevc_cuvid by passing it directly as a PyAV
    stream codec override via the 'video_size' + codec trick.
    Most direct method: open with hevc_cuvid as forced video decoder.
    """
    log.info("Mode: GPU hevc_cuvid (NVDEC)")

    # Method: open container normally, then decode packets manually
    # using a SEPARATE CodecContext configured for hevc_cuvid
    try:
        # Step 1: open container just for demuxing (no decode yet)
        container = av.open(url, options=RTSP_OPTIONS, timeout=15.0)
        video_stream = container.streams.video[0]

        log.info(
            f"Connected — "
            f"{video_stream.codec_context.width}x{video_stream.codec_context.height} "
            f"@ {float(video_stream.average_rate):.0f}fps"
        )

        # Step 2: create a separate GPU codec context for hevc_cuvid
        gpu_codec   = av.codec.Codec("hevc_cuvid", "r")
        gpu_context = av.codec.context.CodecContext.create(gpu_codec)
        gpu_context.options = {"gpu": "0"}

        # Copy stream parameters into the GPU context
        gpu_context.extradata = video_stream.codec_context.extradata
        gpu_context.width     = video_stream.codec_context.width
        gpu_context.height    = video_stream.codec_context.height
        gpu_context.pix_fmt   = video_stream.codec_context.pix_fmt

        gpu_context.open()
        log.info("hevc_cuvid GPU context opened successfully")

        frame_count = 0
        start_time  = time.time()
        log.info("Receiving frames [GPU hevc_cuvid] — press Q to stop")

        try:
            for packet in container.demux(video_stream):
                if packet.size == 0:
                    continue
                try:
                    for frame in gpu_context.decode(packet):
                        img = frame.to_ndarray(format="bgr24")
                        frame_count += 1
                        elapsed = time.time() - start_time
                        fps = frame_count / elapsed if elapsed > 0 else 0
                        cv2.putText(img, f"FPS:{fps:.1f} [GPU NVDEC]", (10, 36),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                        cv2.imshow("Camera Test — Q to quit", img)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            raise KeyboardInterrupt
                        if frame_count % 50 == 0:
                            log.info(f"Frames: {frame_count}  FPS: {fps:.1f}")
                except (av.error.InvalidDataError, Exception):
                    continue

        except KeyboardInterrupt:
            pass
        finally:
            container.close()
            cv2.destroyAllWindows()
            elapsed = time.time() - start_time
            if frame_count > 0:
                log.info(f"Result — {frame_count} frames / {elapsed:.1f}s = {frame_count/elapsed:.1f} FPS")
            else:
                log.warning("0 frames from GPU context — see mode_auto for fallback")

    except Exception as e:
        log.error(f"GPU cuvid mode failed: {e}")
        log.info("Try --mode cpu to confirm camera works, then we debug GPU")


def mode_auto(url):
    """
    Auto mode — tries GPU first, falls back to CPU.
    Most useful for first-time testing.
    """
    log.info("Mode: AUTO (GPU first, CPU fallback)")
    options_gpu = {**RTSP_OPTIONS, "hwaccel": "cuda", "hwaccel_device": "0"}

    try:
        c = av.open(url, options=options_gpu, timeout=15.0)
        s = c.streams.video[0]
        log.info(f"Connected — {s.codec_context.width}x{s.codec_context.height} codec:{s.codec_context.name}")
        n = decode_loop(c, s, "AUTO-GPU")
        if n == 0:
            log.warning("GPU auto gave 0 frames — retrying with CPU")
            mode_cpu(url)
    except Exception as e:
        log.warning(f"GPU auto failed ({e}) — falling back to CPU")
        mode_cpu(url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",  required=True)
    parser.add_argument("--mode", default="gpu",
                        choices=["cpu", "gpu", "auto"],
                        help="cpu=software, gpu=hevc_cuvid, auto=try gpu then cpu")
    args = parser.parse_args()

    if args.mode == "cpu":
        mode_cpu(args.url)
    elif args.mode == "gpu":
        mode_gpu_cuvid(args.url)
    else:
        mode_auto(args.url)