"""Camera stream module – serves a video file as a live HLS camera feed.

The edge node treats its configured video input as a camera: FFmpeg loops
the file forever and publishes HLS segments, which the control-plane API
serves under ``/api/cameras/...`` so the web dashboard can watch it live.

Usage::

    from edge_node.camera_stream import start_camera_stream, stop_camera_stream
    start_camera_stream(video_path, settings)  # non-blocking, FFmpeg in bg thread
    stop_camera_stream()                       # terminates FFmpeg
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)

_ffmpeg_process: Optional[subprocess.Popen] = None
_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


def _find_ffmpeg() -> str:
    """Locate the ffmpeg binary."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it: sudo apt install ffmpeg"
        )
    return ffmpeg


def _run_ffmpeg(video_path: str, hls_dir: str) -> None:
    """Run FFmpeg in a loop, restarting when the process exits."""
    global _ffmpeg_process

    ffmpeg = _find_ffmpeg()
    hls_dir_path = Path(hls_dir)
    hls_dir_path.mkdir(parents=True, exist_ok=True)

    playlist = str(hls_dir_path / "stream.m3u8")

    cmd = [
        ffmpeg,
        "-re",                          # read at native frame rate
        "-stream_loop", "-1",           # loop the input forever
        "-i", video_path,
        "-an",                          # drop audio (traffic footage has none)
        "-c:v", "libx264",              # re-encode to H.264
        "-preset", "ultrafast",         # fast encoding for live
        "-tune", "zerolatency",
        "-g", "60",                     # keyframe every 60 frames
        "-sc_threshold", "0",
        "-f", "hls",                    # HLS output
        "-hls_time", "4",               # 4-second segments
        "-hls_list_size", "5",          # keep 5 segments in playlist
        "-hls_flags", "delete_segments",
        "-hls_segment_filename", str(hls_dir_path / "segment_%03d.ts"),
        playlist,
    ]

    LOGGER.info("Starting camera HLS stream: %s -> %s", video_path, playlist)
    LOGGER.debug("FFmpeg command: %s", " ".join(cmd))

    while not _stop_event.is_set():
        try:
            _ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Wait for FFmpeg to finish (it shouldn't due to -stream_loop -1)
            _ffmpeg_process.wait()

            if _stop_event.is_set():
                break

            LOGGER.warning("FFmpeg exited unexpectedly, restarting in 2s...")
            _stop_event.wait(2)

        except Exception as exc:
            LOGGER.error("FFmpeg error: %s", exc)
            if _stop_event.is_set():
                break
            _stop_event.wait(5)


def start_camera_stream(video_path: str, settings) -> None:
    """Start the HLS camera stream in a background thread.

    ``settings`` supplies ``camera_stream_hls_dir``.
    """
    global _thread, _stop_event

    hls_dir = settings.camera_stream_hls_dir

    if not Path(video_path).exists():
        raise FileNotFoundError(f"Camera video not found: {video_path}")

    _stop_event.clear()
    _thread = threading.Thread(
        target=_run_ffmpeg,
        args=(video_path, hls_dir),
        daemon=True,
        name="camera-stream-ffmpeg",
    )
    _thread.start()
    LOGGER.info("Camera stream started: video=%s, hls_dir=%s", video_path, hls_dir)


def stop_camera_stream() -> None:
    """Stop the camera stream and terminate FFmpeg."""
    global _ffmpeg_process, _thread

    _stop_event.set()

    if _ffmpeg_process is not None:
        try:
            _ffmpeg_process.terminate()
            _ffmpeg_process.wait(timeout=5)
        except Exception:
            try:
                _ffmpeg_process.kill()
            except Exception:
                pass
        _ffmpeg_process = None

    if _thread is not None:
        _thread.join(timeout=5)
        _thread = None

    LOGGER.info("Camera stream stopped")
