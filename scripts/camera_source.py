"""
Camera capture with AVFoundation (via ffmpeg) as the primary backend.

AVFoundation gives USB cameras the same auto-exposure, white-balance, and
color processing that Zoom/FaceTime use.  OpenCV's VideoCapture bypasses all
of that and reads raw UVC frames, which look underexposed/flat by comparison.

Falls back to OpenCV if ffmpeg is not found.

probe_cameras() → [(avf_index, name, width, height), ...]
CameraSource(dst_w, dst_h) → .set_index(i) / .read_bgra() / .release()
"""
from __future__ import annotations

import re
import subprocess
import threading
import cv2
import numpy as np

# ── ffmpeg discovery ───────────────────────────────────────────────────────────
_FFMPEG: str | None = None   # cached path, None = not found yet, "" = not available

def _find_ffmpeg() -> str:
    global _FFMPEG
    if _FFMPEG is not None:
        return _FFMPEG
    for path in ["ffmpeg", "/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        try:
            subprocess.run([path, "-version"], capture_output=True, check=True, timeout=3)
            _FFMPEG = path
            return path
        except Exception:
            pass
    _FFMPEG = ""
    return ""

# ── AVFoundation device enumeration ───────────────────────────────────────────
# Capture at 1280×720 via AVFoundation; Python resizes to dst size.
# 1280×720 is universally supported by modern webcams including the C920.
_AVF_W, _AVF_H = 1280, 720

def _list_avf_devices(ffmpeg: str) -> list[tuple[int, str]]:
    """Return [(avf_index, name), ...] for video-only devices (no screen capture)."""
    try:
        r = subprocess.run(
            [ffmpeg, "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=5,
        )
        devices, in_video = [], False
        for line in r.stderr.splitlines():
            if "AVFoundation video devices" in line:
                in_video = True
            elif "AVFoundation audio devices" in line:
                break
            elif in_video:
                m = re.search(r"\[(\d+)\] (.+)", line)
                if m:
                    name = m.group(2).strip()
                    if "screen" not in name.lower():
                        devices.append((int(m.group(1)), name))
        return devices
    except Exception:
        return []


def probe_cameras(max_index: int = 5) -> list[tuple[int, str, int, int]]:
    """
    Return [(index, name, width, height), ...] for available cameras.

    With ffmpeg: uses AVFoundation indices + real device names.
    Without ffmpeg: falls back to OpenCV indices with generic names.
    """
    ffmpeg = _find_ffmpeg()
    if ffmpeg:
        devices = _list_avf_devices(ffmpeg)
        if devices:
            return [(idx, name, _AVF_W, _AVF_H) for idx, name in devices]

    # OpenCV fallback
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            found.append((i, f"Camera {i}", w, h))
        cap.release()
    return found


# ── CameraSource ───────────────────────────────────────────────────────────────
class CameraSource:
    def __init__(self, dst_w: int, dst_h: int):
        self.dst_w = dst_w
        self.dst_h = dst_h

        self._index: int | None = None

        # AVFoundation path (ffmpeg)
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._reader_running = False
        self._latest: np.ndarray | None = None   # most recent BGRA frame
        self._avf_lock = threading.Lock()

        # OpenCV fallback path
        self._cap: cv2.VideoCapture | None = None

        self._last_frame: np.ndarray | None = None  # returned on dropped frames

    # ── public API ─────────────────────────────────────────────────────────────
    def set_index(self, index: int) -> bool:
        """Switch to camera `index`.  index=-1 = static-image mode."""
        if index == self._index:
            return self.active or index == -1
        self._stop()
        self._index = index
        self._last_frame = None
        if index < 0:
            return True

        ffmpeg = _find_ffmpeg()
        if ffmpeg and self._start_avf(index, ffmpeg):
            return True
        return self._start_cv(index)

    @property
    def active(self) -> bool:
        if self._proc is not None:
            return self._proc.poll() is None
        return self._cap is not None and self._cap.isOpened()

    def read_bgra(self) -> np.ndarray | None:
        if self._proc is not None:
            with self._avf_lock:
                frame = self._latest
            if frame is not None:
                self._last_frame = frame
            return frame if frame is not None else self._last_frame

        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        if not ret:
            return self._last_frame
        frame = cv2.resize(frame, (self.dst_w, self.dst_h))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        self._last_frame = frame
        return frame

    def release(self):
        self._stop()
        self._index = None

    # ── AVFoundation path ──────────────────────────────────────────────────────
    def _start_avf(self, index: int, ffmpeg: str) -> bool:
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-f", "avfoundation",
            "-framerate", "30",
            "-video_size", f"{_AVF_W}x{_AVF_H}",
            "-i", f"{index}:none",
            "-vf", "format=bgr24",
            "-f", "rawvideo",
            "pipe:1",
        ]
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=_AVF_W * _AVF_H * 3 * 8,
            )
            # Read one frame to confirm the camera opened
            n = _AVF_W * _AVF_H * 3
            raw = proc.stdout.read(n)
            if len(raw) != n:
                proc.kill()
                return False

            frame = np.frombuffer(raw, dtype=np.uint8).reshape(_AVF_H, _AVF_W, 3)
            bgra  = self._to_dst(frame)
            with self._avf_lock:
                self._latest = bgra

            self._proc           = proc
            self._reader_running = True
            self._reader = threading.Thread(
                target=self._avf_reader, daemon=True)
            self._reader.start()
            print(f"[camera] AVFoundation: device {index} @ {_AVF_W}×{_AVF_H}")
            return True
        except Exception as e:
            print(f"[camera] AVFoundation failed for device {index}: {e}")
            return False

    def _avf_reader(self):
        n    = _AVF_W * _AVF_H * 3
        proc = self._proc
        while self._reader_running and proc.poll() is None:
            try:
                raw = proc.stdout.read(n)
                if len(raw) != n:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape(_AVF_H, _AVF_W, 3)
                bgra  = self._to_dst(frame)
                with self._avf_lock:
                    self._latest = bgra
            except Exception:
                break

    def _to_dst(self, bgr: np.ndarray) -> np.ndarray:
        """Resize to dst dimensions and add alpha channel."""
        if bgr.shape[1] != self.dst_w or bgr.shape[0] != self.dst_h:
            bgr = cv2.resize(bgr, (self.dst_w, self.dst_h))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)

    # ── OpenCV fallback path ───────────────────────────────────────────────────
    def _start_cv(self, index: int) -> bool:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            print(f"[camera] OpenCV: device {index} not available")
            return False
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        for _ in range(30):      # warmup: let auto-exposure settle
            cap.grab()
        self._cap = cap
        print(f"[camera] OpenCV fallback: device {index}")
        return True

    # ── teardown ───────────────────────────────────────────────────────────────
    def _stop(self):
        self._reader_running = False
        if self._proc is not None:
            self._proc.kill()
            self._proc = None
        if self._reader is not None:
            self._reader.join(timeout=1.0)
            self._reader = None
        with self._avf_lock:
            self._latest = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
