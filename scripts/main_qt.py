"""
Escher Photobooth — two-window PyQt6 UI

  PreviewWindow  — video only; drag to projector, press F to fullscreen
  ControlsWindow — sliders, dropdowns, photobooth buttons

python3 scripts/main_qt.py
"""
from __future__ import annotations

import sys, os, json, time, datetime, smtplib, zipfile, threading
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import numpy as np
import cv2
import mlx.core as mx

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
        QLabel, QComboBox, QSlider, QPushButton, QLineEdit,
        QStackedWidget, QFrame, QSizePolicy, QMessageBox, QCheckBox,
        QScrollArea, QFileDialog,
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtGui import QImage, QPixmap
except ImportError:
    print("PyQt6 not found.  Install with:  pip install PyQt6")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_source import CameraSource, probe_cameras

# ── paths & config ─────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR   = os.path.join(BASE_DIR, "assets")
CAPTURES_DIR = os.path.join(BASE_DIR, "captures")
os.makedirs(CAPTURES_DIR, exist_ok=True)

INPUT_FILE = os.path.join(ASSETS_DIR, "centered_eye.png")
DST_H, DST_W = 800, 800

EXPERIENCES  = ["Droste", "Balcony", "Fisheye", "Möbius", "Möbius Types"]
MOBIUS_TYPES = ["Elliptic", "Hyperbolic", "Loxodromic", "Parabolic"]

DROSTE_DEF = {"Outer": 1280, "Inner": 85, "FOV": 100}
_cfg_path  = os.path.join(ASSETS_DIR, "transform_config.json")
if os.path.exists(_cfg_path):
    try:
        _cfg = json.load(open(_cfg_path))
        DROSTE_DEF.update(_cfg)
        INPUT_FILE = _cfg.get("AssetInput", INPUT_FILE)
    except Exception:
        pass

EMAIL_CFG = {
    "smtp_host": "smtp.gmail.com", "smtp_port": 587,
    "username": "", "password": "", "from_addr": "",
}
_ecfg_path = os.path.join(BASE_DIR, "email_config.json")
if os.path.exists(_ecfg_path):
    try:
        EMAIL_CFG.update(json.load(open(_ecfg_path)))
    except Exception:
        pass

# ── static image ───────────────────────────────────────────────────────────────
try:
    _img = cv2.imread(INPUT_FILE, cv2.IMREAD_UNCHANGED)
    if _img is None:
        raise FileNotFoundError(INPUT_FILE)
    if _img.shape[2] == 3:
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2BGRA)
    STATIC_FULL  = _img
    STATIC_H, STATIC_W = _img.shape[:2]
    STATIC_SMALL = cv2.resize(_img, (DST_W, DST_H))
except Exception as e:
    print(f"Image load error: {e}")
    STATIC_FULL = STATIC_SMALL = np.zeros((DST_H, DST_W, 4), dtype=np.uint8)
    STATIC_H = STATIC_W = DST_W

# ── MLX coordinate grids ───────────────────────────────────────────────────────
Y_GRID, X_GRID = mx.meshgrid(
    mx.arange(DST_H, dtype=mx.float32),
    mx.arange(DST_W, dtype=mx.float32),
    indexing="ij",
)

# ── rendering functions ─────────────────────────────────────────────────────────
MAX_FISH_R = DST_H * 0.23

def _droste_single(zoom, outer, inner, fov, sw, sh):
    lr  = np.log(outer / inner)
    ez  = np.exp(np.log(max(zoom, 1e-10)) % lr)
    d   = lr ** 2 + (2 * np.pi) ** 2
    ar, ai = (2 * np.pi) ** 2 / d,  2 * np.pi * lr / d
    ia  = ar ** 2 + ai ** 2
    iar, iai = ar / ia, -ai / ia
    vs  = (outer / (DST_W * ez)) * (fov / 100.0)
    zx, zy = (X_GRID - DST_W / 2) * vs, (Y_GRID - DST_H / 2) * vs
    zm  = mx.sqrt(zx ** 2 + zy ** 2) + 1e-9
    za  = mx.arctan2(zy, zx)
    rr  = mx.log(zm) * iar - za * iai
    ri  = mx.log(zm) * iai + za * iar
    mw  = mx.exp(rr)
    sh_ = mx.remainder(rr - np.log(inner), lr) + np.log(inner)
    sf  = mx.exp(sh_) / mw
    fa  = ri - np.log(ez) * (lr / (2 * np.pi))
    return mw * mx.cos(fa) * sf + sw / 2, mw * mx.sin(fa) * sf + sh / 2

def render_droste(src, zoom, outer, inner, fov, sw, sh):
    ratio = outer / inner
    mx1, my1 = _droste_single(zoom, outer, inner, fov, sw, sh)
    mx2, my2 = _droste_single(zoom / ratio, outer, inner, fov, sw, sh)
    mx.eval(mx1, my1, mx2, my2)
    f1 = cv2.remap(src, np.array(mx1), np.array(my1), cv2.INTER_LINEAR)
    f2 = cv2.remap(src, np.array(mx2), np.array(my2), cv2.INTER_LINEAR)
    a  = f2[:, :, 3:] / 255.0
    return (f2[:, :, :3] * a + f1[:, :, :3] * (1 - a)).astype(np.uint8)

def render_balcony(src, cx, cy, radius, mag):
    ox, oy = X_GRID - cx, Y_GRID - cy
    r   = mx.sqrt(ox ** 2 + oy ** 2)
    t   = mx.minimum(r / (radius + 1e-9), 1.0)
    k   = (1.0 - t * t) ** 2
    sc  = 1.0 + (mag - 1.0) * k
    mx1, my1 = cx + ox / sc, cy + oy / sc
    mx.eval(mx1, my1)
    out = cv2.remap(src, np.array(mx1), np.array(my1),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.ascontiguousarray(out[:, :, :3])

def render_fisheye(src, cx, cy, radius, depth):
    ox, oy = X_GRID - cx, Y_GRID - cy
    r    = mx.sqrt(ox ** 2 + oy ** 2)
    nr   = r / (radius + 1e-9)
    th   = mx.arcsin(mx.minimum(nr, 0.9999))
    sr   = mx.tanh(mx.tan(mx.minimum(th * depth, float(np.pi) * 0.48)) * radius / MAX_FISH_R) * MAX_FISH_R
    dx   = mx.where(r > 1e-6, ox / r, mx.ones_like(ox))
    dy   = mx.where(r > 1e-6, oy / r, mx.zeros_like(oy))
    mx1, my1 = cx + dx * sr, cy + dy * sr
    mx.eval(mx1, my1, nr)
    out  = cv2.remap(src, np.array(mx1), np.array(my1),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    nrn  = np.array(nr)
    out[nrn >= 1.0] = [0, 0, 0, 255]
    lm   = nrn > 0.7
    t    = ((nrn[lm] - 0.7) / 0.3)
    out[lm, :3] = (out[lm, :3] * (1 - t ** 1.5)[:, None]).astype(np.uint8)
    return np.ascontiguousarray(out[:, :, :3])

def render_mobius(src, cx, cy, scale, rot):
    zx, zy = X_GRID - cx, Y_GRID - cy
    rn  = 2.0 * mx.sqrt(zx ** 2 + zy ** 2) / (scale + 1e-8)
    v   = rot / (rn + 1e-4)
    cv_, sv_ = mx.cos(v), mx.sin(v)
    zxv, zyv = zx * cv_ - zy * sv_, zx * sv_ + zy * cv_
    r2  = zxv ** 2 + zyv ** 2 + 1e-8
    hs  = scale / 2.0
    mx1, my1 = cx + hs ** 2 * zxv / r2, cy - hs ** 2 * zyv / r2
    mx.eval(mx1, my1)
    out = cv2.remap(src, np.array(mx1), np.array(my1),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.ascontiguousarray(out[:, :, :3])

def _cm_r(ax, ay, bx, by): return ax * bx - ay * by
def _cm_i(ax, ay, bx, by): return ax * by + ay * bx
def _cd_r(ax, ay, bx, by): return (ax*bx + ay*by) / (bx**2 + by**2 + 1e-8)
def _cd_i(ax, ay, bx, by): return (ay*bx - ax*by) / (bx**2 + by**2 + 1e-8)

def render_mobius_types(src, cx, cy, scale, pa, pb, ttype):
    hs  = scale / 2.0
    znx = (X_GRID - cx) / (hs + 1e-8)
    zny = (Y_GRID - cy) / (hs + 1e-8)
    if ttype < 3:
        if   ttype == 0: lx, ly = float(np.cos(-pb)), float(np.sin(-pb))
        elif ttype == 1: lx, ly = float(pa), 0.0
        else:            lx, ly = float(pa * np.cos(-pb)), float(pa * np.sin(-pb))
        omx, omy = 1.0 - lx, -ly
        nx = _cm_r(znx, zny, omx, omy) + 2.0
        ny = _cm_i(znx, zny, omx, omy)
        dx = _cm_r(2*lx, 2*ly, znx, zny)
        dy = _cm_i(2*lx, 2*ly, znx, zny)
        wx, wy = _cd_r(nx, ny, dx, dy), _cd_i(nx, ny, dx, dy)
    else:
        tx, ty = float(pa), float(pb)
        nx = 1.0 - _cm_r(tx, ty, znx, zny)
        ny =       - _cm_i(tx, ty, znx, zny)
        wx, wy = _cd_r(nx, ny, znx, zny), _cd_i(nx, ny, znx, zny)
    mx1, my1 = cx + wx * hs, cy + wy * hs
    mx.eval(mx1, my1)
    out = cv2.remap(src, np.array(mx1), np.array(my1),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return np.ascontiguousarray(out[:, :, :3])


# ── EffectsWorker ──────────────────────────────────────────────────────────────
class EffectsWorker(QThread):
    frame_ready = pyqtSignal(object)   # np.ndarray BGR uint8

    def __init__(self, params: dict):
        super().__init__()
        self._p       = params
        self._running = True
        self._cam     = CameraSource(DST_W, DST_H)
        self._zoom    = 1.0
        self._rot     = 0.0
        self._pb      = 0.0
        self._lock    = threading.Lock()
        self.current_frame = None
        self._writer  = None
        self._recording = False

    def run(self):
        last = time.time()
        while self._running:
            now = time.time();  dt = min(now - last, 0.1);  last = now
            p   = self._p

            sidx = p.get("source_idx", -1)
            self._cam.set_index(sidx)
            if self._cam.active:
                frm = self._cam.read_bgra()
                if frm is None:
                    time.sleep(0.01);  continue
                src = cv2.flip(frm, 1) if p.get("mirror") else frm
                gamma = p.get("gamma", 1.0)
                if gamma != 1.0:
                    # Gamma LUT: output = (input/255)^(1/gamma) * 255
                    # Values > 1 lift midtones/shadows without clipping highlights.
                    lut = np.clip(
                        (np.arange(256, dtype=np.float32) / 255.0) ** (1.0 / gamma) * 255,
                        0, 255
                    ).astype(np.uint8)
                    src = np.dstack([cv2.LUT(src[:, :, :3], lut), src[:, :, 3]])
                sw = sh = DST_W
            else:
                exp = p.get("experience", 0)
                src = STATIC_FULL  if exp == 0 else STATIC_SMALL
                sw  = STATIC_W     if exp == 0 else DST_W
                sh  = STATIC_H     if exp == 0 else DST_H

            exp     = p.get("experience", 0)
            playing = p.get("playing", False)

            if playing:
                if exp == 0:   self._zoom *= np.exp(p.get("droste_speed", 0.5) * dt)
                elif exp == 3: self._rot  += p.get("mobius_speed",  1.0) * dt
                elif exp == 4: self._pb   += p.get("mobiust_speed", 0.5) * dt
            else:
                if exp == 3:   self._rot = p.get("mobius_rot", 0.0)
                if exp == 4:   self._pb  = p.get("param_b",    0.0)

            try:
                if exp == 0:
                    result = render_droste(
                        src, self._zoom,
                        p.get("outer", 1280), p.get("inner", 85),
                        p.get("fov", 100), sw, sh)
                elif exp == 1:
                    result = render_balcony(
                        src, p.get("cx", 400), p.get("cy", 400),
                        p.get("radius", 200), p.get("magnification", 2.0))
                elif exp == 2:
                    result = render_fisheye(
                        src, p.get("cx", 400), p.get("cy", 400),
                        p.get("radius", 300), p.get("depth", 0.7))
                elif exp == 3:
                    result = render_mobius(
                        src, p.get("cx", 400), p.get("cy", 400),
                        p.get("scale", 300), self._rot)
                else:
                    result = render_mobius_types(
                        src, p.get("cx", 400), p.get("cy", 400),
                        p.get("scale", 300), p.get("param_a", 1.0),
                        self._pb, p.get("mobius_type", 0))
            except Exception as e:
                print(f"Render error: {e}")
                time.sleep(0.01);  continue

            with self._lock:
                self.current_frame = result
            if self._recording and self._writer is not None:
                self._writer.write(result)
            self.frame_ready.emit(result)

    def start_recording(self, path: str):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer    = cv2.VideoWriter(path, fourcc, 30, (DST_W, DST_H))
        self._recording = True

    def stop_recording(self):
        self._recording = False
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def grab_frame(self):
        with self._lock:
            return self.current_frame.copy() if self.current_frame is not None else None

    def stop(self):
        self._running = False
        self._cam.release()
        if self._writer is not None:
            self._writer.release()


# ── LabeledSlider ──────────────────────────────────────────────────────────────
class LabeledSlider(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(self, label: str, mn: float, mx_: float, default: float,
                 decimals: int = 0, parent=None):
        super().__init__(parent)
        self._scale = 10 ** decimals
        self._dec   = decimals

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(6)

        lbl = QLabel(label)
        lbl.setFixedWidth(108)
        lbl.setStyleSheet("color: #aaa; font-size: 12px;")

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(int(mn * self._scale), int(mx_ * self._scale))
        self._slider.setValue(int(default * self._scale))

        self._val = QLabel(f"{default:.{decimals}f}")
        self._val.setFixedWidth(46)
        self._val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._val.setStyleSheet("color: #ddd; font-size: 12px;")

        self._slider.valueChanged.connect(self._on_slide)

        row.addWidget(lbl)
        row.addWidget(self._slider, 1)
        row.addWidget(self._val)

    def _on_slide(self, raw: int):
        val = raw / self._scale
        self._val.setText(f"{val:.{self._dec}f}")
        self.valueChanged.emit(val)

    def value(self) -> float:
        return self._slider.value() / self._scale


# ── PreviewWindow ──────────────────────────────────────────────────────────────
class PreviewWindow(QMainWindow):
    """Frameless video-only window — drag to projector, F to fullscreen."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Escher — Preview")
        self.resize(800, 800)

        self._countdown  = 0
        self._recording  = False

        central = QWidget()
        central.setStyleSheet("background: #000;")
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self._label)

        # Small fullscreen toggle button overlaid in the corner
        self._fs_btn = QPushButton("⛶", self)
        self._fs_btn.setFixedSize(34, 34)
        self._fs_btn.setToolTip("Toggle fullscreen  (F)")
        self._fs_btn.clicked.connect(self.toggle_fullscreen)
        self._fs_btn.setStyleSheet(
            "QPushButton { background: rgba(0,0,0,160); color: #fff; "
            "border: none; border-radius: 6px; font-size: 16px; }"
            "QPushButton:hover { background: rgba(60,60,60,200); }")
        self._fs_btn.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fs_btn.move(self.width() - 42, 8)

    # called by ControlsWindow when countdown/recording state changes
    def set_countdown(self, n: int):
        self._countdown = n

    def set_recording(self, active: bool):
        self._recording = active

    def update_frame(self, frame: np.ndarray):
        w, h = self._label.width(), self._label.height()
        if w == 0 or h == 0:
            return

        display = frame
        if self._countdown > 0:
            display = frame.copy()
            fh, fw = display.shape[:2]
            cv2.putText(display, str(self._countdown),
                        (fw // 2 - 55, fh // 2 + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 7.0, (255, 255, 255), 16)
        if self._recording:
            if display is frame:
                display = frame.copy()
            cv2.circle(display, (18, 18), 9, (0, 0, 220), -1)

        fh, fw = display.shape[:2]
        rgb  = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, fw, fh, fw * 3, QImage.Format.Format_RGB888).copy()
        pix  = QPixmap.fromImage(qimg).scaled(
            w, h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(pix)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            self._fs_btn.show()
        else:
            self.showFullScreen()
            self._fs_btn.hide()

    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key.Key_F:
            self.toggle_fullscreen()
        elif k == Qt.Key.Key_Escape and self.isFullScreen():
            self.showNormal()
            self._fs_btn.show()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        # Hide rather than destroy — ControlsWindow controls app lifetime
        event.ignore()
        self.hide()


# ── ControlsWindow ─────────────────────────────────────────────────────────────
class ControlsWindow(QMainWindow):
    _send_result = pyqtSignal(bool, str)

    def __init__(self, params: dict, worker: EffectsWorker, preview: PreviewWindow):
        super().__init__()
        self.setWindowTitle("Escher — Controls")
        self.setFixedWidth(320)

        self._params  = params
        self._worker  = worker
        self._preview = preview

        self._captured  = []
        self._recording = False
        self._countdown = 0

        self._countdown_timer = QTimer(self)
        self._countdown_timer.timeout.connect(self._tick_countdown)

        self._send_result.connect(self._on_send_done)

        self._build_ui()
        self._apply_style()

    # ── UI ─────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Wrap everything in a scroll area so it works on small screens
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        self.setCentralWidget(scroll)

        inner = QWidget()
        pv    = QVBoxLayout(inner)
        pv.setContentsMargins(14, 16, 14, 14)
        pv.setSpacing(6)
        scroll.setWidget(inner)

        # ── project to screen ──
        self._fs_btn = QPushButton("⛶  Fullscreen Preview")
        self._fs_btn.setObjectName("fsBtn")
        self._fs_btn.clicked.connect(self._toggle_preview_fullscreen)
        pv.addWidget(self._fs_btn)

        self._show_btn = QPushButton("Show Preview Window")
        self._show_btn.setObjectName("showBtn")
        self._show_btn.clicked.connect(lambda: (
            self._preview.show(), self._preview.raise_()))
        pv.addWidget(self._show_btn)

        pv.addWidget(self._sep())

        # ── experience ──
        pv.addWidget(self._hdr("EXPERIENCE"))
        self._exp_combo = QComboBox()
        self._exp_combo.addItems(EXPERIENCES)
        self._exp_combo.currentIndexChanged.connect(self._on_exp_change)
        pv.addWidget(self._exp_combo)

        # ── source ──
        pv.addWidget(self._hdr("SOURCE"))
        self._src_combo = QComboBox()
        self._src_combo.addItem("Static image")
        for idx, name, w, h in self._cams:
            self._src_combo.addItem(f"{name}  ({w}×{h})")
        self._src_combo.currentIndexChanged.connect(self._on_src_change)
        pv.addWidget(self._src_combo)

        self._img_btn = QPushButton("Choose Image…")
        self._img_btn.setObjectName("imgBtn")
        self._img_btn.clicked.connect(self._choose_image)
        pv.addWidget(self._img_btn)

        self._mirror_chk = QCheckBox("Mirror camera")
        self._mirror_chk.toggled.connect(lambda v: self._params.update({"mirror": v}))
        pv.addWidget(self._mirror_chk)

        self._gamma_slider = LabeledSlider("Gamma", 0.3, 3.0, 1.0, decimals=2)
        self._gamma_slider.valueChanged.connect(
            lambda v: self._params.update({"gamma": v}))
        pv.addWidget(self._gamma_slider)

        pv.addWidget(self._sep())

        # ── per-experience sliders ──
        self._stack = QStackedWidget()
        self._stack.addWidget(self._droste_page())
        self._stack.addWidget(self._balcony_page())
        self._stack.addWidget(self._fisheye_page())
        self._stack.addWidget(self._mobius_page())
        self._stack.addWidget(self._mobius_types_page())
        pv.addWidget(self._stack)

        # ── play / pause ──
        self._play_btn = QPushButton("▶  Play")
        self._play_btn.setObjectName("playBtn")
        self._play_btn.setCheckable(True)
        self._play_btn.clicked.connect(self._on_play)
        pv.addWidget(self._play_btn)

        pv.addWidget(self._sep())

        # ── photobooth ──
        pv.addWidget(self._hdr("PHOTOBOOTH"))

        self._photo_btn = QPushButton("📷  Photo  (3 s timer)")
        self._photo_btn.setObjectName("photoBtn")
        self._photo_btn.clicked.connect(self._start_photo)
        pv.addWidget(self._photo_btn)

        self._rec_btn = QPushButton("⏺  Start recording")
        self._rec_btn.setObjectName("recordBtn")
        self._rec_btn.clicked.connect(self._toggle_record)
        pv.addWidget(self._rec_btn)

        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #666; font-size: 11px;")
        pv.addWidget(self._status_lbl)

        pv.addWidget(self._sep())

        # ── send ──
        pv.addWidget(self._hdr("SEND"))
        self._email_input = QLineEdit()
        self._email_input.setPlaceholderText("email@example.com")
        pv.addWidget(self._email_input)

        self._send_btn = QPushButton("✉  Send to email")
        self._send_btn.setObjectName("sendBtn")
        self._send_btn.clicked.connect(self._send_email)
        pv.addWidget(self._send_btn)

        pv.addStretch()

    # _cams is set before _build_ui by the factory function below
    _cams: list = []

    # ── per-experience control pages ───────────────────────────────────────────
    def _make_page(self, spec: list) -> QWidget:
        w  = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 4, 0, 4)
        vl.setSpacing(2)
        for key, label, mn, mx_, default, dec in spec:
            s = LabeledSlider(label, mn, mx_, default, dec)
            s.valueChanged.connect(lambda val, k=key: self._params.update({k: val}))
            vl.addWidget(s)
        vl.addStretch()
        return w

    def _droste_page(self) -> QWidget:
        return self._make_page([
            ("outer",        "Outer radius",  100,  2000, DROSTE_DEF["Outer"],  0),
            ("inner",        "Inner radius",  5,    500,  DROSTE_DEF["Inner"],  0),
            ("fov",          "Field of view", 1,    300,  DROSTE_DEF["FOV"],    0),
            ("droste_speed", "Speed",        -1.5,  1.5,  0.5,                  2),
        ])

    def _balcony_page(self) -> QWidget:
        return self._make_page([
            ("cx",            "Center X",      0,   DST_W, DST_W // 2, 0),
            ("cy",            "Center Y",      0,   DST_H, DST_H // 2, 0),
            ("radius",        "Radius",        1,   400,   200,        0),
            ("magnification", "Magnification", 0.1, 10.0,  2.0,        1),
        ])

    def _fisheye_page(self) -> QWidget:
        return self._make_page([
            ("cx",     "Center X", 0,   DST_W, DST_W // 2, 0),
            ("cy",     "Center Y", 0,   DST_H, DST_H // 2, 0),
            ("radius", "Radius",   1,   400,   300,        0),
            ("depth",  "Depth",    0.1, 3.0,   0.7,        2),
        ])

    def _mobius_page(self) -> QWidget:
        return self._make_page([
            ("cx",           "Center X",   0,   DST_W, DST_W // 2, 0),
            ("cy",           "Center Y",   0,   DST_H, DST_H // 2, 0),
            ("scale",        "Scale",      10,  DST_W, 300,        0),
            ("mobius_rot",   "Rotation",  -10,  10,    0.0,        2),
            ("mobius_speed", "Spin speed", -3,   3,    1.0,        2),
        ])

    def _mobius_types_page(self) -> QWidget:
        w  = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 4, 0, 4)
        vl.setSpacing(4)

        type_lbl = QLabel("Type")
        type_lbl.setStyleSheet("color: #aaa; font-size: 12px;")
        vl.addWidget(type_lbl)

        type_box = QComboBox()
        type_box.addItems(MOBIUS_TYPES)
        type_box.currentIndexChanged.connect(
            lambda i: self._params.update({"mobius_type": i}))
        vl.addWidget(type_box)

        for key, label, mn, mx_, default, dec in [
            ("cx",            "Center X",   0,     DST_W, DST_W // 2, 0),
            ("cy",            "Center Y",   0,     DST_H, DST_H // 2, 0),
            ("scale",         "Scale",      10,    DST_W, 300,        0),
            ("param_a",       "Param A",    0.0,   5.0,   1.0,        2),
            ("param_b",       "Param B",   -3.14,  3.14,  0.0,        2),
            ("mobiust_speed", "Spin speed", -3.0,  3.0,   0.5,        2),
        ]:
            s = LabeledSlider(label, mn, mx_, default, dec)
            s.valueChanged.connect(lambda val, k=key: self._params.update({k: val}))
            vl.addWidget(s)

        vl.addStretch()
        return w

    # ── helpers ────────────────────────────────────────────────────────────────
    @staticmethod
    def _hdr(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "color: #555; font-size: 10px; font-weight: bold; "
            "letter-spacing: 1.5px; margin-top: 4px;")
        return lbl

    @staticmethod
    def _sep() -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.Shape.HLine)
        f.setStyleSheet("color: #2e2e2e; margin: 4px 0;")
        return f

    # ── event handlers ─────────────────────────────────────────────────────────
    def _choose_image(self):
        global STATIC_FULL, STATIC_SMALL, STATIC_H, STATIC_W
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose Image", "",
            "Images (*.png *.jpg *.jpeg *.tiff *.bmp *.webp)")
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            QMessageBox.warning(self, "Image", f"Could not load:\n{path}")
            return
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        STATIC_FULL  = img
        STATIC_H, STATIC_W = img.shape[:2]
        STATIC_SMALL = cv2.resize(img, (DST_W, DST_H))
        self._worker._zoom = 1.0

    def _toggle_preview_fullscreen(self):
        if not self._preview.isVisible():
            self._preview.show()
        self._preview.toggle_fullscreen()

    def _on_exp_change(self, idx: int):
        self._params["experience"] = idx
        self._stack.setCurrentIndex(idx)
        self._worker._zoom = 1.0
        self._worker._rot  = 0.0
        self._worker._pb   = 0.0
        self._play_btn.setChecked(False)
        self._params["playing"] = False
        self._play_btn.setText("▶  Play")

    def _on_src_change(self, idx: int):
        self._params["source_idx"] = idx - 1

    def _on_play(self, checked: bool):
        self._params["playing"] = checked
        self._play_btn.setText("⏸  Pause" if checked else "▶  Play")

    def _start_photo(self):
        if self._countdown > 0:
            return
        self._countdown = 3
        self._preview.set_countdown(3)
        self._photo_btn.setEnabled(False)
        self._countdown_timer.start(1000)

    def _tick_countdown(self):
        self._countdown -= 1
        self._preview.set_countdown(self._countdown)
        if self._countdown <= 0:
            self._countdown = 0
            self._countdown_timer.stop()
            self._photo_btn.setEnabled(True)
            self._capture_photo()

    def _capture_photo(self):
        frame = self._worker.grab_frame()
        if frame is None:
            return
        ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(CAPTURES_DIR, f"photo_{ts}.jpg")
        cv2.imwrite(path, frame)
        self._captured.append(path)
        self._update_status()

    def _toggle_record(self):
        if not self._recording:
            ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(CAPTURES_DIR, f"video_{ts}.mp4")
            self._worker.start_recording(path)
            self._captured.append(path)
            self._recording = True
            self._preview.set_recording(True)
            self._rec_btn.setText("⏹  Stop recording")
        else:
            self._worker.stop_recording()
            self._recording = False
            self._preview.set_recording(False)
            self._rec_btn.setText("⏺  Start recording")
            self._update_status()

    def _update_status(self):
        n = len(self._captured)
        self._status_lbl.setText(
            f"{n} file{'s' if n != 1 else ''} captured")

    def _send_email(self):
        to = self._email_input.text().strip()
        if not to:
            QMessageBox.warning(self, "Send", "Enter an email address first.")
            return
        if not self._captured:
            QMessageBox.warning(self, "Send", "No photos or videos captured yet.")
            return
        self._send_btn.setEnabled(False)
        self._send_btn.setText("Sending…")
        threading.Thread(target=self._do_send, args=(to,), daemon=True).start()

    def _do_send(self, to: str):
        try:
            cfg = EMAIL_CFG
            if not cfg["username"]:
                raise ValueError(
                    "Email not configured.\n"
                    f"Create  {os.path.join(BASE_DIR, 'email_config.json')}  with:\n"
                    '{"smtp_host":"smtp.gmail.com","smtp_port":587,'
                    '"username":"you@gmail.com","password":"app-password",'
                    '"from_addr":"you@gmail.com"}'
                )
            msg           = MIMEMultipart()
            msg["From"]   = cfg["from_addr"] or cfg["username"]
            msg["To"]     = to
            msg["Subject"] = "Your Escher photobooth captures"
            msg.attach(MIMEText("Here are your captures from the Escher photobooth!", "plain"))

            files = [f for f in self._captured if os.path.exists(f)]
            if len(files) > 5:
                zip_path = os.path.join(CAPTURES_DIR, "captures.zip")
                with zipfile.ZipFile(zip_path, "w") as zf:
                    for f in files:
                        zf.write(f, os.path.basename(f))
                files = [zip_path]

            for path in files:
                with open(path, "rb") as fh:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(fh.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", "attachment",
                                filename=os.path.basename(path))
                msg.attach(part)

            with smtplib.SMTP(cfg["smtp_host"], cfg["smtp_port"]) as srv:
                srv.starttls()
                srv.login(cfg["username"], cfg["password"])
                srv.sendmail(msg["From"], to, msg.as_string())

            self._send_result.emit(True, "Sent successfully!")
        except Exception as e:
            self._send_result.emit(False, str(e))

    def _on_send_done(self, ok: bool, msg: str):
        self._send_btn.setText("✉  Send to email")
        self._send_btn.setEnabled(True)
        if ok:
            QMessageBox.information(self, "Send", msg)
        else:
            QMessageBox.critical(self, "Send error", msg)

    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key.Key_Space:
            self._play_btn.toggle()
            self._on_play(self._play_btn.isChecked())
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self._worker.stop()
        self._worker.wait(2000)
        # Allow preview to truly close now
        self._preview.closeEvent = lambda e: e.accept()
        self._preview.close()
        event.accept()

    # ── stylesheet ─────────────────────────────────────────────────────────────
    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #1a1a1a;
                color: #e0e0e0;
                font-family: -apple-system, 'Helvetica Neue', sans-serif;
                font-size: 13px;
            }
            QScrollArea, QScrollArea > QWidget > QWidget { background: #1a1a1a; }
            QScrollBar:vertical { width: 6px; background: #1a1a1a; }
            QScrollBar::handle:vertical { background: #333; border-radius: 3px; }
            QComboBox {
                background: #2c2c2c; border: 1px solid #3a3a3a;
                border-radius: 6px; padding: 6px 10px; min-height: 28px;
            }
            QComboBox::drop-down { border: none; padding-right: 6px; }
            QComboBox QAbstractItemView {
                background: #2c2c2c; border: 1px solid #444;
                selection-background-color: #383838;
            }
            QSlider::groove:horizontal {
                height: 4px; background: #333; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #4a9eff; width: 14px; height: 14px;
                border-radius: 7px; margin: -5px 0;
            }
            QSlider::sub-page:horizontal { background: #4a9eff; border-radius: 2px; }
            QPushButton {
                border-radius: 7px; padding: 9px; font-size: 13px;
                font-weight: 600; border: none; min-height: 38px;
            }
            QPushButton#fsBtn   { background: #374151; color: #e0e0e0; }
            QPushButton#showBtn { background: #2c2c2c; color: #aaa;
                                  border: 1px solid #3a3a3a; min-height: 30px;
                                  font-size: 12px; padding: 5px; }
            QPushButton#imgBtn  { background: #2c2c2c; color: #aaa;
                                  border: 1px solid #3a3a3a; min-height: 30px;
                                  font-size: 12px; padding: 5px; }
            QPushButton#photoBtn  { background: #2563eb; color: #fff; }
            QPushButton#recordBtn { background: #dc2626; color: #fff; }
            QPushButton#sendBtn   { background: #16a34a; color: #fff; }
            QPushButton#playBtn   {
                background: #2c2c2c; color: #e0e0e0; border: 1px solid #3a3a3a;
            }
            QPushButton#playBtn:checked { background: #1d4ed8; color: #fff; border: none; }
            QPushButton:disabled { opacity: 0.4; }
            QLineEdit {
                background: #2c2c2c; border: 1px solid #3a3a3a;
                border-radius: 6px; padding: 8px 10px;
            }
            QCheckBox { color: #aaa; font-size: 12px; }
        """)


# ── entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cams = probe_cameras()
    print(f"Cameras: {[(name, f'{w}x{h}') for _, name, w, h in cams]}")
    print(f"Captures → {CAPTURES_DIR}")

    app = QApplication(sys.argv)
    app.setApplicationName("Escher")

    params  = {
        "experience":    0,
        "source_idx":   -1,
        "mirror":       False,
        "playing":      False,
        "gamma":        1.0,
        "outer":        float(DROSTE_DEF.get("Outer", 1280)),
        "inner":        float(DROSTE_DEF.get("Inner", 85)),
        "fov":          float(DROSTE_DEF.get("FOV", 100)),
        "droste_speed": 0.5,
        "cx": 400.0, "cy": 400.0, "radius": 200.0,
        "magnification": 2.0, "depth": 0.7,
        "scale": 300.0, "mobius_rot": 0.0, "mobius_speed": 1.0,
        "mobius_type": 0, "param_a": 1.0, "param_b": 0.0, "mobiust_speed": 0.5,
    }

    worker  = EffectsWorker(params)
    preview = PreviewWindow()

    ControlsWindow._cams = cams
    controls = ControlsWindow(params, worker, preview)

    worker.frame_ready.connect(preview.update_frame)
    worker.start()

    preview.show()
    controls.show()

    # Position controls to the right of the preview by default
    preview.move(100, 100)
    controls.move(preview.x() + preview.width() + 10, preview.y())

    sys.exit(app.exec())
