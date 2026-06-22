import os
import sys
import time
import numpy as np
import cv2
import mlx.core as mx
import json
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_source import CameraSource, probe_cameras

# Detect the logical screen size via osascript (avoids tkinter/OpenCV Tcl conflict).
# Drag the main window to your projector/external display first, then press F.
try:
    _out = subprocess.check_output(
        ['osascript', '-e', 'tell application "Finder" to return bounds of window of desktop'],
        text=True, stderr=subprocess.DEVNULL
    ).strip()
    _parts = [int(x.strip()) for x in _out.split(',')]
    SCREEN_W, SCREEN_H = _parts[2], _parts[3]
except Exception:
    SCREEN_W, SCREEN_H = 1920, 1080

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
INPUT_FILE = os.path.join(ASSETS_DIR, "centered_eye.png") # Save your GIMP work here

DST_H, DST_W = 800, 800
MAIN_WINDOW = "Escher Projection"
CTRL_WINDOW = "Projector Controls"

DEFAULTS = {
    'Outer': 1280,
    'Inner': 85,
    'FOV': 100,
    'FocX': 640,
    'FocY': 640,
    'Speed': 150,
    'Play': 1
}

# --- OPTIONAL JSON CONFIG LOAD ---
CONFIG_FILE = os.path.join(ASSETS_DIR, "transform_config.json")
if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, 'r') as f:
            auto_cfg = json.load(f)
            # Update DEFAULTS with any keys found in the JSON
            DEFAULTS.update(auto_cfg)
            INPUT_FILE = auto_cfg["AssetInput"]
            print(f"Successfully loaded auto-config: {auto_cfg}")
    except Exception as e:
        print(f"Warning: Could not parse {CONFIG_FILE}, using hardcoded defaults. ({e})")

# --- LOAD STATIC ASSET (SUPPORTING ALPHA) ---
try:
    # IMREAD_UNCHANGED is vital to keep the transparency you made in GIMP
    static_img = cv2.imread(INPUT_FILE, cv2.IMREAD_UNCHANGED)
    if static_img is None:
        raise FileNotFoundError(f"Could not find {INPUT_FILE}")

    # If image has no alpha channel, add one so the blending logic doesn't break
    if static_img.shape[2] == 3:
        static_img = cv2.cvtColor(static_img, cv2.COLOR_BGR2BGRA)

    static_h, static_w, _ = static_img.shape
except Exception as e:
    print(f"Error: {e}")
    exit()

# Pre-calculate grids
Y_GRID_MX, X_GRID_MX = mx.meshgrid(
    mx.arange(DST_H, dtype=mx.float32),
    mx.arange(DST_W, dtype=mx.float32),
    indexing='ij'
)

def get_droste_map(zoom, params, eff_sw, eff_sh):
    outer_logical, inner_logical, fov_scale = params
    ratio = outer_logical / inner_logical
    log_ratio = np.log(ratio)
    effective_zoom = np.exp(np.log(max(zoom, 1e-10)) % log_ratio)

    denom = log_ratio**2 + (2 * np.pi)**2
    a_real, a_imag = (2 * np.pi * 2 * np.pi) / denom, (2 * np.pi * log_ratio) / denom
    inv_a_real, inv_a_imag = a_real / (a_real**2 + a_imag**2), -a_imag / (a_real**2 + a_imag**2)

    view_scale = (outer_logical / (DST_W * effective_zoom)) * (fov_scale / 100.0)
    zx, zy = (X_GRID_MX - (DST_W / 2)) * view_scale, (Y_GRID_MX - (DST_H / 2)) * view_scale

    z_mag, z_angle = mx.sqrt(zx**2 + zy**2) + 1e-9, mx.arctan2(zy, zx)
    res_real = mx.log(z_mag) * inv_a_real - z_angle * inv_a_imag
    res_imag = mx.log(z_mag) * inv_a_imag + z_angle * inv_a_real

    mag_warped = mx.exp(res_real)
    shift = mx.remainder(res_real - np.log(inner_logical), log_ratio) + np.log(inner_logical)
    scaling_factor = mx.exp(shift) / mag_warped
    final_angle = res_imag - (np.log(effective_zoom) * (log_ratio / (2 * np.pi)))

    return (mag_warped * mx.cos(final_angle) * scaling_factor + (eff_sw / 2)), \
           (mag_warped * mx.sin(final_angle) * scaling_factor + (eff_sh / 2))

def letterbox(img, target_w, target_h):
    """Scale img to fit target_w×target_h with black bars, no distortion."""
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    sw, sh = int(w * scale), int(h * scale)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y = (target_h - sh) // 2
    x = (target_w - sw) // 2
    canvas[y:y+sh, x:x+sw] = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    return canvas

cams = probe_cameras()
print(f"Available cameras: {[(i, f'{w}x{h}') for i, w, h in cams]}")
print(f"Screen size detected: {SCREEN_W}x{SCREEN_H}")
print("Source trackbar: 0=static image, 1=camera 0, 2=camera 1, ...")
print("Keys: F=fullscreen, Space=play/pause, M=mirror camera, Q=quit")

camera = CameraSource(DST_W, DST_H)
mirror = False
fullscreen = False

# --- UI WINDOWS ---
cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_NORMAL)
cv2.namedWindow(CTRL_WINDOW, cv2.WINDOW_NORMAL)

cv2.createTrackbar('Source', CTRL_WINDOW, 0,                  3 + len(cams), lambda x: None)
cv2.createTrackbar('Outer',  CTRL_WINDOW, DEFAULTS['Outer'],  2000,           lambda x: None)
cv2.createTrackbar('Inner',  CTRL_WINDOW, DEFAULTS['Inner'],  1000,           lambda x: None)
cv2.createTrackbar('FOV',    CTRL_WINDOW, DEFAULTS['FOV'],    1000,           lambda x: None)
cv2.createTrackbar('Speed',  CTRL_WINDOW, DEFAULTS['Speed'],  200,            lambda x: None)
cv2.createTrackbar('Play',   CTRL_WINDOW, DEFAULTS['Play'],   1,              lambda x: None)

# --- MAIN LOOP ---
zoom = 1.0
last_time = time.time()
current_src = static_img
eff_sw, eff_sh = static_w, static_h

while True:
    source  = cv2.getTrackbarPos('Source', CTRL_WINDOW)
    outer   = cv2.getTrackbarPos('Outer',  CTRL_WINDOW)
    inner   = max(1, cv2.getTrackbarPos('Inner', CTRL_WINDOW))
    fov     = max(1, cv2.getTrackbarPos('FOV',   CTRL_WINDOW))
    speed   = (cv2.getTrackbarPos('Speed', CTRL_WINDOW) - 100) / 100.0
    playing = cv2.getTrackbarPos('Play',   CTRL_WINDOW)

    dt = time.time() - last_time
    last_time = time.time()
    if playing:
        zoom *= np.exp(speed * dt)

    camera.set_index(source - 1)
    if camera.active:
        frame = camera.read_bgra()
        if frame is not None:
            current_src = cv2.flip(frame, 1) if mirror else frame
        eff_sw, eff_sh = DST_W, DST_H
    else:
        current_src = static_img
        eff_sw, eff_sh = static_w, static_h

    ratio = outer / inner
    mx1, my1 = get_droste_map(zoom,         (outer, inner, fov), eff_sw, eff_sh)
    mx2, my2 = get_droste_map(zoom / ratio, (outer, inner, fov), eff_sw, eff_sh)
    mx.eval(mx1, my1, mx2, my2)

    f_outer = cv2.remap(current_src, np.array(mx1), np.array(my1), cv2.INTER_LINEAR)
    f_inner = cv2.remap(current_src, np.array(mx2), np.array(my2), cv2.INTER_LINEAR)

    bgr_inner   = f_inner[:, :, :3]
    alpha_inner = f_inner[:, :, 3:] / 255.0
    bgr_outer   = f_outer[:, :, :3]

    combined = (bgr_inner * alpha_inner + bgr_outer * (1.0 - alpha_inner)).astype(np.uint8)

    if camera.active:
        src_label = f"Cam {source - 1}" + (" [M]" if mirror else "")
        cv2.putText(combined, src_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    display = letterbox(combined, SCREEN_W, SCREEN_H) if fullscreen else combined
    cv2.imshow(MAIN_WINDOW, display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        fullscreen = not fullscreen
        cv2.setWindowProperty(MAIN_WINDOW, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    elif key == ord(' '):
        cv2.setTrackbarPos('Play', CTRL_WINDOW, 1 - playing)
    elif key == ord('m'):
        mirror = not mirror

camera.release()
cv2.destroyAllWindows()
