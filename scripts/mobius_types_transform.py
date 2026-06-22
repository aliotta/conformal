"""
Mobius Types — four classical Möbius transformation families.

Transform types (trackbar 0-3):
  0  Elliptic   — pure rotation on the Riemann sphere.    ParamB = rotation angle.
  1  Hyperbolic — pure real scaling.                       ParamA = scale factor.
  2  Loxodromic — rotation + scaling combined.             ParamA = magnitude, ParamB = angle.
  3  Parabolic  — complex translation (two fixed pts merge). ParamA = Re(t), ParamB = Im(t).

The map is  w = (zn·(1-λ) + 2) / (2λ·zn)  (types 0-2),  or  w = (1 - t·zn) / zn  (type 3),
where  zn = (pixel - center) / (scale/2).  Result is scaled back by scale/2.
"""

import os
import sys
import time
import numpy as np
import cv2
import mlx.core as mx
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_source import CameraSource, probe_cameras

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
INPUT_FILE = os.path.join(ASSETS_DIR, "raw_photo.jpg")

DST_H, DST_W = 800, 800
MAIN_WINDOW = "Mobius Types"
CTRL_WINDOW = "Mobius Types Controls"

CONFIG_FILE = os.path.join(ASSETS_DIR, "transform_config.json")
if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, 'r') as f:
            cfg = json.load(f)
            if "AssetInput" in cfg:
                INPUT_FILE = cfg["AssetInput"]
    except Exception:
        pass

try:
    static_img = cv2.imread(INPUT_FILE, cv2.IMREAD_UNCHANGED)
    if static_img is None:
        raise FileNotFoundError(f"Could not find {INPUT_FILE}")
    if static_img.shape[2] == 3:
        static_img = cv2.cvtColor(static_img, cv2.COLOR_BGR2BGRA)
    static_img = cv2.resize(static_img, (DST_W, DST_H))
except Exception as e:
    print(f"Error: {e}"); exit()

Y_GRID, X_GRID = mx.meshgrid(
    mx.arange(DST_H, dtype=mx.float32),
    mx.arange(DST_W, dtype=mx.float32),
    indexing='ij'
)

def cmul_r(ax, ay, bx, by): return ax * bx - ay * by
def cmul_i(ax, ay, bx, by): return ax * by + ay * bx
def cdiv_r(ax, ay, bx, by): return (ax*bx + ay*by) / (bx**2 + by**2 + 1e-8)
def cdiv_i(ax, ay, bx, by): return (ay*bx - ax*by) / (bx**2 + by**2 + 1e-8)


def get_mobius_types_map(cx, cy, scale, param_a, param_b, transform_type):
    half_s = scale / 2.0
    zn_x = (X_GRID - cx) / (half_s + 1e-8)
    zn_y = (Y_GRID - cy) / (half_s + 1e-8)

    if transform_type < 3:
        if transform_type == 0:        # Elliptic: λ = e^{-i·paramB}
            li_x = float(np.cos(-param_b));  li_y = float(np.sin(-param_b))
        elif transform_type == 1:      # Hyperbolic: λ = paramA (real)
            li_x = float(param_a);           li_y = 0.0
        else:                          # Loxodromic: λ = paramA · e^{-i·paramB}
            li_x = float(param_a * np.cos(-param_b))
            li_y = float(param_a * np.sin(-param_b))

        oml_x, oml_y = 1.0 - li_x, -li_y

        num_x = cmul_r(zn_x, zn_y, oml_x, oml_y) + 2.0
        num_y = cmul_i(zn_x, zn_y, oml_x, oml_y)
        den_x = cmul_r(2.0 * li_x, 2.0 * li_y, zn_x, zn_y)
        den_y = cmul_i(2.0 * li_x, 2.0 * li_y, zn_x, zn_y)
        wx = cdiv_r(num_x, num_y, den_x, den_y)
        wy = cdiv_i(num_x, num_y, den_x, den_y)

    else:                              # Parabolic: t = (paramA, paramB)
        tx, ty = float(param_a), float(param_b)
        num_x = 1.0 - cmul_r(tx, ty, zn_x, zn_y)
        num_y = 0.0 - cmul_i(tx, ty, zn_x, zn_y)
        wx = cdiv_r(num_x, num_y, zn_x, zn_y)
        wy = cdiv_i(num_x, num_y, zn_x, zn_y)

    return cx + wx * half_s, cy + wy * half_s


cams = probe_cameras()
print(f"Available cameras: {[(i, f'{w}x{h}') for i, w, h in cams]}")
print("Source trackbar: 0=static image, 1=camera 0, 2=camera 1, ...")
print("Keys: Space=play/pause, M=mirror camera, Q=quit")

camera = CameraSource(DST_W, DST_H)
mirror = False

cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_NORMAL)
cv2.namedWindow(CTRL_WINDOW, cv2.WINDOW_NORMAL)

cv2.createTrackbar('Source',     CTRL_WINDOW, 0,           3 + len(cams), lambda _: None)
cv2.createTrackbar('CenterX',    CTRL_WINDOW, DST_W // 2,  DST_W,         lambda _: None)
cv2.createTrackbar('CenterY',    CTRL_WINDOW, DST_H // 2,  DST_H,         lambda _: None)
cv2.createTrackbar('Scale',      CTRL_WINDOW, 300,          DST_W,         lambda _: None)
cv2.createTrackbar('Type 0-3',   CTRL_WINDOW, 0,            3,             lambda _: None)
# ParamA: 0-500 → 0.0-5.0 (÷100). Default 100 = 1.0
cv2.createTrackbar('ParamAx100', CTRL_WINDOW, 100,          500,           lambda _: None)
# ParamB: 0-628 → -π to +π. Default 314 = 0.0
cv2.createTrackbar('ParamBx100', CTRL_WINDOW, 314,          628,           lambda _: None)
cv2.createTrackbar('Speed',      CTRL_WINDOW, 100,          200,           lambda _: None)
cv2.createTrackbar('Play',       CTRL_WINDOW, 0,            1,             lambda _: None)

TYPE_NAMES = ['Elliptic', 'Hyperbolic', 'Loxodromic', 'Parabolic']

param_b_anim = 0.0
last_time = time.time()
current_src = static_img

while True:
    source  = cv2.getTrackbarPos('Source',     CTRL_WINDOW)
    cx      = cv2.getTrackbarPos('CenterX',    CTRL_WINDOW)
    cy      = cv2.getTrackbarPos('CenterY',    CTRL_WINDOW)
    scale   = max(1, cv2.getTrackbarPos('Scale',      CTRL_WINDOW))
    t_type  = cv2.getTrackbarPos('Type 0-3',   CTRL_WINDOW)
    param_a = cv2.getTrackbarPos('ParamAx100', CTRL_WINDOW) / 100.0
    param_b = (cv2.getTrackbarPos('ParamBx100', CTRL_WINDOW) - 314) / 100.0
    speed   = (cv2.getTrackbarPos('Speed',     CTRL_WINDOW) - 100) / 100.0
    playing = cv2.getTrackbarPos('Play',       CTRL_WINDOW)

    dt = time.time() - last_time
    last_time = time.time()

    camera.set_index(source - 1)
    if camera.active:
        frame = camera.read_bgra()
        if frame is not None:
            current_src = cv2.flip(frame, 1) if mirror else frame
    else:
        current_src = static_img

    if playing:
        param_b_anim += speed * dt
        effective_b = param_b_anim
    else:
        param_b_anim = param_b
        effective_b  = param_b

    map_x, map_y = get_mobius_types_map(cx, cy, scale, param_a, effective_b, t_type)
    mx.eval(map_x, map_y)

    result = cv2.remap(current_src, np.array(map_x), np.array(map_y),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    src_label = f"Cam {source - 1}" if camera.active else "Static"
    label = f"{TYPE_NAMES[t_type]}  A={param_a:.2f}  B={effective_b:.2f}  [{src_label}]"
    cv2.putText(result, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow(MAIN_WINDOW, result[:, :, :3])

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        is_fs = cv2.getWindowProperty(MAIN_WINDOW, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(MAIN_WINDOW, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if is_fs == 0 else cv2.WINDOW_NORMAL)
    elif key == ord(' '):
        cv2.setTrackbarPos('Play', CTRL_WINDOW, 1 - playing)
    elif key == ord('m'):
        mirror = not mirror

camera.release()
cv2.destroyAllWindows()
