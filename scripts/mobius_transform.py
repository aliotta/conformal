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
MAIN_WINDOW = "Mobius (Inversion + Vortex)"
CTRL_WINDOW = "Mobius Controls"

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

def get_mobius_map(cx, cy, scale, rotation_angle):
    zx = X_GRID - cx
    zy = Y_GRID - cy
    r_norm = 2.0 * mx.sqrt(zx**2 + zy**2) / (scale + 1e-8)
    vortex = rotation_angle / (r_norm + 1e-4)
    cos_v  = mx.cos(vortex)
    sin_v  = mx.sin(vortex)
    zx_v   = zx * cos_v - zy * sin_v
    zy_v   = zx * sin_v + zy * cos_v
    r2     = zx_v**2 + zy_v**2 + 1e-8
    half_s = scale / 2.0
    return cx + half_s**2 * zx_v / r2, cy + half_s**2 * (-zy_v) / r2

cams = probe_cameras()
print(f"Available cameras: {[(i, f'{w}x{h}') for i, w, h in cams]}")
print("Source trackbar: 0=static image, 1=camera 0, 2=camera 1, ...")
print("Keys: Space=play/pause, M=mirror camera, Q=quit")

camera = CameraSource(DST_W, DST_H)
mirror = False

cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_NORMAL)
cv2.namedWindow(CTRL_WINDOW, cv2.WINDOW_NORMAL)

cv2.createTrackbar('Source',  CTRL_WINDOW, 0,           3 + len(cams), lambda _: None)
cv2.createTrackbar('CenterX', CTRL_WINDOW, DST_W // 2,  DST_W,         lambda _: None)
cv2.createTrackbar('CenterY', CTRL_WINDOW, DST_H // 2,  DST_H,         lambda _: None)
cv2.createTrackbar('Scale',   CTRL_WINDOW, 300,          DST_W,         lambda _: None)
# RotX100: rotation angle ×100, offset 1570 → 0 = no vortex
cv2.createTrackbar('RotX100', CTRL_WINDOW, 1570,         3140,          lambda _: None)
cv2.createTrackbar('Speed',   CTRL_WINDOW, 100,          200,           lambda _: None)
cv2.createTrackbar('Play',    CTRL_WINDOW, 0,            1,             lambda _: None)

rotation_angle = 0.0
last_time = time.time()
current_src = static_img

while True:
    source  = cv2.getTrackbarPos('Source',  CTRL_WINDOW)
    cx      = cv2.getTrackbarPos('CenterX', CTRL_WINDOW)
    cy      = cv2.getTrackbarPos('CenterY', CTRL_WINDOW)
    scale   = max(1, cv2.getTrackbarPos('Scale',   CTRL_WINDOW))
    rot     = (cv2.getTrackbarPos('RotX100', CTRL_WINDOW) - 1570) / 100.0
    speed   = (cv2.getTrackbarPos('Speed',   CTRL_WINDOW) - 100) / 100.0
    playing = cv2.getTrackbarPos('Play',     CTRL_WINDOW)

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
        rotation_angle += speed * dt
    else:
        rotation_angle = rot

    map_x, map_y = get_mobius_map(cx, cy, scale, rotation_angle)
    mx.eval(map_x, map_y)

    result = cv2.remap(current_src, np.array(map_x), np.array(map_y),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    src_label = f"Cam {source - 1}" if camera.active else "Static"
    cv2.putText(result, src_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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
