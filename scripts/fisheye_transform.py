import os
import sys
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
MAIN_WINDOW = "Fisheye (Chrome Sphere)"
CTRL_WINDOW = "Fisheye Controls"

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

# Fixed to 0.23 UV units (matching iOS shader). Keeps samples inside the image
# for portrait-aspect sources; for square 800×800 this is 184 px from center.
MAX_SAMPLE_R = DST_H * 0.23

def get_fisheye_map(cx, cy, radius, depth):
    ox = X_GRID - cx
    oy = Y_GRID - cy
    r  = mx.sqrt(ox**2 + oy**2)
    norm_r = r / (radius + 1e-9)
    theta    = mx.arcsin(mx.minimum(norm_r, 0.9999))
    raw_r    = mx.tan(mx.minimum(theta * depth, float(np.pi) * 0.48)) * radius
    sample_r = mx.tanh(raw_r / MAX_SAMPLE_R) * MAX_SAMPLE_R
    dir_x = mx.where(r > 1e-6, ox / r, mx.ones_like(ox))
    dir_y = mx.where(r > 1e-6, oy / r, mx.zeros_like(oy))
    return cx + dir_x * sample_r, cy + dir_y * sample_r, norm_r

cams = probe_cameras()
print(f"Available cameras: {[(i, f'{w}x{h}') for i, w, h in cams]}")
print("Source trackbar: 0=static image, 1=camera 0, 2=camera 1, ...")
print("Keys: M=mirror camera, Q=quit")

camera = CameraSource(DST_W, DST_H)
mirror = False

cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_NORMAL)
cv2.namedWindow(CTRL_WINDOW, cv2.WINDOW_NORMAL)

cv2.createTrackbar('Source',   CTRL_WINDOW, 0,           3 + len(cams), lambda _: None)
cv2.createTrackbar('CenterX',  CTRL_WINDOW, DST_W // 2,  DST_W,         lambda _: None)
cv2.createTrackbar('CenterY',  CTRL_WINDOW, DST_H // 2,  DST_H,         lambda _: None)
cv2.createTrackbar('Radius',   CTRL_WINDOW, 300,          DST_W,         lambda _: None)
# iOS hardcodes reflectionDepth=0.7 → default trackbar 7
cv2.createTrackbar('DepthX10', CTRL_WINDOW, 7,            30,            lambda _: None)

current_src = static_img

while True:
    source = cv2.getTrackbarPos('Source',   CTRL_WINDOW)
    cx     = cv2.getTrackbarPos('CenterX',  CTRL_WINDOW)
    cy     = cv2.getTrackbarPos('CenterY',  CTRL_WINDOW)
    radius = max(1, cv2.getTrackbarPos('Radius',    CTRL_WINDOW))
    depth  = max(0.1, cv2.getTrackbarPos('DepthX10', CTRL_WINDOW) / 10.0)

    camera.set_index(source - 1)
    if camera.active:
        frame = camera.read_bgra()
        if frame is not None:
            current_src = cv2.flip(frame, 1) if mirror else frame
    else:
        current_src = static_img

    map_x, map_y, norm_r_mx = get_fisheye_map(cx, cy, radius, depth)
    mx.eval(map_x, map_y, norm_r_mx)

    result = cv2.remap(current_src, np.array(map_x), np.array(map_y),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    norm_r = np.array(norm_r_mx)

    result[norm_r >= 1.0] = [0, 0, 0, 255]

    limb_mask = norm_r > 0.7
    t_limb = (norm_r[limb_mask] - 0.7) / 0.3
    darkening = (1.0 - np.power(t_limb, 1.5))[:, np.newaxis]
    result[limb_mask, :3] = (result[limb_mask, :3] * darkening).astype(np.uint8)

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
    elif key == ord('m'):
        mirror = not mirror

camera.release()
cv2.destroyAllWindows()
