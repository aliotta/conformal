import os
import time
import numpy as np
import cv2
import mlx.core as mx
from PIL import Image

# --- DIRECTORY CONFIG ---
# Automatically locates assets/hifi.png relative to this script's location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
INPUT_FILE = os.path.join(ASSETS_DIR, "centered_eye.png")

# --- DISPLAY CONFIG ---
# Set these to your projector's native resolution for the best fit
DST_H, DST_W = 800, 800 
MAIN_WINDOW = "Escher Projection"
CTRL_WINDOW = "Projector Controls"

DEFAULTS = {
    'Outer': 1280,
    'Inner': 85,
    'FOV': 100,
    'FocX': 640, # Optimized Focal Point
    'FocY': 640, # Optimized Focal Point
    'Speed': 150,
    'Play': 1
}

# Load High-Res Image
try:
    img = Image.open(INPUT_FILE).convert('RGB')
    src_pixels = np.array(img)
    src_h, src_w, _ = src_pixels.shape
except Exception as e:
    print(f"Error: {INPUT_FILE} not found. Please ensure it is in the /assets directory.")
    print(f"Details: {e}")
    exit()

# Pre-calculate grids
Y_GRID_MX, X_GRID_MX = mx.meshgrid(
    mx.arange(DST_H, dtype=mx.float32), 
    mx.arange(DST_W, dtype=mx.float32), 
    indexing='ij'
)

# Blending mask for seamless recursion
dist_from_center = np.sqrt((np.array(X_GRID_MX) - DST_W/2)**2 + (np.array(Y_GRID_MX) - DST_H/2)**2)
mask = np.clip(1.5 - (dist_from_center / (DST_W / 2.5)), 0, 1)
mask = cv2.GaussianBlur(mask, (51, 51), 0)[:, :, np.newaxis]

# --- UI WINDOWS ---
cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_NORMAL)
cv2.namedWindow(CTRL_WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(CTRL_WINDOW, 400, 750)

def reset_ui():
    for key, val in DEFAULTS.items():
        cv2.setTrackbarPos(key, CTRL_WINDOW, val)

cv2.createTrackbar('Outer', CTRL_WINDOW, DEFAULTS['Outer'], 2000, lambda x: None)
cv2.createTrackbar('Inner', CTRL_WINDOW, DEFAULTS['Inner'], 1000, lambda x: None)
cv2.createTrackbar('FOV', CTRL_WINDOW, DEFAULTS['FOV'], 1000, lambda x: None)
cv2.createTrackbar('FocX', CTRL_WINDOW, DEFAULTS['FocX'], src_w, lambda x: None)
cv2.createTrackbar('FocY', CTRL_WINDOW, DEFAULTS['FocY'], src_h, lambda x: None)
cv2.createTrackbar('Speed', CTRL_WINDOW, DEFAULTS['Speed'], 200, lambda x: None) 
cv2.createTrackbar('Play', CTRL_WINDOW, DEFAULTS['Play'], 1, lambda x: None)

def get_droste_map(zoom, params):
    outer_logical, inner_logical, fov_scale, focus_x, focus_y = params
    ratio = outer_logical / inner_logical
    log_ratio = np.log(ratio)
    effective_zoom = np.exp(np.log(max(zoom, 1e-10)) % log_ratio)

    denom = log_ratio**2 + (2 * np.pi)**2
    a_real = (2 * np.pi * 2 * np.pi) / denom
    a_imag = (2 * np.pi * log_ratio) / denom
    a_sq_norm = a_real**2 + a_imag**2
    inv_a_real, inv_a_imag = a_real / a_sq_norm, -a_imag / a_sq_norm

    view_scale = (outer_logical / (DST_W * effective_zoom)) * (fov_scale / 100.0)
    zx = (X_GRID_MX - (DST_W / 2)) * view_scale
    zy = (Y_GRID_MX - (DST_H / 2)) * view_scale
    
    z_mag = mx.sqrt(zx**2 + zy**2) + 1e-9
    z_angle = mx.arctan2(zy, zx)
    log_z_mag = mx.log(z_mag)
    
    res_real = log_z_mag * inv_a_real - z_angle * inv_a_imag
    res_imag = log_z_mag * inv_a_imag + z_angle * inv_a_real
    
    twist_factor = log_ratio / (2 * np.pi)
    angle_offset = np.log(effective_zoom) * twist_factor
    
    mag_warped = mx.exp(res_real)
    shift = mx.remainder(res_real - np.log(inner_logical), log_ratio) + np.log(inner_logical)
    scaling_factor = mx.exp(shift) / mag_warped
    final_angle = res_imag - angle_offset
    
    map_x = (mag_warped * mx.cos(final_angle) * scaling_factor + focus_x)
    map_y = (mag_warped * mx.sin(final_angle) * scaling_factor + focus_y)
    return map_x, map_y

# --- MAIN LOOP ---
zoom = 1.0
last_time = time.time()
is_fullscreen = False

while True:
    outer = cv2.getTrackbarPos('Outer', CTRL_WINDOW)
    inner = max(1, cv2.getTrackbarPos('Inner', CTRL_WINDOW))
    fov = max(1, cv2.getTrackbarPos('FOV', CTRL_WINDOW))
    focx = cv2.getTrackbarPos('FocX', CTRL_WINDOW)
    focy = cv2.getTrackbarPos('FocY', CTRL_WINDOW)
    speed = (cv2.getTrackbarPos('Speed', CTRL_WINDOW) - 100) / 100.0 
    playing = cv2.getTrackbarPos('Play', CTRL_WINDOW)

    dt = time.time() - last_time
    last_time = time.time()
    
    if playing:
        zoom *= np.exp(speed * dt)
    
    ratio = outer / inner
    mx1, my1 = get_droste_map(zoom, (outer, inner, fov, focx, focy))
    mx2, my2 = get_droste_map(zoom / ratio, (outer, inner, fov, focx, focy))
    mx.eval(mx1, my1, mx2, my2) 
    
    frame_low = cv2.remap(src_pixels, np.array(mx1), np.array(my1), cv2.INTER_LINEAR)
    frame_high = cv2.remap(src_pixels, np.array(mx2), np.array(my2), cv2.INTER_LINEAR)
    combined = (frame_high * mask + frame_low * (1 - mask)).astype(np.uint8)
    
    final_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    
    # --- PROJECTION CENTERING ---
    _, _, win_w, win_h = cv2.getWindowImageRect(MAIN_WINDOW)
    
    if is_fullscreen:
        # Pad with black bars to keep art centered on the projector
        v_pad = max(0, (win_h - DST_H) // 2)
        h_pad = max(0, (win_w - DST_W) // 2)
        final_bgr = cv2.copyMakeBorder(
            final_bgr, 
            v_pad, win_h - DST_H - v_pad, 
            h_pad, win_w - DST_W - h_pad, 
            cv2.BORDER_CONSTANT, value=[0,0,0]
        )

    cv2.imshow(MAIN_WINDOW, final_bgr)
    
    # --- CONTROL HUD ---
    status_bg = np.zeros((250, 400, 3), dtype=np.uint8)
    y_off = 30
    def draw_text(text, color=(255, 255, 255)):
        global y_off
        cv2.putText(status_bg, text, (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y_off += 25

    draw_text("PROJECTOR CONTROLS:", (0, 255, 255))
    draw_text("[F] Toggle Fullscreen (Projector)")
    draw_text("[R] Reset Animation")
    draw_text("[Space] Play / Pause")
    draw_text("[Q] Quit")
    draw_text("-" * 30, (100, 100, 100))
    draw_text(f"OUTPUT: {win_w}x{win_h}")
    draw_text(f"STATUS: {'PLAYING' if playing else 'PAUSED'}", (0, 255, 0) if playing else (0, 0, 255))

    cv2.imshow(CTRL_WINDOW, status_bg)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord(' '): cv2.setTrackbarPos('Play', CTRL_WINDOW, 1 - playing)
    elif key == ord('r'): reset_ui(); zoom = 1.0
    elif key == ord('f'):
        is_fullscreen = not is_fullscreen
        prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(MAIN_WINDOW, cv2.WND_PROP_FULLSCREEN, prop)

cv2.destroyAllWindows()