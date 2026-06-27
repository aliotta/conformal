"""
Unified Escher projector.
Set the Experience trackbar to switch between effects without restarting.

  0  Droste      — infinite spiral zoom
  1  Balcony     — magnification lens
  2  Fisheye     — chrome sphere reflection
  3  Mobius      — inversion + vortex
  4  MobiusTypes — elliptic / hyperbolic / loxodromic / parabolic

Keys: F = fullscreen toggle   Space = play/pause   M = mirror camera   O = open image   G = generate assets   Q = quit
"""

import os, sys, subprocess, time, threading
import numpy as np
import cv2
import mlx.core as mx
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_source import CameraSource, probe_cameras

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
INPUT_FILE = None

DST_H, DST_W  = 800, 800
MAIN_WINDOW   = "Escher"
CTRL_WINDOW   = "Controls"
EXPERIENCES   = ['Droste', 'Balcony', 'Fisheye', 'Mobius', 'MobiusTypes']

# ── config ───────────────────────────────────────────────────────────────────
DROSTE_DEFAULTS = {'Outer': 1280, 'Inner': 85, 'FOV': 100, 'Speed': 150, 'Play': 1}
CONFIG_FILE = os.path.join(ASSETS_DIR, "transform_config.json")
if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        DROSTE_DEFAULTS.update(cfg)
        INPUT_FILE = cfg.get("AssetInput", None)
        print(f"Loaded config: {cfg}")
    except Exception as e:
        print(f"Warning: could not parse config ({e})")

# ── static image ─────────────────────────────────────────────────────────────
def _make_placeholder():
    img = np.zeros((DST_H, DST_W, 4), dtype=np.uint8)
    cv2.putText(img, "Press O to open an image", (DST_W//2 - 160, DST_H//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180, 255), 2)
    return img

static_img = None
if INPUT_FILE and os.path.exists(INPUT_FILE):
    raw = cv2.imread(INPUT_FILE, cv2.IMREAD_UNCHANGED)
    if raw is not None:
        if raw.ndim == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGRA)
        elif raw.shape[2] == 3:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2BGRA)
        static_img = raw

if static_img is None:
    static_img = _make_placeholder()

static_h, static_w = static_img.shape[:2]
static_small = cv2.resize(static_img, (DST_W, DST_H))

# ── output grid (shared by all effects) ──────────────────────────────────────
Y_GRID, X_GRID = mx.meshgrid(
    mx.arange(DST_H, dtype=mx.float32),
    mx.arange(DST_W, dtype=mx.float32),
    indexing='ij'
)

# ── screen size (osascript avoids tkinter/OpenCV Tcl conflict) ───────────────
try:
    _out = subprocess.check_output(
        ['osascript', '-e', 'tell application "Finder" to return bounds of window of desktop'],
        text=True, stderr=subprocess.DEVNULL
    ).strip()
    _p = [int(x.strip()) for x in _out.split(',')]
    SCREEN_W, SCREEN_H = _p[2], _p[3]
except Exception:
    SCREEN_W, SCREEN_H = 1920, 1080
print(f"Screen: {SCREEN_W}x{SCREEN_H}")

# ── cameras ───────────────────────────────────────────────────────────────────
cams = probe_cameras()
print(f"Cameras: {[(i, name, f'{w}x{h}') for i,name,w,h in cams]}")
camera = CameraSource(DST_W, DST_H)

# ═══════════════════════════════════════════════════════════════════════════════
# Rendering functions
# ═══════════════════════════════════════════════════════════════════════════════

def get_droste_map(zoom, outer, inner, fov, eff_sw, eff_sh):
    ratio         = outer / inner
    log_ratio     = np.log(ratio)
    effective_zoom = np.exp(np.log(max(zoom, 1e-10)) % log_ratio)
    denom         = log_ratio**2 + (2 * np.pi)**2
    a_r = (2*np.pi)**2 / denom;  a_i = (2*np.pi * log_ratio) / denom
    inv_denom     = a_r**2 + a_i**2
    inv_a_r, inv_a_i = a_r / inv_denom, -a_i / inv_denom
    view_scale    = (outer / (DST_W * effective_zoom)) * (fov / 100.0)
    zx = (X_GRID - DST_W/2) * view_scale
    zy = (Y_GRID - DST_H/2) * view_scale
    z_mag   = mx.sqrt(zx**2 + zy**2) + 1e-9
    z_angle = mx.arctan2(zy, zx)
    res_r   = mx.log(z_mag) * inv_a_r - z_angle * inv_a_i
    res_i   = mx.log(z_mag) * inv_a_i + z_angle * inv_a_r
    mag_w   = mx.exp(res_r)
    shift   = mx.remainder(res_r - np.log(inner), log_ratio) + np.log(inner)
    sf      = mx.exp(shift) / mag_w
    fa      = res_i - np.log(effective_zoom) * (log_ratio / (2*np.pi))
    return (mag_w * mx.cos(fa) * sf + eff_sw/2,
            mag_w * mx.sin(fa) * sf + eff_sh/2)

def get_balcony_map(cx, cy, radius, magnification):
    ox = X_GRID - cx;  oy = Y_GRID - cy
    r  = mx.sqrt(ox**2 + oy**2)
    t  = mx.minimum(r / (radius + 1e-9), 1.0)
    k  = (1.0 - t*t)**2
    s  = 1.0 + (magnification - 1.0) * k
    return cx + ox/s, cy + oy/s

MAX_FISHEYE_R = DST_H * 0.23
def get_fisheye_map(cx, cy, radius, depth):
    ox = X_GRID - cx;  oy = Y_GRID - cy
    r  = mx.sqrt(ox**2 + oy**2)
    nr = r / (radius + 1e-9)
    th = mx.arcsin(mx.minimum(nr, 0.9999))
    sr = mx.tanh(mx.tan(mx.minimum(th * depth, float(np.pi)*0.48)) * radius / MAX_FISHEYE_R) * MAX_FISHEYE_R
    dx = mx.where(r > 1e-6, ox/r, mx.ones_like(ox))
    dy = mx.where(r > 1e-6, oy/r, mx.zeros_like(oy))
    return cx + dx*sr, cy + dy*sr, nr

def get_mobius_map(cx, cy, scale, rot):
    zx = X_GRID - cx;  zy = Y_GRID - cy
    rn = 2.0 * mx.sqrt(zx**2 + zy**2) / (scale + 1e-8)
    v  = rot / (rn + 1e-4)
    cv_, sv_ = mx.cos(v), mx.sin(v)
    zxv = zx*cv_ - zy*sv_;  zyv = zx*sv_ + zy*cv_
    r2  = zxv**2 + zyv**2 + 1e-8
    hs  = scale / 2.0
    return cx + hs**2 * zxv/r2, cy + hs**2 * (-zyv)/r2

def _cmul_r(ax,ay,bx,by): return ax*bx - ay*by
def _cmul_i(ax,ay,bx,by): return ax*by + ay*bx
def _cdiv_r(ax,ay,bx,by): return (ax*bx + ay*by) / (bx**2+by**2+1e-8)
def _cdiv_i(ax,ay,bx,by): return (ay*bx - ax*by) / (bx**2+by**2+1e-8)

def get_mobius_types_map(cx, cy, scale, pa, pb, ttype):
    hs  = scale / 2.0
    znx = (X_GRID - cx) / (hs + 1e-8)
    zny = (Y_GRID - cy) / (hs + 1e-8)
    if ttype < 3:
        if   ttype == 0: lx, ly = float(np.cos(-pb)),        float(np.sin(-pb))
        elif ttype == 1: lx, ly = float(pa),                  0.0
        else:            lx, ly = float(pa*np.cos(-pb)),      float(pa*np.sin(-pb))
        ox2, oy2 = 1.0-lx, -ly
        nx = _cmul_r(znx,zny,ox2,oy2) + 2.0
        ny = _cmul_i(znx,zny,ox2,oy2)
        dx = _cmul_r(2*lx,2*ly,znx,zny)
        dy = _cmul_i(2*lx,2*ly,znx,zny)
        wx, wy = _cdiv_r(nx,ny,dx,dy), _cdiv_i(nx,ny,dx,dy)
    else:
        tx, ty = float(pa), float(pb)
        nx = 1.0 - _cmul_r(tx,ty,znx,zny)
        ny =     - _cmul_i(tx,ty,znx,zny)
        wx, wy = _cdiv_r(nx,ny,znx,zny), _cdiv_i(nx,ny,znx,zny)
    return cx + wx*hs, cy + wy*hs

# ═══════════════════════════════════════════════════════════════════════════════
# Display helpers
# ═══════════════════════════════════════════════════════════════════════════════

def letterbox(img, tw, th):
    h, w = img.shape[:2]
    s = min(tw/w, th/h)
    sw, sh = int(w*s), int(h*s)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    y, x = (th-sh)//2, (tw-sw)//2
    canvas[y:y+sh, x:x+sw] = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    return canvas

# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic control panel
# ═══════════════════════════════════════════════════════════════════════════════

def setup_controls(exp_idx):
    """Destroy and recreate the control window with trackbars for exp_idx."""
    cv2.destroyWindow(CTRL_WINDOW)
    cv2.namedWindow(CTRL_WINDOW, cv2.WINDOW_NORMAL)

    n = lambda _: None
    cv2.createTrackbar('Experience', CTRL_WINDOW, exp_idx, len(EXPERIENCES)-1, n)
    cv2.createTrackbar('Source',     CTRL_WINDOW, 0, max(1, len(cams)),          n)

    if exp_idx == 0:   # ── Droste ──────────────────────────────────────────
        cv2.createTrackbar('Outer',  CTRL_WINDOW, DROSTE_DEFAULTS['Outer'], 2000, n)
        cv2.createTrackbar('Inner',  CTRL_WINDOW, DROSTE_DEFAULTS['Inner'], 1000, n)
        cv2.createTrackbar('FOV',    CTRL_WINDOW, DROSTE_DEFAULTS['FOV'],   1000, n)
        cv2.createTrackbar('Speed',  CTRL_WINDOW, DROSTE_DEFAULTS['Speed'],  200, n)
        cv2.createTrackbar('Play',   CTRL_WINDOW, DROSTE_DEFAULTS['Play'],     1, n)

    elif exp_idx == 1: # ── Balcony ─────────────────────────────────────────
        cv2.createTrackbar('CenterX', CTRL_WINDOW, DST_W//2, DST_W, n)
        cv2.createTrackbar('CenterY', CTRL_WINDOW, DST_H//2, DST_H, n)
        cv2.createTrackbar('Radius',  CTRL_WINDOW, 200,      DST_W, n)
        cv2.createTrackbar('MagX10',  CTRL_WINDOW, 20,       100,   n)

    elif exp_idx == 2: # ── Fisheye ─────────────────────────────────────────
        cv2.createTrackbar('CenterX',   CTRL_WINDOW, DST_W//2, DST_W, n)
        cv2.createTrackbar('CenterY',   CTRL_WINDOW, DST_H//2, DST_H, n)
        cv2.createTrackbar('Radius',    CTRL_WINDOW, 300,      DST_W, n)
        cv2.createTrackbar('DepthX10',  CTRL_WINDOW, 7,        30,    n)

    elif exp_idx == 3: # ── Mobius ──────────────────────────────────────────
        cv2.createTrackbar('CenterX', CTRL_WINDOW, DST_W//2, DST_W, n)
        cv2.createTrackbar('CenterY', CTRL_WINDOW, DST_H//2, DST_H, n)
        cv2.createTrackbar('Scale',   CTRL_WINDOW, 300,      DST_W, n)
        cv2.createTrackbar('RotX100', CTRL_WINDOW, 1570,     3140,  n)
        cv2.createTrackbar('Speed',   CTRL_WINDOW, 100,      200,   n)
        cv2.createTrackbar('Play',    CTRL_WINDOW, 0,        1,     n)

    elif exp_idx == 4: # ── MobiusTypes ─────────────────────────────────────
        cv2.createTrackbar('CenterX',    CTRL_WINDOW, DST_W//2, DST_W, n)
        cv2.createTrackbar('CenterY',    CTRL_WINDOW, DST_H//2, DST_H, n)
        cv2.createTrackbar('Scale',      CTRL_WINDOW, 300,      DST_W, n)
        cv2.createTrackbar('Type 0-3',   CTRL_WINDOW, 0,        3,     n)
        cv2.createTrackbar('ParamAx100', CTRL_WINDOW, 100,      500,   n)
        cv2.createTrackbar('ParamBx100', CTRL_WINDOW, 314,      628,   n)
        cv2.createTrackbar('Speed',      CTRL_WINDOW, 100,      200,   n)
        cv2.createTrackbar('Play',       CTRL_WINDOW, 0,        1,     n)

def tb(name, fallback=0):
    """Safe trackbar read — returns fallback if window is being rebuilt."""
    try:
        return cv2.getTrackbarPos(name, CTRL_WINDOW)
    except Exception:
        return fallback

# ═══════════════════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════════════════

def _run_gif_gen(img, outer, inner, fov, src_stem="_gen_source"):
    global gif_gen_status
    try:
        os.makedirs(ASSETS_DIR, exist_ok=True)
        src_path = os.path.join(ASSETS_DIR, f"{src_stem}.png")
        cv2.imwrite(src_path, img)
        cfg = {"Outer": outer, "Inner": inner, "FOV": fov,
               "FocX": img.shape[1] / 2, "FocY": img.shape[0] / 2, "AssetInput": src_path}
        with open(CONFIG_FILE, 'w') as f:
            json.dump(cfg, f, indent=4)
        print("gif_gen: starting…")
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gif_gen.py")
        subprocess.run([sys.executable, script], check=True)
        gif_gen_status = "Done"
        print("gif_gen: complete")
    except Exception as e:
        gif_gen_status = f"Error: {e}"
        print(f"gif_gen error: {e}")

gif_gen_running = False
gif_gen_status  = ""
opened_file_stem = "_gen_source"

cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_NORMAL)
setup_controls(0)

current_exp   = 0
mirror        = False
fullscreen    = False
zoom          = 1.0
rotation_angle = 0.0
param_b_anim  = 0.0
last_time     = time.time()
current_src   = static_img   # updated each frame

MOBIUS_TYPE_NAMES = ['Elliptic', 'Hyperbolic', 'Loxodromic', 'Parabolic']

while True:
    dt = time.time() - last_time
    last_time = time.time()

    # ── experience switch ────────────────────────────────────────────────────
    exp = tb('Experience', current_exp)
    if exp != current_exp:
        setup_controls(exp)
        current_exp = exp
        zoom = 1.0;  rotation_angle = 0.0;  param_b_anim = 0.0

    # ── camera / source ──────────────────────────────────────────────────────
    source = tb('Source', 0)
    camera.set_index(source - 1)
    if camera.active:
        frame = camera.read_bgra()
        if frame is not None:
            current_src = cv2.flip(frame, 1) if mirror else frame
        cam_src = current_src          # DST×DST BGRA
        eff_sw = eff_sh = DST_W
    else:
        cam_src   = static_small       # DST×DST for non-Droste
        eff_sw, eff_sh = static_w, static_h

    # ── render ───────────────────────────────────────────────────────────────
    result = None

    if current_exp == 0:               # ── Droste ──────────────────────────
        outer   = max(2, tb('Outer', DROSTE_DEFAULTS['Outer']))
        inner   = max(1, tb('Inner', DROSTE_DEFAULTS['Inner']))
        fov     = max(1, tb('FOV',   DROSTE_DEFAULTS['FOV']))
        speed   = (tb('Speed', DROSTE_DEFAULTS['Speed']) - 100) / 100.0
        playing = tb('Play', DROSTE_DEFAULTS['Play'])

        if playing:
            zoom *= np.exp(speed * dt)

        src  = cam_src if camera.active else static_img
        ratio = outer / inner
        mx1, my1 = get_droste_map(zoom,         outer, inner, fov, eff_sw, eff_sh)
        mx2, my2 = get_droste_map(zoom / ratio, outer, inner, fov, eff_sw, eff_sh)
        mx.eval(mx1, my1, mx2, my2)
        f_out = cv2.remap(src, np.array(mx1), np.array(my1), cv2.INTER_LINEAR)
        f_in  = cv2.remap(src, np.array(mx2), np.array(my2), cv2.INTER_LINEAR)
        alpha = f_in[:, :, 3:] / 255.0
        result = (f_in[:,:,:3]*alpha + f_out[:,:,:3]*(1-alpha)).astype(np.uint8)

    elif current_exp == 1:             # ── Balcony ─────────────────────────
        cx  = tb('CenterX', DST_W//2)
        cy  = tb('CenterY', DST_H//2)
        rad = max(1, tb('Radius', 200))
        mag = max(0.1, tb('MagX10', 20) / 10.0)
        mx1, my1 = get_balcony_map(cx, cy, rad, mag)
        mx.eval(mx1, my1)
        result = cv2.remap(cam_src, np.array(mx1), np.array(my1),
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[:,:,:3]

    elif current_exp == 2:             # ── Fisheye ─────────────────────────
        cx     = tb('CenterX',  DST_W//2)
        cy     = tb('CenterY',  DST_H//2)
        radius = max(1, tb('Radius', 300))
        depth  = max(0.1, tb('DepthX10', 7) / 10.0)
        mx1, my1, nr_mx = get_fisheye_map(cx, cy, radius, depth)
        mx.eval(mx1, my1, nr_mx)
        out = cv2.remap(cam_src, np.array(mx1), np.array(my1),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        nr  = np.array(nr_mx)
        out[nr >= 1.0] = [0, 0, 0, 255]
        lm  = nr > 0.7
        t_l = ((nr[lm] - 0.7) / 0.3)
        out[lm, :3] = (out[lm, :3] * (1.0 - t_l**1.5)[:, None]).astype(np.uint8)
        result = out[:,:,:3]

    elif current_exp == 3:             # ── Mobius ──────────────────────────
        cx      = tb('CenterX', DST_W//2)
        cy      = tb('CenterY', DST_H//2)
        scale   = max(1, tb('Scale', 300))
        rot_tb  = (tb('RotX100', 1570) - 1570) / 100.0
        speed   = (tb('Speed', 100) - 100) / 100.0
        playing = tb('Play', 0)
        if playing:
            rotation_angle += speed * dt
        else:
            rotation_angle = rot_tb
        mx1, my1 = get_mobius_map(cx, cy, scale, rotation_angle)
        mx.eval(mx1, my1)
        result = cv2.remap(cam_src, np.array(mx1), np.array(my1),
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[:,:,:3]

    elif current_exp == 4:             # ── MobiusTypes ─────────────────────
        cx      = tb('CenterX',    DST_W//2)
        cy      = tb('CenterY',    DST_H//2)
        scale   = max(1, tb('Scale', 300))
        ttype   = tb('Type 0-3', 0)
        pa      = tb('ParamAx100', 100) / 100.0
        pb_tb   = (tb('ParamBx100', 314) - 314) / 100.0
        speed   = (tb('Speed', 100) - 100) / 100.0
        playing = tb('Play', 0)
        if playing:
            param_b_anim += speed * dt
            pb = param_b_anim
        else:
            param_b_anim = pb_tb
            pb = pb_tb
        mx1, my1 = get_mobius_types_map(cx, cy, scale, pa, pb, ttype)
        mx.eval(mx1, my1)
        out = cv2.remap(cam_src, np.array(mx1), np.array(my1),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        label = f"{MOBIUS_TYPE_NAMES[ttype]}  A={pa:.2f}  B={pb:.2f}"
        cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        result = out[:,:,:3]

    # ── overlay experience name (top-right) ──────────────────────────────────
    if result is not None:
        result = np.ascontiguousarray(result)
        exp_label = EXPERIENCES[current_exp]
        if camera.active:
            exp_label += f"  Cam {source-1}" + ("  [M]" if mirror else "")
        cv2.putText(result, exp_label, (DST_W - 220, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        if gif_gen_status:
            cv2.putText(result, gif_gen_status, (10, DST_H - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
            if gif_gen_status in ("Done",) or gif_gen_status.startswith("Error"):
                gif_gen_running = False

        display = letterbox(result, SCREEN_W, SCREEN_H) if fullscreen else result
        cv2.imshow(MAIN_WINDOW, display)

    # ── keys ─────────────────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        fullscreen = not fullscreen
        cv2.setWindowProperty(MAIN_WINDOW, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    elif key == ord(' '):
        playing = tb('Play', 0)
        try:
            cv2.setTrackbarPos('Play', CTRL_WINDOW, 1 - playing)
        except Exception:
            pass
    elif key == ord('m'):
        mirror = not mirror
    elif key == ord('g'):
        if not gif_gen_running:
            outer = max(2, tb('Outer', DROSTE_DEFAULTS['Outer']))
            inner = max(1, tb('Inner', DROSTE_DEFAULTS['Inner']))
            fov   = max(1, tb('FOV',   DROSTE_DEFAULTS['FOV']))
            gif_gen_running = True
            gif_gen_status  = "Generating…"
            t = threading.Thread(target=_run_gif_gen,
                                 args=(static_img.copy(), outer, inner, fov, opened_file_stem), daemon=True)
            t.start()
    elif key == ord('o'):
        try:
            result = subprocess.run(
                ['osascript', '-e',
                 'POSIX path of (choose file with prompt "Choose Image"'
                 ' of type {"public.image", "com.adobe.pdf"})'],
                capture_output=True, text=True
            )
            path = result.stdout.strip() if result.returncode == 0 else None
            if path:
                opened_file_stem = os.path.splitext(os.path.basename(path))[0]
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if img.ndim == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                    elif img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    static_img   = img
                    static_h, static_w = img.shape[:2]
                    static_small = cv2.resize(img, (DST_W, DST_H))
                    zoom = 1.0
        except Exception as e:
            print(f"File dialog error: {e}")

camera.release()
cv2.destroyAllWindows()
