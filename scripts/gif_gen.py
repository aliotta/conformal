# Copyright (c) 2026 Alexander Liotta
# Licensed under the MIT License

import os
import numpy as np
import cv2
import mlx.core as mx
from PIL import Image
import imageio

# --- DIRECTORY CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

ASSETS_DIR = os.path.join(BASE_DIR, "assets")
OUT_DIR = os.path.join(BASE_DIR, "out")

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# --- GLOBAL MATH CONFIG ---
INPUT_FILE = os.path.join(ASSETS_DIR, "centered_eye.png")
OUTER = 1280
INNER = 85
FOV = 100
FOC_X = 640 
FOC_Y = 640
SPEED = 0.5 

# --- OPTIONAL JSON CONFIG LOAD ---
CONFIG_FILE = os.path.join(ASSETS_DIR, "transform_config.json")
if os.path.exists(CONFIG_FILE):
    try:
        import json
        with open(CONFIG_FILE, 'r') as f:
            cfg = json.load(f)
            OUTER = cfg.get('Outer', OUTER)
            INNER = cfg.get('Inner', INNER)
            FOC_X = cfg.get('FocX', FOC_X)
            FOC_Y = cfg.get('FocY', FOC_Y)
            INPUT_FILE = cfg["AssetInput"]
            print(f"GIF Generator using auto-config from {CONFIG_FILE}")
    except Exception as e:
        print(f"GIF Generator using hardcoded defaults: {e}")

# Recalculate derived math
ratio = OUTER / INNER
log_ratio = np.log(ratio)
loop_duration = log_ratio / SPEED

def get_droste_map(zoom, grid_h, grid_w, Y_GRID, X_GRID):
    eff_zoom = np.exp(np.log(max(zoom, 1e-10)) % log_ratio)
    denom = log_ratio**2 + (2 * np.pi)**2
    a_real, a_imag = (2 * np.pi * 2 * np.pi) / denom, (2 * np.pi * log_ratio) / denom
    a_sq_norm = a_real**2 + a_imag**2
    inv_a_real, inv_a_imag = a_real / a_sq_norm, -a_imag / a_sq_norm

    view_scale = (OUTER / (grid_w * eff_zoom)) * (FOV / 100.0)
    zx, zy = (X_GRID - (grid_w / 2)) * view_scale, (Y_GRID - (grid_h / 2)) * view_scale
    z_mag, z_angle = mx.sqrt(zx**2 + zy**2) + 1e-9, mx.arctan2(zy, zx)
    log_z_mag = mx.log(z_mag)
    
    res_real = log_z_mag * inv_a_real - z_angle * inv_a_imag
    res_imag = log_z_mag * inv_a_imag + z_angle * inv_a_real
    
    angle_offset = np.log(eff_zoom) * (log_ratio / (2 * np.pi))
    mag_warped = mx.exp(res_real)
    shift = mx.remainder(res_real - np.log(INNER), log_ratio) + np.log(INNER)
    scaling_factor = mx.exp(shift) / mag_warped
    final_angle = res_imag - angle_offset
    
    return (mag_warped * mx.cos(final_angle) * scaling_factor + FOC_X), \
           (mag_warped * mx.sin(final_angle) * scaling_factor + FOC_Y)

def generate_single_frame(h, w, zoom_val):
    Y_GRID_MX, X_GRID_MX = mx.meshgrid(mx.arange(h, dtype=mx.float32), mx.arange(w, dtype=mx.float32), indexing='ij')

    mx1, my1 = get_droste_map(zoom_val, h, w, Y_GRID_MX, X_GRID_MX)
    mx2, my2 = get_droste_map(zoom_val / ratio, h, w, Y_GRID_MX, X_GRID_MX)
    mx.eval(mx1, my1, mx2, my2) 
    
    f_low = cv2.remap(src_pixels, np.array(mx1), np.array(my1), cv2.INTER_LINEAR)
    f_high = cv2.remap(src_pixels, np.array(mx2), np.array(my2), cv2.INTER_LINEAR)
    
    bgr_inner = f_high[:, :, :3]
    alpha_inner = f_high[:, :, 3:] / 255.0
    bgr_outer = f_low[:, :, :3]
    
    combined = (bgr_inner * alpha_inner + bgr_outer * (1.0 - alpha_inner)).astype(np.uint8)
    return cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

def save_static_png(filename, h, w):
    try:
        output_path = os.path.join(OUT_DIR, filename)
        print(f"\n--- Generating Static PNG: {filename} ---")
        frame_rgb = generate_single_frame(h, w, 1.0)
        Image.fromarray(frame_rgb).save(output_path)
        print(f"SUCCESS: Static frame saved to {output_path}")
    except Exception as e:
        print(f"FAILED Static PNG: {e}")

def bake_sequence(filename, h, w, fps, subrects=False, colors=256):
    try:
        output_path = os.path.join(OUT_DIR, filename)
        is_video = filename.endswith('.mp4')
        print(f"\n--- Baking {filename} ({w}x{h} @ {fps}fps) ---")
        
        total_frames = int(loop_duration * fps)
        frames = []
        
        writer = imageio.get_writer(output_path, fps=fps) if is_video else None
        
        for i in range(total_frames):
            zoom = np.exp(SPEED * (i / fps))
            frame_rgb = generate_single_frame(h, w, zoom)
            
            if is_video:
                writer.append_data(frame_rgb)
            else:
                frames.append(frame_rgb)
                
            if i % 10 == 0: print(f"Progress: {i}/{total_frames}")

        if is_video:
            writer.close()
        else:
            imageio.mimsave(output_path, frames, format='GIF', fps=fps, loop=0, 
                            subrectangles=subrects, palettesize=colors, plugin='pillow')
            
        print(f"SUCCESS: Saved to {output_path}")
    except Exception as e:
        print(f"FAILED {filename}: {e}")

if __name__ == '__main__':
    try:
        # Load with Alpha channel (BGRA)
        src_pixels = cv2.imread(INPUT_FILE, cv2.IMREAD_UNCHANGED)
        if src_pixels is None:
            raise FileNotFoundError(f"Could not find {INPUT_FILE}")
            
        if src_pixels.shape[2] == 3:
            src_pixels = cv2.cvtColor(src_pixels, cv2.COLOR_BGR2BGRA)

        base_name = os.path.splitext(os.path.basename(INPUT_FILE))[0]
        
        # 1. STATIC PNG
        save_static_png(f'{base_name}_static.png', h=1350, w=1080)
        
        # 2. iOS APP ICON (1024x1024, Opaque RGB)
        print("\n--- Generating iOS App Icon ---")
        icon_frame = generate_single_frame(1024, 1024, 1.0)
        icon_img = Image.fromarray(icon_frame).convert("RGB")
        icon_img.save(os.path.join(OUT_DIR, "AppIcon.png"))
        print(f"SUCCESS: iOS Icon saved to {os.path.join(OUT_DIR, 'AppIcon.png')}")
        
        # 3. REDDIT HIGH-RES MP4
        bake_sequence(f'{base_name}_reddit.mp4', h=1350, w=1080, fps=30)
        
        # 4. EXISTING GIFS
        bake_sequence(f'{base_name}_discord.gif', h=400, w=400, fps=15, subrects=True)
        bake_sequence(f'{base_name}_slack.gif', h=300, w=300, fps=10, subrects=True, colors=128)
        
    except Exception as e:
        print(f"Error: {e}")