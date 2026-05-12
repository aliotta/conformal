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
    """Helper to generate a single processed frame at a specific zoom level."""
    Y_GRID_MX, X_GRID_MX = mx.meshgrid(mx.arange(h, dtype=mx.float32), mx.arange(w, dtype=mx.float32), indexing='ij')
    dist_from_center = np.sqrt((np.array(X_GRID_MX) - w/2)**2 + (np.array(Y_GRID_MX) - h/2)**2)
    mask = np.clip(1.5 - (dist_from_center / (w / 2.5)), 0, 1)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)[:, :, np.newaxis]

    mx1, my1 = get_droste_map(zoom_val, h, w, Y_GRID_MX, X_GRID_MX)
    mx2, my2 = get_droste_map(zoom_val / ratio, h, w, Y_GRID_MX, X_GRID_MX)
    mx.eval(mx1, my1, mx2, my2) 
    
    f_low = cv2.remap(src_pixels, np.array(mx1), np.array(my1), cv2.INTER_LINEAR)
    f_high = cv2.remap(src_pixels, np.array(mx2), np.array(my2), cv2.INTER_LINEAR)
    return (f_high * mask + f_low * (1 - mask)).astype(np.uint8)

def save_static_png(filename, h, w):
    try:
        output_path = os.path.join(OUT_DIR, filename)
        print(f"--- Saving Static PNG: {filename} ---")
        # Generate the frame at the start of the loop (zoom = 1.0)
        frame = generate_single_frame(h, w, 1.0)
        Image.fromarray(frame).save(output_path)
        print(f"SUCCESS: Static frame saved to {output_path}")
    except Exception as e:
        print(f"FAILED Static PNG: {e}")

def bake_gif(filename, h, w, fps, subrects, colors=256):
    try:
        output_path = os.path.join(OUT_DIR, filename)
        print(f"\n--- Baking {filename} ({w}x{h} @ {fps}fps) ---")
        total_frames = int(loop_duration * fps)
        
        frames = []
        for i in range(total_frames):
            zoom = np.exp(SPEED * (i / fps))
            frame = generate_single_frame(h, w, zoom)
            frames.append(frame)
            if i % 10 == 0: print(f"Progress: {i}/{total_frames}")

        imageio.mimsave(output_path, frames, format='GIF', fps=fps, loop=0, 
                        subrectangles=subrects, palettesize=colors, plugin='pillow')
        print(f"SUCCESS: Saved to {output_path}")
    except Exception as e:
        print(f"FAILED {filename}: {e}")

if __name__ == '__main__':
    try:
        img = Image.open(INPUT_FILE).convert('RGB')
        src_pixels = np.array(img)
        
        # Output static frame first
        save_static_png('eye_spiral_static.png', h=1080, w=1080)
        
        # Then proceed with animated GIFs
        bake_gif('eye_spiral_reddit.gif', h=800, w=800, fps=30, subrects=False)
        bake_gif('eye_spiral_discord.gif', h=400, w=400, fps=15, subrects=True)
        bake_gif('eye_spiral_slack.gif', h=300, w=300, fps=10, subrects=True, colors=128)
    except FileNotFoundError:
        print(f"Error: Could not find {INPUT_FILE}")