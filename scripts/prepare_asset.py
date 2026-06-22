import cv2
import numpy as np
import json
import os
import argparse

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Prepare an image asset for Droste Effect.")
parser.add_argument('input', help="Path to the raw image file (e.g., assets/shrek.jpg)")
args = parser.parse_args()

PREPROCESSED_DIR = 'preprocessed'
if not os.path.exists(PREPROCESSED_DIR):
    os.makedirs(PREPROCESSED_DIR)

# --- CONFIG ---
INPUT_PATH = args.input # Now using the path passed via command line
file_base = os.path.splitext(os.path.basename(INPUT_PATH))[0]
OUTPUT_PATH = os.path.join(PREPROCESSED_DIR, f"{file_base}.png")
JSON_PATH = 'assets/transform_config.json'

def nothing(x): pass

# Global-ish state for the callback
state = {
    'pad': 0,
    'm': 0,
    'cx': 0,
    'cy': 0,
    'w': 0,
    'h': 0
}

def handle_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Calculate where the top-left corner of the CURRENT crop is in padded space
        top_left_x = state['cx'] - state['m']
        top_left_y = state['cy'] - state['m']
        
        # New center in padded space = top_left + mouse_click
        new_cx_padded = top_left_x + x
        new_cy_padded = top_left_y + y
        
        # Convert back to raw image space (subtract padding)
        new_raw_x = new_cx_padded - state['pad']
        new_raw_y = new_cy_padded - state['pad']
        
        # Constrain to trackbar limits to prevent crashes
        new_raw_x = max(0, min(state['w'], new_raw_x))
        new_raw_y = max(0, min(state['h'], new_raw_y))
        
        cv2.setTrackbarPos('Center X', 'Droste Pre-processor', int(new_raw_x))
        cv2.setTrackbarPos('Center Y', 'Droste Pre-processor', int(new_raw_y))

img = cv2.imread(INPUT_PATH, cv2.IMREAD_UNCHANGED)
if img is None:
    print(f"Error: {INPUT_PATH} not found."); exit()

if img.shape[2] == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

h, w = img.shape[:2]
pad = max(h, w) 
padded_img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0,0,0,0])

# Initialize state
state['w'], state['h'], state['pad'] = w, h, pad

cv2.namedWindow('Droste Pre-processor', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Center X', 'Droste Pre-processor', w//2, w, nothing)
cv2.createTrackbar('Center Y', 'Droste Pre-processor', h//2, h, nothing)
cv2.createTrackbar('Diameter', 'Droste Pre-processor', min(h, w)//2, min(h, w), nothing)
cv2.createTrackbar('Scale %', 'Droste Pre-processor', 15, 100, nothing)

cv2.setMouseCallback('Droste Pre-processor', handle_click)

print("\n--- CONTROLS ---")
print("CLICK: Set Center Point")
print("ARROWS: Fine-tune (1px) | SHIFT + ARROWS: Move 10px")
print("'S': Save and Purge | 'Q': Quit")

while True:
    cx_raw = cv2.getTrackbarPos('Center X', 'Droste Pre-processor')
    cy_raw = cv2.getTrackbarPos('Center Y', 'Droste Pre-processor')
    diam = max(10, cv2.getTrackbarPos('Diameter', 'Droste Pre-processor'))
    scale = cv2.getTrackbarPos('Scale %', 'Droste Pre-processor') / 100.0
    
    # Update global state for the click handler
    state['cx'], state['cy'] = cx_raw + pad, cy_raw + pad
    state['m'] = int(diam * 0.7) # Current zoom margin
    
    cx, cy, m = state['cx'], state['cy'], state['m']
    r = diam // 2
    
    display = padded_img.copy()
    cv2.circle(display, (cx, cy), r, (0, 255, 0), 4) 
    cv2.circle(display, (cx, cy), int(r * scale), (0, 0, 255), 2)
    
    # Final display view
    crop_view = display[cy-m:cy+m, cx-m:cx+m]
    cv2.imshow('Droste Pre-processor', crop_view)

    # Use waitKeyEx to catch Arrow Keys properly
    key = cv2.waitKeyEx(1)
    
    # Detect if Shift (or other modifiers) are held for 10px steps
    # Note: 0x10000 is a common flag for Shift in waitKeyEx
    step = 10 if (key & 0x10000) else 1

    # Standard Arrow Key Codes (Platform dependent, common codes included)
    if key in [63232, 2490368, 82]: # UP
        cv2.setTrackbarPos('Center Y', 'Droste Pre-processor', cy_raw - step)
    elif key in [63233, 2621440, 84]: # DOWN
        cv2.setTrackbarPos('Center Y', 'Droste Pre-processor', cy_raw + step)
    elif key in [63234, 2424832, 81]: # LEFT
        cv2.setTrackbarPos('Center X', 'Droste Pre-processor', cx_raw - step)
    elif key in [63235, 2555904, 83]: # RIGHT
        cv2.setTrackbarPos('Center X', 'Droste Pre-processor', cx_raw + step)

    # Standard ASCII keys
    elif (key & 0xFF) == ord('q'): 
        break
    elif (key & 0xFF) == ord('s'):
        # --- PURGE DATA FIX ---
        square_crop = np.zeros((diam, diam, 4), dtype=np.uint8)
        source_region = padded_img[cy-r:cy+r, cx-r:cx+r]
        
        # Blit source onto clean canvas
        sh, sw = source_region.shape[:2]
        square_crop[:sh, :sw] = source_region

        # Create mask
        mask = np.zeros((diam, diam), dtype=np.uint8)
        cv2.circle(mask, (diam // 2, diam // 2), r, 255, -1)
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        # Kill "Ghost" RGB pixels
        for i in range(3): 
            square_crop[:, :, i] = (square_crop[:, :, i] * (mask / 255.0)).astype(np.uint8)
        square_crop[:, :, 3] = mask 

        inner_dim = int(diam * scale)
        inner_img = cv2.resize(square_crop, (inner_dim, inner_dim), interpolation=cv2.INTER_LANCZOS4)
        
        final_canvas = square_crop.copy()
        offset = (diam - inner_dim) // 2
        
        alpha_s = inner_img[:, :, 3:] / 255.0
        src_rgb = inner_img[:, :, :3]
        roi = final_canvas[offset:offset+inner_dim, offset:offset+inner_dim, :3]
        
        final_canvas[offset:offset+inner_dim, offset:offset+inner_dim, :3] = \
            (src_rgb * alpha_s + roi * (1.0 - alpha_s)).astype(np.uint8)

        cv2.imwrite(OUTPUT_PATH, final_canvas)
        
        with open(JSON_PATH, 'w') as f:
            json.dump({
                "Outer": diam, "Inner": inner_dim,
                "FocX": diam / 2, "FocY": diam / 2,
                "Ratio": diam / inner_dim,
                "AssetInput": OUTPUT_PATH,
            }, f, indent=4)

        print(f"\nSaved with Purge: {OUTPUT_PATH}")
        break

cv2.destroyAllWindows()