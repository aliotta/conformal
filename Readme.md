# Conformal Droste Animation Engine

A Python-based implementation of the Droste Effect (recursive image-in-image) using conformal mapping, optimized with Apple's **MLX** framework for GPU-accelerated real-time performance.

![Droste Spiral Example](out/eye_spiral_static.png)

## Workflow

### 1. Asset Preparation (`prepare_asset.py`)
Run this script to prepare your source image. This GUI tool automates the geometry and masking needed for a perfect loop.

- **Interaction:**
  - **Click:** Set the focal point (center of the spiral) directly on the image.
  - **Arrow Keys:** Fine-tune the focal point 1 pixel at a time.
  - **Shift + Arrows:** Move the focal point 10 pixels at a time.
  - **Sliders:** Adjust the `Diameter` of the circular mask and the `Scale %` (the size of the inner recursion).
- **Auto-Sync:** Pressing **'S'** saves a processed `centered_eye.png` and a `transform_config.json`. These files ensure that the transformation math in the projection and generation scripts perfectly aligns with your chosen focal point.

### 2. Live Projection (`mlx_transform.py`)
Launch this to explore the Droste spiral in real-time. It automatically detects your configuration from the assets folder.

- **Controls:**
  - `F`: Toggle Fullscreen.
  - `Space`: Play/Pause the zoom animation.
  - `R`: Reset to default parameters.
  - `Q`: Quit.

### 3. Multi-Platform Exporter (`gif_gen.py`)
Generates high-quality, perfectly looping animations optimized for specific platforms:
- **Reddit:** High-quality MP4 (800x800, 30fps).
- **Discord:** Optimized GIF under 10MB.
- **Slack:** Ultra-light GIF under 2MB for autoplay.

## Project Structure
- `/assets`: Put your raw image here (e.g., `shrek.jpg`). After preparation, this folder will contain `centered_eye.png` and `transform_config.json`.
- `/scripts`: Execution logic (`prepare_asset.py`, `mlx_transform.py`, `gif_gen.py`).
- `/out`: Generated GIFs and videos.

## Inspiration & Attribution
- **3Blue1Brown:** [Visual explanation of conformal mapping](https://www.youtube.com/watch?v=ldxFjLJ3rVY) — The mathematical foundation for this project.