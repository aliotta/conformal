# Contributing to Conformal Droste Animation Engine

First off, thank you for considering contributing to this project! It's through people like you that this tool gets better for everyone.

## Pull Request Guidelines

To ensure the highest quality and to help me review your changes efficiently, please follow these guidelines when submitting a Pull Request:

### 1. Visual Verification (Required for Merging)
I am significantly more likely to merge in code changes—especially those affecting the mathematical mapping or rendering logic—if you include visual proof of the output.

* **Use the Reference Asset:** When testing your changes, please use the existing `assets/centered_eye.png` file as your source. 
* **Upload the Output:** Upload the resulting `.gif` files directly into your Pull Request description. 
* **Compare Outputs:** If your PR is an optimization or a bug fix, providing a "before and after" comparison of the rendered GIFs is highly encouraged.

### 2. Performance and Optimization
Since this engine is optimized for Apple's **MLX** framework and GPU acceleration, please ensure that your contributions do not negatively impact real-time performance. If you are introducing a new dependency or a change in the processing pipeline, include a brief note on the performance impact.

### 3. Code Style
* Ensure your code is well-commented, particularly around complex mathematical implementations.
* Follow standard Python PEP 8 guidelines where possible.

## How to Report a Bug
If you find a bug, please open an issue and include:
* Your hardware specifications (specifically which Mac chip you are using).
* The exact steps to reproduce the behavior.
* Any error logs or screenshots of the distorted output.

## Feature Requests
Have an idea for a new platform preset or a mapping effect? Feel free to open an issue to discuss it first!