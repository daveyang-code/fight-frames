# Real-Time Punch Detection with MediaPipe and OpenCV

This project provides a real-time punch detection system using MediaPipe and OpenCV. It analyzes hand movements from webcam input to detect punching motions based on velocity, direction, and distance from a starting position.

# Features

- Real-time video processing via OpenCV

- Hand tracking using MediaPipe

- Detection of punch extension and retraction phases

- Visualization of:

  - Hand velocity and direction

  - Punch state (ready, extension, retraction)

  - Distance from starting position over time (graph)

  - Punch statistics

- FPS counter and debugging information overlay

# Dependencies
Make sure you have the following installed:
```
pip install opencv-python-headless pillow
```
