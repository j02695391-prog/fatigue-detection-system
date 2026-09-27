# Multi-Feature Fatigue Detection System

A real-time, camera-based (or video-file-based) driver/user fatigue detection system built on classical computer vision techniques, combining eye-closure, yawning, and head-pose signals into a single composite fatigue score with an alarm trigger.

## How it works

The system uses dlib's pretrained 68-point facial landmark detector to locate facial features in each frame, then computes three independent signals:

- **Eye Aspect Ratio (EAR)** — a standard geometric measure (Soukupova & Cech, 2016) of eye openness, used to detect blinks and sustained eye closure
- **Mouth Aspect Ratio (MAR)** — the same geometric approach applied to the mouth, used to detect yawning
- **Head pose (pitch/yaw)** — estimated via OpenCV's `solvePnP` against a standard 3D face model, used to detect nodding or head-drooping

These three signals are combined into a weighted 0-40 fatigue score (blink frequency, yawn count, longest continuous eye closure, and head pose each contribute), which is mapped to a fatigue level (Normal / Mild / Moderate / Severe). A severe rating triggers a visual on-screen alert and an audio alarm on a separate thread so it doesn't block frame processing.

This project builds on well-established computer vision techniques (EAR/MAR-based blink and yawn detection is a widely used method in the drowsiness-detection literature) rather than introducing a new detection algorithm. The original contribution here is the multi-signal fusion into a single composite score and alert system, rather than relying on any one signal alone.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the pretrained facial landmark model (not included in this repo -- see note below) and place it in the project root:
   ```
   shape_predictor_68_face_landmarks.dat
   ```
   Available from [dlib's official model repository](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) (decompress the `.bz2` file after downloading).

## Usage

Run on a video file:
```bash
python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video test.mp4
```

Run on a live webcam feed:
```bash
python detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat
```

Press `ESC` to exit.

## Platform note

The audio alarm uses Python's built-in `winsound` module, which is **Windows-only**. On macOS/Linux, either comment out the `sound_alarm()` call or swap in a cross-platform audio library (e.g. `playsound` or `simpleaudio`).

## Demo

`test.mp4` is included as a sample input for testing the pipeline without a webcam.
