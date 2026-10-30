# Identity-Lock: Intelligent Whole-Body Recognition & Person Re-Identification

Identity-Lock is a real-time, CPU-optimized computer vision system that identifies a person using their face as a verified identity anchor, associates that identity with their whole body, learns their visual body appearance across multiple angles (including front, side, and rear views), and continues tracking and re-identifying them through 180-degree rotations when their face is completely invisible.

---

## 1. System Architecture

The pipeline processes webcam frames through a multi-rate, CPU-optimized architecture:

```
Webcam Frame (BGR)
  ├──> YOLOX-Nano ONNX: Whole-person detection (detects visible bodies regardless of face orientation)
  ├──> YuNet ONNX: Face detection & 5-point facial keypoints
  ├──> ArcFace ONNX: 512-D facial embedding extraction & enrolled identity matching
  ├──> Spatial Face-to-Body Association: Bipartite upper-body containment gating
  ├──> OSNet-x0.25 Re-ID ONNX + HSV Histograms: Multi-modal body appearance feature extraction
  ├──> Multi-View Appearance Gallery: Dynamic prototype learning during 180° rotation
  ├──> Appearance-Aware Multi-Object Tracking: Motion prediction & gated association
  ├──> Three-State Identity Model: CONFIRMED, TEMPORARY, and UNKNOWN with exponential decay
  ├──> Anti-Identity Confusion: Distinguishes target from other back-facing people
  └──> Whole-Body Target Centring: Smoothed centroid error calculation for camera tracking
```

---

## 2. Models & Setup

The application runs locally on CPU using ONNX Runtime. Model weights are located in `models/`:

| Component | Model Name | Format / Framework | Input Dimensions | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Person Detection** | `yolox_nano.onnx` | ONNX (CPU) | `1 x 3 x 416 x 416` | Whole-body detection (front/side/rear) |
| **Face Detection** | `face_detection_yunet_2023mar.onnx` | OpenCV DNN / ONNX | Adaptive (e.g. `640 x 480`) | Face detection & 5-point landmarks |
| **Face Recognition** | `w600k_r50.onnx` (ArcFace) | ONNX (CPU) | `1 x 3 x 112 x 112` | 512-D facial identity embedding |
| **Person Re-ID** | `osnet_x0_25_msmt17.onnx` | ONNX (CPU) | `1 x 3 x 256 x 128` | 512-D body appearance embedding |
| **Facial Signals** | MediaPipe FaceMesh | CPU Library | Adaptive | 468-point landmarks for blink/smile |

### Installation

Requires Python 3.11 or higher.

```powershell
python -m pip install -r requirements.txt
```

Verify dependencies: `opencv-python`, `onnxruntime`, `numpy`, `mediapipe`, `paho-mqtt`.

---

## 3. Usage & Workflows

### Step 1: Enroll an Identity

Capture facial biometric samples of a person using YuNet and ArcFace:

```powershell
python -m src.enroll --name="Michael"
```

- Press `SPACE` to capture individual samples, or press `a` for automatic capture.
- Once 15 samples are collected, press `s` to save to `data/db/face_db.npz`.
- Press `q` to quit.

### Step 2: Run Whole-Body Tracking & Re-Identification

Track an enrolled person using whole-person detection, body Re-ID, and three-state identity continuity:

```powershell
python -m src.face_tracking --name="Michael"
```

### Options & Performance Tuning

```powershell
python -m src.face_tracking --name="Michael" --samples=20 --width=1280 --height=720 --half-life=2.0 --reid-threshold=0.48 --yolox-interval=2 --debug
```

- `--name`: Exact enrolled identity name (required).
- `--samples`: Number of body appearance takes to collect across 360° rotation before tracking starts (default `20`).
- `--skip-sampling`: Skip initial calibration sampling and start tracking immediately.
- `--camera`: Camera index (default `0`).
- `--width`, `--height`: Capture resolution (default `1280x720`).
- `--mirror` / `--no-mirror`: Horizontal mirror flip (default `True`).
- `--fullscreen` / `--no-fullscreen`: Fullscreen window mode (default `True`).
- `--threshold`: ArcFace distance cutoff (default `0.42`).
- `--half-life`: Exponential confidence decay half-life in seconds (default `2.0`).
- `--temp-threshold`: Cutoff below which `TEMPORARY` transitions to `UNKNOWN` (default `0.35`).
- `--reid-threshold`: Re-ID appearance similarity threshold (default `0.48`).
- `--yolox-interval`: Run YOLOX person detector every $N$ frames (default `2` for smooth 25+ FPS CPU execution).
- `--debug`: Enable runtime diagnostics and verbose telemetry.
- `--mqtt` / `--no-mqtt`: Toggle MQTT publishing for stepper motors (disabled by default).

### Interactive Hotkeys

- `Q`: Quit application.
- `F`: Toggle fullscreen / windowed mode.
- `C`: Trigger manual 360° body re-sampling calibration at any time.

---

## 4. Two-Phase Workflow: Calibration Sampling then Active Tracking

To avoid falsely dropping to `UNKNOWN` when the user turns their back before rear-view samples are learned, Identity-Lock employs an explicit **Two-Phase Workflow**:

### Phase 1: Calibration Body Sampling (Look & Rotate 360°)
1. **Face Anchor Lock**: When the target person stands in front of the camera, ArcFace verifies their facial embedding against the database and locks onto their body track.
2. **360° Multi-Angle Takes**: The tracker enters `SAMPLING` mode. An onscreen HUD banner prompts: `[BODY SAMPLING: X / 20 TAKES] >>> PLEASE ROTATE SLOWLY 360° <<<` with a real-time progress bar.
3. **Protected Track State**: As the user turns around (front, 45°, 90°, 135°, 180° back view), the track is protected in `CONFIRMED` state and **never drops to `UNKNOWN`**.
4. **Gallery Population**: The system collects $N$ diverse appearance observations (OSNet deep feature vectors + HSV torso/leg histograms) sampled across rotation angles into the appearance gallery.

### Phase 2: Active Whole-Body Tracking & Re-Identification
1. **Transition**: Once all required samples (e.g. 20 takes) are finished, a green banner flashes: `*** ALL SAMPLES FINISHED! 360° WHOLE-BODY TRACKING ACTIVE ***` and the system switches to `TRACKING` mode.
2. **Back-Facing Continuity**: When Michael turns completely away, his rear view is matched against the rear-view prototypes collected during sampling ($\ge 0.65$–$0.85$ Re-ID similarity). The tracker smoothly transitions to **`TEMPORARY`** with exponential confidence decay $C(t) = C_0 \cdot e^{-\lambda t}$ instead of dropping to `UNKNOWN`.
3. **Anti-Identity Confusion**: If a second person enters facing away, their appearance is compared against Michael's 360° gallery. Because their features do not match, they remain separate as `UNKNOWN`.
4. **Face Re-Emergence**: When Michael turns back to face the camera, ArcFace re-verifies his identity and resets state to **`CONFIRMED`** with confidence 100%.

---

## 5. Three-State Identity Model

- **`CONFIRMED`** *(Cyan/Green Bounding Box)*:
  Face is currently visible inside the body box and verified biometrically by ArcFace. Direct evidence is active.
- **`TEMPORARY`** *(Orange/Yellow Bounding Box)*:
  Face is temporarily lost (head turned, profile, occlusion, back-facing), but body tracking continuity and Re-ID appearance matching support provisional identity. Displays provisional label `[C(t)=XX%]`.
- **`UNKNOWN`** *(Gray/Red Bounding Box)*:
  The person has not been identified, temporal confidence has expired ($C(t) < 0.35$), or path-crossing ambiguity occurred. The system represents honest uncertainty.

---

## 6. Whole-Body Target Centring

Target centring calculates the displacement from the webcam frame center:
$$dx = x_{\text{body\_center}} - x_{\text{frame\_center}}, \quad dy = y_{\text{body\_center}} - y_{\text{frame\_center}}$$
Target positions are smoothed via an Exponential Moving Average (EMA, $\alpha = 0.65$) to eliminate video jitter while maintaining responsive tracking. Centring remains continuous and uninterrupted even when the person turns their head completely away.

---

## 7. Known Technical Limitations

1. **Extreme Lighting Shifts**: Drastic illumination changes (e.g. turning off room lights) alter clothing color distribution and can degrade Re-ID similarity.
2. **Identical Clothing / Uniforms**: If two people wear identical uniforms and cross paths while facing away, appearance Re-ID cannot uniquely distinguish them without facial biometrics. The system will correctly transition to `UNKNOWN` to avoid false identification.
3. **Severe Long-Term Occlusion**: If a person is completely hidden behind an obstacle for longer than `max_missing_seconds` (2.5s), the physical track expires and requires re-acquisition upon reappearance.
