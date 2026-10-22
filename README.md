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
python -m src.face_tracking --name="Michael" --width=1280 --height=720 --half-life=2.0 --reid-threshold=0.48 --yolox-interval=2 --debug
```

- `--name`: Exact enrolled identity name (required).
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

---

## 4. How Body Learning & 180-Degree Rotation Works

1. **Front-Facing Arrival**:
   When Michael stands in front of the webcam, YOLOX detects his body and YuNet detects his face. ArcFace verifies his identity. Spatial association links the face to his body track, entering **`CONFIRMED`**.
2. **Appearance Profile Learning**:
   The system extracts a 512-D OSNet embedding, an upper-torso HSV color histogram, and a lower-body HSV histogram, saving them into Michael's bounded multi-view appearance gallery.
3. **180-Degree Rotation**:
   As Michael rotates through $90^\circ$ and $180^\circ$, the body tracker maintains continuous physical tracking. Reliable side-view and rear-view observations are added to the gallery.
4. **Face Disappearance & Back-Facing Continuity**:
   When Michael faces completely away, his face is no longer visible. The state transitions to **`TEMPORARY`**, and temporal confidence decays exponentially:
   $$C(t) = C_0 \cdot e^{-\lambda t}$$
   where $\lambda = \frac{\ln(2)}{t_{1/2}}$. Re-ID appearance matching against Michael's multi-view gallery confirms that the tracked body is indeed Michael.
5. **Anti-Identity Confusion**:
   If a second person enters the frame (also facing away), the system assigns them a separate anonymous track ID (`UNKNOWN`). Michael's target identity is **not** assigned to the new person.
6. **Face Re-Emergence**:
   When Michael turns back around, YuNet re-detects his face, ArcFace confirms his identity, and the state returns to **`CONFIRMED`**, restoring confidence to 100%.

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
