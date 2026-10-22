# FaceLocking

Face recognition and whole-person tracking that locks onto one enrolled identity and displays that person's expression (`SMILE` or `NEUTRAL`) and eye state (`EYES_OPEN` or `EYES_CLOSED`), while maintaining temporal identity continuity across temporary facial loss through an honest three-state model (`CONFIRMED`, `TEMPORARY`, `UNKNOWN`) with exponential confidence decay.

## Setup

Use Python 3.11 (or compatible Python with requirements installed), then install dependencies:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place a compatible ArcFace ONNX model at either:

```text
models/embedder_arcface.onnx
models/w600k_r50.onnx
```

Model weights and face databases are deliberately ignored by Git.

## Enroll identities

```powershell
python -m src.enroll --name="Enrolled_name"
```

Press `SPACE` to take samples, `a` for automatic capture, `s` to save after the requested sample count, or `q` to quit. Enrollment is stored locally at `data/db/face_db.npz`.

To keep your existing enrollments when moving from the original project, copy these local files yourself (do not commit them):

```text
old-project/data/db/face_db.npz -> FaceLocking/data/db/face_db.npz
old-project/data/db/face_db.json -> FaceLocking/data/db/face_db.json
```

## Track one person with Whole-Person Tracking & Face Association

```powershell
python -m src.face_tracking --name="Enrolled_name"
```

`--name` must be an exact enrolled name.

### Whole-Person Tracking & Three-State Model
The tracker integrates whole-person tracking and spatial face-to-body association with an honest three-state model and exponential confidence decay:
- **`CONFIRMED`**: Face is directly detected inside the person body and biometrically verified via ArcFace ($d \le 0.42$). Tracking confidence is restored to $C_0 \approx 1.0$.
- **`TEMPORARY`**: Face is temporarily lost (e.g., head turned away, profile, hand occlusion), but the person's body continues to be tracked physically. Identity is maintained provisionally while confidence decays exponentially according to:
  $$C(t) = C_0 \cdot e^{-\lambda t}$$
  where $\lambda = \frac{\ln(2)}{t_{1/2}}$ (configurable via `--half-life`, default $2.0\text{ s}$).
- **`UNKNOWN`**: When confidence drops below threshold ($C(t) < 0.35$, configurable via `--temp-threshold`), or when tracking ambiguity occurs (e.g. unverified path crossing), the system explicitly represents genuine uncertainty and transitions to `UNKNOWN`.

### Target Centering
Target centering offset (`dx`, `dy`) and normalized offsets are calculated relative to the webcam frame center $(c_x, c_y)$. When the face is visible, centring anchors to the facial landmarks/nose; during head turns or facial occlusions in `TEMPORARY` state, centring smoothly transitions to the tracked upper-body / torso centroid.

Run automated unit and integration tests:
```powershell
pytest tests/test_whole_person_tracker.py -v
```

## Tuning signals

`src/face_signals.py` contains the eye and smile thresholds. Smile tolerance is adjusted slightly for smaller faces to account for noisier distant landmarks. `--stable-frames` controls how many matching frames are required before an expression or eye state display changes.

## Hardware & MQTT Stepper Motor Tracking

FaceLocking can publish real-time face centering and angular error to an ESP8266 microcontroller over MQTT to steer a 28BYJ-48 stepper motor ("tacho" pointer indicator or turret).

### Hardware Wiring

| ULN2003 Driver Pin | ESP8266 (NodeMCU) Pin | Description |
| :--- | :--- | :--- |
| **IN1** | **D1 (GPIO 5)** | Coil A |
| **IN2** | **D2 (GPIO 4)** | Coil B |
| **IN3** | **D5 (GPIO 14)** | Coil C |
| **IN4** | **D6 (GPIO 12)** | Coil D |
| **GND (-)** | **GND** | Ground |
| **+ (VCC / 5V)** | **VIN / VU** | 5V Power (Do NOT use 3V3!) |

*Plug the 5-pin 28BYJ-48 motor connector directly into the ULN2003 white socket.*

### Microcontroller Firmware

Open and upload the Arduino sketch at:
[`arduino/esp8266_face_tracker/esp8266_face_tracker.ino`](arduino/esp8266_face_tracker/esp8266_face_tracker.ino)

Required Arduino libraries:
- `PubSubClient` (Nick O'Leary)
- `AccelStepper` (Mike McCauley)

### Running Face Tracking with MQTT

```powershell
python -m src.face_tracking --name="Michael"
```

Options:
- `--mqtt` / `--no-mqtt`: Toggle MQTT publishing (enabled by default).
- `--mqtt-broker`: Broker IP address (default `127.0.0.1`).
- `--mqtt-port`: Broker port (default `1883`).
- `--mqtt-topic`: Topic name (default `face_track`).
- `--mqtt-mode`: `steps` (default for tacho pointer) or `direction` (`LEFT`/`RIGHT`/`CENTER`).
- `--half-life`: Exponential confidence decay half-life in seconds (default `2.0`).
- `--temp-threshold`: Confidence cutoff below which TEMPORARY becomes UNKNOWN (default `0.35`).
- `--person-tracking` / `--no-person-tracking`: Toggle whole-person tracking mode (enabled by default).
