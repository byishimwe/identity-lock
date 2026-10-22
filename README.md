# FaceLocking

Face recognition that locks onto one enrolled identity and displays that
person's expression (`SMILE` or `NEUTRAL`) and eye state (`EYES_OPEN` or
`EYES_CLOSED`). Other faces never provide signals for the target.

## Setup

Use Python 3.11, then install dependencies:

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

Press `SPACE` to take samples, `a` for automatic capture, `s` to save after
the requested sample count, or `q` to quit. Enrollment is stored locally at
`data/db/face_db.npz`.

To keep your existing enrollments when moving from the original project,
copy these local files yourself (do not commit them):

```text
old-project/data/db/face_db.npz -> FaceLocking/data/db/face_db.npz
old-project/data/db/face_db.json -> FaceLocking/data/db/face_db.json
```

## Track one person

```powershell
python -m src.face_tracking --name="Enrolled_name"
```

`--name` must be an exact enrolled name. The tracker shows `NOT_DETECTED`
when that person leaves the frame and does not switch to another face. Press
`Q` to quit.

## Tuning signals

`src/face_signals.py` contains the eye and smile thresholds. Smile tolerance
is adjusted slightly for smaller faces to account for noisier distant
landmarks. `--stable-frames` controls how many matching frames are required
before an expression or eye state display changes.

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

