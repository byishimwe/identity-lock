"""Lock tracking to one enrolled identity using Whole-Person Tracking and Face Association.

Integrates:
- Object / whole-person detection and tracking across frames (CPU-only)
- Spatial face-to-body mapping
- Three-state identity model (CONFIRMED, TEMPORARY, UNKNOWN)
- Exponential confidence decay C(t) = C0 * exp(-lambda * t)
- Target centring logic relative to webcam center
- Real-world handling of head turns, occlusions, and path crossings

Run: python -m src.face_tracking --name="Michael" [--debug]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .face_signals import (
    EyeSignal,
    ExpressionSignal,
    PerSessionAnalyzer,
    SignalStabilizer,
    BlinkCounter,
    eye_aspect_ratio,
    nose_position_in_frame,
)
from .recognition import (
    ArcFaceEmbedderONNX,
    DetectStats,
    FaceDBMatcher,
    FaceDet,
    FaceMeshDetector,
    MatchResult,
    align_face_5pt,
    load_db_npz,
)
from .whole_person_tracker import (
    CenteringResult,
    IdentityState,
    TrackedPerson,
    WholePersonTracker,
)
from .mqtt_publisher import FaceTrackMQTTPublisher

DB_PATH = Path("data/db/face_db.npz")

HUD_CYAN = (0, 255, 255)
HUD_YELLOW = (0, 215, 255)
HUD_ORANGE = (0, 140, 255)
HUD_RED = (0, 0, 255)
HUD_GREEN = (0, 255, 0)
HUD_WHITE = (240, 240, 240)
HUD_DARK = (30, 30, 30)
HUD_GRAY = (140, 140, 140)

BOX_CONFIRMED = HUD_CYAN
BOX_TEMPORARY = HUD_ORANGE
BOX_UNKNOWN = HUD_GRAY

NOSE_TIP_IDX = 1
DEFAULT_DIST_THRESHOLD = 0.42


def _recognize(frame, faces: List[FaceDet], embedder: ArcFaceEmbedderONNX, matcher: FaceDBMatcher) -> List[MatchResult]:
    return [matcher.match(embedder.embed(align_face_5pt(frame, face.kps))) for face in faces]


def _label(
    frame,
    text: str,
    xy: Tuple[int, int],
    color: Tuple[int, int, int],
    scale: float = 0.7,
    thickness: int = 2,
) -> None:
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, HUD_DARK, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_background_box(frame, top_left: Tuple[int, int], bottom_right: Tuple[int, int]) -> None:
    sub = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    if sub.size == 0:
        return
    dim = cv2.addWeighted(sub, 0.45, sub * 0, 0.55, 0)
    frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = dim


def _draw_crosshair(frame, cx: int, cy: int, color: Tuple[int, int, int], size: int = 16) -> None:
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 2, color, -1, cv2.LINE_AA)


def _draw_frame_border(frame, color: Tuple[int, int, int], thickness: int = 6) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness, cv2.LINE_AA)


def _draw_confidence_bar(
    frame,
    origin: Tuple[int, int],
    confidence: float,
    width: int = 160,
    height: int = 14,
    color: Tuple[int, int, int] = HUD_ORANGE,
) -> None:
    """Renders a confidence progress bar."""
    ox, oy = origin
    cv2.rectangle(frame, (ox, oy), (ox + width, oy + height), HUD_DARK, -1)
    fill_w = int(max(0.0, min(1.0, confidence)) * width)
    if fill_w > 0:
        cv2.rectangle(frame, (ox, oy), (ox + fill_w, oy + height), color, -1)
    cv2.rectangle(frame, (ox, oy), (ox + width, oy + height), HUD_WHITE, 1)


def open_camera(
    preferred_camera: Optional[int] = None,
    width: int = 1920,
    height: int = 1080,
) -> Tuple[cv2.VideoCapture, int]:
    candidates: List[int] = []
    if preferred_camera is not None:
        candidates.append(preferred_camera)
    for c in (0, 1, 2):
        if c not in candidates:
            candidates.append(c)

    for idx in candidates:
        for api in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            cap = cv2.VideoCapture(idx, api)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"[Camera] Active camera index {idx} ({frame.shape[1]}x{frame.shape[0]})")
                    return cap, idx
            cap.release()
    raise RuntimeError("No operational camera found on this system.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Whole-person tracking and face association with 3-state exponential decay."
    )
    parser.add_argument("--name", required=True, help="Exact name enrolled in data/db/face_db.npz.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0 for external webcam).")
    parser.add_argument("--width", type=int, default=1920, help="Camera width (default 1920).")
    parser.add_argument("--height", type=int, default=1080, help="Camera height (default 1080).")
    parser.add_argument("--fullscreen", action=argparse.BooleanOptionalAction, default=True, help="Display window in fullscreen mode.")
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True, help="Mirror video horizontally like a phone selfie camera.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_DIST_THRESHOLD,
        help=f"ArcFace distance cutoff (default {DEFAULT_DIST_THRESHOLD:.2f}); higher is more forgiving.",
    )
    parser.add_argument("--stable-frames", type=int, default=3)
    parser.add_argument("--half-life", type=float, default=2.0, help="Exponential confidence decay half-life in seconds (default 2.0).")
    parser.add_argument("--temp-threshold", type=float, default=0.35, help="Confidence threshold below which TEMPORARY becomes UNKNOWN (default 0.35).")
    parser.add_argument("--person-tracking", action=argparse.BooleanOptionalAction, default=True, help="Enable whole-person tracking with face association.")
    parser.add_argument("--debug", action="store_true", help="Print per-frame diagnostics and show on-screen debug HUD.")
    parser.add_argument("--mqtt", action=argparse.BooleanOptionalAction, default=True, help="Publish face centering data to MQTT.")
    parser.add_argument("--mqtt-broker", type=str, default="127.0.0.1", help="MQTT broker host IP (default 127.0.0.1).")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port (default 1883).")
    parser.add_argument("--mqtt-topic", type=str, default="face_track", help="MQTT topic to publish to (default face_track).")
    parser.add_argument("--mqtt-mode", choices=["steps", "direction"], default="steps", help="Publish target motor steps or direction.")
    args = parser.parse_args()

    target_name = args.name.strip()
    database = load_db_npz(DB_PATH)
    if not target_name or target_name not in database:
        parser.error(f'Target "{target_name}" was not found in the face database.')

    mqtt_pub: Optional[FaceTrackMQTTPublisher] = None
    if args.mqtt:
        mqtt_pub = FaceTrackMQTTPublisher(
            broker_host=args.mqtt_broker,
            broker_port=args.mqtt_port,
            topic=args.mqtt_topic,
        )
        mqtt_pub.connect()
    last_mqtt_msg = "IDLE"

    detector = FaceMeshDetector()
    embedder = ArcFaceEmbedderONNX()
    matcher = FaceDBMatcher(database, args.threshold)
    person_tracker = WholePersonTracker(
        target_name=target_name,
        half_life_seconds=args.half_life,
        temp_threshold=args.temp_threshold,
    )

    expression_stabilizer = SignalStabilizer(ExpressionSignal.NEUTRAL, args.stable_frames)
    eyes_stabilizer = SignalStabilizer(EyeSignal.EYES_OPEN, args.stable_frames)
    blink_counter = BlinkCounter(min_closure_frames=1, min_open_frames=2)
    analyzer = PerSessionAnalyzer()

    capture, active_camera_idx = open_camera(args.camera, width=args.width, height=args.height)

    window_name = "FaceLocking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    locked_once = False
    frame_idx = 0
    consecutive_drop_frames = 0
    prev_state: Optional[IdentityState] = None

    try:
        while True:
            frame_idx += 1
            ok, frame = capture.read()
            if not ok or frame is None:
                consecutive_drop_frames += 1
                if consecutive_drop_frames > 20:
                    print("[Camera] Camera stopped providing frames.")
                    break
                continue
            consecutive_drop_frames = 0
            if args.mirror:
                frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]
            cx, cy = width // 2, height // 2

            # 1. Detect faces and run ArcFace biometrics
            faces, stats = detector.detect_with_stats(frame)
            matches = _recognize(frame, faces, embedder, matcher) if faces else []

            # Package face detections for whole-person tracker
            face_detections_for_tracker: List[Tuple[Tuple[int, int, int, int], Optional[str], float, bool]] = []
            for face, match in zip(faces, matches):
                face_detections_for_tracker.append(
                    ((face.x1, face.y1, face.x2, face.y2), match.name, match.distance, match.accepted)
                )

            # 2. Update Whole-Person Tracking with face-to-body association
            active_tracks = person_tracker.update(frame, face_detections_for_tracker)
            target_track = person_tracker.get_target_track()

            # Target State & Centering
            if target_track is not None:
                target_state = target_track.state
                confidence = target_track.confidence
                is_ambiguous = target_track.is_ambiguous
                locked_once = True
            else:
                target_state = IdentityState.UNKNOWN
                confidence = 0.0
                is_ambiguous = False

            centering: Optional[CenteringResult] = person_tracker.calculate_centering(width, height, target_track)

            # 3. Facial signals extraction (when face is visible for target)
            target_face: Optional[FaceDet] = None
            target_match: Optional[MatchResult] = None
            if target_track is not None and target_track.face_bbox is not None:
                # Find matching face
                for f, m in zip(faces, matches):
                    if (f.x1, f.y1, f.x2, f.y2) == target_track.face_bbox:
                        target_face = f
                        target_match = m
                        break

            expression, eyes = "NOT_DETECTED", "NOT_DETECTED"
            blink_counted_this_frame = False
            blink_label_for_hud: Optional[str] = None
            if target_face is not None:
                observed = analyzer.analyze(target_face.landmarks)
                raw_expr = expression_stabilizer.update(observed.expression)
                raw_eyes = eyes_stabilizer.update(observed.eyes)
                blink_counted_this_frame = blink_counter.update(raw_eyes)
                if blink_counted_this_frame:
                    blink_label_for_hud = blink_counter.last_cycle_label
                expression = raw_expr.value
                eyes = raw_eyes.value
            elif target_state == IdentityState.TEMPORARY:
                expression = "HEAD_TURNED / OCCLUDED"
                eyes = "HEAD_TURNED / OCCLUDED"

            # 4. Target Centering & MQTT Transmission
            target_dx = float(centering.offset_px[0]) if centering is not None else 0.0
            is_active_target = target_state in (IdentityState.CONFIRMED, IdentityState.TEMPORARY)
            if mqtt_pub is not None:
                sent_msg = mqtt_pub.publish_face_position(
                    detected=is_active_target,
                    dx=target_dx,
                    frame_width=width,
                    mode=args.mqtt_mode,
                )
                if sent_msg is not None:
                    last_mqtt_msg = sent_msg

            # 5. Debug Logging
            if args.debug and target_state != prev_state:
                print(
                    f"[frame {frame_idx}] IDENTITY STATE -> {target_state.value} | "
                    f"conf={confidence:.3f} | tracks={len(active_tracks)} | "
                    f"ambiguous={is_ambiguous} | centering={centering.direction_text if centering else 'NONE'}",
                    file=sys.stderr, flush=True,
                )
                prev_state = target_state

            # 6. Render HUD View
            view = frame.copy()

            # Warning border for unconfirmed or lost states
            if not is_active_target and locked_once:
                _draw_frame_border(view, HUD_RED)
            elif target_state == IdentityState.TEMPORARY:
                _draw_frame_border(view, HUD_ORANGE, thickness=4)

            # Draw tracked person bodies
            for track in active_tracks:
                bx1, by1, bx2, by2 = track.body_bbox
                is_target_t = track is target_track

                if track.state == IdentityState.CONFIRMED:
                    box_col = BOX_CONFIRMED
                    state_lbl = f"CONFIRMED ({track.assigned_name})"
                elif track.state == IdentityState.TEMPORARY:
                    box_col = BOX_TEMPORARY
                    loss_s = time.time() - track.last_confirmed_time
                    state_lbl = f"TEMPORARY ({track.assigned_name}) C(t)={track.confidence:.2f} [{loss_s:.1f}s]"
                else:
                    box_col = BOX_UNKNOWN
                    state_lbl = f"UNKNOWN #{track.track_id}"

                if track.is_ambiguous:
                    state_lbl += " [AMBIGUOUS: CROSSING]"

                # Person body rectangle
                thick = 3 if is_target_t else 1
                cv2.rectangle(view, (bx1, by1), (bx2, by2), box_col, thick)
                _label(view, state_lbl, (bx1 + 5, max(24, by1 - 8)), box_col, 0.55, 2)

                # Draw associated face box if detected
                if track.face_bbox is not None:
                    fx1, fy1, fx2, fy2 = track.face_bbox
                    cv2.rectangle(view, (fx1, fy1), (fx2, fy2), (255, 255, 255), 1)

            # HUD Telemetry & Banners
            hud_color = HUD_CYAN if target_state == IdentityState.CONFIRMED else (
                HUD_YELLOW if target_state == IdentityState.TEMPORARY else HUD_ORANGE
            )

            # Status Banner
            banner_y_offset = 0
            if target_state == IdentityState.TEMPORARY:
                banner = f"  TEMPORARY CONTINUITY: FACE LOST  (DECAYING)  "
                (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                _draw_background_box(view, (10, 105), (10 + tw + 20, 105 + th + 25))
                _label(view, banner, (15, 105 + th + 12), HUD_ORANGE, 0.8, 2)
                banner_y_offset = 32
            elif not is_active_target and locked_once:
                banner = f"  WARNING: TARGET LOST (UNKNOWN)  "
                (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                _draw_background_box(view, (10, 105), (10 + tw + 20, 105 + th + 25))
                _label(view, banner, (15, 105 + th + 12), HUD_RED, 0.8, 2)
                banner_y_offset = 32

            # Telemetry text
            state_text = f"TARGET = {target_name} [{target_state.value}]"
            _label(view, state_text, (10, 30), hud_color)

            # Confidence bar readout
            conf_str = f"CONFIDENCE = {confidence * 100:.1f}%  (Half-life: {args.half_life:.1f}s)"
            _label(view, conf_str, (10, 60), hud_color)
            _draw_confidence_bar(view, (450, 48), confidence, width=150, height=14, color=hud_color)

            _label(view, f"EXPRESSION = {expression}", (10, 90), hud_color)

            blink_line = f"BLINKS = {blink_counter.total}  (full={blink_counter.full_blinks}  L={blink_counter.winks_left}  R={blink_counter.winks_right})"
            if blink_counter.cycle_highlight and blink_label_for_hud:
                blink_line = f"BLINKS = {blink_counter.total}  [{blink_label_for_hud}!]"
                blink_color = (80, 255, 80)
            else:
                blink_color = hud_color
            _label(view, blink_line, (10, 120 + banner_y_offset), blink_color, 0.7, 2)

            # Centering line
            cent_y = 155 + banner_y_offset
            if centering is not None:
                cent_text = (
                    f"CENTERING = {centering.direction_text}  "
                    f"dx={centering.offset_px[0]:+4d}px ({centering.normalized_offset[0]:+.2f})  "
                    f"dy={centering.offset_px[1]:+4d}px  dist={centering.distance_px:.0f}px"
                )
                _label(view, cent_text, (10, cent_y), (80, 255, 80) if abs(centering.offset_px[0]) < 20 else hud_color, 0.65, 2)
            else:
                _label(view, "CENTERING = TARGET NOT FOUND", (10, cent_y), HUD_GRAY, 0.65, 2)

            # MQTT status
            mqtt_y = cent_y + 28
            if mqtt_pub is not None:
                if mqtt_pub.is_connected:
                    mqtt_text = f"MQTT [{args.mqtt_topic}] => {last_mqtt_msg}"
                    _label(view, mqtt_text, (10, mqtt_y), (80, 255, 80), 0.65, 2)
                else:
                    _label(view, f"MQTT: CONNECTING ({args.mqtt_broker})...", (10, mqtt_y), HUD_ORANGE, 0.60, 2)
            else:
                _label(view, "MQTT: DISABLED", (10, mqtt_y), HUD_GRAY, 0.55, 1)

            # Center Crosshair & Target Vector
            cross_color = HUD_CYAN if target_state == IdentityState.CONFIRMED else (
                HUD_YELLOW if target_state == IdentityState.TEMPORARY else HUD_GRAY
            )
            _draw_crosshair(view, cx, cy, cross_color)

            if centering is not None:
                tx, ty = centering.target_xy
                # Line from center crosshair to target
                cv2.line(view, (cx, cy), (tx, ty), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(view, (tx, ty), 5, (0, 255, 255) if target_state == IdentityState.CONFIRMED else (0, 140, 255), -1, cv2.LINE_AA)

            _label(view, "Q = quit  |  F = fullscreen" + ("  [DEBUG]" if args.debug else ""), (10, view.shape[0] - 15), (220, 220, 220), 0.55, 1)
            cv2.imshow(window_name, view)

            try:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key in (ord("f"), ord("F")):
                    is_fs = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
                    cv2.setWindowProperty(
                        window_name,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_NORMAL if is_fs else cv2.WINDOW_FULLSCREEN,
                    )
            except KeyboardInterrupt:
                break
    finally:
        if mqtt_pub is not None:
            mqtt_pub.disconnect()
        try:
            capture.release()
        finally:
            detector.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
