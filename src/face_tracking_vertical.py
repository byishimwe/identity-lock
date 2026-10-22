# """Vertical face tracking script.
#
# This script is a copy of `src/face_tracking.py` but publishes the vertical
# offset (Y‑axis) of the detected nose tip instead of the horizontal offset.
# The sign is inverted so that moving the face **up** results in a **positive**
# step count (motor turns upward) and moving **down** results in a negative
# step count.  All other functionality – camera handling, MQTT publishing,
# full‑screen window, mirroring, etc. – is identical.
#
# Usage example:
#   python -m src.face_tracking_vertical --name="Michael" --camera=0
#
# The script publishes the same MQTT topic (`face_track`) as the horizontal
# version.  The Arduino sketch for the vertical motor should subscribe to the
# same topic and drive a separate stepper motor.
# """

from __future__ import annotations

import argparse
import sys
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
from .mqtt_publisher import FaceTrackMQTTPublisher

DB_PATH = Path("data/db/face_db.npz")

HUD_CYAN = (0, 255, 255)
HUD_YELLOW = (0, 200, 255)
HUD_ORANGE = (0, 140, 255)
HUD_RED = (0, 0, 255)
HUD_WHITE = (240, 240, 240)
HUD_DARK = (30, 30, 30)
BOX_TARGET = HUD_CYAN
BOX_OTHER_KNOWN = (0, 255, 0)
BOX_UNKNOWN = HUD_RED

NOSE_TIP_IDX = 1

DEFAULT_DIST_THRESHOLD = 0.42
TEMPORAL_HOLD_FRAMES = 4
TEMPORAL_HOLD_SLACK = 0.08


@dataclass
class TemporalLock:
    last_accepted_index: int = -1
    last_accepted_frames_ago: int = 9999
    best_distance: float = 1.0


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


def _hud_color(locked_once: bool, detected: bool) -> Tuple[int, int, int]:
    if not locked_once:
        return HUD_CYAN if detected else HUD_RED
    return HUD_YELLOW if detected else HUD_ORANGE


def _warning_color(locked_once: bool, detected: bool) -> Optional[Tuple[int, int, int]]:
    if locked_once and not detected:
        return HUD_RED
    if not locked_once and not detected:
        return HUD_ORANGE
    return None


def _draw_frame_border(frame, color: Tuple[int, int, int], thickness: int = 6) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness, cv2.LINE_AA)


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
    parser = argparse.ArgumentParser(description="Vertical face‑tracking for neck pitch.")
    parser.add_argument("--name", required=True, help="Exact name enrolled in data/db/face_db.npz.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0).")
    parser.add_argument("--width", type=int, default=1920, help="Camera width.")
    parser.add_argument("--height", type=int, default=1080, help="Camera height.")
    parser.add_argument("--fullscreen", action=argparse.BooleanOptionalAction, default=True, help="Fullscreen window.")
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=False, help="Mirror video horizontally.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_DIST_THRESHOLD,
        help=f"ArcFace distance cutoff (default {DEFAULT_DIST_THRESHOLD:.2f}); higher is more forgiving.")
    parser.add_argument("--stable-frames", type=int, default=3)
    parser.add_argument("--debug", action="store_true", help="Show debug HUD.")
    parser.add_argument("--mqtt", action=argparse.BooleanOptionalAction, default=True, help="Publish to MQTT.")
    parser.add_argument("--mqtt-broker", type=str, default="127.0.0.1")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument("--mqtt-topic", type=str, default="face_track")
    parser.add_argument("--mqtt-mode", choices=["steps", "direction"], default="steps")
    parser.add_argument("--invert-vertical", action=argparse.BooleanOptionalAction, default=False,
        help="Invert Y offset so moving face up yields positive steps.")
    args = parser.parse_args()

    target_name = args.name.strip()
    database = load_db_npz(DB_PATH)
    if not target_name or target_name not in database:
        parser.error(f'Target "{target_name}" was not found in the face database.')

    mqtt_pub: Optional[FaceTrackMQTTPublisher] = None
    if args.mqtt:
        mqtt_pub = FaceTrackMQTTPublisher(broker_host=args.mqtt_broker, broker_port=args.mqtt_port, topic=args.mqtt_topic)
        mqtt_pub.connect()
    last_mqtt_msg = "IDLE"

    detector = FaceMeshDetector()
    embedder = ArcFaceEmbedderONNX()
    matcher = FaceDBMatcher(database, args.threshold)
    expression_stabilizer = SignalStabilizer(ExpressionSignal.NEUTRAL, args.stable_frames)
    eyes_stabilizer = SignalStabilizer(EyeSignal.EYES_OPEN, args.stable_frames)
    blink_counter = BlinkCounter(min_closure_frames=1, min_open_frames=2)
    analyzer = PerSessionAnalyzer()

    capture, active_camera_idx = open_camera(args.camera, width=args.width, height=args.height)

    window_name = "FaceLockingVertical"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    locked_once = False
    frame_idx = 0
    consecutive_drop_frames = 0
    prev_detected: Optional[bool] = None
    prev_eyes_value: Optional[str] = None
    prev_expr_value: Optional[str] = None
    temporal_lock = TemporalLock()
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
            faces, stats = detector.detect_with_stats(frame)
            matches = _recognize(frame, faces, embedder, matcher) if faces else []
            # select target face as in the horizontal script
            accepted_index: Optional[int] = next(
                (i for i, m in enumerate(matches) if m.accepted and m.name == target_name), None,
            )
            target_index = accepted_index
            lock_relaxed = False
            if target_index is None and locked_once and faces and temporal_lock.last_accepted_frames_ago <= TEMPORAL_HOLD_FRAMES:
                best_idx: Optional[int] = None
                best_score = args.threshold + TEMPORAL_HOLD_SLACK
                for i, m in enumerate(matches):
                    if m.distance <= best_score:
                        best_idx = i
                        best_score = m.distance
                if best_idx is not None:
                    target_index = best_idx
                    lock_relaxed = True

            if accepted_index is not None:
                temporal_lock.last_accepted_index = accepted_index
                temporal_lock.last_accepted_frames_ago = 0
                temporal_lock.best_distance = matches[accepted_index].distance
            else:
                temporal_lock.last_accepted_frames_ago = min(temporal_lock.last_accepted_frames_ago + 1, 9999)

            detected = target_index is not None
            nose_hud = "NOT_DETECTED"
            nose_xy: Optional[Tuple[int, int]] = None
            if detected:
                locked_once = True
                face = faces[target_index]
                observed = analyzer.analyze(face.landmarks)
                expression = expression_stabilizer.update(observed.expression).value
                eyes = eyes_stabilizer.update(observed.eyes).value
                blink_counter.update(eyes)
                nose = nose_position_in_frame(face.landmarks, width, height)
                nose_hud = f"OFFSET ({nose.offset_text})  DIST {nose.distance_px:.0f}px"
                nose_xy = (int(round(face.landmarks[NOSE_TIP_IDX, 0])), int(round(face.landmarks[NOSE_TIP_IDX, 1])))

            # ----- vertical offset computation -----
            # nose.offset_px returns (dx, dy) where dy is positive **downwards** in image space.
            # To make "up" correspond to positive motor steps we optionally invert the sign.
            vertical_offset = float(nose.offset_px[1]) if detected and nose is not None else 0.0
            if args.invert_vertical:
                vertical_offset = -vertical_offset

            if mqtt_pub is not None:
                sent_msg = mqtt_pub.publish_face_position(
                    detected=detected,
                    dx=vertical_offset,  # using dx field for the vertical value
                    frame_width=height,  # use frame HEIGHT for vertical normalization
                    mode=args.mqtt_mode,
                )
                if sent_msg is not None:
                    last_mqtt_msg = sent_msg

            # ----- HUD (mostly unchanged) -----
            view = frame.copy()
            warn = _warning_color(locked_once, detected)
            if warn is not None:
                _draw_frame_border(view, warn)

            for i, (face, match) in enumerate(zip(faces, matches)):
                is_target = i == target_index
                color = BOX_TARGET if is_target else (BOX_OTHER_KNOWN if match.accepted else BOX_UNKNOWN)
                label = match.name if match.accepted else "Unknown"
                if is_target:
                    label += " [LOCKED]"
                    if lock_relaxed:
                        label += " *"
                cv2.rectangle(view, (face.x1, face.y1), (face.x2, face.y2), color, 3 if is_target else 2)
                _label(view, f"{label} d={match.distance:.3f}", (face.x1, max(24, face.y1 - 10)), color, 0.6, 2)

            hud_color = _hud_color(locked_once, detected)
            state = "LOCKED" if locked_once else "WAITING"
            _label(view, f"TARGET = {target_name} ({state})", (10, 30), hud_color)
            _label(view, f"NOSE VERTICAL = {nose_hud}", (10, 60), hud_color)
            if mqtt_pub is not None:
                if mqtt_pub.is_connected:
                    _label(view, f"MQTT [{args.mqtt_topic}] => {last_mqtt_msg}", (10, 90), (80, 255, 80), 0.65, 2)
                else:
                    _label(view, f"MQTT: CONNECTING ({args.mqtt_broker})...", (10, 90), HUD_ORANGE, 0.60, 2)
            else:
                _label(view, "MQTT: DISABLED", (10, 90), (160, 160, 160), 0.55, 1)

            cross_color = HUD_CYAN if detected else (HUD_YELLOW if not locked_once else HUD_ORANGE)
            _draw_crosshair(view, cx, cy, cross_color)
            if nose_xy is not None:
                cv2.line(view, (cx, cy), nose_xy, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(view, nose_xy, 4, (255, 80, 80), -1, cv2.LINE_AA)

            _label(view, "Q = quit  |  F = toggle fullscreen" + ("  [DEBUG]" if args.debug else ""),
                (10, view.shape[0] - 15), (220, 220, 220), 0.55, 1)
            cv2.imshow(window_name, view)
            try:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q')):
                    break
                if key in (ord('f'), ord('F')):
                    is_fs = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL if is_fs else cv2.WINDOW_FULLSCREEN)
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
