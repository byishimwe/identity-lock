"""Intelligent Whole-Body Identity Tracking and Person Re-Identification.

Integrates:
- YOLOX ONNX whole-person detection (CPU-optimized)
- YuNet ONNX face detection & ArcFace identity recognition
- Spatial face-to-body association
- Multi-view body appearance learning through 180-degree rotation
- Appearance-aware tracking and person Re-ID across facial loss
- Three-state identity model (CONFIRMED, TEMPORARY, UNKNOWN) with exponential confidence decay
- Whole-body target centring with jitter smoothing
- Preserved facial expressions and eye analysis when face is visible
- Multi-rate inference scheduling for high-efficiency CPU operation

Run:
  python -m src.face_tracking --name="Michael"
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .body_features import BodyAppearanceExtractor, BodyObservation
from .face_signals import (
    BlinkCounter,
    ExpressionSignal,
    EyeSignal,
    PerSessionAnalyzer,
    SignalStabilizer,
)
from .mqtt_publisher import FaceTrackMQTTPublisher
from .recognition import (
    ArcFaceEmbedderONNX,
    FaceDBMatcher,
    FaceDet,
    FaceMeshDetector,
    MatchResult,
    YuNetFaceDetector,
    align_face_5pt,
    load_db_npz,
)
from .whole_person_tracker import (
    CenteringResult,
    IdentityState,
    TrackedPerson,
    WholePersonTracker,
)
from .yolox_detector import PersonDetection, YOLOXPersonDetector

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


def _label(
    frame: np.ndarray,
    text: str,
    xy: Tuple[int, int],
    color: Tuple[int, int, int],
    scale: float = 0.65,
    thickness: int = 2,
) -> None:
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, HUD_DARK, thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_background_box(frame: np.ndarray, top_left: Tuple[int, int], bottom_right: Tuple[int, int]) -> None:
    sub = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    if sub.size == 0:
        return
    dim = cv2.addWeighted(sub, 0.45, sub * 0, 0.55, 0)
    frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = dim


def _draw_crosshair(frame: np.ndarray, cx: int, cy: int, color: Tuple[int, int, int], size: int = 16) -> None:
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 2, color, -1, cv2.LINE_AA)


def _draw_frame_border(frame: np.ndarray, color: Tuple[int, int, int], thickness: int = 6) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness, cv2.LINE_AA)


def _draw_confidence_bar(
    frame: np.ndarray,
    origin: Tuple[int, int],
    confidence: float,
    width: int = 160,
    height: int = 14,
    color: Tuple[int, int, int] = HUD_ORANGE,
) -> None:
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
        description="Whole-body recognition, tracking, and Re-ID with 3-state exponential decay."
    )
    parser.add_argument("--name", required=True, help="Exact name enrolled in data/db/face_db.npz.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0 for webcam).")
    parser.add_argument("--width", type=int, default=1280, help="Camera width (default 1280).")
    parser.add_argument("--height", type=int, default=720, help="Camera height (default 720).")
    parser.add_argument("--fullscreen", action=argparse.BooleanOptionalAction, default=True, help="Fullscreen display mode.")
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True, help="Mirror video horizontally.")
    parser.add_argument("--threshold", type=float, default=0.42, help="ArcFace distance cutoff (default 0.42).")
    parser.add_argument("--half-life", type=float, default=2.0, help="Confidence decay half-life in seconds (default 2.0).")
    parser.add_argument("--temp-threshold", type=float, default=0.35, help="Confidence cutoff for TEMPORARY->UNKNOWN (default 0.35).")
    parser.add_argument("--reid-threshold", type=float, default=0.48, help="Body Re-ID similarity threshold (default 0.48).")
    parser.add_argument("--yolox-interval", type=int, default=2, help="Run YOLOX detection every N frames (default 2 for CPU speed).")
    parser.add_argument("--debug", action="store_true", help="Show performance and diagnostic telemetry.")
    parser.add_argument("--mqtt", action=argparse.BooleanOptionalAction, default=False, help="Publish centering data to MQTT (default False).")
    parser.add_argument("--mqtt-broker", type=str, default="127.0.0.1", help="MQTT broker host IP.")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port.")
    parser.add_argument("--mqtt-topic", type=str, default="face_track", help="MQTT topic name.")
    parser.add_argument("--mqtt-mode", choices=["steps", "direction"], default="steps")
    args = parser.parse_args()

    target_name = args.name.strip()
    database = load_db_npz(DB_PATH)
    if not target_name or target_name not in database:
        parser.error(f'Target "{target_name}" was not found in the face database (data/db/face_db.npz).')

    mqtt_pub: Optional[FaceTrackMQTTPublisher] = None
    if args.mqtt:
        mqtt_pub = FaceTrackMQTTPublisher(
            broker_host=args.mqtt_broker,
            broker_port=args.mqtt_port,
            topic=args.mqtt_topic,
        )
        mqtt_pub.connect()
    last_mqtt_msg = "IDLE"

    # Initialize CPU models
    print("[Pipeline] Initializing CPU-optimized detection & Re-ID models...")
    yolox_detector = YOLOXPersonDetector(conf_threshold=0.35)
    yunet_detector = YuNetFaceDetector(score_threshold=0.50)
    arcface_embedder = ArcFaceEmbedderONNX()
    body_extractor = BodyAppearanceExtractor()
    facemesh_detector = FaceMeshDetector()

    matcher = FaceDBMatcher(database, args.threshold)
    person_tracker = WholePersonTracker(
        target_name=target_name,
        half_life_seconds=args.half_life,
        temp_threshold=args.temp_threshold,
        reid_sim_threshold=args.reid_threshold,
    )

    expression_stabilizer = SignalStabilizer(ExpressionSignal.NEUTRAL, 3)
    eyes_stabilizer = SignalStabilizer(EyeSignal.EYES_OPEN, 3)
    blink_counter = BlinkCounter(min_closure_frames=1, min_open_frames=2)
    analyzer = PerSessionAnalyzer()

    capture, _ = open_camera(args.camera, width=args.width, height=args.height)

    window_name = "Identity-Lock | Whole-Body Tracking & Re-ID"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    locked_once = False
    frame_idx = 0
    consecutive_drop_frames = 0
    prev_state: Optional[IdentityState] = None

    # Cached detections for multi-rate scheduling
    cached_person_dets: List[PersonDetection] = []

    # FPS and performance monitoring
    fps_history = deque(maxlen=30)
    last_loop_time = time.perf_counter()

    try:
        while True:
            loop_start = time.perf_counter()
            frame_idx += 1

            ok, frame = capture.read()
            if not ok or frame is None:
                consecutive_drop_frames += 1
                if consecutive_drop_frames > 20:
                    print("[Camera] Stopped providing frames.")
                    break
                continue
            consecutive_drop_frames = 0
            if args.mirror:
                frame = cv2.flip(frame, 1)

            height, width = frame.shape[:2]
            cx, cy = width // 2, height // 2
            now = time.time()

            # Multi-rate inference scheduling:
            # Run YOLOX person detector every N frames (or if no active tracks)
            run_yolox = (frame_idx % args.yolox_interval == 0) or (len(person_tracker.tracks) == 0)
            if run_yolox:
                cached_person_dets = yolox_detector.detect(frame)

            # Face detection via YuNet
            faces, _ = yunet_detector.detect_with_stats(frame, max_faces=4)

            # Face recognition via ArcFace
            face_matches: List[MatchResult] = []
            for face in faces:
                aligned = align_face_5pt(frame, face.kps)
                emb = arcface_embedder.embed(aligned)
                face_matches.append(matcher.match(emb))

            # Update Whole-Person Tracker with Re-ID & Face-to-Body Association
            active_tracks = person_tracker.update(
                frame_bgr=frame,
                person_detections=cached_person_dets,
                face_detections=faces,
                face_matches=face_matches,
                body_extractor=body_extractor,
                current_time=now,
                extract_reid_on_all=True,
            )

            target_track = person_tracker.get_target_track()

            # Target State & Confidence
            if target_track is not None:
                target_state = target_track.state
                confidence = target_track.confidence
                is_ambiguous = target_track.is_ambiguous
                locked_once = True
            else:
                target_state = IdentityState.UNKNOWN
                confidence = 0.0
                is_ambiguous = False

            # Whole-Body Target Centring
            centering: Optional[CenteringResult] = person_tracker.calculate_centering(width, height, target_track)

            # Facial expressions & eye signals (computed only when target face is visible)
            expression = "NOT_DETECTED"
            eyes = "NOT_DETECTED"
            blink_counted_this_frame = False
            blink_label_for_hud: Optional[str] = None

            if target_track is not None and target_track.face_bbox is not None:
                # Find matching face
                fx1, fy1, fx2, fy2 = target_track.face_bbox
                # Run FaceMesh on target face crop for detailed expressions if available
                face_crop = frame[max(0, fy1):min(height, fy2), max(0, fx1):min(width, fx2)]
                mesh_faces = facemesh_detector.detect(frame, max_faces=1) if face_crop.size > 0 else []
                if mesh_faces:
                    observed = analyzer.analyze(mesh_faces[0].landmarks)
                    raw_expr = expression_stabilizer.update(observed.expression)
                    raw_eyes = eyes_stabilizer.update(observed.eyes)
                    blink_counted_this_frame = blink_counter.update(raw_eyes)
                    if blink_counted_this_frame:
                        blink_label_for_hud = blink_counter.last_cycle_label
                    expression = raw_expr.value
                    eyes = raw_eyes.value
                else:
                    expression = "NEUTRAL"
                    eyes = "EYES_OPEN"
            elif target_state == IdentityState.TEMPORARY:
                expression = "(FACE ROTATED AWAY / OCCLUDED)"
                eyes = "(FACE ROTATED AWAY / OCCLUDED)"

            # MQTT Telemetry
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

            # Debug logging
            if args.debug and target_state != prev_state:
                print(
                    f"[frame {frame_idx}] TARGET STATE -> {target_state.value} | "
                    f"conf={confidence:.3f} | reid_sim={target_track.reid_similarity if target_track else 0.0:.2f} | "
                    f"tracks={len(active_tracks)} | gallery={len(person_tracker.target_profile.gallery)}",
                    file=sys.stderr, flush=True,
                )
                prev_state = target_state

            # Visualization & HUD
            view = frame.copy()

            if not is_active_target and locked_once:
                _draw_frame_border(view, HUD_RED)
            elif target_state == IdentityState.TEMPORARY:
                _draw_frame_border(view, HUD_ORANGE, thickness=4)

            # Draw Whole-Person Bounding Boxes
            for track in active_tracks:
                bx1, by1, bx2, by2 = track.body_bbox
                is_target_t = track is target_track

                if track.state == IdentityState.CONFIRMED:
                    box_col = BOX_CONFIRMED
                    state_lbl = f"TRACK {track.track_id:02d} | {track.assigned_name} | CONFIRMED"
                    thick = 3
                elif track.state == IdentityState.TEMPORARY:
                    box_col = BOX_TEMPORARY
                    state_lbl = f"TRACK {track.track_id:02d} | {track.assigned_name} (provisional) | TEMPORARY [C(t)={track.confidence*100:.0f}%]"
                    thick = 2
                else:
                    box_col = BOX_UNKNOWN
                    state_lbl = f"TRACK {track.track_id:02d} | UNKNOWN"
                    thick = 1

                if track.is_ambiguous:
                    state_lbl += " [AMBIGUOUS: CROSSING]"

                # Whole-body rectangle
                cv2.rectangle(view, (bx1, by1), (bx2, by2), box_col, thick)
                _label(view, state_lbl, (bx1 + 5, max(24, by1 - 8)), box_col, 0.55, 2)

                # Face box if visible
                if track.face_bbox is not None:
                    fx1, fy1, fx2, fy2 = track.face_bbox
                    cv2.rectangle(view, (fx1, fy1), (fx2, fy2), (255, 255, 255), 1)

            # Telemetry readout
            hud_color = HUD_CYAN if target_state == IdentityState.CONFIRMED else (
                HUD_YELLOW if target_state == IdentityState.TEMPORARY else HUD_ORANGE
            )

            banner_y_offset = 0
            if target_state == IdentityState.TEMPORARY:
                loss_s = now - target_track.last_confirmed_time if target_track else 0.0
                banner = f"  BODY TRACKING CONTINUITY [FACE AWAY / OCCLUDED {loss_s:.1f}s]  "
                (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                _draw_background_box(view, (10, 105), (10 + tw + 20, 105 + th + 25))
                _label(view, banner, (15, 105 + th + 12), HUD_ORANGE, 0.75, 2)
                banner_y_offset = 32
            elif not is_active_target and locked_once:
                banner = f"  TARGET LOST: IDENTITY UNKNOWN  "
                (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                _draw_background_box(view, (10, 105), (10 + tw + 20, 105 + th + 25))
                _label(view, banner, (15, 105 + th + 12), HUD_RED, 0.75, 2)
                banner_y_offset = 32

            # Status lines
            state_text = f"TARGET = {target_name} [{target_state.value}]"
            _label(view, state_text, (10, 30), hud_color)

            # Confidence bar
            conf_str = f"CONFIDENCE = {confidence * 100:.1f}%  (Half-life: {args.half_life:.1f}s)"
            _label(view, conf_str, (10, 60), hud_color)
            _draw_confidence_bar(view, (460, 48), confidence, width=150, height=14, color=hud_color)

            _label(view, f"EXPRESSION = {expression}", (10, 90), hud_color, 0.60, 2)

            blink_line = f"BLINKS = {blink_counter.total}  (full={blink_counter.full_blinks}  L={blink_counter.winks_left}  R={blink_counter.winks_right})"
            if blink_counter.cycle_highlight and blink_label_for_hud:
                blink_line = f"BLINKS = {blink_counter.total}  [{blink_label_for_hud}!]"
                blink_color = (80, 255, 80)
            else:
                blink_color = hud_color
            _label(view, blink_line, (10, 120 + banner_y_offset), blink_color, 0.65, 2)

            # Centering readout
            cent_y = 150 + banner_y_offset
            if centering is not None:
                cent_text = (
                    f"CENTERING = {centering.direction_text}  "
                    f"dx={centering.offset_px[0]:+4d}px ({centering.normalized_offset[0]:+.2f})  "
                    f"dy={centering.offset_px[1]:+4d}px  dist={centering.distance_px:.0f}px"
                )
                _label(view, cent_text, (10, cent_y), (80, 255, 80) if abs(centering.offset_px[0]) < 25 else hud_color, 0.65, 2)
            else:
                _label(view, "CENTERING = TARGET NOT FOUND", (10, cent_y), HUD_GRAY, 0.65, 2)

            # Gallery status & Re-ID similarity
            gal_y = cent_y + 26
            gal_size = len(person_tracker.target_profile.gallery)
            target_reid_sim = target_track.reid_similarity if target_track else 0.0
            gal_text = f"GALLERY = {gal_size}/12 multi-view prototypes | Target Re-ID Sim: {target_reid_sim:.2f}"
            _label(view, gal_text, (10, gal_y), (200, 200, 200), 0.55, 1)

            # Performance monitoring: FPS and latency
            fps_history.append(1.0 / max(1e-4, time.perf_counter() - loop_start))
            avg_fps = sum(fps_history) / len(fps_history)
            perf_text = (
                f"PERF: {avg_fps:.1f} FPS | YOLOX: {yolox_detector.last_inference_time_ms:.0f}ms | "
                f"YuNet: {yunet_detector.last_inference_time_ms:.0f}ms | Re-ID: {body_extractor.last_inference_time_ms:.0f}ms"
            )
            _label(view, perf_text, (10, gal_y + 24), (160, 255, 160), 0.52, 1)

            # Center Crosshair & Target Vector
            cross_color = HUD_CYAN if target_state == IdentityState.CONFIRMED else (
                HUD_YELLOW if target_state == IdentityState.TEMPORARY else HUD_GRAY
            )
            _draw_crosshair(view, cx, cy, cross_color)

            if centering is not None:
                tx, ty = centering.target_xy
                cv2.line(view, (cx, cy), (tx, ty), (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(view, (tx, ty), 6, (0, 255, 255) if target_state == IdentityState.CONFIRMED else (0, 140, 255), -1, cv2.LINE_AA)

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
            facemesh_detector.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
