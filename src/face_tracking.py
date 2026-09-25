"""Lock facial-signal tracking to one exact enrolled identity.

Run: python -m src.face_tracking --name="Winner" [--debug]
"""
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Display facial signals for one enrolled person only.")
    parser.add_argument("--name", required=True, help="Exact name enrolled in data/db/face_db.npz.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_DIST_THRESHOLD,
        help=f"ArcFace distance cutoff (default {DEFAULT_DIST_THRESHOLD:.2f}); higher is more forgiving.",
    )
    parser.add_argument("--stable-frames", type=int, default=3)
    parser.add_argument("--debug", action="store_true", help="Print per-frame diagnostics and show on-screen debug HUD.")
    args = parser.parse_args()

    target_name = args.name.strip()
    database = load_db_npz(DB_PATH)
    if not target_name or target_name not in database:
        parser.error(f'Target "{target_name}" was not found in the face database.')

    detector = FaceMeshDetector()
    embedder = ArcFaceEmbedderONNX()
    matcher = FaceDBMatcher(database, args.threshold)
    expression_stabilizer = SignalStabilizer(ExpressionSignal.NEUTRAL, args.stable_frames)
    eyes_stabilizer = SignalStabilizer(EyeSignal.EYES_OPEN, args.stable_frames)
    blink_counter = BlinkCounter(min_closure_frames=1, min_open_frames=2)
    analyzer = PerSessionAnalyzer()
    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not capture.isOpened():
        detector.close()
        raise RuntimeError("Camera not available. Try --camera 1.")

    locked_once = False
    frame_idx = 0
    prev_detected: Optional[bool] = None
    prev_eyes_value: Optional[str] = None
    prev_expr_value: Optional[str] = None
    temporal_lock = TemporalLock()
    try:
        while True:
            frame_idx += 1
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            cx, cy = width // 2, height // 2
            faces, stats = detector.detect_with_stats(frame)
            matches = _recognize(frame, faces, embedder, matcher) if faces else []
            accepted_index: Optional[int] = next(
                (index for index, match in enumerate(matches) if match.accepted and match.name == target_name), None,
            )
            target_index = accepted_index
            lock_relaxed = False
            if target_index is None and locked_once and faces and temporal_lock.last_accepted_frames_ago <= TEMPORAL_HOLD_FRAMES:
                best_idx: Optional[int] = None
                best_score = args.threshold + TEMPORAL_HOLD_SLACK
                for idx, match in enumerate(matches):
                    if match.distance <= best_score and (match.name in (None, target_name) or True):
                        best_idx = idx
                        best_score = match.distance
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
            expression, eyes = "NOT_DETECTED", "NOT_DETECTED"
            nose_hud = "NOT_DETECTED"
            nose_xy: Optional[Tuple[int, int]] = None
            debug_target_lines: List[str] = ["[target stats: none]"]
            blink_counted_this_frame = False
            blink_label_for_hud: Optional[str] = None
            if detected:
                locked_once = True
                face = faces[target_index]
                observed = analyzer.analyze(face.landmarks)
                raw_expr = expression_stabilizer.update(observed.expression)
                raw_eyes = eyes_stabilizer.update(observed.eyes)
                blink_counted_this_frame = blink_counter.update(raw_eyes)
                if blink_counted_this_frame:
                    blink_label_for_hud = blink_counter.last_cycle_label
                expression = raw_expr.value
                eyes = raw_eyes.value
                nose = nose_position_in_frame(face.landmarks, width, height)
                nose_hud = f"OFFSET ({nose.offset_text})  DIST {nose.distance_px:.0f}px"
                nose_xy = (int(round(face.landmarks[NOSE_TIP_IDX, 0])), int(round(face.landmarks[NOSE_TIP_IDX, 1])))
                pts = np.asarray(face.landmarks, dtype=np.float32)
                left_ear = eye_aspect_ratio(pts, (33, 160, 158, 133, 153, 144))
                right_ear = eye_aspect_ratio(pts, (362, 385, 387, 263, 373, 380))
                match = matches[target_index]
                relaxed_tag = " [RELAXED]" if lock_relaxed else ""
                baseline = analyzer.eye_baseline
                left_base = baseline.left_baseline()
                right_base = baseline.right_baseline()
                _, _, asym_thr = baseline.closed_threshold()
                debug_target_lines = [
                    f"EAR L={left_ear:.3f}(baseline {left_base:.3f} ≤{observed.closed_threshold_ear:.3f}=closed)  "
                    f"R={right_ear:.3f}(base {right_base:.3f})  asym_thr={asym_thr:.3f}",
                    f"eyeDist={observed.eye_distance_px:.1f}px  mouth MAR={observed.mouth_mar:.3f}  "
                    f"width/eye={observed.mouth_width_ratio:.3f}",
                    f"match name={match.name or 'None'}  d={match.distance:.4f}  "
                    f"thr={args.threshold:.2f}  acc={'T' if match.accepted else 'F'}{relaxed_tag}",
                    f"temporal: last_acc={temporal_lock.last_accepted_frames_ago} frames ago  d_best={temporal_lock.best_distance:.4f}",
                ]

            debug_match_lines = ["[all faces: 0]"] if not faces else []
            for i, (face, match) in enumerate(zip(faces, matches)):
                is_target = i == target_index
                is_accepted_tag = "YES" if (match.accepted and match.name == target_name) else (
                    "NAME" if (match.accepted and match.name != target_name) else ("DIST" if match.distance <= args.threshold + TEMPORAL_HOLD_SLACK else "NO"))
                debug_match_lines.append(
                    f"face#{i} box=({face.x1},{face.y1})-({face.x2},{face.y2}) "
                    f"match={match.name or 'Unknown'} d={match.distance:.3f} ok={is_accepted_tag}"
                    f"{' [TARGET]' if is_target else ''}"
                )

            if args.debug:
                if detected != prev_detected:
                    print(
                        f"[frame {frame_idx}] DETECTION {'GAINED' if detected else 'LOST'} | "
                        f"faces={stats.describe()} | "
                        f"matches={len(matches)} | target_index={target_index} | relaxed={lock_relaxed} | "
                        f"temporal_last={temporal_lock.last_accepted_frames_ago}",
                        file=sys.stderr, flush=True,
                    )
                if detected and (eyes != prev_eyes_value or expression != prev_expr_value):
                    print(
                        f"[frame {frame_idx}] SIGNAL expr={expression} eyes={eyes} | "
                        + " | ".join(debug_target_lines),
                        file=sys.stderr, flush=True,
                    )
                if blink_counted_this_frame:
                    print(
                        f"[frame {frame_idx}] BLINK COUNTED total={blink_counter.total} "
                        f"(full={blink_counter.full_blinks} Lwink={blink_counter.winks_left} "
                        f"Rwink={blink_counter.winks_right}) kind={blink_label_for_hud!r}",
                        file=sys.stderr, flush=True,
                    )
                prev_detected, prev_eyes_value, prev_expr_value = detected, eyes, expression

            view = frame.copy()
            warn = _warning_color(locked_once, detected)
            if warn is not None:
                _draw_frame_border(view, warn)

            for index, (face, match) in enumerate(zip(faces, matches)):
                is_target = index == target_index
                if is_target:
                    color = BOX_TARGET
                else:
                    color = BOX_OTHER_KNOWN if match.accepted else BOX_UNKNOWN
                label = match.name if match.accepted else "Unknown"
                if is_target:
                    label += " [LOCKED]"
                    if lock_relaxed:
                        label += " *"
                cv2.rectangle(view, (face.x1, face.y1), (face.x2, face.y2), color, 3 if is_target else 2)
                _label(view, f"{label} d={match.distance:.3f}", (face.x1, max(24, face.y1 - 10)), color, 0.6, 2)

            hud_color = _hud_color(locked_once, detected)
            state = "LOCKED" if locked_once else "WAITING"
            banner_y_offset = 0
            if warn is not None:
                if locked_once:
                    banner = f"  WARNING: TARGET LOST  "
                else:
                    banner = f"  WARNING: LOOKING FOR TARGET  "
                (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 3)
                bx, by = 10, 105
                _draw_background_box(view, (bx, by), (bx + tw + 30, by + th + 30))
                _label(view, banner, (bx + 15, by + th + 15), warn, 0.85, 3)
                banner_y_offset = 30

            _label(view, f"TARGET = {target_name} ({state})", (10, 30), hud_color)
            _label(view, f"EXPRESSION = {expression}", (10, 60), hud_color)
            _label(view, f"EYES = {eyes}", (10, 90), hud_color)
            blink_line = f"BLINKS = {blink_counter.total}  (full={blink_counter.full_blinks}  L={blink_counter.winks_left}  R={blink_counter.winks_right})"
            if blink_counter.cycle_highlight and blink_label_for_hud:
                blink_line = f"BLINKS = {blink_counter.total}  [{blink_label_for_hud}!]"
                blink_color = (80, 255, 80)
                blink_scale, blink_thick = 0.85, 3
            else:
                blink_color = hud_color
                blink_scale, blink_thick = 0.75, 2
            _label(view, blink_line, (10, 120), blink_color, blink_scale, blink_thick)
            nose_y = 170 if warn is None else 200
            _label(view, f"NOSE = {nose_hud}", (10, nose_y + banner_y_offset), hud_color, 0.65, 2)

            if args.debug:
                det_line = (
                    f"DET: {stats.describe()}   matches: {len(matches)}  "
                    f"thr={args.threshold:.2f}+{TEMPORAL_HOLD_SLACK:.2f}(hold {TEMPORAL_HOLD_FRAMES}f)"
                )
                _label(view, det_line, (10, nose_y + 30 + banner_y_offset), HUD_WHITE, 0.52, 1)
                for i, line in enumerate(debug_target_lines):
                    _label(view, line, (10, nose_y + 52 + i * 22 + banner_y_offset), HUD_WHITE, 0.5, 1)
                blink_debug = (
                    f"blink state: in_closure={blink_counter._in_closure}  "
                    f"close_frames={blink_counter._closure_seen_frames}  "
                    f"open_frames={blink_counter._open_seen_frames}"
                )
                extra_y = nose_y + 52 + len(debug_target_lines) * 22 + banner_y_offset
                _label(view, blink_debug, (10, extra_y), HUD_WHITE, 0.48, 1)
                matches_y = extra_y + 22
                for i, line in enumerate(debug_match_lines[:6]):
                    _label(view, line, (10, matches_y + i * 20), HUD_WHITE, 0.46, 1)

            cross_color = HUD_CYAN if detected else (HUD_YELLOW if not locked_once else HUD_ORANGE)
            _draw_crosshair(view, cx, cy, cross_color)
            if nose_xy is not None:
                cv2.line(view, (cx, cy), nose_xy, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(view, nose_xy, 4, (255, 80, 80), -1, cv2.LINE_AA)

            _label(view, "Q = quit" + ("  [DEBUG]" if args.debug else ""), (10, view.shape[0] - 15), (220, 220, 220), 0.55, 1)
            cv2.imshow("FaceLocking", view)
            try:
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q")):
                    break
            except KeyboardInterrupt:
                break
    finally:
        try:
            capture.release()
        finally:
            detector.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
