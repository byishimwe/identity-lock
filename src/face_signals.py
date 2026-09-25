"""Facial-expression and eye-state classification from FaceMesh landmarks."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Generic, Optional, Sequence, Tuple, TypeVar

import numpy as np


class ExpressionSignal(str, Enum):
    NEUTRAL = "NEUTRAL"
    SMILE = "SMILE"
    FROWN = "FROWN"
    SADNESS = "SADNESS"
    GRIMACE = "GRIMACE"


class EyeSignal(str, Enum):
    EYES_OPEN = "EYES_OPEN"
    EYES_CLOSED = "EYES_CLOSED"
    WINK_LEFT = "WINK_LEFT"
    WINK_RIGHT = "WINK_RIGHT"
    BLINKING = "BLINKING"


@dataclass(frozen=True)
class FacialSignals:
    expression: ExpressionSignal
    eyes: EyeSignal
    eye_distance_px: float
    left_ear: float
    right_ear: float
    mouth_mar: float
    mouth_width_ratio: float
    closed_threshold_ear: float
    eye_reason: str


SMILE_WIDTH_THRESHOLD = 0.64
SMILE_REFERENCE_EYE_DISTANCE_PX = 90.0
SMILE_FAR_FACE_TOLERANCE = 0.06
STABLE_FRAMES = 3

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
MOUTH_CORNERS = (61, 291)
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
LEFT_INNER_BROW = 107
RIGHT_INNER_BROW = 336
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
NOSE_TIP = 1

BASELINE_WINDOW = 60
CLOSED_RATIO_OF_BASELINE = 0.86
WINK_ASYM_RATIO_OF_BASELINE = 0.10
MIN_EAR_ABSOLUTE_FLOOR = 0.12

SMILE_CORNER_LIFT_MIN = 0.006
FROWN_CORNER_DROP_RATIO = 0.07
SAD_BROW_RAISE_RATIO = 0.14
SAD_MOUTH_DROP_RATIO = 0.03
GRIMACE_MAR_MIN = 0.28
GRIMACE_WIDTH_MIN = 0.61


def _distance(points: np.ndarray, first: int, second: int) -> float:
    return float(np.linalg.norm(points[first] - points[second]))


def eye_aspect_ratio(points: np.ndarray, indices: Sequence[int]) -> float:
    a = _distance(points, indices[1], indices[5])
    b = _distance(points, indices[2], indices[4])
    c = _distance(points, indices[0], indices[3])
    return (a + b) / (2.0 * c) if c > 1e-6 else 0.0


def mouth_aspect_ratio(points: np.ndarray) -> float:
    height = _distance(points, MOUTH_TOP, MOUTH_BOTTOM)
    width = _distance(points, *MOUTH_CORNERS)
    return height / max(width, 1e-6)


def smile_width_threshold(eye_distance: float) -> float:
    if eye_distance >= SMILE_REFERENCE_EYE_DISTANCE_PX:
        return SMILE_WIDTH_THRESHOLD
    fraction = 1.0 - max(0.0, eye_distance) / SMILE_REFERENCE_EYE_DISTANCE_PX
    return SMILE_WIDTH_THRESHOLD - SMILE_FAR_FACE_TOLERANCE * fraction


def _mouth_corner_vertical_offset(points: np.ndarray, eye_distance: float) -> float:
    mouth_top_y = points[MOUTH_TOP, 1]
    return (mouth_top_y - (points[MOUTH_CORNERS[0], 1] + points[MOUTH_CORNERS[1], 1]) / 2.0) / max(eye_distance, 1e-6)


def _inner_brow_raise_over_eye(points: np.ndarray, eye_distance: float) -> float:
    left_brow_y = points[LEFT_INNER_BROW, 1]
    right_brow_y = points[RIGHT_INNER_BROW, 1]
    left_eye_y = points[LEFT_EYE[0], 1]
    right_eye_y = points[RIGHT_EYE[0], 1]
    brow_y = (left_brow_y + right_brow_y) / 2.0
    eye_y = (left_eye_y + right_eye_y) / 2.0
    return (eye_y - brow_y) / max(eye_distance, 1e-6)


def _classify_expression(points: np.ndarray, eye_distance: float, mar: float, width_ratio: float) -> ExpressionSignal:
    corner_y_ratio = _mouth_corner_vertical_offset(points, eye_distance)
    brow_raise = _inner_brow_raise_over_eye(points, eye_distance)

    smile_threshold = smile_width_threshold(eye_distance)
    if width_ratio >= smile_threshold and corner_y_ratio >= SMILE_CORNER_LIFT_MIN:
        return ExpressionSignal.SMILE

    if mar >= GRIMACE_MAR_MIN and width_ratio >= GRIMACE_WIDTH_MIN and corner_y_ratio > -0.02:
        return ExpressionSignal.GRIMACE

    if corner_y_ratio <= -FROWN_CORNER_DROP_RATIO and brow_raise < SAD_BROW_RAISE_RATIO * 0.4:
        return ExpressionSignal.FROWN

    if brow_raise >= SAD_BROW_RAISE_RATIO and corner_y_ratio <= -SAD_MOUTH_DROP_RATIO:
        return ExpressionSignal.SADNESS

    return ExpressionSignal.NEUTRAL


@dataclass
class EyeBaselineTracker:
    """Tracks a recent window of "eyes open" EAR values to derive a person-specific closed threshold."""

    window_size: int = BASELINE_WINDOW
    closed_ratio: float = CLOSED_RATIO_OF_BASELINE
    asym_threshold_ratio: float = WINK_ASYM_RATIO_OF_BASELINE
    _left_open: Deque[float] = field(default_factory=lambda: deque(maxlen=BASELINE_WINDOW))
    _right_open: Deque[float] = field(default_factory=lambda: deque(maxlen=BASELINE_WINDOW))

    def _percentile(self, values: Deque[float], pct: float) -> float:
        if not values:
            return 0.0
        arr = np.asarray(values, dtype=np.float32)
        return float(np.percentile(arr, int(pct * 100)))

    def left_baseline(self) -> float:
        return max(self._percentile(self._left_open, 0.75), MIN_EAR_ABSOLUTE_FLOOR)

    def right_baseline(self) -> float:
        return max(self._percentile(self._right_open, 0.75), MIN_EAR_ABSOLUTE_FLOOR)

    def closed_threshold(self) -> Tuple[float, float, float]:
        left = self.left_baseline() * self.closed_ratio
        right = self.right_baseline() * self.closed_ratio
        asym = (self.left_baseline() + self.right_baseline()) * 0.5 * self.asym_threshold_ratio
        return left, right, max(asym, 0.02)

    def update(self, left_ear: float, right_ear: float, looks_open: bool) -> None:
        if looks_open:
            self._left_open.append(left_ear)
            self._right_open.append(right_ear)


def _classify_eyes(
    left_ear: float,
    right_ear: float,
    baseline: EyeBaselineTracker,
) -> Tuple[EyeSignal, float, str]:
    left_thr, right_thr, asym_thr = baseline.closed_threshold()
    left_closed = left_ear <= left_thr
    right_closed = right_ear <= right_thr
    asym = abs(left_ear - right_ear)

    reason = (
        f"L={left_ear:.3f}/{left_thr:.3f}{'C' if left_closed else 'O'} "
        f"R={right_ear:.3f}/{right_thr:.3f}{'C' if right_closed else 'O'} "
        f"asym={asym:.3f}/{asym_thr:.3f}"
    )

    if left_closed and right_closed:
        if asym <= asym_thr * 0.8:
            return EyeSignal.EYES_CLOSED, (left_thr + right_thr) * 0.5, reason + " -> BOTH_CLOSED_SYMM"
        return EyeSignal.BLINKING, (left_thr + right_thr) * 0.5, reason + " -> BOTH_CLOSED_ASYM"
    if left_closed and not right_closed and asym >= asym_thr and left_ear <= left_thr:
        return EyeSignal.WINK_LEFT, (left_thr + right_thr) * 0.5, reason + " -> ONLY_LEFT_CLOSED"
    if right_closed and not left_closed and asym >= asym_thr and right_ear <= right_thr:
        return EyeSignal.WINK_RIGHT, (left_thr + right_thr) * 0.5, reason + " -> ONLY_RIGHT_CLOSED"
    if (not left_closed and not right_closed) and asym >= asym_thr * 0.6:
        return EyeSignal.BLINKING, (left_thr + right_thr) * 0.5, reason + " -> OPEN_ASYM"
    return EyeSignal.EYES_OPEN, (left_thr + right_thr) * 0.5, reason + " -> BOTH_OPEN"


@dataclass
class PerSessionAnalyzer:
    eye_baseline: EyeBaselineTracker = field(default_factory=EyeBaselineTracker)

    def analyze(self, face_landmarks: np.ndarray) -> FacialSignals:
        points = np.asarray(face_landmarks, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) <= max(RIGHT_EYE):
            raise ValueError("Expected MediaPipe FaceMesh landmarks with x/y coordinates.")
        eye_distance = _distance(points, LEFT_EYE_OUTER, RIGHT_EYE_OUTER)
        left_ear = eye_aspect_ratio(points, LEFT_EYE)
        right_ear = eye_aspect_ratio(points, RIGHT_EYE)
        baseline = self.eye_baseline
        eyes, closed_thr, eye_reason = _classify_eyes(left_ear, right_ear, baseline)
        looks_open = eyes is EyeSignal.EYES_OPEN
        baseline.update(left_ear, right_ear, looks_open=looks_open)
        if eyes in (EyeSignal.BLINKING,) and (
            self.eye_baseline.left_baseline() < MIN_EAR_ABSOLUTE_FLOOR * 1.5
            or self.eye_baseline.right_baseline() < MIN_EAR_ABSOLUTE_FLOOR * 1.5
        ):
            eyes = EyeSignal.EYES_OPEN
            eye_reason = eye_reason + " [OVERRIDE: low baseline -> OPEN]"
        mar = mouth_aspect_ratio(points)
        mouth_width = _distance(points, *MOUTH_CORNERS)
        width_ratio = mouth_width / max(eye_distance, 1e-6)
        expression = _classify_expression(points, eye_distance, mar, width_ratio)
        return FacialSignals(
            expression=expression,
            eyes=eyes,
            eye_distance_px=eye_distance,
            left_ear=left_ear,
            right_ear=right_ear,
            mouth_mar=mar,
            mouth_width_ratio=width_ratio,
            closed_threshold_ear=closed_thr,
            eye_reason=eye_reason,
        )


def detect_face_signals(face_landmarks: np.ndarray) -> FacialSignals:
    """Backwards-compatible convenience: uses a module-level analyzer state."""
    return _default_analyzer.analyze(face_landmarks)


_default_analyzer = PerSessionAnalyzer()


@dataclass(frozen=True)
class NosePosition:
    offset_px: Tuple[int, int]
    direction: str
    distance_px: float

    @property
    def offset_text(self) -> str:
        dx, dy = self.offset_px
        h = "CENTER" if abs(dx) < 6 else ("LEFT" if dx < 0 else "RIGHT")
        v = "" if abs(dy) < 6 else ("-UP" if dy < 0 else "-DOWN")
        return h + v


def nose_position_in_frame(face_landmarks: np.ndarray, frame_width: int, frame_height: int) -> NosePosition:
    points = np.asarray(face_landmarks, dtype=np.float32)
    nose = points[NOSE_TIP]
    cx, cy = frame_width / 2.0, frame_height / 2.0
    dx = float(nose[0] - cx)
    dy = float(nose[1] - cy)
    return NosePosition(
        offset_px=(int(round(dx)), int(round(dy))),
        direction="",
        distance_px=math.hypot(dx, dy),
    )


SignalType = TypeVar("SignalType", bound=Enum)


class SignalStabilizer(Generic[SignalType]):
    def __init__(self, initial_signal: SignalType, stable_frames: int = STABLE_FRAMES):
        if stable_frames < 1:
            raise ValueError("stable_frames must be at least 1")
        self.current = initial_signal
        self.stable_frames = stable_frames
        self._candidate: Optional[SignalType] = None
        self._candidate_frames = 0

    def update(self, observed: SignalType) -> SignalType:
        if observed == self.current:
            self._candidate, self._candidate_frames = None, 0
        elif observed == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate, self._candidate_frames = observed, 1
        if self._candidate_frames >= self.stable_frames:
            self.current = observed
            self._candidate, self._candidate_frames = None, 0
        return self.current


class BlinkCounter:
    """Count full blink / wink cycles: any eye closure followed by a return to fully open."""

    OPEN_STATES = frozenset({EyeSignal.EYES_OPEN})
    CLOSURE_STATES = frozenset(
        {EyeSignal.EYES_CLOSED, EyeSignal.WINK_LEFT, EyeSignal.WINK_RIGHT, EyeSignal.BLINKING}
    )

    def __init__(self, min_closure_frames: int = 1, min_open_frames: int = 2) -> None:
        self.total = 0
        self.winks_left = 0
        self.winks_right = 0
        self.full_blinks = 0
        self.last_closure: Optional[EyeSignal] = None
        self._in_closure = False
        self._closure_seen_frames = 0
        self._open_seen_frames = 0
        self._min_closure_frames = max(1, min_closure_frames)
        self._min_open_frames = max(1, min_open_frames)
        self._last_cycle_label: Optional[str] = None
        self._cycle_hot_frames = 0
        self._hot_ttl = 8

    def update(self, stabilized_eyes: EyeSignal) -> bool:
        """Returns True when a new blink/wink cycle was just counted."""
        triggered = False
        if stabilized_eyes in self.CLOSURE_STATES:
            self._closure_seen_frames += 1
            self._open_seen_frames = 0
            if self._closure_seen_frames >= self._min_closure_frames and not self._in_closure:
                self._in_closure = True
                self.last_closure = stabilized_eyes
        elif stabilized_eyes in self.OPEN_STATES:
            self._open_seen_frames += 1
            if self._in_closure and self._open_seen_frames >= self._min_open_frames:
                triggered = True
                closure = self.last_closure
                if closure is EyeSignal.WINK_LEFT:
                    self.winks_left += 1
                    self._last_cycle_label = "WINK LEFT"
                elif closure is EyeSignal.WINK_RIGHT:
                    self.winks_right += 1
                    self._last_cycle_label = "WINK RIGHT"
                elif closure is EyeSignal.EYES_CLOSED:
                    self.full_blinks += 1
                    self._last_cycle_label = "FULL BLINK"
                else:
                    self.full_blinks += 1
                    self._last_cycle_label = "BLINK"
                self.total += 1
                self._cycle_hot_frames = self._hot_ttl
                self._in_closure = False
                self._closure_seen_frames = 0
        if self._cycle_hot_frames > 0:
            self._cycle_hot_frames -= 1
        return triggered

    @property
    def cycle_highlight(self) -> bool:
        """True for a handful of frames after a new blink is counted (for HUD flash)."""
        return self._cycle_hot_frames > 0

    @property
    def last_cycle_label(self) -> Optional[str]:
        return self._last_cycle_label if self.cycle_highlight else None
