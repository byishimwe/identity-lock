"""Whole-Person Tracking with Spatial Face-to-Body Association and Three-State Identity Continuity.

This module implements:
1. Whole-person detection and temporal tracking across webcam frames (CPU-only).
2. Spatial face-to-body mapping based on upper-body geometric containment.
3. Three-state identity model:
   - CONFIRMED: Face identity actively verified via ArcFace biometrics within the body.
   - TEMPORARY: Face temporarily lost (head turn, occlusion), body tracked with exponential confidence decay C(t) = C0 * e^(-lambda * t).
   - UNKNOWN: Unidentified person, decayed confidence, or ambiguous track.
4. Path crossing & occlusion ambiguity detection to prevent false identity swaps.
5. Target centring calculation relative to webcam frame center.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


class IdentityState(str, Enum):
    """Honest three-state identity model."""
    CONFIRMED = "CONFIRMED"
    TEMPORARY = "TEMPORARY"
    UNKNOWN = "UNKNOWN"


@dataclass
class CenteringResult:
    """Target position and offset relative to webcam frame center."""
    target_xy: Tuple[int, int]
    frame_center: Tuple[int, int]
    offset_px: Tuple[int, int]          # (dx, dy) where dx > 0 is target to right of center
    normalized_offset: Tuple[float, float]  # (-1.0 to 1.0)
    distance_px: float
    direction_h: str                    # "CENTER", "LEFT", "RIGHT"
    direction_v: str                    # "CENTER", "UP", "DOWN"

    @property
    def direction_text(self) -> str:
        h = self.direction_h
        v = f"-{self.direction_v}" if self.direction_v != "CENTER" else ""
        return f"{h}{v}"


@dataclass
class TrackedPerson:
    """Represents a person tracked across frames with identity evidence."""
    track_id: int
    body_bbox: Tuple[int, int, int, int]    # (x1, y1, x2, y2)
    state: IdentityState = IdentityState.UNKNOWN
    assigned_name: Optional[str] = None
    confidence: float = 0.0                # Range [0.0, 1.0]
    initial_confidence: float = 1.0        # C0 at moment of last confirmation
    last_confirmed_time: float = 0.0       # Timestamp of last facial confirmation
    last_seen_time: float = 0.0            # Timestamp when body was last updated
    frames_since_confirmed: int = 9999
    frames_since_seen: int = 0
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    face_distance: float = 1.0             # ArcFace cosine distance
    is_ambiguous: bool = False             # True during path crossing / high track overlap
    head_turned: bool = False              # True when body tracked but no face detected in upper region
    occluded: bool = False                 # True when face or body is partially obstructed

    # Velocity estimation for smooth tracking (dx, dy per frame)
    velocity: Tuple[float, float] = (0.0, 0.0)

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.body_bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def upper_body_center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.body_bbox
        h = y2 - y1
        return (x1 + x2) // 2, int(y1 + 0.25 * h)


class ExponentialDecayConfidence:
    """Computes confidence decay C(t) = C0 * exp(-lambda * t).

    Parameters:
        half_life_seconds: Time (s) for confidence to decay to 50% of C0.
        threshold_temp_to_unknown: Confidence floor below which TEMPORARY becomes UNKNOWN.
        lambda_rate: Computed decay rate ln(2) / half_life_seconds.
    """

    def __init__(self, half_life_seconds: float = 2.0, threshold_temp_to_unknown: float = 0.35) -> None:
        if half_life_seconds <= 0:
            raise ValueError("half_life_seconds must be positive.")
        self.half_life_seconds = half_life_seconds
        self.lambda_rate = math.log(2.0) / half_life_seconds
        self.threshold_temp_to_unknown = threshold_temp_to_unknown

    def compute(self, initial_confidence: float, elapsed_seconds: float, ambiguity_multiplier: float = 1.0) -> float:
        """Calculate C(t) = C0 * exp(-lambda * multiplier * t)."""
        if elapsed_seconds <= 0.0:
            return max(0.0, min(1.0, initial_confidence))
        effective_lambda = self.lambda_rate * ambiguity_multiplier
        decayed = initial_confidence * math.exp(-effective_lambda * elapsed_seconds)
        return float(max(0.0, min(1.0, decayed)))

    def is_expired(self, current_confidence: float) -> bool:
        """Returns True if confidence has fallen below the temporary retention threshold."""
        return current_confidence < self.threshold_temp_to_unknown


def compute_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Compute Intersection-over-Union (IoU) of two bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - intersection

    return float(intersection / union) if union > 0 else 0.0


def spatial_face_in_body_score(face_box: Tuple[int, int, int, int], body_box: Tuple[int, int, int, int]) -> float:
    """Calculates spatial compatibility score of a face within a body bounding box.

    A face belongs to a body if:
    1. It is horizontally aligned with the body (face center lies within body x-span + small tolerance).
    2. It is vertically located in the upper 55% of the body box.
    3. The face area intersects significantly with the upper body.

    Returns a score from 0.0 (incompatible) to 1.0 (perfect fit).
    """
    fx1, fy1, fx2, fy2 = face_box
    bx1, by1, bx2, by2 = body_box

    body_w = max(1, bx2 - bx1)
    body_h = max(1, by2 - by1)
    face_w = max(1, fx2 - fx1)
    face_h = max(1, fy2 - fy1)

    face_cx = (fx1 + fx2) / 2.0
    face_cy = (fy1 + fy2) / 2.0

    # Horizontal containment: face center should be within body margins
    h_margin = body_w * 0.15
    if not (bx1 - h_margin <= face_cx <= bx2 + h_margin):
        return 0.0

    # Vertical containment: face should be in upper 55% of body
    v_top_margin = body_h * 0.15
    v_bottom_limit = by1 + body_h * 0.55
    if not (by1 - v_top_margin <= face_cy <= v_bottom_limit):
        return 0.0

    # Intersection over face area
    ix1 = max(fx1, bx1)
    iy1 = max(fy1, by1)
    ix2 = min(fx2, bx2)
    iy2 = min(fy2, by2)
    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    face_area = face_w * face_h
    io_face = inter_area / float(face_area)

    # Upper vertical alignment bonus: face higher up scores higher
    norm_y = (face_cy - by1) / body_h
    vertical_score = max(0.0, 1.0 - norm_y * 1.5)

    return float(0.6 * io_face + 0.4 * vertical_score)


class WholePersonTracker:
    """Maintains whole-person tracks across webcam frames, associates faces,

    and enforces honest three-state identity transitions with exponential decay.
    """

    def __init__(
        self,
        target_name: str,
        half_life_seconds: float = 2.0,
        temp_threshold: float = 0.35,
        max_missing_seconds: float = 6.0,
        iou_track_threshold: float = 0.25,
        ambiguity_iou_threshold: float = 0.20,
    ) -> None:
        self.target_name = target_name
        self.decay_model = ExponentialDecayConfidence(
            half_life_seconds=half_life_seconds,
            threshold_temp_to_unknown=temp_threshold,
        )
        self.max_missing_seconds = max_missing_seconds
        self.iou_track_threshold = iou_track_threshold
        self.ambiguity_iou_threshold = ambiguity_iou_threshold

        self.next_track_id = 1
        self.tracks: Dict[int, TrackedPerson] = {}
        self.last_timestamp: float = time.time()

        # Background HOG detector for full bodies (CPU only)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def _infer_body_from_face(
        self, face_box: Tuple[int, int, int, int], frame_w: int, frame_h: int
    ) -> Tuple[int, int, int, int]:
        """Infers an upper-body/torso bounding box when a face is detected

        (typical webcam head-and-shoulders desktop setup).
        """
        fx1, fy1, fx2, fy2 = face_box
        fw = fx2 - fx1
        fh = fy2 - fy1

        # Torso and shoulders expand ~2.4x face width and ~3.8x face height downward
        bx1 = max(0, int(fx1 - 0.7 * fw))
        bx2 = min(frame_w - 1, int(fx2 + 0.7 * fw))
        by1 = max(0, int(fy1 - 0.2 * fh))
        by2 = min(frame_h - 1, int(fy1 + 3.8 * fh))

        return (bx1, by1, bx2, by2)

    def detect_bodies(
        self,
        frame_bgr: np.ndarray,
        faces: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    ) -> List[Tuple[int, int, int, int]]:
        """Detects person bodies in the frame using HOG people detector and

        face-anchored torso propagation.
        """
        fh, fw = frame_bgr.shape[:2]
        detected_bodies: List[Tuple[int, int, int, int]] = []

        # 1. Face-anchored bodies (always active for webcam portrait / desk views)
        if faces:
            for face in faces:
                inferred = self._infer_body_from_face(face, fw, fh)
                detected_bodies.append(inferred)

        # 2. HOG full-body detection (for standing/walking persons)
        scale = 1.0
        proc_frame = frame_bgr
        if fw > 640:
            scale = 640.0 / fw
            proc_frame = cv2.resize(frame_bgr, (640, int(fh * scale)))

        try:
            boxes, _ = self.hog.detectMultiScale(
                proc_frame,
                winStride=(8, 8),
                padding=(4, 4),
                scale=1.1,
            )
            for (x, y, w, h) in boxes:
                orig_box = (
                    int(x / scale),
                    int(y / scale),
                    int((x + w) / scale),
                    int((y + h) / scale),
                )
                if not any(compute_iou(orig_box, b) > 0.4 for b in detected_bodies):
                    detected_bodies.append(orig_box)
        except Exception:
            pass

        return detected_bodies

    def update(
        self,
        frame_bgr: np.ndarray,
        face_detections: Sequence[Tuple[Tuple[int, int, int, int], Optional[str], float, bool]],
        current_time: Optional[float] = None,
    ) -> List[TrackedPerson]:
        """Updates tracks with new frame observations.

        Parameters:
            frame_bgr: Current video frame.
            face_detections: List of tuples (face_box, matched_name, arcface_distance, is_accepted).
            current_time: Optional timestamp (defaults to time.time()).

        Returns:
            List of active TrackedPerson objects.
        """
        now = time.time() if current_time is None else current_time
        dt = max(1e-4, now - self.last_timestamp)
        self.last_timestamp = now
        fh, fw = frame_bgr.shape[:2]

        face_boxes = [f[0] for f in face_detections]
        detected_bodies = self.detect_bodies(frame_bgr, faces=face_boxes)

        # Predict new positions for existing tracks based on velocity
        for track in self.tracks.values():
            vx, vy = track.velocity
            bx1, by1, bx2, by2 = track.body_bbox
            predicted_box = (
                max(0, min(fw - 1, int(bx1 + vx * dt))),
                max(0, min(fh - 1, int(by1 + vy * dt))),
                max(0, min(fw - 1, int(bx2 + vx * dt))),
                max(0, min(fh - 1, int(by2 + vy * dt))),
            )
            track.body_bbox = predicted_box
            track.frames_since_seen += 1
            track.frames_since_confirmed += 1

        # Associate detected bodies with existing tracks using IoU matching
        matched_track_ids = set()
        matched_body_indices = set()

        track_ids = list(self.tracks.keys())
        matches: List[Tuple[float, int, int]] = []
        for t_idx, t_id in enumerate(track_ids):
            for b_idx, body in enumerate(detected_bodies):
                iou = compute_iou(self.tracks[t_id].body_bbox, body)
                if iou >= self.iou_track_threshold:
                    matches.append((iou, t_id, b_idx))

        matches.sort(key=lambda m: m[0], reverse=True)
        for iou, t_id, b_idx in matches:
            if t_id not in matched_track_ids and b_idx not in matched_body_indices:
                matched_track_ids.add(t_id)
                matched_body_indices.add(b_idx)
                track = self.tracks[t_id]
                old_cx, old_cy = track.center
                track.body_bbox = detected_bodies[b_idx]
                new_cx, new_cy = track.center
                track.velocity = ((new_cx - old_cx) / dt, (new_cy - old_cy) / dt)
                track.last_seen_time = now
                track.frames_since_seen = 0

        # Create new tracks for unmatched bodies
        for b_idx, body in enumerate(detected_bodies):
            if b_idx not in matched_body_indices:
                new_track = TrackedPerson(
                    track_id=self.next_track_id,
                    body_bbox=body,
                    state=IdentityState.UNKNOWN,
                    assigned_name=None,
                    confidence=0.0,
                    initial_confidence=1.0,
                    last_confirmed_time=0.0,
                    last_seen_time=now,
                    frames_since_confirmed=9999,
                    frames_since_seen=0,
                )
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1

        # Check for path crossings / track overlap ambiguity
        track_list = list(self.tracks.values())
        for t in track_list:
            t.is_ambiguous = False
        for i in range(len(track_list)):
            for j in range(i + 1, len(track_list)):
                iou = compute_iou(track_list[i].body_bbox, track_list[j].body_bbox)
                if iou >= self.ambiguity_iou_threshold:
                    track_list[i].is_ambiguous = True
                    track_list[j].is_ambiguous = True

        # Associate detected faces to bodies
        face_matches: List[Tuple[float, int, int]] = []
        for f_idx, (f_box, _, _, _) in enumerate(face_detections):
            for t_id, track in self.tracks.items():
                score = spatial_face_in_body_score(f_box, track.body_bbox)
                if score > 0.3:
                    face_matches.append((score, f_idx, t_id))

        face_matches.sort(key=lambda m: m[0], reverse=True)
        assigned_faces = set()
        assigned_tracks = set()
        track_face_map: Dict[int, int] = {}
        for score, f_idx, t_id in face_matches:
            if f_idx not in assigned_faces and t_id not in assigned_tracks:
                assigned_faces.add(f_idx)
                assigned_tracks.add(t_id)
                track_face_map[t_id] = f_idx

        # Update three-state model and confidence decay for all tracks
        for t_id, track in list(self.tracks.items()):
            if t_id in track_face_map:
                f_idx = track_face_map[t_id]
                f_box, f_name, f_dist, f_accepted = face_detections[f_idx]
                track.face_bbox = f_box
                track.face_distance = f_dist
                track.head_turned = False
                track.occluded = False
                track.last_seen_time = now

                # Did face match the target identity?
                if f_accepted and f_name == self.target_name:
                    # Verified face biometric evidence!
                    track.state = IdentityState.CONFIRMED
                    track.assigned_name = self.target_name
                    c0 = max(0.5, min(1.0, 1.0 - (f_dist * 0.5)))
                    track.initial_confidence = c0
                    track.confidence = c0
                    track.last_confirmed_time = now
                    track.frames_since_confirmed = 0
                elif f_accepted and f_name is not None and f_name != self.target_name:
                    # Confirmed as a different person
                    track.state = IdentityState.UNKNOWN
                    track.assigned_name = f_name
                    track.confidence = 0.0
                else:
                    # Face detected but not verified as target
                    if track.state == IdentityState.CONFIRMED:
                        track.state = IdentityState.TEMPORARY
                    elif track.state == IdentityState.TEMPORARY:
                        elapsed = now - track.last_confirmed_time
                        mult = 2.0 if track.is_ambiguous else 1.0
                        track.confidence = self.decay_model.compute(
                            track.initial_confidence, elapsed, ambiguity_multiplier=mult
                        )
                        if self.decay_model.is_expired(track.confidence) or track.is_ambiguous:
                            track.state = IdentityState.UNKNOWN
            else:
                # No face detected inside this body (Head turned, occluded, or back of head)
                track.face_bbox = None
                track.head_turned = True

                if track.state == IdentityState.CONFIRMED:
                    track.state = IdentityState.TEMPORARY

                if track.state == IdentityState.TEMPORARY:
                    # Body is actively tracked in TEMPORARY state
                    track.last_seen_time = now
                    # Apply exponential confidence decay C(t) = C0 * exp(-lambda * t)
                    elapsed = now - track.last_confirmed_time
                    mult = 2.5 if track.is_ambiguous else 1.0
                    track.confidence = self.decay_model.compute(
                        track.initial_confidence, elapsed, ambiguity_multiplier=mult
                    )
                    # When confidence drops below threshold, transition to UNKNOWN
                    if self.decay_model.is_expired(track.confidence):
                        track.state = IdentityState.UNKNOWN
                        track.assigned_name = None
                        track.confidence = 0.0
                else:
                    track.confidence = 0.0

            # Prune lost tracks
            time_since_seen = now - track.last_seen_time
            if time_since_seen > self.max_missing_seconds:
                del self.tracks[t_id]

        return list(self.tracks.values())

    def get_target_track(self) -> Optional[TrackedPerson]:
        """Returns the primary target track if in CONFIRMED or TEMPORARY state."""
        # Prefer CONFIRMED target
        for track in self.tracks.values():
            if track.state == IdentityState.CONFIRMED and track.assigned_name == self.target_name:
                return track

        # Otherwise look for TEMPORARY target with highest confidence
        temp_candidates = [
            t for t in self.tracks.values()
            if t.state == IdentityState.TEMPORARY and t.assigned_name == self.target_name
        ]
        if temp_candidates:
            temp_candidates.sort(key=lambda t: t.confidence, reverse=True)
            return temp_candidates[0]

        return None

    def calculate_centering(
        self,
        frame_width: int,
        frame_height: int,
        target_track: Optional[TrackedPerson] = None,
    ) -> Optional[CenteringResult]:
        """Calculates tracked target position relative to webcam frame center."""
        track = target_track or self.get_target_track()
        if track is None:
            return None

        # Choose anchor point: face center if available, otherwise upper-body/torso center
        if track.face_bbox is not None:
            fx1, fy1, fx2, fy2 = track.face_bbox
            target_x = (fx1 + fx2) // 2
            target_y = (fy1 + fy2) // 2
        else:
            target_x, target_y = track.upper_body_center

        cx = frame_width // 2
        cy = frame_height // 2

        dx = int(target_x - cx)
        dy = int(target_y - cy)

        norm_dx = float(dx / (frame_width / 2.0))
        norm_dy = float(dy / (frame_height / 2.0))

        dist_px = float(math.hypot(dx, dy))

        dir_h = "CENTER" if abs(dx) < 15 else ("LEFT" if dx < 0 else "RIGHT")
        dir_v = "CENTER" if abs(dy) < 15 else ("UP" if dy < 0 else "DOWN")

        return CenteringResult(
            target_xy=(target_x, target_y),
            frame_center=(cx, cy),
            offset_px=(dx, dy),
            normalized_offset=(norm_dx, norm_dy),
            distance_px=dist_px,
            direction_h=dir_h,
            direction_v=dir_v,
        )
