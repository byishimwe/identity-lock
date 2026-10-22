"""Intelligent Whole-Body Recognition, Tracking, and Person Re-Identification Module.

Integrates:
1. YOLOX whole-person detections and multi-object tracking.
2. YuNet face detection & ArcFace identity recognition.
3. Robust spatial face-to-body association.
4. Multi-modal body appearance learning across 180-degree rotations.
5. Appearance-aware tracking and person Re-ID when faces are invisible.
6. Three-state identity management (CONFIRMED, TEMPORARY, UNKNOWN) with exponential confidence decay.
7. Anti-confusion protection against misidentifying other back-facing people.
8. Whole-body target centring with jitter smoothing.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .appearance_profile import IdentityAppearanceProfile
from .body_features import BodyAppearanceExtractor, BodyObservation
from .recognition import FaceDet, MatchResult
from .yolox_detector import PersonDetection


class IdentityState(str, Enum):
    """Honest three-state identity model."""
    CONFIRMED = "CONFIRMED"
    TEMPORARY = "TEMPORARY"
    UNKNOWN = "UNKNOWN"


class TrackLifecycle(str, Enum):
    """Physical multi-object tracking state."""
    TENTATIVE = "TENTATIVE"
    ACTIVE = "ACTIVE"
    MISSING = "MISSING"
    TERMINATED = "TERMINATED"


@dataclass
class CenteringResult:
    """Whole-body target position and centring offsets relative to frame center."""
    target_xy: Tuple[int, int]
    frame_center: Tuple[int, int]
    offset_px: Tuple[int, int]              # (dx, dy)
    normalized_offset: Tuple[float, float]  # (-1.0 to 1.0)
    distance_px: float
    direction_h: str                        # "CENTER", "LEFT", "RIGHT"
    direction_v: str                        # "CENTER", "UP", "DOWN"

    @property
    def direction_text(self) -> str:
        h = self.direction_h
        v = f"-{self.direction_v}" if self.direction_v != "CENTER" else ""
        return f"{h}{v}"


@dataclass
class TrackedPerson:
    """Represents a physically tracked body and its associated identity evidence."""
    track_id: int
    body_bbox: Tuple[int, int, int, int]    # (x1, y1, x2, y2)
    lifecycle: TrackLifecycle = TrackLifecycle.TENTATIVE
    state: IdentityState = IdentityState.UNKNOWN
    assigned_name: Optional[str] = None
    confidence: float = 0.0                 # Exponentially decayed confidence [0.0, 1.0]
    initial_confidence: float = 1.0         # C0 at last facial confirmation

    # Timestamps
    created_time: float = 0.0
    last_seen_time: float = 0.0             # Updated ONLY on real detection
    last_confirmed_time: float = 0.0        # Updated ONLY on ArcFace face match
    time_since_last_face: float = 9999.0

    # Observation evidence
    detection_score: float = 0.0
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    face_distance: float = 1.0
    latest_body_obs: Optional[BodyObservation] = None
    reid_similarity: float = 0.0            # Similarity against target gallery

    # Motion & geometry
    velocity: Tuple[float, float] = (0.0, 0.0)
    smoothed_center: Optional[Tuple[float, float]] = None
    consecutive_hits: int = 1
    consecutive_misses: int = 0
    is_ambiguous: bool = False              # True during path crossing / overlapping tracks
    is_predicted: bool = False              # True when coasted without detection

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.body_bbox
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def width(self) -> int:
        return max(1, self.body_bbox[2] - self.body_bbox[0])

    @property
    def height(self) -> int:
        return max(1, self.body_bbox[3] - self.body_bbox[1])


class ExponentialDecayModel:
    """Computes identity confidence decay C(t) = C0 * exp(-lambda * delta_t)."""

    def __init__(self, half_life_seconds: float = 2.0, threshold_temp_to_unknown: float = 0.35) -> None:
        self.half_life_seconds = max(0.1, half_life_seconds)
        self.lambda_rate = math.log(2.0) / self.half_life_seconds
        self.threshold_temp_to_unknown = threshold_temp_to_unknown

    def compute(self, c0: float, delta_t: float, ambiguity_multiplier: float = 1.0) -> float:
        if delta_t <= 0.0:
            return float(max(0.0, min(1.0, c0)))
        decay = c0 * math.exp(-self.lambda_rate * ambiguity_multiplier * delta_t)
        return float(max(0.0, min(1.0, decay)))

    def is_expired(self, current_confidence: float) -> bool:
        return current_confidence < self.threshold_temp_to_unknown


def compute_box_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    """Computes IoU between two bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def spatial_face_in_body(face_box: Tuple[int, int, int, int], body_box: Tuple[int, int, int, int]) -> float:
    """Computes spatial face-to-body compatibility score.

    Requires face center to be horizontally aligned and in the upper 50% of the body.
    """
    fx1, fy1, fx2, fy2 = face_box
    bx1, by1, bx2, by2 = body_box

    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)
    fw = max(1, fx2 - fx1)
    fh = max(1, fy2 - fy1)

    fcx = (fx1 + fx2) / 2.0
    fcy = (fy1 + fy2) / 2.0

    # Horizontal containment: face center must be within body box with small margin
    h_margin = bw * 0.15
    if not (bx1 - h_margin <= fcx <= bx2 + h_margin):
        return 0.0

    # Vertical containment: face must reside in upper 50% of the body
    v_top_margin = bh * 0.15
    v_bottom_limit = by1 + bh * 0.50
    if not (by1 - v_top_margin <= fcy <= v_bottom_limit):
        return 0.0

    # Intersection over face area
    ix1 = max(fx1, bx1)
    iy1 = max(fy1, by1)
    ix2 = min(fx2, bx2)
    iy2 = min(fy2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    face_area = fw * fh
    io_face = inter / float(face_area)

    return float(max(0.0, min(1.0, io_face)))


class WholePersonTracker:
    """Intelligent whole-person tracking and person re-identification system."""

    def __init__(
        self,
        target_name: str,
        half_life_seconds: float = 2.0,
        temp_threshold: float = 0.35,
        reid_sim_threshold: float = 0.48,
        max_missing_seconds: float = 2.5,
        ambiguity_iou_threshold: float = 0.20,
    ) -> None:
        self.target_name = target_name
        self.decay_model = ExponentialDecayModel(half_life_seconds, temp_threshold)
        self.reid_sim_threshold = reid_sim_threshold
        self.max_missing_seconds = max_missing_seconds
        self.ambiguity_iou_threshold = ambiguity_iou_threshold

        self.next_track_id = 1
        self.tracks: Dict[int, TrackedPerson] = {}
        self.target_profile = IdentityAppearanceProfile(identity_name=target_name, max_gallery_size=12)
        self.last_update_time = time.time()

        # Target center smoothing filter
        self._target_ema_center: Optional[Tuple[float, float]] = None
        self._ema_alpha = 0.65

    def update(
        self,
        frame_bgr: np.ndarray,
        person_detections: Sequence[PersonDetection],
        face_detections: Sequence[FaceDet],
        face_matches: Sequence[MatchResult],
        body_extractor: BodyAppearanceExtractor,
        current_time: Optional[float] = None,
        extract_reid_on_all: bool = True,
    ) -> List[TrackedPerson]:
        """Main update step executing multi-object tracking, spatial association,

        appearance learning, and 3-state identity continuity.
        """
        now = time.time() if current_time is None else current_time
        dt = max(1e-4, min(1.0, now - self.last_update_time))
        self.last_update_time = now
        fh, fw = frame_bgr.shape[:2]

        # 1. Motion prediction: coast existing tracks
        for track in self.tracks.values():
            vx, vy = track.velocity
            bx1, by1, bx2, by2 = track.body_bbox
            pred_x1 = max(0, min(fw - 1, int(bx1 + vx * dt)))
            pred_y1 = max(0, min(fh - 1, int(by1 + vy * dt)))
            pred_x2 = max(0, min(fw - 1, int(bx2 + vx * dt)))
            pred_y2 = max(0, min(fh - 1, int(by2 + vy * dt)))
            track.body_bbox = (pred_x1, pred_y1, pred_x2, pred_y2)
            track.is_predicted = True

        # 2. Extract appearance observations for detections if requested
        detection_obs: List[Optional[BodyObservation]] = []
        if extract_reid_on_all:
            for det in person_detections:
                obs = body_extractor.extract_observation(frame_bgr, det.box, timestamp=now)
                detection_obs.append(obs)
        else:
            detection_obs = [None] * len(person_detections)

        # 3. Data Association: Match person detections to active tracks
        matched_track_ids = set()
        matched_det_indices = set()

        if self.tracks and person_detections:
            track_ids = list(self.tracks.keys())
            cost_matrix = np.full((len(track_ids), len(person_detections)), 10.0, dtype=np.float32)

            for i, tid in enumerate(track_ids):
                track = self.tracks[tid]
                tcx, tcy = track.center
                for j, det in enumerate(person_detections):
                    iou = compute_box_iou(track.body_bbox, det.box)
                    dcx, dcy = det.center
                    dist = math.hypot(tcx - dcx, tcy - dcy)
                    diag = math.hypot(fw, fh)
                    norm_dist = dist / diag

                    # Re-ID similarity bonus if available
                    reid_bonus = 0.0
                    obs = detection_obs[j]
                    if obs is not None and track.latest_body_obs is not None:
                        sim = BodyAppearanceExtractor.compare_observations(obs, track.latest_body_obs)
                        reid_bonus = sim

                    # Gating: must have reasonable IoU or close spatial proximity
                    if iou < 0.08 and norm_dist > 0.18:
                        continue

                    cost = 0.55 * (1.0 - iou) + 0.25 * norm_dist - 0.20 * reid_bonus
                    cost_matrix[i, j] = cost

            # Greedy assignment
            while True:
                min_val = np.min(cost_matrix)
                if min_val >= 1.0:
                    break
                min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
                i, j = min_idx
                tid = track_ids[i]
                matched_track_ids.add(tid)
                matched_det_indices.add(j)

                # Update matched track with new detection
                track = self.tracks[tid]
                old_cx, old_cy = track.center
                new_box = person_detections[j].box
                track.body_bbox = new_box
                new_cx, new_cy = (new_box[0] + new_box[2]) // 2, (new_box[1] + new_box[3]) // 2
                track.velocity = ((new_cx - old_cx) / dt, (new_cy - old_cy) / dt)
                track.detection_score = person_detections[j].score
                track.last_seen_time = now
                track.consecutive_hits += 1
                track.consecutive_misses = 0
                track.is_predicted = False
                if detection_obs[j] is not None:
                    track.latest_body_obs = detection_obs[j]
                if track.consecutive_hits >= 2:
                    track.lifecycle = TrackLifecycle.ACTIVE

                cost_matrix[i, :] = 10.0
                cost_matrix[:, j] = 10.0

        # 4. Handle unmatched tracks
        for tid, track in list(self.tracks.items()):
            if tid not in matched_track_ids:
                track.consecutive_misses += 1
                track.consecutive_hits = 0
                track.lifecycle = TrackLifecycle.MISSING
                # Check expiration
                if now - track.last_seen_time > self.max_missing_seconds:
                    track.lifecycle = TrackLifecycle.TERMINATED
                    del self.tracks[tid]

        # 5. Create new tracks for unmatched high-confidence detections
        for j, det in enumerate(person_detections):
            if j not in matched_det_indices and det.score >= 0.40:
                new_t = TrackedPerson(
                    track_id=self.next_track_id,
                    body_bbox=det.box,
                    lifecycle=TrackLifecycle.TENTATIVE,
                    state=IdentityState.UNKNOWN,
                    assigned_name=None,
                    confidence=0.0,
                    initial_confidence=1.0,
                    created_time=now,
                    last_seen_time=now,
                    last_confirmed_time=0.0,
                    detection_score=det.score,
                    latest_body_obs=detection_obs[j],
                    is_predicted=False,
                )
                self.tracks[self.next_track_id] = new_t
                self.next_track_id += 1

        # 6. Ambiguity check: detect track overlaps / path crossings
        track_list = list(self.tracks.values())
        for t in track_list:
            t.is_ambiguous = False
        for i in range(len(track_list)):
            for k in range(i + 1, len(track_list)):
                iou = compute_box_iou(track_list[i].body_bbox, track_list[k].body_bbox)
                if iou >= self.ambiguity_iou_threshold:
                    track_list[i].is_ambiguous = True
                    track_list[k].is_ambiguous = True

        # 7. Spatial Face-to-Body Association
        # Match each detected face to at most one body track
        assigned_faces = set()
        track_face_map: Dict[int, int] = {}

        face_pairs = []
        for f_idx, face in enumerate(face_detections):
            f_box = (face.x1, face.y1, face.x2, face.y2)
            for tid, track in self.tracks.items():
                score = spatial_face_in_body(f_box, track.body_bbox)
                if score >= 0.50:
                    face_pairs.append((score, f_idx, tid))

        face_pairs.sort(key=lambda x: x[0], reverse=True)
        assigned_tracks_for_face = set()
        for score, f_idx, tid in face_pairs:
            if f_idx not in assigned_faces and tid not in assigned_tracks_for_face:
                assigned_faces.add(f_idx)
                assigned_tracks_for_face.add(tid)
                track_face_map[tid] = f_idx

        # 8. Three-State Identity Machine & Appearance Gallery Learning
        for tid, track in list(self.tracks.items()):
            # A) Face is visible and associated with this body
            if tid in track_face_map:
                f_idx = track_face_map[tid]
                face = face_detections[f_idx]
                match = face_matches[f_idx] if f_idx < len(face_matches) else MatchResult(None, 1.0, False)

                track.face_bbox = (face.x1, face.y1, face.x2, face.y2)
                track.face_distance = match.distance

                if match.accepted and match.name == self.target_name:
                    # Biometrically CONFIRMED!
                    track.state = IdentityState.CONFIRMED
                    track.assigned_name = self.target_name
                    track.initial_confidence = 1.0
                    track.confidence = 1.0
                    track.last_confirmed_time = now
                    track.time_since_last_face = 0.0

                    # Learn body appearance while face is confirmed!
                    if track.latest_body_obs is not None:
                        can_learn = self.target_profile.can_update(
                            is_confirmed=True,
                            is_temporary=False,
                            is_ambiguous=track.is_ambiguous,
                            is_predicted=track.is_predicted,
                            quality=track.latest_body_obs.quality_score,
                        )
                        if can_learn:
                            self.target_profile.add_observation(track.latest_body_obs, is_face_verified=True)

                elif match.accepted and match.name is not None and match.name != self.target_name:
                    # Confirmed as a different enrolled person
                    track.state = IdentityState.UNKNOWN
                    track.assigned_name = match.name
                    track.confidence = 0.0

                else:
                    # Face visible but not verified
                    if track.state == IdentityState.CONFIRMED:
                        track.state = IdentityState.TEMPORARY
                    track.time_since_last_face = now - track.last_confirmed_time

            # B) Face is not visible (Head turned, 180-deg rotation, occlusion, or back-facing)
            else:
                track.face_bbox = None
                track.time_since_last_face = now - track.last_confirmed_time if track.last_confirmed_time > 0 else 9999.0

                # Compute appearance Re-ID similarity against target profile
                reid_sim = 0.0
                if self.target_profile.is_initialized and track.latest_body_obs is not None:
                    max_sim, _ = self.target_profile.query_similarity(track.latest_body_obs)
                    reid_sim = max_sim
                track.reid_similarity = reid_sim

                # If track was previously CONFIRMED, transition to TEMPORARY
                if track.state == IdentityState.CONFIRMED:
                    track.state = IdentityState.TEMPORARY

                # While in TEMPORARY:
                if track.state == IdentityState.TEMPORARY:
                    # Evaluate Re-ID appearance evidence
                    # If appearance contradicts target gallery (reid_sim < threshold), demote
                    reid_match = (reid_sim >= self.reid_sim_threshold) or (not self.target_profile.is_initialized)

                    if not reid_match:
                        # Body appearance does not match Michael's learned gallery
                        track.state = IdentityState.UNKNOWN
                        track.assigned_name = None
                        track.confidence = 0.0
                    else:
                        # Exponential confidence decay: C(t) = C0 * exp(-lambda * delta_t)
                        delta_t = now - track.last_confirmed_time
                        mult = 2.5 if track.is_ambiguous else 1.0
                        decayed_conf = self.decay_model.compute(
                            track.initial_confidence, delta_t, ambiguity_multiplier=mult
                        )
                        track.confidence = decayed_conf

                        # If confidence drops below cutoff, acknowledge honest uncertainty
                        if self.decay_model.is_expired(decayed_conf):
                            track.state = IdentityState.UNKNOWN
                            track.assigned_name = None
                            track.confidence = 0.0
                        else:
                            # Safe multi-view appearance learning during rotation
                            if track.latest_body_obs is not None:
                                can_learn = self.target_profile.can_update(
                                    is_confirmed=False,
                                    is_temporary=True,
                                    is_ambiguous=track.is_ambiguous,
                                    is_predicted=track.is_predicted,
                                    quality=track.latest_body_obs.quality_score,
                                )
                                if can_learn:
                                    self.target_profile.add_observation(track.latest_body_obs, is_face_verified=False)

                elif track.state == IdentityState.UNKNOWN:
                    track.confidence = 0.0
                    track.assigned_name = None

        # 9. Anti-confusion enforcement:
        # Guarantee at most one track is provisionally associated with the target name
        target_tracks = [
            t for t in self.tracks.values()
            if t.state in (IdentityState.CONFIRMED, IdentityState.TEMPORARY) and t.assigned_name == self.target_name
        ]
        if len(target_tracks) > 1:
            # Prefer CONFIRMED over TEMPORARY; if both TEMPORARY, prefer highest confidence + ReID
            target_tracks.sort(
                key=lambda t: (1 if t.state == IdentityState.CONFIRMED else 0, t.confidence, t.reid_similarity),
                reverse=True,
            )
            # Demote competitors to UNKNOWN
            for loser in target_tracks[1:]:
                loser.state = IdentityState.UNKNOWN
                loser.assigned_name = None
                loser.confidence = 0.0

        return list(self.tracks.values())

    def get_target_track(self) -> Optional[TrackedPerson]:
        """Returns the active target track if in CONFIRMED or TEMPORARY state."""
        for track in self.tracks.values():
            if track.state == IdentityState.CONFIRMED and track.assigned_name == self.target_name:
                return track

        temp_tracks = [
            t for t in self.tracks.values()
            if t.state == IdentityState.TEMPORARY and t.assigned_name == self.target_name
        ]
        if temp_tracks:
            temp_tracks.sort(key=lambda t: t.confidence, reverse=True)
            return temp_tracks[0]

        return None

    def calculate_centering(
        self,
        frame_width: int,
        frame_height: int,
        target_track: Optional[TrackedPerson] = None,
    ) -> Optional[CenteringResult]:
        """Calculates whole-body target position relative to frame center with jitter smoothing."""
        track = target_track or self.get_target_track()
        if track is None:
            self._target_ema_center = None
            return None

        # Primary anchor: whole-body centroid
        bx1, by1, bx2, by2 = track.body_bbox
        raw_cx = float((bx1 + bx2) / 2.0)
        raw_cy = float((by1 + by2) / 2.0)

        # Smooth target position with exponential moving average (EMA)
        if self._target_ema_center is None:
            smooth_cx, smooth_cy = raw_cx, raw_cy
        else:
            prev_cx, prev_cy = self._target_ema_center
            smooth_cx = self._ema_alpha * raw_cx + (1.0 - self._ema_alpha) * prev_cx
            smooth_cy = self._ema_alpha * raw_cy + (1.0 - self._ema_alpha) * prev_cy

        self._target_ema_center = (smooth_cx, smooth_cy)
        target_x = int(round(smooth_cx))
        target_y = int(round(smooth_cy))

        fcx = frame_width // 2
        fcy = frame_height // 2

        dx = int(target_x - fcx)
        dy = int(target_y - fcy)

        norm_dx = float(dx / (frame_width / 2.0))
        norm_dy = float(dy / (frame_height / 2.0))

        dist_px = float(math.hypot(dx, dy))

        dir_h = "CENTER" if abs(dx) < 20 else ("LEFT" if dx < 0 else "RIGHT")
        dir_v = "CENTER" if abs(dy) < 20 else ("UP" if dy < 0 else "DOWN")

        return CenteringResult(
            target_xy=(target_x, target_y),
            frame_center=(fcx, fcy),
            offset_px=(dx, dy),
            normalized_offset=(norm_dx, norm_dy),
            distance_px=dist_px,
            direction_h=dir_h,
            direction_v=dir_v,
        )
