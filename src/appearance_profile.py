"""Identity-Linked Multi-View Body Appearance Profile and Gallery Manager.

Manages:
- Bounded multi-view appearance gallery (front, side, rear prototypes).
- Strict profile contamination prevention (no updates on UNKNOWN or ambiguous tracks).
- Safe online prototype aggregation across 180-degree rotations.
- Gallery matching and similarity scoring against candidate body tracks.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .body_features import BodyAppearanceExtractor, BodyObservation


@dataclass
class IdentityAppearanceProfile:
    """Maintains an identity-linked body appearance gallery across multiple viewpoints."""
    identity_name: str
    max_gallery_size: int = 12
    gallery: List[BodyObservation] = field(default_factory=list)
    last_updated_time: float = 0.0
    last_face_verified_time: float = 0.0
    total_samples_observed: int = 0

    @property
    def is_initialized(self) -> bool:
        return len(self.gallery) > 0

    def can_update(
        self,
        is_confirmed: bool,
        is_temporary: bool,
        is_ambiguous: bool,
        is_predicted: bool,
        quality: float,
    ) -> bool:
        """Strict safeguard policy preventing appearance profile contamination."""
        # Never update if track is ambiguous, predicted without detection, or low quality
        if is_ambiguous or is_predicted or quality < 0.25:
            return False

        # Only allow learning when face is directly confirmed, or during high-confidence
        # continuous rotation in TEMPORARY state
        if is_confirmed:
            return True
        if is_temporary and self.is_initialized:
            # Allow careful multi-view learning only within 3.0s of facial loss
            elapsed_loss = time.time() - self.last_face_verified_time
            return elapsed_loss <= 3.0

        return False

    def add_observation(self, obs: BodyObservation, is_face_verified: bool = False) -> bool:
        """Adds or updates representative appearance prototype in the multi-view gallery."""
        now = obs.timestamp
        if is_face_verified:
            self.last_face_verified_time = now
        self.last_updated_time = now
        self.total_samples_observed += 1

        if not self.gallery:
            self.gallery.append(obs)
            return True

        # Check similarity to existing prototypes in gallery
        sims = [
            BodyAppearanceExtractor.compare_observations(obs, proto)
            for proto in self.gallery
        ]
        max_sim = max(sims)
        max_idx = int(np.argmax(sims))

        # If very similar to an existing prototype (> 0.90), update prototype if higher quality
        if max_sim > 0.90:
            if obs.quality_score > self.gallery[max_idx].quality_score:
                self.gallery[max_idx] = obs
                return True
            return False

        # If adds new viewpoint diversity (e.g. side or rear view):
        if len(self.gallery) < self.max_gallery_size:
            self.gallery.append(obs)
            return True

        # If gallery is full, find the pair with highest pairwise redundancy
        # and replace the one with lower quality
        redundant_idx = self._find_most_redundant_index()
        self.gallery[redundant_idx] = obs
        return True

    def _find_most_redundant_index(self) -> int:
        """Identifies index of the most redundant prototype to replace."""
        n = len(self.gallery)
        if n <= 1:
            return 0
        best_sim = -1.0
        replace_cand = 0
        for i in range(n):
            for j in range(i + 1, n):
                s = BodyAppearanceExtractor.compare_observations(self.gallery[i], self.gallery[j])
                if s > best_sim:
                    best_sim = s
                    # Pick whichever has lower quality
                    replace_cand = i if self.gallery[i].quality_score < self.gallery[j].quality_score else j
        return replace_cand

    def query_similarity(self, obs: BodyObservation) -> Tuple[float, float]:
        """Compares candidate observation against all prototypes in gallery.

        Returns (max_similarity, top_mean_similarity).
        """
        if not self.gallery:
            return 0.0, 0.0

        sims = [
            BodyAppearanceExtractor.compare_observations(obs, proto)
            for proto in self.gallery
        ]
        sims.sort(reverse=True)
        max_sim = sims[0]
        top_k = min(3, len(sims))
        top_mean = float(np.mean(sims[:top_k]))
        return max_sim, top_mean
