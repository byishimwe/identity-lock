"""Body Feature Extraction and Re-Identification Module (CPU-Optimized).

Extracts:
1. Deep body appearance embeddings using OSNet ONNX (512-D, L2-normalized).
2. Regional clothing color histograms (Upper torso and lower body HSV distributions).
3. Multi-modal similarity computation combining deep Re-ID and regional appearance.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

DEFAULT_OSNET_PATH = Path("models/osnet_x0_25_msmt17.onnx")


@dataclass
class BodyObservation:
    """Multi-modal visual appearance observation of a person's body."""
    reid_embedding: np.ndarray            # 512-D float32 normalized
    upper_hist: np.ndarray                 # HSV 2D histogram (16, 8)
    lower_hist: Optional[np.ndarray]       # HSV 2D histogram (16, 8) or None if occluded
    timestamp: float
    bbox: Tuple[int, int, int, int]
    quality_score: float                   # Quality metric based on resolution & sharpness


class BodyAppearanceExtractor:
    """Extracts deep Re-ID embeddings and clothing color descriptors on CPU."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_OSNET_PATH,
        input_size: Tuple[int, int] = (128, 256),  # (width, height)
        intra_op_threads: int = 2,
    ) -> None:
        model_file = Path(model_path)
        if not model_file.exists() or model_file.stat().st_size == 0:
            raise FileNotFoundError(f"OSNet Re-ID model not found at: {model_file.resolve()}")

        self.input_size = input_size
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_file),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.required_batch_size = self.session.get_inputs()[0].shape[0]
        self.last_inference_time_ms = 0.0

        # Standard ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def _extract_color_hist(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extracts normalized 2D HSV histogram (Hue: 16 bins, Saturation: 8 bins)."""
        if crop_bgr.size == 0 or crop_bgr.shape[0] < 5 or crop_bgr.shape[1] < 5:
            return None
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        norm = np.linalg.norm(hist)
        return (hist / (norm + 1e-7)).astype(np.float32)

    def extract_observation(
        self,
        frame_bgr: np.ndarray,
        body_bbox: Tuple[int, int, int, int],
        timestamp: Optional[float] = None,
    ) -> Optional[BodyObservation]:
        """Extracts complete multi-modal body appearance observation from a person crop."""
        t0 = time.perf_counter()
        now = time.time() if timestamp is None else timestamp
        h_frame, w_frame = frame_bgr.shape[:2]
        x1, y1, x2, y2 = body_bbox

        x1 = max(0, min(w_frame - 1, x1))
        y1 = max(0, min(h_frame - 1, y1))
        x2 = max(0, min(w_frame - 1, x2))
        y2 = max(0, min(h_frame - 1, y2))

        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w < 20 or crop_h < 40:
            return None

        person_crop = frame_bgr[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None

        # 1. Deep OSNet Embedding
        in_w, in_h = self.input_size
        resized_rgb = cv2.cvtColor(cv2.resize(person_crop, (in_w, in_h), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        normalized = (resized_rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        chw = np.transpose(normalized, (2, 0, 1))

        # Handle batch size requirement
        if isinstance(self.required_batch_size, int) and self.required_batch_size > 1:
            batch = np.zeros((self.required_batch_size, 3, in_h, in_w), dtype=np.float32)
            batch[0] = chw
            raw_out = self.session.run([self.output_name], {self.input_name: batch})[0][0]
        else:
            batch = chw[None, :].astype(np.float32)
            raw_out = self.session.run([self.output_name], {self.input_name: batch})[0][0]

        reid_emb = raw_out.reshape(-1).astype(np.float32)
        reid_emb /= (np.linalg.norm(reid_emb) + 1e-12)

        # 2. Regional Clothing Color Histograms
        # Upper body (torso): top 15% to 55% of height
        top_y1 = int(0.15 * crop_h)
        top_y2 = int(0.55 * crop_h)
        upper_crop = person_crop[top_y1:top_y2, :]
        upper_hist = self._extract_color_hist(upper_crop)
        if upper_hist is None:
            upper_hist = np.zeros((16, 8), dtype=np.float32)

        # Lower body (pants/legs): 55% to 90% of height
        bot_y1 = int(0.55 * crop_h)
        bot_y2 = int(0.90 * crop_h)
        lower_crop = person_crop[bot_y1:bot_y2, :]
        lower_hist = self._extract_color_hist(lower_crop)

        # 3. Quality score (resolution + sharpness)
        gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        res_factor = min(1.0, (crop_w * crop_h) / (120 * 240))
        sharp_factor = min(1.0, laplacian_var / 120.0)
        quality = 0.6 * res_factor + 0.4 * sharp_factor

        self.last_inference_time_ms = (time.perf_counter() - t0) * 1000.0

        return BodyObservation(
            reid_embedding=reid_emb,
            upper_hist=upper_hist,
            lower_hist=lower_hist,
            timestamp=now,
            bbox=body_bbox,
            quality_score=quality,
        )

    @staticmethod
    def compare_observations(
        obs1: BodyObservation,
        obs2: BodyObservation,
        w_reid: float = 0.70,
        w_upper: float = 0.20,
        w_lower: float = 0.10,
    ) -> float:
        """Computes composite appearance similarity [0.0, 1.0] between two body observations."""
        # Cosine similarity of Re-ID embeddings
        sim_reid = float(np.dot(obs1.reid_embedding, obs2.reid_embedding))
        sim_reid = max(0.0, min(1.0, sim_reid))

        # Upper clothing histogram similarity (Bhattacharyya distance)
        d_upper = cv2.compareHist(obs1.upper_hist, obs2.upper_hist, cv2.HISTCMP_BHATTACHARYYA)
        sim_upper = max(0.0, 1.0 - float(d_upper))

        # Lower clothing histogram similarity
        if obs1.lower_hist is not None and obs2.lower_hist is not None:
            d_lower = cv2.compareHist(obs1.lower_hist, obs2.lower_hist, cv2.HISTCMP_BHATTACHARYYA)
            sim_lower = max(0.0, 1.0 - float(d_lower))
            total_sim = w_reid * sim_reid + w_upper * sim_upper + w_lower * sim_lower
        else:
            # Fallback to upper body and Re-ID if lower body occluded
            total_sim = (w_reid / (w_reid + w_upper)) * sim_reid + (w_upper / (w_reid + w_upper)) * sim_upper

        return float(max(0.0, min(1.0, total_sim)))
