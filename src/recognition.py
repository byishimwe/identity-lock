"""Shared camera, landmark, alignment, embedding, and identity matching tools."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

try:
    import mediapipe as mp
except Exception as error:  # pragma: no cover - depends on local installation
    mp = None
    _MP_IMPORT_ERROR = error


DEFAULT_EMBEDDER_MODEL = Path("models/embedder_arcface.onnx")
FALLBACK_EMBEDDER_MODEL = Path("models/w600k_r50.onnx")


@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    kps: np.ndarray  # five ArcFace alignment points, (5, 2)
    landmarks: np.ndarray  # all FaceMesh points, (N, 2)


@dataclass
class MatchResult:
    name: Optional[str]
    distance: float
    accepted: bool


@dataclass
class DetectStats:
    mesh_count: int = 0
    rejected_eyespan: int = 0
    output_count: int = 0

    def describe(self) -> str:
        rej = (f" reject-eyespan={self.rejected_eyespan}" if self.rejected_eyespan else "")
        return f"mesh={self.mesh_count}{rej} kept={self.output_count}"


def _clip_xyxy(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Tuple[int, int, int, int]:
    x1, x2 = sorted((int(round(x1)), int(round(x2))))
    y1, y2 = sorted((int(round(y1)), int(round(y2))))
    return max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)


def _bbox_from_5pt(kps: np.ndarray) -> Tuple[float, float, float, float]:
    x_min, y_min = np.min(kps, axis=0)
    x_max, y_max = np.max(kps, axis=0)
    span_x, span_y = max(1.0, x_max - x_min), max(1.0, y_max - y_min)
    return x_min - .55 * span_x, y_min - .85 * span_y, x_max + .55 * span_x, y_max + 1.15 * span_y


class FaceMeshDetector:
    """Multi-face MediaPipe detector that retains full landmarks per face,
    with automatic OpenCV Haar cascade fallback if MediaPipe solutions is unavailable.
    """

    def __init__(self) -> None:
        self.mesh = None
        self.cascade = None
        if mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            try:
                self.mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=5,
                    refine_landmarks=False,
                    min_detection_confidence=.3,
                    min_tracking_confidence=.3,
                )
            except Exception:
                self.mesh = None

        if self.mesh is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.cascade = cv2.CascadeClassifier(cascade_path)
            if self.cascade.empty():
                raise RuntimeError("Failed to load OpenCV face cascade fallback.")

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 5) -> List[FaceDet]:
        faces, stats = self.detect_with_stats(frame_bgr, max_faces)
        self._last_stats = stats
        return faces

    def detect_with_stats(
        self, frame_bgr: np.ndarray, max_faces: int = 5,
    ) -> Tuple[List[FaceDet], DetectStats]:
        height, width = frame_bgr.shape[:2]
        stats = DetectStats()

        if self.mesh is not None:
            result = self.mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if not result.multi_face_landmarks:
                self._last_faces, self._last_stats = [], stats
                return [], stats
            stats.mesh_count = len(result.multi_face_landmarks)
            faces: List[FaceDet] = []
            indices = [33, 263, 1, 61, 291]
            for result_landmarks in result.multi_face_landmarks[:max_faces]:
                landmarks = np.array(
                    [[point.x * width, point.y * height] for point in result_landmarks.landmark], dtype=np.float32,
                )
                kps = landmarks[indices].copy()
                if kps[0, 0] > kps[1, 0]:
                    kps[[0, 1]] = kps[[1, 0]]
                if kps[3, 0] > kps[4, 0]:
                    kps[[3, 4]] = kps[[4, 3]]
                if np.linalg.norm(kps[1] - kps[0]) < 6.0:
                    stats.rejected_eyespan += 1
                    continue
                faces.append(FaceDet(*_clip_xyxy(*_bbox_from_5pt(kps), width, height), kps=kps, landmarks=landmarks))
            stats.output_count = len(faces)
            self._last_faces, self._last_stats = faces, stats
            return faces, stats

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        boxes = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        if len(boxes) == 0:
            self._last_faces, self._last_stats = [], stats
            return [], stats

        stats.mesh_count = len(boxes)
        faces = []
        indices = [33, 263, 1, 61, 291]
        for (x, y, w, h) in boxes[:max_faces]:
            left_eye = np.array([x + 0.35 * w, y + 0.38 * h], dtype=np.float32)
            right_eye = np.array([x + 0.65 * w, y + 0.38 * h], dtype=np.float32)
            nose = np.array([x + 0.50 * w, y + 0.55 * h], dtype=np.float32)
            mouth_l = np.array([x + 0.36 * w, y + 0.75 * h], dtype=np.float32)
            mouth_r = np.array([x + 0.64 * w, y + 0.75 * h], dtype=np.float32)
            kps = np.stack([left_eye, right_eye, nose, mouth_l, mouth_r])

            landmarks = np.zeros((468, 2), dtype=np.float32)
            landmarks[indices] = kps
            landmarks[13] = [x + 0.50 * w, y + 0.72 * h]
            landmarks[14] = [x + 0.50 * w, y + 0.78 * h]
            landmarks[107] = [x + 0.38 * w, y + 0.30 * h]
            landmarks[336] = [x + 0.62 * w, y + 0.30 * h]

            faces.append(FaceDet(*_clip_xyxy(*_bbox_from_5pt(kps), width, height), kps=kps, landmarks=landmarks))

        stats.output_count = len(faces)
        self._last_faces, self._last_stats = faces, stats
        return faces, stats

    def close(self) -> None:
        if self.mesh is not None:
            self.mesh.close()


def align_face_5pt(frame_bgr: np.ndarray, kps: np.ndarray, out_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    destination = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
        dtype=np.float32,
    )
    out_width, out_height = out_size
    destination *= np.array([out_width / 112.0, out_height / 112.0], dtype=np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), destination, method=cv2.LMEDS)
    if matrix is None:
        matrix = cv2.getAffineTransform(kps[:3].astype(np.float32), destination[:3])
    return cv2.warpAffine(frame_bgr, matrix, out_size, flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))


def resolve_embedder_model_path(model_path: Path = DEFAULT_EMBEDDER_MODEL) -> str:
    if model_path.exists() and model_path.stat().st_size > 0:
        return str(model_path)
    if model_path == DEFAULT_EMBEDDER_MODEL and FALLBACK_EMBEDDER_MODEL.exists() and FALLBACK_EMBEDDER_MODEL.stat().st_size > 0:
        return str(FALLBACK_EMBEDDER_MODEL)
    raise FileNotFoundError(f"ArcFace model not found: {model_path} (or {FALLBACK_EMBEDDER_MODEL}).")


class ArcFaceEmbedderONNX:
    def __init__(self, model_path: Path = DEFAULT_EMBEDDER_MODEL) -> None:
        self.session = ort.InferenceSession(resolve_embedder_model_path(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(cv2.resize(aligned_bgr, (112, 112)), cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = np.transpose((rgb - 127.5) / 128.0, (2, 0, 1))[None].astype(np.float32)
        embedding = self.session.run([self.output_name], {self.input_name: tensor})[0].reshape(-1).astype(np.float32)
        return embedding / (np.linalg.norm(embedding) + 1e-12)


def load_db_npz(db_path: Path | str) -> Dict[str, np.ndarray]:
    p = Path(db_path)
    if not p.exists():
        return {}
    data = np.load(str(p), allow_pickle=True)
    return {name: np.asarray(data[name], dtype=np.float32).reshape(-1) for name in data.files}


class FaceDBMatcher:
    def __init__(self, db: Dict[str, np.ndarray], dist_thresh: float = .34) -> None:
        self.names = sorted(db)
        self.matrix = np.stack([db[name].reshape(-1) for name in self.names]) if self.names else None
        self.dist_thresh = dist_thresh

    def match(self, embedding: np.ndarray) -> MatchResult:
        if self.matrix is None:
            return MatchResult(None, 1.0, False)
        similarities = self.matrix @ embedding.reshape(-1)
        index = int(np.argmax(similarities))
        distance = float(1.0 - similarities[index])
        accepted = distance <= self.dist_thresh
        return MatchResult(self.names[index] if accepted else None, distance, accepted)
