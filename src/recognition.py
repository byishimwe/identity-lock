"""Shared camera, landmark, alignment, embedding, and identity matching tools.

Includes:
- YuNet ONNX face detection (Primary detector).
- MediaPipe FaceMesh (Detailed facial expressions & eye signals).
- ArcFace ONNX embedder & cosine similarity identity matcher.
"""
from __future__ import annotations

import time
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


DEFAULT_YUNET_MODEL = Path("models/face_detection_yunet_2023mar.onnx")
DEFAULT_EMBEDDER_MODEL = Path("models/embedder_arcface.onnx")
FALLBACK_EMBEDDER_MODEL = Path("models/w600k_r50.onnx")


@dataclass
class FaceDet:
    x1: int
    y1: int
    x2: int
    y2: int
    kps: np.ndarray        # five ArcFace alignment points: (5, 2)
    landmarks: np.ndarray  # FaceMesh or synthetic landmarks: (468, 2)
    score: float = 1.0


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


class YuNetFaceDetector:
    """Primary OpenCV YuNet ONNX face detector.

    Outputs 5-point facial landmarks compatible with ArcFace alignment.
    """

    def __init__(
        self,
        model_path: Path | str = DEFAULT_YUNET_MODEL,
        score_threshold: float = 0.50,
        nms_threshold: float = 0.30,
    ) -> None:
        p = Path(model_path)
        if not p.exists() or p.stat().st_size == 0:
            raise FileNotFoundError(f"YuNet ONNX model not found at: {p.resolve()}")

        self.model_path = str(p)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.current_size: Tuple[int, int] = (320, 320)

        self.detector = cv2.FaceDetectorYN.create(
            self.model_path,
            "",
            self.current_size,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
        )
        self.last_inference_time_ms = 0.0

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 5) -> List[FaceDet]:
        faces, _ = self.detect_with_stats(frame_bgr, max_faces)
        return faces

    def detect_with_stats(
        self, frame_bgr: np.ndarray, max_faces: int = 5
    ) -> Tuple[List[FaceDet], DetectStats]:
        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]
        if self.current_size != (w, h):
            self.detector.setInputSize((w, h))
            self.current_size = (w, h)

        ret, raw_faces = self.detector.detect(frame_bgr)
        self.last_inference_time_ms = (time.perf_counter() - t0) * 1000.0

        stats = DetectStats()
        if ret == 0 or raw_faces is None or len(raw_faces) == 0:
            return [], stats

        stats.mesh_count = len(raw_faces)
        faces: List[FaceDet] = []

        # Sort by confidence score descending
        sorted_faces = sorted(raw_faces, key=lambda f: f[14], reverse=True)[:max_faces]

        indices = [33, 263, 1, 61, 291]
        for f in sorted_faces:
            bx, by, bw, bh = f[0], f[1], f[2], f[3]
            score = float(f[14])

            # Five landmarks from YuNet:
            # f[4:6]: right eye (subject's right)
            # f[6:8]: left eye (subject's left)
            # f[8:10]: nose tip
            # f[10:12]: right mouth corner
            # f[12:14]: left mouth corner
            re = np.array([f[4], f[5]], dtype=np.float32)
            le = np.array([f[6], f[7]], dtype=np.float32)
            nose = np.array([f[8], f[9]], dtype=np.float32)
            rm = np.array([f[10], f[11]], dtype=np.float32)
            lm = np.array([f[12], f[13]], dtype=np.float32)

            # Ensure image left eye (lower x) is point 0, image right eye is point 1
            if le[0] > re[0]:
                eye_l, eye_r = re, le
            else:
                eye_l, eye_r = le, re

            if lm[0] > rm[0]:
                mouth_l, mouth_r = rm, lm
            else:
                mouth_l, mouth_r = lm, rm

            kps = np.stack([eye_l, eye_r, nose, mouth_l, mouth_r]).astype(np.float32)

            # Check eye span validity
            if np.linalg.norm(kps[1] - kps[0]) < 6.0:
                stats.rejected_eyespan += 1
                continue

            x1 = max(0, min(w - 1, int(round(bx))))
            y1 = max(0, min(h - 1, int(round(by))))
            x2 = max(0, min(w - 1, int(round(bx + bw))))
            y2 = max(0, min(h - 1, int(round(by + bh))))

            # Synthetic 468 landmark array
            landmarks = np.zeros((468, 2), dtype=np.float32)
            landmarks[indices] = kps
            landmarks[13] = [nose[0], mouth_l[1] - 0.2 * (mouth_l[1] - nose[1])]
            landmarks[14] = [nose[0], mouth_l[1] + 0.2 * (mouth_l[1] - nose[1])]
            landmarks[107] = [eye_l[0], eye_l[1] - 0.2 * (nose[1] - eye_l[1])]
            landmarks[336] = [eye_r[0], eye_r[1] - 0.2 * (nose[1] - eye_r[1])]

            faces.append(FaceDet(x1=x1, y1=y1, x2=x2, y2=y2, kps=kps, landmarks=landmarks, score=score))

        stats.output_count = len(faces)
        return faces, stats


class FaceMeshDetector:
    """MediaPipe detector retaining full 468 landmarks for facial expression/eye analysis."""

    def __init__(self) -> None:
        self.mesh = None
        if mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            try:
                self.mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=3,
                    refine_landmarks=False,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3,
                )
            except Exception:
                self.mesh = None

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 3) -> List[FaceDet]:
        faces, _ = self.detect_with_stats(frame_bgr, max_faces)
        return faces

    def detect_with_stats(
        self, frame_bgr: np.ndarray, max_faces: int = 3
    ) -> Tuple[List[FaceDet], DetectStats]:
        height, width = frame_bgr.shape[:2]
        stats = DetectStats()
        if self.mesh is None:
            return [], stats

        result = self.mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return [], stats

        stats.mesh_count = len(result.multi_face_landmarks)
        faces: List[FaceDet] = []
        indices = [33, 263, 1, 61, 291]
        for result_landmarks in result.multi_face_landmarks[:max_faces]:
            landmarks = np.array(
                [[point.x * width, point.y * height] for point in result_landmarks.landmark],
                dtype=np.float32,
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
    """ArcFace 512-D ONNX Embedder running strictly on CPU."""

    def __init__(self, model_path: Path = DEFAULT_EMBEDDER_MODEL, intra_op_threads: int = 2) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            resolve_embedder_model_path(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.last_inference_time_ms = 0.0

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(cv2.resize(aligned_bgr, (112, 112)), cv2.COLOR_BGR2RGB).astype(np.float32)
        tensor = np.transpose((rgb - 127.5) / 128.0, (2, 0, 1))[None].astype(np.float32)
        embedding = self.session.run([self.output_name], {self.input_name: tensor})[0].reshape(-1).astype(np.float32)
        norm = np.linalg.norm(embedding)
        self.last_inference_time_ms = (time.perf_counter() - t0) * 1000.0
        return embedding / (norm + 1e-12)


def load_db_npz(db_path: Path | str) -> Dict[str, np.ndarray]:
    p = Path(db_path)
    if not p.exists():
        return {}
    data = np.load(str(p), allow_pickle=True)
    return {name: np.asarray(data[name], dtype=np.float32).reshape(-1) for name in data.files}


class FaceDBMatcher:
    """Matches ArcFace 512-D embeddings against enrolled identities in the database."""

    def __init__(self, db: Dict[str, np.ndarray], dist_thresh: float = 0.42) -> None:
        self.names = sorted(db)
        self.matrix = np.stack([db[name].reshape(-1) for name in self.names]) if self.names else None
        self.dist_thresh = dist_thresh

    def match(self, embedding: np.ndarray) -> MatchResult:
        if self.matrix is None or len(self.names) == 0:
            return MatchResult(None, 1.0, False)
        similarities = self.matrix @ embedding.reshape(-1)
        index = int(np.argmax(similarities))
        distance = float(1.0 - similarities[index])
        accepted = distance <= self.dist_thresh
        return MatchResult(self.names[index] if accepted else None, distance, accepted)
