"""YOLOX ONNX Whole-Person Detector (CPU-Optimized).

Handles:
- Letterboxing and preprocessing (416x416 default).
- ONNX Runtime CPUExecutionProvider inference.
- Multi-stride anchor decoding (strides 8, 16, 32).
- Person class filtering (COCO class 0).
- Non-Maximum Suppression (NMS).
- Coordinate restoration and clipping to original frame boundaries.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

DEFAULT_YOLOX_PATH = Path("models/yolox_nano.onnx")


@dataclass(frozen=True)
class PersonDetection:
    """Detected whole-person bounding box and score."""
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    score: float
    class_id: int = 0               # 0 = person

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        return self.box

    @property
    def width(self) -> int:
        return max(1, self.box[2] - self.box[0])

    @property
    def height(self) -> int:
        return max(1, self.box[3] - self.box[1])

    @property
    def center(self) -> Tuple[int, int]:
        return (self.box[0] + self.box[2]) // 2, (self.box[1] + self.box[3]) // 2


class YOLOXPersonDetector:
    """CPU-optimized YOLOX detector for whole-person detection."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_YOLOX_PATH,
        input_size: Tuple[int, int] = (416, 416),
        conf_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        intra_op_threads: int = 2,
    ) -> None:
        model_file = Path(model_path)
        if not model_file.exists() or model_file.stat().st_size == 0:
            raise FileNotFoundError(f"YOLOX model not found at: {model_file.resolve()}")

        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_file),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Precompute decoding grids for the input size
        self.grids, self.strides = self._generate_grids(self.input_size, strides=(8, 16, 32))
        self.last_inference_time_ms = 0.0

    @staticmethod
    def _generate_grids(
        input_size: Tuple[int, int], strides: Tuple[int, ...] = (8, 16, 32)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Precomputes coordinate grids and stride maps for decoding."""
        grids = []
        stride_map = []
        for stride in strides:
            h = input_size[0] // stride
            w = input_size[1] // stride
            xv, yv = np.meshgrid(np.arange(w), np.arange(h))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            stride_map.append(np.full((1, grid.shape[1], 1), stride, dtype=np.float32))

        return (
            np.concatenate(grids, axis=1).astype(np.float32),
            np.concatenate(stride_map, axis=1).astype(np.float32),
        )

    def _preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Letterboxes frame into input_size with color 114 padding."""
        orig_h, orig_w = frame_bgr.shape[:2]
        target_h, target_w = self.input_size
        scale = min(target_h / orig_h, target_w / orig_w)
        scaled_w = int(orig_w * scale)
        scaled_h = int(orig_h * scale)

        resized = cv2.resize(frame_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[:scaled_h, :scaled_w] = resized

        # Layout: CHW, float32, no division by 255 (standard YOLOX raw BGR input)
        tensor = np.transpose(padded.astype(np.float32), (2, 0, 1))[None, :]
        return tensor, scale, (scaled_w, scaled_h)

    def detect(self, frame_bgr: np.ndarray) -> List[PersonDetection]:
        """Detects all whole-person bounding boxes in frame."""
        t0 = time.perf_counter()
        orig_h, orig_w = frame_bgr.shape[:2]
        tensor, scale, _ = self._preprocess(frame_bgr)

        outputs = self.session.run([self.output_name], {self.input_name: tensor})[0]

        # Decode YOLOX output: [1, 3549, 85]
        # Box coordinates: cx, cy, w, h
        decoded = outputs.copy()
        decoded[..., :2] = (decoded[..., :2] + self.grids) * self.strides
        decoded[..., 2:4] = np.exp(decoded[..., 2:4]) * self.strides

        predictions = decoded[0]
        # Class 0 is person in COCO
        obj_conf = predictions[:, 4]
        person_cls_conf = predictions[:, 5]
        person_scores = obj_conf * person_cls_conf

        # Filter by score
        mask = person_scores >= self.conf_threshold
        if not np.any(mask):
            self.last_inference_time_ms = (time.perf_counter() - t0) * 1000.0
            return []

        boxes = predictions[mask, :4]
        scores = person_scores[mask]

        # Convert cx, cy, w, h -> x1, y1, w, h for OpenCV NMS
        cx = boxes[:, 0]
        cy = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]

        x1 = (cx - w / 2.0) / scale
        y1 = (cy - h / 2.0) / scale
        w = w / scale
        h = h / scale

        # Prepare for NMS
        nms_boxes = [[int(x), int(y), int(bw), int(bh)] for x, y, bw, bh in zip(x1, y1, w, h)]
        nms_scores = [float(s) for s in scores]

        indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, self.conf_threshold, self.nms_threshold)

        detections: List[PersonDetection] = []
        if len(indices) > 0:
            for idx in indices.flatten():
                bx, by, bw, bh = nms_boxes[idx]
                score = nms_scores[idx]
                # Clamp to frame boundaries
                clamped_x1 = max(0, min(orig_w - 1, bx))
                clamped_y1 = max(0, min(orig_h - 1, by))
                clamped_x2 = max(0, min(orig_w - 1, bx + bw))
                clamped_y2 = max(0, min(orig_h - 1, by + bh))

                if clamped_x2 > clamped_x1 + 5 and clamped_y2 > clamped_y1 + 10:
                    detections.append(
                        PersonDetection(
                            box=(clamped_x1, clamped_y1, clamped_x2, clamped_y2),
                            score=score,
                            class_id=0,
                        )
                    )

        self.last_inference_time_ms = (time.perf_counter() - t0) * 1000.0
        return detections
