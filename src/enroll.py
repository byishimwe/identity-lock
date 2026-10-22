"""Enroll or replace an identity in the local FaceLocking database."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .recognition import ArcFaceEmbedderONNX, YuNetFaceDetector, align_face_5pt, load_db_npz

DB_PATH = Path("data/db/face_db.npz")
META_PATH = Path("data/db/face_db.json")


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
    parser = argparse.ArgumentParser(description="Enroll an identity for FaceLocking using YuNet + ArcFace.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0).")
    parser.add_argument("--width", type=int, default=1920, help="Frame width (default 1920).")
    parser.add_argument("--height", type=int, default=1080, help="Frame height (default 1080).")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--mirror", action=argparse.BooleanOptionalAction, default=True, help="Mirror video horizontally like a phone selfie camera.")
    args = parser.parse_args()
    name = args.name.strip()
    if not name:
        parser.error("--name must not be blank")

    detector = YuNetFaceDetector()
    embedder = ArcFaceEmbedderONNX()
    capture, active_cam_idx = open_camera(args.camera, width=args.width, height=args.height)
    samples: List[np.ndarray] = []
    auto, last_auto = False, 0.0
    print("SPACE=capture, a=auto capture, s=save, q=quit")
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            if args.mirror:
                frame = cv2.flip(frame, 1)
            faces = detector.detect(frame, max_faces=1)
            aligned = align_face_5pt(frame, faces[0].kps) if faces else None
            view = frame.copy()
            if faces:
                face = faces[0]
                cv2.rectangle(view, (face.x1, face.y1), (face.x2, face.y2), (0, 255, 0), 2)
            cv2.putText(view, f"ENROLL {name}: {len(samples)}/{args.samples}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
            cv2.putText(view, "SPACE=capture a=auto s=save q=quit", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
            cv2.imshow("FaceLocking Enrollment", view)
            key = cv2.waitKey(1) & 0xFF
            now = time.time()
            capture_sample = key == ord(" ") or (auto and aligned is not None and now - last_auto >= .25)
            if capture_sample and aligned is not None:
                samples.append(embedder.embed(aligned))
                last_auto = now
            if key == ord("a"):
                auto = not auto
            if key == ord("s"):
                if len(samples) < args.samples:
                    print(f"Need {args.samples} samples; have {len(samples)}.")
                    continue
                template = np.mean(np.stack(samples), axis=0)
                template /= np.linalg.norm(template) + 1e-12
                db = load_db_npz(DB_PATH)
                db[name] = template.astype(np.float32)
                DB_PATH.parent.mkdir(parents=True, exist_ok=True)
                np.savez(DB_PATH, **db)
                META_PATH.write_text(json.dumps({"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "names": sorted(db)}, indent=2), encoding="utf-8")
                print(f"Saved {name} ({len(samples)} samples).")
                break
            if key == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
