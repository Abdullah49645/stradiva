"""
Vehicle detector on OpenCV 5's DNN module: YOLOX-S from the OpenCV Model Zoo
(Apache-2.0; github.com/opencv/opencv_zoo, models/object_detection_yolox),
COCO classes car / bus / truck / motorcycle.

Why a learned detector at all. Background subtraction finds *moving pixels*,
not vehicles: two vehicles that touch in the image become one blob, and a
vehicle that enters the frame already overlapping another is never separated
(the dominant end-to-end failure in docs/EVALUATION_RESULTS.md). A detector
gives one box per vehicle even when their silhouettes merge. The tracker
(cv/tracking/mot.py) uses detector boxes to SPLIT fused foreground blobs; the
ground contact is still measured on the foreground mask inside each box
(pixel-accurate), never from the box alone.

The model is not in the repository (36 MB): models/fetch_models.sh downloads
it and checks its SHA-256. Without it, Odolite runs exactly as before
(background subtraction only) and says so in the case record.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL = os.environ.get("ODOLITE_DETECTOR_MODEL",
                               os.path.join(ROOT, "models", "object_detection_yolox_2022nov.onnx"))
MODEL_SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class VehicleDetector:
    """YOLOX-S, 640x640 letterboxed input, decoded as in the OpenCV Zoo reference."""

    def __init__(self, model_path=DEFAULT_MODEL, conf=0.35, nms=0.5, input_size=640):
        self.net = cv2.dnn.readNet(model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.size = input_size
        self.conf, self.nms = conf, nms
        grids, strides = [], []
        for s in (8, 16, 32):
            n = input_size // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        self.grids = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)

    def detect(self, bgr):
        """Returns a list of {bbox: [x, y, w, h], score, cls} for vehicles, in image pixels."""
        h, w = bgr.shape[:2]
        r = min(self.size / h, self.size / w)
        img = np.full((self.size, self.size, 3), 114.0, np.float32)
        rs = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        img[: rs.shape[0], : rs.shape[1]] = rs
        self.net.setInput(np.transpose(img, (2, 0, 1))[None])
        out = self.net.forward(self.net.getUnconnectedOutLayersNames())[0][0]
        xy = (out[:, :2] + self.grids) * self.strides
        wh = np.exp(out[:, 2:4]) * self.strides
        scores = out[:, 4:5] * out[:, 5:]
        cls = np.argmax(scores, axis=1)
        sc = scores[np.arange(len(cls)), cls]
        keep = np.isin(cls, list(VEHICLE_CLASSES)) & (sc >= self.conf)
        if not keep.any():
            return []
        xy, wh, sc, cls = xy[keep], wh[keep], sc[keep], cls[keep]
        boxes = np.column_stack([xy - wh / 2, wh]) / r
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), sc.tolist(), self.conf, self.nms)
        dets = []
        for i in np.array(idx).reshape(-1):
            x, y, bw, bh = boxes[i]
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(w), x + bw), min(float(h), y + bh)
            if x1 - x0 > 2 and y1 - y0 > 2:
                dets.append({"bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                             "score": float(sc[i]), "cls": VEHICLE_CLASSES[int(cls[i])]})
        return dets


_DETECTOR = None


def get_detector():
    """Shared detector instance, or None if the model file is not installed."""
    global _DETECTOR
    if _DETECTOR is None and os.path.exists(DEFAULT_MODEL) and os.environ.get("ODOLITE_DETECTOR", "1") != "0":
        _DETECTOR = VehicleDetector(DEFAULT_MODEL)
    return _DETECTOR
