"""YOLOv8n-face detector: boxes, scores and 5 landmarks (eyes, nose, mouth corners)."""
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort

REG_MAX = 16
STRIDES = (8, 16, 32)


@dataclass
class Detection:
    bbox: np.ndarray  # [x1, y1, x2, y2] float, source-frame pixels
    score: float
    kps: np.ndarray  # (5, 2): left eye, right eye, nose, left mouth, right mouth (image left/right)

    @property
    def size(self):
        return min(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])


class Detector:
    def __init__(self, model_path, input_size, conf, nms_iou=0.45):
        if input_size % 32:
            raise ValueError("det_input_size must be a multiple of 32")
        self.model_path = model_path
        self.size = input_size
        self.conf = conf
        self.nms_iou = nms_iou
        # The exported network has a fixed det_input_size x det_input_size input, so the anchor grids are
        # fixed too and are built once here.
        self.anchors = {}
        for s in STRIDES:
            n = input_size // s
            ys, xs = np.mgrid[0:n, 0:n]
            self.anchors[s] = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32) + 0.5

    def detect(self, frame):
        h, w = frame.shape[:2]
        scale = self.size / max(h, w)
        # INTER_AREA averages every source pixel; plain bilinear skips most of them at a 4x shrink and
        # aliases fine detail, which costs the detector its smallest faces.
        resized = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        rh, rw = resized.shape[:2]

        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[:rh, :rw] = resized
        # YOLO models are trained on RGB; swapRB converts the BGR frame.
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, swapRB=True)
        session = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        outputs = session.run(None, {session.get_inputs()[0].name: blob})

        boxes, scores, kpss = [], [], []
        for out in outputs:
            stride = self.size // out.shape[2]
            p = out[0].reshape(out.shape[1], -1).T  # (cells, 64 + 1 + 15)
            score = 1 / (1 + np.exp(-p[:, 4 * REG_MAX]))
            keep = score >= self.conf
            if not keep.any():
                continue
            p, score, anchor = p[keep], score[keep], self.anchors[stride][keep]
            dfl = p[:, : 4 * REG_MAX].reshape(-1, 4, REG_MAX)
            dfl = np.exp(dfl - dfl.max(axis=2, keepdims=True))
            dist = (dfl / dfl.sum(axis=2, keepdims=True)) @ np.arange(REG_MAX, dtype=np.float32)
            box = np.concatenate([anchor - dist[:, :2], anchor + dist[:, 2:]], axis=1) * stride
            kps = p[:, 4 * REG_MAX + 1 :].reshape(-1, 5, 3)[:, :, :2]
            kps = (kps * 2.0 + (anchor[:, None, :] - 0.5)) * stride
            boxes.append(box)
            scores.append(score)
            kpss.append(kps)
        if not boxes:
            return []

        boxes = np.concatenate(boxes) / scale
        scores = np.concatenate(scores)
        kpss = np.concatenate(kpss) / scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        xywh = np.concatenate([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], axis=1)
        idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), self.conf, self.nms_iou)
        return [Detection(boxes[i], float(scores[i]), kpss[i]) for i in np.array(idx).ravel()]
