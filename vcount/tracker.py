"""IoU tracker run on detection cycles. Track lifetime is in seconds of video time."""
from dataclasses import dataclass
from itertools import count

import numpy as np


def iou(a, b):
    """IoU matrix between box arrays a (N, 4) and b (M, 4)."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


@dataclass(eq=False)
class Track:
    track_id: int
    det: object  # latest Detection
    first_t: float
    last_t: float
    first_bbox: np.ndarray
    face_id: int = None
    first_crop: np.ndarray = None


class Tracker:
    def __init__(self, max_age_s, iou_threshold=0.3):
        self.max_age_s = max_age_s
        self.iou_threshold = iou_threshold
        self.tracks = []
        self._ids = count(1)

    def update(self, dets, t):
        """Associate detections with tracks. Returns (matched tracks, new tracks, removed tracks)."""
        matched, used = [], set()
        if self.tracks and dets:
            m = iou([tr.det.bbox for tr in self.tracks], [d.bbox for d in dets])
            # ponytail: greedy matching by IoU; Hungarian only matters in dense crowds.
            for ti, di in sorted(np.argwhere(m >= self.iou_threshold), key=lambda p: -m[p[0], p[1]]):
                tr = self.tracks[ti]
                if di in used or tr in matched:
                    continue
                tr.det, tr.last_t = dets[di], t
                matched.append(tr)
                used.add(di)

        new = [Track(next(self._ids), d, t, t, d.bbox.copy()) for i, d in enumerate(dets) if i not in used]
        removed = [tr for tr in self.tracks if tr not in matched and t - tr.last_t > self.max_age_s]
        self.tracks = [tr for tr in self.tracks if tr not in removed] + new
        return matched, new, removed
