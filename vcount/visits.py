"""Entry/exit state per face_id: one entry when a visit opens, one exit after exit_grace_s unseen."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Event:
    face_id: int
    event: str  # "entry" or "exit"
    video_time_s: float
    bbox: np.ndarray
    crop: np.ndarray


@dataclass
class Visit:
    last_t: float
    last_bbox: np.ndarray
    last_crop: np.ndarray


class Visits:
    def __init__(self, exit_grace_s):
        self.exit_grace_s = exit_grace_s
        self.open = {}  # face_id -> Visit

    def seen(self, face_id, t, bbox, crop, entry_t, entry_bbox, entry_crop):
        """Record a sighting. Returns the entry Event if this sighting opens a visit."""
        visit = self.open.get(face_id)
        if visit is not None:
            visit.last_t, visit.last_bbox, visit.last_crop = t, bbox, crop
            return None
        self.open[face_id] = Visit(t, bbox, crop)
        return Event(face_id, "entry", entry_t, entry_bbox, entry_crop)

    def expire(self, t):
        """Close visits not seen for more than exit_grace_s. The exit is stamped at the last sighting."""
        gone = [fid for fid, v in self.open.items() if t - v.last_t > self.exit_grace_s]
        return [self._close(fid) for fid in gone]

    def close_all(self):
        return [self._close(fid) for fid in list(self.open)]

    def _close(self, face_id):
        v = self.open.pop(face_id)
        return Event(face_id, "exit", v.last_t, v.last_bbox, v.last_crop)
