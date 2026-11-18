"""Everything written to --out: visitors.db, events.jsonl, events.log and the face crops."""
import json
import logging
import os
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS faces(face_id INTEGER PRIMARY KEY, created_video_s REAL, created_at TEXT, embedding BLOB);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, face_id INTEGER, event TEXT, video_time_s REAL,
    x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER, crop TEXT, created_at TEXT);
"""


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


class Store:
    def __init__(self, out_dir):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.out / "visitors.db")
        self.db.executescript(SCHEMA)
        self._reconcile_jsonl()
        self.jsonl = open(self.out / "events.jsonl", "a", encoding="utf-8")
        self.pending_lines = []
        self.pending_crops = []

        self.log = logging.getLogger(f"vcount.{self.out.resolve()}")
        self.log.setLevel(logging.INFO)
        self.log.propagate = False
        self._handler = logging.FileHandler(self.out / "events.log", encoding="utf-8")
        self._handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        self.log.addHandler(self._handler)

    def _reconcile_jsonl(self):
        """After a crash: drop a torn last line, then re-append lines for committed rows that missed theirs.
        (Rows are committed before their line is written, so the database is never behind events.jsonl.)"""
        path = self.out / "events.jsonl"
        if not path.exists():
            return
        lines = []
        for raw in path.read_text(encoding="utf-8").splitlines(keepends=True):
            if not raw.endswith("\n"):
                continue
            try:
                lines.append((raw, json.loads(raw)))
            except json.JSONDecodeError:
                continue
        have = Counter((d["face_id"], d["event"], d["video_time_s"]) for _, d in lines)
        missing = []
        rows = self.db.execute("SELECT face_id, event, video_time_s, x1, y1, x2, y2, crop FROM events ORDER BY id")
        for face_id, event, t, x1, y1, x2, y2, crop in rows:
            if have[(face_id, event, t)] > 0:
                have[(face_id, event, t)] -= 1
            else:
                line = {"face_id": face_id, "event": event, "video_time_s": t, "bbox": [x1, y1, x2, y2], "crop": crop}
                missing.append(json.dumps(line) + "\n")
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("".join(raw for raw, _ in lines) + "".join(missing))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def load_faces(self):
        rows = self.db.execute("SELECT face_id, embedding FROM faces ORDER BY face_id").fetchall()
        return [(fid, np.frombuffer(blob, dtype=np.float32).copy()) for fid, blob in rows]

    def add_face(self, emb, video_s):
        cur = self.db.execute(
            "INSERT INTO faces(created_video_s, created_at, embedding) VALUES (?, ?, ?)",
            (round(video_s, 2), now_iso(), emb.astype(np.float32).tobytes()),
        )
        self.db.commit()
        return cur.lastrowid

    def add_event(self, ev):
        t = round(float(ev.video_time_s), 2)
        bbox = [int(round(v)) for v in ev.bbox]
        folder = "entries" if ev.event == "entry" else "exits"
        rel = f"logs/{folder}/{datetime.now():%Y-%m-%d}/{ev.face_id}_{ev.event}_{t:.2f}.jpg"
        path = self.out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        self.pending_crops.append((path, ev.crop))

        self.db.execute(
            "INSERT INTO events(face_id, event, video_time_s, x1, y1, x2, y2, crop, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ev.face_id, ev.event, t, *bbox, rel, now_iso()),
        )
        self.db.commit()
        line = {"face_id": ev.face_id, "event": ev.event, "video_time_s": t, "bbox": bbox, "crop": rel}
        self.pending_lines.append(json.dumps(line))
        self.log.info("t=%.2f %s face_id=%d bbox=%s crop=%s", t, ev.event, ev.face_id, bbox, rel)

    def flush(self):
        """Commit this frame's rows, append their events.jsonl lines, then save the crops."""
        if self.db.in_transaction:
            self.db.commit()
        if self.pending_lines:
            self.jsonl.write("".join(line + "\n" for line in self.pending_lines))
            self.jsonl.flush()
            os.fsync(self.jsonl.fileno())
            self.pending_lines = []
        for path, crop in self.pending_crops:
            cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.pending_crops = []

    def close(self):
        self.flush()
        self.db.close()
        self.jsonl.close()
        self.log.removeHandler(self._handler)
        self._handler.close()

    def save_frame(self, frame, ev):
        path = self.out / f"logs/frames/{datetime.now():%Y-%m-%d}/{ev.face_id}_{ev.event}_{ev.video_time_s:.2f}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame)
