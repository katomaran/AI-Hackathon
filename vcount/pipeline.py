"""Per-frame processing: detect -> track -> recognise/register -> entry/exit events."""
import queue
import threading
import time

import numpy as np

from .align import align_face, face_quality
from .detector import Detection
from .recognition import Gallery
from .tracker import Tracker
from .visits import Visits


def crop_face(frame, bbox, margin=0.25):
    x1, y1, x2, y2 = bbox
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    h, w = frame.shape[:2]
    x1, y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    x2, y2 = int(min(w, x2 + mx)), int(min(h, y2 + my))
    return frame[y1:y2, x1:x2].copy()


def unmirror(dets, width):
    """Detections found on a horizontally flipped frame, in the original frame's coordinates."""
    out = []
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        kps = d.kps[[1, 0, 2, 4, 3]].copy()  # left and right swap in a mirror image
        kps[:, 0] = width - kps[:, 0]
        out.append(Detection(np.array([width - x2, y1, width - x1, y2], np.float32), d.score, kps))
    return out


class Pipeline:
    def __init__(self, cfg, detector, embedder, store):
        self.detector = detector
        self.embedder = embedder
        self.store = store
        self.log = store.log
        self.skip = max(1, int(cfg["detection_frame_skip"]))
        self.min_face_px = cfg["min_face_px"]
        self.registration_min_quality = cfg["registration_min_quality"]
        self.gallery = Gallery(cfg["match_threshold"])
        for face_id, emb in store.load_faces():
            self.gallery.add(face_id, emb)
        self.tracker = Tracker(cfg["track_max_age_s"])
        self.visits = Visits(cfg["exit_grace_s"])
        self.frame_idx = 0
        self.log.info("start: %d known faces", len(self.gallery.embs))

    def process(self, frame, t):
        self.frame_idx += 1
        if (self.frame_idx - 1) % self.skip:
            return  # detection runs on every `detection_frame_skip`-th frame

        dets = self.detector.detect(frame)
        if not dets:
            # Nothing found: try the mirror image as well. The detector misses more faces turned to
            # one side than to the other, and mirroring swaps the sides.
            dets = unmirror(self.detector.detect(np.ascontiguousarray(frame[:, ::-1])), frame.shape[1])
        dets = [d for d in dets if d.size >= self.min_face_px]
        matched, new, removed = self.tracker.update(dets, t)
        for tr in new:
            tr.first_crop = crop_face(frame, tr.det.bbox)
            self.log.info("t=%.2f tracking start track=%d bbox=%s", t, tr.track_id, tr.det.bbox.astype(int).tolist())
        for tr in removed:
            self.log.info("t=%.2f tracking lost track=%d face_id=%s", t, tr.track_id, tr.face_id)

        for tr in matched + new:
            self._recognise(tr, frame, t)
            if tr.face_id is None:
                continue
            ev = self.visits.seen(
                tr.face_id, t, tr.det.bbox.copy(), crop_face(frame, tr.det.bbox), tr.first_t, tr.first_bbox, tr.first_crop
            )
            if ev:
                self.store.add_event(ev)
                self.store.save_frame(frame, ev)
        for ev in self.visits.expire(t):
            self.store.add_event(ev)
            self.store.save_frame(frame, ev)
        self.store.flush()

    def _recognise(self, tr, frame, t):
        aligned = align_face(frame, tr.det)
        quality = face_quality(tr.det, aligned)
        if quality < self.registration_min_quality:
            return  # blurry, small or side-on: wait for a better crop
        emb = self.embedder.embed(aligned)
        best_id, sim = self.gallery.match(emb)
        self.log.info("t=%.2f embedding track=%d quality=%.2f best=%s sim=%.3f", t, tr.track_id, quality, best_id, sim)

        if tr.face_id is None:
            if best_id is not None:
                tr.face_id = best_id
                self.log.info("t=%.2f recognition track=%d face_id=%d sim=%.3f", t, tr.track_id, best_id, sim)
            else:
                tr.face_id = self.store.add_face(emb, t)
                self.gallery.add(tr.face_id, emb)
                self.log.info("t=%.2f registration face_id=%d track=%d quality=%.2f", t, tr.face_id, tr.track_id, quality)
            return

        own = self.gallery.similarity(tr.face_id, emb)
        if best_id is not None and best_id != tr.face_id and own < self.gallery.threshold:
            # The tracker followed the wrong face (people crossing): switch identity from here on.
            self.log.info("t=%.2f recognition track=%d face_id=%d->%d sim=%.3f", t, tr.track_id, tr.face_id, best_id, sim)
            tr.face_id = best_id
            tr.first_t, tr.first_bbox, tr.first_crop = t, tr.det.bbox.copy(), crop_face(frame, tr.det.bbox)
        elif own >= self.gallery.threshold:
            self.gallery.add(tr.face_id, emb)

    def finish(self):
        """End of a file: close every open visit."""
        for ev in self.visits.close_all():
            self.store.add_event(ev)
        self.store.flush()


PREFETCH_FRAMES = 48  # two seconds of 25 fps video, so a slow detector frame never stalls the decoder
END = object()


class Prefetcher:
    """Reads the source on a background thread, up to `depth` frames ahead of the frame loop."""

    def __init__(self, source, depth=PREFETCH_FRAMES):
        self.source = source
        self.queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self):
        while not self._stop.is_set():
            item = self.source.read(timeout=0.5)
            if item is None and self.source.ended:
                item = END
            while not self._stop.is_set():
                try:
                    self.queue.put(item, timeout=0.5)
                    break
                except queue.Full:
                    pass
            if item is END:
                return

    def get(self, timeout=0.5):
        """Next (frame, video_time_s, read_time), END after the last frame, or None if nothing came in time."""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5)


def run(pipeline, source, stop):
    """Frame loop. Returns the metrics.json dict."""
    latencies, done = [], []  # per frame: read-to-finished in ms, finish time in s since start
    frames = Prefetcher(source)
    ended = False
    start = time.perf_counter()
    try:
        while not stop.is_set():
            item = frames.get()
            if item is END:
                ended = True
                break
            if item is None:
                continue
            frame, video_t, read_t = item
            pipeline.process(frame, video_t)
            now = time.perf_counter()
            latencies.append((now - read_t) * 1000.0)
            done.append(now - start)
    finally:
        frames.close()
    wall = time.perf_counter() - start
    if ended:
        pipeline.finish()

    lat = np.array(latencies) if latencies else np.zeros(1)
    return {
        "frames_read": source.frames_read,
        "frames_processed": len(latencies),
        "wall_s": round(wall, 2),
        "fps": round(len(latencies) / wall, 2) if wall > 0 else 0.0,
        "latency_ms_p50": round(float(np.percentile(lat, 50)), 1),
        "latency_ms_p95": round(float(np.percentile(lat, 95)), 1),
        "latency_ms_max": round(float(lat.max()), 1),
        "fps_by_tenth": fps_by_tenth(done),
    }


def fps_by_tenth(done):
    """Throughput over each tenth of the run (equal frame counts): shows a pipeline that slows down as it runs."""
    if len(done) < 20:
        return []
    edges = np.linspace(0, len(done), 11).astype(int)
    t = np.concatenate([[0.0], done])
    return [round(float((b - a) / max(t[b] - t[a], 1e-9)), 2) for a, b in zip(edges[:-1], edges[1:])]
