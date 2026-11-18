"""ArcFace embeddings and the per-visitor gallery they are matched against."""
import cv2
import numpy as np


class Embedder:
    """InsightFace ArcFace (w600k_mbf). Takes an aligned 112x112 BGR crop, as InsightFace does."""

    def __init__(self, session):
        self.session = session
        self.input_name = session.get_inputs()[0].name

    def embed(self, aligned_bgr):
        # Same preprocessing as insightface ArcFaceONNX: (x - 127.5) / 127.5, BGR -> RGB via swapRB.
        blob = cv2.dnn.blobFromImage(aligned_bgr, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        out = self.session.run(None, {self.input_name: blob})
        v = out[0][0]
        # Unit length, so the dot product of two embeddings is their cosine similarity.
        return v / np.linalg.norm(v)


class Gallery:
    """A few embeddings per face_id; a query matches the id holding its most similar embedding."""

    def __init__(self, threshold, max_per_id=8):
        # match_threshold is a cosine similarity. The default in config.json is a round starting
        # value for w600k_mbf, not tuned on any dataset.
        self.threshold = threshold
        self.max_per_id = max_per_id
        self.embs = {}  # face_id -> list of embeddings; index 0 is the registration embedding
        self._matrix = None

    def add(self, face_id, emb):
        lst = self.embs.setdefault(face_id, [])
        lst.append(emb.astype(np.float32))
        if len(lst) > self.max_per_id:
            lst = lst[:1] + lst[2:]  # keep the registration embedding, drop the oldest extra
        self._matrix = None

    def similarity(self, face_id, emb):
        return float(np.max(np.stack(self.embs[face_id]) @ emb)) if face_id in self.embs else -1.0

    def match(self, emb):
        """Best (face_id, similarity); face_id is None when nothing reaches the threshold."""
        if not self.embs:
            return None, -1.0
        if self._matrix is None:
            ids = [fid for fid, lst in self.embs.items() for _ in lst]
            self._matrix = (np.array(ids), np.stack([e for lst in self.embs.values() for e in lst]))
        ids, mat = self._matrix
        sims = mat @ emb
        i = int(np.argmax(sims))
        best_id, best = int(ids[i]), float(sims[i])
        return (best_id if best >= self.threshold else None), best
