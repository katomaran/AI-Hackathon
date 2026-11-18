"""Face alignment to the ArcFace 112x112 template, and a crop quality score."""
import cv2
import numpy as np

# Landmark positions ArcFace models were trained with (InsightFace's arcface_dst).
ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)

GOOD_FACE_PX = 80  # face size (short side) that scores 1.0
SHARP_LAPLACIAN_VAR = 100.0  # Laplacian variance of the aligned crop that scores 1.0


def similarity_transform(src, dst):
    """Least-squares rotation + uniform scale + translation mapping src points onto dst (Umeyama)."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(d.T @ s / len(src))
    e = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        e[1, 1] = -1
    r = u @ e @ vt
    scale = np.trace(np.diag(sig) @ e) / s.var(0).sum()
    t = mu_d - scale * r @ mu_s
    return np.hstack([scale * r, t[:, None]]).astype(np.float32)


def align_face(frame, det, size=112):
    """Warp the face so its 5 landmarks land on the ArcFace template."""
    m = similarity_transform(det.kps, ARCFACE_DST * (size / 112.0))
    return cv2.warpAffine(frame, m, (size, size), borderValue=0)


def frontalness(kps):
    """1.0 when the nose sits midway between the eyes, towards 0 as the head turns side-on."""
    le, re, nose = kps[0], kps[1], kps[2]
    axis = re - le
    eye_dist = np.linalg.norm(axis)
    if eye_dist < 1e-6:
        return 0.0
    along = float(np.dot(nose - le, axis / eye_dist))  # nose position along the eye line
    left, right = along, eye_dist - along
    if left <= 0 or right <= 0:
        return 0.0
    return min(left, right) / max(left, right)


def sharpness(aligned):
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    return min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / SHARP_LAPLACIAN_VAR)


def face_quality(det, aligned):
    """Lowest of detector score, size, sharpness and frontalness, each in [0, 1]."""
    size = min(1.0, det.size / GOOD_FACE_PX)
    return min(det.score, size, sharpness(aligned), frontalness(det.kps))
