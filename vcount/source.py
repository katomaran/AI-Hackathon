"""Video input. Every frame comes with its video time in seconds."""
import threading
import time

import cv2


def open_source(src):
    return LiveSource(src) if "://" in src else VideoSource(src)


def open_stream(url, timeout_ms=5000):
    params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms]
    return cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)


class VideoSource:
    """Reads every frame of a video in order."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video {path}")
        self.frames_read = 0
        self.ended = False

    def read(self, timeout=None):
        """(frame, video_time_s, read_time) or None at the end of the file. Video time is the frame's PTS."""
        ok, frame = self.cap.read()
        if not ok:
            self.ended = True
            return None
        self.frames_read += 1
        return frame, self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, time.perf_counter()

    def close(self):
        self.cap.release()


class LiveSource:
    """RTSP (or any live URL). A reader thread decodes continuously and keeps only the latest frame,
    so a slow pipeline skips frames instead of falling behind. Reconnects when the stream drops."""

    def __init__(self, url, open_capture=open_stream, reconnect_s=1.0):
        self.url = url
        self.open_capture = open_capture
        self.reconnect_s = reconnect_s
        self.frames_read = 0
        self.ended = False  # a live stream never ends on its own
        self._latest = None
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        offset = 0.0  # keeps video time increasing across reconnects
        last_t = 0.0
        while not self._stop.is_set():
            cap = self.open_capture(self.url)
            first_pts = None
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if first_pts is None:
                    first_pts = pts
                last_t = offset + pts - first_pts
                with self._cond:
                    self._latest = (frame, last_t, time.perf_counter())
                    self.frames_read += 1
                    self._cond.notify()
            cap.release()
            offset = last_t + 1e-3
            self._stop.wait(self.reconnect_s)

    def read(self, timeout=0.5):
        """Latest frame not yet returned, or None if none arrived within timeout."""
        with self._cond:
            if self._latest is None:
                self._cond.wait(timeout)
            item, self._latest = self._latest, None
            return item

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5)
