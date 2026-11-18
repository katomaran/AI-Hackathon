"""Model loading and CPU thread settings. Models are loaded once, before the frame loop."""
import os

import cv2
import onnxruntime as ort

from .detector import Detector
from .recognition import Embedder


def set_threads(num_threads):
    n = num_threads if num_threads > 0 else os.cpu_count()
    cv2.setNumThreads(n)
    return n


def _session(path, num_threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = num_threads
    so.inter_op_num_threads = num_threads
    so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def load_models(cfg):
    n = set_threads(cfg["num_threads"])
    detector = Detector(cfg["models"]["detector"], cfg["det_input_size"], cfg["det_conf"])
    embedder = Embedder(_session(cfg["models"]["recognizer"], n))
    return detector, embedder
