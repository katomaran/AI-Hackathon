"""Download the model weights into models/ and check their sha256.

    python scripts/download_models.py

Sources (pinned):
- yolov8n-face.onnx: YOLOv8n-face (derronqi/yolov8-face, GPL-3.0), the ONNX export from
  hpc203/yolov8-face-landmarks-opencv-dnn, the implementation the upstream README links to.
  The input height/width are marked dynamic after download so `det_input_size` can be changed;
  the weights themselves are not touched.
- w600k_mbf.onnx: ArcFace MobileFaceNet from InsightFace's buffalo_s pack (official release).
"""
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import onnx

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

DETECTOR = {
    "file": "yolov8n-face.onnx",
    "url": "https://raw.githubusercontent.com/hpc203/yolov8-face-landmarks-opencv-dnn/"
    "3deb14bb4454527fe81e94383c771078efc7c465/weights/yolov8n-face.onnx",
    "sha256": "d22820ff253677420e56562f8351955f741bfed7ab4960cf74aed543f52afeb5",
}
RECOGNIZER = {
    "file": "w600k_mbf.onnx",
    "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip",
    "sha256": "d85a87f503f691807cd8bb97128bdf7a0660326cd9cd02657127fa978bab8b5e",
    "member": "w600k_mbf.onnx",
    "member_sha256": "9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f",
}


def fetch(url, sha256):
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        sys.exit(f"sha256 mismatch for {url}\n  expected {sha256}\n  got      {got}")
    return data


def make_input_dynamic(model_bytes):
    m = onnx.load_from_string(model_bytes)
    dims = m.graph.input[0].type.tensor_type.shape.dim
    dims[2].dim_param, dims[3].dim_param = "height", "width"
    for out in m.graph.output:
        od = out.type.tensor_type.shape.dim
        od[2].dim_param, od[3].dim_param = f"{out.name}_h", f"{out.name}_w"
    del m.graph.value_info[:]
    return m.SerializeToString()


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    det_path = MODELS_DIR / DETECTOR["file"]
    if not det_path.exists():
        det_path.write_bytes(make_input_dynamic(fetch(DETECTOR["url"], DETECTOR["sha256"])))

    rec_path = MODELS_DIR / RECOGNIZER["file"]
    if not rec_path.exists():
        with zipfile.ZipFile(io.BytesIO(fetch(RECOGNIZER["url"], RECOGNIZER["sha256"]))) as z:
            data = z.read(RECOGNIZER["member"])
        if hashlib.sha256(data).hexdigest() != RECOGNIZER["member_sha256"]:
            sys.exit(f"sha256 mismatch for {RECOGNIZER['member']}")
        rec_path.write_bytes(data)

    print(f"models ready in {MODELS_DIR}")


if __name__ == "__main__":
    main()
