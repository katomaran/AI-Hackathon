"""Unique visitor counter.

    python run.py --source <video file | rtsp://...> --config config.json --out <dir>
"""
import argparse
import json
import os
import signal
import threading
from pathlib import Path

from vcount.models import load_models
from vcount.pipeline import Pipeline, run
from vcount.source import open_source
from vcount.store import Store


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    base = Path(path).resolve().parent
    cfg["models"] = {k: str(base / v) for k, v in cfg["models"].items()}
    return cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file or rtsp:// URL")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--out", required=True, help="output directory (rerunning into it continues it)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    store = Store(args.out)
    detector, embedder = load_models(cfg)
    pipeline = Pipeline(cfg, detector, embedder, store)
    source = open_source(args.source)
    try:
        metrics = run(pipeline, source, stop)
    finally:
        source.close()
        visitors = store.db.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
        store.close()

    tmp = Path(args.out) / "metrics.json.tmp"
    tmp.write_text(json.dumps(metrics, indent=2))
    os.replace(tmp, Path(args.out) / "metrics.json")
    print(f"{metrics['frames_processed']} frames at {metrics['fps']} fps, {visitors} unique visitors")


if __name__ == "__main__":
    main()
