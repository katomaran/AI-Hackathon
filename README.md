# Visitor Counter

Counts unique visitors in a video file or a live RTSP stream by their faces, and logs when each one enters and leaves.

Each frame goes through: face detection (YOLOv8n-face, with 5 landmarks) → tracking → ArcFace embedding (InsightFace `w600k_mbf`) → matching against known faces, or registering a new one → entry and exit events. Everything runs on the CPU with ONNX Runtime and OpenCV.

## Setup

Python 3.11 or 3.12.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_models.py    # downloads into models/ and checks sha256
```

## Videos

Open https://drive.google.com/drive/folders/1ItGjsE9Txn2nJzQ8VAMUNOMkwaDMzCw8 and download the 23 videos in its **Video Datasets** folder (22 `record_20250620_*.mp4` recordings and `video_sample1.mp4`) into `data/`. Use whichever you like. Video files are git-ignored.

## Run

```bash
python run.py --source data/record_20250620_190324.mp4 --config config.json --out out/
```

`--source` takes a video file or an `rtsp://` URL; `--out` is the output directory.

- A file is processed to the end, then the program exits with code 0.
- A live stream runs until Ctrl+C or SIGTERM.
- If a live stream drops, it is reconnected.
- Ctrl+C or SIGTERM stops cleanly at any time: pending events are written and `metrics.json` is saved.
- Running again into an existing output directory continues it: known faces are loaded from `visitors.db`, so a returning visitor keeps their `face_id`.

Speed figures for each run (FPS and per-frame latency) are written to `out/metrics.json`, described below.

## Scorecard (`eval.py`)

```bash
python eval.py --source data/record_20250620_190324.mp4 --config config.json --out out/
```

`eval.py` deletes earlier results in `--out`, runs `run.py` on the video as a separate process, measures it from outside, and prints the scorecard (also saved as `out/scorecard.json`):

| Row | What it reports |
| --- | --- |
| Speed | Frames per second, as a multiple of real time, and the first and last tenth of the run (a pipeline that slows down as it runs shows here). With `--baseline`, the speedup over the baseline. |
| Latency | Per frame, from read to processed: typical (p50), p95 and worst case. |
| CPU | CPU-seconds used, and the average number of cores busy. |
| Memory | Peak resident memory, and growth from early in the run to the end. |
| Output check | Every entry has an exit, and every event has its face crop (present and decodable), its database row and its `events.log` line. With `--baseline`, whether the events still match the baseline's. |

Options:

- `--save-baseline baseline.json` saves this scorecard and its events. Run it once on the unmodified code, then compare every later run with `--baseline baseline.json`: it shows the speedup and whether your changes altered the output.
- `--repeat N` plays the video N times back to back in one run, to see a longer run (needs `ffmpeg` on `PATH`).
- `--kill-after S` waits S seconds, then kills the run without warning the moment it next writes to its database, reruns into the same `--out` to the end, then checks the output: a crash must not leave broken or inconsistent output.

The scorecard needs no labels and does not judge who was counted; it shows how fast and how heavy the pipeline is, and that its output is complete and consistent.

## Configuration (`config.json`)

| Key | Meaning |
| --- | --- |
| `detection_frame_skip` | Run detection on every N-th frame (1 = every frame). |
| `det_input_size` | Detector input size in pixels; a multiple of 32. |
| `det_conf` | Minimum detector confidence. |
| `min_face_px` | Faces whose shorter side is smaller than this are ignored. |
| `track_max_age_s` | Seconds a track is kept without a matching detection. |
| `match_threshold` | Cosine similarity at or above which a face matches a known visitor. |
| `registration_min_quality` | Crop quality threshold (0–1). Quality is the lowest of detector score, face size, sharpness and frontalness. |
| `exit_grace_s` | Grace period for exits, in seconds. |
| `num_threads` | CPU threads for ONNX Runtime and OpenCV (0 = all cores). |
| `models` | Paths to the detector and recogniser ONNX files, relative to `config.json`. |

## Output (`--out` directory)

```
events.jsonl            one JSON object per event
events.log              human-readable log: tracking, embedding, recognition, registration, entry, exit
visitors.db             SQLite: faces and events
metrics.json            speed figures, written at shutdown
logs/entries/YYYY-MM-DD/<face_id>_entry_<video_time_s>.jpg
logs/exits/YYYY-MM-DD/<face_id>_exit_<video_time_s>.jpg
```

`events.jsonl`:

```json
{"face_id": 7, "event": "entry", "video_time_s": 12.48, "bbox": [412, 188, 498, 290], "crop": "logs/entries/2026-12-01/7_entry_12.48.jpg"}
```

- `face_id`: the visitor's id, the same across visits and reruns into the same output directory.
- `event`: `entry` or `exit`. Each visit has one of each; visits still open when a live stream is stopped have no exit.
- `video_time_s`: seconds from the start of the source, from the stream's timestamps. An entry is stamped when the face is first seen, an exit when it was last seen.
- `bbox`: `[x1, y1, x2, y2]` in source-frame pixels.
- `crop`: the saved face image, relative to the output directory.

`visitors.db`:

```sql
faces(face_id INTEGER PRIMARY KEY, created_video_s REAL, created_at TEXT, embedding BLOB)
events(id INTEGER PRIMARY KEY, face_id INTEGER, event TEXT, video_time_s REAL,
       x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER, crop TEXT, created_at TEXT)
```

Unique visitors: `SELECT COUNT(*) FROM faces`. Embeddings are 512 float32 values.

`metrics.json`:

```json
{"frames_read": 4500, "frames_processed": 4500, "wall_s": 180.2, "fps": 24.97, "latency_ms_p50": 31.0, "latency_ms_p95": 58.0}
```

Latency is per frame, from when the frame is read to when its processing finishes. `metrics.json` also has `latency_ms_max` (worst frame) and `fps_by_tenth` (throughput over each tenth of the run), which `eval.py` reads.

## Code

```
run.py                  command line, signals, metrics.json
eval.py                 the scorecard
vcount/source.py        video file and live stream input
vcount/models.py        model loading and thread settings
vcount/detector.py      YOLOv8n-face detector
vcount/align.py         face crops for the recogniser, and crop quality
vcount/recognition.py   ArcFace embeddings and the face gallery
vcount/tracker.py       IoU tracker
vcount/visits.py        entry and exit logic
vcount/pipeline.py      the per-frame pipeline and the frame loop
vcount/store.py         database, events.jsonl, events.log and crops
```

## Model licences

- `w600k_mbf.onnx` (InsightFace `buffalo_s`): InsightFace's pretrained models are available for **non-commercial research purposes only**. See https://github.com/deepinsight/insightface#license.
- `yolov8n-face.onnx`: YOLOv8-face by derronqi (https://github.com/derronqi/yolov8-face), GPL-3.0, built on Ultralytics YOLOv8 (AGPL-3.0). The ONNX export comes from https://github.com/hpc203/yolov8-face-landmarks-opencv-dnn.
