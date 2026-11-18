"""Scorecard: runs the visitor counter on a video, then reports speed, latency, CPU, memory and an output check.

    python eval.py --source data/record_20250620_190324.mp4 --config config.json --out out/

It starts `run.py` as a separate process and measures CPU and memory from outside it; latency comes from the
`metrics.json` the run writes. No labels are used: the output check only tests that the output is complete and
consistent. Earlier results in --out (events, database, crops, metrics) are deleted first.

  --save-baseline FILE  also save this scorecard and its events as a baseline (do it once on the unmodified code)
  --baseline FILE       compare with a saved baseline: speedup, and whether the events still match
  --repeat N            play the video N times back to back in one run (needs ffmpeg on PATH): a long run
  --kill-after S        after S seconds, kill the run the moment it next writes to its database (no clean-up),
                        rerun into the same --out to the end, then check the output
"""
import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
import psutil

HERE = Path(__file__).resolve().parent
OUTPUTS = ("events.jsonl", "events.log", "visitors.db", "visitors.db-wal", "visitors.db-shm", "visitors.db-journal",
           "metrics.json", "metrics.json.tmp", "scorecard.json", "logs")
EVENT_TIME_TOLERANCE_S = 0.25  # when comparing events with a baseline


def clear_outputs(out):
    for name in OUTPUTS:
        p = out / name
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()


def looped_copy(src, n, tmp):
    """The video n times back to back, without re-encoding (ffmpeg concat keeps the timestamps increasing)."""
    if not shutil.which("ffmpeg"):
        sys.exit("--repeat needs ffmpeg on PATH")
    lst = tmp / "list.txt"
    lst.write_text("".join(f"file '{Path(src).resolve().as_posix()}'\n" for _ in range(n)))
    dst = tmp / f"{Path(src).stem}_x{n}{Path(src).suffix}"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(dst)],
                   check=True)
    return dst


class Sampler(threading.Thread):
    """Samples resident memory and CPU time of a process and its children every `every_s` seconds."""

    def __init__(self, proc, every_s=0.2):
        super().__init__(daemon=True)
        self.proc, self.every_s = psutil.Process(proc.pid), every_s
        self.samples = []  # (seconds since start, rss bytes, cpu seconds)
        self.done = threading.Event()

    def run(self):
        start = time.perf_counter()
        while not self.done.is_set():
            rss = cpu = 0.0
            try:
                for p in [self.proc] + self.proc.children(recursive=True):
                    try:
                        rss += p.memory_info().rss
                        c = p.cpu_times()
                        cpu += c.user + c.system
                    except psutil.Error:
                        pass
            except psutil.Error:
                break
            self.samples.append((time.perf_counter() - start, rss, cpu))
            self.done.wait(self.every_s)


def run_pipeline(source, config, out, kill_after=None):
    """run.py on the source. Returns {"returncode", "wall_s", "cpu_s", "peak_rss_mb", "samples", "killed"}."""
    cmd = [sys.executable, str(HERE / "run.py"), "--source", str(source), "--config", str(config), "--out", str(out)]
    before = _children_rusage()
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=HERE)
    sampler = Sampler(proc)
    sampler.start()
    killed = False
    try:
        proc.wait(timeout=kill_after)
    except subprocess.TimeoutExpired:
        stamp, deadline = db_stamp(out), time.monotonic() + 30
        while proc.poll() is None and db_stamp(out) == stamp and time.monotonic() < deadline:
            time.sleep(0.001)
        if proc.poll() is None:
            proc.kill()  # SIGKILL on Linux and macOS, TerminateProcess on Windows: no chance to clean up
            killed = True
        proc.wait()
    wall = time.perf_counter() - start
    sampler.done.set()
    sampler.join()
    after = _children_rusage()
    samples = sampler.samples
    cpu = samples[-1][2] if samples else 0.0
    peak = max((s[1] for s in samples), default=0)
    if before and after:  # exact figures where the OS keeps them for finished children
        cpu = max(cpu, after[0] - before[0])
        peak = max(peak, after[1])
    return {"returncode": proc.returncode, "wall_s": wall, "cpu_s": cpu, "peak_rss_mb": peak / 2**20,
            "samples": samples, "killed": killed}


def db_stamp(out):
    """Sizes and modification times of the database files: changes on every write."""
    stamp = []
    for suffix in ("", "-wal", "-journal"):
        try:
            st = (Path(out) / f"visitors.db{suffix}").stat()
            stamp.append((st.st_size, st.st_mtime_ns))
        except OSError:
            stamp.append(None)
    return stamp


def _children_rusage():
    """(CPU seconds, max RSS bytes) of finished child processes, or None on Windows."""
    try:
        import resource
    except ImportError:
        return None
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime, r.ru_maxrss * (1 if sys.platform == "darwin" else 1024)


def memory_growth(samples):
    """MB from after warm-up (20-30% into the run) to the end (last 10%), and per minute of run time between them."""
    if len(samples) < 20:
        return None, None
    end_t = samples[-1][0]

    def median_rss(a, b):
        v = sorted(s[1] for s in samples if a * end_t <= s[0] <= b * end_t)
        return v[len(v) // 2] if v else None

    early, late = median_rss(0.2, 0.3), median_rss(0.9, 1.0)
    if early is None or late is None:
        return None, None
    growth = (late - early) / 2**20
    return growth, growth / max((0.7 * end_t) / 60, 1e-9)


def read_events(out):
    """(events, unparseable line numbers) from events.jsonl."""
    events, bad = [], []
    path = out / "events.jsonl"
    if not path.exists():
        return events, bad
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            bad.append(n)
    return events, bad


def db_events(out):
    """(events rows as (face_id, event, video_time_s, crop), faces count), or raises sqlite3.Error."""
    db = sqlite3.connect(f"file:{(out / 'visitors.db').as_posix()}?mode=ro", uri=True)
    try:
        rows = db.execute("SELECT face_id, event, video_time_s, crop FROM events").fetchall()
        return rows, db.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
    finally:
        db.close()


def check_after_kill(out):
    """Right after a kill, before anything repairs it: every database row and every complete events.jsonl
    line must point at a crop that is there and decodes, and every line must have its database row."""
    problems = []
    path = out / "events.jsonl"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines = lines[:-1]  # a line cut off by the kill; the next start drops it
    events = []
    for raw in lines:
        try:
            events.append(json.loads(raw))
        except json.JSONDecodeError:
            problems.append("a complete events.jsonl line that is not JSON")
    try:
        rows, _ = db_events(out)
    except sqlite3.Error as err:
        return problems + [f"visitors.db unreadable after the kill: {err}"], set()
    broken = sorted({r[3] for r in rows if cv2.imread(str(out / r[3])) is None})
    broken += sorted({e.get("crop") for e in events if cv2.imread(str(out / str(e.get("crop")))) is None} - set(broken))
    if broken:
        problems.append(f"after the kill, {len(broken)} events point at a crop that is missing or incomplete, e.g. {broken[:2]}")
    in_db = Counter((r[0], r[1], round(float(r[2]), 2)) for r in rows)
    only_jsonl = Counter((e.get("face_id"), e.get("event"), round(float(e.get("video_time_s", -1)), 2)) for e in events) - in_db
    if only_jsonl:
        problems.append(f"after the kill, {sum(only_jsonl.values())} events.jsonl lines have no database row")
    open_faces = set()
    for r in sorted(rows, key=lambda r: r[2]):
        (open_faces.add if r[1] == "entry" else open_faces.discard)(r[0])
    return problems, open_faces


def check_output(out, open_at_kill=frozenset()):
    """Label-free checks of the output in --out. Returns (list of problems, summary dict).
    open_at_kill: faces whose visit was open when a kill test killed the run; the rerun replays the video from
    the start, so those faces get a second entry, which is expected."""
    problems = []
    events, bad = read_events(out)
    if bad:
        problems.append(f"events.jsonl lines that are not JSON: {bad[:5]}")
    ok_events = []
    for e in events:
        fields_ok = (isinstance(e.get("face_id"), int) and e.get("event") in ("entry", "exit")
                     and isinstance(e.get("video_time_s"), (int, float)) and isinstance(e.get("crop"), str)
                     and isinstance(e.get("bbox"), list) and len(e["bbox"]) == 4
                     and all(isinstance(v, int) for v in e["bbox"]))
        if fields_ok:
            ok_events.append(e)
        else:
            problems.append(f"malformed event {json.dumps(e)[:120]}")
    if not events:
        problems.append("no events in events.jsonl")

    open_visits = {}
    for e in ok_events:
        if e["event"] == "entry":
            if open_visits.get(e["face_id"]) and e["face_id"] not in open_at_kill:
                problems.append(f"face {e['face_id']}: two entries without an exit between them")
            open_visits[e["face_id"]] = True
        else:
            if not open_visits.get(e["face_id"]):
                problems.append(f"face {e['face_id']}: exit at {e['video_time_s']} s without an entry")
            open_visits[e["face_id"]] = False
    unclosed = sorted(fid for fid, is_open in open_visits.items() if is_open)
    if unclosed:
        problems.append(f"entries with no exit: faces {unclosed[:10]}")

    bad_crops = [e["crop"] for e in ok_events if cv2.imread(str(out / e["crop"])) is None]
    if bad_crops:
        problems.append(f"{len(bad_crops)} events whose crop is missing or does not decode, e.g. {bad_crops[:3]}")

    rows, faces = [], None
    try:
        rows, faces = db_events(out)
    except sqlite3.Error as err:
        problems.append(f"visitors.db unreadable: {err}")
    key = lambda e: (e["face_id"], e["event"], round(float(e["video_time_s"]), 2), e["crop"])  # noqa: E731
    in_jsonl = Counter(key(e) for e in ok_events)
    in_db = Counter((r[0], r[1], round(float(r[2]), 2), r[3]) for r in rows)
    if faces is not None and in_jsonl != in_db:
        only_j, only_d = in_jsonl - in_db, in_db - in_jsonl
        problems.append(f"database and events.jsonl disagree: {sum(only_j.values())} events only in events.jsonl, "
                        f"{sum(only_d.values())} only in the database")
    if faces is not None and faces < len({e["face_id"] for e in ok_events}):
        problems.append(f"faces table has {faces} rows but events name {len({e['face_id'] for e in ok_events})} faces")

    log = (out / "events.log").read_text(encoding="utf-8", errors="replace") if (out / "events.log").exists() else ""
    no_log = [e for e in ok_events if f"crop={e['crop']}" not in log]
    if no_log:
        problems.append(f"{len(no_log)} events with no line in events.log, e.g. {no_log[0]['event']} face {no_log[0]['face_id']}")
    return problems, {"events": len(ok_events), "visitors": len({e["face_id"] for e in ok_events}), "faces_table": faces}


def compare_events(events, base_events):
    """How many events match the baseline's: same type, same order, times within EVENT_TIME_TOLERANCE_S."""
    def by_type(evs):
        return {k: sorted(float(e["video_time_s"]) for e in evs if e.get("event") == k) for k in ("entry", "exit")}
    a, b = by_type(events), by_type(base_events)
    matched = sum(abs(x - y) <= EVENT_TIME_TOLERANCE_S for k in a for x, y in zip(a[k], b[k]))
    return matched, len(base_events), len(events)


def pct(a, b):
    return f"{100 * (a / b - 1):+.0f}%" if a and b else "n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 epilog=__doc__.split("\n\n", 2)[2], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="a video file")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-baseline", metavar="FILE")
    ap.add_argument("--baseline", metavar="FILE")
    ap.add_argument("--repeat", type=int, default=1, metavar="N")
    ap.add_argument("--kill-after", type=float, metavar="S")
    args = ap.parse_args()

    if "://" in args.source:
        sys.exit("eval.py scores a video file; use run.py directly for a live stream")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    clear_outputs(out)
    config = Path(args.config).resolve()
    tmp = Path(tempfile.mkdtemp(prefix="scorecard_"))
    try:
        source = looped_copy(args.source, args.repeat, tmp) if args.repeat > 1 else Path(args.source).resolve()
        cap = cv2.VideoCapture(str(source))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        kill, kill_problems, open_at_kill = None, [], set()
        if args.kill_after:
            first = run_pipeline(source, config, out, kill_after=args.kill_after)
            if first["killed"]:
                kill_problems, open_at_kill = check_after_kill(out)
                kill = "killed right after a database write; output checked, then rerun into the same --out"
            else:
                kill = f"not killed: it ended before writing again (exit code {first['returncode']})"
        r = run_pipeline(source, config, out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    try:
        m = json.loads((out / "metrics.json").read_text())
    except (OSError, json.JSONDecodeError):
        m = {}
    problems, summary = check_output(out, open_at_kill)
    problems = kill_problems + problems
    if r["returncode"] != 0:
        problems.insert(0, f"run.py exited with code {r['returncode']}")
    growth, growth_per_min = memory_growth(r["samples"])
    fps = m.get("fps")
    tenths = m.get("fps_by_tenth") or []
    cores = r["cpu_s"] / r["wall_s"] if r["wall_s"] else 0.0
    card = {
        "source": Path(args.source).name, "repeat": args.repeat, "video_frames": video_frames, "video_fps": video_fps,
        "speed": {"fps": fps, "x_real_time": round(fps / video_fps, 2) if fps else None,
                  "frames_read": m.get("frames_read"), "frames_processed": m.get("frames_processed"),
                  "fps_first_tenth": tenths[0] if tenths else None, "fps_last_tenth": tenths[-1] if tenths else None,
                  "wall_s_end_to_end": round(r["wall_s"], 2)},
        "latency_ms": {"p50": m.get("latency_ms_p50"), "p95": m.get("latency_ms_p95"), "max": m.get("latency_ms_max")},
        "cpu": {"cpu_s": round(r["cpu_s"], 1), "avg_cores": round(cores, 2),
                "avg_load_pct_of_machine": round(100 * cores / (psutil.cpu_count() or 1), 1)},
        "memory_mb": {"peak": round(r["peak_rss_mb"]), "growth": None if growth is None else round(growth, 1),
                      "growth_per_min": None if growth_per_min is None else round(growth_per_min, 2)},
        "output_check": {"passed": not problems, "problems": problems, **summary, "kill_test": kill},
    }
    events, _ = read_events(out)

    base = None
    if args.baseline:
        base = json.loads(Path(args.baseline).read_text())
        bc = base["scorecard"]
        matched, n_base, n_now = compare_events(events, base["events"])
        card["baseline"] = {
            "speedup": round(fps / bc["speed"]["fps"], 2) if fps and bc["speed"]["fps"] else None,
            "events_matching": matched, "events_in_baseline": n_base, "events_now": n_now,
            "visitors_in_baseline": bc["output_check"]["visitors"]}
    (out / "scorecard.json").write_text(json.dumps(card, indent=2))
    if args.save_baseline:
        Path(args.save_baseline).write_text(json.dumps({"scorecard": card, "events": events}))

    f = lambda v, fmt="": "n/a" if v is None else format(v, fmt)  # noqa: E731
    sp, lat, cpu, mem, oc = card["speed"], card["latency_ms"], card["cpu"], card["memory_mb"], card["output_check"]
    run_note = f", played {args.repeat} times" if args.repeat > 1 else ""
    print(f"\nScorecard: {card['source']} ({video_frames} frames, {video_frames / video_fps:.1f} s of video{run_note})")
    line = f"{f(fps)} fps, {f(sp['x_real_time'])}x real time (video is {video_fps:.2f} fps)"
    if sp["fps_first_tenth"]:
        line += (f"; over the run: first tenth {sp['fps_first_tenth']} fps, last tenth {sp['fps_last_tenth']} fps "
                 f"({pct(sp['fps_last_tenth'], sp['fps_first_tenth'])})")
    print(f"  Speed         {line}")
    if base:
        b = card["baseline"]
        print(f"                vs baseline: {f(b['speedup'])}x the baseline's {base['scorecard']['speed']['fps']} fps")
    print(f"  Latency       per frame: typical (p50) {f(lat['p50'])} ms, p95 {f(lat['p95'])} ms, worst {f(lat['max'])} ms")
    print(f"  CPU           {cpu['cpu_s']} CPU-seconds, {cpu['avg_cores']} cores busy on average "
          f"({cpu['avg_load_pct_of_machine']}% of {psutil.cpu_count()} logical cores)")
    print(f"  Memory        peak {mem['peak']} MB, growth during the run {f(mem['growth'], '+.1f')} MB "
          f"({f(mem['growth_per_min'], '+.2f')} MB per minute)")
    verdict = "PASS" if oc["passed"] else "FAIL"
    print(f"  Output check  {verdict}: {oc['events']} events, {oc['visitors']} visitors"
          + (" - every entry has an exit; every event has its crop, database row and log line" if oc["passed"] else ""))
    for p in problems[:10]:
        print(f"                - {p}")
    if kill:
        print(f"                kill test: {kill}")
    if base:
        b = card["baseline"]
        same = b["events_matching"] == b["events_in_baseline"] == b["events_now"]
        print(f"                vs baseline: {b['events_matching']} of {b['events_in_baseline']} baseline events matched "
              f"(within {EVENT_TIME_TOLERANCE_S} s), {b['events_now']} events now"
              + ("; output unchanged" if same else "; output differs from the baseline"))
    print(f"\nFull scorecard: {out / 'scorecard.json'}")
    return 0 if oc["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
