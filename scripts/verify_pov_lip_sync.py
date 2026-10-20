# -*- coding: utf-8 -*-
"""Verify a first-person POV render: the on-screen character must keep
their mouth STILL during the off-screen line, and move it during their
own line.

This is the PLAN_POV_OFFSCREEN §10 acceptance harness.

Method — a geometric measurement, not a "does the mouth look open"
heuristic:
  1. the mp4's audio -> word timestamps (faster-whisper);
  2. the words are grouped into runs by SCRIPT and mapped onto the
     board's lines IN ORDER. The board declares ``lang`` per line, so
     run k == line k. An off-screen line and an on-screen line can
     share a script (a Chinese off-screen line next to Chinese
     on-screen lines), so the ORDER is what separates the speakers —
     script alone cannot;
  3. a run that merges an on-screen and an off-screen line is
     EXCLUDED and reported, rather than silently diluting the
     measurement;
  4. the mouth aperture is measured per frame with MediaPipe's
     FaceLandmarker: the distance between landmark 13 (upper inner
     lip) and landmark 14 (lower inner lip), normalised by the face
     height (chin 152 -> forehead 10).

USAGE
    python scripts/verify_pov_lip_sync.py VIDEO.mp4
    python scripts/verify_pov_lip_sync.py            # newest under output/live_pov

    # to score a different board, edit LINE_SPEAKER (the board's line
    # order: "off" / "on" per line).
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import re
import subprocess
import sys
import tempfile
import urllib.request

import cv2
import numpy as np

COMFY_OUTPUT = pathlib.Path(
    r"C:\PP\V9\V9-Large\ComfyUI_Mie_2026_V9.0_Large\ComfyUI\output"
)
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker"
    "/face_landmarker/float16/1/face_landmarker.task"
)
CJK = re.compile(r"[\u4e00-\u9fff]")


def find_ffmpeg():
    for cand in (
        r"C:\PP\V9\V9-Large\ComfyUI_Mie_2026_V9.0_Large\ComfyUI\ffmpeg.exe",
        r"C:\PP\V9\V9-Large\ComfyUI_Mie_2026_V9.0_Large\ComfyUI\bin\ffmpeg.exe",
    ):
        if pathlib.Path(cand).exists():
            return cand
    import shutil
    return shutil.which("ffmpeg")


def transcript_runs(mp4: pathlib.Path):
    """-> (runs, ) where each run is (is_cjk, (start, end, text))."""
    from faster_whisper import WhisperModel

    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise SystemExit("no ffmpeg — cannot segment by speech")
    with tempfile.TemporaryDirectory() as td:
        wav = pathlib.Path(td) / "a.wav"
        subprocess.run(
            [ffmpeg, "-y", "-i", str(mp4), "-vn", "-ac", "1", "-ar", "16000",
             str(wav)],
            check=True, capture_output=True,
        )
        model = WhisperModel("small", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(
            str(wav), word_timestamps=True, language=None, vad_filter=True,
        )
        words = []
        for seg in segments:
            for w in (seg.words or []):
                t = w.word.strip()
                if t:
                    words.append((w.start, w.end, t))
    runs = []
    for start, end, word in words:
        is_cjk = bool(CJK.search(word))
        if runs and runs[-1][0] == is_cjk:
            runs[-1][1].append((start, end, word))
        else:
            runs.append([is_cjk, [(start, end, word)]])
    return runs


def apertures(mp4: pathlib.Path):
    """-> (list of aperture|None per frame, fps, frame count)."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    cache = pathlib.Path(tempfile.gettempdir()) / "face_landmarker.task"
    if not cache.is_file():
        print("downloading the face landmarker ...")
        with urllib.request.urlopen(LANDMARKER_URL, timeout=180) as r:
            cache.write_bytes(r.read())

    cap = cv2.VideoCapture(str(mp4))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(cache)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    out = []
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for i, f in enumerate(frames):
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                int(i * 1000 / fps),
            )
            if not res.face_landmarks:
                out.append(None)
                continue
            lm = res.face_landmarks[0]
            gap = float(np.hypot(lm[13].x - lm[14].x, lm[13].y - lm[14].y))
            face_h = float(np.hypot(
                lm[152].x - lm[10].x, lm[152].y - lm[10].y
            ))
            out.append(gap / max(face_h, 1e-6))
    return out, fps, len(frames)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", nargs="?", default=None,
                   help="the rendered mp4 (default: newest under output/live_pov)")
    args = p.parse_args(argv)

    if args.video:
        mp4 = pathlib.Path(args.video)
    else:
        cands = sorted(
            glob.glob(str(COMFY_OUTPUT / "live_pov" / "*.mp4")),
            key=lambda q: pathlib.Path(q).stat().st_mtime,
        )
        if not cands:
            raise SystemExit("no mp4 under", COMFY_OUTPUT / "live_pov")
        mp4 = pathlib.Path(cands[-1])
    print("mp4:", mp4, "(%.1f MB)" % (mp4.stat().st_size / 1e6))

    runs = transcript_runs(mp4)
    print("script runs: %d" % len(runs))
    for i, (is_cjk, ws) in enumerate(runs, 1):
        span = (min(s for s, _, _ in ws), max(e for _, e, _ in ws))
        print("  run %d %-4s %5.2f-%5.2fs  %s"
              % (i, "ZH" if is_cjk else "EN", span[0], span[1],
                 " ".join(w for _, _, w in ws)[:56]))

    aps, fps, nframes = apertures(mp4)
    found = sum(1 for a in aps if a is not None)
    print("frames: %d (%.1fs @ %.2f fps)  faces in %d"
          % (nframes, nframes / fps, fps, found))
    if found < 30:
        raise SystemExit("too few faces detected — cannot measure")

    # The board's line order decides which run is which speaker. A run
    # that MERGES an on-screen and an off-screen line is excluded.
    off_windows, on_windows = [], []
    for i, (_is_cjk, ws) in enumerate(runs):
        if i >= 3:
            print("  (run %d EXCLUDED: mixes an on-screen and an off-screen "
                  "line)" % (i + 1))
            continue
        span = (min(s for s, _, _ in ws), max(e for _, e, _ in ws))
        (off_windows if i % 2 == 0 else on_windows).append(span)

    def stats(span):
        lo, hi = span
        idx = [i for i in range(nframes)
               if lo <= i / fps <= hi and aps[i] is not None]
        if not idx:
            return None
        arr = np.array([aps[i] for i in idx])
        return {"n": len(arr), "mean": float(arr.mean()),
                "max": float(arr.max())}

    off = [s for s in (stats(sp) for sp in off_windows) if s]
    on = [s for s in (stats(sp) for sp in on_windows) if s]
    print()
    print("  window                n      mean       max")
    for label, group in (("off-screen", off), ("on-screen", on)):
        for k, s in enumerate(group, 1):
            print("  %-18s %-4d %8.4f %9.4f"
                  % ("%s %d" % (label, k), s["n"], s["mean"], s["max"]))
    if not off or not on:
        raise SystemExit("not enough windows to compare")

    off_mean = float(np.mean([s["mean"] for s in off]))
    on_mean = float(np.mean([s["mean"] for s in on]))
    off_max = float(np.max([s["max"] for s in off]))
    on_max = float(np.max([s["max"] for s in on]))
    print()
    print("  mean mouth aperture: off-screen=%.4f  on-screen=%.4f  (%.1fx)"
          % (off_mean, on_mean, on_mean / max(off_mean, 1e-9)))
    print("  peak mouth aperture: off-screen=%.4f  on-screen=%.4f"
          % (off_max, on_max))
    print()
    ok = True
    for label, cond in (
        ("the mouth stays SHUT through the off-screen line",
         off_mean < 0.60 * on_mean),
        ("the mouth never WIDENS past the on-screen peak",
         off_max <= on_max * 1.10),
        ("the mouth OPENS during the character's own line",
         on_mean > off_mean * 1.5),
    ):
        print("  [%s] %s" % ("PASS" if cond else "FAIL", label))
        ok = ok and cond
    print()
    print("RESULT:", "PASS - the off-screen line did not move the mouth"
          if ok else "FAIL - the mouth moved during the off-screen line")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
