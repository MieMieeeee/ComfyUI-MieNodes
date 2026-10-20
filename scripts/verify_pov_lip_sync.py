# -*- coding: utf-8 -*-
"""Complete, line-by-line POV acceptance against an explicit storyboard.

Usage: python scripts/verify_pov_lip_sync.py VIDEO --storyboard BOARD --report REPORT
Voice attribution and visible identity need per-line review evidence; missing
or uncertain evidence produces a nonzero exit status. No transcript runs are skipped.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import subprocess
import sys
import tempfile
import urllib.request

COMFY_OUTPUT = pathlib.Path(r"C:\PP\V9\V9-Large\ComfyUI_Mie_2026_V9.0_Large\ComfyUI\output")
LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

def find_ffmpeg():
    for cand in (
        r"C:\PP\V9\V9-Large\ComfyUI_Mie_2026_V9.0_Large\ComfyUI\ffmpeg.exe",
        r"C:\PP\V9\V9-Large\ComfyUI_Mie_2026_V9.0_Large\ComfyUI\bin\ffmpeg.exe",
    ):
        if pathlib.Path(cand).exists():
            return cand
    import shutil
    return shutil.which("ffmpeg")


def _normalized(text):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKC", str(text)).casefold() if c.isalnum())


def align_lines(lines, words):
    """Strict, ordered text alignment. Missing/extra speech never disappears.

    ASR mismatches require listening to confirm; they cannot be a PASS.
    A word shared by two lines cannot supply a trustworthy speaker boundary.
    """
    from difflib import SequenceMatcher
    expected_parts = [_normalized(line["text"]) for line in lines]
    expected = "".join(expected_parts)
    observed, owners = "", []
    errors, invalid_words = [], set()
    previous_start = -1.0
    for index, word in enumerate(words):
        start, end = float(word["start"]), float(word["end"])
        if start < previous_start or start < 0 or end <= start:
            errors.append("invalid transcript timestamp at word %d" % index)
            invalid_words.add(index)
        previous_start = start
        token = _normalized(word.get("text", word.get("word", "")))
        observed += token
        owners.extend([index] * len(token))
    mapping = {}
    for block in SequenceMatcher(None, expected, observed, autojunk=False).get_matching_blocks():
        mapping.update((block.a + j, block.b + j) for j in range(block.size))
    if expected != observed:
        errors.append("transcript differs from the storyboard: missing, changed or extra speech; listen to confirm")
    result, offset = [], 0
    for index, (line, token) in enumerate(zip(lines, expected_parts)):
        positions = [mapping.get(i) for i in range(offset, offset + len(token))]
        exact = bool(token) and all(i is not None for i in positions)
        if exact:
            exact = positions == list(range(positions[0], positions[0] + len(token)))
        word_ids = sorted({owners[i] for i in positions if i is not None})
        entry = dict(line, id=line.get("id", "line_%02d" % (index + 1)), text_ok=exact,
                     start=None, end=None, boundary_ok=not bool(set(word_ids) & invalid_words))
        if exact:
            entry.update(start=float(words[word_ids[0]]["start"]), end=float(words[word_ids[-1]]["end"]))
        result.append(entry)
        offset += len(token)
    for left, right in zip(result, result[1:]):
        if left["end"] is not None and right["start"] is not None and left["end"] > right["start"]:
            left["boundary_ok"] = right["boundary_ok"] = False
            errors.append("speaker boundary overlaps an ASR word: %s / %s" % (left["id"], right["id"]))
    return {"ok": not errors and bool(result) and all(x["text_ok"] for x in result),
            "errors": errors, "lines": result, "observed_text": observed}


def evaluate_lines(lines, words, apertures_by_name, fps, review=None,
                   max_closed_aperture=0.02, max_closed_motion=0.015,
                   min_speaking_motion=0.015, min_face_coverage=0.9):
    """Measure EVERY line and listener. Voice/identity require separate evidence."""
    import math
    if fps <= 0:
        raise ValueError("fps must be positive")
    alignment = align_lines(lines, words)
    reviews = review or {}
    for entry in alignment["lines"]:
        visible = entry.get("visible_speakers", list(apertures_by_name))
        entry["mouths"] = {}
        timed = entry["text_ok"] and entry["boundary_ok"] and entry["start"] is not None
        for name in visible:
            values = apertures_by_name.get(name, [])
            start = math.ceil(entry["start"] * fps - 1e-8) if timed else 0
            end = math.ceil(entry["end"] * fps - 1e-8) if timed else 0
            expected_count = max(0, end - start)
            samples = [x for x in values[start:end] if x is not None and math.isfinite(x)]
            coverage = len(samples) / expected_count if expected_count else 0.0
            span = max(samples) - min(samples) if samples else None
            peak = max(samples) if samples else None
            speaking = not entry.get("offscreen", False) and name == entry["speaker"]
            measured = timed and len(samples) >= 3 and coverage >= min_face_coverage
            ok = bool(measured and (span >= min_speaking_motion if speaking else
                      peak <= max_closed_aperture and span <= max_closed_motion))
            entry["mouths"][name] = {"ok": ok, "coverage": coverage, "samples": len(samples),
                                     "peak": peak, "motion": span, "speaking": speaking}
        entry["mouth_ok"] = bool(visible) and all(x["ok"] for x in entry["mouths"].values())
        evidence = reviews.get(entry["id"], {})
        entry["voice_ok"] = evidence.get("voice_speaker") == entry["speaker"]
        entry["visual_ok"] = sorted(evidence.get("visible_speakers", [])) == sorted(visible) and bool(visible)
        entry["ok"] = bool(timed and entry["mouth_ok"] and entry["voice_ok"] and entry["visual_ok"])
    return {"ok": alignment["ok"] and all(x["ok"] for x in alignment["lines"]),
            "errors": alignment["errors"], "lines": alignment["lines"],
            "thresholds": {"max_closed_aperture": max_closed_aperture,
                           "max_closed_motion": max_closed_motion,
                           "min_speaking_motion": min_speaking_motion,
                           "min_face_coverage": min_face_coverage},
            "notes": ["ASR mismatches need listening confirmation.",
                      "Unreviewed voice attribution or visible identities cannot pass.",
                      "Mouth thresholds are explicit geometric checks, not a perceptual guarantee."]}


def storyboard_lines(board):
    cast = board["cast"]
    shots = board.get("shots")
    if not shots:
        raise ValueError("acceptance requires explicit storyboard shots matching the rendered plan")
    result = []
    for shot_index, shot in enumerate(shots, 1):
        visible = [n for n, info in cast.items() if not info.get("offscreen") and n not in shot.get("absent", [])]
        for line_index, line in enumerate(shot["lines"], 1):
            result.append(dict(line, id="scene_%02d.line_%02d" % (shot_index, line_index),
                               offscreen=bool(cast[line["speaker"]].get("offscreen")), visible_speakers=visible))
    return result


def transcribe_words(mp4, model_name="small"):
    from faster_whisper import WhisperModel
    import inspect
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise ValueError("ffmpeg is required")
    with tempfile.TemporaryDirectory() as directory:
        wav = pathlib.Path(directory) / "speech.wav"
        subprocess.run([ffmpeg, "-y", "-i", str(mp4), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
                       check=True, capture_output=True)
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        options = dict(word_timestamps=True, language=None, vad_filter=True, task="transcribe")
        if "multilingual" in inspect.signature(model.transcribe).parameters:
            options["multilingual"] = True
        segments, _ = model.transcribe(str(wav), **options)
        return [{"text": word.word, "start": word.start, "end": word.end}
                for segment in segments for word in segment.words or [] if word.word.strip()]


def apertures(mp4: pathlib.Path):
    """-> (list of aperture|None per frame, fps, frame count)."""
    import cv2
    import numpy as np
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
        num_faces=2,
    )
    out = []
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        for i, f in enumerate(frames):
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                int(i * 1000 / fps),
            )
            if len(res.face_landmarks) != 1:
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
    import hashlib
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=pathlib.Path)
    parser.add_argument("--storyboard", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--transcript", type=pathlib.Path, help="JSON words with text/start/end")
    parser.add_argument("--mouth-data", type=pathlib.Path, help="JSON fps and named face tracks")
    parser.add_argument("--review", type=pathlib.Path, help="JSON keyed by scene_XX.line_XX: voice_speaker/visible_speakers")
    parser.add_argument("--model", default="small")
    args = parser.parse_args(argv)
    board = json.loads(args.storyboard.read_text(encoding="utf-8"))
    lines = storyboard_lines(board)
    words = json.loads(args.transcript.read_text(encoding="utf-8")) if args.transcript else transcribe_words(args.video, args.model)
    if isinstance(words, dict):
        words = words["words"]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.with_suffix(".words.json").write_text(json.dumps(words, ensure_ascii=False, indent=2), encoding="utf-8")
    extraction_errors = []
    if args.mouth_data:
        data = json.loads(args.mouth_data.read_text(encoding="utf-8"))
        tracks, fps = data["faces"], data["fps"]
    else:
        names = list(dict.fromkeys(n for line in lines for n in line["visible_speakers"]))
        if len(names) == 1:
            values, fps, _ = apertures(args.video)
            tracks = {names[0]: values}
        else:
            tracks, fps = {}, 24.0
            extraction_errors.append("multiple visible identities: supply --mouth-data with identified face tracks; automatic single-face measurements cannot prove this board")
    with args.video.open("rb") as video_file:
        video_hash = hashlib.file_digest(video_file, "sha256").hexdigest()
    storyboard_hash = hashlib.sha256(args.storyboard.read_bytes()).hexdigest()
    review = {}
    if args.review:
        evidence = json.loads(args.review.read_text(encoding="utf-8"))
        if evidence.get("video_sha256") != video_hash or evidence.get("storyboard_sha256") != storyboard_hash:
            extraction_errors.append("review hash mismatch: evidence must identify this exact video and storyboard")
        else:
            review = evidence.get("lines", {})
    args.report.with_suffix(".mouths.json").write_text(
        json.dumps({"faces": tracks, "fps": fps}, indent=2), encoding="utf-8")
    result = evaluate_lines(lines, words, tracks, fps, review=review)
    result["errors"].extend(extraction_errors)
    result["ok"] = result["ok"] and not extraction_errors
    result.update(video=str(args.video.resolve()), storyboard=str(args.storyboard.resolve()),
                  storyboard_sha256=storyboard_hash, words=words,
                  video_sha256=video_hash)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for line in result["lines"]:
        print("%s text=%s mouth=%s voice=%s visual=%s" %
              (line["id"], line["text_ok"], line["mouth_ok"], line["voice_ok"], line["visual_ok"]))
    print("RESULT:", "PASS" if result["ok"] else "FAIL / NEEDS REVIEW", str(args.report))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
