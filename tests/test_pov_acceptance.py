"""A complete transcript and every speech window are required for acceptance."""
import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def verifier():
    path = Path(__file__).resolve().parents[1] / "scripts" / "verify_pov_lip_sync.py"
    spec = importlib.util.spec_from_file_location("pov_verifier", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def words(text):
    return [{"text": char, "start": i * 0.1, "end": (i + 1) * 0.1} for i, char in enumerate(text)]


def test_adjacent_chinese_lines_are_aligned_separately(verifier):
    lines = [{"speaker": "N", "text": "真真的吗"}, {"speaker": "A", "text": "你的账号已被封禁"}]
    out = verifier.align_lines(lines, words("真真的吗你的账号已被封禁"))
    assert out["ok"]
    assert len(out["lines"]) == 2
    assert out["lines"][0]["end"] <= out["lines"][1]["start"]


def test_missing_later_line_cannot_pass(verifier):
    lines = [{"speaker": "N", "text": str(i)} for i in range(1, 7)]
    out = verifier.align_lines(lines, words("12345"))
    assert not out["ok"]
    assert len(out["lines"]) == 6
    assert not out["lines"][-1]["text_ok"]


def test_changed_words_and_extra_speech_cannot_pass(verifier):
    lines = [{"speaker": "N", "text": "hello"}]
    assert not verifier.align_lines(lines, words("help"))["ok"]
    assert not verifier.align_lines(lines, words("helloagain"))["ok"]


def test_a_word_spanning_two_speakers_is_inconclusive(verifier):
    lines = [{"speaker": "N", "text": "你好"}, {"speaker": "A", "text": "好的"}]
    out = verifier.align_lines(lines, [{"text": "你好好的", "start": 0.0, "end": 2.0}])
    assert not out["ok"]
    assert any("boundary" in x for x in out["errors"])


def test_zero_duration_asr_word_is_reported_without_losing_lines(verifier):
    out = verifier.align_lines(
        [{"speaker": "N", "text": "hello"}],
        [{"text": "hello", "start": 1.0, "end": 1.0}],
    )
    assert not out["ok"]
    assert len(out["lines"]) == 1
    assert any("timestamp" in x for x in out["errors"])


def test_open_mouth_during_sixth_offscreen_line_fails(verifier):
    lines = [{"speaker": "N", "text": str(i), "offscreen": True} for i in range(1, 7)]
    samples = {"A": [0.005] * 50 + [0.08] * 10}
    out = verifier.evaluate_lines(lines, words("123456"), samples, fps=100)
    assert len(out["lines"]) == 6
    assert out["lines"][0]["mouth_ok"]
    assert not out["lines"][-1]["mouth_ok"]
    assert not out["ok"]


def test_missing_face_samples_and_unreviewed_voice_do_not_pass(verifier):
    lines = [{"speaker": "N", "text": "1", "offscreen": True}]
    out = verifier.evaluate_lines(lines, words("1"), {"A": [None] * 10}, fps=100)
    assert not out["ok"]
    assert not out["lines"][0]["mouth_ok"]
    out = verifier.evaluate_lines(lines, words("1"), {"A": [0.005] * 10}, fps=100)
    assert out["lines"][0]["mouth_ok"]
    assert not out["ok"]
    assert not out["lines"][0]["voice_ok"]


def test_cli_review_is_bound_to_the_video_and_storyboard(verifier, tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video identity for supplied measurements")
    board = tmp_path / "board.json"
    board.write_text(json.dumps({"cast": {"N": {"offscreen": True}, "A": {}},
                                "shots": [{"lines": [{"speaker": "N", "text": "1"}]}]}))
    transcript = tmp_path / "words.json"
    transcript.write_text(json.dumps(words("1")))
    mouths = tmp_path / "mouths.json"
    mouths.write_text(json.dumps({"fps": 100, "faces": {"A": [0.005] * 10}}))
    review = tmp_path / "review.json"
    evidence = {"video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                "storyboard_sha256": hashlib.sha256(board.read_bytes()).hexdigest(),
                "lines": {"scene_01.line_01": {"voice_speaker": "N", "visible_speakers": ["A"]}}}
    review.write_text(json.dumps(evidence))
    report = tmp_path / "report.json"
    args = [str(video), "--storyboard", str(board), "--report", str(report),
            "--transcript", str(transcript), "--mouth-data", str(mouths), "--review", str(review)]
    assert verifier.main(args) == 0
    evidence["video_sha256"] = "another-video"
    review.write_text(json.dumps(evidence))
    assert verifier.main(args) == 1
    assert any("hash" in x for x in json.loads(report.read_text())["errors"])
