# -*- coding: utf-8 -*-
"""Smoke test for the storyboard-harvest helper script.

Verifies:
  * the script is executable as ``python -m``;
  * the dumped JSON shape matches the schema the save node accepts;
  * the picture_captions and cast_suggestions surface the right speakers;
  * the per-line dialogue TEXT is preserved verbatim (locked contract).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_DIR / "scripts" / "harvest_storyboard_from_plan.py"
FIXTURE = PROJECT_DIR / "tests" / "fixtures" / "plan_english_kitten_grandpa.json"


def test_harvest_runs_and_emits_schema_compatible_json():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(FIXTURE), "--print-summary"],
        check=True, capture_output=True, text=True,
    )
    out = proc.stdout
    err = proc.stderr
    # The stderr summary names the speakers and Picture N captions.
    assert "小猫妹妹" in err
    assert "猫爷爷" in err
    # The stdout JSON parses.
    payload = json.loads(out)
    # Schema fields are present (the save validator only accepts these).
    for key in (
        "version", "language", "camera_discipline", "pacing",
        "seed_policy", "prompt_prefix", "cast", "setting_line", "shots",
    ):
        assert key in payload, "missing schema key %r" % key
    # The harvest meta is a comment, not part of the schema.
    assert "_harvest_meta" in payload
    assert payload["_harvest_meta"]["cast_suggestions_by_sid"] == {
        "1": "小猫妹妹", "2": "猫爷爷",
    }
    # The locked dialogue text is verbatim.
    lines = [
        ln
        for shot in payload["shots"]
        for ln in shot["lines"]
    ]
    texts = {ln["text"] for ln in lines}
    assert "Grandpa, I have had a psychological shadow since I was little." in texts
    assert "What is it?" in texts
    # Setting line is BLANK by design (PLAN §3.1 refuses Chinese residue).
    assert payload["setting_line"] == ""


def test_harvest_can_strip_meta_to_fit_save_validator():
    """The save node's ``require_shots=True`` path runs the full
    validator, which rejects unknown keys. The caller must drop
    ``_harvest_meta`` before pasting — this test enforces the contract
    that the meta block, removed, leaves a valid storyboard payload
    (cast and name still to be authored)."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), str(FIXTURE)],
        check=True, capture_output=True, text=True,
    )
    payload = json.loads(proc.stdout)
    cleaned = {k: v for k, v in payload.items() if k != "_harvest_meta"}
    # Schema keys still present, no extras.
    schema_keys = {
        "name", "version", "language", "camera_discipline", "pacing",
        "seed_policy", "prompt_prefix", "cast", "setting_line", "shots",
        "spatial_layout", "ambience",
    }
    assert set(cleaned.keys()) <= schema_keys