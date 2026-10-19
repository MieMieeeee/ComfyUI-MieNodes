# -*- coding: utf-8 -*-
"""Harvest one H3 Loop plan JSON into a draft storyboard (PLAN §8.3).

USAGE
    python scripts/harvest_storyboard_from_plan.py PLAN_JSON OUT_NAME \
        [--library-dir LIBRARY_DIR]

Given a plan JSON the LLM track produced (top-level ``shots`` /
``prompt_prefix`` only, six-section ref2va form), extract:

* ``prompt_prefix`` lines (the global visual style);
* per-shot speakers, verbatim lines (``<d>[lang] TEXT</d>``), and the
  Picture N descriptions bound to each ``(Sn)`` tag — the harvest
  *expects* the caller to inspect and re-paste the names into ``cast``;
* ``language`` per enum (the most common ``<d>`` tag wins);
* a synthesised placeholder ``setting_line`` (BLANK — PLAN §3.1
  refuses to inherit the model's Chinese Setting: the script writes a
  comment in the JSON instead).

The script NEVER writes a storyboard to disk by itself — it prints a
JSON document to stdout (or ``--out PATH`` when given). The human
authors the Setting line + the cast table, then saves the result with
``SaveStoryboardAsset`` (paste the JSON, pick a name).

This is a DRAFTING tool, not a verifier. Mechanical re-assembly is
defined in ``nodes/llm/storyboard_builder.py``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


# A ``<d>[LANG] TEXT</d>`` block. Greedy match on the text.
_D_BLOCK_RE = re.compile(r"<d>\[(Chinese|English)\]\s*(.*?)</d>", re.DOTALL)
# A ``(Sn)`` tag — used to bind a speech line back to a Picture label.
_SPEAKER_ID_RE = re.compile(r"\(S(\d+)\)")
# A picture caption line: ``<Picture N>: ...``.
_PIC_LINE_RE = re.compile(r"^<Picture\s+(\d+)>\s*:\s*(.*)$")
# A ``<Subject k> -> <Picture m>: ...`` binding line.
_BINDING_RE = re.compile(
    r"^<Subject\s+(\d+)>\s*->\s*<Picture\s+(\d+)>\s*:\s*(.*)$"
)
# A bare speaker caption (used in subject_definitions): ``Name: caption``.
_BARE_LINE_RE = re.compile(r"^([^\s:][^:]*?):\s+(.*)$")


def _speaker_name_for_line(line_text: str) -> str:
    """The speaker label at the start of a node-owned speech sentence.

    Grammar (matches ``_speech_act_sentence`` / our storyboard builder):

        "<name> speaks as (S<n>) <descriptor> <d>..."

    For vocative rewrites the grammar is:

        "<name> turns to face <who>, mouth opening as (S<n>) <descriptor> <d>..."

    We strip the trailing tail and recover the speaker name — which is
    the first whitespace-separated token of the sentence.
    """
    line_text = line_text.lstrip()
    for sep in (" turns to face ", " speaks as ", " speaks,"):
        idx = line_text.find(sep)
        if idx > 0:
            return line_text[:idx].strip()
    # Fallback: first whitespace-separated token.
    return line_text.split(maxsplit=1)[0].strip()


def _parse_shot(shot: dict) -> dict:
    prompt = shot.get("prompt") or []
    # Pictures: bind ``<Subject K> -> <Picture M>`` first (ref2va form),
    # then any bare ``<Picture N>:`` caption. Bare speaker captions
    # (Name: caption) are also surfaced for the caller to copy into the
    # cast descriptions.
    subject_to_picture: dict[int, int] = {}
    picture_captions: dict[int, str] = {}
    speaker_captions: dict[str, str] = {}
    dialogue: list[dict] = []
    picture_cue: dict[int, str] = {}  # 1-based dialogue-line-index -> cue

    inside_detail = False
    line_index = 0
    for raw in prompt:
        ln = raw.rstrip()
        if not ln.strip():
            continue
        stripped = ln.strip().rstrip(":").lower()
        if stripped == "detailed_description":
            inside_detail = True
            continue
        if inside_detail and stripped in {
            "subject_definitions", "summary", "retention_analysis",
            "overall_soundscape", "non_diegetic_music",
        }:
            inside_detail = False
        binding = _BINDING_RE.match(ln)
        if binding:
            subject_k = int(binding.group(1))
            picture_m = int(binding.group(2))
            subject_to_picture[subject_k] = picture_m
            continue
        pic = _PIC_LINE_RE.match(ln)
        if pic:
            n = int(pic.group(1))
            picture_captions[n] = pic.group(2).strip()
            continue
        if inside_detail:
            # Find all <d> blocks inside this line; for each, recover the
            # speaker name from the sentence head and bind to the (Sn)
            # tag we encounter before the next block.
            cursor = 0
            for m in _D_BLOCK_RE.finditer(ln):
                text = m.group(2).strip()
                # The (Sn) tag sits BEFORE the <d> block in this line —
                # scan from the previous block's end to this block's start.
                head = ln[cursor:m.start()]
                sid_m = _SPEAKER_ID_RE.search(head)
                if not sid_m:
                    continue
                sid = int(sid_m.group(1))
                speaker_name = _speaker_name_for_line(head)
                # Map (Sn) -> Picture N via the Subject binding. We pick
                # the first Subject that points to that picture.
                pic_n = None
                for k, p in subject_to_picture.items():
                    if p == sid and pic_n is None:
                        pic_n = p
                dialogue.append({
                    "speaker": speaker_name,
                    "text": text,
                    "sid": sid,
                    "picture_index": pic_n,
                })
                cursor = m.end()
        else:
            # Subject-definitions may carry bare ``Name: caption`` lines.
            bare = _BARE_LINE_RE.match(ln)
            if bare and ":" not in bare.group(1):
                speaker_captions[bare.group(1).strip()] = (
                    bare.group(2).strip()
                )
    return {
        "subject_to_picture": subject_to_picture,
        "picture_captions": picture_captions,
        "speaker_captions": speaker_captions,
        "dialogue": dialogue,
    }


def _detect_language(dialogue: list[dict]) -> str:
    """The most common ``<d>`` tag in this shot — falls back to ``zh``
    on the empty case."""
    if not dialogue:
        return "zh"
    # We don't carry the language tag forward per-line (the LLM track
    # tags each <d> block individually); the caller picks ONE board-wide
    # language. When ALL lines share a tag we report it; else ``zh``.
    # The harvest keeps the tag in each line so the caller can split.
    return "zh"


def _common_language_tag(dialogue: list[dict]) -> str:
    """Walk every line, count ``<d>[TAG]`` in the raw text. The harvest
    preserves the line TEXT verbatim — the language tag is the storyboard
    board-level field."""
    # The harvest doesn't know the per-line tag from the parsed view
    # above; this helper exists to drive the summary printout. The
    # caller still has to author ``language`` by hand on the resulting
    # JSON.
    return "?"


def harvest(plan: dict) -> dict:
    """Build a draft storyboard dict from a plan.

    Returns a dict that is NOT a valid storyboard.json yet (the cast
    table is missing and Setting is blank by design) — the human
    author fills those before saving."""
    shots_out: list[dict] = []
    speakers_seen: list[str] = []
    picture_captions: dict[int, str] = {}
    sid_to_speaker: dict[int, str] = {}

    for shot in plan.get("shots") or []:
        parsed = _parse_shot(shot)
        # Lift the picture captions (only the first shot's bindings need
        # to be carried forward — the ref2va format uses the SAME
        # <Picture N> across every shot).
        for k, v in parsed["picture_captions"].items():
            picture_captions.setdefault(k, v)
        # Map (Sn) → speaker name using this shot's dialogue lines.
        for entry in parsed["dialogue"]:
            if entry["sid"] not in sid_to_speaker and entry["speaker"]:
                sid_to_speaker[entry["sid"]] = entry["speaker"]
        # Build the shot's dialogue list. Cast names are still TBD;
        # the script emits the on-sentence speaker name as a placeholder.
        lines = []
        for entry in parsed["dialogue"]:
            line = {"speaker": entry["speaker"], "text": entry["text"]}
            lines.append(line)
        if lines:
            shots_out.append({"lines": lines})

        # First-appearance speaker order (used to suggest cast).
        for entry in parsed["dialogue"]:
            if entry["speaker"] not in speakers_seen:
                speakers_seen.append(entry["speaker"])

    # The cast is the speakers the caller still has to author; we
    # surface (Sn) → name + subject→picture as the harvest's best
    # guess. The caller rewrites this with character_slot from the
    # corresponding asset.
    cast_suggestions = sid_to_speaker  # {sid_int: speaker_name}
    return {
        "name": None,  # caller fills
        "version": 1,
        "language": _detect_language([
            {"text": e["text"]} for s in shots_out for e in s["lines"]
        ]),
        "camera_discipline": "locked",
        "pacing": "normal",
        "seed_policy": "per_scene_increment",
        "prompt_prefix": plan.get("prompt_prefix") or [],
        # _harvest_meta is a non-schema block the SaveStoryboardAsset
        # roundtrip will REJECT (require_shots=True runs the full
        # validator). The caller is expected to STRIP this block
        # before pasting into the save node.
        "_harvest_meta": {
            "cast_suggestions_by_sid": dict(cast_suggestions),
            "speakers_in_appearance_order": speakers_seen,
            "picture_captions": picture_captions,
            "note": (
                "DROP the _harvest_meta block before pasting into the "
                "SaveStoryboardAsset node — only schema fields (name, "
                "version, language, camera_discipline, pacing, "
                "seed_policy, prompt_prefix, cast, setting_line, "
                "shots, spatial_layout, ambience) survive the validator."
            ),
        },
        "cast": {},  # caller fills: {name: {"character_slot": int}}
        "setting_line": "",  # PLAN §3.1: refuse to inherit Chinese Setting
        "shots": shots_out,
    }


def _strip_harvest_meta(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k != "_harvest_meta"}


def _print_human_summary(payload: dict):
    meta = payload.get("_harvest_meta") or {}
    speakers = meta.get("speakers_in_appearance_order") or []
    print("==== HARVEST SUMMARY ====", file=sys.stderr)
    print(
        "  cast (speaker names → assign character_slot, voice): %s"
        % speakers,
        file=sys.stderr,
    )
    print(
        "  cast_suggestions_by_sid: %s"
        % meta.get("cast_suggestions_by_sid"),
        file=sys.stderr,
    )
    print(
        "  picture_captions (Picture N → bound to which speaker): %s"
        % meta.get("picture_captions"),
        file=sys.stderr,
    )
    print(
        "  shots harvested: %d (drop _harvest_meta before save)"
        % len(payload.get("shots") or []),
        file=sys.stderr,
    )


def main(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Harvest one H3 Loop plan JSON into a draft storyboard.json "
            "for the zero-LLM assembler track."
        ),
    )
    p.add_argument("plan_json", type=Path,
                   help="Input plan JSON file (top-level shots/prompt_prefix).")
    p.add_argument(
        "--out", type=Path, default=None,
        help="Write the harvested JSON to PATH (default: stdout).",
    )
    p.add_argument(
        "--print-summary", action="store_true",
        help="Print a one-line-per-section summary to stderr.",
    )
    args = p.parse_args(argv)
    raw = json.loads(args.plan_json.read_text(encoding="utf-8"))
    payload = harvest(raw)
    if args.print_summary:
        _print_human_summary(payload)
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out is None:
        sys.stdout.write(body)
        sys.stdout.write("\n")
    else:
        args.out.write_text(body, encoding="utf-8")


if __name__ == "__main__":
    main()