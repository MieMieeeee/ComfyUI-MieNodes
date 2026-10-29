# -*- coding: utf-8 -*-
"""SYNC PAIR guard: the node prompt and the docs prompt must stay in step.

The two system prompts below are a SYNC PAIR (2026-10 ruling: shared rules
land in BOTH files in the same commit):

* ``nodes/llm/prompts/h3_storyboard/system_asset.txt`` — used inside
  ComfyUI by the 剧本资产生成器 node (asset_header + locked_events input).
* ``docs/script_asset_system_prompt.txt`` — used outside ComfyUI.

The 2026-10-28 review caught the pair drifted: the docs copy carried 12
CRAFT RULES while the node copy had only 5 of them — precisely the rules
from the latest live-render rounds (light anchor, locale text, one active
hand). Commit discipline alone did not hold the pair together, so this
test mechanically enforces it:

1. the numbered CRAFT RULES section must be IDENTICAL in both files
   (same rules, same order, same wording);
2. every shared scattered rule marker (naming, voice timbre, speech
   rate, scene purity, …) must appear in BOTH files.

Adding a shared rule means: write it into BOTH files in one commit and,
if it is a scattered rule, add its marker to SHARED_MARKERS below.
"""
from __future__ import annotations

import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
NODE_PROMPT = PROJECT_DIR / "nodes" / "llm" / "prompts" / "h3_storyboard" / "system_asset.txt"
DOCS_PROMPT = PROJECT_DIR / "docs" / "script_asset_system_prompt.txt"

CRAFT_HEADING = "CRAFT RULES FROM LIVE RENDERS"

# Distinctive substrings of rules that live OUTSIDE the numbered craft
# section. Case-insensitive (the two files capitalize differently).
SHARED_MARKERS = (
    "its OWN new name",                    # one name per concept revision
    "timbre, the register and the pace ONLY",  # voice = timbre, never behaviour
    "INITIAL STAGING ONLY",                # setting narrowed to staging (2026-10-29)
    "overrides it mechanically",           # wired scene asset beats script scene text
    "light preset",                        # time of day lives in the light selector
    "mode=concept",                        # free-creation mode contract (2026-10-29)
    "environments only",                   # scene purity: never a figure
    "a dark sedan at the curb below",      # script-asserted visible object
    "5-6 characters per second",           # measured speech-rate budget
    "trims the first 22 frames",           # continuation blind window
    "<d>",                                 # dialogue tag ownership
    "audio_mode",                          # generated/reference split
)


def _read(path: Path) -> str:
    assert path.is_file(), "missing SYNC PAIR file: %s" % path
    return path.read_text(encoding="utf-8")


def _craft_rules(text: str) -> list[str]:
    """The numbered rules after the craft heading, in order.

    Collection stops at the first non-empty line that is not a numbered
    rule, so the section can sit between different neighbours in the two
    files.
    """
    lines = text.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.startswith(CRAFT_HEADING)),
        None,
    )
    assert start is not None, "missing %s heading" % CRAFT_HEADING
    rules: list[str] = []
    for line in lines[start + 1:]:
        if re.match(r"^\d+\)", line):
            rules.append(line.strip())
        elif not line.strip():
            if rules:
                break  # blank line after the block ends the section
            continue    # blank line between heading and rule 1
        else:
            break
    return rules


def test_craft_rules_section_identical_in_both_prompts():
    node_rules = _craft_rules(_read(NODE_PROMPT))
    docs_rules = _craft_rules(_read(DOCS_PROMPT))
    assert len(docs_rules) >= 12, (
        "docs prompt lost craft rules (found %d)" % len(docs_rules)
    )
    assert node_rules == docs_rules, (
        "CRAFT RULES drifted between the node prompt and the docs prompt "
        "(SYNC PAIR): only-one-side rules are exactly how the 2026-10-28 "
        "drift happened. Patch BOTH files in the same commit.\n"
        + "\n".join(
            "node: %s" % r for r in node_rules if r not in docs_rules
        )
        + "\n"
        + "\n".join(
            "docs: %s" % r for r in docs_rules if r not in node_rules
        )
    )


def test_scattered_shared_rule_markers_in_both_prompts():
    for path in (NODE_PROMPT, DOCS_PROMPT):
        text = _read(path).casefold()
        missing = [m for m in SHARED_MARKERS if m.casefold() not in text]
        assert not missing, (
            "%s is missing shared SYNC PAIR rule marker(s): %s" % (path, missing)
        )
