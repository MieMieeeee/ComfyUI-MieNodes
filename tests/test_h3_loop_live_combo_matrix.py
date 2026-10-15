# -*- coding: utf-8 -*-
"""Live replay matrix of the saved ref2va workflows (2026-10-15 request).

Extracts every ``MiniMaxH3LoopPromptGeneratorV2`` combo from the
user's workflow drop folder, dedupes on (character/scene roster,
user_input, 润色 on/off), then replays each UNIQUE combo on the local
ComfyUI with ``SetMiniMaxTokenPlanLLMServiceConnector`` pinned to
MiniMax-M3 and verifies the generated plan mechanically — the plan is
the final artifact, so every guard runs on the assembled scene
prompts (2026-10-14 contract-checks ruling):

* every user dialogue line lands VERBATIM, in order, in exactly one
  scene's ``<d>`` block (locked dialogue is a hard contract);
* each spoken line is attributed to the user's speaker name;
* every roster character is named in EVERY scene prompt (the
  2026-10-15 visible-cast fix: silent-but-placed characters stay);
* the scene asset's ``<Picture N>`` reference is taught in every
  scene prompt;
* forbidden markers absent: ``[Shot 2]``, clock notation, ``@alias``,
  ``#tag``.

Known residuals are WARNINGS, not failures (``Setting:`` verbatim
line, ``fur`` wording on human-subject combos).

Run (server must be up on freshly-restarted code):

    MIE_LIVE_E2E=1 python -m pytest tests/test_h3_loop_live_combo_matrix.py -v

Each combo's plan text is dumped under ``tests/live_combo_results/``.
"""

import importlib.util
import json
import os
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

import pytest

PROJECT_DIR = pathlib.Path(__file__).resolve().parent.parent
WORKFLOW_DIR = pathlib.Path(
    os.environ.get("MIE_WORKFLOW_DIR", r"C:/Users/administered/Downloads/工作流")
)
BASE = os.environ.get("MIE_COMFYUI", "http://127.0.0.1:8188").rstrip("/")
MODEL = os.environ.get("MIE_COMBO_MODEL", "MiniMax-M3")
POLL_TIMEOUT = int(os.environ.get("MIE_COMBO_TIMEOUT", "900"))
PLUGIN_LOGS = pathlib.Path(
    os.environ.get(
        "MIE_PLUGIN_LOGS",
        r"C:/PP/V9/V9-Large/ComfyUI_Mie_2026_V9.0_Large/ComfyUI/custom_nodes/"
        "comfyui_mienodes/logs",
    )
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("MIE_LIVE_E2E"),
    reason="live replay against the local ComfyUI; set MIE_LIVE_E2E=1",
)

# dialogue_segmenter is deliberately self-contained (math + dataclasses
# only) so it loads in test environments without ComfyUI.
_spec = importlib.util.spec_from_file_location(
    "_mie_live_dialogue_segmenter",
    str(PROJECT_DIR / "nodes" / "llm" / "dialogue_segmenter.py"),
)
_dlg = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _dlg  # dataclasses need the module registered
_spec.loader.exec_module(_dlg)


# ------------------------------------------------------------------ #
# combo extraction
# ------------------------------------------------------------------ #
def _widgets(node):
    wv = node.get("widgets_values") or []
    if wv and isinstance(wv[0], dict) and "values" in wv[0]:
        wv = wv[0]["values"]  # widgets_values_named form
    return wv


def extract_combos(workflow_dir=WORKFLOW_DIR):
    """Deduped [(index, combo_dict)] in file order."""
    seen = {}
    for path in sorted(workflow_dir.glob("*.json")):
        try:
            wf = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        gen = next(
            (n for n in wf.get("nodes", [])
             if "MiniMaxH3LoopPromptGeneratorV2" in str(n.get("type", ""))),
            None,
        )
        if gen is None:
            continue
        wv = _widgets(gen)
        if not wv or not isinstance(wv[0], str) or not wv[0].strip():
            continue
        characters, scene = [], None
        for n in wf.get("nodes", []):
            if "LoadAssetV2" not in str(n.get("type", "")):
                continue
            aw = _widgets(n)
            if len(aw) < 3:
                continue
            kind, name = str(aw[0]), str(aw[2]).strip()
            if not name or "下拉" in name:
                continue
            if kind == "character":
                characters.append(name)
            elif kind == "scene":
                scene = name
        key = (
            tuple(characters), scene, " ".join(wv[0].split()),
            str(wv[1]).startswith("on"),
        )
        if key in seen:
            seen[key]["source"] += " | " + path.name
            continue
        seen[key] = {
            "user_input": wv[0],
            "enhance": key[3],
            "seed": wv[2] if isinstance(wv[2], int) else 0,
            "characters": characters,
            "scene": scene,
            "source": path.name,
        }
    return list(enumerate(seen.values()))


COMBOS = extract_combos() if WORKFLOW_DIR.exists() else []
COMBO_PARAMS = [c for _i, c in COMBOS]
COMBO_IDS = [
    "%02d_%s%s%s" % (
        i, "+".join(c["characters"]) or "nochar",
        "+" + c["scene"] if c["scene"] else "+noscene",
        "_润色" if c["enhance"] else "_raw",
    )
    for i, c in COMBOS
]


# ------------------------------------------------------------------ #
# live server plumbing
# ------------------------------------------------------------------ #
def _post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def build_graph(combo, combo_index):
    chars = combo["characters"]
    graph = {}
    for i, name in enumerate(chars[:3]):
        graph[str(70 + i)] = {
            "class_type": "LoadAssetV2|Mie",
            "inputs": {"kind": "character", "library_dir": "", "name": name},
            "_meta": {"title": "Load %s" % name},
        }
    if combo["scene"]:
        graph["81"] = {
            "class_type": "LoadAssetV2|Mie",
            "inputs": {
                "kind": "scene", "library_dir": "", "name": combo["scene"],
            },
            "_meta": {"title": "Load %s" % combo["scene"]},
        }
    graph["60"] = {
        "class_type": "SetMiniMaxTokenPlanLLMServiceConnector|Mie",
        "inputs": {
            "api_token": "",
            "model_select": MODEL,
            "custom_model": "",
            "config_file": "mie_llm_keys.json",
            "config_key": "minimax",
            "prefer_local_config": True,
        },
        "_meta": {"title": "MiniMax Token Plan"},
    }
    gen_inputs = {
        "user_input": combo["user_input"],
        "enhance_user_input": (
            "on - 自动润色后再规划" if combo["enhance"]
            else "off - 不润色(默认)"
        ),
        "seed": int(combo["seed"] or 0),
        "scene_count": 0,
        "total_duration_seconds": 0,
        "pacing": "normal - 正常（语速·推荐）",
        "generation_mode": "per_shot - 逐场生成(推荐)",
        "camera_discipline": "locked - 固定机位(构图全场不变)",
        "category": "none - 不指定",
        "output_language": "en",
        "seed_mode": "per_scene_increment - 每场seed递增(推荐)",
        "temperature": 0.4,
        "max_tokens": 16384,
        "timeout": 300,
        "reference_mode": "ref2va - 参考图(N张/全场景)",
        "caption_mode": "cache_memory_disk - 缓存:内存+磁盘(推荐)",
        "llm_service_connector": ["60", 0],
    }
    for i in range(len(chars[:3])):
        gen_inputs["character_%d" % (i + 1)] = [str(70 + i), 5]
    if combo["scene"]:
        gen_inputs["scene_1"] = ["81", 5]
    graph["59"] = {
        "class_type": "MiniMaxH3LoopPromptGeneratorV2|Mie",
        "inputs": gen_inputs,
        "_meta": {"title": "Generator"},
    }
    graph["50"] = {
        "class_type": "ShowAndSaveAnything|Mie",
        "inputs": {
            "anything": ["59", 0],
            "save_to_log": True,
            "log_file_name": "live_combo_%02d.log" % combo_index,
        },
        "_meta": {"title": "Show Plan"},
    }
    return graph


def run_combo(graph):
    resp = _post(BASE + "/prompt", {
        "prompt": graph, "client_id": "mie-combo-matrix",
    })
    pid = resp["prompt_id"]
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        time.sleep(5)
        with urllib.request.urlopen(BASE + "/history/" + pid, timeout=30) as r:
            hist = json.loads(r.read().decode("utf-8"))
        if pid not in hist:
            continue
        status = hist[pid].get("status", {})
        if status.get("status_str") == "error":
            errs = [
                m for m in (status.get("messages") or [])
                if m and m[0] in ("execution_error", "execution_interrupted")
            ]
            raise AssertionError("server-side error: %s" % (errs or status,))
        if status.get("completed"):
            return hist[pid]
    raise AssertionError("timeout after %ss (prompt_id %s)" % (POLL_TIMEOUT, pid))


def read_plan_text(combo_index):
    """Unwrap the ShowAnything log line: {"ts",..., "result": "<plan json>"}.

    The file holds one JSON object per run; the LAST one is ours.
    """
    log = PLUGIN_LOGS / ("live_combo_%02d.log" % combo_index)
    text = log.read_text(encoding="utf-8", errors="replace")
    lines = [l for l in text.splitlines() if l.strip().startswith("{")]
    if not lines:
        raise AssertionError("no JSON entry in %s" % log)
    outer = json.loads(lines[-1])
    if isinstance(outer.get("result"), str):
        return outer["result"]
    if "shots" in outer:
        return lines[-1]
    raise AssertionError(
        "log entry has neither result nor shots: %r" % list(outer)[:5]
    )


# ------------------------------------------------------------------ #
# mechanical verification on the final artifact
# ------------------------------------------------------------------ #
def expected_turns(user_input):
    turns, _prologue = _dlg.parse_structured_dialogue_turns(user_input)
    if turns is None:
        pytest.skip(
            "user_input has no canonical speaker lines: %r" % user_input[:40]
        )
    return [(t.speaker, list(t.lines)) for t in turns]


_SPEECH_RE = re.compile(
    r"^(?P<pre>.{0,60}?)\s*speaks as \(S\d+\).*?"
    r"<d>\[[^\]]*\]\s*(?P<text>.*?)\s*</d>\s*$"
)

_HUMANISH = {"黑白", "银白", "爷爷", "孙女"}


def verify_plan(plan_text, combo):
    errors, warnings = [], []
    plan = json.loads(plan_text)
    scenes = plan.get("shots") or []
    scene_texts = [
        (s.get("id"), "\n".join(str(x) for x in (s.get("prompt") or [])))
        for s in scenes
    ]
    if not scenes:
        errors.append("plan has no shots")
        return errors, warnings

    # 1+2. verbatim lines in order, owned by the user's speaker
    expected = expected_turns(combo["user_input"])
    want = [(spk, ln) for spk, lines in expected for ln in lines]
    got = []
    for sid, text in scene_texts:
        for line in text.split("\n"):
            m = _SPEECH_RE.match(line.strip())
            if m:
                got.append((sid, m.group("pre").strip(), m.group("text")))
    for i, (spk, ln) in enumerate(want):
        if i >= len(got):
            errors.append("line %d of %d missing: %r" % (i, len(want), ln))
            break
        sid, pre, text = got[i]
        if text.strip() != ln.strip():
            errors.append(
                "scene %s line %d not verbatim:\n  want %r\n  got  %r"
                % (sid, i, ln, text)
            )
        if spk not in pre:
            errors.append(
                "scene %s line %d owned by %r, expected speaker %r"
                % (sid, i, pre, spk)
            )
    if len(got) != len(want):
        errors.append("spoken-line count %d != expected %d" % (len(got), len(want)))

    # 3. cast completeness. The plan faces H3 with the USER's dialogue
    # names (asset names need not appear when they differ — combo 01:
    # roster 小猫妹妹/猫爷爷, speakers 孙女/爷爷), so the guarantee is:
    # every SPEAKER name stays in every scene (the visible-cast fix),
    # and every character slot's <Picture i> is anchored in every scene.
    for name in dict.fromkeys(
        spk for spk, _lines in expected
        if spk not in ("旁白", "(narrator)")
    ):
        for sid, text in scene_texts:
            if name not in text:
                errors.append(
                    "scene %s never names speaking character %s "
                    "(visible-cast guarantee)" % (sid, name)
                )
    for i in range(1, len(combo["characters"]) + 1):
        tok = "<Picture %d>" % i
        for sid, text in scene_texts:
            if tok not in text:
                errors.append(
                    "scene %s never anchors character slot %s" % (sid, tok)
                )

    # 3b. spatial continuity: every scene pins every speaking character
    # (code-built directive; "X stays at ..." / "X stays beside ...").
    for name in dict.fromkeys(
        spk for spk, _lines in expected
        if spk not in ("旁白", "(narrator)")
    ):
        pin_re = re.compile(re.escape(name) + r"\s+stays\b")
        for sid, text in scene_texts:
            if not pin_re.search(text):
                errors.append(
                    "scene %s has no spatial pin for %s" % (sid, name)
                )

    # 4. scene asset taught in every scene
    if combo["scene"]:
        pic = "<Picture %d>" % (len(combo["characters"]) + 1)
        for sid, text in scene_texts:
            if pic not in text:
                errors.append(
                    "scene %s never references the scene asset %s (%s)"
                    % (sid, pic, combo["scene"])
                )

    # 5. forbidden markers
    for sid, text in scene_texts:
        for bad in ("[Shot 2]", "At 00:", "@alias", "#tag"):
            if bad in text:
                errors.append("scene %s carries forbidden marker %r" % (sid, bad))

    # known residuals — warn only
    for sid, text in scene_texts:
        if re.search(r"^Setting: ", text, re.MULTILINE):
            warnings.append("scene %s still has a verbatim 'Setting:' line" % sid)
    if any(n in _HUMANISH for n in combo["characters"]) and re.search(
        # "fur or hair as the subject actually has" is the CORRECT
        # template wording; bare "fur" on a human combo is the residual.
        r"\bfur\b(?! or hair)", plan_text, re.IGNORECASE
    ):
        warnings.append("human-subject combo carries 'fur' wording")
    return errors, warnings


@pytest.mark.parametrize("combo", COMBO_PARAMS, ids=COMBO_IDS)
def test_combo_generates_a_valid_plan(combo, request):
    combo_index = next(i for i, c in COMBOS if c is combo)
    graph = build_graph(combo, combo_index)
    run_combo(graph)
    plan_text = read_plan_text(combo_index)
    out_dir = pathlib.Path(__file__).parent / "live_combo_results"
    out_dir.mkdir(exist_ok=True)
    safe = re.sub(r"[^\w\u4e00-\u9fff]+", "_", combo["source"])[:60]
    (out_dir / ("%02d_%s.json" % (combo_index, safe))).write_text(
        plan_text, encoding="utf-8"
    )
    errors, warnings = verify_plan(plan_text, combo)
    if warnings:
        print("\nWARNINGS:\n  - " + "\n  - ".join(warnings))
    assert not errors, "\n  - ".join(["combo %s" % combo["source"]] + errors)
