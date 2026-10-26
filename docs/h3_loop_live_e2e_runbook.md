# H3 Loop Live E2E Runbook

Sync the local plugin to a running ComfyUI, restart the server, submit a workflow via the HTTP API, and assert against the persisted plan / rendered mp4.

This is the working recipe validated end-to-end on 2026-09-22 against commit `3724188` (grok glue-to-`<d>`). It is **not** an automated test — it's the manual driving loop you run after committing code that affects a running pipeline.

## When to use

- A code change touches a node that ComfyUI is already running (most LLM nodes: `MiniMaxH3LoopPromptGenerator|Mie`, prompt builders, validators).
- Stubbed unit tests pass but you want to confirm the live LLM (MiniMax-M3) produces the contract you asserted.
- The previous live run is referenced in [[full-workflow-e2e-testing]] as the "live-tier verification" pattern.

## Pre-flight checklist

- [ ] LLM connector key file: `ComfyUI/custom_nodes/comfyui_mienodes/mie_llm_keys.json` — has `minimax` entry (string api_key, ~100 chars), no nested dict.
- [ ] Reference images: the workflow's three `LoadImage` nodes point to files that exist in `ComfyUI/input/`. Get the names from the workflow JSON (the `image` field on each `LoadImage` class node).
- [ ] Connector bound to the right model: `SetMiniMaxTokenPlanLLMServiceConnector|Mie` with `model_select=MiniMax-M3`, `config_key=minimax`, `prefer_local_config=true`.
- [ ] Working tree clean or at least on the commit you want to test. The deployed copy is overwritten by sync — uncommitted local edits will land in the deployed tree verbatim (sync is mirror mode).

## 1. Sync to local ComfyUI

```bash
cd /c/Users/administered/PycharmProjects/ComfyUI-MieNodes
python scripts/sync_to_comfyui.py --target local
```

Built-in targets (defined in `scripts/sync_to_comfyui.py`):

| name | path |
|---|---|
| `local` | `C:\PP\V9\V9-Large\ComfyUI_Mie_2026_V9.0_Large\ComfyUI\custom_nodes\comfyui_mienodes` |
| `lan`   | `\\192.168.31.150\f\V9_cu130\V9-Large_cu130\ComfyUI_Mie_2026_V9.0_Large\ComfyUI\custom_nodes\comfyui_mienodes` |

Default mode is **mirror** — stale destination files that no longer exist in src are deleted. `mie_llm_keys.json` in destination is preserved (never overwritten). The LAN target may fail if the box is offline; the script reports per-target failure tolerance.

After sync, confirm the deployed tree has your edits:

```bash
cd /c/PP/V9/V9-Large/ComfyUI_Mie_2026_V9.0_Large/ComfyUI/custom_nodes/comfyui_mienodes
grep -nE "def _speech_act_sentence|def assemble_dialogue_line_blocks" \
  nodes/llm/minimax_h3_loop_prompts.py
```

## 2. Restart ComfyUI via the launcher CLI

```bash
cd "/c/PP/V9/V9-Large/ComfyUI_Mie_2026_V9.0_Large"
cmd //c "ComfyUI启动器-CLI.cmd restart --json"
```

Gotcha: `restart` may print `started: true` but the inner ComfyUI process then dies (the launcher only checks port reachability after a short wait, and the first boot can fail on DB lock). If the next `status` says `running: false` and `http_reachable: false`, run `start` explicitly:

```bash
cmd //c "ComfyUI启动器-CLI.cmd start --json"
# {"started": true, "pid": 7496, "port": 8188, "ready": true, ...}
```

Probe reachability with a real endpoint, not just `/`:

```bash
curl -s http://127.0.0.1:8188/system_stats | python -c \
  "import sys,json; d=json.load(sys.stdin); print(d['system']['comfyui_version'], d['devices'][0]['type'])"
# 0.37.0 cuda
```

## 3. Locate and patch the workflow

Reference workflow lives at `C:\Users\administered\Downloads\MiniMax H3_LongVideo_Ref2V_full.json` (or the variant the user names). Inspect key nodes:

```bash
python -c "
import json
wf = json.load(open(r'C:\Users\administered\Downloads\MiniMax H3_LongVideo_Ref2V_full.json', encoding='utf-8'))
for nid in ['61','3','7','54','37','43','58']:
    n = wf[nid]
    print(nid, n['class_type'], list(n['inputs'].keys())[:5])
"
```

| node id | class | purpose |
|---|---|---|
| 61 | `MiniMaxH3LoopPromptGenerator\|Mie` | generates the plan (LLM call). Patch `user_input` here. |
| 54 | `SetMiniMaxTokenPlanLLMServiceConnector\|Mie` | LLM connector (MiniMax-M3). Leave alone. |
| 37/43/58 | `LoadImage` | reference images. Verify filenames exist in `ComfyUI/input/`. |
| 3 | `MiniMaxH3ChainPlanModern` | plans per-scene. Patch `run_name` for uniqueness. |
| 7 | `MiniMaxH3ChainAssemble` | final mp4 assembly. Patch `filename`/`copy_to_output`/`output_subfolder`. |

The plan flows from node 61 to node 3 via `plan_json_input: ["61", 0]`. Node 3 has a hardcoded literal `plan_json` as the default value; the link overrides it. Patching node 61's `user_input` is sufficient to drive a new plan.

## 4. Submit the workflow

Driver script template (`_live_<run_tag>.py`):

```python
import json, pathlib, time, urllib.request, datetime
API = "http://127.0.0.1:8188"
WF = pathlib.Path(r"C:\Users\administered\Downloads\MiniMax H3_LongVideo_Ref2V_full.json")
RUN_TAG = f"e2e_<descriptive>_{datetime.datetime.now():%Y%m%d_%H%M%S}"

wf = json.loads(WF.read_text(encoding="utf-8"))
wf["61"]["inputs"]["user_input"] = "<your patched input>"
wf["3"]["inputs"]["run_name"] = RUN_TAG           # unique per run
wf["7"]["inputs"]["filename"] = f"{RUN_TAG}_%date:yyyy-MM-dd%"
wf["7"]["inputs"]["copy_to_output"] = True
wf["7"]["inputs"]["output_subfolder"] = "live_e2e"

req = urllib.request.Request(API + "/prompt",
    data=json.dumps({"prompt": wf, "client_id": RUN_TAG}).encode("utf-8"),
    headers={"Content-Type": "application/json"}, method="POST")
resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
prompt_id = resp["prompt_id"]
```

Poll `/history/{prompt_id}` every 15 s, up to 25 min. Status field is `entry["status"]["status_str"]` (`"success"` when done) and `entry["status"]["completed"]`.

## 5. The artifacts you'll find on disk

The chain writes per-run artifacts under `ComfyUI/output/h3_chains/<run_name>/`:

```
plan_studio_presentation.json                # initial empty plan shell
prompt_history/scene_01/<sha256>.json        # ← THE plan prompt per scene
prompt_history/scene_01/index.json
prompt_history/scene_02/<sha256>.json        # second scene lands when sampler
prompt_history/scene_02/index.json           # finishes scene_01
final/<run_name>_<date>.mp4                  # ← final rendered video
final/<run_name>_<date>.generated.wav        # ← the TTS-mixed audio track
```

Per-scene prompts (the actual text node 1 sends to H3) are the contract surface for **plan-level** assertions. The mp4 + wav are for **rendered-output** assertions (lip sync, audio consistency, voice matching).

The user-level mp4 also lands at `ComfyUI/output/<output_subfolder>/<run_name>_<date>.mp4` when `copy_to_output=true`.

## 6. The five-point review pattern

For the grok glue-to-`<d>` commit, the live plan was checked against:

1. **Mommy's two lines same adult-female voice** — `adult female, warm mid-range pitch, soft rounded timbre` appears identically on both `<d>` blocks across both scenes.
2. **Daddy's line is adult-male** — `adult male, low-mid pitch, steady gravel-edged timbre` on `(S3)`.
3. **Vocative triggers facing** — `turns to face 白猫` / `turns to face 黑猫` clauses appear in scene_01's prompt for the Mommy/Daddy vocatives.
4. **Voice+tag+lip cue glued to `<d>`, no colon gap** — last prose line before each `<d>` ends with `...timbre` and contains either `speaks as (S<n>)` or `mouth opening as (S<n>)`. Regex check: `(S\d+)\s*:` must NOT appear in that last line.
5. **One input line per `<d>`, no doubles** — total `<d>` count across all scene prompts equals input dialogue line count, no duplicates.

### Locked-dialogue contract

User-supplied dialogue lines are **verbatim** in `<d>[...]` blocks. `enhance_user_input=on` and `output_language=en` translate the prose directions only; they never translate the locked lines. If a user pastes Chinese lines, the `<d>` blocks will say `[Chinese]` even with `output_language=en` — the user's words are the contract, not a suggestion. To get `[English]` blocks the user must type English in `user_input`. See [[h3-loop-locked-dialogue-is-contract]] for the verdict and [[h3-loop-cn-vocative-and-locked-lang]] for the live finding that proved it.

Equivalent Python snippet (drives 1-5 against a persisted scene prompt):

```python
import json, re, pathlib
prompt = json.load(open("<path>/<sha256>.json", encoding="utf-8'))["prompt"]
matches = list(re.finditer(r"<d>\[([^\]]*)\]\s*(.*?)</d>", prompt, re.DOTALL))
for m in matches:
    pre = prompt[max(0, m.start()-260):m.start()]
    last = pre.split("\n")[-1]
    assert ("speaks as (" in last) or ("mouth opening as (" in last), last
    assert last.rstrip().endswith(("timbre", "pitch")), last
    assert not re.search(r"\(S\d+\)\s*:", last), last
```

## 7. Sync semantics pitfalls

- `scripts/sync_to_comfyui.py` runs in **mirror mode by default** — files deleted in src get deleted in dst. Anything custom in the deployed tree (notes, debug scripts) will be wiped.
- `mie_llm_keys.json` is the **only** file preserved on dst. Sync restores it explicitly after the mirror walk.
- Python file changes do NOT auto-reload in ComfyUI. A restart is mandatory after sync.
- The deployed tree is a **separate git checkout** with its own HEAD. `git status` there will look weird — that's expected. The sync doesn't touch dst's git state.

## 8. Launcher CLI reference

`ComfyUI启动器-CLI.cmd` (path: `C:\PP\V9\V9-Large\ComfyUI_Mie_2026_V9.0_Large\`) exposes:

| command | purpose |
|---|---|
| `start [--json]` | start ComfyUI in background |
| `stop [--json]` | kill the ComfyUI process (parses port 8188) |
| `restart [--json]` | stop then start |
| `status [--json]` | running state + pid + http_reachable |
| `info [--json]` | effective config |
| `logs [launcher\|comfyui]` | tail a log file |
| `plugins list/install/uninstall/disable/enable/check-updates` | manage custom_nodes |
| `update [component]` | update ComfyUI core or plugins |

`--json` is required for scripted parsing. `status` reports `http_reachable` separately from `running` — the launcher sometimes mis-tracks internal ComfyUI; trust `http_reachable` + a real endpoint probe.

## 9. Related files

- `scripts/sync_to_comfyui.py` — the sync script with `TARGETS`, `IGNORE_DIRS`, `PRESERVE_DST_FILES`
- `Downloads/MiniMax H3_LongVideo_Ref2V_full.json` — the workflow used for live-tier verification
- `tests/test_h3_loop_english_dialogue_and_meta.py::test_blocks_carry_voice_per_scene` — the canonical assertion for the grok format
- `nodes/llm/minimax_h3_loop_prompts.py` lines ~2407-2566 — `_speech_act_sentence` two-branch grammar
- `output/live_e2e/e2e_*.mp4` — the 8 full-video renders from 2026-09-22 across A/B/C/C2/C3/C4/C5/D scenarios