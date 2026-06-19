# Bernini prompts — upstream sync

The `.txt` / `.json` files in this directory are a verbatim copy of the system
prompts and task templates from **[bytedance/Bernini](https://github.com/bytedance/Bernini)**,
module `bernini.prompt_enhancer` (Apache-2.0). They are data only; the runtime
loads them via `nodes/llm/prompts/loader.py`.

## File ← upstream symbol map

| File | Upstream symbol |
| :--- | :--- |
| `system_prompts.json` | `SYSTEM_PROMPTS` (per-task-code dict) |
| `t2v_a14b_en.txt` | `T2V_A14B_EN_SYS_PROMPT` |
| `_t2i_note.txt` | `_T2I_NOTE` (build block, not a standalone prompt) |
| `t2i_a14b_en.txt` | **derived** — see below |
| `r2v.txt` | `R2V_TEMPLATE` |
| `r2i.txt` | `R2I_TEMPLATE` |
| `vr2v.txt` | `VR2V_TEMPLATE` |
| `v2v.txt` | `V2V_TEMPLATE` |
| `i2i.txt` | `I2I_TEMPLATE` |
| `i2v.txt` | `I2V_TEMPLATE` |
| `vi2v.txt` | `VI2V_TEMPLATE` |
| `ads2v.txt` | `ADS2V_TEMPLATE` |
| `ri2i.txt` | `RI2I_TEMPLATE` — **MieNodes extension**, NOT upstream (fills the i2i↔r2i gap) |

Placeholders (`{image_num}`, `{original_text}`, `{user_prompt}`, `{ref_num}`) are
kept verbatim — `BerniniPromptGenerator` calls `.format(**kwargs)` on them.

## Derived T2I (why a snapshot, not runtime derivation)

Upstream defines `T2I_A14B_EN_SYS_PROMPT = _T2I_NOTE + T2V_A14B_EN_SYS_PROMPT.replace(old, new)`.
Doing that `.replace()` at runtime is fragile: if upstream ever reworded the
targeted sentence, the replace would silently no-op and T2I would degrade into
the motion-describing T2V. So `t2i_a14b_en.txt` stores the **expanded result**,
and the derivation is replayed at build time by `scripts/regen_bernini_prompts.py`.

## Sync procedure (when upstream updates)

1. Copy the updated upstream values into the matching files above (except
   `t2i_a14b_en.txt`, which is regenerated).
2. Run `python scripts/regen_bernini_prompts.py` to rebuild `t2i_a14b_en.txt`.
3. Run `pytest tests/test_prompt_snapshot.py` — intentional upstream changes
   turn the snapshot tests RED; review and update expectations deliberately.
