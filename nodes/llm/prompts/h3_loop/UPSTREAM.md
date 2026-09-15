# H3 Loop Plan prompts

Prompt templates for the `MiniMaxH3LoopPromptGenerator` ComfyUI node
(nodes/llm/minimax_h3_loop_prompt_generator.py). The node turns a concept
plus an optional IMAGE stream (or `references_text` fallback) into a
`plan_json` string that plugs into ethanfel/ComfyUI-MiniMaxH3-Context-Loop's
`MiniMaxH3ChainPlanModern.plan_json_input` (the "Production Plan" node).

## Source

| Source | URL | Used for |
| --- | --- | --- |
| H3_CHAIN_FORMAT_GUIDE.md | https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop/blob/main/H3_CHAIN_FORMAT_GUIDE.md | plan JSON schema (`defaults`/`shots[]`/`prompt_prefix`), 17k+5 length grid, seed-as-string rule, 1-128 shots |
| docs/SCENE_AUTHORING.md | https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop/blob/main/docs/SCENE_AUTHORING.md | prompt line arrays, prompt_prefix prepended with one blank line, seamless-chain authoring (end mid-action, next shot continues) |
| docs/AUDIO_AND_CONTINUITY.md | https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop/blob/main/docs/AUDIO_AND_CONTINUITY.md | audio continuity across scene boundaries (carry the bed, don't restart) |
| User workflow `1 (2).json` (Production Plan node) | local | real plan_json shape: `prompt` as string[] with bare three-section headers, `seed` as digit string, per-shot `length`; no steps/continuation_mode/context_length/width/height in the JSON (generation parameters stay on the Plan node's widgets) |
| Project H3 system prompt | `prompts/h3/system_t2v.txt` | Stage-2 base system prompt (three-section structure, camera vocabulary, sound rules) reused via `h3_prompts.system_t2v_prompt()` |

## File map

| File | Purpose |
| --- | --- |
| `shot_system_addendum.txt` | Appended to the shared H3 t2v system prompt: single-continuous-clip rules, bare headers, mid-action endings, chain no-reset rule |
| `shot_user_template.txt` | per_shot user turn; placeholders `.format()`-ed by `minimax_h3_loop_prompts.build_shot_user_text` |
| `shot_continuation.txt` | continuation block injected for clips 2+; placeholders `{previous_id}` `{previous_tail}` |
| `prefix_synth_system.txt` | Stage-1 shared prompt_prefix derivation system role |
| `prefix_synth_user.txt` | Stage-1 user turn; placeholders `{concept}` `{genre_advice}` `{language_name}` |
| `single_call_format.txt` | format directive appended in `single_call` generation mode; placeholder `{clip_count}` |
| `ref2v_addendum.txt` | Stage-2 system prompt append for `ref2va` mode (six bare headers, summary `[reference generation]`, retention markers, deterministic subject binding) |
| `shot_continuation_ref2v.txt` | continuation block for ref2va clips 2+; six-section carry-over with subject_definitions re-assertion; placeholders `{previous_id}` `{previous_subject_definitions}` `{previous_description}` `{previous_soundscape}` |

## Image inputs (Stage 0.6)

The node accepts ONE `IMAGE` socket (a ComfyUI IMAGE batch) named
`images`. Wiring one batch covers all three reference modes — the node
routes frames to the right upstream picture sink based on
`reference_mode`:

| Mode | What the `images` batch becomes | Upstream wiring |
| --- | --- | --- |
| `t2va` | ignored (warning in preflight; no upstream image wiring) | — |
| `i2va` | only the FIRST frame is captioned; extras dropped with a preflight warning | `MiniMax H3 First-Scene Image Gate` `image` = batch[0] (one image, scene 1 only) |
| `fl2va` | batch[0] → `Picture 1` = the OPENING frame; batch[k≥1] → `Picture k+1` = scene k's end target | gate `image` = batch[0]; `MiniMax H3 Chain Frame Index Switch` `frame_1` = batch[1], `frame_2` = batch[2], … (scene j ends on `frame_j`; selection wraps after the last connected slot) |
| `ref2va` | every frame → `Picture 1..N`; six-section schema; all active every scene | `MiniMax H3 Reference to Video images` = the whole batch |

Wiring example — a 4-image batch `[A, B, C, D]` on `fl2va`:

- `A` (Picture 1) → First-Scene Image Gate `image` — the opening frame
- `B` (Picture 2) → switch `frame_1` — scene 1's end target
- `C` (Picture 3) → switch `frame_2` — scene 2's end target
- `D` (Picture 4) → switch `frame_3` — scene 3's end target (later scenes wrap back to `frame_1`)

For an A→B→A chain, wire a 3-frame batch `[A, B, A]` and write the
intent in `user_input` (any natural phrasing: `从图一到图二再回到图一`
or `A→B→A`). The deterministic per-scene keyframe idiom in the plan
follows the slot math above, so the storyboard and the wiring stay in
step. 1–9 frames are supported; the node hard-caps at
`_MAX_REFERENCE_IMAGES = 9` and warns when extra frames are dropped.

When `images` is connected, the node runs one LLM caption call per
frame (using `h3_prompts.caption_reference_prompt()` and the shared
`core.utils.image_tensor_batch_to_data_urls` helper) and auto-builds
the manifest. The resulting `references_text` semantics fully
supersede anything typed into the `references_text` widget — a
preflight warning names the conflict so users know their text was
ignored.
