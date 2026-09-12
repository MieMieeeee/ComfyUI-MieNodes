# H3 Loop Plan prompts

Prompt templates for the `MiniMaxH3LoopPromptGenerator` ComfyUI node
(nodes/llm/minimax_h3_loop_prompt_generator.py). The node turns a concept
plus a storyboard (e.g. `MiniMaxH3StoryboardGenerator.shots_json`) into a
`plan_json` string that plugs into ethanfel/ComfyUI-MiniMaxH3-Context-Loop's
`MiniMaxH3ChainPlanModern.plan_json_input` (the "Production Plan" node).

## Source

| Source | URL | Used for |
| --- | --- | --- |
| H3_CHAIN_FORMAT_GUIDE.md | https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop/blob/main/H3_CHAIN_FORMAT_GUIDE.md | plan JSON schema (`defaults`/`shots[]`/`prompt_prefix`), 17k+5 length grid, seed-as-string rule, 1-128 shots |
| docs/SCENE_AUTHORING.md | https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop/blob/main/docs/SCENE_AUTHORING.md | prompt line arrays, prompt_prefix prepended with one blank line, seamless-chain authoring (end mid-action, next shot continues) |
| docs/AUDIO_AND_CONTINUITY.md | https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop/blob/main/docs/AUDIO_AND_CONTINUITY.md | audio continuity across scene boundaries (carry the bed, don't restart) |
| User workflow `1 (2).json` (Production Plan node) | local | real plan_json shape: `prompt` as string[] with bare three-section headers, `seed` as digit string, per-shot `steps`/`length`; no continuation_mode/context_length/width/height in the JSON |
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
