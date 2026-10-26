# H3 Storyboard prompts

Prompt templates for the `MiniMaxH3StoryboardGenerator` ComfyUI node
(nodes/llm/minimax_h3_storyboard_generator.py). The node turns a concept +
target shot count into a structured storyboard (JSON shot list), which feeds
`MiniMaxH3LoopPromptGenerator.shots_text` or any other plan builder.

## Source

The storyboard methodology encoded in `system_storyboard.txt` (beats before
shots, hook, shot-size variety, character anchor, transition logic, pacing,
continuity bible) is synthesized from mainstream storyboarding practice for
AI video pipelines:

| Source | URL | Used for |
| --- | --- | --- |
| Runway — How Do You Create a Storyboard? | https://runway.com/resources/ai-storyboard | 7-step workflow, per-panel metadata (frame, camera direction, shot type, duration, shot number), continuity review pass |
| The AI Prompt Shop — Storyboard a Consistent Multi-Shot AI Video | https://theaipromptshop.com/blogs/news/storyboard-multi-shot-ai-video | Beat sheet rule ("each beat changes the viewer's understanding"), continuity bible field list, shot-card fields, transition logic (matched movement / detail handoff / object crossing frame / shared shapes) |
| MindStudio — Storyboards and Character Sheets in AI Video | https://mindstudio.ai/blog/storyboards-character-sheets-ai-video-generation | Storyboard as a planning/prompt-management document with shot metadata |
| r/StableDiffusion — AI Video Prompt Structured Templates | https://www.reddit.com/r/StableDiffusion/comments/1j4yhlb/ai_video_prompt_writing_structured_templates_and/ | Structured per-shot templates for short-clip generators |

The JSON shot schema itself (id/description/shot_type/camera_movement/
transition_in/duration_seconds/narrative_beat/characters/props/notes) is
project-specific; the Python-side whitelists live in
`nodes/llm/minimax_h3_storyboard_prompts.py`.

## File map

| File | Purpose |
| --- | --- |
| `system_storyboard.txt` | System role: storyboard methodology + strict JSON output contract |
| `user_storyboard_template.txt` | User turn; placeholders `{concept}` `{shot_count}` `{style_name}` `{style_advice}` `{genre_advice}` `{language}` are `.format()`-ed by `minimax_h3_storyboard_prompts.build_user_text` |
| `system_asset.txt` | System role for portable v2 storyboard assets; loaded verbatim by `storyboard_asset_generator.PROMPT_PATH` (see sync pair below) |
| `asset.schema.json` | JSON schema of the v2 asset for external tooling |

## SYNC PAIR — system_asset.txt (never edit one side alone)

`system_asset.txt` (this directory) and
`docs/storyboard_asset_system_prompt.txt` (the external/no-ComfyUI path)
are two sibling renderings of ONE v2 contract. Their openings differ by
design (asset_header JSON flow vs free-form conversation flow); every
shared rule — roles, camera, shots/events, execution copy, duration,
live-render craft rules — must be updated in BOTH files in the same
commit, or the two generation paths drift. The 2026-10-26 review update
(fewest segments, no floor-pinned budgets, first-frame visible_state,
spoken-word numerals, role-name scope, constraints-not-compiled,
script-asserted visible objects) landed in both.
