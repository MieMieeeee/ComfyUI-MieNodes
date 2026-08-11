# H3 prompts

Prompt templates for the `MiniMaxH3PromptGenerator` ComfyUI node.

These templates encode the official MiniMax H3 video-prompt-writing guidance so
the node can rewrite a user's rough idea into a structured H3 prompt.

## Source

The structure, alignment directives, material-role taxonomy, and output rules
are derived from the official MiniMax H3 documentation:

| Source | URL | Used for |
| --- | --- | --- |
| HF base guide | https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md | T2VA/I2VA/FL2VA/L2VA task structure, alignment directives, camera vocabulary, dialogue/sound writing |
| HF ref guide | https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md | Ref2VA six-section structure + `<Picture N>`/`<Subject N>`/`<Video N>`/`<Audio N>` material tags |
| Official skill (core) | https://github.com/MiniMax-AI/MiniMax-H3/tree/main/skills/h3-prompt-writing | The five-principle "core mental model" condensed into the system prompts |
| Official Chinese example page | https://platform.minimaxi.com/docs/guides/video-prompt | The brand-film / MV reference samples adapted into `examples_i2v.txt` |

The few-shot examples in `examples_t2v.txt` / `examples_i2v.txt` are adapted
from the H3 project's locally-validated prompt set (see the project knowledge
base) and the official example page; they are rewritten to fit this node's
field layout and are NOT byte-level copies of any single upstream file.

## File map

### System prompts (stage-2 system role)

| File | Purpose |
| --- | --- |
| `system_t2v.txt` | T2VA system prompt — five core principles, three-field structure, per-shot seven-element checklist, output rules |
| `system_reference.txt` | I2VA/FL2VA/L2VA/Ref2VA/S2V system prompt — adds material-role taxonomy, alignment-directive rule, "do not re-describe the reference" rule, label-consistency rule |
| `caption_reference.txt` | Stage-1 system prompt — caption reference images / sampled video frames (identity, wardrobe, scene, features to preserve) before the stage-2 enhance |

### Alignment directive snippets (prepended to the I2V/R2V user prompt)

| File | Purpose |
| --- | --- |
| `align_i2v_first.txt` | I2VA first-frame directive (verbatim, `at 0.00 seconds ... <Picture 1> ... is fully referenced`) |
| `align_i2v_first_last.txt` | FL2VA first+last directive; contains a `{duration}` placeholder substituted at runtime |
| `align_i2v_last.txt` | L2VA last-frame directive; contains a `{duration}` placeholder substituted at runtime |

### User templates (stage-2 user role, `.format()`-substituted)

| File | Placeholders |
| --- | --- |
| `user_t2v_template.txt` | `{idea}` `{duration}` `{aspect_ratio}` `{category_advice}` `{examples}` |
| `user_i2v_template.txt` | `{align_directive}` `{idea}` `{caption}` `{duration}` `{aspect_ratio}` `{category_advice}` `{examples}` |

### Few-shot examples

| File | Source |
| --- | --- |
| `examples_t2v.txt` | Adapted from a locally-validated kinetic-typography T2VA prompt + an environmental T2VA sample |
| `examples_i2v.txt` | Adapted from a brand-film I2VA sample + a five-character Ref2VA MV + an ancient-Chinese Ref2VA MV |

## Placeholder convention

Placeholders like `{duration}`, `{idea}`, `{caption}` are preserved verbatim by
the loader (`prompts/loader.py` never calls `.format()`); they are substituted
at runtime by the wrapper functions in `h3_prompts.py`. Angle-bracket material
tags such as `<Picture 1>`, `<d>[zh] ...</d>` are NOT str.format placeholders
and pass through untouched. If a future example contains a literal `{` or `}`
it must be doubled (`{{` / `}}`).
