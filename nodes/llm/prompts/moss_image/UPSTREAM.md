# Upstream provenance — MOSS-Image prompt enhancer

Snapshot date: **2026-10-25** (Asia/Shanghai).

## Files

| File | Source | Status |
|---|---|---|
| `upsampler_system_prompt.txt` | Comfy-Org official template `image_moss_image_t2i.json`, subgraph node 14 (the `TextGenerate` node's `system_prompt` widget). Byte-identical to `E:\CC\data\moss\raw\moss_upsampler_system_prompt.txt`. | **official, verbatim — never edit** |
| `layout_system_prompt.txt` | Written for MieNodes. Encodes the official MOSS-Image Layout *skill* contract (`OpenMOSS/MOSS-Image → skills/moss-image-prompt-enhancer`, `SKILL.md` + `references/layout.md`) in the form the node needs: a JSON draft with absolute-pixel boxes. | **MieNodes-authored** |

Character counts / digests of the official file (also asserted in
`tests/test_moss_image_prompt_generator.py`):

```
upsampler_system_prompt.txt:  8398 characters
sha1 prefix:                  ab424ab68233
```

The official prompt is shared byte-for-byte by all four official templates
(`t2i`, `base_t2i`, `t2i_bboxes`, `base_t2i_bboxes`); only the bbox variants
add a trailing newline. It is one artifact, used here for the caption mode.

## Why there are two prompts

The official upsampler prompt and the official Layout skill are **two
separate official artifacts** with different output contracts:

- The upsampler prompt answers with **prose**. Its "Bounding boxes:" section
  explicitly forbids JSON (`No coordinates, no JSON, no prose paragraphs`) and
  asks for one `index | English description` line per box.
- The Layout skill answers with a **JSON draft** (`summary` + a flat `items`
  array, absolute-pixel `xyxy` boxes) which `scripts/compile_prompt.py` turns
  into the one-line compact JSON the image model actually reads.

MOSS-Image consumes the compiled JSON, so the layout mode has to speak the
skill's contract. The node therefore asks for the official draft schema in
the layout mode and compiles it with a faithful port of the official
compiler (`compile_layout_draft` in `nodes/llm/moss_image_prompts.py`).

## The three coordinate spaces

This is where a MOSS Layout implementation goes wrong. The node keeps the
three apart:

| Space | Order / unit | Where it appears |
|---|---|---|
| Canvas grid | `[y_min, x_min, y_max, x_max]`, 0–1000, per axis, origin top-left | `CreateBoundingBoxes` → `elements` |
| Draft | `[x_min, y_min, x_max, y_max]`, absolute pixels, origin top-left | what the LLM is asked to produce |
| Model | `[y_min, x_min, y_max, x_max]`, absolute pixels **centred** on the canvas | the compiled one-line JSON |

The official upsampler prompt says to use the given coordinates as they are
and never convert them, so a raw 0–1000 grid handed to the model would be
read as pixels. `elements_to_boxes` converts the grid to the draft space
before the model sees it.

The compiler encloses each draft box in whole pixels (floor on the near
edge, ceil on the far edge) and then subtracts the canvas centre, exactly as
the official script does. The canvas size is **not** part of the compiled
prompt: it has to be passed to the image workflow separately.

## The upsampler rewrites layouts

The official bbox template ships this warning:

> Prompt upsampler is **on** in this template, so it can rewrite the prompt
> and drop the layout — set `enable_prompt_upsampler` to `false` (inside the
> subgraph) when the boxes must survive verbatim.

The node's `keep_layout_verbatim` (default on) is the same guarantee: the
user's drawn pixels win, and only the descriptions come from the model.
Turning it off lets the model plan the layout entirely.

## Decoding parameters

The official template uses `max_length=4096`, `temperature=0.7`, `top_k=64`,
`top_p=0.95`, `min_p=0.05`, `repetition_penalty=1.0`, `presence_penalty=0`,
`thinking=false`. The node exposes `temperature` and `max_tokens` only: no
connector in this project forwards the rest, and the bundled service layer
deliberately does not carry them. `max_tokens` defaults to 16384 rather than
the official 4096 because reasoning models spend part of the budget thinking
before the caption — the same reasoning as the FLUX.3 Klein node.

## Public links

- <https://github.com/OpenMOSS/MOSS-Image>
- <https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md>
- <https://github.com/OpenMOSS/MOSS-Image/tree/main/skills/moss-image-prompt-enhancer>
- <https://docs.comfy.org/tutorials/image/moss-image/moss-image>
