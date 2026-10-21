# Upstream provenance — FLUX.3 Image Klein prompt upsampler

Snapshot date: **2026-10-19** (Asia/Shanghai).

## Files

| File | Source | Size |
|---|---|---|
| `system_prompts.json` | HF [`black-forest-labs/flux-3-image-klein-pup`](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup) → [`system_prompts.json`](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup/blob/main/system_prompts.json) | 12,602 bytes |
| `pup_prompts_external_vlm.json` | Inference repo [PR #5](https://github.com/black-forest-labs/flux-3-image-klein/pull/5) → `src/flux3/pup_prompts_external_vlm.json` | 103,088 bytes |

Both are byte-for-byte copies. **Never edit their contents.** Upstream states the PUP
model is a full fine-tune of Qwen3.5-9B on the captions it writes, and the HF model
card says any other LLM can reproduce the behaviour using these system prompts.
Rewriting them invalidates the node.

Character counts (upstream self-check, also asserted in
`tests/test_flux3_klein_prompt_generator.py`):

```
short  system_prompts.json:            t2i 3404 · i2i 2677 · t2i_bbox 3302 · i2i_bbox 3034
long   pup_prompts_external_vlm.json:  t2i 46615 · i2i 6194 · t2i_bbox 27258 · i2i_bbox 21614
```

The two packs are *not* the same text — same four keys, same purpose family, two
separately published official prompt sets. The short pack is the default; the long
pack (PR #5, "Smaller GPUs") carries many more few-shot examples and is roughly an
order of magnitude longer.

## Read this before wiring a workflow

**Turn Klein's own upsampler off.** Upstream is explicit that Klein is trained on the
expanded captions, but running both means the text is expanded twice:

- ComfyUI: `enable_prompt_upsampler = false`
- CLI: `--no-pup`

**Editing voice is instruction-style here.** BFL's Comfy tutorial says "Write the
result, not the change", while the PUP `i2i` system prompt, BFL's single-reference
guide and the official Space examples all say the opposite — describe the change and
close with what stays unchanged. Official sources disagree with each other. This node
follows the **PUP `i2i`** convention (instruction style), because that is what the
bundled prompt was trained for. The open question is tracked as TO20 in the source
research doc; if a side-by-side test settles it on result-style, add a separate
bundled system prompt rather than editing `i2i`.

## Decoding parameters

Official PUP uses `max_new_tokens=3584`, `enable_thinking=False`, and
`presence_penalty` of **1.5 for t2i** / **0 for editing**. The node exposes
`temperature` and `max_tokens` only: no connector in this project forwards
`presence_penalty`, and the bundled service layer deliberately does not carry it.
`max_tokens` defaults higher than 3584 because reasoning models spend part of the
budget thinking before the caption.

## Public links

- <https://docs.bfl.ml/flux_3/flux3_image_klein>
- <https://docs.comfy.org/tutorials/flux/flux-3-klein>
- <https://docs.bfl.ml/guides/prompting_editing_single_reference>
- <https://docs.bfl.ml/guides/prompting_layout>
