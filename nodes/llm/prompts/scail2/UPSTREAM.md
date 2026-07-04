# SCAIL-2 prompts

Prompt templates for the `Scail2PromptGenerator` ComfyUI node.

## Source

[zai-org/SCAIL-2](https://github.com/zai-org/SCAIL-2), branch `wan-scail2`.

SCAIL-2 is a unified character animation / replacement model from Zhipu AI
(arXiv 2606.10804). It supports two modes:

- **Replacement** (`character_replacement` task): replace a subject in
  a source video with a reference-image character.
- **Animation** (`motion_transfer` task): animate a reference-image
  character with motion from a driving video.

## File map

### Replacement mode -- upstream verbatim

| File | Upstream source |
| --- | --- |
| `caption_replacement.txt` | `VIDEO_CAPTION_PROMPT` in [`prompt_enhancer.py`](https://raw.githubusercontent.com/zai-org/SCAIL-2/wan-scail2/prompt_enhancer.py) |
| `enhance_replacement.txt` | `REPLACEMENT_PROMPT_TEMPLATE` in the same file |
| `examples_replacement.txt` | [`examples/prompt_examples.txt`](https://raw.githubusercontent.com/zai-org/SCAIL-2/wan-scail2/examples/prompt_examples.txt) in the same repo |

These three files are byte-level copies of the upstream contents (as of
2026-06). Placeholders (`{instruction}`, `{caption}`, `{examples}`) are
preserved verbatim and substituted at runtime via `.format(...)`.

### Motion-transfer mode -- MieNodes original

Upstream has no prompt enhancer for animation mode -- the canonical input is
a 4-word prompt like "the girl is dancing" (see [`examples/input.txt`](https://raw.githubusercontent.com/zai-org/SCAIL-2/wan-scail2/examples/input.txt)).
The three files below are MieNodes designs and are NOT from upstream:

| File | Purpose |
| --- | --- |
| `caption_motion_transfer.txt` | Stage 1: caption the driving video (focus on motion) |
| `enhance_motion_transfer.txt` | Stage 2: rewrite as a positive animation prompt |
| `examples_motion_transfer.txt` | MieNodes-original few-shot examples (dancing / walking / yoga motions, one descriptive paragraph each, matching the prompt style required by `enhance_motion_transfer.txt`) |

If upstream later ships a motion-transfer prompt enhancer, these three
files should be either deleted (in favor of the upstream files) or moved
under a `mie_nodes_extension/` subfolder.

## Sync procedure (replacement files only)

1. Fetch upstream into a temp dir:

   ```bash
   curl -sL https://raw.githubusercontent.com/zai-org/SCAIL-2/wan-scail2/prompt_enhancer.py -o upstream_prompt_enhancer.py
   curl -sL https://raw.githubusercontent.com/zai-org/SCAIL-2/wan-scail2/examples/prompt_examples.txt -o upstream_prompt_examples.txt
   ```

2. Diff against the local copies.
3. If upstream changed, copy verbatim into the matching files here.
4. Re-run `pytest tests/test_scail2_prompts.py` -- a RED test is a signal
   to review the upstream change deliberately.