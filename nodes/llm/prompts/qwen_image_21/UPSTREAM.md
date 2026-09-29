# qwen_image_21 prompts

System prompts for the `QwenImage21PromptGenerator` ComfyUI node.

## Source

- Model: [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (T2I + image edit; not the legacy Qwen-Image / Edit-2511)
- Prompt-Enhancer rule source (distilled into the .txt files below; **weights not required at runtime**):
  - [Qwen/Qwen-Image-2.1-PE-T2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I) — ~9 GB PE weights
  - [Qwen/Qwen-Image-2.1-PE-I2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I) — ~9 GB PE weights
- Official demo space (prompt patterns + 26 case studies):
  [Qwen/Qwen-Image-2.1](https://huggingface.co/spaces/Qwen/Qwen-Image-2.1)
- ComfyUI tutorial: <https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1>

The PE route is intentionally **not** the default path: each PE weight is ~9 GB
on top of the 7 B DiT itself, which most users can't afford. The system
prompts below encode the same rules so the API-only path reproduces the
behaviour.

## File map

| File | Mode | Source | Notes |
| --- | --- | --- | --- |
| `_enhance_t2i.txt` | `t2i` | Official PE-T2I observer-8-step system + MieNodes wrapper | English observer paragraph; fires RGBA sandwich only when user asks |
| `_enhance_t2i_rgba.txt` | `t2i_rgba` | `_enhance_t2i.txt` + forced Step 9 (RGBA sandwich always on) | RGBA bookend sentences guaranteed regardless of user phrasing |
| `_enhance_edit.txt` | `edit` | Official Edit Prompt Enhancer v2 + Space / MieNodes hardening + English-default language policy | Rewrite body always English; verbatim text in straight quotes preserves user script. Caption DATA injected by node. |
| `_enhance_multi_ref.txt` | `multi_ref` | `_enhance_edit.txt` + multi-image slot bookkeeping | For N≥2; node passes `reference_count` so the LLM writes exactly that many slots, never more |
| `_enhance_remove_bg.txt` | `remove_bg` | (Node short-circuits; file kept for documentation only) | Node returns a fixed `--- BEGIN enhanced_prompt ---...--- END enhanced_prompt ---` block, **no LLM call** |
| `_caption_image.txt` | caption stage | MieNodes original (mirrors MiniMax H3 Loop caption prompt, sharpened for Qwen-Image-2.1 use case) | Per-image caption system prompt. Used only when `reference_images` is wired AND mode in `{edit, multi_ref}` |

Files prefixed with `_` are module-private: `load_prompt_text` and
`list_usable_builtin_prompts` skip them, so they never appear in the
`CustomSystemPromptGenerator` dropdown. The generator node reads them via its
own `_MODE_TO_PROMPT` map.

## Sync procedure

1. Fetch the upstream `.txt` from
   `E:\CC\data\qwenimage21\_enhance_t2i_system.txt` and `_enhance_edit.txt`.
2. Diff against the local copies in this directory.
3. If upstream changed, copy verbatim into the matching local file. For T2I
   edits, the `## MieNodes wrapper` tail (forcing `--- BEGIN enhanced_prompt ---`
   … `--- END enhanced_prompt ---`) must remain the last section; without it
   the node regex extractor will raise. For Edit, the `【输出格式 — MieNodes】`
   section already embeds the same wrapper and must remain.
4. Re-run `pytest tests/test_qwen_image_21_prompt_generator.py`. A red test is
   a signal to review the upstream change deliberately.

## Handoff

- Brief: `E:\CC\data\qwenimage21\Qwen-Image-2.1-提示词增强节点开发brief.md`
- Handoff doc: `E:\CC\data\qwenimage21\MieNodes-Qwen-Image-2.1-提示词增强-交接文档.md`
- Handoff owner: SweetValberry / XLHL — 2026-09-27 (Asia/Shanghai)

## Out of scope (do not add here)

- Video / audio scripts (this is a prompt-only node).
- Cloud-space API model names — never hardcode them in the node or the prompts.
- Hardcoded aspect ratios / resolutions — the sampler handles that; the .txt
  files explicitly forbid writing them into the prompt.

## Widget shape (v1)

| Widget | Type | Purpose |
| --- | --- | --- |
| `llm_service_connector` | `LLMServiceConnector` (required) | Connector wired by the user. |
| `mode` | dropdown (required) | One of `t2i` / `t2i_rgba` / `edit` / `multi_ref` / `remove_bg`. |
| `user_input` | `STRING` multiline (required) | The rough draft. |
| `seed` | `INT` (required) | Forwarded to the LLM; `control_after_generate = True`. |
| `reference_images` | `IMAGE` (optional) | A batch of reference images. For `mode in {edit, multi_ref}`, the node captions each image via the bundled `_caption_image.txt` system prompt and injects the captions into the rewrite user message as locked identity / wardrobe / palette DATA. Pixels are NOT visible to the rewrite LLM (only to the captioner LLM). Recommended ceiling: 10; beyond that the node logs a warning but passes the full count. |
| `caption_mode` | dropdown (optional) | Cache strategy for the captioner. Mirrors `MiniMaxH3LoopPromptGenerator.CAPTION_MODES`: `cache_memory_disk` (recommended, default; in-memory + persistent on-disk), `cache_memory_only`, `no_cache`, `force_recaption_once`. |
| `temperature` | `FLOAT` (optional, default 0.7) | 0.0 – 2.0, step 0.05. |
| `max_tokens` | `INT` (optional, default 16384) | 64 – 32768. Reasoning models count their thinking against this cap; 4096 ran out on the first H3 enhancer iteration. |
| `timeout` | dropdown (optional, default 300) | 60 / 120 / 300 / 600 seconds. |

`remove_bg` short-circuits: the node returns a fixed BG-remove block
without an LLM call and ignores `user_input` / `reference_images`.

The earlier handoff draft proposed `slot_roles` and `verbatim_text`
optional STRING widgets (so users could hand-number slots and force
verbatim strings into the rewritten prompt). Both were removed in
v1 — the model routinely accumulates "任意字符串" / "image1=人物" / "店招
逐字保留" in plain user_input just fine, and the two widgets added UX
friction without giving the LLM anything it couldn't infer from the
draft. They can be re-introduced in v2 if a concrete need shows up.

## Caption disk cache

When `caption_mode = cache_memory_disk` (default) and `reference_images`
is connected, each image goes through a two-tier cache:

- **In-memory**: per-node-instance dict keyed by `sha256(image_data_url)
  | sha256(caption_prompt)`. Lives for the lifetime of the node;
  cleared on ComfyUI restart or workflow reload.
- **On-disk**: `<ComfyUI output dir>/mien_nodes/caption_cache/<sha256>.txt`
  — text files containing the caption. Resolution order:
  1. `folder_paths.get_output_directory()` (ComfyUI runtime)
  2. `<repo>/output/mien_nodes/caption_cache/` (fallback for tests /
     standalone imports)

Cache key includes a sha256 of the caption system prompt, so editing
`_caption_image.txt` invalidates the entire cache atomically — next
run re-captions every image. Use `force_recaption_once` after a prompt
upgrade to bypass in-memory and disk tiers once.

`MiniMaxH3LoopPromptGenerator` uses the same key derivation and disk
root, so users who run both nodes can share the cache directory if
they ever want to (the two caption prompts are *not* identical, so the
sha256-of-prompt half of the key always differs — no false hits, just
shared layout).