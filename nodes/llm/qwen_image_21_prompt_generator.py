"""ComfyUI compatibility shell; generation implementation lives in mie_prompt."""
from __future__ import annotations
from mie_prompt import qwen_image_21_prompt_generator as _impl
for _name in dir(_impl):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_impl, _name)

class QwenImage21PromptGenerator:
    """ComfyUI node: rough draft → Qwen-Image-2.1-ready STRING prompt.

    Sits next to ``MiniMaxH3LoopUserInputEnhancer`` in the same Prompt
    Generator category. The ``enhanced_prompt`` output plugs straight
    into the Qwen-Image-2.1 sampler / clip-text-encode nodes.

    Raises ``RuntimeError`` if the LLM reply does not contain a
    ``--- BEGIN enhanced_prompt ---`` ... ``--- END enhanced_prompt ---``
    block; the error includes the raw reply head so the user can
    diagnose. Silent fallback would let a misclassified reply flow into
    the sampler and waste the whole pipeline.

    ``mode = remove_bg`` is intentionally short-circuited: the node
    returns a fixed BG-remove block without an LLM call.

    The rewrite body is **always English**, regardless of the user's
    input language. The verbatim-language rule (text inside straight
    double quotes in the rewrite keeps the user's script) is encoded
    by the rewrite system prompt and the node does not second-guess it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "mode": (
                    list(MODES),
                    {
                        "default": MODES[0],
                        "tooltip": (
                            # English first, then Chinese — mirrors
                            # MiniMaxH3LoopPromptGenerator dropdown
                            # convention so users can read either.
                            "Mode / 模式:  "
                            "t2i 文生图(默认) — observer English paragraph.  "
                            "t2i_rgba 透明文生图 — same plus forced RGBA bookend sandwich.  "
                            "edit 单图编辑 — single-image edit instruction (rewrite body "
                            "is always English).  "
                            "multi_ref 多参考编辑 — multi-reference edit instruction "
                            "(<image1>…<imageN>; rewrite body always English).  "
                            "remove_bg 抠图 — fixed BG-remove block, no LLM call."
                        ),
                    },
                ),
                "user_input": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Rough idea / 草稿: a sentence, an outline, "
                            "a few lines, or a free paragraph. The rewriter "
                            "expands it into the prompt the Qwen-Image-2.1 "
                            "sampler consumes. Output is always English; "
                            "verbatim text inside straight double quotes in "
                            "your input is preserved as-is. / 扩写正文一律"
                            "英文；直双引号内的字符串保留原语种与逐字。"
                        ),
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": (
                            "Seed / 种子 forwarded to the LLM call. 0 lets "
                            "the connector pick a fresh seed."
                        ),
                    },
                ),
            },
            "optional": {
                "reference_images": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional IMAGE batch / 可选 IMAGE 批次.  "
                            "For mode=edit / multi_ref 单图编辑 / 多参考编辑, "
                            "the node captions each image (<image1>…<imageN>) "
                            "with the bundled _caption_image.txt system prompt "
                            "and injects the captions into the rewrite user "
                            "message as locked identity / wardrobe / palette "
                            "DATA. Pixels are visible to the captioner LLM "
                            "but NOT to the rewrite LLM. / 改写 LLM 看不到图。  "
                            "Recommended ceiling: 10 (model slot limit); "
                            "beyond that the node logs a warning but still "
                            "passes the full count.  "
                            "IMPORTANT / 注意: 多视图设定表是版式。四身横排喂进去，"
                            "成图会复制这排人，每个人一个姿势。先裁成单人"
                            "（正面或四分之三侧面）再接 reference_images，"
                            "输出用竖幅。编辑节点看见的仍是你接上的那张图；"
                            "这个节点只写提示词。 "
                            "A multi-view character sheet is a layout: four "
                            "standing views in, four poses out. Crop to one "
                            "person before wiring the image. "
                            "Wire any sampler / loader output that emits an "
                            "IMAGE batch — e.g. LoadImageBatch or a custom "
                            "collector."
                        ),
                    },
                ),
                "caption_mode": (
                    list(CAPTION_MODES),
                    {
                        "default": CAPTION_MODES[0],
                        "tooltip": (
                            "Caption cache strategy / 标注缓存策略 (only used "
                            "when reference_images is connected and mode in "
                            "{edit, multi_ref}).  "
                            "cache_memory_disk (recommended) 缓存:内存+磁盘 — "
                            "in-memory + persistent on-disk cache under "
                            "<ComfyUI output>/mien_nodes/caption_cache/.  "
                            "cache_memory_only 缓存:仅内存 — RAM-only cache, "
                            "lost on restart.  "
                            "no_cache 禁用缓存 — always bypass cache.  "
                            "force_recaption_once 本次强制重打标 — force fresh "
                            "captions now (e.g. after editing "
                            "_caption_image.txt)."
                        ),
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": _DEFAULT_TEMPERATURE,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": (
                            "Sampling temperature / 采样温度. 0.0–2.0, step 0.05. "
                            "Lower = strict adherence to system prompt; "
                            "higher = more surface variation."
                        ),
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": _MAX_TOKENS_DEFAULT,
                        "min": _MIN_MAX_TOKENS,
                        "max": _MAX_MAX_TOKENS,
                        "tooltip": (
                            "Token budget / Token 上限. 64–32768. Reasoning "
                            "models count their thinking against this cap; "
                            "16384 is the safe default for the whole plugin."
                        ),
                    },
                ),
                "timeout": (
                    [60, 120, 300, 600],
                    {
                        "default": _DEFAULT_TIMEOUT,
                        "tooltip": (
                            "Per-call timeout (seconds) / 单次调用超时(秒)."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    FUNCTION = "enhance"
    CATEGORY = MY_CATEGORY

    def enhance(
        self,
        llm_service_connector,
        mode,
        user_input,
        seed=None,
        reference_images=None,
        caption_mode=None,
        temperature=_DEFAULT_TEMPERATURE,
        max_tokens=_MAX_TOKENS_DEFAULT,
        timeout=_DEFAULT_TIMEOUT,
    ):
        # Resolve bilingual widget value (``t2i - 文生图(默认)``) to
        # the bare code the rest of the node branches on.
        mode_code = parse_mode_label(mode) or mode
        n_images = count_reference_images(reference_images)
        if n_images > _MAX_REFERENCE_IMAGES:
            mie_log(
                f"qwen_image_21_prompt_generator: {n_images} reference "
                f"images connected (recommended ceiling "
                f"{_MAX_REFERENCE_IMAGES}); passing full count to LLM, "
                "model may misbehave on >10 refs"
            )

        # Captions: only run for modes that consume them. The caption
        # task is its own LLM call per image (cache-fronted), so we
        # gate on both `n_images > 0` AND mode in {edit, multi_ref} —
        # t2i / t2i_rgba / remove_bg never see reference data.
        captions: Optional[list[str]] = None
        if n_images > 0 and mode_code in {"edit", "multi_ref"}:
            force, scope = resolve_caption_controls(
                caption_mode or CAPTION_MODES[0]
            )
            if scope != "disabled" or force:
                captioner = _ReferenceCaptioner(
                    llm_service_connector,
                    cache_scope=scope,
                    force_recaption=force,
                )
                try:
                    captions = caption_references(
                        captioner, reference_images, seed=seed
                    )
                except RuntimeError as exc:
                    # Empty-reply or encode failures are surfaced as a
                    # clear error — silent fallback would let the rewrite
                    # LLM hallucinate identity, which is the exact
                    # failure mode the captioner is here to prevent.
                    raise RuntimeError(
                        f"qwen_image_21_prompt_generator: captioning failed "
                        f"({exc}); retry with caption_mode=no_cache to "
                        "bypass the cache, or check the image content"
                    ) from exc
            else:
                # scope == "disabled" AND not force_recaption → user
                # opted out of caching AND captions, so emit no captions.
                # The rewrite LLM will fall back to "describe what the
                # user said in user_input" — same as v1 behaviour.
                captions = None

        block = run_enhancer(
            llm_service_connector,
            user_input,
            mode=mode_code,
            reference_count=n_images,
            captions=captions,
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return (block,)

    def is_changed(
        self,
        llm_service_connector,
        mode,
        user_input,
        seed=None,
        reference_images=None,
        caption_mode=None,
        temperature=_DEFAULT_TEMPERATURE,
        max_tokens=_MAX_TOKENS_DEFAULT,
        timeout=_DEFAULT_TIMEOUT,
    ):
        # Bilingual widget value → bare code, so the same hash lands on
        # identical configurations regardless of the menu language.
        mode_code = parse_mode_label(mode) or mode
        h = hashlib.md5(usedforsecurity=False)
        for part in (
            user_input or "",
            mode_code or "",
            str(seed),
            str(caption_mode or ""),
            str(temperature),
            str(timeout),
            str(max_tokens),
        ):
            h.update(part.encode("utf-8"))
        # Hash the IMAGE batch enough to detect a same-shape content
        # swap (mirrors MiniMaxH3LoopPromptGenerator.is_changed).
        if reference_images is None:
            h.update(b"images:none")
        else:
            try:
                shape = tuple(reference_images.shape)
            except AttributeError:
                shape = ()
            h.update(repr(shape).encode("utf-8"))
            h.update(str(getattr(reference_images, "dtype", "")).encode("utf-8"))
            sample_bytes = b""
            try:
                flat = (
                    reference_images.detach().cpu().reshape(-1)
                    if hasattr(reference_images, "detach")
                    and hasattr(reference_images, "cpu")
                    else (
                        reference_images.reshape(-1)
                        if hasattr(reference_images, "reshape")
                        else None
                    )
                )
                if flat is not None:
                    n = (
                        int(flat.shape[0])
                        if hasattr(flat, "shape")
                        else len(flat)
                    )
                    chunks = []
                    if n > 0:
                        head = flat[:1024]
                        mid_start = max(0, (n // 2) - 512)
                        mid = flat[mid_start:mid_start + 1024]
                        tail = flat[-1024:] if n > 1024 else flat
                        for part in (head, mid, tail):
                            if hasattr(part, "numpy"):
                                chunks.append(bytes(part.numpy().tobytes()))
                            elif hasattr(part, "tobytes"):
                                chunks.append(bytes(part.tobytes()))
                    sample_bytes = b"".join(chunks)
            except Exception:
                sample_bytes = b""
            h.update(
                hashlib.md5(
                    sample_bytes, usedforsecurity=False
                ).hexdigest().encode("ascii")
            )
        # Hash the bundled system-prompt bodies (rewrite + caption) so
        # editing either .txt file in development invalidates ComfyUI's
        # node cache automatically.
        try:
            sys_text = load_mode_prompt(mode_code)
        except (KeyError, FileNotFoundError):
            sys_text = ""
        h.update(sys_text.encode("utf-8"))
        try:
            sys_caption = caption_image_prompt()
        except FileNotFoundError:
            sys_caption = ""
        h.update(sys_caption.encode("utf-8"))
        try:
            h.update(llm_service_connector.get_state().encode("utf-8"))
        except AttributeError:
            h.update(str(getattr(llm_service_connector, "api_url", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "api_token", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "model", "")).encode("utf-8"))
        return h.hexdigest()