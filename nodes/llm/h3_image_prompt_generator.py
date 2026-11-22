"""ComfyUI compatibility shell; generation implementation lives in mie_prompt."""
from __future__ import annotations
from mie_prompt import h3_image_prompt_generator as _impl
for _name in dir(_impl):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_impl, _name)

class H3ImagePromptGenerator:
    """ComfyUI node: rough draft -> H3-Image-ready prompt + aspect ratio.

    Sits next to ``MiniMaxH3PromptGenerator`` (the VIDEO model) in the same
    Prompt Generator category -- the two are different models and must not
    be swapped.

    The ``prompt`` output plugs into the H3-Image conditioning node;
    ``aspect_ratio`` plugs into the resolution control. ``raw_reply`` is
    for debugging only.
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
                            # English first, then Chinese -- mirrors the other
                            # prompt generators so users can read either.
                            "Mode / 模式:  "
                            "t2i 文生图增强 (默认) — the brief becomes an "
                            "English caption plus an aspect ratio, as one "
                            "JSON object.  "
                            "edit 图像编辑增强 (含参考生图) — an edit "
                            "instruction plus the reference images (the "
                            "picture being edited first, extra references "
                            "after) become a three-part caption plus a "
                            "target ratio; needs a vision-capable connector."
                        ),
                    },
                ),
                "user_input": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Rough idea / 草稿: a sentence, an outline, a few "
                            "lines, or a free paragraph. The rewrite expands "
                            "it into the prompt H3-Image consumes. The body "
                            "is always English; text you put in quotes is "
                            "kept verbatim, in its own language. / 扩写正文"
                            "一律英文；引号内的文字逐字保留。"
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
                            "the connector pick a fresh seed. The upstream PE "
                            "is not deterministic, so a fixed seed + a fixed "
                            "model is the only lever for reproducing a run."
                        ),
                    },
                ),
            },
            "optional": {
                "reference_images": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Reference images, attached in order. Used by "
                            "edit -- that mode needs a "
                            "vision-capable connector, because the official "
                            "prompt requires the model to look at the "
                            "sources (<Picture N>, <Subject N>, transcribed "
                            "source text). The images are visible to the LLM "
                            "only; the image model receives them separately "
                            "through the H3-Image conditioning node. "
                            "按序附上的参考图。edit 模式必填，需使用"
                            "支持视觉的 LLM 连接器：官方 prompt 要求模型"
                            "真的去看参考图。图片只给 LLM 看，出图仍由"
                            "H3-Image Conditioning 节点单独接收。"
                        ),
                    },
                ),
                "aspect_ratio": (
                    "STRING",
                    {
                        "default": ADAPTIVE,
                        "tooltip": (
                            "Aspect ratio / 宽高比. 'adaptive' lets the "
                            "model choose. For t2i only the ten "
                            "official values apply: "
                            + " ".join(ALLOWED_T2I_RATIOS)
                            + ". For edit any positive-integer "
                            "W:H is allowed -- the official edit prompt "
                            "explicitly refuses a preset list. The ratio is "
                            "never pixels: the canvas (4 MP to 20 MP, best "
                            "around 16 MP, every edge a multiple of 32) "
                            "belongs to the image workflow. / 比例不是像素："
                            "画布尺寸由出图工作流决定。"
                        ),
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": DEFAULT_TEMPERATURE,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": (
                            "Sampling temperature / 采样温度. 0.0–2.0, step "
                            "0.05. The default is 1.0, the official PE "
                            "value -- do not copy the 0.7 the MOSS node "
                            "uses; MOSS's own official value is 0.7."
                        ),
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": DEFAULT_MAX_TOKENS,
                        "min": 64,
                        "max": 32768,
                        "tooltip": (
                            "Token budget / Token 上限. 64–32768. The "
                            "official PE asks for 4096, but a reasoning "
                            "model spends part of the budget thinking before "
                            "the caption starts, and a reply that runs out "
                            "mid-caption is the common failure -- so the "
                            "default is 16384, and the call auto-escalates "
                            "when the think chain starves the answer."
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

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "aspect_ratio", "raw_reply")
    FUNCTION = "enhance"
    CATEGORY = MY_CATEGORY

    def enhance(
        self,
        llm_service_connector,
        mode,
        user_input,
        seed=None,
        reference_images=None,
        aspect_ratio=ADAPTIVE,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        timeout=_DEFAULT_TIMEOUT,
    ):
        mode_code = parse_mode(mode) or mode
        urls = self._reference_urls(mode_code, reference_images)
        ratios = _reference_ratios(reference_images) if urls else []
        requested = self._requested_ratio(mode_code, aspect_ratio)

        enhancer = H3ImagePromptEnhancer(
            llm_service_connector,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        prompt, ratio, raw = enhancer(
            mode=mode_code,
            user_text=user_input,
            image_urls=urls,
            reference_ratios=ratios,
            aspect_ratio=requested,
            seed=seed,
        )
        return (prompt, ratio, raw)

    @staticmethod
    def _reference_urls(mode: str, reference_images) -> list[str]:
        """Validate the reference images for ``mode`` and encode them.

        The editing modes are VLM-only by design: the official edit prompt
        makes the model look at the sources, so there is no caption-then-
        rewrite fallback -- a text-only reply would be the model inventing
        what it supposedly saw.
        """
        urls = (
            image_tensor_batch_to_data_urls(reference_images)
            if reference_images is not None
            else []
        )
        if is_image_mode(mode) and not urls:
            raise ValueError(
                f"H3Image: mode {mode!r} needs the reference images and a "
                f"vision-capable connector -- connect them to "
                f"reference_images / {mode} 模式需要参考图：请把图片接到 "
                f"reference_images，并使用支持视觉的 LLM 连接器"
            )
        if not is_image_mode(mode) and urls:
            mie_log(
                f"H3Image: mode {mode!r} takes no reference images, but "
                f"{len(urls)} arrived on reference_images -- ignoring them / "
                f"{mode} 模式不使用参考图，已忽略"
            )
            return []
        if len(urls) > MAX_REFERENCE_IMAGES:
            raise ValueError(
                f"H3Image: {len(urls)} reference images exceed the official "
                f"limit of {MAX_REFERENCE_IMAGES} / 参考图超过官方上限"
            )
        if len(urls) > _DEFAULT_REFERENCE_CAP:
            mie_log(
                f"H3Image: {len(urls)} reference images exceed the node cap "
                f"of {_DEFAULT_REFERENCE_CAP}; using the first "
                f"{_DEFAULT_REFERENCE_CAP} / 参考图超过节点上限，已截断"
            )
            urls = urls[:_DEFAULT_REFERENCE_CAP]
        return urls

    @staticmethod
    def _requested_ratio(mode: str, aspect_ratio: str) -> str:
        """The ratio the user asked for, checked against the mode's rules."""
        value = (aspect_ratio or "").strip() or ADAPTIVE
        if value.lower() == ADAPTIVE:
            return ADAPTIVE
        if mode == "t2i":
            if value not in ALLOWED_T2I_RATIOS:
                mie_log(
                    f"H3Image: {value!r} is not one of the ten official t2i "
                    f"ratios ({' '.join(ALLOWED_T2I_RATIOS)}); using "
                    f"adaptive / 该比例不在官方十值内，改用 adaptive"
                )
                return ADAPTIVE
            return value
        # edit: any positive-integer W:H.
        left, _, right = value.partition(":")
        try:
            if int(left) > 0 and int(right) > 0:
                return value
        except ValueError:
            pass
        mie_log(
            f"H3Image: {value!r} is not a W:H ratio; using adaptive / 该值"
            f"不是 W:H 比例，改用 adaptive"
        )
        return ADAPTIVE

    def is_changed(
        self,
        llm_service_connector,
        mode,
        user_input,
        seed=None,
        reference_images=None,
        aspect_ratio=ADAPTIVE,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        timeout=_DEFAULT_TIMEOUT,
    ):
        mode_code = parse_mode(mode) or mode
        h = hashlib.md5(usedforsecurity=False)
        for part in (
            user_input or "",
            mode_code or "",
            str(seed),
            str(aspect_ratio or ""),
            str(temperature),
            str(timeout),
            str(max_tokens),
        ):
            h.update(part.encode("utf-8"))
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
        # Hash the official prompt bodies so editing one of the bundled
        # files in development invalidates ComfyUI's node cache.
        try:
            from .h3_image_prompts import build_system_prompt

            h.update(build_system_prompt(mode_code).encode("utf-8"))
        except Exception:
            pass
        try:
            h.update(llm_service_connector.get_state().encode("utf-8"))
        except AttributeError:
            h.update(str(getattr(llm_service_connector, "api_url", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "api_token", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "model", "")).encode("utf-8"))
        return h.hexdigest()
