"""ComfyUI compatibility shell; generation implementation lives in mie_prompt."""
from __future__ import annotations
from mie_prompt import moss_image_prompt_generator as _impl
for _name in dir(_impl):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_impl, _name)

class MossImagePromptUpsampler:
    """ComfyUI node: short request -> MOSS-Image caption or Layout prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "mode": (list(MODES), {"default": MODES[0]}),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "待扩写的简短请求，可中文；输出恒为英文描述，仅引号内"
                            "待渲染文字保留原文。布局模式下还用于说明整张画面。\n"
                            "Short request to expand. Any language is fine -- "
                            "the caption is always written in English, and only "
                            "text the model is asked to render (inside double "
                            "quotes) keeps its original language."
                        ),
                    },
                ),
                "canvas_width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 16,
                        "max": 16384,
                        "step": 16,
                        "tooltip": (
                            "画布宽度（像素，16 的倍数）。仅布局模式使用；必须与"
                            "出图工作流的 width 一致。\n"
                            "Canvas width in pixels (a multiple of 16). Layout "
                            "mode only; must match the width of the image "
                            "workflow."
                        ),
                    },
                ),
                "canvas_height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 16,
                        "max": 16384,
                        "step": 16,
                        "tooltip": (
                            "画布高度（像素，16 的倍数）。仅布局模式使用；必须与"
                            "出图工作流的 height 一致。\n"
                            "Canvas height in pixels (a multiple of 16). Layout "
                            "mode only; must match the height of the image "
                            "workflow."
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
                    },
                ),
            },
            "optional": {
                "ref_image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "只给 LLM 看的参考图（最多 4 张）。MOSS 本身不做"
                            "图生图，图像模型不会收到这张图。\n"
                            "Reference image for the LLM only (up to 4). MOSS "
                            "has no image-to-image mode -- the image model "
                            "never sees this."
                        )
                    },
                ),
                "elements": (
                    "ARRAY",
                    {
                        "tooltip": (
                            "Create Bounding Boxes 的 elements 输出。坐标是"
                            "0-1000 网格的 [y_min, x_min, y_max, x_max]，本节点"
                            "会自动换算成 MOSS 要求的绝对像素 xyxy，无需手写"
                            "JSON。仅布局模式使用。\n"
                            "Elements output from Create Bounding Boxes. The "
                            "boxes arrive on the official 0-1000 grid as "
                            "[y_min, x_min, y_max, x_max]; the node converts "
                            "them to the absolute-pixel xyxy that MOSS "
                            "expects, so no JSON is needed. Layout mode only."
                        )
                    },
                ),
                "character_boxes": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "把哪些元素框标成角色（人物），填元素序号，如 "
                            "\"1, 3\"。留空则全部按物体处理。仅布局模式使用。\n"
                            "Numbers of the element boxes to label as characters "
                            "instead of objects, e.g. \"1, 3\". Layout mode "
                            "only."
                        ),
                    },
                ),
                "keep_layout_verbatim": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "勾选时以你画的框为准：官方文档明确指出提示词扩写"
                            "会改写布局，勾选可保证框的像素不被模型改写，只取用"
                            "模型的描述文字。取消勾选则完全采用模型规划的布局。\n"
                            "Keep the boxes you drew. The official docs warn "
                            "that the prompt upsampler rewrites the layout, so "
                            "this keeps your pixels and takes only the "
                            "descriptions from the model. Turn it off to let "
                            "the model plan the layout entirely."
                        ),
                    },
                ),
                "skip_enhance": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "跳过扩写，直接用你的提示词（布局模式则直接把画好的"
                            "框编译成 Layout）。官方文档把\"关掉扩写以自己控制"
                            "措辞\"列为一项提示词技巧。\n"
                            "Skip the expansion and use your own wording. The "
                            "official guide lists turning the upsampler off as "
                            "a prompting technique."
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
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": _MAX_TOKENS_DEFAULT,
                        "min": _MIN_MAX_TOKENS,
                        "max": _MAX_MAX_TOKENS,
                    },
                ),
                "timeout": ([60, 120, 300, 600], {"default": _DEFAULT_TIMEOUT}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("upsampled_prompt", "caption", "layout_json")
    FUNCTION = "enhance"
    CATEGORY = MY_CATEGORY

    def enhance(
        self,
        llm_service_connector,
        mode,
        prompt,
        canvas_width=1024,
        canvas_height=1024,
        seed=None,
        ref_image=None,
        elements=None,
        character_boxes="",
        keep_layout_verbatim=True,
        skip_enhance=False,
        temperature=_DEFAULT_TEMPERATURE,
        max_tokens=_MAX_TOKENS_DEFAULT,
        timeout=_DEFAULT_TIMEOUT,
    ):
        mode_code = parse_mode_label(mode) or parse_mode_label(MODES[0])

        if mode_code == LAYOUT_MODE:
            width, height = _validate_canvas(canvas_width, canvas_height)
            boxes = (
                elements_to_boxes(elements, width, height, character_boxes)
                if elements is not None
                else []
            )
        else:
            width, height = 0, 0
            boxes = []

        if skip_enhance:
            return self._skip(mode_code, prompt, width, height, boxes)

        enhancer = MossImagePromptEnhancer(
            llm_service_connector,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        full, caption, layout_json = enhancer(
            mode=mode_code,
            prompt=prompt,
            width=width,
            height=height,
            boxes=boxes,
            reference_images=ref_image,
            keep_layout_verbatim=keep_layout_verbatim,
            seed=seed,
        )
        if mode_code == LAYOUT_MODE:
            return (full, caption, layout_json)
        return (full, full, "")

    @staticmethod
    def _skip(mode_code, prompt, width, height, boxes):
        """Answer without calling the LLM.

        The caption mode hands the request straight to the image model, which
        is the official "turn the upsampler off" technique. The layout mode
        compiles the boxes the user drew and uses the request as the global
        description, so the layout still reaches the model as exact geometry.
        """
        request = (prompt or "").strip()
        if not request:
            raise ValueError(
                "MossImage: prompt is empty; there is nothing to enhance / "
                "prompt 为空，无内容可增强"
            )
        if mode_code != LAYOUT_MODE:
            return (request, request, "")
        if not boxes:
            raise ValueError(
                "MossImage: skip_enhance in layout mode needs the boxes drawn "
                "on the canvas, otherwise there is no layout to compile / "
                "布局模式下跳过扩写需要画布上的框，否则没有布局可编译"
            )
        draft = {
            "width": width,
            "height": height,
            "bbox_coord_space": "abs",
            "bbox_format": "xyxy",
            "summary": {"caption": request},
            "items": [
                {
                    key: value
                    for key, value in box.items()
                    if key in ("label", "caption", "bbox", "text")
                }
                for box in boxes
            ],
        }
        compiled = compile_layout_draft(draft)
        readable = json.dumps(draft, ensure_ascii=False, indent=2)
        return (compiled, request, readable)

    def is_changed(
        self,
        llm_service_connector,
        mode,
        prompt,
        canvas_width=1024,
        canvas_height=1024,
        seed=None,
        ref_image=None,
        elements=None,
        character_boxes="",
        keep_layout_verbatim=True,
        skip_enhance=False,
        temperature=_DEFAULT_TEMPERATURE,
        max_tokens=_MAX_TOKENS_DEFAULT,
        timeout=_DEFAULT_TIMEOUT,
    ):
        h = hashlib.md5(usedforsecurity=False)
        mode_code = parse_mode_label(mode) or parse_mode_label(MODES[0])
        for part in (
            prompt,
            mode_code,
            str(canvas_width),
            str(canvas_height),
            str(seed),
            str(temperature),
            str(timeout),
            str(max_tokens),
            str(keep_layout_verbatim),
            str(skip_enhance),
            character_boxes,
        ):
            h.update((part or "").encode("utf-8"))
        # A canvas edit changes the boxes without touching any widget, so the
        # resolved boxes have to participate in the hash.
        if elements:
            try:
                h.update(
                    json.dumps(
                        elements_to_boxes(elements, canvas_width, canvas_height),
                        ensure_ascii=False,
                        sort_keys=True,
                    ).encode("utf-8")
                )
            except (ValueError, TypeError):
                h.update(repr(elements).encode("utf-8"))
        h.update(b"|")
        h.update(_image_sample_bytes(ref_image))

        # Hash the system prompt actually loaded, so editing a bundled prompt
        # invalidates cached results instead of silently reusing them.
        try:
            h.update(load_system_prompt(mode_code).encode("utf-8"))
        except (ValueError, KeyError, FileNotFoundError):
            pass

        try:
            h.update(llm_service_connector.get_state().encode("utf-8"))
        except AttributeError:
            h.update(str(getattr(llm_service_connector, "api_url", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "api_token", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "model", "")).encode("utf-8"))
        return h.hexdigest()


def _validate_canvas(width, height) -> tuple[int, int]:
    """Check the canvas is a positive multiple of 16 on both sides.

    MOSS-Image requires multiples of 16, and the compiled boxes are centred
    on the canvas, so a wrong canvas silently misplaces every element.
    """
    values = []
    for name, value in (("canvas_width", width), ("canvas_height", height)):
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"MossImage: {name} must be an integer, got {value!r} / "
                f"{name} 必须是整数"
            ) from exc
        if number <= 0 or number % 16 != 0:
            raise ValueError(
                f"MossImage: {name} must be a positive multiple of 16, got "
                f"{number} / {name} 必须是 16 的正整数倍，当前为 {number}"
            )
        values.append(number)
    return values[0], values[1]


def _image_sample_bytes(reference_images) -> bytes:
    """A cheap, stable fingerprint of an IMAGE batch for ``is_changed``.

    Hashing full pixels would be wasteful, and hashing only the frame count
    would miss a swapped image of the same count. The shape always
    participates; on top of it, three 1024-element windows (head / middle /
    tail) are sampled to catch the common edits. The tensor calls are
    duck-typed and failure-tolerant, so an object that is not a torch tensor
    still contributes its shape rather than nothing at all.
    """
    if reference_images is None:
        return b"none"
    out = [str(getattr(reference_images, "shape", "")).encode("utf-8")]
    try:
        flat = reference_images.detach().cpu().reshape(-1).numpy().tobytes()
    except Exception:  # noqa: BLE001 - any tensor-like failure just weakens the hash
        return b"|".join(out)
    step = max(1, len(flat) // 1024)
    for start in (0, len(flat) // 2, max(0, len(flat) - 1024)):
        chunk = flat[start : start + 1024 * step]
        if chunk:
            out.append(hashlib.md5(chunk, usedforsecurity=False).digest())
    return b"|".join(out)
