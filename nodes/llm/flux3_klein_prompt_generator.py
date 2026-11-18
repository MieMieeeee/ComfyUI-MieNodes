"""ComfyUI compatibility shell; generation implementation lives in mie_prompt."""
from __future__ import annotations
from mie_prompt import flux3_klein_prompt_generator as _impl
for _name in dir(_impl):
    if not _name.startswith('__'):
        globals()[_name] = getattr(_impl, _name)

class Flux3KleinPromptUpsampler:
    """ComfyUI node: short request -> official FLUX.3 Klein upsampled prompt.

    Sends the bundled official PUP system prompt for the selected mode and
    prompt pack to the configured LLM, and returns the caption the image model
    expects. For the ``*_bbox`` modes the reply is also split into its caption
    paragraph and its JSON element table so each can drive its own node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "mode": (list(MODES), {"default": MODES[0]}),
                "prompt_pack": (list(PROMPT_PACKS), {"default": PROMPT_PACKS[0]}),
                "prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Short request to upsample. Any language is fine — "
                            "the caption is always written in English, and only "
                            "text the model is asked to render (inside double "
                            "quotes) keeps its original language. Required, "
                            "except in the *_bbox modes when an element table "
                            "is supplied — then the table's own descriptions "
                            "serve as the request. "
                            "短提示词，可中文；输出恒为英文，仅引号内待渲染文字保留原文。"
                            "*_bbox 模式下若已填元素表，则可留空。"
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
                "reference_images": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Reference images, attached in order (up to 10). "
                            "Required by i2i / i2i_bbox — those prompts open "
                            "with \"You see the reference image(s)\", so they "
                            "run on a vision-capable connector. "
                            "按序附上的参考图（最多 10 张）。i2i / i2i_bbox 必填，"
                            "需使用支持视觉的连接器。"
                        )
                    },
                ),
                "elements": (
                    "ARRAY",
                    {
                        "tooltip": (
                            "Element table from Create Bounding Boxes' "
                            "elements output. Boxes arrive already normalized "
                            "to the official [y_min, x_min, y_max, x_max] on "
                            "a 0-1000 grid, so no JSON is needed. *_bbox "
                            "modes only. To MOVE something, draw two boxes "
                            "and label their descriptions 'src' and 'tgt', "
                            "then name those labels in the instruction "
                            "('Move the bird from src to tgt'). "
                            "直接接 Create Bounding Boxes 的 elements 输出："
                            "坐标已是官方 0-1000 网格的 [y_min, x_min, y_max, "
                            "x_max]，无需手写 JSON。要移动物体，就画两个框，"
                            "desc 分别写 src 和 tgt，指令里引用这两个词。"
                        )
                    },
                ),
                "canvas_preview": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Create Bounding Boxes' preview output. Its "
                            "dimensions ARE the canvas, so the aspect ratio is "
                            "taken from here instead of being typed — the "
                            "canvas and the prompt can never disagree. The "
                            "official box grid normalizes each axis "
                            "independently, so the frame's proportions cannot "
                            "be recovered from the boxes alone; without this, "
                            "objects are sized as if the frame were square. "
                            "Only the t2i modes use it (the editing modes take "
                            "the frame from the reference image). "
                            "接 Create Bounding Boxes 的 preview 输出："
                            "画布比例自动取自这里，无需手填——官方网格两轴各自"
                            "归一化，只看框无法知道画幅宽高比。"
                        )
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
    RETURN_NAMES = ("upsampled_prompt", "caption", "bbox_json")
    FUNCTION = "upsample"
    CATEGORY = MY_CATEGORY

    def upsample(
        self,
        llm_service_connector,
        mode,
        prompt_pack,
        prompt,
        seed=None,
        reference_images=None,
        elements=None,
        canvas_preview=None,
        temperature=_DEFAULT_TEMPERATURE,
        max_tokens=_MAX_TOKENS_DEFAULT,
        timeout=_DEFAULT_TIMEOUT,
    ):
        mode_code = parse_mode_label(mode) or parse_mode_label(MODES[0])
        pack_code = parse_pack_label(prompt_pack) or parse_pack_label(PROMPT_PACKS[0])

        enhancer = Flux3KleinPromptUpsamplerEnhancer(
            llm_service_connector,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        full, caption, table_json = enhancer(
            mode=mode_code,
            prompt_pack=pack_code,
            prompt=prompt,
            canvas_preview=canvas_preview,
            elements=elements if mode_code in BBOX_MODES else None,
            reference_images=reference_images,
            seed=seed,
        )
        return (full, caption, table_json)

    def is_changed(
        self,
        llm_service_connector,
        mode,
        prompt_pack,
        prompt,
        seed=None,
        reference_images=None,
        elements=None,
        canvas_preview=None,
        temperature=_DEFAULT_TEMPERATURE,
        max_tokens=_MAX_TOKENS_DEFAULT,
        timeout=_DEFAULT_TIMEOUT,
    ):
        h = hashlib.md5(usedforsecurity=False)
        mode_code = parse_mode_label(mode) or parse_mode_label(MODES[0])
        pack_code = parse_pack_label(prompt_pack) or parse_pack_label(PROMPT_PACKS[0])
        for part in (
            prompt,
            mode_code,
            pack_code,
            _canvas_ratio(canvas_preview),
            str(seed),
            str(temperature),
            str(timeout),
            str(max_tokens),
        ):
            h.update((part or "").encode("utf-8"))
        # A canvas edit changes the boxes without touching any widget, so the
        # resolved table has to participate in the hash.
        if elements:
            try:
                h.update(
                    json.dumps(
                        elements_to_table(elements), ensure_ascii=False, sort_keys=True
                    ).encode("utf-8")
                )
            except (ValueError, TypeError):
                h.update(repr(elements).encode("utf-8"))
        h.update(b"|")
        h.update(_image_sample_bytes(reference_images))

        # Hash the system prompt actually loaded, so editing a bundled official
        # prompt invalidates cached results instead of silently reusing them.
        try:
            h.update(load_system_prompt(mode_code, pack_code).encode("utf-8"))
        except (ValueError, KeyError, FileNotFoundError):
            pass

        try:
            h.update(llm_service_connector.get_state().encode("utf-8"))
        except AttributeError:
            h.update(str(getattr(llm_service_connector, "api_url", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "api_token", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "model", "")).encode("utf-8"))
        return h.hexdigest()


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
