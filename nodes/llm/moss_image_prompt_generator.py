"""MOSS-Image prompt-enhancement ComfyUI node.

MOSS-Image is a 5B single-stream DiT trained on full scene captions, and a
one-word request fed straight into it produces a visibly weaker image than
the 200-500 word English caption it was trained on. The official workflow
therefore ships an expander: the model's own text encoder (Qwen3.5-9B) is
run as an LLM over the request before the image model sees it.

This node reproduces that behaviour over HTTP instead of in-process. Upstream
MOSS has **no** image-to-image mode, so there are only two modes:

* ``t2i`` -- the request is expanded into an English caption, using the
  official upsampler system prompt verbatim.
* ``t2i_layout`` -- the request is expanded into a spatial Layout: a JSON
  draft with absolute-pixel boxes, compiled by the node into the one-line
  compact JSON the image model consumes (``abs_centralized`` / ``yxyx``).

Unlike Klein, MOSS needs no separate upsampler weights, so the node occupies
no VRAM, works with any OpenAI-compatible endpoint, and accepts Chinese
input.

Two things make the layout mode different from every other Layout node:

* The three coordinate spaces are kept apart. ``CreateBoundingBoxes`` emits a
  0-1000 ``[y_min, x_min, y_max, x_max]`` grid; the model reads absolute
  pixels in ``xyxy`` from the top-left; the model finally consumes centered
  ``yxyx``. The node converts between them, because the official prompt says
  to use the given coordinates as they are and never convert them.
* The official docs warn that the prompt upsampler rewrites the layout. So
  the user's drawn boxes are authoritative by default: the node keeps the
  user's pixels and takes only the descriptions from the model.

Sits beside ``Flux3KleinPromptUpsampler`` in the Prompt Generator category
and keeps the family conventions: bilingual dropdown labels, per-call timeout
override, ``is_changed`` hashing.
"""
from __future__ import annotations

import hashlib
import json
import time

try:
    from _mienodes_internal.core.utils import (
        build_multimodal_user_content,
        image_tensor_batch_to_data_urls,
        mie_log,
    )
except ImportError:
    try:
        from ...core.utils import (
            build_multimodal_user_content,
            image_tensor_batch_to_data_urls,
            mie_log,
        )
    except ImportError:

        def mie_log(msg):  # noqa: ARG001
            pass

        def build_multimodal_user_content(text, image_urls=None, image_detail="auto"):
            return [{"type": "text", "text": text or ""}]

        def image_tensor_batch_to_data_urls(image, fmt=".jpg"):
            return []


try:
    from _mienodes_internal.nodes.llm.moss_image_prompts import (
        CAPTION_MODE,
        LAYOUT_MODE,
        MODES,
        build_user_message,
        compile_layout_draft,
        elements_to_boxes,
        enforce_supplied_boxes,
        load_system_prompt,
        parse_layout_reply,
        parse_mode_label,
    )
except ImportError:
    from .moss_image_prompts import (
        CAPTION_MODE,
        LAYOUT_MODE,
        MODES,
        build_user_message,
        compile_layout_draft,
        elements_to_boxes,
        enforce_supplied_boxes,
        load_system_prompt,
        parse_layout_reply,
        parse_mode_label,
    )


MY_CATEGORY = "\U0001F411 MieNodes/\U0001F411 Prompt Generator"

# Official MOSS caps the prompt at roughly 4K tokens, so the *model* never
# needs a bigger budget than that. The node's budget is a different thing:
# reasoning models (M3, DeepSeek-R1, GLM-5.x) spend part of it thinking
# before the caption, and a reply that runs out of budget mid-caption is
# the common failure. The default therefore sits well above upstream's
# figure, matching the rest of this project's LLM nodes.
_MAX_TOKENS_DEFAULT = 16384
_MIN_MAX_TOKENS = 64
_MAX_MAX_TOKENS = 32768

# Official decoding parameters (MOSS template): temperature 0.7, top_p 0.95,
# top_k 64, min_p 0.05, repetition_penalty 1.0, presence_penalty 0.
# Only temperature is exposed -- no connector in this project forwards the
# rest, and the bundled service layer deliberately does not carry them.
_DEFAULT_TEMPERATURE = 0.7

# One call, no JSON round-trips. 300 s covers a long caption from a
# reasoning model on a busy backend.
_DEFAULT_TIMEOUT = 300

# The official template feeds a single ``ref_image`` to the upsampler. The
# node accepts an IMAGE batch, so extra frames are capped rather than
# silently dropped.
_MAX_REFERENCE_IMAGES = 4

# How many samples to take when the layout reply carries no parseable JSON
# draft. Verified live: models habitually fence the object or add a sentence
# of prose around it, and one resample usually lands the official shape.
_LAYOUT_PARSE_ATTEMPTS = 2


# --------------------------------------------------------------------------- #
# Enhancer
# --------------------------------------------------------------------------- #
class MossImagePromptEnhancer:
    """Run one MOSS enhancement call and turn the reply into a prompt."""

    def __init__(
        self,
        llm_service_connector,
        *,
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: int = _MAX_TOKENS_DEFAULT,
        timeout: int = _DEFAULT_TIMEOUT,
    ):
        self.llm = llm_service_connector
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        # Per-call timeout override, restored after the call. ``None`` would
        # mean "leave the connector's own timeout alone".
        self._timeout_override = int(timeout) if timeout else None

    def _invoke(self, messages: list[dict], seed=None) -> str:
        prev_timeout = getattr(self.llm, "timeout", None)
        try:
            if self._timeout_override is not None and prev_timeout is not None:
                self.llm.timeout = self._timeout_override
            t0 = time.perf_counter()
            out = self.llm.invoke(
                messages,
                seed=seed,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                # The caption is the answer; a provider that returns only its
                # reasoning channel would hand back text with no caption and
                # no layout. Refusing that fallback turns the failure into the
                # empty-reply error below, which names the real cause.
                allow_reasoning_fallback=False,
            )
            elapsed = time.perf_counter() - t0
            model_name = getattr(self.llm, "model", "?")
            if not out or not out.strip():
                # The connector returns "" rather than raising on an empty
                # 200. Reasoning models usually hit this by spending the
                # whole budget thinking, so say so instead of returning "".
                raise RuntimeError(
                    f"MossImage: LLM returned an empty reply after "
                    f"{elapsed:.2f}s (model={model_name}). Reasoning models "
                    f"often exhaust the budget before writing the caption -- "
                    f"raise max_tokens / 模型返回空回复，推理模型可能已用尽 "
                    f"token 预算，请调高 max_tokens"
                )
            mie_log(
                f"MossImage: model={model_name} ok in {elapsed:.2f}s "
                f"response_chars={len(out)}"
            )
            return out
        finally:
            if prev_timeout is not None:
                self.llm.timeout = prev_timeout

    def __call__(
        self,
        *,
        mode: str,
        prompt: str,
        width: int,
        height: int,
        boxes: list[dict] | None = None,
        reference_images=None,
        keep_layout_verbatim: bool = True,
        seed=None,
    ) -> tuple[str, str, str]:
        """Enhance ``prompt`` and return ``(upsampled_prompt, caption, layout_json)``.

        In the caption mode the reply is the caption and the layout fields
        are empty. In the layout mode the reply is a JSON draft that the
        node compiles into the one-line compact JSON the image model reads;
        ``layout_json`` is the readable draft, kept so the layout can be
        edited and re-compiled.
        """
        system_prompt = load_system_prompt(mode)
        boxes = boxes or []

        if mode == CAPTION_MODE:
            user_text = build_user_message(mode, prompt)
            messages = self._messages(system_prompt, user_text, reference_images)
            raw = self._invoke(messages, seed=seed)
            caption = raw.strip()
            return caption, caption, ""

        user_text = build_user_message(
            mode, prompt, width=width, height=height, boxes=boxes
        )
        messages = self._messages(system_prompt, user_text, reference_images)

        last_error: Exception | None = None
        for attempt in range(1, _LAYOUT_PARSE_ATTEMPTS + 1):
            raw = self._invoke(messages, seed=seed)
            try:
                draft = parse_layout_reply(raw)

                # The canvas belongs to the image workflow, not to the model.
                # If the model picked a different one, its boxes would
                # describe a different picture, so the requested canvas wins.
                if draft.get("width") != width or draft.get("height") != height:
                    mie_log(
                        f"MossImage: model returned canvas "
                        f"{draft.get('width')}x{draft.get('height')}, using the "
                        f"requested {width}x{height}"
                    )
                    draft["width"] = width
                    draft["height"] = height

                if boxes and keep_layout_verbatim:
                    # The official docs warn that the upsampler rewrites the
                    # layout. The user's pixels win; only the descriptions
                    # come from the model.
                    draft = enforce_supplied_boxes(draft, boxes)

                compiled = compile_layout_draft(draft)
            except ValueError as exc:
                last_error = exc
                mie_log(
                    f"MossImage: layout mode got no usable JSON draft: {exc} "
                    f"(attempt {attempt}/{_LAYOUT_PARSE_ATTEMPTS}); "
                    f"raw_tail={raw[-300:]!r}"
                )
                if attempt < _LAYOUT_PARSE_ATTEMPTS:
                    messages = messages + [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                "Your previous reply could not be used as a "
                                "JSON Layout draft: "
                                f"{exc}\n"
                                "Reply with ONE JSON object and nothing else: "
                                "no prose, no code fence. Keep every bbox "
                                "inside the canvas."
                            ),
                        },
                    ]
                continue

            summary = draft.get("summary")
            caption = (
                str(summary.get("caption") or "")
                if isinstance(summary, dict)
                else ""
            )
            readable = json.dumps(draft, ensure_ascii=False, indent=2)
            return compiled, caption, readable

        raise last_error

    @staticmethod
    def _messages(system_prompt: str, user_text: str, reference_images) -> list[dict]:
        """Build the chat messages, attaching a reference image when given.

        The reference image is only ever shown to the LLM -- MOSS-Image does
        not take a reference image -- so it rides in the user turn ahead of
        the text, the way the official template feeds ``ref_image``.
        """
        urls = (
            image_tensor_batch_to_data_urls(reference_images)
            if reference_images is not None
            else []
        )
        if len(urls) > _MAX_REFERENCE_IMAGES:
            mie_log(
                f"MossImage: {len(urls)} reference images exceed the official "
                f"single-image reference; using the first "
                f"{_MAX_REFERENCE_IMAGES} / 参考图超过官方上限，已截断"
            )
            urls = urls[:_MAX_REFERENCE_IMAGES]
        messages = [{"role": "system", "content": system_prompt}]
        if urls:
            messages.append(
                {
                    "role": "user",
                    "content": build_multimodal_user_content(user_text, urls),
                }
            )
        else:
            messages.append({"role": "user", "content": user_text})
        return messages


# --------------------------------------------------------------------------- #
# ComfyUI node
# --------------------------------------------------------------------------- #
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
