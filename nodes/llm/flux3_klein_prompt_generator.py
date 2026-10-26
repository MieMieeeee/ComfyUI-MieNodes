"""FLUX.3 Image Klein prompt-upsampler ComfyUI node.

Reproduces Black Forest Labs' official Prompt Upsampling (PUP) behaviour by
sending the bundled official system prompt for the selected mode to the
configured ``LLMServiceConnector``. Upstream is emphatic that Klein is trained
on the captions its upsampler writes -- BFL's docs say "Do not use Klein
without the prompt upsampler" -- so the short request a user types and the
prompt the image model finally sees are deliberately not the same string.

Four modes, straight from the upstream prompt packs: ``t2i`` (no image, no
layout), ``i2i`` (edit against reference images), and the ``*_bbox`` pair,
whose contract is one line holding the global caption paragraph followed by a
JSON element table on a 0-1000 grid.

Two official prompt packs ship side by side (see
``nodes/llm/prompts/flux3_klein/UPSTREAM.md``): the short pack released with the
PUP weights, which upstream nominates for "any other LLM", and the far longer
external-VLM pack from inference repo PR #5. They are separate official
publications, not one text in two spellings, so both are exposed for
side-by-side comparison.

Sits beside ``QwenImage21PromptGenerator`` and ``Krea2PromptGenerator`` in the
Prompt Generator category and keeps the family conventions: bilingual dropdown
labels, per-call timeout override, ``is_changed`` hashing.
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
    from _mienodes_internal.nodes.llm.flux3_klein_prompts import (
        BBOX_MODES,
        MAX_REFERENCE_IMAGES,
        MODES,
        PROMPT_PACKS,
        VISION_MODES,
        build_user_message,
        check_box_provenance,
        elements_to_table,
        is_edit_shaped,
        load_system_prompt,
        parse_mode_label,
        parse_pack_label,
        split_bbox_output,
        supplied_boxes,
    )
except ImportError:
    from .flux3_klein_prompts import (
        BBOX_MODES,
        MAX_REFERENCE_IMAGES,
        MODES,
        PROMPT_PACKS,
        VISION_MODES,
        build_user_message,
        check_box_provenance,
        elements_to_table,
        is_edit_shaped,
        load_system_prompt,
        parse_mode_label,
        parse_pack_label,
        split_bbox_output,
        supplied_boxes,
    )


MY_CATEGORY = "\U0001F411 MieNodes/\U0001F411 Prompt Generator"

# Official PUP spends ``max_new_tokens=3584``. Reasoning models (M3, DeepSeek-R1,
# GLM-5.x) emit a chain-of-thought block inside the same budget before the
# caption, and a 200-with-empty-content reply is the usual symptom of too small
# a budget -- so the default sits well above upstream's figure, matching the
# rest of this project's LLM nodes.
_MAX_TOKENS_DEFAULT = 16384
_MIN_MAX_TOKENS = 64
_MAX_MAX_TOKENS = 32768

# Official PUP runs ``do_sample=False``. External APIs are told to stay in the
# 0-0.3 band rather than at a true zero, which some backends reject.
_DEFAULT_TEMPERATURE = 0.2

# One upsampling call, no JSON round-trips. 300 s covers a long caption from a
# reasoning model on a busy backend; the long prompt pack alone is ~12k tokens
# of system prompt.
_DEFAULT_TIMEOUT = 300

# How many samples to take when a bbox-mode reply has an unparseable or altered
# supplied table. Verified live against MiniMax-M3.1: it sometimes writes rows as
# loose ``id [y x y x] desc`` prose instead of JSON. One resample usually lands
# the official shape, and failing the whole run over it would discard an
# otherwise usable caption.
_TABLE_PARSE_ATTEMPTS = 2


# --------------------------------------------------------------------------- #
# Enhancer
# --------------------------------------------------------------------------- #
class Flux3KleinPromptUpsamplerEnhancer:
    """Run one official-PUP upsampling call and split its reply."""

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
                # The upsampled caption is the answer; a provider that
                # returns only its reasoning channel would hand back text
                # with no caption and no element table. Refusing that
                # fallback turns the failure into the empty-reply error
                # below, which names the real cause.
                allow_reasoning_fallback=False,
            )
            elapsed = time.perf_counter() - t0
            model_name = getattr(self.llm, "model", "?")
            if not out or not out.strip():
                # The connector returns "" rather than raising on an empty
                # 200. Reasoning models usually hit this by spending the
                # whole budget thinking, so say so instead of returning "".
                raise RuntimeError(
                    f"Flux3Klein: LLM returned an empty reply after "
                    f"{elapsed:.2f}s (model={model_name}). Reasoning models "
                    f"often exhaust the budget before writing the caption -- "
                    f"raise max_tokens / 模型返回空回复，推理模型可能已用尽 "
                    f"token 预算，请调高 max_tokens"
                )
            mie_log(
                f"Flux3Klein: model={model_name} ok in {elapsed:.2f}s "
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
        prompt_pack: str,
        prompt: str,
        canvas_preview=None,
        elements=None,
        reference_images=None,
        seed=None,
    ) -> tuple[str, str, str]:
        """Upsample ``prompt`` and return ``(upsampled_prompt, caption, bbox_json)``.

        ``upsampled_prompt`` is the reply verbatim -- that is the string the
        image model is trained to receive. ``caption`` and ``bbox_json`` are
        the same reply deterministically split for the bbox modes (caption
        paragraph / JSON element table), so they can drive separate conditioning
        and bounding-box nodes. Non-bbox modes return ``("", "")`` for those.

        The element table comes from ``elements`` -- straight from
        ``CreateBoundingBoxes``, which already emits boxes in the official
        coordinate convention.
        """
        system_prompt = load_system_prompt(mode, prompt_pack)
        rows = _resolve_table(elements, mode)
        expected_table = rows or None
        user_text = build_user_message(
            mode,
            prompt,
            target_canvas=_canvas_ratio(canvas_preview),
            rows=rows,
        )
        if mode in BBOX_MODES and expected_table:
            if is_edit_shaped(expected_table):
                user_text += (
                    "\nThe supplied element table is authoritative. Copy every "
                    "row exactly, including ids, coordinates, source references "
                    "and descs. Do not add, remove or rename rows. Put requested "
                    "appearance changes in the instruction paragraph, not in "
                    "source descriptions."
                )
            else:
                # Generation-shaped rows: the user marked regions and the
                # system prompt is the annotation system that turns them into
                # an edit table. State what must hold without forbidding the
                # transformation the system prompt asks for.
                user_text += (
                    "\nThe supplied rows mark the regions the user selected. "
                    "Every marked region must be represented by the source or "
                    "target box of the row that handles it. Tighten a marked "
                    "box onto the object it actually marks when the marked "
                    "region is larger than the object, and describe what is "
                    "really there. You may add rows for the rest of the "
                    "scene. Do not place an object in a region the user did "
                    "not mark."
                )
        messages = [{"role": "system", "content": system_prompt}]

        urls = self._reference_urls(mode, reference_images)
        if urls:
            # Upstream i2i attaches the references first, then the request
            # text; ``build_multimodal_user_content`` orders it that way.
            messages.append(
                {
                    "role": "user",
                    "content": build_multimodal_user_content(user_text, urls),
                }
            )
        else:
            messages.append({"role": "user", "content": user_text})

        raw = self._invoke(messages, seed=seed)
        if mode not in BBOX_MODES:
            caption, table_json, _ = split_bbox_output(raw, mode=mode)
            return raw.strip(), caption, table_json

        last_error: Exception | None = None
        for attempt in range(1, _TABLE_PARSE_ATTEMPTS + 1):
            try:
                caption, table_json, returned_table = split_bbox_output(raw, mode=mode)
                self._check_faithfulness(expected_table, returned_table)
            except ValueError as exc:
                last_error = exc
                mie_log(
                    f"Flux3Klein: mode={mode} no parseable or faithful "
                    f"element table: {exc} (attempt {attempt}/"
                    f"{_TABLE_PARSE_ATTEMPTS}); raw_tail={raw[-300:]!r}"
                )
                if attempt < _TABLE_PARSE_ATTEMPTS:
                    correction = {
                        "role": "user",
                        "content": (
                            f"Your previous reply failed validation: {exc}. "
                            "Return the complete instruction paragraph and ONE "
                            "JSON array. Preserve the supplied table exactly."
                        ),
                    }
                    raw = self._invoke(
                        messages + [{"role": "assistant", "content": raw}, correction],
                        seed=seed,
                    )
                continue
            return raw.strip(), caption, table_json
        raise last_error

    @staticmethod
    def _check_faithfulness(expected_table, returned_table) -> None:
        """Reject a reply whose table drifted away from what the user supplied.

        The rule depends on the shape of the table the user gave us, because
        the two shapes carry different promises:

        * **Edit shape** (``from`` / ``src_bbox`` / ``tgt_bbox``) -- the user
          wrote the final table by hand. The model must return it verbatim:
          same ids, same coordinates, same descriptions. A model that
          rewrites a desc to describe the *end* state (the classic
          "white cup" -> "cobalt blue cup" slip) changes what the row means.

        * **Generation shape** (``bbox`` only, which is what
          ``CreateBoundingBoxes`` emits) -- the user marked regions and the
          official system prompt is the annotation system that turns those
          into an edit table. The row shape *must* change, so equality is
          meaningless. What must hold is that the coordinates stay honest:
          every box the model emits has to be one the user drew. A box it
          invents lands the object somewhere nobody asked for, which is the
          failure worth catching.
        """
        if not expected_table:
            return
        if is_edit_shaped(expected_table):
            if returned_table != expected_table:
                raise ValueError(
                    "Flux3Klein: LLM changed the supplied element table; "
                    "refusing altered ids, coordinates or descriptions / "
                    "模型改写了指定元素表，拒绝使用错误的框"
                )
            return
        supplied = supplied_boxes(expected_table)
        missing = check_box_provenance(returned_table, supplied)
        if missing:
            raise ValueError(
                f"Flux3Klein: the reply does not account for the marked "
                f"regions {missing}; refusing a table that drops the user's "
                f"boxes / 模型输出没有覆盖用户指定的框，拒绝使用丢失区域的表"
            )

    @staticmethod
    def _reference_urls(mode: str, reference_images) -> list[str]:
        """Validate the reference images for ``mode`` and encode them.

        The editing modes are VLM-only by design: their official system prompt
        opens with "You see the reference image(s), attached in order", so
        there is no caption-then-rewrite fallback -- a text-only reply would be
        the model inventing what it supposedly saw.
        """
        urls = image_tensor_batch_to_data_urls(reference_images) if reference_images is not None else []

        if mode in VISION_MODES and not urls:
            raise ValueError(
                f"Flux3Klein: mode {mode!r} upsample reference image(s) and "
                f"needs a vision-capable connector with the images connected "
                f"to reference_images / {mode} 模式需要参考图：请把图片接到 "
                f"reference_images，并使用支持视觉的 LLM 连接器"
            )
        if mode not in VISION_MODES and urls:
            mie_log(
                f"Flux3Klein: mode {mode!r} takes no reference images, but "
                f"{len(urls)} arrived on reference_images -- ignoring them / "
                f"{mode} 模式不使用参考图，已忽略"
            )
            return []
        if len(urls) > MAX_REFERENCE_IMAGES:
            mie_log(
                f"Flux3Klein: {len(urls)} reference images exceed the "
                f"upstream ceiling of {MAX_REFERENCE_IMAGES}; using the first "
                f"{MAX_REFERENCE_IMAGES} / 参考图超过官方上限，已截断"
            )
            urls = urls[:MAX_REFERENCE_IMAGES]
        return urls


def _canvas_ratio(canvas_preview) -> str:
    """Read the canvas aspect ratio from the canvas preview image.

    ``CreateBoundingBoxes`` renders its preview at exactly the canvas size, so
    the preview's dimensions ARE the canvas -- no separate ratio field to keep
    in sync with it.

    Returns a bare ``W:H`` string, or ``""`` when there is no preview.
    """
    if canvas_preview is None:
        return ""
    shape = getattr(canvas_preview, "shape", None)
    if not shape or len(shape) < 3:
        return ""
    height, width = int(shape[1]), int(shape[2])
    if height <= 0 or width <= 0:
        return ""
    return f"{width}:{height}"


def _resolve_table(elements, mode: str = "") -> list:
    """Turn the canvas rows into the element table the model receives.

    The canvas is the only source of a user-supplied table: hand-writing the
    JSON was always a footgun (y/x order, 0-1000 grid vs pixels) and the
    canvas already emits boxes in the official convention.

    An absent socket arrives as ``None`` and a connected-but-empty one as
    ``[]``; those are different states and must not collapse. Treating an
    empty array as "no table" would let the model quietly plan its own layout
    while the user believes their canvas boxes are being honoured.

    Canvas rows carry one ``bbox`` each, while the official editing table
    wants a source and a target box per row. That is not a reason to refuse:
    the official ``i2i_bbox`` system prompt is the annotation system that
    turns "a marked region plus an instruction" into src/tgt rows, and the
    native ComfyUI flow does exactly this -- ``Flux3LayoutPrompt`` hands the
    generation-shaped rows to the model and the model produces the
    edit-shaped table. The rows are transported as the user drew them and
    the model converts them, which is the upstream contract.

    A *move* is expressed the way the community does it: draw two boxes and
    label their descriptions "src" and "tgt", then name those labels in the
    instruction ("Move the bird from src to tgt"). Verified end to end.
    """
    if elements is None:
        return []
    if not elements:
        raise ValueError(
            "Flux3Klein: elements is connected but empty, so the boxes drawn "
            "on the canvas would be silently ignored. Draw at least one box in "
            "Create Bounding Boxes, or disconnect it. / "
            "elements 已连接但为空：画布上的框会被静默忽略。"
            "请至少画一个框，或断开该输入"
        )
    return elements_to_table(elements)


# --------------------------------------------------------------------------- #
# ComfyUI node
# --------------------------------------------------------------------------- #
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
