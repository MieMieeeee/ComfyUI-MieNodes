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
        load_system_prompt,
        parse_mode_label,
        parse_pack_label,
        split_bbox_output,
    )
except ImportError:
    from .flux3_klein_prompts import (
        BBOX_MODES,
        MAX_REFERENCE_IMAGES,
        MODES,
        PROMPT_PACKS,
        VISION_MODES,
        build_user_message,
        load_system_prompt,
        parse_mode_label,
        parse_pack_label,
        split_bbox_output,
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
        target_canvas: str = "",
        bbox_json: str = "",
        reference_images=None,
        seed=None,
    ) -> tuple[str, str, str]:
        """Upsample ``prompt`` and return ``(upsampled_prompt, caption, bbox_json)``.

        ``upsampled_prompt`` is the reply verbatim -- that is the string the
        image model is trained to receive. ``caption`` and ``bbox_json`` are
        the same reply deterministically split for the bbox modes (caption
        paragraph / JSON element table), so they can drive separate conditioning
        and bounding-box nodes. Non-bbox modes return ``("", "")`` for those.
        """
        system_prompt = load_system_prompt(mode, prompt_pack)
        user_text = build_user_message(
            mode, prompt, target_canvas=target_canvas, bbox_json=bbox_json
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
        caption, table_json, _table = split_bbox_output(raw, mode=mode)
        return raw.strip(), caption, table_json

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
                            "quotes) keeps its original language. "
                            "短提示词，可中文；输出恒为英文，仅引号内待渲染文字保留原文。"
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
                "target_canvas": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": (
                            "Optional canvas ratio (16:9, 1:1, or auto). It only "
                            "steers composition — the model is instructed never "
                            "to mention it. Leave empty to compose freely. "
                            "可选画布比例，仅影响构图，输出中不会出现该比例。"
                        ),
                    },
                ),
                "bboxes_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Optional element table as a JSON array; the model "
                            "keeps it exactly. Only for the *_bbox modes — "
                            "leave empty to let the model plan the layout. "
                            "可选的 JSON 元素表，模型会原样保留；仅 *_bbox 模式使用，"
                            "留空则由模型自拟布局。"
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
        target_canvas="",
        bboxes_json="",
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
            target_canvas=target_canvas,
            bbox_json=bboxes_json if mode_code in BBOX_MODES else "",
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
        target_canvas="",
        bboxes_json="",
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
            target_canvas or "",
            bboxes_json or "",
            str(seed),
            str(temperature),
            str(timeout),
            str(max_tokens),
        ):
            h.update((part or "").encode("utf-8"))
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
