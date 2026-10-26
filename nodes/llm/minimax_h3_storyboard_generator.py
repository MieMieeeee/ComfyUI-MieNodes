"""MiniMax H3 role-event storyboard asset generator.

The public node emits a validated portable asset, its JSON, a readable
preview and the shot count. V3 compiles events without another LLM call.
"""
from __future__ import annotations

import hashlib
import re
import time
from typing import Any, Optional

try:
    from _mienodes_internal.core.utils import mie_log
except ImportError:
    try:
        from ...core.utils import mie_log
    except ImportError:
        from core.utils import mie_log

try:
    from _mienodes_internal.nodes.llm.minimax_h3_storyboard_prompts import (
        DEFAULT_LANGUAGE,
        DEFAULT_OUTPUT_FORMAT,
        DEFAULT_STYLE,
        LANGUAGES,
        MAX_SHOTS,
        OUTPUT_FORMATS,
        STYLES,
        STYLE_CODES,
        SYSTEM_STORYBOARD_PROMPT,
        PARSE_RETRY_CORRECTION,
        build_user_text,
        extract_json_array,
        normalize_shots,
        parse_output_format,
        parse_style,
        render_storyboard_markdown,
        shots_to_json_string,
    )
    from _mienodes_internal.nodes.llm.h3_prompts import CATEGORIES
except ImportError:
    from .minimax_h3_storyboard_prompts import (
        DEFAULT_LANGUAGE,
        DEFAULT_OUTPUT_FORMAT,
        DEFAULT_STYLE,
        LANGUAGES,
        MAX_SHOTS,
        OUTPUT_FORMATS,
        STYLES,
        STYLE_CODES,
        SYSTEM_STORYBOARD_PROMPT,
        PARSE_RETRY_CORRECTION,
        build_user_text,
        extract_json_array,
        normalize_shots,
        parse_output_format,
        parse_style,
        render_storyboard_markdown,
        shots_to_json_string,
    )
    from .h3_prompts import CATEGORIES


MY_CATEGORY = "\U0001F411 MieNodes/\U0001F411 Prompt Generator"

# Token budget: sized for reasoning models that burn the budget on the
# think chain before the visible answer (see the LTX-2.5 sibling note).
_DEFAULT_MAX_TOKENS = 8192
_MIN_MAX_TOKENS = 64
_MAX_MAX_TOKENS = 32768

# Creative board layout: 0.8 matches the creative siblings (LTX-2.5).
_DEFAULT_TEMPERATURE = 0.8

# One LLM call at 120 s covers the typical response window with headroom.
_DEFAULT_TIMEOUT = 120

_MIN_SHOT_COUNT = 1
_MAX_SHOT_COUNT_WIDGET = 20  # Phase-1 widget cap; normalize_shots caps at MAX_SHOTS

# Strip a leading <think>...</think> block that reasoning models may emit
# before the JSON array.
_THINK_BLOCK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)

# One retry on unparseable replies, then fail the node (no partial boards).
_PARSE_RETRIES = 1


def postprocess_reply(raw_text: str) -> str:
    """Strip a leading ``<think>...</think>`` block and outer whitespace."""
    if not raw_text:
        return ""
    text = raw_text.strip()
    text = _THINK_BLOCK_RE.sub("", text, count=1).strip()
    return text


def _default_concept() -> str:
    return (
        "A short, visually striking cinematic moment with one clear "
        "protagonist, a simple emotional arc, and a satisfying final image."
    )


class H3StoryboardEnhancer:
    """Storyboard planner that talks to the project's LLMServiceConnector.

    Mirrors ``LTX25PromptEnhancer`` / ``H3PromptEnhancer`` (shared logging
    + timeout-override pattern) so all LLM nodes behave alike.
    """

    def __init__(
        self,
        llm_service_connector: Any,
        *,
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        timeout: Optional[int] = None,
    ):
        self.llm = llm_service_connector
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        # Saved/restored around invoke() so a shared connector keeps its
        # own timeout for other nodes.
        self._timeout_override = int(timeout) if timeout else None

    def _invoke(
        self,
        messages: list[dict],
        *,
        temperature: float,
        seed: Optional[int],
        stage: str,
    ) -> str:
        prev_timeout = getattr(self.llm, "timeout", None)
        try:
            if self._timeout_override is not None:
                self.llm.timeout = self._timeout_override
            t0 = time.perf_counter()
            out = self.llm.invoke(
                messages,
                seed=seed,
                temperature=temperature,
                max_tokens=self.max_tokens,
            )
            elapsed = time.perf_counter() - t0
            model_name = getattr(self.llm, "model", "?")
            if not out:
                mie_log(
                    f"H3SB {stage}: model={model_name} returned empty after {elapsed:.2f}s"
                )
                return ""
            mie_log(
                f"H3SB {stage}: model={model_name} ok in {elapsed:.2f}s response_chars={len(out)}"
            )
            return out.strip()
        finally:
            if prev_timeout is not None:
                self.llm.timeout = prev_timeout

    @staticmethod
    def _build_messages(system_prompt: str, user_text: str) -> list[dict]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

    def _call_storyboard(
        self, messages: list[dict], *, seed: Optional[int]
    ) -> list[Any]:
        """One storyboard call with a corrective retry.

        The retry appends an explicit "reply with ONLY the JSON array"
        user turn instead of replaying the identical messages — a model
        that answered in prose the first time needs the correction, not
        a repeat. Raises ``RuntimeError`` (with the reply head for
        diagnosis) when every attempt fails; a partial board must never
        reach the outputs.
        """
        last_error: Optional[Exception] = None
        last_head = "<no reply>"
        for attempt in range(1 + _PARSE_RETRIES):
            attempt_messages = messages
            if attempt > 0:
                attempt_messages = messages + [
                    {"role": "user", "content": PARSE_RETRY_CORRECTION}
                ]
            raw = postprocess_reply(
                self._invoke(
                    attempt_messages,
                    temperature=self.temperature,
                    seed=seed,
                    stage=f"plan[attempt {attempt + 1}]",
                )
            )
            if not raw:
                last_error = ValueError("empty storyboard reply")
                last_head = "<empty reply>"
                continue
            last_head = raw[:200]
            try:
                return extract_json_array(raw)
            except ValueError as exc:
                last_error = exc
                mie_log(
                    f"H3SB: parse failed ({exc}); retry head={last_head!r}"
                )
        raise RuntimeError(
            f"storyboard reply unparseable after {1 + _PARSE_RETRIES} attempts: "
            f"{last_error}; last reply head: {last_head!r}"
        )

    def __call__(
        self,
        concept: str,
        shot_count: int,
        *,
        style: str = DEFAULT_STYLE,
        genre: str = "",
        language: str = DEFAULT_LANGUAGE,
        output_format: str = DEFAULT_OUTPUT_FORMAT,
        seed: Optional[int] = None,
    ) -> dict:
        """Run the storyboard pipeline.

        Returns ``{"storyboard_text", "shots_json", "shot_count"}``.
        """
        style_code = parse_style(style)
        if style_code not in STYLE_CODES:
            mie_log(
                f"H3SB: unknown style {style!r} (parsed {style_code!r}); "
                f"falling back to {DEFAULT_STYLE}"
            )
            style_code = DEFAULT_STYLE
        fmt_code = parse_output_format(output_format)
        if fmt_code not in ("table", "detailed", "minimal"):
            fmt_code = DEFAULT_OUTPUT_FORMAT
        count = max(_MIN_SHOT_COUNT, min(int(shot_count or 1), MAX_SHOTS))

        idea = (concept or "").strip() or _default_concept()
        if not (concept or "").strip():
            mie_log(f"H3SB: empty concept; using default idea: {idea[:80]!r}")

        user_text = build_user_text(idea, count, style_code, genre, language)
        messages = self._build_messages(SYSTEM_STORYBOARD_PROMPT, user_text)

        raw_shots = self._call_storyboard(messages, seed=seed)
        shots, warnings = normalize_shots(raw_shots, count)

        storyboard_text = render_storyboard_markdown(shots, fmt_code, warnings)
        return {
            "storyboard_text": storyboard_text,
            "shots_json": shots_to_json_string(shots),
            "shot_count": len(shots),
        }


# --------------------------------------------------------------------------- #
# ComfyUI node
# --------------------------------------------------------------------------- #
from .storyboard_asset_generator import AssetStoryboardEnhancer, preview_asset

class MiniMaxH3StoryboardGenerator:
    """User role events -> validated portable asset -> V3 (no prompt rewriting)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "llm_service_connector": ("LLMServiceConnector",),
            "name": ("STRING", {"default": "新分镜", "tooltip": "分镜资产名，可直接保存入库"}),
            "concept": ("STRING", {"multiline": True, "default": "角色2动作：双手拿着苹果递向角色1。\n角色2：吃苹果吗？\n角色1：不吃。",
                "tooltip": "按时间顺序，每行一个事件：角色1：台词 / 角色1动作：动作 / 角色1入场：动作 / 角色1离场：动作；可加[zh]或[en]。原文逐字锁定。"}),
            "character_count": ("INT", {"default": 2, "min": 1, "max": 8}),
            "character_settings": ("STRING", {"multiline": True, "default": "角色1：角色1 | 画面内 | 左侧 | adult male\n角色2：角色2 | 画面内 | 右侧 | adult female",
                "tooltip": "每角色一行：角色1：名字 | 画面内/画外/稍后入场 | 左侧/右侧/中间/无 | 声音要求。数量须与角色数一致；留空使用编号角色。外观来自后续角色资产。"}),
            "shot_count": ("INT", {"default": 1, "min": 1, "max": 20,
                                  "tooltip": "Loop生成片段数。约14秒内能装下就填1：每多一段就多一次续生成，单人独占台词的续生成片段是已知崩角色形态。一个场景内的切镜写在constraints中，由camera_cuts编译为同一H3提示词的[Shot 1]/[Shot 2]。"}),
            "target_duration_seconds": ("FLOAT", {"default": 20, "min": 1, "max": 600, "step": 1,
                "tooltip": "目标交付总时长；按H3帧网格分配并计入续接裁剪，实际值在预览中显示。过短会报错，不删减台词。"}),
            "camera_framing": (["medium_shot", "two_shot", "medium_close_up", "close_up", "wide_shot"],),
            "camera_movement": (["locked", "gentle_follow"],),
            "camera_perspective": (["observer", "first_person"],),
            "language": (["zh", "en"], {"default": "zh"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
        }, "optional": {
            "look": ("STRING", {"default": "Natural lighting, coherent visual style.", "multiline": True}),
            "setting": ("STRING", {"default": "", "multiline": True, "tooltip": "人物位置与情境；具体环境可由V3的可选场景资产提供。"}),
            "constraints": ("STRING", {"default": "", "multiline": True, "tooltip": "导演约束，原样保留。例如：只拍两人对话，不把台词中的回忆拍出来。"}),
            "continuation_overlap_frames": ("INT", {"default": 22, "min": 0, "max": 120, "tooltip": "需与Loop续接重叠帧一致；独立片段不裁剪时设0。"}),
            "temperature": ("FLOAT", {"default": 0.2, "min": 0, "max": 2, "step": 0.05}),
            "max_tokens": ("INT", {"default": 8192, "min": 256, "max": 32768}),
            "timeout": ([30, 60, 120, 300], {"default": 120}),
            "audio_mode": (["generated", "reference"], {"default": "generated", "tooltip": "generated：H3自选声音；reference：V3中每个角色都接带参考音频的人物资产，画外角色仅使用声音。最多3个角色；Asset→Slots音频需接到H3参考音频输入。"}),
        }}

    RETURN_TYPES = ("MIE_STORYBOARD", "STRING", "STRING", "INT")
    RETURN_NAMES = ("storyboard", "storyboard_json", "storyboard_text", "shot_count")
    FUNCTION = "generate_storyboard"
    CATEGORY = MY_CATEGORY

    def generate_storyboard(self, llm_service_connector, concept, shot_count=2, seed=0,
                            temperature=0.2, max_tokens=8192, timeout=120, **options):
        planner = AssetStoryboardEnhancer(llm_service_connector, temperature=temperature,
                                           max_tokens=max_tokens, timeout=timeout)
        asset = planner(concept, shot_count, seed=seed, **options)
        import json
        return asset, json.dumps(asset, ensure_ascii=False, indent=2), preview_asset(asset), len(asset["shots"])

    @classmethod
    def IS_CHANGED(cls, llm_service_connector, **options):
        import json
        from .storyboard_asset_generator import PROMPT_PATH
        digest = hashlib.sha256(PROMPT_PATH.read_bytes())
        digest.update(json.dumps(options, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8"))
        state = getattr(llm_service_connector, "get_state", None)
        if callable(state):
            digest.update(str(state()).encode("utf-8"))
        else:
            for key in ("api_url", "api_token", "model"):
                digest.update(str(getattr(llm_service_connector, key, "")).encode("utf-8"))
        return digest.hexdigest()

    is_changed = IS_CHANGED
