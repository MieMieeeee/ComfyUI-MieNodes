"""ComfyUI node shells for the host-neutral mie_craft.loop_generator."""
from __future__ import annotations
import hashlib
import json
import os
import re
import time
from typing import Any, Optional
from mie_craft import loop_generator as _impl
from mie_craft.loop_generator import LoopRuntime
from mie_llm.media import image_tensor_batch_to_data_urls
from mie_llm.utils import mie_log

# Preserve the historical helper API; implementation globals live in Craft.
for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)

try:
    import nodes as _comfy_nodes  # type: ignore
    from comfy_execution.execution import InterruptProcessingException  # type: ignore

    def _comfy_interrupt_pressed() -> bool:
        try:
            return bool(_comfy_nodes.interrupt_processing())
        except Exception:  # pragma: no cover - defensive
            return False
except Exception:  # pragma: no cover - tests / standalone
    InterruptProcessingException = Exception  # type: ignore

    def _comfy_interrupt_pressed() -> bool:
        return False


def _check_interrupt(stage: str) -> None:
    """Raise ``InterruptProcessingException`` if the executor signalled
    a stop. No-op outside ComfyUI's runtime (tests / standalone)."""
    if _comfy_interrupt_pressed():
        mie_log(f"H3LOOP {stage}: interrupt pressed; aborting")
        raise InterruptProcessingException(
            f"H3LOOP {stage}: interrupt pressed by user"
        )


def _caption_cache_disk_root() -> str:
    """Resolve the on-disk caption cache directory.

    Resolution order:
      1) ComfyUI's runtime output dir via ``folder_paths.get_output_directory()``.
      2) Fallback ``<repo>/output/mienodes/caption_cache`` for tests /
         standalone import contexts where ``folder_paths`` is unavailable.
    """
    try:
        import folder_paths  # type: ignore

        output_dir = folder_paths.get_output_directory()
        if output_dir:
            return os.path.join(
                os.path.abspath(str(output_dir)),
                "mien_nodes",
                "caption_cache",
            )
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        pass
    # __file__ is nodes/llm/<thisfile>.py; walk up two to the repo root.
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))
    return os.path.join(repo_root, "output", "mien_nodes", "caption_cache")


class H3LoopPromptEnhancer(_impl.H3LoopPromptEnhancer):
    """Bind only ComfyUI interruption, media conversion and cache location."""
    def __init__(self, *args, runtime=None, **kwargs):
        super().__init__(*args, runtime=runtime or LoopRuntime(
            check_interrupt=lambda stage: _check_interrupt(stage),
            cache_root=lambda: _caption_cache_disk_root(),
            image_urls=lambda image: image_tensor_batch_to_data_urls(image),
            interrupt_exception=InterruptProcessingException,
        ), **kwargs)

class MiniMaxH3LoopPromptGenerator:
    """ComfyUI node: free-form user_input -> Production Plan ``plan_json``.

    Paste a concept paragraph or one beat per line into ``user_input``;
    the node splits it into ``scene_count`` scenes whose durations sum
    close to ``total_duration_seconds``, then feeds ``plan_json`` into
    ``MiniMaxH3ChainPlanModern.plan_json_input``. The generated JSON only
    carries ``shots`` / ``prompt_prefix`` — sampler steps, canvas size,
    continuation mode, run name etc. stay on the Plan node's widgets.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "user_input": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": (
                            "Everything you want, in any shape: a concept "
                            "paragraph, speaker：line dialogue, or mixed. "
                            "Dialogue is extracted then packed "
                            "deterministically; narration still goes through "
                            "the storyboard LLM. Blank = default concept."
                        ),
                    },
                ),
                "enhance_user_input": (
                    list(_ENHANCE_USER_INPUT_LABELS),
                    {
                        "default": _ENHANCE_USER_INPUT_LABELS[0],
                        "tooltip": (
                            "on: rewrite user_input with the User Input "
                            "Enhancer before planning. A dialogue board or "
                            "a single spoken line is cached and kept "
                            "verbatim — a paraphrase is pinned back on the "
                            "first try, not retried. A colon line inside "
                            "ordinary prose is rewritten, not pinned. "
                            "off (default): use "
                            "user_input as written. Keep off when the "
                            "standalone Enhancer is already wired upstream. "
                            "The rewrite is appended to the run summary. "
                            "Changing scene_count or duration re-runs "
                            "planning but reuses a cached rewrite of the "
                            "same draft and reference_mode."
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
                            "Seed base for the per-scene seed chain "
                            "(scene N gets base+N as a string). 0 = derive "
                            "from the clock each run."
                        ),
                    },
                ),
            },
            "optional": {
                "scene_count": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 128,
                        "tooltip": (
                            "Number of scenes.\n\n"
                            "0 = auto. Dialogue boards hold at most 2 "
                            "spoken lines per scene, inside the 14s H3 "
                            "window (a speaker change does NOT force a "
                            "cut). Pacing does not change this cut count — "
                            "it only changes speech tempo and the BRISK/"
                            "MEASURED prompt directive. Narration boards "
                            "let the storyboard LLM decide, using pacing as "
                            "split bias (fast=aggressive, slow=conservative)."
                            "\n\n"
                            ">0 on a dialogue board: speech is spread evenly "
                            "across that many scenes, capped at one line per "
                            "scene — a count above the spoken-line count is "
                            "capped with a warning; NO silent reaction scenes "
                            "are manufactured to reach a number (没事尽可能"
                            "不切 scene, 2026-10-12). A line is never split "
                            "mid-utterance. 场景数：0=自动（最小切分）；显式"
                            "数字在台词数以内精确生效，超出封顶为每场一句并"
                            "警告，绝不添加无声反应镜头凑数。\n\n"
                            ">0 on a narration board: the storyboard LLM "
                            "produces exactly that many scenes."
                        ),
                    },
                ),
                "total_duration_seconds": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1800,
                        "tooltip": (
                            "Whole-board duration budget. The storyboard LLM "
                            "distributes it across scenes; each scene rounds up "
                            "onto the 17k+5 frame grid, so the delivered total "
                            "lands close to this (±20% tolerated silently).\n\n"
                            "0 = auto: derived from dialogue line count and the "
                            "selected pacing preset (TTS rate + per-turn pause). "
                            "For concepts without speaker:line dialogue the "
                            "auto budget falls back to 15s. Set >0 to override "
                            "the auto estimate. The override actually shrinks "
                            "shots (clamped to 4-14s each); the old '15 default' "
                            "no longer applies.\n\n"
                            "MiniMax-H3 single-generation window: 4-15 s per "
                            "upstream MiniMax-AI/MiniMax-H3 README. This node "
                            "hard-caps every per-shot duration_seconds at 14 s "
                            "and emits a preflight warning when clamping occurred."
                        ),
                    },
                ),
                "pacing": (
                    list(_PACING_LABELS),
                    {
                        "default": _PACING_LABELS[1],
                        "tooltip": (
                            "SPEECH TEMPO (语速) of the whole board — "
                            "this is the faster/slower rhythm control, "
                            "NOT a cut-density control.\n\n"
                            "On a dialogue board with scene_count=0, "
                            "pacing does NOT change how many scenes you "
                            "get (auto packing is at most 2 spoken lines "
                            "per scene, inside the 14s window). It does:\n"
                            "1. Prompt tempo: BRISK / natural / MEASURED "
                            "injected into the prefix and every shot.\n"
                            "2. Auto length (total=0): TTS rate + turn "
                            "pause (fast CN 4.5 chars/s + 1.0s, normal "
                            "3.5 + 1.5s, slow 2.8 + 2.0s).\n"
                            "Narration boards also map fast/normal/slow "
                            "to the storyboard LLM's aggressive/balanced/"
                            "conservative split bias. An explicit "
                            "scene_count is the cut-count override; "
                            "total_duration_seconds stays the duration "
                            "constraint."
                        ),
                    },
                ),
                "generation_mode": (
                    list(GENERATION_MODES),
                    {
                        "default": GENERATION_MODES[0],
                        "tooltip": (
                            "per_shot: one LLM call per scene, best continuity "
                            "(recommended). single_call: one LLM call for the "
                            "whole board (cheaper/faster).",
                        ),
                    },
                ),
                "camera_discipline": (
                    list(CAMERA_DISCIPLINES),
                    {
                        "default": CAMERA_DISCIPLINES[0],
                        "tooltip": (
                            "镜头纪律。auto：分镜自由构图（默认——注意分镜 "
                            "LLM 有一条「景别交替」硬规则，会主动切特写，"
                            "概念里的「镜头固定」只是软性提示）。locked："
                            "固定机位——全场一个构图（two_shot/medium_shot），"
                            "禁止景别交替和特写，camera_movement 一律 "
                            "static_lockoff；概念里写的构图就是全片的构图。"
                            "no_closeups：禁特写——景别保持中景及以上，机位"
                            "运动自由。locked/no_closeups 由代码在分镜归一"
                            "化后强制执行，不依赖模型自觉。",
                        ),
                    },
                ),
                "category": (
                    list(LOOP_CATEGORIES),
                    {
                        "default": LOOP_CATEGORIES[0],
                        "tooltip": (
                            "none: no extra cinematography contract — "
                            "but once dialogue lines are extracted the "
                            "board is automatically treated as dialogue "
                            "(spoken-scene framing, shot/reverse-shot, "
                            "no on-screen subtitles). "
                            "dialogue: the same contract, always. "
                            "action: motion-blur / camera-shake advice. "
                            "The dialogue verbatim contract runs "
                            "whenever lines are extracted, on any "
                            "category."
                        ),
                    },
                ),
                "output_language": (
                    ["en", "zh"],
                    {
                        "default": "en",
                        "tooltip": (
                            "Narrative language for generated scene prose. "
                            "English default is recommended for H3 stability; "
                            "spoken lines still follow dialogue-tag policy."
                        ),
                    },
                ),
                "seed_mode": (
                    list(SEED_MODES),
                    {
                        "default": SEED_MODES[0],
                        "tooltip": (
                            "per_scene_increment (recommended): seed_base+index "
                            "per scene — deterministic but DISTINCT per scene. "
                            "This mirrors the upstream ComfyUI-MiniMaxH3-Context-"
                            "Loop plugin's own default, which derives per-scene "
                            "seeds from one base (sha256(base:index:shot_id)) "
                            "when the plan omits seeds and relies on that "
                            "determinism for checkpoint recovery; identical "
                            "seeds on every clip make consecutive clips sample "
                            "near-identical noise (repeated motion rhythm). "
                            "same_across_scenes: every scene shares one seed — "
                            "use it only when you deliberately want the "
                            "identical-noise look."
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
                            "Advanced tuning: LLM creativity/randomness. "
                            "Most workflows should keep the default."
                        ),
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": _DEFAULT_MAX_TOKENS,
                        "min": _MIN_MAX_TOKENS,
                        "max": _MAX_MAX_TOKENS,
                        "tooltip": (
                            "Advanced tuning: upper token budget for each LLM call."
                        ),
                    },
                ),
                "timeout": (
                    [60, 120, 300, 600],
                    {
                        "default": _DEFAULT_TIMEOUT,
                        "tooltip": (
                            "Advanced tuning: per-call timeout (seconds)."
                        ),
                    },
                ),
                "reference_mode": (
                    list(REFERENCE_MODES),
                    {
                        "default": REFERENCE_MODES[0],
                        "tooltip": (
                            "t2va: text-only chain (default). "
                            "i2va: scene 1 anchored to one <Picture 1>. "
                            "fl2va: scene 1 + alternating end targets "
                            "<Picture (N % 2) + 1>. "
                            "ref2va: N pictures active for every scene "
                            "(six-section prompt contract)."
                        ),
                    },
                ),
                "images": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "One IMAGE batch, all modes. Wiring (mirror the "
                            "upstream H3 workflow): batch[0] (Picture 1) is "
                            "the OPENING frame -> MiniMax H3 First-Scene "
                            "Image Gate 'image' (i2va/fl2va). For fl2va, "
                            "batch[1..] (Picture 2..N) are per-scene end "
                            "targets -> Chain Frame Index Switch "
                            "frame_1..frame_{N-1} (scene j ends on "
                            "frame_j, wraps after the last slot). For "
                            "ref2va, the whole batch -> Reference to Video "
                            "images (all active every scene). i2va keeps "
                            "only batch[0] and drops the rest with a "
                            "warning; t2va ignores images (warning). "
                            "Scene-transition intent (e.g. '从图一到图二再"
                            "回到图一' / 'A→B→A') goes in user_input. "
                            "1-9 frames. REQUIRED for i2va / fl2va / ref2va."
                        ),
                    },
                ),
                "caption_mode": (
                    list(CAPTION_MODES),
                    {
                        "default": CAPTION_MODES[0],
                        "tooltip": (
                            "Caption cache strategy for images. "
                            "cache_memory_disk (recommended): persistent cache. "
                            "cache_memory_only: RAM-only cache. "
                            "no_cache: always bypass cache. "
                            "force_recaption_once: force fresh captions now."
                        ),
                    },
                ),
                # 连续性审片（2026-11-05）：旧轨自动分镜也一样要有人跨镜头
                # 复核一遍。纯文本审片，不碰渲染。
                "critic": (
                    ["on", "off"],
                    {
                        "default": "on",
                        "tooltip": (
                            "连续性审片 · continuity critic:\n"
                            "on：自动分镜生成后再让一个“读者”复核一遍（覆盖、连续性、"
                            "单镜可演、节奏），发现观众看得见的矛盾就回修重摇；"
                            "每次多一次模型调用（约 30~60 秒），修不好则保留原分镜并附警告。\n"
                            "off：不做复核（快、零额外调用）。\n"
                            "on: a second reader checks the auto-storyboard across shots "
                            "(coverage, continuity, one beat per shot, pacing) and repairs "
                            "what a viewer would notice. One extra model call; an "
                            "unrepaired board is kept with the notes attached."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "plan_json",
        "summary",
        "board_kind",
    )
    FUNCTION = "generate"
    CATEGORY = MY_CATEGORY

    def generate(
        self,
        llm_service_connector,
        *,
        user_input: str = "",
        seed=None,
        scene_count=0,
        total_duration_seconds=0,
        generation_mode=GENERATION_MODES[0],
        category="",
        output_language="en",
        seed_mode="",
        temperature=_DEFAULT_TEMPERATURE,
        max_tokens=_DEFAULT_MAX_TOKENS,
        timeout=_DEFAULT_TIMEOUT,
        reference_mode=REFERENCE_MODES[0],
        references_text: str = "",
        images=None,
        caption_mode="",
        force_recaption: bool = False,
        caption_cache_scope: str = "memory_disk",
        pacing: str = _PACING_LABELS[1],
        enhance_user_input: str = _ENHANCE_USER_INPUT_LABELS[0],
        camera_discipline: str = CAMERA_DISCIPLINES[0],
        critic: str = "on",
    ):
        enhancer = H3LoopPromptEnhancer(
            llm_service_connector,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        out = enhancer(
            user_input=user_input,
            scene_count=scene_count,
            total_duration_seconds=total_duration_seconds,
            generation_mode=generation_mode,
            category=category,
            output_language=output_language,
            seed=seed,
            seed_mode=seed_mode,
            reference_mode=reference_mode,
            references_text=references_text,
            images=images,
            caption_mode=caption_mode,
            force_recaption=force_recaption,
            caption_cache_scope=caption_cache_scope,
            pacing=pacing,
            enhance_user_input=parse_enhance_user_input(enhance_user_input),
            camera_discipline=camera_discipline,
            critic=critic,
        )
        return (
            out["plan_json"],
            out["summary"],
            out["board_kind"],
        )

    def is_changed(
        self,
        llm_service_connector,
        *,
        user_input: str = "",
        seed=None,
        scene_count=0,
        total_duration_seconds=0,
        generation_mode=GENERATION_MODES[0],
        category="",
        output_language="en",
        seed_mode="",
        temperature=_DEFAULT_TEMPERATURE,
        max_tokens=_DEFAULT_MAX_TOKENS,
        timeout=_DEFAULT_TIMEOUT,
        reference_mode=REFERENCE_MODES[0],
        images=None,
        references_text: str = "",
        caption_mode="",
        force_recaption: bool = False,
        caption_cache_scope: str = "memory_disk",
        pacing: str = _PACING_LABELS[1],
        enhance_user_input: str = _ENHANCE_USER_INPUT_LABELS[0],
        camera_discipline: str = CAMERA_DISCIPLINES[0],
        critic: str = "on",
    ):
        h = hashlib.md5(usedforsecurity=False)
        for part in (
            user_input,
            str(seed),
            str(scene_count),
            parse_seed_mode(seed_mode),
            str(total_duration_seconds),
            generation_mode,
            category,
            output_language,
            str(temperature),
            str(max_tokens),
            str(timeout),
            parse_reference_mode(reference_mode),
            parse_caption_mode(caption_mode),
            str(bool(force_recaption)),
            str(caption_cache_scope or "memory_disk"),
            pacing,
            str(parse_enhance_user_input(enhance_user_input)),
            parse_camera_discipline(camera_discipline),
            # The critic widget changes the board (it can trigger a repair
            # round), so it belongs in the cache key like every other
            # output-shaping widget.
            str(critic or "on"),
            (references_text or "").strip(),
        ):
            h.update((part or "").encode("utf-8"))
        # images: hash tensor shape plus head/mid/tail samples so a
        # same-shape swap still invalidates the node.
        if images is None:
            h.update(b"none")
        else:
            try:
                shape = tuple(images.shape)
            except AttributeError:
                shape = ()
            h.update(repr(shape).encode("utf-8"))
            h.update(str(getattr(images, "dtype", "")).encode("utf-8"))
            sample_bytes = b""
            try:
                if hasattr(images, "detach") and hasattr(images, "cpu"):
                    flat = images.detach().cpu().reshape(-1)
                elif hasattr(images, "reshape"):
                    flat = images.reshape(-1)
                else:
                    flat = None
                if flat is not None:
                    n = int(getattr(flat, "shape", [0])[0] if hasattr(flat, "shape") else len(flat))
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
            h.update(hashlib.md5(sample_bytes, usedforsecurity=False).hexdigest().encode("ascii"))
        try:
            h.update(llm_service_connector.get_state().encode("utf-8"))
        except AttributeError:
            h.update(str(getattr(llm_service_connector, "api_url", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "api_token", "")).encode("utf-8"))
            h.update(str(getattr(llm_service_connector, "model", "")).encode("utf-8"))
        return h.hexdigest()


# Asset slot sockets are character_1..9 / scene_1..9 ONLY. A bare
# startswith("scene_") scan also swallows the scene_count WIDGET —
# the exact live bug that silently ignored every explicit scene
# count on the V2 node path (2026-10-12).
_SLOT_KEY_RE = re.compile(r"^(?:character|scene)_\d+$")


class MiniMaxH3LoopPromptGeneratorV2(MiniMaxH3LoopPromptGenerator):
    """The same generator, with assets wired in from the Load nodes.

    Per 2026-10-10 (final) ruling: NO selection widgets on this node. The
    character/scene slots are SOCKETS — wire the ``asset`` output of Load
    Character Asset / Load Scene Asset into ``character_1..9`` /
    ``scene_1..9``; a new slot appears once the previous one is wired (JS
    autogrow). One wire carries the whole asset (name, description, voice,
    image, audio), so nothing is read from the library here and no vision
    caption runs: the manifest is assembled straight from the bundles, and
    the appended ``images`` output returns the assembled picture batch for
    the video chain. Slot order = character_1..N then scene_1..M — that
    order IS ``<Picture N>``.

    ``user_input`` stays STORY ONLY. Outputs: the same ``plan_json`` schema
    as V1 (Production Plan wiring unchanged) plus ``images``.
    """

    CATEGORY = MY_CATEGORY
    FUNCTION = "generate_v2"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "IMAGE", "MIE_ASSET_ROSTER")
    RETURN_NAMES = ("plan_json", "summary", "board_kind", "images", "assets")

    MAX_SLOTS = 9  # per kind; execution enforces the H3 9-picture cap

    @classmethod
    def INPUT_TYPES(cls):
        # Derived from V1 rather than copied: the widget set must never
        # drift, and a duplicated 280-line INPUT_TYPES is a drift waiting
        # to happen.
        spec = dict(MiniMaxH3LoopPromptGenerator.INPUT_TYPES())
        optional = dict(spec.get("optional") or {})
        for i in range(1, cls.MAX_SLOTS + 1):
            optional["character_%d" % i] = (
                "MIE_ASSET",
                {
                    "tooltip": (
                        "角色槽位 %d：接 Load Character Asset 的 asset 输出。"
                        "接上本槽才会出现下一槽。槽位顺序（角色在前、场景在后）"
                        "就是 <Picture N>。" % i
                    ),
                },
            )
        for i in range(1, cls.MAX_SLOTS + 1):
            optional["scene_%d" % i] = (
                "MIE_ASSET",
                {
                    "tooltip": (
                        "场景槽位 %d：接 Load Scene Asset 的 asset 输出（环境"
                        "参考，不占 <Subject N>）。" % i
                    ),
                },
            )
        spec["optional"] = optional
        return spec

    # ---- bundle collection ------------------------------------------------
    @classmethod
    def _slot_bundles(cls, kwargs, prefix, kind):
        """Connected slot values in order (unconnected slots are skipped;
        a gap simply compacts — the slot ORDER of the wired ones is what
        becomes <Picture N>)."""
        bundles = []
        for i in range(1, cls.MAX_SLOTS + 1):
            value = kwargs.get("%s_%d" % (prefix, i))
            if value is None:
                continue
            if not isinstance(value, dict) or value.get("kind") != kind:
                raise RuntimeError(
                    "%s_%d is not a %s asset bundle (got %s); wire the "
                    "'asset' output of the matching Load node"
                    % (prefix, i, kind, type(value).__name__)
                )
            bundles.append(value)
        return bundles

    @classmethod
    def _build_roster_from_bundles(cls, characters, scenes, offscreen_names=None):
        """Assemble everything the enhancer needs from the wired bundles:
        roster JSON (same shape the retired batch node emitted) + a preset
        manifest (about = authored description, role = identity /
        environment) + the picture batch for the images output."""
        offscreen = set(offscreen_names or ())
        visible_characters = [b for b in characters if b["name"] not in offscreen]
        if len(visible_characters) + len(scenes) > 9:
            raise RuntimeError(
                "wired slots give %d pictures (characters + scenes); the H3 "
                "manifest supports at most 9" % (len(characters) + len(scenes))
            )
        names = [b["name"] for b in characters + scenes]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise RuntimeError(
                "asset wired into more than one slot: %s; one asset occupies "
                "one slot" % ", ".join(sorted(dupes))
            )

        roster = []
        manifest = []
        tensors = []
        slot = 0
        for bundle in visible_characters:
            slot += 1
            roster.append({
                "name": bundle["name"], "kind": "character", "slot": slot,
                "description": bundle["description"],
                "appearance": bundle["description"],
                "voice": bundle["voice"],
                "image_count": 1,
                "audio_count": 1 if bundle.get("audio") is not None else 0,
            })
            manifest.append({
                "slot": "Picture %d" % slot,
                "about": bundle["description"],
                "role": "identity", "source": "asset",
            })
            tensors.append(bundle["image"])
        for bundle in scenes:
            slot += 1
            roster.append({
                "name": bundle["name"], "kind": "scene", "slot": slot,
                "description": bundle["description"],
                "image_count": 1,
            })
            manifest.append({
                "slot": "Picture %d" % slot,
                "about": bundle["description"],
                "role": "environment", "source": "asset",
            })
            tensors.append(bundle["image"])

        # The images BATCH needs one shape, but the H3 chain does not:
        # MiniMaxH3ReferenceToVideo takes each picture on its OWN socket and
        # resizes each independently ('match' to the generation's pixel area,
        # 'max' to a 2048 short edge). So mixed sizes are legal — only the
        # convenience batch output steps aside (None) and the plan tells the
        # user to wire Asset → Slots' picture_i outputs, which carry each
        # asset's native tensor without any cat.
        import torch

        shapes = {}
        for entry, tensor in zip(roster, tensors):
            shapes.setdefault(tuple(tensor.shape[1:]), []).append(entry["name"])
        images = None
        size_note = ""
        if len(shapes) > 1:
            detail = "; ".join(
                "%dx%d: %s" % (w, h, ", ".join(ns))
                for (h, w, _c), ns in sorted(shapes.items())
            )
            size_note = (
                "picture sizes differ (%s): the images batch output is off "
                "for this run — wire Asset → Slots picture_1..N to "
                "ref_image_0..N-1; H3 resizes each reference independently"
                % detail
            )
        elif tensors:
            images = torch.cat(tensors, dim=0)
        return json.dumps(roster, ensure_ascii=False), manifest, images, size_note

    def generate_v2(
        self,
        llm_service_connector,
        *,
        enhance_user_input: str = _ENHANCE_USER_INPUT_LABELS[0],
        temperature=_DEFAULT_TEMPERATURE,
        max_tokens=_DEFAULT_MAX_TOKENS,
        timeout=_DEFAULT_TIMEOUT,
        **kwargs,
    ):
        characters = self._slot_bundles(kwargs, "character", "character")
        scenes = self._slot_bundles(kwargs, "scene", "scene")
        offscreen = _dlg_declared_offscreen_names(kwargs.get("user_input", ""))
        speaker_references = None
        if offscreen:
            speaker_references = {}
            audio_index = 0
            for bundle in characters:
                if bundle.get("audio") is not None:
                    audio_index += 1
                    speaker_references[bundle["name"]] = {
                        "sid": "S%d" % audio_index, "voice": bundle.get("voice", ""),
                        "description": bundle.get("description", ""),
                    }
        # The slot sockets are consumed HERE; they must not leak into the
        # enhancer's keyword contract.
        for key in [k for k in list(kwargs)
                     if _SLOT_KEY_RE.match(k)]:
            kwargs.pop(key)
        roster_manifest = None
        images_out = None
        roster_arg = ""
        assets_out = None
        size_note = ""
        if characters or scenes:
            roster_arg, roster_manifest, images_out, size_note = (
                self._build_roster_from_bundles(characters, scenes, offscreen_names=offscreen)
            )
            if size_note:
                # generate_v2 has no warnings list of its own — fold the
                # note into the summary via the enhancer's warning path.
                self._slot_size_note = size_note
                log_pipeline("asset slots: " + size_note)
            # The wired bundles, aggregated in slot order — what Asset →
            # Slots splits back into picture/audio slots downstream.
            assets_out = [dict(b, image=None) if b["name"] in offscreen else b for b in characters] + scenes
            kwargs.pop("images", None)  # the images socket is not consulted

        enhancer = H3LoopPromptEnhancer(
            llm_service_connector,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        out = enhancer(
            asset_roster=roster_arg,
            roster_manifest=roster_manifest,
            offscreen_names=offscreen,
            speaker_references=speaker_references,
            enhance_user_input=parse_enhance_user_input(enhance_user_input),
            camera_discipline=kwargs.pop("camera_discipline", CAMERA_DISCIPLINES[0]),
            **kwargs,
        )
        note = getattr(self, "_slot_size_note", "")
        if note:
            out["summary"] = out["summary"] + "\n- " + note
            self._slot_size_note = ""
        return (
            out["plan_json"],
            out["summary"],
            out["board_kind"],
            images_out,
            assets_out,
        )

    def is_changed(self, llm_service_connector, **kwargs):
        # Slot sockets are wired inputs: upstream Load nodes invalidate via
        # their own content-hash IS_CHANGED, which naturally re-runs this
        # node. Nothing widget-side left to hash for them.
        for key in [k for k in list(kwargs)
                     if _SLOT_KEY_RE.match(k)]:
            kwargs.pop(key)
        return super().is_changed(llm_service_connector, **kwargs)
