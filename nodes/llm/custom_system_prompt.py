"""Generic custom-system-prompt three-node set.

Pick any builtin or user-defined system prompt and drive any LLM with it —
optionally feeding in source / reference images or reference video frames
(multimodal). Mirrors the Kontext three-node pattern for persistence and reuses
the same ``core.utils`` media helpers BerniniPromptGenerator uses.

Nodes:
  - CustomSystemPromptGenerator  builtin/user prompt + optional media
  - AddCustomSystemPrompt        create a user prompt
  - RemoveCustomSystemPrompt     delete a user prompt

User data lives in ``user_system_prompts.json`` (gitignored) so a ``git pull``
never clobbers it. All media inputs are optional: when none are wired the user
content stays a plain string (identical to text-only behavior).

Media sampling mirrors BerniniPromptGenerator's two-knob design. Both default
to 3 so a large batch doesn't blow up the request size:
  - ``video_frames``           frames sampled from the ``source`` batch (>=1)
  - ``reference_video_frames`` frames sampled from ``reference_video``
                               (default 3; 0 = forward all legacy, 1..16 = sample)
"""
import hashlib
import json
import os

try:
    from _mienodes_internal.nodes.llm.prompts.loader import (
        list_usable_builtin_prompts,
        load_prompt_text,
    )
except ImportError:
    from .prompts.loader import list_usable_builtin_prompts, load_prompt_text

try:
    from _mienodes_internal.core.utils import (
        build_multimodal_user_content,
        image_tensor_batch_to_data_urls,
    )
except ImportError:
    from ...core.utils import (
        build_multimodal_user_content,
        image_tensor_batch_to_data_urls,
    )

MY_CATEGORY = "🐑 MieNodes/🐑 Prompt Generator"

# nodes/llm/custom_system_prompt.py -> up 3 levels = repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
USER_CUSTOM_FILE = os.path.join(_REPO_ROOT, "user_system_prompts.json")

__all__ = [
    "CustomSystemPromptGenerator",
    "AddCustomSystemPrompt",
    "RemoveCustomSystemPrompt",
]


def load_user_custom_prompts() -> dict:
    """Read user-defined prompts as {name: str}. Malformed entries are skipped."""
    if os.path.exists(USER_CUSTOM_FILE):
        try:
            with open(USER_CUSTOM_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {k: v for k, v in data.items() if isinstance(v, str)}
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_user_custom_prompts(prompts: dict) -> None:
    with open(USER_CUSTOM_FILE, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)


def get_all_custom_system_prompts() -> dict:
    """Builtin usable prompts merged with user-defined ones (user wins on conflict)."""
    result = {name: load_prompt_text(name) for name in list_usable_builtin_prompts()}
    result.update(load_user_custom_prompts())
    return result


def _sample_urls(urls, n):
    """Uniformly sample n items preserving endpoints.

    n <= 0 (or fewer urls than n) -> passthrough all. Used so
    reference_video_frames == 0 means "forward all" (legacy Bernini behavior).
    """
    if not urls or n <= 0 or len(urls) <= n:
        return list(urls or [])
    return [urls[round(i * (len(urls) - 1) / (n - 1))] for i in range(n)]


def _collect_media_urls(source, reference_images, reference_video, video_frames, reference_video_frames):
    """Flatten non-None media inputs into a list of data URLs.

    Order: source -> reference_images -> reference_video.
      - source: sampled down to ``video_frames`` (>=1).
      - reference_images: forwarded in full.
      - reference_video: ``reference_video_frames`` == 0 forwards all;
        1..16 uniformly samples that many frames.
    """
    urls = []
    if source is not None:
        urls.extend(_sample_urls(image_tensor_batch_to_data_urls(source), video_frames))
    if reference_images is not None:
        urls.extend(image_tensor_batch_to_data_urls(reference_images))
    if reference_video is not None:
        urls.extend(_sample_urls(
            image_tensor_batch_to_data_urls(reference_video), reference_video_frames
        ))
    return urls


class CustomSystemPromptGenerator(object):
    """Run any builtin/user system prompt against an LLM, with optional media."""

    @classmethod
    def INPUT_TYPES(cls):
        all_prompts = get_all_custom_system_prompts()
        keys = list(all_prompts.keys())
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "input_text": ("STRING", {"default": "", "multiline": True}),
                "system_prompt_name": (
                    keys,
                    {
                        "default": next(iter(keys), ""),
                        "tooltip": (
                            "Pick a builtin system prompt (from prompts/*.txt, no placeholders) "
                            "or one you added via Add Custom System Prompt."
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
                "source": ("IMAGE", {"tooltip": "Source image or video-frame batch. Leave unwired for text-only."}),
                "reference_images": ("IMAGE", {"tooltip": "Reference image batch (forwarded in full)."}),
                "reference_video": ("IMAGE", {"tooltip": "Reference video-frame batch (sampled per reference_video_frames)."}),
                "video_frames": ("INT", {"default": 3, "min": 1, "max": 16, "tooltip": "Frames sampled from the source batch when it is video frames."}),
                "reference_video_frames": ("INT", {"default": 3, "min": 0, "max": 16, "tooltip": "Frames sampled from reference_video. Default 3 samples 3; 0 = forward all (legacy)."}),
                "image_detail": (["auto", "low", "high"], {"default": "auto"}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("custom_prompt",)
    FUNCTION = "generate"
    CATEGORY = MY_CATEGORY

    def generate(
        self,
        llm_service_connector,
        input_text,
        system_prompt_name,
        seed=None,
        source=None,
        reference_images=None,
        reference_video=None,
        video_frames=3,
        reference_video_frames=3,
        image_detail="auto",
        temperature=0.8,
    ):
        all_prompts = get_all_custom_system_prompts()
        sys_text = all_prompts.get(system_prompt_name)
        if sys_text is None:
            raise ValueError(f"Unknown system prompt: {system_prompt_name}")
        user_msg = input_text.strip() or "Generate a result following the system prompt."
        media_urls = _collect_media_urls(
            source, reference_images, reference_video, video_frames, reference_video_frames
        )
        # text-only -> plain string (unchanged); media present -> multimodal parts list
        user_content = (
            build_multimodal_user_content(user_msg, media_urls, image_detail)
            if media_urls
            else user_msg
        )
        messages = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": user_content},
        ]
        out = llm_service_connector.invoke(messages, seed=seed, temperature=temperature, top_p=0.9)
        return out.strip(),

    def is_changed(
        self,
        llm_service_connector,
        input_text,
        system_prompt_name,
        seed,
        source=None,
        reference_images=None,
        reference_video=None,
        video_frames=3,
        reference_video_frames=3,
        image_detail="auto",
        temperature=0.8,
    ):
        hasher = hashlib.md5()
        hasher.update(input_text.encode("utf-8"))
        hasher.update(system_prompt_name.encode("utf-8"))
        hasher.update(str(seed).encode("utf-8"))
        hasher.update(get_all_custom_system_prompts().get(system_prompt_name, "").encode("utf-8"))
        hasher.update(str(video_frames).encode("utf-8"))
        hasher.update(str(reference_video_frames).encode("utf-8"))
        hasher.update(str(image_detail).encode("utf-8"))
        hasher.update(str(temperature).encode("utf-8"))
        for tensor in (source, reference_images, reference_video):
            urls = image_tensor_batch_to_data_urls(tensor)
            hasher.update("".join(u[:64] for u in urls).encode("utf-8"))
        try:
            hasher.update(llm_service_connector.get_state().encode("utf-8"))
        except AttributeError:
            hasher.update(str(getattr(llm_service_connector, "api_url", "")).encode("utf-8"))
            hasher.update(str(getattr(llm_service_connector, "api_token", "")).encode("utf-8"))
            hasher.update(str(getattr(llm_service_connector, "model", "")).encode("utf-8"))
        return hasher.hexdigest()


class AddCustomSystemPrompt(object):
    """Persist a user-defined system prompt to user_system_prompts.json."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_name": ("STRING", {"default": ""}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("success", "log")
    FUNCTION = "add_prompt"
    CATEGORY = MY_CATEGORY

    def add_prompt(self, prompt_name, system_prompt):
        if not prompt_name or not system_prompt:
            return False, "Prompt name and system prompt must not be empty."
        customs = load_user_custom_prompts()
        if prompt_name in customs:
            return False, f"Custom prompt '{prompt_name}' already exists."
        customs[prompt_name] = system_prompt
        save_user_custom_prompts(customs)
        return True, f"Custom prompt '{prompt_name}' added."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")


class RemoveCustomSystemPrompt(object):
    """Delete a user-defined system prompt (dropdown lists only user prompts)."""

    @classmethod
    def INPUT_TYPES(cls):
        customs = load_user_custom_prompts()
        keys = list(customs.keys())
        return {
            "required": {
                "preset_name": (keys, {"default": next(iter(keys), "")}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("success", "log")
    FUNCTION = "remove_preset"
    CATEGORY = MY_CATEGORY

    def remove_preset(self, preset_name):
        customs = load_user_custom_prompts()
        if preset_name in customs:
            del customs[preset_name]
            save_user_custom_prompts(customs)
            return True, f"Custom prompt '{preset_name}' removed."
        return False, f"Custom prompt '{preset_name}' not found."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")
