"""General aspect-ratio helpers (not tied to any single model)."""

from __future__ import annotations

# Batch 2a: the two pure string helpers moved to mie_prompt.aspect_ratio;
# the ComfyUI node classes below stay here and import them.
from mie_prompt.aspect_ratio import normalize_ratio_string, ratio_from_size  # noqa: F401

MY_CATEGORY = "🐑 MieNodes/🐑 Common"

from math import gcd



class AspectRatioFromSize:
    """Compute a simplified W:H ratio string from width and height in pixels."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": 16384,
                        "tooltip": "Image width in pixels.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": 16384,
                        "tooltip": "Image height in pixels.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("aspect_ratio",)
    FUNCTION = "convert"
    CATEGORY = MY_CATEGORY

    def convert(self, width, height):
        return (ratio_from_size(width, height),)
