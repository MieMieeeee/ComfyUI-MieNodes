"""General aspect-ratio helpers (not tied to any single model)."""

from __future__ import annotations

from math import gcd


def ratio_from_size(width: int, height: int) -> str:
    """Return simplified ``W:H`` from pixel dimensions (gcd-reduced, no preset snapping)."""
    w, h = int(width), int(height)
    if w <= 0 or h <= 0:
        return "1:1"
    g = gcd(w, h)
    return f"{w // g}:{h // g}"


def normalize_ratio_string(text: str) -> str:
    """Parse ``W:H`` labels; strip parenthetical suffixes like ``(Widescreen)``."""
    raw = (text or "").strip()
    if not raw:
        return ""
    if raw.lower() == "auto":
        return "auto"
    if "(" in raw:
        raw = raw.split("(", 1)[0].strip()
    if ":" not in raw:
        return raw
    left, right = raw.split(":", 1)
    try:
        w, h = int(left.strip()), int(right.strip())
    except ValueError:
        return raw
    if w <= 0 or h <= 0:
        return raw
    return f"{w}:{h}"


MY_CATEGORY = "\ud83d\udc11 MieNodes/\ud83d\udc11 Common"


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
