"""Numeric helpers and nodes (round-to-multiple, etc.)."""

from __future__ import annotations

import math
from typing import Tuple

MY_CATEGORY = "🐑 MieNodes/🐑 Common"


def _round_half_away_from_zero(x: float) -> int:
    """Round a number to the nearest integer, breaking ties away from zero.

    Mirrors the schoolbook 四舍五入 rule rather than Python's banker's
    rounding, so values like 2.5 -> 3 and -2.5 -> -3.
    """
    if x >= 0:
        return int(math.floor(x + 0.5))
    return -int(math.floor(-x + 0.5))


def _snap_to_multiple(value: float, multiple: float, rounding: str) -> float:
    """Snap ``value`` to the nearest multiple of ``multiple`` (signed).

    ``rounding`` selects the tie-breaking direction:

    * ``"ceil"``  - the result is >= value
    * ``"floor"`` - the result is <= value
    * ``"round"`` - half-away-from-zero rounding (standard 四舍五入)

    A zero or NaN ``multiple`` falls back to ``1`` so the node never raises
    mid-workflow. The result preserves the sign of the requested multiple.
    """
    if not math.isfinite(multiple) or multiple == 0:
        multiple = 1.0
    step = abs(float(multiple))
    sign = 1.0 if multiple >= 0 else -1.0
    ratio = float(value) / step
    if rounding == "ceil":
        snapped = math.ceil(ratio)
    elif rounding == "floor":
        snapped = math.floor(ratio)
    elif rounding == "round":
        snapped = _round_half_away_from_zero(ratio)
    else:
        raise ValueError(f"Unknown rounding mode: {rounding!r}")
    return float(snapped) * step * sign


class RoundToMultiple:
    """Snap a number to a multiple of another number (ceil / floor / round)."""

    ROUNDING_MODES = ("round", "floor", "ceil")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -1.0e15,
                        "max": 1.0e15,
                        "step": 0.01,
                        "tooltip": "The number to snap. Accepts int and float.",
                    },
                ),
                "multiple": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.000001,
                        "max": 1.0e15,
                        "step": 0.01,
                        "tooltip": "The base unit to snap to (e.g. 8, 16, 32 for image dims).",
                    },
                ),
                "rounding": (
                    list(cls.ROUNDING_MODES),
                    {
                        "default": "round",
                        "tooltip": (
                            "ceil -> snap up, floor -> snap down, "
                            "round -> nearest multiple (half away from zero)."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("float", "int")
    FUNCTION = "round"
    CATEGORY = MY_CATEGORY

    def round(self, value: float, multiple: float, rounding: str) -> Tuple[float, int]:
        snapped = _snap_to_multiple(value, multiple, rounding)
        return (snapped, int(snapped))