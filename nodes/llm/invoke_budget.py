# -*- coding: utf-8 -*-
"""Shared LLM invoke with token-budget escalation for the prompt generators.

Reasoning models (MiniMax-M3, DeepSeek-R1, GLM-5.x) emit their think chain
INSIDE the caller's max_tokens budget before the answer. When the chain
consumes the whole budget the reply comes back EMPTY with
``finish_reason='length'`` — the #1 live cause of "H3/Scail2/LTX25
enhance[...]: returned empty after Xs" logs:

  2026-11-04 (H3, MiniMax-M3): ``enhance[i2v_first]`` burned all 8192
  tokens on 51s of thinking and the answer never started.

The budget belongs to the CALLER, so only the caller can fix starvation —
and it can, mechanically: double the budget and retry. A bigger budget
never adds latency (the model stops when the answer is done); it only
removes the ceiling the think chain was hitting. This helper implements
that loop ONCE for every generator's ``_invoke`` so the three copies
cannot drift apart:

  * retry only when the reply is empty AND the connector reports
    ``last_finish_reason == 'length'`` (set by services/llm.py on every
    200 response) — a provider-side empty reply is NOT retried at a
    bigger budget, that would just burn tokens;
  * double until the answer arrives or the ceiling (the widget max,
    32768) is reached;
  * the per-stage log lines are byte-identical to the pre-helper
    generators' ("returned empty after Xs" / "ok in Xs
    response_chars=N"), with one new escalation line between retries.
"""

from __future__ import annotations

import time
from typing import Any, Optional

try:
    from _mienodes_internal.core.utils import mie_log
except ImportError:
    from ...core.utils import mie_log

# The ceiling for auto-escalation: the max_tokens widget maximum shared
# by the H3 / SCAIL-2 / LTX-2.5 generator nodes (min 64, max 32768).
MAX_TOKENS_CEILING = 32768


def invoke_with_budget_escalation(
    connector: Any,
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float,
    seed: Optional[int],
    stage_label: str,
    ceiling: int = MAX_TOKENS_CEILING,
) -> str:
    """One stage call with length-starvation auto-retry. Returns the
    stripped answer, or ``""`` when every attempt came back empty.

    ``stage_label`` is the full log prefix, e.g. ``"H3 enhance[i2v_first]"``
    or ``"Scail2 caption[motion_transfer]"`` — timeout plumbing stays in
    the caller's ``_invoke``, this owns budget + logging only.
    """
    t0 = time.perf_counter()
    budget = int(max_tokens)
    out = connector.invoke(
        messages, seed=seed, temperature=temperature, max_tokens=budget
    )
    # finish_reason='length' + empty = the think chain starved the answer;
    # connectors that never set the attribute (Gemini path, foreign
    # connectors) simply never escalate.
    while (
        not out
        and budget < ceiling
        and getattr(connector, "last_finish_reason", None) == "length"
    ):
        new_budget = min(budget * 2, ceiling)
        mie_log(
            f"{stage_label}: empty answer starved by the think chain "
            f"(finish_reason=length); retrying with max_tokens "
            f"{budget} -> {new_budget}"
        )
        budget = new_budget
        out = connector.invoke(
            messages, seed=seed, temperature=temperature, max_tokens=budget
        )
    elapsed = time.perf_counter() - t0
    model_name = getattr(connector, "model", "?")
    if not out:
        mie_log(
            f"{stage_label}: model={model_name} returned empty after "
            f"{elapsed:.2f}s"
        )
        return ""
    mie_log(
        f"{stage_label}: model={model_name} ok in {elapsed:.2f}s "
        f"response_chars={len(out)}"
    )
    return out.strip()
