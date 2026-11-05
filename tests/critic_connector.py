"""Shared test helper: the continuity critic is a second reader.

The scripted connectors in this suite serve queued replies POSITIONALLY:
reply 1 answers stage 1, reply 2 answers stage 2, and so on. The
continuity critic (2026-11-05) inserts one extra provider call in the
middle of a pipeline — a call that is NOT a pipeline stage — which would
otherwise eat the next stage's scripted reply and shift the whole
sequence.

The connectors therefore route critic calls (recognizable by their
system prompt) into their OWN queue, so a scripted pipeline keeps its
order while the critic still gets a say.
"""
from __future__ import annotations

import json

CRITIC_SYSTEM_PREFIX = "You are a SKEPTICAL QA reviewer"

NO_FINDINGS = '{"findings": []}'


def is_critic_call(messages) -> bool:
    """True when this provider call is the continuity critic."""
    if not messages:
        return False
    system = (messages[0] or {}).get("content") or ""
    return str(system).startswith(CRITIC_SYSTEM_PREFIX)


def critic_reply(findings=()) -> str:
    """A critic reply in the critic's own protocol."""
    return json.dumps({"findings": list(findings)}, ensure_ascii=False)
