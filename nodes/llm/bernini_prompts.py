# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bernini task-aware prompt-enhancer templates.

Verbatim copy of the system prompts and task templates from
`bernini.prompt_enhancer` (https://github.com/bytedance/Bernini), exposed
here as plain module-level strings so that `BerniniPromptGenerator` can
build a chat request per task type. Data only - no I/O, no LLM client.

The mode-specific response parsing and media-frame handling live in
`bernini_prompt_generator.py`; this module is the prompt library."""

import re

try:
    from _mienodes_internal.nodes.llm.prompts.loader import load_prompt_dict, load_prompt_text
except ImportError:
    from .prompts.loader import load_prompt_dict, load_prompt_text

# Display strings shown in the task_type dropdown. Format: "code - 中文".
# The short code is what routing / JSON_MODE_TASKS / SYSTEM_PROMPTS use,
# extracted via parse_task_code() at call time.
TASK_TYPES = (
    "t2i - 文生图",
    "t2v - 文生视频",
    "i2i - 图像编辑",
    "r2i - 参考主体生图",
    "ri2i (扩展) - 参考图引导图像编辑",
    "i2v - 图生视频",
    "v2v - 视频编辑",
    "mv2v - 多源视频编辑",
    "r2v - 参考图生视频",
    "vi2v - 视频插入参考图",
    "rv2v - 参考图引导视频编辑",
    "vrc2v - 参考内容视频编辑",
    "ads2v - 视频植入视频",
)

# The original short codes, in display order. Useful for iterating
# or for sanity checks against the dropdown contents.
TASK_CODES = (
    "t2i", "t2v", "i2i", "r2i", "ri2i", "i2v", "v2v",
    "mv2v", "r2v", "vi2v", "rv2v", "vrc2v", "ads2v",
)


def parse_task_code(task_type):
    """Extract the short task code from a display string.

    Accepts:
      - new display strings like ``"i2v - 图生视频"``
      - extension display strings like ``"ri2i (扩展) - 参考图引导图像编辑"``
      - the legacy bare code ``"i2v"`` (saved workflows from before the
        bilingual labels)
      - None / empty (passed through unchanged)

    For display strings, splits on the first ``" - "`` and strips an
    optional parenthesized tag (e.g. ``"(扩展)"``) from the leading
    code, returning just the short code. Bare codes pass through
    unchanged.
    """
    if not task_type:
        return task_type
    head = task_type.split(" - ", 1)[0]
    # Strip optional parenthesized suffix like " (扩展)" off the head.
    head = re.sub(r"\s*\(.*?\)\s*", "", head)
    return head.strip()

# Tasks that are expected to return a JSON object with a single
# "rewritten_text" key, which the caller must parse out.
JSON_MODE_TASKS = frozenset({"r2i", "r2v", "rv2v", "vrc2v", "ri2i"})

# --------------------------------------------------------------------------- #
# System prompts per task type
# --------------------------------------------------------------------------- #
SYSTEM_PROMPTS = load_prompt_dict("bernini/system_prompts")

def get_system_prompt_for_task(task_type: str) -> str:
    """Return the system-prompt prefix for `task_type` (default if unknown)."""
    return SYSTEM_PROMPTS.get(task_type, SYSTEM_PROMPTS["default"])

# --------------------------------------------------------------------------- #
# Task prompt templates
# --------------------------------------------------------------------------- #
T2V_A14B_EN_SYS_PROMPT = load_prompt_text("bernini/t2v_a14b_en")

T2I_A14B_EN_SYS_PROMPT = load_prompt_text("bernini/t2i_a14b_en")

R2V_TEMPLATE = load_prompt_text("bernini/r2v")

R2I_TEMPLATE = load_prompt_text("bernini/r2i")

VR2V_TEMPLATE = load_prompt_text("bernini/vr2v")

V2V_TEMPLATE = load_prompt_text("bernini/v2v")

I2I_TEMPLATE = load_prompt_text("bernini/i2i")

I2V_TEMPLATE = load_prompt_text("bernini/i2v")

VI2V_TEMPLATE = load_prompt_text("bernini/vi2v")

ADS2V_TEMPLATE = load_prompt_text("bernini/ads2v")



# --------------------------------------------------------------------------- #
# ri2i - Reference-image-guided image editing (MieNodes extension)
# --------------------------------------------------------------------------- #
# This task is NOT part of the upstream bytedance/Bernini 12-task set; it
# fills the symmetric gap between `i2i` (single source image) and `r2i`
# (reference-driven new image generation). Mirrors the structure of
# `VR2V_TEMPLATE` but for the still-image domain.
RI2I_TEMPLATE = load_prompt_text("bernini/ri2i")
