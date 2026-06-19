"""Regenerate the derived Bernini T2I prompt (build-time derivation).

The runtime no longer derives ``T2I_A14B_EN_SYS_PROMPT`` from T2V — it loads a
snapshot from ``prompts/bernini/t2i_a14b_en.txt``. That avoids a silent failure
if upstream ever reworded the sentence the old ``.replace()`` targeted (a
no-op replace would silently degrade T2I into the motion-describing T2V).

This script keeps the derivation reproducible: when upstream bytedance/Bernini
updates the T2V prompt or the T2I note, copy them into
``prompts/bernini/t2v_a14b_en.txt`` and ``_t2i_note.txt``, then run this to
regenerate ``t2i_a14b_en.txt``. See prompts/bernini/UPSTREAM.md.

Usage: python scripts/regen_bernini_prompts.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BERNINI = ROOT / "nodes" / "llm" / "prompts" / "bernini"

# The exact sentence upstream T2V contains, and its T2I replacement. If upstream
# rewords the source sentence, update _OLD_LINE here and verify the result.
_OLD_LINE = (
    "4. 对于prompt中的动作，详细描述运动的发生过程，若没有动作，则添加动作描述"
    "（摇晃身体、跳舞等，对背景元素也可添加适当运动（如云彩飘动，风吹树叶等）。"
)
_NEW_LINE = (
    "4. 不要描述运动 / 摄像机运动 / 动作过程，只描写主体和背景的静态状态、姿态、表情、构图等。"
)


def main() -> None:
    note = (BERNINI / "_t2i_note.txt").read_text(encoding="utf-8")
    t2v = (BERNINI / "t2v_a14b_en.txt").read_text(encoding="utf-8")
    if _OLD_LINE not in t2v:
        print("WARNING: the sentence targeted for replacement is not in t2v_a14b_en.txt.")
        print("Upstream may have reworded it — update _OLD_LINE in this script and verify.")
        return
    t2i = note + t2v.replace(_OLD_LINE, _NEW_LINE)
    out = BERNINI / "t2i_a14b_en.txt"
    out.write_text(t2i, encoding="utf-8", newline="\n")
    print(f"regenerated {out.relative_to(ROOT)} ({len(t2i)} chars)")


if __name__ == "__main__":
    main()
