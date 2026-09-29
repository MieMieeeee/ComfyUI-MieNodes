# -*- coding: utf-8 -*-
"""Tests for the frontend-only annotation nodes in
``nodes/common/text_nodes.py``: SimpleTextNode, RichTextNode,
AboutAuthorNode (Chinese) and AboutAuthorNodeEn (English).

These classes are no-op shells — the Python side just exists so ComfyUI
can list the node types in the menu and round-trip them through workflow
save/load. All visual behavior lives in ``js/textNodes.js``. Tests here
cover the contract that has to hold on the Python side: registration
keys, no-op execution, and the i18n contract (each language variant
points to its own JSON profile).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
COMMON_DIR = PROJECT_DIR / "nodes" / "common"
PROFILES_DIR = PROJECT_DIR / "js" / "profiles"


def _ensure_pkg(fqn: str, path: Path | None = None):
    if fqn in sys.modules:
        return sys.modules[fqn]
    mod = types.ModuleType(fqn)
    if path is not None:
        mod.__path__ = [str(path)]
    mod.__package__ = fqn
    sys.modules[fqn] = mod
    return mod


def _load_file(fqn: str, path: Path):
    if fqn in sys.modules:
        del sys.modules[fqn]
    spec = importlib.util.spec_from_file_location(fqn, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def text_nodes():
    _ensure_pkg("_mienodes_internal", PROJECT_DIR)
    _ensure_pkg("_mienodes_internal.core", PROJECT_DIR / "core")
    _load_file("_mienodes_internal.core.utils", PROJECT_DIR / "core" / "utils.py")
    _ensure_pkg("_mienodes_internal.nodes", PROJECT_DIR / "nodes")
    _ensure_pkg("_mienodes_internal.nodes.common", COMMON_DIR)
    return _load_file(
        "_mienodes_internal.nodes.common.text_nodes",
        COMMON_DIR / "text_nodes.py",
    )


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
def test_about_author_zh_and_en_both_registered(text_nodes):
    """Both AboutAuthorNode variants are listed in the local
    NODE_CLASS_MAPPINGS. The root ``__init__.py`` adds the ``|Mie``
    suffix when copying them into the global registry; that lives in
    a separate test (test_node_contracts.py)."""
    assert "AboutAuthorNode" in text_nodes.NODE_CLASS_MAPPINGS
    assert "AboutAuthorNodeEn" in text_nodes.NODE_CLASS_MAPPINGS
    # English must be a DIFFERENT class object, not a re-export.
    zh = text_nodes.NODE_CLASS_MAPPINGS["AboutAuthorNode"]
    en = text_nodes.NODE_CLASS_MAPPINGS["AboutAuthorNodeEn"]
    assert en is not zh


def test_about_author_display_names_are_distinct(text_nodes):
    """Menu labels are distinct so users can tell the two apart."""
    disp = text_nodes.NODE_DISPLAY_NAME_MAPPINGS
    assert disp["AboutAuthorNode"] == "About Author"
    assert disp["AboutAuthorNodeEn"] == "About Author EN"


def test_annotation_base_is_noop(text_nodes):
    """Each annotation node shares the same no-op shell: empty
    INPUT_TYPES, RETURN_TYPES=(), FUNCTION=noop, category under Mie Extra."""
    for key in (
        "SimpleTextNode",
        "RichTextNode",
        "AboutAuthorNode",
        "AboutAuthorNodeEn",
    ):
        cls = text_nodes.NODE_CLASS_MAPPINGS[key]
        assert cls.INPUT_TYPES() == {"required": {}}
        assert cls.RETURN_TYPES == ()
        assert cls.FUNCTION == "noop"
        # Sanity: the noop() call returns an empty dict (a few extensions
        # check this to distinguish annotation nodes from real ones).
        assert cls().noop() == {}
        assert "MieNodes" in cls.CATEGORY
        assert "Extra" in cls.CATEGORY


# --------------------------------------------------------------------------- #
# Profile i18n contract
# --------------------------------------------------------------------------- #
def test_zh_profile_parses_and_has_required_keys():
    """``author.json`` (Chinese) parses and exposes the keys the JS
    card renderer reads."""
    data = json.loads((PROFILES_DIR / "author.json").read_text(encoding="utf-8"))
    for k in ("name", "handle", "tagline", "avatar", "links"):
        assert k in data, f"author.json missing key: {k}"
    assert isinstance(data["links"], list) and data["links"], "links must be non-empty"
    # Original Chinese card has at least one Chinese-char group label.
    assert any(
        any("\u4e00" <= ch <= "\u9fff" for ch in str(link.get("group", "")))
        for link in data["links"]
    ), "author.json link labels should contain CJK characters"


def test_en_profile_parses_and_has_required_keys():
    """``author_en.json`` (English) parses and exposes the keys the JS
    card renderer reads."""
    data = json.loads((PROFILES_DIR / "author_en.json").read_text(encoding="utf-8"))
    for k in ("name", "handle", "tagline", "avatar", "links"):
        assert k in data, f"author_en.json missing key: {k}"
    assert isinstance(data["links"], list) and data["links"], "links must be non-empty"
    # English card link labels should be ASCII (no CJK).
    for link in data["links"]:
        group = str(link.get("group", ""))
        assert not any("\u4e00" <= ch <= "\u9fff" for ch in group), (
            f"author_en.json link label still contains CJK chars: {group!r}"
        )


def test_en_profile_shares_avatar_with_zh_profile():
    """The avatar path is the same in both profiles — the JS rebases it
    against PROFILES_BASE so a relative "./Summer.png" works regardless
    of language. If the English profile drifted to a different avatar
    the rendered card would no longer match its sibling, so we lock it
    down here."""
    zh = json.loads((PROFILES_DIR / "author.json").read_text(encoding="utf-8"))
    en = json.loads((PROFILES_DIR / "author_en.json").read_text(encoding="utf-8"))
    assert en["avatar"] == zh["avatar"]


def test_en_profile_has_same_link_count_as_zh():
    """Both profiles advertise the same number of link buttons — the card
    layout (vertical stack of buttons) expects a stable count so the
    Chinese and English cards line up when placed next to each other."""
    zh = json.loads((PROFILES_DIR / "author.json").read_text(encoding="utf-8"))
    en = json.loads((PROFILES_DIR / "author_en.json").read_text(encoding="utf-8"))
    assert len(en["links"]) == len(zh["links"])


def test_en_profile_shares_all_urls_with_zh():
    """Same URLs in both profiles — only the labels are translated. A
    copy-paste mistake that drops a URL would otherwise silently remove
    a button from the English card."""
    zh_urls = {str(link["url"]) for link in json.loads(
        (PROFILES_DIR / "author.json").read_text(encoding="utf-8")
    )["links"]}
    en_urls = {str(link["url"]) for link in json.loads(
        (PROFILES_DIR / "author_en.json").read_text(encoding="utf-8")
    )["links"]}
    assert en_urls == zh_urls


# --------------------------------------------------------------------------- #
# js/textNodes.js dispatch (structural — the JS code itself is not
# executed, but we grep the file to make sure both node types are wired
# up so a future refactor doesn't accidentally drop one of them).
# --------------------------------------------------------------------------- #
JS_DISPATCH_NEEDLES = (
    'nodeData.name === "AboutAuthorNode|Mie"',
    'nodeData.name === "AboutAuthorNodeEn|Mie"',
    'AUTHOR_PROFILE_URL_EN',
)


@pytest.mark.parametrize("needle", JS_DISPATCH_NEEDLES)
def test_text_nodes_js_dispatches_both_languages(needle):
    """The JS-side dispatch must keep both languages wired up. A typo or
    accidental rename in js/textNodes.js would silently leave one of
    the variants un-rendered (its LiteGraph shell would draw but the
    profile fetch would never fire)."""
    js_path = PROJECT_DIR / "js" / "textNodes.js"
    text = js_path.read_text(encoding="utf-8")
    assert needle in text, f"js/textNodes.js missing dispatch token: {needle!r}"