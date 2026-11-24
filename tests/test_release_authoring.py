"""The release rule, checked.

Releasing the three authoring packages on demand means the versions do
NOT have to be equal any more. What has to hold is the chain: the floor
a package declares on its dependency cannot be above the version that
dependency has, and a package must not report a version different from
the one it was published under.

These tests use throwaway package directories - nothing here touches the
real repositories.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "release_authoring.py"
_spec = importlib.util.spec_from_file_location("release_authoring", SCRIPT)
release = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release)


def _package(tmp_path, repo, package, version, *, deps=(), init_version=None):
    """A package directory: pyproject.toml plus the package's __init__."""
    root = tmp_path / repo
    pkg = root / package
    pkg.mkdir(parents=True)
    dist = "comfyui-mie-" + package.replace("mie_", "")
    lines = ["[project]", 'name = "%s"' % dist,
             'version = "%s"' % version, "dependencies = ["]
    for dependency, floor in deps:
        lines.append('    "%s>=%s",' % (dependency, floor))
    lines += ["]", "", "[tool.setuptools]", 'packages = ["%s"]' % package]
    (root / "pyproject.toml").write_text("\n".join(lines) + "\n",
                                         encoding="utf-8")
    (pkg / "__init__.py").write_text(
        '__version__ = "%s"\n' % (init_version or version), encoding="utf-8")
    return root


def _three(tmp_path, versions, deps=None):
    """The three packages, in the order the script expects them."""
    deps = deps or {}
    repos = ("ComfyUI-Mie-LLM", "ComfyUI-Mie-Prompt", "ComfyUI-Mie-Craft")
    packages = ("mie_llm", "mie_prompt", "mie_craft")
    roots = []
    for repo, package, version in zip(repos, packages, versions):
        roots.append(_package(tmp_path, repo, package, version,
                               deps=deps.get(repo, ())))
    return roots


def test_the_chain_is_consistent_when_the_versions_match(tmp_path):
    roots = _three(tmp_path, ["0.2.0", "0.2.0", "0.2.0"])
    versions = [release.current_version(root) for root in roots]
    assert release.consistency_problems(roots, versions) == []


def test_different_versions_are_fine_now(tmp_path):
    """Releasing on demand: only the package that changed moves."""
    roots = _three(tmp_path, ["0.2.0", "0.2.1", "0.2.0"])
    versions = [release.current_version(root) for root in roots]
    assert release.consistency_problems(roots, versions) == []


def test_a_floor_above_the_dependency_is_a_problem(tmp_path):
    """craft asks for prompt>=0.3.0, but prompt is at 0.2.1."""
    roots = _three(tmp_path, ["0.2.0", "0.2.1", "0.2.0"], deps={
        "ComfyUI-Mie-Craft": (("comfyui-mie-prompt", "0.3.0"),),
    })
    versions = [release.current_version(root) for root in roots]
    problems = release.consistency_problems(roots, versions)
    assert problems, "the unsatisfiable floor must be reported"
    assert "comfyui-mie-prompt" in problems[0]


def test_a_package_that_lies_about_its_version_is_a_problem(tmp_path):
    """The wheel says 0.2.0, pyproject says 0.2.1."""
    roots = _three(tmp_path, ["0.2.0", "0.2.0", "0.2.0"])
    init = roots[1] / "mie_prompt" / "__init__.py"
    init.write_text('__version__ = "0.1.9"\n', encoding="utf-8")
    versions = [release.current_version(root) for root in roots]
    problems = release.consistency_problems(roots, versions)
    assert problems, "the drift must be reported"
    assert "0.1.9" in problems[0]


def test_the_floors_are_read_from_the_pyproject(tmp_path):
    roots = _three(tmp_path, ["0.2.0", "0.2.1", "0.2.0"], deps={
        "ComfyUI-Mie-Craft": (("comfyui-mie-prompt", "0.2.1"),
                              ("comfyui-mie-llm", "0.2.0")),
        "ComfyUI-Mie-Prompt": (("comfyui-mie-llm", "0.2.0"),),
    })
    floors = release.dependency_floors(roots[2])
    assert floors == {"comfyui-mie-prompt": "0.2.1",
                      "comfyui-mie-llm": "0.2.0"}


def test_versions_compare_as_numbers_not_as_text(tmp_path):
    """0.2.10 is newer than 0.2.9 - a string comparison would say otherwise."""
    assert release.version_tuple("0.2.10") > release.version_tuple("0.2.9")
    assert release.version_tuple("0.3.0") > release.version_tuple("0.2.99")


def test_only_selects_the_package_it_names(tmp_path):
    roots = _three(tmp_path, ["0.2.0", "0.2.0", "0.2.0"])
    assert release.selected_indexes(["prompt"], roots) == [1]
    assert release.selected_indexes(["mie_prompt"], roots) == [1]
    assert release.selected_indexes(["ComfyUI-Mie-Prompt"], roots) == [1]
    assert release.selected_indexes(["craft"], roots) == [2]
    assert release.selected_indexes(None, roots) == [0, 1, 2]
    assert release.selected_indexes(["prompt", "craft"], roots) == [1, 2]


def test_only_refuses_a_package_that_does_not_exist(tmp_path):
    roots = _three(tmp_path, ["0.2.0", "0.2.0", "0.2.0"])
    with pytest.raises(SystemExit):
        release.selected_indexes(["nonsense"], roots)


def test_bump_raises_the_selected_package_only(tmp_path):
    """The other two keep their version - that is the whole point."""
    roots = _three(tmp_path, ["0.2.0", "0.2.0", "0.2.0"])
    assert release.bump("0.2.0", "patch") == "0.2.1"
    assert release.bump("0.2.1", "minor") == "0.3.0"
    assert release.bump("0.2.0", "major") == "1.0.0"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
