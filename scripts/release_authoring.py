#!/usr/bin/env python3
"""Release the three authoring packages.

They are three separate repositories on purpose, but they form a chain -
``mie_prompt`` depends on ``mie_llm`` and ``mie_craft`` depends on both -
so a change to one can break the others.

Keeping three repositories buys one thing: the H3 engine, which is a fork
of upstream and merges it periodically, stays far away from the authoring
code.

Releasing them together was the old rule, and it was a safe one, but it
also meant that a change to ONE package put three identical wheels on
PyPI and wrote "no change" into two version histories. The rule that
actually matters is narrower:

    **A package is released when it changed. A package that did not
    change keeps its version.**

What the chain demands is that the floor a package declares on its
dependency matches what that dependency actually is. ``mie_craft``
imports six modules of ``mie_prompt`` (the storyboard prompts, the H3
prompts, the prompt loader), so a change to those has to reach craft as
well - otherwise pip happily installs a combination that does not work.
A change that stays behind the package boundary - a new prompt family, a
refactor of the inside - only needs the one package.

So the versions do not have to be equal. They have to be CONSISTENT, and
that is what --check now verifies.

This script does the whole train:

    python scripts/release_authoring.py --check
    python scripts/release_authoring.py --bump patch --only prompt
    python scripts/release_authoring.py --publish --only prompt
    python scripts/release_authoring.py --bump patch           # all three
    python scripts/release_authoring.py --publish              # all three

``--only`` takes a package name (``prompt``), an import name
(``mie_prompt``) or a repository name (``ComfyUI-Mie-Prompt``). Without
it, all three are bumped and published together, which is still the
right thing to do when the change reaches across the chain.

The three repositories are expected as siblings of this one, or listed in
the MIE_AUTHORING_REPOS environment variable, separated by the platform
path separator.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys


PACKAGES = (
    ("ComfyUI-Mie-LLM", "mie_llm"),
    ("ComfyUI-Mie-Prompt", "mie_prompt"),
    ("ComfyUI-Mie-Craft", "mie_craft"),
)

#: The distribution names on PyPI.
DIST_NAMES = {
    "mie_llm": "comfyui-mie-llm",
    "mie_prompt": "comfyui-mie-prompt",
    "mie_craft": "comfyui-mie-craft",
}

#: The import name of the distribution each package depends on.
DEP_OF = {
    "comfyui-mie-llm": "mie_llm",
    "comfyui-mie-prompt": "mie_prompt",
    "comfyui-mie-craft": "mie_craft",
}

FLOOR_RE = re.compile(r'"(comfyui-mie-[a-z]+)\s*>=\s*([0-9][^"]*)"')

VERSION_RE = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)
VERSION_ASSIGN_RE = re.compile(r'^__version__\s*=\s*"[^"]*"', re.MULTILINE)


def repo_roots() -> list[pathlib.Path]:
    override = os.environ.get("MIE_AUTHORING_REPOS")
    if override:
        parts = [p for p in override.split(os.pathsep) if p.strip()]
        return [pathlib.Path(p) for p in parts]
    here = pathlib.Path(__file__).resolve().parents[2]
    # The three repositories are not always siblings of this one - on the
    # deployment machine they live on another drive - so the usual roots are
    # tried in turn.
    for base in (here, pathlib.Path("F:/"), pathlib.Path("C:/")):
        roots = [base / name for name, _ in PACKAGES]
        if all(root.is_dir() for root in roots):
            return roots
    print("could not find the three repositories; set MIE_AUTHORING_REPOS to "
          "their paths, separated by %r" % os.pathsep, file=sys.stderr)
    return []


def current_version(root: pathlib.Path) -> str:
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = VERSION_RE.search(text)
    if not match:
        raise SystemExit("%s: no version in pyproject.toml" % root)
    return match.group(1)


def bump(version: str, part: str) -> str:
    pieces = [int(p) for p in version.split(".")]
    while len(pieces) < 3:
        pieces.append(0)
    if part == "major":
        pieces[0] += 1
        pieces[1] = 0
        pieces[2] = 0
    elif part == "minor":
        pieces[1] += 1
        pieces[2] = 0
    else:
        pieces[2] += 1
    return ".".join(str(p) for p in pieces[:3])


def write_version(root: pathlib.Path, version: str) -> None:
    path = root / "pyproject.toml"
    text = path.read_text(encoding="utf-8")
    text = VERSION_RE.sub('version = "%s"' % version, text, count=1)
    path.write_text(text, encoding="utf-8")
    write_package_version(root, version)


def write_package_version(root: pathlib.Path, version: str) -> None:
    """Keep ``__version__`` in the package in step with pyproject.toml.

    Without this the published wheel reports the version it had when the
    train started: ``import mie_llm`` said 0.1.0 while the wheel on PyPI
    was 0.1.1, and nobody noticed because only pip was reading the real one.
    """
    for init in root.glob("*/__init__.py"):
        text = init.read_text(encoding="utf-8")
        if "__version__" not in text:
            continue
        text = VERSION_ASSIGN_RE.sub('__version__ = "%s"' % version, text, count=1)
        init.write_text(text, encoding="utf-8")
        print("  %s -> %s" % (init.relative_to(root), version))


def git(root: pathlib.Path, *args: str, capture: bool = False) -> str:
    result = subprocess.run(
        ["git", "-C", str(root)] + list(args),
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        text=True)
    if result.returncode:
        raise SystemExit("git %s failed in %s" % (" ".join(args), root))
    return (result.stdout or "") if capture else ""


def version_tuple(version: str) -> tuple:
    pieces = []
    for part in str(version).split("."):
        digits = "".join(c for c in part if c.isdigit())
        pieces.append(int(digits) if digits else 0)
    while len(pieces) < 3:
        pieces.append(0)
    return tuple(pieces[:3])


def dependency_floors(root: pathlib.Path) -> dict:
    """The floors this package declares on the other two.

    ``{"comfyui-mie-prompt": "0.2.0"}`` - the version of that
    distribution this package needs at least.
    """
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    return {match.group(1): match.group(2)
            for match in FLOOR_RE.finditer(text)}


def published_versions(dist: str) -> set:
    """The versions of ``dist`` that are on the index.

    The index is whichever one the upload will go to: TWINE_REPOSITORY_URL
    says where, and that is what decides whether a version is "already
    there". A network that cannot be reached is not a reason to block a
    release - the upload will complain if it really is a duplicate.
    """
    index = os.environ.get("TWINE_REPOSITORY_URL", "")
    host = "test.pypi.org" if "test.pypi.org" in index else "pypi.org"
    url = "https://%s/pypi/%s/json" % (host, dist)
    try:
        import json
        import urllib.request
        with urllib.request.urlopen(url, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - offline is not a verdict
        return set()
    return set((payload.get("releases") or {}).keys())


def selected_indexes(only, roots) -> list:
    """Which of the three packages an invocation is about.

    A package answers to its repository name, its import name, its
    distribution name and the short name in between ("prompt").
    """
    if not only:
        return list(range(len(roots)))
    wanted = {str(name).strip().lower() for name in only if str(name).strip()}
    picked = []
    for index, (repo_name, package) in enumerate(PACKAGES):
        dist = DIST_NAMES[package]
        names = {repo_name.lower(), package.lower(), dist.lower(),
                 dist.rsplit("-", 1)[-1]}
        if names & wanted:
            picked.append(index)
    if not picked:
        raise SystemExit("no package matches %s; use prompt, mie_prompt or "
                         "ComfyUI-Mie-Prompt" % ", ".join(sorted(wanted)))
    return picked


def consistency_problems(roots, versions) -> list:
    """Everything that has to hold before the chain may be released.

    Two things, and neither of them is "the three versions are equal" -
    that stopped being the rule when releasing on demand came in:

    * a package must not report a version different from the one it was
      published under (a wheel that lies about itself at runtime), and
    * the floor a package declares on its dependency cannot be above the
      version that dependency has: the chain has to be satisfiable.
    """
    problems = []
    for root, version in zip(roots, versions):
        for init in root.glob("*/__init__.py"):
            text = init.read_text(encoding="utf-8")
            match = VERSION_ASSIGN_RE.search(text)
            if not match:
                continue
            found = re.search(r'"([^"]+)"', match.group(0)).group(1)
            if found != version:
                problems.append("%s says %s, pyproject says %s" % (
                    init.relative_to(root), found, version))
    by_package = {package: version
                  for (_, package), version in zip(PACKAGES, versions)}
    for root, version in zip(roots, versions):
        for dist, floor in dependency_floors(root).items():
            dependency = DEP_OF.get(dist)
            if not dependency or dependency not in by_package:
                continue
            have = by_package[dependency]
            if version_tuple(floor) > version_tuple(have):
                problems.append("%s wants %s>=%s, but that package is at %s"
                                % (root.name, dist, floor, have))
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bump", choices= ("major", "minor", "patch"),
                       help="raise the version of the selected packages")
    group.add_argument("--check", action="store_true",
                       help="verify that the chain is consistent")
    group.add_argument("--publish", action="store_true",
                       help="build and upload the selected distributions")
    parser.add_argument("--tag", default=None,
                        help="the git tag to apply (defaults to v<version>)")
    parser.add_argument("--only", action="append", default=None,
                        metavar="PACKAGE",
                        help="only this package: prompt, mie_prompt or "
                             "ComfyUI-Mie-Prompt. May be repeated. Without "
                             "it, all three are included.")
    args = parser.parse_args()

    roots = repo_roots()
    if not roots:
        return 1
    if len(roots) != len(PACKAGES):
        print("expected %d repositories, got %d"
              % (len(PACKAGES), len(roots)), file=sys.stderr)
        return 1

    versions = [current_version(root) for root in roots]
    picked = selected_indexes(args.only, roots)

    if args.check:
        problems = consistency_problems(roots, versions)
        if problems:
            print("the authoring packages are not consistent:", file=sys.stderr)
            for line in problems:
                print("  " + line, file=sys.stderr)
            return 1
        print("the chain is consistent: " + ", ".join(
            "%s %s" % (package, version)
            for (_, package), version in zip(PACKAGES, versions)))
        return 0

    if args.bump:
        for index in picked:
            root = roots[index]
            version = bump(versions[index], args.bump)
            write_version(root, version)
            # Both files go into one commit: the package's __version__ and
            # pyproject's version must never disagree, not even for a moment
            # between the two commits.
            git(root, "add", "pyproject.toml")
            git(root, "add", "-A", "--", "*.py")
            git(root, "commit", "-m", "chore: release %s" % version)
            print("bumped %s to %s" % (root.name, version))
        return 0

    # publish
    for index in picked:
        root = roots[index]
        name, package = PACKAGES[index]
        version = versions[index]
        dist = DIST_NAMES[package]
        if version in published_versions(dist):
            print("%s %s is already on the index - skipped"
                  % (dist, version))
        else:
            print("building", name)
            subprocess.run([sys.executable, "-m", "build", "--wheel"],
                           cwd=str(root), check=True)
            subprocess.run([sys.executable, "-m", "twine", "upload", "dist/*"],
                           cwd=str(root), check=True)
            print("published", package, version)
        tag = args.tag or ("v" + version)
        existing = git(root, "tag", "--list", tag, capture=True).strip()
        if not existing:
            git(root, "tag", "-a", tag, "-m", tag)
        git(root, "push", "origin", "main")
        if not existing:
            git(root, "push", "origin", tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
