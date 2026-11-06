#!/usr/bin/env python3
"""Release the three authoring packages in lockstep.

They are three separate repositories on purpose, but they form a chain -
``mie_prompt`` depends on ``mie_llm`` and ``mie_craft`` depends on both -
so a change to one can break the others. They therefore have to be released
together, with one version number.

Keeping three repositories buys one thing: the H3 engine, which is a fork
of upstream and merges it periodically, stays far away from the authoring
code. That separation is worth the coordination, as long as the
coordination is a command and not a habit.

This script does the whole train:

    python scripts/release_authoring.py --bump patch
    python scripts/release_authoring.py --bump minor --tag v0.2.0
    python scripts/release_authoring.py --check
    python scripts/release_authoring.py --publish

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bump", choices=("major", "minor", "patch"),
                       help="raise the version in all three repositories")
    group.add_argument("--check", action="store_true",
                       help="verify the three versions match")
    group.add_argument("--publish", action="store_true",
                       help="build and upload the three distributions")
    parser.add_argument("--tag", default=None,
                        help="the git tag to apply (defaults to v<version>)")
    args = parser.parse_args()

    roots = repo_roots()
    if not roots:
        return 1
    if len(roots) != len(PACKAGES):
        print("expected %d repositories, got %d"
              % (len(PACKAGES), len(roots)), file=sys.stderr)
        return 1

    versions = [current_version(root) for root in roots]

    if args.check:
        if len(set(versions)) != 1:
            print("the versions differ: %s" % ", ".join(
                "%s=%s" % (root.name, version)
                for root, version in zip(roots, versions)), file=sys.stderr)
            return 1
        # A package that reports a different version than the one it was
        # published under is a wheel that lies about itself at runtime.
        drift = []
        for root, version in zip(roots, versions):
            for init in root.glob("*/__init__.py"):
                text = init.read_text(encoding="utf-8")
                match = VERSION_ASSIGN_RE.search(text)
                if not match:
                    continue
                found = re.search(r'"([^"]+)"', match.group(0)).group(1)
                if found != version:
                    drift.append("%s says %s, pyproject says %s" % (
                        init.relative_to(root), found, version))
        if drift:
            print("the package versions drift from pyproject:", file=sys.stderr)
            for line in drift:
                print("  " + line, file=sys.stderr)
            return 1
        print("the three packages are at %s" % versions[0])
        return 0

    if args.bump:
        version = bump(versions[0], args.bump)
        for root in roots:
            write_version(root, version)
            # Both files go into one commit: the package's __version__ and
            # pyproject's version must never disagree, not even for a moment
            # between the two commits.
            git(root, "add", "pyproject.toml")
            git(root, "add", "-A", "--", "*.py")
            git(root, "commit", "-m", "chore: release %s" % version)
        print("bumped the three packages to %s" % version)
        return 0

    # publish
    if len(set(versions)) != 1:
        print("the versions differ: %s" % versions, file=sys.stderr)
        return 1
    version = versions[0]
    for root, (name, package) in zip(roots, PACKAGES):
        print("building", name)
        subprocess.run([sys.executable, "-m", "build", "--wheel"],
                       cwd=str(root), check=True)
        subprocess.run([sys.executable, "-m", "twine", "upload", "dist/*"],
                       cwd=str(root), check=True)
        tag = args.tag or ("v" + version)
        git(root, "tag", "-a", tag, "-m", tag)
        git(root, "push", "origin", "main")
        git(root, "push", "origin", tag)
        print("published", package, version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
