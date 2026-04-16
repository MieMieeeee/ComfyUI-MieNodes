import tempfile
from pathlib import Path

import torch

from loop import (
    MieLoopCollectImage,
    MieLoopFinalizeImages,
    MieLoopCleanupImages,
    RUNTIME_STORE,
)


def test_offload_images_to_disk_and_finalize_cleans_files(sample_loop_ctx):
    collect = MieLoopCollectImage()
    finalize = MieLoopFinalizeImages()

    img1 = torch.rand(1, 16, 16, 3)
    img2 = torch.rand(1, 16, 16, 3)

    with tempfile.TemporaryDirectory() as td:
        ctx = collect.execute(sample_loop_ctx, img1, True, td)[0]
        ctx = collect.execute(ctx, img2, True, td)[0]
        ref = ctx["collectors"]["image"]["ref"]
        stored = RUNTIME_STORE["collectors"]["image"][ref]
        assert len(stored) == 2
        paths = [Path(x["disk_path"]) for x in stored]
        assert all(p.exists() for p in paths)

        merged = finalize.execute(ctx, True)[0]
        assert isinstance(merged, torch.Tensor)
        assert merged.shape == (2, 16, 16, 3)
        assert ref not in RUNTIME_STORE["collectors"]["image"]
        assert all(not p.exists() for p in paths)


def test_offload_images_to_disk_and_cleanup_cleans_files(sample_loop_ctx):
    collect = MieLoopCollectImage()
    cleanup = MieLoopCleanupImages()

    img1 = torch.rand(1, 16, 16, 3)

    with tempfile.TemporaryDirectory() as td:
        ctx = collect.execute(sample_loop_ctx, img1, True, td)[0]
        ref = ctx["collectors"]["image"]["ref"]
        stored = RUNTIME_STORE["collectors"]["image"][ref]
        paths = [Path(x["disk_path"]) for x in stored]
        assert all(p.exists() for p in paths)

        cleaned_ctx, cleaned = cleanup.execute(ctx)
        assert cleaned is True
        assert cleaned_ctx["collectors"]["image"] == {"ref": None, "count": 0}
        assert ref not in RUNTIME_STORE["collectors"]["image"]
        assert all(not p.exists() for p in paths)

