"""Tests for the `opacity` (透明度) widget added to both image watermark nodes.

AddTextWatermarkForImage|Mie and AddNumberWatermarkForImage|Mie both accept a
new required ``opacity`` widget (FLOAT 0..1, default 1.0).

Both implementations render a fully-opaque watermark first, then blend it
against the original image with a per-pixel weight ``coverage * opacity``.
Because of that, for any opacity ``o`` the output is exactly linear:

    out(o) = base + o * (out(1.0) - base)   (pixel-wise)

The schema test guards the widget contract, the zero/one tests pin the
endpoints, and the linearity tests pin the blend math in between.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_plugin_imports import load_plugin_module  # noqa: E402

import torch  # noqa: E402


WATERMARK_NODES = {
    "AddTextWatermarkForImage|Mie": dict(
        text="MieNodes",
        font_size=24,
        position_x=50.0,
        position_y=50.0,
        color_r=255,
        color_g=255,
        color_b=255,
        outline=False,
        outline_width=0,
        align="center",
    ),
    "AddNumberWatermarkForImage|Mie": dict(
        start_number=1,
        position_x=50.0,
        position_y=50.0,
        font_scale=1.0,
        color_r=255,
        color_g=255,
        color_b=255,
        thickness=2,
        outline=False,
        outline_thickness=1,
    ),
}


@pytest.fixture(scope="module")
def plugin():
    return load_plugin_module()


def make_images(batch=1, h=64, w=128, value=0.0):
    # Black background: uint8 quantization is lossless (0 -> 0), so the
    # linearity assertions below hold without quantization ambiguity.
    return torch.full((batch, h, w, 3), value, dtype=torch.float32)


# ------------------------------ schema ----------------------------------- #

@pytest.mark.parametrize("node_name", WATERMARK_NODES)
def test_opacity_widget_schema(plugin, node_name):
    cls = plugin.NODE_CLASS_MAPPINGS[node_name]
    spec = cls.INPUT_TYPES()
    # Backward-compat: opacity is OPTIONAL (not required) so saved workflows
    # that predate this widget still evaluate with the schema default.
    assert "opacity" not in spec.get("required", {}), (
        f"{node_name}: opacity must live under 'optional', not 'required'"
    )
    opacity = spec["optional"]["opacity"]
    assert opacity[0] == "FLOAT"
    opts = opacity[1]
    assert opts["default"] == 1.0
    assert opts["min"] == 0.0
    assert opts["max"] == 1.0
    # step drives the UI slider granularity; pin it so a typo here can't
    # quietly ship a 0.05-step slider.
    assert opts["step"] == 0.01


@pytest.mark.parametrize("node_name", WATERMARK_NODES)
def test_opacity_default_renders_full_watermark(plugin, node_name):
    """Old-workflow regression guard.

    A workflow saved before `opacity` was introduced won't pass the kwarg;
    ComfyUI fills the missing optional slot with the schema default. The
    default must therefore be 1.0, and the rendering must be bit-identical
    to an explicit opacity=1.0 call.
    """
    cls = plugin.NODE_CLASS_MAPPINGS[node_name]
    node = cls()
    images = make_images()
    kwargs = dict(WATERMARK_NODES[node_name])
    default = node.apply_watermark(images, **kwargs)[0]
    explicit = node.apply_watermark(images, opacity=1.0, **kwargs)[0]
    assert torch.equal(default, explicit), (
        f"{node_name}: default opacity must render identically to opacity=1.0"
    )


@pytest.mark.parametrize("node_name", WATERMARK_NODES)
def test_apply_watermark_without_opacity_kwarg(plugin, node_name):
    """Defensive: the function must not require `opacity` as a positional
    argument. Old saved workflows + cache-stale sessions can both reach the
    function with the kwarg absent or set to None.
    """
    cls = plugin.NODE_CLASS_MAPPINGS[node_name]
    node = cls()
    images = make_images()
    kwargs = dict(WATERMARK_NODES[node_name])
    kwargs.pop("opacity", None)  # simulate widget not present in saved workflow
    out = node.apply_watermark(images, **kwargs)[0]
    assert out.shape == images.shape
    # Default behavior = full-opacity watermark = identical to opacity=1.0
    expected = node.apply_watermark(
        images, opacity=1.0, **dict(WATERMARK_NODES[node_name])
    )[0]
    assert torch.equal(out, expected)


@pytest.mark.parametrize("node_name", WATERMARK_NODES)
def test_apply_watermark_with_opacity_none(plugin, node_name):
    """None must be treated as 'use the default' (1.0), not crash."""
    cls = plugin.NODE_CLASS_MAPPINGS[node_name]
    node = cls()
    images = make_images()
    kwargs = dict(WATERMARK_NODES[node_name])
    out = node.apply_watermark(images, opacity=None, **kwargs)[0]
    expected = node.apply_watermark(images, opacity=1.0, **kwargs)[0]
    assert torch.equal(out, expected), (
        f"{node_name}: opacity=None must clamp to the default, not raise"
    )


# ------------------------------ endpoints -------------------------------- #

@pytest.mark.parametrize("node_name", WATERMARK_NODES)
def test_opacity_zero_leaves_image_unchanged(plugin, node_name):
    cls = plugin.NODE_CLASS_MAPPINGS[node_name]
    node = cls()
    images = make_images()
    kwargs = dict(WATERMARK_NODES[node_name])
    out = node.apply_watermark(images, opacity=0.0, **kwargs)[0]
    assert out.shape == images.shape
    assert out.dtype == images.dtype
    assert torch.equal(out, images), "opacity=0 must leave the image untouched"


@pytest.mark.parametrize("node_name", WATERMARK_NODES)
def test_opacity_one_renders_full_watermark(plugin, node_name):
    cls = plugin.NODE_CLASS_MAPPINGS[node_name]
    node = cls()
    images = make_images()
    kwargs = dict(WATERMARK_NODES[node_name])
    out = node.apply_watermark(images, opacity=1.0, **kwargs)[0]
    assert out.shape == images.shape
    assert not torch.equal(out, images), "opacity=1 must actually draw the watermark"
    diff = (out - images).abs()
    assert diff.max().item() > 0.5, "watermark pixels should move toward the text color"


# ----------------------------- blend math -------------------------------- #

@pytest.mark.parametrize("node_name", WATERMARK_NODES)
@pytest.mark.parametrize("opacity", [0.25, 0.5, 0.75])
def test_opacity_blends_linearly(plugin, node_name, opacity):
    cls = plugin.NODE_CLASS_MAPPINGS[node_name]
    node = cls()
    images = make_images()
    kwargs = dict(WATERMARK_NODES[node_name])

    # Use the node's own opacity=0 output as `base`: both pipelines quantize
    # the image to uint8 internally, so the linear blend is exact relative to
    # that baseline rather than to the raw float input tensor.
    base = node.apply_watermark(images, opacity=0.0, **kwargs)[0]
    full = node.apply_watermark(images, opacity=1.0, **kwargs)[0]
    partial = node.apply_watermark(images, opacity=opacity, **kwargs)[0]

    expected = base + opacity * (full - base)
    assert torch.allclose(partial, expected, atol=2.0 / 255.0), (
        f"{node_name}: out({opacity}) must equal base + {opacity}*(out(1) - base)"
    )


@pytest.mark.parametrize("node_name", WATERMARK_NODES)
def test_opacity_clamps_out_of_range(plugin, node_name):
    cls = plugin.NODE_CLASS_MAPPINGS[node_name]
    node = cls()
    images = make_images()
    kwargs = dict(WATERMARK_NODES[node_name])

    over = node.apply_watermark(images, opacity=5.0, **kwargs)[0]
    one = node.apply_watermark(images, opacity=1.0, **kwargs)[0]
    under = node.apply_watermark(images, opacity=-1.0, **kwargs)[0]
    zero = node.apply_watermark(images, opacity=0.0, **kwargs)[0]

    assert torch.equal(over, one), "opacity > 1 must clamp to 1.0"
    assert torch.equal(under, zero), "opacity < 0 must clamp to 0.0"


# --------------------------- regression guards --------------------------- #

@pytest.mark.parametrize("node_name", WATERMARK_NODES)
@pytest.mark.parametrize("opacity", [0.25, 0.5, 0.75])
def test_opacity_blends_linearly_on_colored_background(plugin, node_name, opacity):
    """The black-background linearity test can't catch per-pixel bugs that
    only surface on a non-zero base (e.g. anti-alias edge quantization,
    alpha-channel truncation). Repeat the linear-blend identity on a
    mid-gray background to give the math real partial-coverage pixels
    to work with.
    """
    cls = plugin.NODE_CLASS_MAPPINGS[node_name]
    node = cls()
    images = torch.full((1, 64, 128, 3), 0.5, dtype=torch.float32)
    kwargs = dict(WATERMARK_NODES[node_name])

    base = node.apply_watermark(images, opacity=0.0, **kwargs)[0]
    full = node.apply_watermark(images, opacity=1.0, **kwargs)[0]
    partial = node.apply_watermark(images, opacity=opacity, **kwargs)[0]

    expected = base + opacity * (full - base)
    # 4/255 tolerance covers uint8 round-trip + straight-alpha quantization
    # on edge pixels, which is the regime we care about here.
    assert torch.allclose(partial, expected, atol=4.0 / 255.0), (
        f"{node_name}: out({opacity}) on colored bg must equal "
        f"base + {opacity}*(out(1) - base) within 4/255"
    )


@pytest.mark.parametrize("node_name", WATERMARK_NODES)
def test_opacity_per_image_consistency_in_batch(plugin, node_name):
    """Per-image state in the watermark loop must not leak: every frame in
    a batch must match the same frame rendered alone with its own number
    / text. We compare via [0] to drop the batch-1 leading dim so the
    shapes line up.
    """
    cls = plugin.NODE_CLASS_MAPPINGS[node_name]
    node = cls()
    base_kwargs = dict(WATERMARK_NODES[node_name])

    # Render the batch in one shot (batch=3)
    batch = node.apply_watermark(
        make_images(batch=3), opacity=0.5, **base_kwargs
    )[0]
    assert batch.shape[0] == 3

    # Render each frame individually and compare to the corresponding batch slot.
    # `start_number + i` (number node) cycles 1,2,3 across the batch; the text
    # node always uses the same string, so all three frames should match a
    # single batch=1 render.
    for i in range(3):
        per_frame_kwargs = dict(base_kwargs)
        if node_name.startswith("AddNumber"):
            per_frame_kwargs["start_number"] = base_kwargs["start_number"] + i
        single = node.apply_watermark(
            make_images(batch=1), opacity=0.5, **per_frame_kwargs
        )[0][0]  # [0] drops the batch=1 leading dim
        assert torch.equal(batch[i], single), (
            f"{node_name}: batch frame {i} must match the same frame "
            f"rendered alone (start_number+{i})"
        )
