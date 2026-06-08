/**
 * Frontend-only annotation nodes: SimpleTextNode and RichTextNode.
 *
 * Each node is a pure canvas annotation. It has no inputs, no outputs, and
 * no widget area on the node body — the body itself is the rendered text.
 * The text content lives in `this.properties.text` (auto-serialized into
 * the workflow JSON) and is edited via a small modal that opens on
 * double-click (or automatically the first time the node is added).
 *
 * SimpleText: drawn on the LiteGraph Canvas with `draw(ctx)`, called
 *             from a global LGraphCanvas.drawNode hook so the LiteGraph
 *             shell stays transparent. Mirrors rgthree-comfy`s Label.
 * RichText:   injected as an HTML DOM widget so we can render full Markdown
 *             (via `marked`) sanitized with `DOMPurify` and auto-grow the
 *             node height to fit the rendered content.
 */

import { app } from "../../scripts/app.js";

const CATEGORY = "🐑 MieNodes/🐑 Extra";

// Markdown rendering dependencies are vendored under js/lib/ so the extension
// has no external network requirements. `import.meta.url` resolves to the URL
// this script was loaded from, regardless of how ComfyUI mounts the
// extension's web directory.
const LIB_BASE = new URL("./lib/", import.meta.url).href;
const MARKED_URL = `${LIB_BASE}marked.min.js`;
const PURIFY_URL = `${LIB_BASE}purify.min.js`;

// ---------------------------------------------------------------------------
// Shared: roundRect (Canvas helper) and the text-editor modal
// ---------------------------------------------------------------------------

function roundRect(ctx, x, y, w, h, r) {
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

let _editorStylesInjected = false;
function injectEditorStyles() {
  if (_editorStylesInjected) return;
  _editorStylesInjected = true;
  const style = document.createElement("style");
  style.textContent = `
    .mie-text-editor-mask {
      position: fixed; inset: 0; z-index: 99999;
      background: rgba(0, 0, 0, 0.55);
      display: flex; align-items: center; justify-content: center;
      backdrop-filter: blur(2px);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .mie-text-editor-card {
      width: 600px; max-width: 92vw; height: 70vh; max-height: 640px;
      background: #1f2937; color: #e5e7eb;
      border-radius: 10px; box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
      display: flex; flex-direction: column; overflow: hidden;
    }
    .mie-text-editor-header {
      display: flex; align-items: center; justify-content: space-between;
      padding: 12px 16px; border-bottom: 1px solid #374151;
    }
    .mie-text-editor-title { font-weight: 600; font-size: 14px; }
    .mie-text-editor-hint { color: #9ca3af; font-size: 12px; }
    .mie-text-editor-close {
      background: none; border: none; color: #9ca3af; font-size: 22px;
      cursor: pointer; line-height: 1; padding: 0 4px;
    }
    .mie-text-editor-close:hover { color: #e5e7eb; }
    .mie-text-editor-card textarea {
      flex: 1; resize: none; border: none; outline: none;
      background: #111827; color: #e5e7eb;
      padding: 14px 16px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 13px; line-height: 1.55;
    }
    .mie-text-editor-actions {
      display: flex; gap: 8px; justify-content: flex-end;
      padding: 10px 16px; border-top: 1px solid #374151; background: #1f2937;
    }
    .mie-text-editor-actions button {
      padding: 6px 14px; border-radius: 6px; border: 1px solid #4b5563;
      background: #374151; color: #e5e7eb; cursor: pointer; font-size: 13px;
    }
    .mie-text-editor-actions button:hover { background: #4b5563; }
    .mie-text-editor-save {
      background: #2563eb !important; border-color: #2563eb !important; color: #fff !important;
    }
    .mie-text-editor-save:hover { background: #1d4ed8 !important; }
    .mie-anno-empty-hint {
      display: flex; align-items: center; justify-content: center;
      width: 100%; height: 100%; box-sizing: border-box;
      color: rgba(229, 231, 235, 0.45);
      font-style: italic;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif;
      font-size: 13px; text-align: center;
      pointer-events: none; user-select: none;
    }
    .mie-anno-empty-hint .mie-anno-cta {
      display: inline-block; margin-left: 6px; padding: 1px 8px;
      border: 1px solid rgba(229, 231, 235, 0.25); border-radius: 4px;
      color: rgba(229, 231, 235, 0.7);
    }
    .mie-text-editor-fields:empty { display: none; }
    .mie-text-editor-fields {
      display: flex; flex-direction: column; gap: 8px;
      padding: 12px 16px; border-top: 1px solid #374151;
      background: #1f2937; max-height: 35vh; overflow-y: auto;
    }
    .mie-field-row { display: grid; grid-template-columns: 90px 1fr; gap: 10px; align-items: center; }
    .mie-field-row:has(.mie-field-boolean) { grid-template-columns: 1fr; }
    /* Push the boolean checkbox + label to the right edge of the row, */
    /* matching the convention of placing toggles on the trailing side. */
    .mie-field-row:has(.mie-field-boolean) .mie-field-ctrl {
      justify-content: flex-end;
    }
    .mie-field-label { color: #9ca3af; font-size: 12px; text-align: right; }
    .mie-field-ctrl { display: flex; align-items: center; min-width: 0; }
    .mie-field-number {
      width: 110px; background: #111827; border: 1px solid #4b5563; color: #e5e7eb;
      border-radius: 4px; padding: 4px 8px; font-size: 12px; outline: none;
    }
    .mie-field-number:focus { border-color: #2563eb; }
    .mie-field-color-wrap { display: flex; flex-direction: column; gap: 6px; width: 100%; }
    .mie-field-color {
      width: 40px; height: 32px; padding: 0; border: 1px solid #4b5563;
      background: transparent; border-radius: 4px; cursor: pointer; flex-shrink: 0;
    }
    .mie-field-color-palette {
      display: inline-flex; align-items: center; gap: 8px;
      padding: 3px 10px 3px 3px;
      background: #111827; border: 1px solid #4b5563;
      border-radius: 5px; cursor: pointer; user-select: none;
      color: #d1d5db; font-size: 12px; line-height: 1;
      transition: border-color 0.12s ease, background 0.12s ease;
    }
    .mie-field-color-palette:hover {
      border-color: #6b7280; background: #1f2937; color: #f3f4f6;
    }
    .mie-field-color-palette .mie-field-color {
      width: 32px; height: 26px; border-radius: 3px;
    }
    .mie-field-color-text {
      flex: 1; min-width: 0; background: #111827; border: 1px solid #4b5563; color: #e5e7eb;
      border-radius: 4px; padding: 4px 8px; font-family: ui-monospace, monospace; font-size: 12px; outline: none;
    }
    .mie-field-color-text:focus { border-color: #2563eb; }
    .mie-field-segmented {
      display: inline-flex; background: #111827; border: 1px solid #4b5563;
      border-radius: 6px; overflow: hidden;
    }
    .mie-field-segmented button {
      background: transparent; color: #9ca3af; border: none;
      padding: 5px 12px; cursor: pointer; font-size: 12px;
      border-right: 1px solid #4b5563; min-width: 48px;
    }
    .mie-field-segmented button:last-child { border-right: none; }
    .mie-field-segmented button:hover:not(.active) { background: #1f2937; color: #e5e7eb; }
    .mie-field-segmented button.active { background: #2563eb; color: #fff; }
    .mie-field-boolean {
      display: inline-flex; align-items: center; gap: 8px; cursor: pointer;
      user-select: none; color: #e5e7eb; font-size: 13px;
    }
    .mie-field-boolean input { width: 16px; height: 16px; accent-color: #2563eb; cursor: pointer; }
    .mie-field-disabled { opacity: 0.45; }
    .mie-field-disabled input,
    .mie-field-disabled button { pointer-events: none; }
    .mie-field-color-presets {
      display: flex; flex-wrap: wrap; gap: 4px;
      margin-bottom: 6px;
    }
    .mie-field-color-swatch {
      width: 22px; height: 22px; border-radius: 4px;
      border: 1px solid rgba(255,255,255,0.18);
      cursor: pointer; padding: 0;
      transition: transform 0.1s, border-color 0.1s;
    }
    .mie-field-color-swatch:hover { transform: scale(1.12); border-color: rgba(255,255,255,0.45); }
    .mie-field-color-swatch.active {
      border-color: #60a5fa;
      box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.4);
    }
    .mie-field-color-picker-row {
      display: inline-flex; align-items: center; gap: 6px; width: 100%;
    }
  `;
  document.head.appendChild(style);
}

/**
 * Open a modal editor for a multi-line text value, plus optional style
 * fields. The resolved value is:
 *   - `null` if the user cancels
 *   - the new string value (legacy shape) if no `fields` were provided
 *   - `{ text, values }` if `fields` were provided; `values` maps each
 *     field key to its current value (string, number, or boolean)
 */
function openMieTextEditor({ title, hint, value, placeholder, fields = [] }) {
  return new Promise((resolve) => {
    injectEditorStyles();
    const wrap = document.createElement("div");
    wrap.className = "mie-text-editor-mask";
    wrap.innerHTML = `
      <div class="mie-text-editor-card" role="dialog" aria-label="${title}">
        <div class="mie-text-editor-header">
          <div>
            <div class="mie-text-editor-title">${title}</div>
            ${hint ? `<div class="mie-text-editor-hint">${hint}</div>` : ""}
          </div>
          <button class="mie-text-editor-close" type="button" aria-label="close">\u00D7</button>
        </div>
        <textarea spellcheck="false" placeholder="${(placeholder || "").replace(/"/g, "&quot;")}"></textarea>
        <div class="mie-text-editor-fields" data-fields-root></div>
        <div class="mie-text-editor-actions">
          <button type="button" data-act="cancel">\u53D6\u6D88</button>
          <button type="button" data-act="save" class="mie-text-editor-save">\u4FDD\u5B58</button>
        </div>
      </div>
    `;
    document.body.appendChild(wrap);

    const ta = wrap.querySelector("textarea");
    ta.value = value ?? "";

    // Build style field rows. sharedValues mirrors the current form
    // state so disableWhen predicates can reference sibling fields.
    // updateDisabled is re-run on every change.
    const fieldRows = [];
    const sharedValues = {};
    const fieldsRoot = wrap.querySelector("[data-fields-root]");
    for (const field of fields) {
      sharedValues[field.key] = field.value;
    }
    const updateDisabled = () => {
      for (const row of fieldRows) {
        const f = row._field;
        if (f && typeof f.disableWhen === "function") {
          row._setDisabled(!!f.disableWhen(sharedValues));
        }
      }
    };
    for (const field of fields) {
      const row = buildFieldRow(field, (v) => {
        sharedValues[field.key] = v;
        updateDisabled();
      });
      fieldsRoot.appendChild(row);
      fieldRows.push(row);
    }
    updateDisabled();

    const cleanup = (result) => {
      wrap.remove();
      document.removeEventListener("keydown", onKey);
      resolve(result);
    };
    const onSave = () => {
      // When no style fields are configured, keep the legacy "return a
      // string" shape so existing callers (RichText) can keep doing
      // `this.properties.text = result` without changes.
      if (fields.length === 0) {
        cleanup(ta.value);
        return;
      }
      const values = {};
      for (const row of fieldRows) {
        values[row.dataset.fkey] = row._getValue();
      }
      cleanup({ text: ta.value, values });
    };
    const onCancel = () => cleanup(null);
    const onKey = (e) => {
      if (e.key === "Escape") onCancel();
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) onSave();
    };

    wrap.querySelector(".mie-text-editor-close").addEventListener("click", onCancel);
    wrap.querySelector('[data-act="cancel"]').addEventListener("click", onCancel);
    wrap.querySelector('[data-act="save"]').addEventListener("click", onSave);
    document.addEventListener("keydown", onKey);
    // Click outside the card to cancel
    wrap.addEventListener("mousedown", (e) => {
      if (e.target === wrap) onCancel();
    });

    setTimeout(() => {
      ta.focus();
      ta.setSelectionRange(ta.value.length, ta.value.length);
    }, 0);
  });
}

function buildFieldRow(field, onChange) {
  const row = document.createElement("div");
  row.className = "mie-field-row";
  row.dataset.fkey = field.key;
  // Stash the config so updateDisabled can re-evaluate disableWhen later.
  row._field = field;

  const label = document.createElement("div");
  label.className = "mie-field-label";
  label.textContent = field.label;
  row.appendChild(label);

  const ctrl = document.createElement("div");
  ctrl.className = "mie-field-ctrl";

  // Toggle the row's disabled state. The native `disabled` attribute
  // blocks interaction; the CSS class lowers opacity and disables
  // pointer events on remaining interactive children.
  const setDisabled = (disabled) => {
    row.classList.toggle("mie-field-disabled", !!disabled);
    for (const el of row.querySelectorAll("input, button")) {
      el.disabled = !!disabled;
    }
  };
  row._setDisabled = setDisabled;

  switch (field.type) {
    case "number": {
      const input = document.createElement("input");
      input.type = "number";
      input.className = "mie-field-number";
      if (field.min !== undefined) input.min = field.min;
      if (field.max !== undefined) input.max = field.max;
      input.value = field.value;
      input.addEventListener("input", () => {
        const n = Number(input.value);
        onChange?.(Number.isFinite(n) ? n : field.value);
      });
      ctrl.appendChild(input);
      row._getValue = () => {
        const n = Number(input.value);
        return Number.isFinite(n) ? n : field.value;
      };
      break;
    }
    case "color": {
      const w = document.createElement("div");
      w.className = "mie-field-color-wrap";

      const presets = Array.isArray(field.presets) ? field.presets : [];

      // Build color + text inputs first so the swatch click handler can
      // reference them in its closure.
      const pickerRow = document.createElement("div");
      pickerRow.className = "mie-field-color-picker-row";
      // Wrap the native <input type="color"> in a labeled button so the
      // user can see it as a "调色板" (color palette) and not just a tiny
      // swatch. Clicking the button (or the embedded input) opens the
      // system color picker for arbitrary colors.
      const paletteBtn = document.createElement("label");
      paletteBtn.className = "mie-field-color-palette";
      paletteBtn.title = "\u6253\u5F00\u8C03\u8272\u76D8\u9009\u62E9\u4EFB\u610F\u989C\u8272";
      const colorIn = document.createElement("input");
      colorIn.type = "color";
      colorIn.className = "mie-field-color";
      colorIn.value = field.value;
      const paletteLabel = document.createElement("span");
      paletteLabel.textContent = "\u8C03\u8272\u677F";
      paletteBtn.appendChild(colorIn);
      paletteBtn.appendChild(paletteLabel);
      const textIn = document.createElement("input");
      textIn.type = "text";
      textIn.className = "mie-field-color-text";
      textIn.value = field.value;

      let presetRow = null;
      if (presets.length > 0) {
        presetRow = document.createElement("div");
        presetRow.className = "mie-field-color-presets";
        for (const preset of presets) {
          const sw = document.createElement("button");
          sw.type = "button";
          sw.className = "mie-field-color-swatch";
          sw.style.background = preset;
          sw.title = preset;
          if (String(preset).toLowerCase() === String(field.value).toLowerCase()) {
            sw.classList.add("active");
          }
          sw.addEventListener("click", () => {
            colorIn.value = preset;
            textIn.value = preset;
            for (const s of presetRow.querySelectorAll(".mie-field-color-swatch")) {
              s.classList.remove("active");
            }
            sw.classList.add("active");
            onChange?.(preset);
          });
          presetRow.appendChild(sw);
        }
        w.appendChild(presetRow);
      }

      const updateSwatchHighlight = (color) => {
        if (!presetRow) return;
        for (const s of presetRow.querySelectorAll(".mie-field-color-swatch")) {
          s.classList.toggle("active", s.title.toLowerCase() === color.toLowerCase());
        }
      };

      colorIn.addEventListener("input", () => {
        textIn.value = colorIn.value;
        onChange?.(colorIn.value);
        updateSwatchHighlight(colorIn.value);
      });
      textIn.addEventListener("input", () => {
        const v = textIn.value.trim();
        if (/^#[0-9a-f]{6}$/i.test(v)) {
          colorIn.value = v;
          onChange?.(v);
          updateSwatchHighlight(v);
        }
      });
      pickerRow.appendChild(paletteBtn);
      pickerRow.appendChild(textIn);
      w.appendChild(pickerRow);
      ctrl.appendChild(w);
      row._getValue = () => {
        const v = textIn.value.trim();
        return /^#[0-9a-f]{6}$/i.test(v) ? v : field.value;
      };
      break;
    }
    case "segmented": {
      const group = document.createElement("div");
      group.className = "mie-field-segmented";
      const buttons = [];
      for (const opt of field.options) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = opt.label;
        if (opt.value === field.value) btn.classList.add("active");
        btn.addEventListener("click", () => {
          for (const b of buttons) b.classList.remove("active");
          btn.classList.add("active");
          onChange?.(opt.value);
        });
        group.appendChild(btn);
        buttons.push(btn);
      }
      ctrl.appendChild(group);
      row._getValue = () => {
        const i = buttons.findIndex((b) => b.classList.contains("active"));
        return i >= 0 ? field.options[i].value : field.value;
      };
      break;
    }
    case "boolean": {
      const lbl = document.createElement("label");
      lbl.className = "mie-field-boolean";
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = !!field.value;
      const txt = document.createElement("span");
      txt.textContent = field.label;
      lbl.appendChild(cb);
      lbl.appendChild(txt);
      cb.addEventListener("change", () => onChange?.(cb.checked));
      ctrl.appendChild(lbl);
      // Boolean carries its own label; hide the row label.
      label.style.display = "none";
      row._getValue = () => cb.checked;
      break;
    }
    default: {
      const input = document.createElement("input");
      input.type = "text";
      input.className = "mie-field-text";
      input.value = field.value ?? "";
      input.addEventListener("input", () => onChange?.(input.value));
      ctrl.appendChild(input);
      row._getValue = () => input.value;
    }
  }

  row.appendChild(ctrl);
  return row;
}

// ---------------------------------------------------------------------------
// Shared: defaults for annotation nodes
// ---------------------------------------------------------------------------

const ANNOTATION_DEFAULTS = {
  bg_color: "transparent",
  font_color: "#e5e7eb",
  padding: 12,
  border_radius: 8,
  font_family: "sans-serif",
  font_weight: "normal",
  font_style: "normal",
};

function applyAnnotationDefaults(node) {
  // Ensure the property bag is populated even when the workflow JSON is missing
  // some keys (e.g. an older workflow loaded into a newer version).
  for (const [k, v] of Object.entries(ANNOTATION_DEFAULTS)) {
    if (node.properties[k] === undefined) node.properties[k] = v;
  }
  if (node.properties.text === undefined) node.properties.text = "";
}

function configureAnnotationShell(node) {
  // Strip the LiteGraph chrome so the node body is just the content.
  // title_mode is exposed as a getter-only property in some ComfyUI /
  // LiteGraph builds, so a plain assignment throws. Use defineProperty
  // to force-overwrite, falling back to a no-op if even that fails.
  try {
    node.title_mode = LiteGraph.NO_TITLE;
  } catch (_) {
    try {
      Object.defineProperty(node, "title_mode", {
        value: LiteGraph.NO_TITLE,
        writable: true,
        configurable: true,
      });
    } catch (_) { /* leave the default; title bar may still show */ }
  }
  node.collapsable = false;
  node.flags = node.flags || {};
  // No inputs, no outputs, no slots.
  node.inputs = [];
  node.outputs = [];
  // No widgets of our own; let LiteGraph compute a zero widget area.
  node.serialize_widgets = true;
  // Belt-and-suspenders transparency. The real enforcement happens in the
  // LGraphCanvas.prototype.drawNode wrap installed by
  // installMieDrawNodeHook(), which forces these to "transparent" on every
  // frame. We still set them here for the brief window between node creation
  // and the first draw pass.
  node.color = "transparent";
  node.bgcolor = "transparent";

  // LiteGraph draws a drop shadow on every node by default. For a clean
  // floating-annotation look (matching rgthree's `Label`), turn it off.
  node.flags = node.flags || {};
  node.flags.shadow = false;

  // Users can drag the bottom-right resize handle to make the box bigger
  // (handy for laying out annotations alongside other nodes, or for giving
  // RichText extra breathing room around rendered Markdown). The draw
  // methods below enforce a "grow-only" floor: if the user drags the box
  // smaller than the natural content size, the next frame snaps it back up
  // so text is never clipped.
  node.resizable = true;
}

// ---------------------------------------------------------------------------
// Global drawNode hook: re-enforce transparency on every frame.
//
// LiteGraph (and some themes) will sometimes reset `node.bgcolor` /
// `node.color` between onNodeCreated and the actual draw pass. The
// rgthree-comfy `Label` solves this by wrapping LGraphCanvas.prototype.
// drawNode and forcing the colors back to transparent right before the
// shell is drawn. We do the same, but scope the override to the node
// types registered through this extension so we don't disturb anything else.
// Set of node constructors that should render with a fully transparent shell.
// Populated via registerMieTextNodeType() when each annotation node type
// is installed.
const _mieTextNodeCtors = new Set();
function registerMieTextNodeType(nodeType) {
  _mieTextNodeCtors.add(nodeType.prototype.constructor);
}

// Mirrors rgthree-comfy's `Label` drawNode wrap: for our annotation nodes
// we (1) clear the LiteGraph body / border colors to the CSS keyword
// `"transparent"` right before LiteGraph draws the shell, (2) call the
// original drawNode so LiteGraph still does its selection box / etc. (with
// no body / border painted), and (3) call our node's own `draw(ctx)` which
// paints the background rect and the text on top of the now-invisible shell.
// This is the only sequence that survives other extensions / themes that
// "aggressively" reset node.bgcolor between frames -- see the comment in
// rgthree-comfy src_web/comfyui/label.ts.
let _mieDrawNodeHookInstalled = false;
function installMieDrawNodeHook() {
  if (_mieDrawNodeHookInstalled) return;
  if (typeof LGraphCanvas === "undefined" || !LGraphCanvas.prototype?.drawNode) return;
  _mieDrawNodeHookInstalled = true;
  const oldDrawNode = LGraphCanvas.prototype.drawNode;
  LGraphCanvas.prototype.drawNode = function (node, ctx) {
    if (node && _mieTextNodeCtors.has(node.constructor)) {
      // Force a fully transparent shell so the LiteGraph body / border /
      // title-color slot never paint anything, no matter what other code did
      // to the node in between frames.
      node.bgcolor = "transparent";
      node.color = "transparent";
      const v = oldDrawNode.apply(this, arguments);
      // Our custom draw runs AFTER LiteGraph, so we can paint the real
      // background + text on top of the (transparent) shell. If our node
      // type doesn't implement `draw`, just return what oldDrawNode gave us.
      if (typeof node.draw === "function") {
        node.draw(ctx);
      }
      return v;
    }
    return oldDrawNode.apply(this, arguments);
  };
}

// ---------------------------------------------------------------------------
// Curated palette shared by every color picker. The first row of the
// picker is these 12 swatches; below them is the native color input for
// arbitrary values. Hoisted to module scope so both SimpleText and
// RichText editors can share the same swatch row.
const COLOR_PRESETS = [
  "#000000", "#ffffff", "#6b7280", "#1f2937",
  "#ef4444", "#f97316", "#eab308", "#22c55e",
  "#06b6d4", "#3b82f6", "#8b5cf6", "#ec4899",
];

// SimpleTextNode — Canvas self-drawing plain text
// ---------------------------------------------------------------------------

const SIMPLE_DEFAULTS = {
  ...ANNOTATION_DEFAULTS,
  font_size: 14,
  align: "left",
};

let _simpleMeasureCtx = null;
function getMeasureCtx() {
  if (!_simpleMeasureCtx) {
    _simpleMeasureCtx = document.createElement("canvas").getContext("2d");
  }
  return _simpleMeasureCtx;
}

function installSimpleTextBehavior(nodeType) {
  const origOnNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    origOnNodeCreated?.apply(this, arguments);
    applyAnnotationDefaults(this);
    for (const [k, v] of Object.entries({ font_size: 14, align: "left" })) {
      if (this.properties[k] === undefined) this.properties[k] = v;
    }
    configureAnnotationShell(this);
    if (!this.size || this.size[0] < 80 || this.size[1] < 40) {
      this.size = [220, 80];
    }
  };

  // COLOR_PRESETS is hoisted to module scope (above installSimpleTextBehavior)
  // so both SimpleText and RichText editors can share the same swatch row.
  nodeType.prototype.openEditor = async function () {
    const isTransparent = this.properties.bg_color === "transparent";
    const result = await openMieTextEditor({
      title: "\u7F16\u8F91\u6587\u672C (SimpleText)",
      hint: "\u652F\u6301\u591A\u884C\uFF0C\u7528 \\n \u6216\u76F4\u63A5\u56DE\u8F66",
      value: this.properties.text,
      placeholder: "\u8F93\u5165\u8981\u663E\u793A\u7684\u6587\u5B57...",
      fields: [
        { key: "font_size", label: "\u5B57\u53F7", type: "number", min: 8, max: 72, value: this.properties.font_size },
        { key: "font_color", label: "\u6587\u5B57\u989C\u8272", type: "color", value: this.properties.font_color, presets: COLOR_PRESETS },
        { key: "font_weight", label: "\u5B57\u91CD", type: "segmented", options: [
          { value: "normal", label: "\u6B63\u5E38" },
          { value: "bold", label: "\u7C97\u4F53" },
        ], value: this.properties.font_weight || "normal" },
        { key: "font_style", label: "\u5B57\u5F62", type: "segmented", options: [
          { value: "normal", label: "\u6B63\u5E38" },
          { value: "italic", label: "\u503E\u659C" },
        ], value: this.properties.font_style || "normal" },
        { key: "align", label: "\u6392\u5217\u65B9\u5F0F", type: "segmented", options: [
          { value: "left", label: "\u5DE6" },
          { value: "center", label: "\u4E2D" },
          { value: "right", label: "\u53F3" },
        ], value: this.properties.align },
        { key: "bg_color", label: "\u80CC\u666F\u989C\u8272", type: "color", value: isTransparent ? "#1f2937" : this.properties.bg_color, presets: COLOR_PRESETS, disableWhen: (v) => !!v.bg_transparent },
        { key: "bg_transparent", label: "\u80CC\u666F\u900F\u660E", type: "boolean", value: isTransparent },
        { key: "padding", label: "\u5185\u8FB9\u8DDD", type: "number", min: 0, max: 48, value: this.properties.padding },
        { key: "border_radius", label: "\u5706\u89D2", type: "number", min: 0, max: 32, value: this.properties.border_radius },
      ],
    });
    if (!result) return;
    this.properties.text = result.text;
    const v = result.values;
    // Clamp numeric values to their declared ranges to defend against
    // tampered inputs (the picker shows min/max but doesn't enforce).
    this.properties.font_size = Math.max(8, Math.min(72, Number(v.font_size) || 14));
    this.properties.font_color = v.font_color;
    this.properties.font_weight = v.font_weight === "bold" ? "bold" : "normal";
    this.properties.font_style = v.font_style === "italic" ? "italic" : "normal";
    this.properties.align = v.align;
    this.properties.padding = Math.max(0, Math.min(48, Number(v.padding) || 0));
    this.properties.border_radius = Math.max(0, Math.min(32, Number(v.border_radius) || 0));
    this.properties.bg_color = v.bg_transparent ? "transparent" : v.bg_color;
    this.setDirtyCanvas?.(true, true);
  };

  nodeType.prototype.onDblClick = function () {
    this.openEditor();
  };

  nodeType.prototype.onPropertyChanged = function () {
    this.setDirtyCanvas?.(true, true);
  };

  // Right-click menu: prepend an "Edit" item so the user has a fallback
  // discoverability path alongside double-click.
  const _origGetMenuOptions_S = nodeType.prototype.getMenuOptions;
  nodeType.prototype.getMenuOptions = function (canvas) {
    const base = _origGetMenuOptions_S ? _origGetMenuOptions_S.call(this, canvas) : [];
    return [
      { content: "✎  编辑文本 (Edit Text)", callback: () => this.openEditor() },
      null,
      ...base,
    ];
  };

  nodeType.prototype.computeSize = function () {
    const text = String(this.properties.text ?? "").replace(/\\n/g, "\n").replace(/\n+$/, "");
    const lines = text === "" ? [""] : text.split("\n");
    const fontSize = Math.max(Number(this.properties.font_size) || 14, 6);
    const padding = Math.max(Number(this.properties.padding) || 0, 0);
    const fontStyle = this.properties.font_style === "italic" ? "italic" : "normal";
    const fontWeight = this.properties.font_weight === "bold" ? "bold" : "normal";
    const ctx = getMeasureCtx();
    ctx.font = `${fontStyle} ${fontWeight} ${fontSize}px ${this.properties.font_family || "sans-serif"}`;
    let maxW = 0;
    for (const line of lines) {
      const w = ctx.measureText(line).width;
      if (w > maxW) maxW = w;
    }
    const lineH = fontSize * 1.45;
    return [
      Math.max(80, Math.ceil(maxW + padding * 2)),
      Math.max(40, Math.ceil(lines.length * lineH + padding * 2)),
    ];
  };

  nodeType.prototype.draw = function (ctx) {
    // Belt-and-suspenders: even though the global drawNode hook forces
    // these to "transparent" right before LiteGraph paints the shell,
    // some extensions / themes "aggressively" reset the colors between
    // frames. Forcing them here means the only thing that can ever be
    // visible is what we draw in this method. Mirrors rgthree-comfy
    // Label.draw().
    this.color = "transparent";
    this.bgcolor = "transparent";
    ctx.save();
    try {
      if (this.flags?.collapsed) return;
      const raw = String(this.properties.text ?? "");
    const fontSize = Math.max(Number(this.properties.font_size) || 14, 6);
    const padding = Math.max(Number(this.properties.padding) || 0, 0);
    const radius = Math.max(Number(this.properties.border_radius) || 0, 0);
    const fontColor = this.properties.font_color || "#ffffff";
    const bgColor = this.properties.bg_color || "transparent";
    const align = this.properties.align || "left";
    const fontFamily = this.properties.font_family || "sans-serif";
    const fontWeight = this.properties.font_weight === "bold" ? "bold" : "normal";
    const fontStyle = this.properties.font_style === "italic" ? "italic" : "normal";

    // Auto-grow to fit content, but never shrink below the user's manual size
    // for the other axis.
    const [contentW, contentH] = this.computeSize();
    if (this.size[0] < contentW) this.size[0] = contentW;
    if (this.size[1] < contentH) this.size[1] = contentH;

    // Background
    if (bgColor && bgColor !== "transparent") {
      ctx.fillStyle = bgColor;
      roundRect(ctx, 0, 0, this.size[0], this.size[1], radius);
      ctx.fill();
    }

    // Empty state: show a subtle hint to encourage double-click
    if (raw === "") {
      ctx.save();
      ctx.fillStyle = "rgba(255,255,255,0.45)";
      ctx.font = `italic 13px -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("\uD83D\uDC11  \u53CC\u51FB\u8282\u70B9\u7F16\u8F91\u5185\u5BB9  \uD83D\uDC11", this.size[0] / 2, this.size[1] / 2);
      ctx.restore();
      return;
    }

    // Text
    const text = raw.replace(/\\n/g, "\n").replace(/\n+$/, "");
    const lines = text.split("\n");
    ctx.fillStyle = fontColor;
    // Canvas font shorthand: [style] [variant] [weight] [size] [family].
    // Two "normal" tokens in a row is valid (e.g. "normal normal 14px ...").
    ctx.font = `${fontStyle} ${fontWeight} ${fontSize}px ${fontFamily}`;
    const lineH = fontSize * 1.45;
    // Vertical centering: anchor each line by its visual middle so the
    // text block sits on the node's vertical center, regardless of
    // how tall the user has dragged the node. The previous
    // (textBaseline = "top" + yStart = (size - textHeight) / 2) approach
    // was off by roughly 0.225 * fontSize visually because the EM-box top
    // is not the visual top of the glyphs.
    ctx.textBaseline = "middle";
    const yCenter = this.size[1] / 2;
    const yLineOffset = (i) => (i - (lines.length - 1) / 2) * lineH;
    for (let i = 0; i < lines.length; i++) {
      let x = padding;
      ctx.textAlign = "left";
      if (align === "center") { x = this.size[0] / 2; ctx.textAlign = "center"; }
      else if (align === "right") { x = this.size[0] - padding; ctx.textAlign = "right"; }
      ctx.fillText(lines[i], x, yCenter + yLineOffset(i));
    }
    } finally {
      ctx.restore();
    }
  };
}

// ---------------------------------------------------------------------------
// RichTextNode — Markdown rendered as HTML inside the node, auto-grow height
// ---------------------------------------------------------------------------

const RICH_DEFAULTS = {
  ...ANNOTATION_DEFAULTS,
  font_size: 14,
  width: 360,
};

let _mdLibs = null; // { marked, DOMPurify }
function ensureMarkdownLibs() {
  if (_mdLibs) return Promise.resolve(_mdLibs);
  if (_mdLibs && _mdLibs.failed) return Promise.reject(_mdLibs.error);
  return new Promise((resolve, reject) => {
    let pending = 2;
    const onOne = () => { if (--pending === 0) tryAttach(); };
    const tryAttach = () => {
      const marked = window.marked;
      const DOMPurify = window.DOMPurify;
      if (marked && DOMPurify) {
        _mdLibs = { marked, DOMPurify };
        resolve(_mdLibs);
      } else {
        _mdLibs = { failed: true, error: new Error("markdown libs not available") };
        reject(_mdLibs.error);
      }
    };
    const loadScript = (src) => {
      if (document.querySelector(`script[data-mie-md="${src}"]`)) {
        onOne();
        return;
      }
      const s = document.createElement("script");
      s.src = src;
      s.async = true;
      s.dataset.mieMd = src;
      s.onload = onOne;
      s.onerror = () => { _mdLibs = { failed: true, error: new Error("failed to load " + src) }; reject(_mdLibs.error); };
      document.head.appendChild(s);
    };
    loadScript(MARKED_URL);
    loadScript(PURIFY_URL);
    // Safety timeout in case both scripts 404 silently (no onerror fired
    // reliably in some browsers when blocked by extensions).
    setTimeout(() => {
      if (!_mdLibs) tryAttach();
    }, 4000);
  });
}

let _richStylesInjected = false;
function injectRichStyles() {
  if (_richStylesInjected) return;
  _richStylesInjected = true;
  const style = document.createElement("style");
  style.textContent = `
    .mie-rich-frame {
      /* Outer flex container; holds the padding, fills the node body, */
      /* and vertically centers the inner markdown block.         */
      background: transparent;
    }
    .mie-rich-content {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      line-height: 1.6; word-wrap: break-word;
      box-sizing: border-box;
      background: transparent;
    }
    .mie-rich-content h1, .mie-rich-content h2, .mie-rich-content h3,
    .mie-rich-content h4, .mie-rich-content h5, .mie-rich-content h6 {
      margin: 0.6em 0 0.3em; font-weight: 600; line-height: 1.3;
    }
    .mie-rich-content h1 { font-size: 1.6em; }
    .mie-rich-content h2 { font-size: 1.4em; }
    .mie-rich-content h3 { font-size: 1.2em; }
    .mie-rich-content p { margin: 0.4em 0; }
    .mie-rich-content ul, .mie-rich-content ol { margin: 0.4em 0; padding-left: 1.6em; }
    .mie-rich-content li { margin: 0.15em 0; }
    .mie-rich-content code {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      background: rgba(127, 127, 127, 0.18);
      padding: 0.1em 0.35em; border-radius: 3px; font-size: 0.9em;
    }
    .mie-rich-content pre {
      background: rgba(0, 0, 0, 0.35);
      padding: 10px 12px; border-radius: 6px; overflow: auto;
      font-size: 0.85em; line-height: 1.5;
    }
    .mie-rich-content pre code { background: none; padding: 0; font-size: inherit; }
    .mie-rich-content blockquote {
      border-left: 3px solid rgba(127, 127, 127, 0.4);
      padding-left: 0.8em; margin: 0.4em 0; color: rgba(229, 231, 235, 0.85);
    }
    .mie-rich-content table {
      border-collapse: collapse; margin: 0.5em 0; width: 100%;
    }
    .mie-rich-content th, .mie-rich-content td {
      border: 1px solid rgba(127, 127, 127, 0.3);
      padding: 4px 8px; text-align: left;
    }
    .mie-rich-content th { background: rgba(127, 127, 127, 0.15); font-weight: 600; }
    .mie-rich-content a { color: #60a5fa; text-decoration: none; }
    .mie-rich-content a:hover { text-decoration: underline; }
    .mie-rich-content hr { border: none; border-top: 1px solid rgba(127, 127, 127, 0.3); margin: 0.6em 0; }
    .mie-rich-content input[type="checkbox"] { margin-right: 0.3em; }
    .mie-rich-content img { max-width: 100%; height: auto; }
    .mie-rich-empty { color: inherit; }
  `;
  document.head.appendChild(style);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "\u0027": "&#39;",
  }[c]));
}

function installRichTextBehavior(nodeType) {
  injectRichStyles();

  const origOnNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    origOnNodeCreated?.apply(this, arguments);
    applyAnnotationDefaults(this);
    for (const [k, v] of Object.entries({ font_size: 14, width: 360 })) {
      if (this.properties[k] === undefined) this.properties[k] = v;
    }
    configureAnnotationShell(this);
    if (!this.size || this.size[0] < 200 || this.size[1] < 100) {
      this.size = [360, 200];
    }
    // Pre-create the content DOM so the user can see something immediately.
    this._ensureContentElement();
    this._renderContent();
  };

  nodeType.prototype.openEditor = async function () {
    const isTransparent = this.properties.bg_color === "transparent";
    const result = await openMieTextEditor({
      title: "\u7F16\u8F91 Markdown (RichText)",
      hint: "\u652F\u6301 GFM\uFF1A**\u7C97\u4F53** *\u659C\u4F53* `\u4EE3\u7801` \u6807\u9898 \u5217\u8868 \u8868\u683C \u4EFB\u52A1\u5217\u8868 \u5F15\u7528",
      value: this.properties.text,
      placeholder: "# \u6807\u9898\n\n**\u7C97\u4F53** *\u659C\u4F53* `inline code`\n\n- \u5217\u8868\u9879\n- \u5217\u8868\u9879\n\n```js\nconsole.log(\u0027hi\u0027)\n```",
      fields: [
        { key: "font_size", label: "\u5B57\u53F7", type: "number", min: 8, max: 72, value: this.properties.font_size },
        { key: "font_color", label: "\u6587\u5B57\u989C\u8272", type: "color", value: this.properties.font_color, presets: COLOR_PRESETS },
        { key: "font_weight", label: "\u5B57\u91CD", type: "segmented", options: [
          { value: "normal", label: "\u6B63\u5E38" },
          { value: "bold", label: "\u7C97\u4F53" },
        ], value: this.properties.font_weight || "normal" },
        { key: "font_style", label: "\u5B57\u5F62", type: "segmented", options: [
          { value: "normal", label: "\u6B63\u5E38" },
          { value: "italic", label: "\u503E\u659C" },
        ], value: this.properties.font_style || "normal" },
        { key: "bg_color", label: "\u80CC\u666F\u989C\u8272", type: "color", value: isTransparent ? "#1f2937" : this.properties.bg_color, presets: COLOR_PRESETS, disableWhen: (v) => !!v.bg_transparent },
        { key: "bg_transparent", label: "\u80CC\u666F\u900F\u660E", type: "boolean", value: isTransparent },
        { key: "padding", label: "\u5185\u8FB9\u8DDD", type: "number", min: 0, max: 48, value: this.properties.padding },
        { key: "border_radius", label: "\u5706\u89D2", type: "number", min: 0, max: 32, value: this.properties.border_radius },
      ],
    });
    if (!result) return;
    this.properties.text = result.text;
    const v = result.values;
    // Clamp numeric values to their declared ranges to defend against
    // tampered inputs (the picker shows min/max but doesn\u0027t enforce).
    this.properties.font_size = Math.max(8, Math.min(72, Number(v.font_size) || 14));
    this.properties.font_color = v.font_color;
    this.properties.font_weight = v.font_weight === "bold" ? "bold" : "normal";
    this.properties.font_style = v.font_style === "italic" ? "italic" : "normal";
    this.properties.padding = Math.max(0, Math.min(48, Number(v.padding) || 0));
    this.properties.border_radius = Math.max(0, Math.min(32, Number(v.border_radius) || 0));
    this.properties.bg_color = v.bg_transparent ? "transparent" : v.bg_color;
    // Re-render the DOM widget so new font_size / padding / color
    // values take effect on the rendered Markdown immediately.
    this._renderContent();
    this.setDirtyCanvas?.(true, true);
  };

  nodeType.prototype.onDblClick = function () {
    this.openEditor();
  };

  nodeType.prototype.onPropertyChanged = function () {
    this._renderContent();
    this.setDirtyCanvas?.(true, true);
  };

  // Right-click menu: prepend an "Edit" item so the user has a fallback
  // discoverability path alongside double-click.
  const _origGetMenuOptions_R = nodeType.prototype.getMenuOptions;
  nodeType.prototype.getMenuOptions = function (canvas) {
    const base = _origGetMenuOptions_R ? _origGetMenuOptions_R.call(this, canvas) : [];
    return [
      { content: "✎  编辑 Markdown (Edit)", callback: () => this.openEditor() },
      null,
      ...base,
    ];
  };

  nodeType.prototype._ensureContentElement = function () {
    // The outer `el` is a flex column that holds the padding, fills the
    // node body, and vertically centers the inner content. The inner
    // `inner` is a plain block that holds the rendered Markdown. We
    // keep these as two separate elements so the inner's scrollHeight
    // reflects the natural content height (no padding included) and
    // so flex can center it without disturbing the markdown flow.
    if (this._mieContentEl && this._mieInnerEl) {
      return this._mieContentEl;
    }

    const el = document.createElement("div");
    el.className = "mie-rich-frame";
    el.style.boxSizing = "border-box";
    el.style.overflow = "hidden";
    el.style.display = "flex";
    el.style.flexDirection = "column";
    el.style.justifyContent = "center";
    el.style.background = "transparent";

    const inner = document.createElement("div");
    inner.className = "mie-rich-content";
    inner.style.boxSizing = "border-box";
    inner.style.width = "100%";
    inner.style.background = "transparent";
    el.appendChild(inner);

    // Cache BEFORE addDOMWidget: if addDOMWidget throws, the next call
    // still returns the cached element instead of stacking more widgets
    // at different Y positions.
    this._mieContentEl = el;
    this._mieInnerEl = inner;

    // Use the LiteGraph DOM-widget API so the element is properly attached
    // to the node's widget area and survives graph redraws.
    if (typeof this.addDOMWidget === "function") {
      try {
        this._mieDomWidget = this.addDOMWidget("mie_rich", "MIE_RICH", el, {
          serialize: false, hideOnZoom: false,
        });
      } catch (e) {
        console.warn("[MieText] addDOMWidget failed, using stub:", e);
        this._mieDomWidget = { element: el };
      }
    } else {
      // Fallback: attach to the node's graph canvas wrapper. Older
      // ComfyUI versions may not have addDOMWidget.
      this._mieDomWidget = { element: el };
    }

    // The DOM widget sits on top of the node, so the canvas underneath
    // never sees clicks on the node body. Without this, onDblClick is
    // dead for RichText. We catch the native dblclick on the DOM element
    // and route it to openEditor.
    el.addEventListener("dblclick", (ev) => {
      ev.stopPropagation();
      this.openEditor();
    });

    return el;
  };

  nodeType.prototype._renderContent = function () {
    const el = this._ensureContentElement();
    const inner = this._mieInnerEl;
    const raw = String(this.properties.text ?? "");
    const fontSize = Math.max(Number(this.properties.font_size) || 14, 6);
    const padding = Math.max(Number(this.properties.padding) || 0, 0);
    // Padding lives on the outer flex frame so flex centering accounts
    // for it. Typography lives on the inner block so scrollHeight gives
    // us the natural content height without padding being mixed in.
    el.style.padding = `${padding}px`;
    inner.style.fontSize = `${fontSize}px`;
    inner.style.color = this.properties.font_color || "#e5e7eb";

    if (raw.trim() === "") {
      inner.classList.add("mie-rich-empty");
      inner.innerHTML = '<div class="mie-anno-empty-hint">\uD83D\uDC11  \u53CC\u51FB\u8282\u70B9\u7F16\u8F91 Markdown  \uD83D\uDC11</div>';
      return;
    }
    inner.classList.remove("mie-rich-empty");
    if (_mdLibs) {
      try {
        const html = _mdLibs.marked.parse(raw, { breaks: true, gfm: true });
        inner.innerHTML = _mdLibs.DOMPurify.sanitize(html);
        return;
      } catch (e) {
        // fall through to plain text
      }
    }
    inner.textContent = raw;
  };

  nodeType.prototype._measureHeight = function () {
    const inner = this._mieInnerEl || this._ensureContentElement();
    // Measure the inner (markdown) block, NOT the outer flex frame.
    // The inner has no padding so scrollHeight is purely the content
    // height. The outer's padding is added on top by the caller.
    // Reading offsetHeight first forces a synchronous layout so the
    // returned scrollHeight reflects the current width.
    void inner.offsetHeight;
    return inner.scrollHeight;
  };

  nodeType.prototype.draw = function (ctx) {
    // Belt-and-suspenders: keep the LiteGraph shell invisible even
    // if some other extension/theme overrides bgcolor mid-frame.
    // Mirrors rgthree-comfy Label.draw().
    this.color = "transparent";
    this.bgcolor = "transparent";
    ctx.save();
    try {
      if (this.flags?.collapsed) return;

      // Background (drawn on Canvas; the Markdown content sits on top via DOM)
    const bgColor = this.properties.bg_color || "transparent";
    const radius = Math.max(Number(this.properties.border_radius) || 0, 0);
    if (bgColor !== "transparent") {
      ctx.fillStyle = bgColor;
      roundRect(ctx, 0, 0, this.size[0], this.size[1], radius);
      ctx.fill();
    }

    // Lazy-load the markdown libs the first time we actually need them.
    if (!this.properties.text) {
      this._ensureContentElement();
      this._renderContent();
    } else if (!_mdLibs) {
      ensureMarkdownLibs().then(() => {
        this._renderContent();
        this.setDirtyCanvas?.(true, true);
      }).catch(() => {
        // Libraries failed to load; the fallback in _renderContent will
        // display plain text. Trigger a redraw to apply it.
        this._renderContent();
        this.setDirtyCanvas?.(true, true);
      });
    }

    // Sync the outer flex frame's height to the node height so flex
    // centering can put the inner content in the middle of any extra
    // space the user added by dragging the node taller. We set it
    // explicitly (rather than via CSS height: 100%) because the widget
    // container's intrinsic height would otherwise be driven by the
    // content, defeating the purpose of vertical centering.
    const el = this._mieContentEl;
    if (el) {
      el.style.width = `${this.size[0]}px`;
      el.style.height = `${this.size[1]}px`;
    }

    // Auto-fit is "grow only": never shrink below the natural content
    // height (content + 2*padding) so the user can drag the node taller
    // and have the content vertically centered via flex. The previous
    // behavior (always reset to contentH + padding) made manual resize
    // impossible and broke vertical centering for RichText.
    const padding = Math.max(Number(this.properties.padding) || 0, 0);
    const contentH = this._measureHeight();
    const minH = contentH + padding * 2;
    if (this.size[1] < minH) {
      this.size[1] = minH;
      if (el) el.style.height = `${this.size[1]}px`;
    }
    } finally {
      ctx.restore();
    }
  };
}

// ---------------------------------------------------------------------------
// Register the extension
// ---------------------------------------------------------------------------

app.registerExtension({
  name: "MieTextAnnotations",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeData) return;
    if (nodeData.category !== CATEGORY) {
      // Only log on first miss per category to avoid spam
      if (!app.__mieCatWarned) app.__mieCatWarned = new Set();
      if (!app.__mieCatWarned.has(nodeData.category)) {
        app.__mieCatWarned.add(nodeData.category);
        console.log("[MieText] skip", nodeData.name, "category mismatch:", JSON.stringify(nodeData.category), "vs", JSON.stringify(CATEGORY));
      }
      return;
    }
    if (nodeData.name === "SimpleTextNode|Mie") {
      console.log("[MieText] installing SimpleText behavior");
      installSimpleTextBehavior(nodeType);
      registerMieTextNodeType(nodeType);
      installMieDrawNodeHook();
    } else if (nodeData.name === "RichTextNode|Mie") {
      console.log("[MieText] installing RichText behavior");
      installRichTextBehavior(nodeType);
      registerMieTextNodeType(nodeType);
      installMieDrawNodeHook();
    }
  },
});
