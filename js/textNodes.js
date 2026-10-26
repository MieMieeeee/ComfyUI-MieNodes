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
 *             In ComfyUI's "Nodes 2.0" renderer the text is rendered by a
 *             DOM widget instead (see the Nodes 2.0 section below).
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

// ---------------------------------------------------------------------------
// Shared: Nodes 2.0 (Vue node renderer) compatibility
// ---------------------------------------------------------------------------
// ComfyUI's "Nodes 2.0" mode — the setting `Comfy.VueNodes.Enabled`, labelled
// "Nodes 2.0" in the settings UI — replaces the LiteGraph canvas chrome with
// Vue DOM components (renderer/extensions/vueNodes). Two things change for
// our annotation nodes:
//
//  1. Interaction. LiteGraph never calls `node.onDblClick` in that mode: the
//     pointer events are consumed by the DOM node element, so the modal
//     editor would never open. The browser still delivers a native
//     `dblclick`, and it bubbles up to the document — so we listen once on
//     the document (capture phase), resolve the Vue node the event happened
//     in, and open its editor. The classic frontend keeps using
//     `node.onDblClick`; the bridge stays inert there.
//
//  2. Layout. The Vue node is sized from its DOM content, and it renders a
//     footer row holding the frontend's node badge ("#12 comfyui_mienodes")
//     pinned to the BOTTOM of the node body. Canvas-drawn content therefore
//     does not work there: the box would not grow with the text and the
//     badge would land on top of it. So in Nodes 2.0 the annotation content
//     is rendered as a DOM widget (which the node box is sized from) and the
//     canvas copy is hidden. The classic frontend keeps the canvas
//     rendering exactly as it is today.

function isVueNodesMode() {
  return !!(typeof window !== "undefined" && window.LiteGraph && window.LiteGraph.vueNodesMode);
}

// Node types whose double-click opens an editor modal.
const MIE_EDITABLE_NODE_TYPES = new Set([
  "SimpleTextNode|Mie",
  "RichTextNode|Mie",
]);

// All annotation node types (they share the Nodes 2.0 handling).
const MIE_ANNOTATION_NODE_TYPES = new Set([
  "SimpleTextNode|Mie",
  "RichTextNode|Mie",
  "AboutAuthorNode|Mie",
  "AboutAuthorNodeEn|Mie",
]);

// Visit every annotation node in the current graph.
function forEachAnnotationNode(fn) {
  const graphs = [];
  if (app?.canvas?.graph) graphs.push(app.canvas.graph);
  if (app?.graph) graphs.push(app.graph);
  const seen = new Set();
  for (const graph of graphs) {
    for (const node of graph?._nodes || []) {
      if (seen.has(node)) continue;
      seen.add(node);
      if (MIE_ANNOTATION_NODE_TYPES.has(node.type)) fn(node);
    }
  }
}

// The user can switch between the classic frontend and Nodes 2.0 at any
// time, and the frontend does not notify extensions about it. The canvas
// does not necessarily repaint on the switch either, so we watch the mode
// flag and re-render the annotation nodes when it flips — otherwise a node
// could keep the presentation of the other renderer (e.g. a hidden DOM
// copy in Nodes 2.0, which would make the node look empty).
function installRendererModeWatcher() {
  if (typeof window === "undefined") return;
  if (window.__mieRendererModeWatcher) return;
  window.__mieRendererModeWatcher = true;
  let last = isVueNodesMode();
  setInterval(() => {
    const vue = isVueNodesMode();
    if (vue === last) return;
    last = vue;
    forEachAnnotationNode((node) => {
      try {
        node._onRendererModeChanged?.();
        node.setDirtyCanvas?.(true, true);
      } catch (e) {
        console.warn("[MieText] renderer mode switch failed:", e);
      }
    });
  }, 500);
}

// Map a Vue node element's `data-node-id` back to its LiteGraph node.
// Subgraph nodes are addressed as "<graphId>:<nodeId>", so compare the last
// segment only.
function findGraphNodeByDomId(rawId) {
  if (!rawId) return null;
  const id = String(rawId).split(":").pop();
  const graphs = [];
  if (app?.canvas?.graph) graphs.push(app.canvas.graph);
  if (app?.graph) graphs.push(app.graph);
  for (const graph of graphs) {
    if (!graph) continue;
    try {
      if (typeof graph.getNodeById === "function") {
        const node = graph.getNodeById(id);
        if (node) return node;
      }
    } catch (_) {
      // fall through to the manual scan below
    }
    const hit = (graph._nodes || []).find((n) => String(n.id) === id);
    if (hit) return hit;
  }
  return null;
}

function installVueDblClickBridge() {
  if (typeof document === "undefined" || typeof window === "undefined") return;
  if (window.__mieVueDblClickBridge) return;
  window.__mieVueDblClickBridge = true;
  document.addEventListener(
    "dblclick",
    (event) => {
      // Classic frontend: LiteGraph calls node.onDblClick itself.
      if (!isVueNodesMode()) return;
      const target = event.target;
      if (!target || typeof target.closest !== "function") return;
      const host = target.closest("[data-node-id]");
      if (!host) return;
      const node = findGraphNodeByDomId(host.getAttribute("data-node-id"));
      if (!node || !MIE_EDITABLE_NODE_TYPES.has(node.type)) return;
      if (typeof node.openEditor !== "function") return;
      node.openEditor();
    },
    true,
  );
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
    this._ensureSimpleElement();
    this._renderSimpleText();
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
    // Re-render the DOM copy right away: in Nodes 2.0 the DOM is what is
    // on screen, and the canvas may not repaint on its own.
    this._renderSimpleText();
    this.setDirtyCanvas?.(true, true);
  };

  nodeType.prototype.onDblClick = function () {
    this.openEditor();
  };

  nodeType.prototype.onPropertyChanged = function () {
    this._renderSimpleText();
    this.setDirtyCanvas?.(true, true);
  };

  // Called when the user switches between the classic frontend and
  // Nodes 2.0: the DOM copy has to follow the renderer.
  nodeType.prototype._onRendererModeChanged = function () {
    this._renderSimpleText();
  };

  // -------------------------------------------------------------------------
  // Nodes 2.0: the text lives in a DOM widget instead of on the canvas.
  //
  // In Nodes 2.0 the node box is sized by the Vue layout from the node's DOM
  // content, and the frontend renders its node badge ("#12 comfyui_mienodes")
  // in a footer row at the bottom of the node body. A canvas-drawn text can
  // do neither: the box would not grow with the text and the badge would sit
  // on top of it. So the text is mounted as a DOM widget there, and the
  // canvas copy is hidden. In the classic frontend nothing changes: the
  // canvas still paints the text and the DOM widget stays hidden.
  // -------------------------------------------------------------------------
  nodeType.prototype._ensureSimpleElement = function () {
    if (this._mieSimpleEl) return this._mieSimpleEl;

    const frame = document.createElement("div");
    frame.className = "mie-simple-frame";
    const text = document.createElement("div");
    text.className = "mie-simple-text";
    frame.appendChild(text);
    this._mieSimpleEl = frame;
    this._mieSimpleTextEl = text;

    if (typeof this.addDOMWidget === "function") {
      try {
        this._mieSimpleWidget = this.addDOMWidget("mie_simple", "MIE_SIMPLE", frame, {
          serialize: false, hideOnZoom: false,
        });
      } catch (e) {
        console.warn("[MieText] addDOMWidget failed, using stub:", e);
        this._mieSimpleWidget = { element: frame };
      }
    } else {
      this._mieSimpleWidget = { element: frame };
    }

    // Collapse the auto-created .dom-widget wrapper height (same fix as
    // RichText) so the widget's height never stacks under the title bar.
    const w = this._mieSimpleWidget;
    if (w && !w.computeSize) {
      w.computeSize = function () { return [0, 0]; };
    }

    return frame;
  };

  nodeType.prototype._renderSimpleText = function () {
    const frame = this._ensureSimpleElement();
    const textEl = this._mieSimpleTextEl;
    if (!textEl) return;

    const vue = isVueNodesMode();
    const raw = String(this.properties.text ?? "");
    const fontSize = Math.max(Number(this.properties.font_size) || 14, 6);
    const padding = Math.max(Number(this.properties.padding) || 0, 0);
    const fontColor = this.properties.font_color || "#ffffff";
    const bgColor = this.properties.bg_color || "transparent";
    const align = this.properties.align || "left";
    const fontFamily = this.properties.font_family || "sans-serif";
    const fontWeight = this.properties.font_weight === "bold" ? "bold" : "normal";
    const fontStyle = this.properties.font_style === "italic" ? "italic" : "normal";
    const radius = Math.max(Number(this.properties.border_radius) || 0, 0);
    const isEmpty = raw.trim() === "";

    // Skip the DOM writes when nothing changed — draw() runs on every
    // canvas repaint.
    const sig = [
      vue, raw, fontSize, padding, fontColor, bgColor, align, fontFamily,
      fontWeight, fontStyle, radius, isEmpty,
    ].join("\u0000");
    if (this._mieSimpleSig === sig) return;
    this._mieSimpleSig = sig;

    // Hidden in the classic frontend (the canvas paints the text there),
    // visible in Nodes 2.0 where the DOM is the only renderer that works.
    frame.style.display = vue ? "flex" : "none";
    frame.style.flexDirection = "column";
    frame.style.justifyContent = "center";
    frame.style.boxSizing = "border-box";
    frame.style.width = "100%";
    // Content-driven height in Nodes 2.0 so the node box grows with the
    // text and the badge row ends up BELOW it instead of on top of it.
    frame.style.height = "auto";
    frame.style.padding = `${padding}px`;
    frame.style.background = bgColor === "transparent" ? "transparent" : bgColor;
    frame.style.borderRadius = `${radius}px`;
    frame.style.overflow = "hidden";
    frame.style.pointerEvents = "none";

    textEl.style.whiteSpace = "pre-wrap";
    textEl.style.wordBreak = "break-word";
    textEl.style.fontFamily = fontFamily;
    textEl.style.lineHeight = "1.45";
    textEl.style.textAlign = align;
    textEl.style.fontSize = isEmpty ? "13px" : `${fontSize}px`;
    textEl.style.fontWeight = isEmpty ? "normal" : fontWeight;
    textEl.style.fontStyle = isEmpty ? "italic" : fontStyle;
    textEl.style.color = fontColor;
    textEl.style.opacity = isEmpty ? "0.45" : "1";
    textEl.textContent = isEmpty
      ? "\uD83D\uDC11  \u53CC\u51FB\u8282\u70B9\u7F16\u8F91\u5185\u5BB9  \uD83D\uDC11"
      : raw.replace(/\\n/g, "\n");
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

  // Natural size of the text itself.
  nodeType.prototype._contentSize = function () {
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

  nodeType.prototype.computeSize = function () {
    return this._contentSize();
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

      // Keep the DOM copy in sync with the current renderer mode — the user
      // can switch between the classic frontend and Nodes 2.0 at any time,
      // and the DOM element survives the switch.
      this._renderSimpleText();

      // Nodes 2.0: the text is rendered by the DOM widget (see
      // _renderSimpleText) because the node box there is sized from the
      // DOM content — canvas text would neither grow the box nor dodge the
      // frontend's node badge.
      if (isVueNodesMode()) return;

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
    /* ComfyUI wraps our element inside a div.dom-widget overlay with the
       default pointer-events: auto. That wrapper absorbs clicks on the
       node body even though the inner .mie-rich-frame is now none,
       so the bottom-right resize handle stays unreachable. Use :has() to
       flip the wrapper to none only when it actually contains a
       .mie-rich-frame direct child (i.e. our RichText widget), so we do
       not disturb DOM widgets from other extensions. */
    .dom-widget:has(> .mie-rich-frame) { pointer-events: none !important; }
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

  // Called when the user switches between the classic frontend and
  // Nodes 2.0: the DOM copy has to follow the renderer.
  nodeType.prototype._onRendererModeChanged = function () {
    this._renderContent();
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
    el.style.pointerEvents = "none";

    const inner = document.createElement("div");
    inner.className = "mie-rich-content";
    inner.style.boxSizing = "border-box";
    inner.style.width = "100%";
    inner.style.background = "transparent";
    inner.style.pointerEvents = "none";
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

    // Collapse the auto-created .dom-widget wrapper height to zero so
    // the new Vue-driven frontend's domWidget layout does not stack
    // this widget's height under the title bar (which would push the
    // node taller on every layout pass -- a runaway feedback loop).
    // The wrapper has pointer-events: none (via the :has() rule in
    // injectRichStyles), so its visual size does not affect clicks.
    // The Markdown band still fills the full node body because the
    // inner .mie-rich-frame is sized explicitly in _renderContent.
    const w = this._mieDomWidget;
    if (w && !w.computeSize) {
      w.computeSize = function () { return [0, 0]; };
    }

    return el;
  };

  // Sync the outer flex frame to the node: size, background and padding.
  //
  // In the classic frontend the frame is pinned to the node size (the
  // Canvas paints the background, the frame only carries the content). In
  // Nodes 2.0 the frame is content-driven instead — the node box is sized
  // from the DOM — so the background moves into the frame and the height is
  // left to the content. That way the box grows with the Markdown and the
  // frontend's node badge row ends up BELOW it.
  nodeType.prototype._syncFrameStyle = function () {
    const el = this._mieContentEl;
    if (!el) return;
    const vue = isVueNodesMode();
    const bgColor = this.properties.bg_color || "transparent";
    const radius = Math.max(Number(this.properties.border_radius) || 0, 0);
    const padding = Math.max(Number(this.properties.padding) || 0, 0);
    el.style.width = vue ? "100%" : `${this.size[0]}px`;
    el.style.height = vue ? "auto" : `${this.size[1]}px`;
    el.style.background = vue ? (bgColor === "transparent" ? "transparent" : bgColor) : "transparent";
    el.style.borderRadius = vue ? `${radius}px` : "0px";
    el.style.padding = `${padding}px`;
  };

  nodeType.prototype._renderContent = function () {
    const el = this._ensureContentElement();
    const inner = this._mieInnerEl;
    const raw = String(this.properties.text ?? "");
    const fontSize = Math.max(Number(this.properties.font_size) || 14, 6);
    inner.style.fontSize = `${fontSize}px`;
    inner.style.color = this.properties.font_color || "#e5e7eb";

    if (raw.trim() === "") {
      inner.classList.add("mie-rich-empty");
      inner.innerHTML = '<div class="mie-anno-empty-hint">\uD83D\uDC11  \u53CC\u51FB\u8282\u70B9\u7F16\u8F91 Markdown  \uD83D\uDC11</div>';
      this._syncFrameStyle();
      return;
    }
    inner.classList.remove("mie-rich-empty");
    if (_mdLibs) {
      try {
        const html = _mdLibs.marked.parse(raw, { breaks: true, gfm: true });
        inner.innerHTML = _mdLibs.DOMPurify.sanitize(html);
        this._syncFrameStyle();
        return;
      } catch (e) {
        // fall through to plain text
      }
    }
    inner.textContent = raw;
    this._syncFrameStyle();
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

      const vue = isVueNodesMode();
      const bgColor = this.properties.bg_color || "transparent";
      const radius = Math.max(Number(this.properties.border_radius) || 0, 0);
      const padding = Math.max(Number(this.properties.padding) || 0, 0);

      // Background. In the classic frontend the Canvas paints it and the
      // Markdown content sits on top via DOM; in Nodes 2.0 the DOM element
      // paints it, because the node box there is sized from the DOM content
      // and the Canvas region would not match it.
      if (!vue && bgColor !== "transparent") {
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

    // Sync the outer flex frame to the node so flex centering can put the
    // inner content in the middle of any extra space the user added by
    // dragging the node taller. In Nodes 2.0 the frame is content-driven
    // instead, so the box grows with the Markdown and the frontend's node
    // badge row lands BELOW it instead of on top of it.
    this._syncFrameStyle();
    const el = this._mieContentEl;

    if (!vue) {
      // Auto-fit is "grow only": never shrink below the natural content
      // height (content + 2*padding) so the user can drag the node taller
      // and have the content vertically centered via flex. The previous
      // behavior (always reset to contentH + padding) made manual resize
      // impossible and broke vertical centering for RichText.
      const contentH = this._measureHeight();
      const minH = contentH + padding * 2;
      if (this.size[1] < minH) {
        this.size[1] = minH;
        if (el) el.style.height = `${this.size[1]}px`;
      }
    }
    } finally {
      ctx.restore();
    }
  };
}

// ---------------------------------------------------------------------------
// AboutAuthorNode / AboutAuthorNodeEn — Read-only structured author card.
//
// Content (name / handle / tagline / avatar / links) is sourced from
// `js/profiles/author.json` (Chinese) or `js/profiles/author_en.json`
// (English), depending on which node type was placed. Each variant
// resolves its own profile URL on first render and caches it; subsequent
// nodes of the same language re-use the cached profile. All fields are
// read-only; the only per-instance state is the theme (Dark / Light /
// Minimal / Leaf), chosen from the right-click menu.
// ---------------------------------------------------------------------------

let _authorStylesInjected = false;
function injectAboutAuthorStyles() {
  if (_authorStylesInjected) return;
  _authorStylesInjected = true;
  const style = document.createElement("style");
  style.textContent = `
    /* --- Card shell ------------------------------------------------------- */
    /* The default card is a vertically centered stack: header on top, links */
    /* below, everything horizontally centered. Banner is the one horizontal */
    /* variant (header on the left, links on the right) for placing at the */
    /* top of a workflow. */
    .mie-author-card {
      box-sizing: border-box;
      width: 100%; height: 100%;
      display: flex; flex-direction: column;
      align-items: stretch;              /* children fill card width; the
                                            links list needs full width so its
                                            buttons fill the card. The header
                                            centers itself via its own rules. */
      padding: 16px 18px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      pointer-events: none;          /* the inner <a>/widget handle hit-test */
      overflow: hidden;
    }
    /* Leaf theme: Animal-Crossing-inspired warm cream card. */
    .mie-author-card.theme-leaf {
      color: #5d4037;
    }
    .theme-leaf .mie-author-link:hover { background: rgba(124, 179, 66, 0.18); }

    /* --- Header: avatar + name + tagline, vertically stacked & centered --- */
    .mie-author-header {
      display: flex; flex-direction: column; align-items: center;
      gap: 4px;
      margin-bottom: 12px;
      pointer-events: auto;
      text-align: center;
    }
    /* Banner theme removed — replaced by Leaf. This slot kept for any future
       /* horizontal-variant theme. */
    .mie-author-avatar {
      width: 72px; height: 72px; flex-shrink: 0;
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 38px; line-height: 1;
      user-select: none;
      margin-bottom: 8px;
      /* Glow ring: a soft halo around the avatar. Each theme overrides the */
      /* halo color via its --author-accent CSS variable. */
      box-shadow: 0 0 0 3px var(--author-accent, #60a5fa),
                  0 0 18px 2px var(--author-glow, rgba(96, 165, 250, 0.35));
    }
    .mie-author-avatar img {
      width: 100%; height: 100%;
      border-radius: 50%; object-fit: cover; display: block;
    }
    .mie-author-text { min-width: 0; }
    .mie-author-name {
      font-size: 18px; font-weight: 700; line-height: 1.25;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .mie-author-handle {
      font-size: 12px; opacity: 0.65; margin-top: 2px;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .mie-author-tagline {
      font-size: 12px; opacity: 0.8; margin-top: 8px; line-height: 1.5;
      max-width: 200px;                   /* constrain so it wraps cleanly */
    }
    .theme-leaf .mie-author-tagline { color: #6d4c41; }   /* warm brown */

    /* --- Links list (Linktree-style big buttons) ------------------------- */
    .mie-author-links {
      display: flex; flex-direction: column; align-items: stretch; gap: 8px;
      margin-top: 4px;
      width: 100%;                        /* buttons fill the card width */
      pointer-events: auto;
    }
    /* Each link is a full-width button: translucent background, thin border,
       generous padding, large radius. Hover deepens the background and
       brightens the border toward the theme accent. Inspired by Linktree /
       creator bio-link cards. */
    .mie-author-link {
      display: flex; align-items: center; justify-content: center;
      gap: 6px;
      padding: 11px 14px;
      border-radius: 12px;
      font-size: 13px; font-weight: 500; line-height: 1.3;
      text-decoration: none;
      border: 1px solid var(--author-link-border, rgba(127, 127, 127, 0.25));
      background: var(--author-link-bg, rgba(127, 127, 127, 0.10));
      transition: background 0.14s ease, border-color 0.14s ease, transform 0.14s ease;
      cursor: pointer;
      width: 100%; box-sizing: border-box;
    }
    .mie-author-link:hover {
      background: var(--author-link-bg-hover, rgba(127, 127, 127, 0.20));
      border-color: var(--author-accent, #60a5fa);
    }
    .mie-author-link-left { min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; text-align: center; }
    .mie-author-link-platform { opacity: 0.6; font-size: 12px; }

    /* --- Theme color variants -------------------------------------------- */
    /* Each theme defines its accent (avatar ring + link hover border) and a
       pair of link background tones (resting / hover) via CSS variables on
       the card root. This keeps the per-theme rules tiny. */
    .mie-author-card.theme-dark {
      color: #e5e7eb;
      --author-accent: #60a5fa;
      --author-glow: rgba(96, 165, 250, 0.35);
      --author-link-bg: rgba(255, 255, 255, 0.06);
      --author-link-bg-hover: rgba(96, 165, 250, 0.20);
      --author-link-border: rgba(255, 255, 255, 0.12);
    }
    .mie-author-card.theme-dark .mie-author-link { color: #e5e7eb; }

    .mie-author-card.theme-light {
      color: #1f2937;
      --author-accent: #2563eb;
      --author-glow: rgba(37, 99, 235, 0.25);
      --author-link-bg: rgba(0, 0, 0, 0.04);
      --author-link-bg-hover: rgba(37, 99, 235, 0.12);
      --author-link-border: rgba(0, 0, 0, 0.10);
    }
    .mie-author-card.theme-light .mie-author-link { color: #1f2937; }

    /* Minimal: no card background, links revert to plain text (no button
       chrome) so the theme stays as light as the name implies. */
    .mie-author-card.theme-minimal {
      color: inherit; padding: 10px 12px;
      --author-accent: #9ca3af;
      --author-glow: transparent;
    }
    .mie-author-card.theme-minimal .mie-author-avatar {
      background: transparent; width: 44px; height: 44px; font-size: 26px;
      box-shadow: none;
    }
    .mie-author-card.theme-minimal .mie-author-link {
      background: transparent; border-color: transparent; padding: 6px 8px;
    }
    .mie-author-card.theme-minimal .mie-author-link:hover { background: transparent; border-color: transparent; }
    .mie-author-card.theme-minimal .mie-author-link { color: inherit; }

    /* Leaf: Animal-Crossing warm cream with grass-green accent. */
    .mie-author-card.theme-leaf {
      color: #5d4037;
      --author-accent: #7cb342;
      --author-glow: rgba(124, 179, 66, 0.40);
      --author-link-bg: rgba(124, 179, 66, 0.08);
      --author-link-bg-hover: rgba(124, 179, 66, 0.20);
      --author-link-border: rgba(124, 179, 66, 0.25);
    }
    .mie-author-card.theme-leaf .mie-author-link { color: #5d4037; }

    /* --- Theme picker modal ---------------------------------------------- */
    /* Custom theme selector — used because this LiteGraph build does not */
    /* render native submenus (has_submenu / submenu fields are ignored), */
    /* and flattening 4 themes into 4 top-level items lets other extensions */
    /* (rgthree) inject between them. A single menu item that opens this */
    /* modal sidesteps both problems. */
    .mie-theme-picker-mask {
      position: fixed; inset: 0; z-index: 99999;
      background: rgba(0, 0, 0, 0.45);
      display: flex; align-items: center; justify-content: center;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif;
    }
    .mie-theme-picker {
      width: 320px; max-width: 92vw;
      background: #1f2937; color: #e5e7eb;
      border-radius: 10px; box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
      padding: 16px;
    }
    .mie-theme-picker-title {
      font-weight: 600; font-size: 14px; margin-bottom: 4px;
    }
    .mie-theme-picker-hint {
      color: #9ca3af; font-size: 12px; margin-bottom: 12px;
    }
    .mie-theme-picker-grid {
      display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
    }
    .mie-theme-option {
      display: flex; flex-direction: column; align-items: center; gap: 6px;
      padding: 12px 8px; border-radius: 8px;
      background: #111827; border: 2px solid transparent;
      color: #e5e7eb; cursor: pointer;
      transition: border-color 0.12s ease, background 0.12s ease;
      font-family: inherit;
    }
    .mie-theme-option:hover { background: #1f2937; border-color: #4b5563; }
    .mie-theme-option.active { border-color: #2563eb; background: #1e293b; }
    .mie-theme-option-swatches {
      display: flex; gap: 4px; height: 28px; width: 100%;
      border-radius: 4px; overflow: hidden;
    }
    .mie-theme-option-label { font-size: 12px; }
  `;
  document.head.appendChild(style);
}

// Module-level cache for the loaded profiles, keyed by absolute URL so the
// Chinese (`author.json`) and English (`author_en.json`) variants don't
// collide. Each URL has its own resolved value + in-flight promise so
// multiple AboutAuthorNode instances — of either language — share one fetch.
const _authorProfiles = new Map();   // url -> resolved profile
const _authorProfilePromises = new Map();  // url -> Promise<profile>

// Default (Chinese) profile used by the legacy AboutAuthorNode|Mie. The
// English AboutAuthorNodeEn|Mie variant passes its own URL.
const AUTHOR_PROFILE_URL = new URL("./profiles/author.json", import.meta.url).href;
const AUTHOR_PROFILE_URL_EN = new URL("./profiles/author_en.json", import.meta.url).href;
// Base for resolving relative avatar paths inside the profile (e.g.
// "./profiles/about_me.png") into absolute URLs the browser can fetch.
const PROFILES_BASE = new URL("./profiles/", import.meta.url).href;

/**
 * Load (and cache) an author profile JSON by absolute URL. Resolves to a
 * normalized profile object even on failure — a minimal fallback so the
 * card never blanks out just because the profile file is missing or the
 * network is down.
 */
function loadAuthorProfile(url) {
  const target = url || AUTHOR_PROFILE_URL;
  if (_authorProfiles.has(target)) return Promise.resolve(_authorProfiles.get(target));
  if (_authorProfilePromises.has(target)) return _authorProfilePromises.get(target);
  const promise = new Promise((resolve) => {
    const fallback = {
      name: "Author", handle: "", tagline: "", avatar: "🐑", links: [],
    };
    const timer = setTimeout(() => {
      // 5s safety timeout: resolve with fallback so callers never hang.
      _authorProfiles.set(target, fallback);
      resolve(fallback);
    }, 5000);
    fetch(target, { cache: "no-cache" })
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error("http " + r.status))))
      .then((data) => {
        clearTimeout(timer);
        const profile = {
          name: String(data?.name ?? fallback.name),
          handle: String(data?.handle ?? ""),
          tagline: String(data?.tagline ?? ""),
          avatar: String(data?.avatar ?? fallback.avatar),
          links: Array.isArray(data?.links) ? data.links : [],
        };
        _authorProfiles.set(target, profile);
        resolve(profile);
      })
      .catch(() => {
        clearTimeout(timer);
        _authorProfiles.set(target, fallback);
        resolve(fallback);
      });
  });
  _authorProfilePromises.set(target, promise);
  return promise;
}

/**
 * Decide whether an avatar string is an image source (URL / data URI) or an
 * emoji/text fallback, and append the right element into `parent`.
 * If an image fails to load, it falls back to the 🐑 emoji.
 */
function renderAvatar(parent, avatar) {
  parent.textContent = "";
  const v = String(avatar ?? "");
  const isImage = (/^https?:\/\//i.test(v) || v.startsWith("/") || v.startsWith("data:image"));
  if (isImage) {
    const img = document.createElement("img");
    img.alt = "";
    img.src = v;
    img.addEventListener("error", () => {
      // Replace the broken image with the default emoji so the header
      // never has an empty avatar slot.
      parent.textContent = "🐑";
    });
    parent.appendChild(img);
  } else {
    parent.textContent = v === "" ? "🐑" : v;
  }
}

/**
 * Open a modal theme picker and resolve to the chosen theme value, or `null`
 * if the user cancels. Used instead of a LiteGraph submenu because this build
 * does not render native submenus (has_submenu is ignored), and flattening
 * themes into top-level items lets other extensions inject between them.
 *
 * Each option shows a 3-swatch preview (card bg / text / link) so the choice
 * is visual, not just textual.
 */
const AUTHOR_THEMES = [
  {
    value: "leaf",
    label: "Leaf / 动森",
    swatches: ["#fef9e7", "#5d4037", "#7cb342"],
  },
  {
    value: "dark",
    label: "Dark / 深色",
    swatches: ["#1f2937", "#e5e7eb", "#60a5fa"],
  },
  {
    value: "light",
    label: "Light / 亮色",
    swatches: ["#f3f4f6", "#1f2937", "#2563eb"],
  },
  {
    value: "minimal",
    label: "Minimal / 极简",
    swatches: ["transparent", "#e5e7eb", "#9ca3af"],
  },
];

function openThemePicker(currentTheme) {
  return new Promise((resolve) => {
    const mask = document.createElement("div");
    mask.className = "mie-theme-picker-mask";
    mask.innerHTML = `
      <div class="mie-theme-picker" role="dialog" aria-label="选择主题">
        <div class="mie-theme-picker-title">🎨  主题 (Theme)</div>
        <div class="mie-theme-picker-hint">选择卡片的展示样式</div>
        <div class="mie-theme-picker-grid"></div>
      </div>
    `;
    document.body.appendChild(mask);

    const grid = mask.querySelector(".mie-theme-picker-grid");
    for (const t of AUTHOR_THEMES) {
      const opt = document.createElement("button");
      opt.type = "button";
      opt.className = "mie-theme-option" + (t.value === currentTheme ? " active" : "");
      const sw = document.createElement("div");
      sw.className = "mie-theme-option-swatches";
      for (const color of t.swatches) {
        const s = document.createElement("div");
        s.style.flex = "1";
        s.style.background = color === "transparent" ? "repeating-conic-gradient(#4b5563 0% 25%, #374151 0% 50%) 50% / 12px 12px" : color;
        sw.appendChild(s);
      }
      opt.appendChild(sw);
      const lbl = document.createElement("div");
      lbl.className = "mie-theme-option-label";
      lbl.textContent = t.label;
      opt.appendChild(lbl);
      opt.addEventListener("click", () => {
        mask.remove();
        document.removeEventListener("keydown", onKey);
        resolve(t.value);
      });
      grid.appendChild(opt);
    }

    const onKey = (e) => {
      if (e.key === "Escape") {
        mask.remove();
        document.removeEventListener("keydown", onKey);
        resolve(null);
      }
    };
    // Click outside the dialog to cancel.
    mask.addEventListener("mousedown", (e) => {
      if (e.target === mask) {
        mask.remove();
        document.removeEventListener("keydown", onKey);
        resolve(null);
      }
    });
    document.addEventListener("keydown", onKey);
  });
}

function installAboutAuthorBehavior(nodeType, profileUrl) {
  injectAboutAuthorStyles();

  const getProfile = () => loadAuthorProfile(profileUrl);

  const origOnNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    origOnNodeCreated?.apply(this, arguments);
    // Theme is the ONLY per-instance writable state — the only thing that
    // needs to survive workflow save/load. All content (name/handle/tagline/
    // avatar/links) is read live from the language-specific profile JSON
    // on every render so updating the plugin (and restarting ComfyUI)
    // propagates the new content to every existing node automatically. This
    // means the workflow file stays tiny ({theme: "..."} only) and every
    // user always sees the content matching their installed MieNodes version.
    if (this.properties.theme === undefined) this.properties.theme = "leaf";
    // Initial size — draw() will grow-only fit to the rendered content.
    if (!this.size || this.size[0] < 200 || this.size[1] < 100) {
      this.size = [260, 160];
    }
    configureAnnotationShell(this);
    this._ensureAuthorCardElement();
    // Drop any legacy content fields serialized by older MieNodes versions,
    // so they don't override the live profile read below.
    for (const k of ["author_name", "author_handle", "author_tagline", "author_avatar", "author_links"]) {
      delete this.properties[k];
    }
    // Content is read live: once the profile resolves, re-render to paint
    // the real data (until then _renderAuthorCard shows the fallback).
    getProfile().then(() => {
      this._renderAuthorCard();
      this.setDirtyCanvas?.(true, true);
    });
    this._renderAuthorCard();
  };

  nodeType.prototype._ensureAuthorCardElement = function () {
    if (this._mieAuthorCard) return this._mieAuthorCard;

    const card = document.createElement("div");
    card.className = "mie-author-card";
    card.style.boxSizing = "border-box";
    card.style.overflow = "hidden";
    card.style.background = "transparent";
    card.style.pointerEvents = "none";
    // Pin the card to the node's top-left corner. LiteGraph's DOM-widget
    // host otherwise offsets the widget horizontally (e.g. past the title
    // bar slot), which makes the cream card + its content slide right
    // while the Canvas-drawn background stays at 0,0 — visibly off-center.
    // Absolute positioning relative to the node wrapper keeps the DOM card
    // aligned with the Canvas background rect on every frame.
    card.style.position = "absolute";
    card.style.left = "0";
    card.style.top = "0";
    this._mieAuthorCard = card;

    // Use the LiteGraph DOM-widget API so the element is properly attached
    // to the node body and survives graph redraws — same pattern as RichText.
    if (typeof this.addDOMWidget === "function") {
      try {
        this._mieAuthorWidget = this.addDOMWidget("mie_author", "MIE_AUTHOR", card, {
          serialize: false, hideOnZoom: false,
        });
      } catch (e) {
        console.warn("[MieAuthor] addDOMWidget failed, using stub:", e);
        this._mieAuthorWidget = { element: card };
      }
    } else {
      this._mieAuthorWidget = { element: card };
    }

    // Collapse the auto-created .dom-widget wrapper height (same fix as
    // RichText: without this the Vue-driven frontend stacks the widget's
    // height under the title bar and the Markdown band misaligns).
    const w = this._mieAuthorWidget;
    if (w && !w.computeSize) {
      w.computeSize = function () { return [0, 0]; };
    }

    return card;
  };

  nodeType.prototype._renderAuthorCard = function () {
    const card = this._ensureAuthorCardElement();
    // Clear children but keep the cached class list base. Rebuild from scratch
    // on every render so theme switches and profile reloads are clean.
    card.className = `mie-author-card theme-${this.properties.theme || "leaf"}`;
    card.textContent = "";

    // Content is read LIVE from the cached profile on every render — never
    // from properties. This is what makes "update profile JSON + restart
    // ComfyUI" propagate to all existing nodes. If the profile hasn't
    // resolved yet (or failed), we fall back to the placeholder.
    const profile = _authorProfiles.get(profileUrl) || {};
    const name = profile.name || "Author";
    const handle = profile.handle || "";
    const tagline = profile.tagline || "";
    const links = profile.links || [];
    // Resolve the avatar: emoji stays as-is; a relative path (e.g.
    // "./Summer.png") is rebased against this extension's URL so
    // the <img src> works regardless of how ComfyUI mounts /js.
    let avatar = profile.avatar || "🐑";
    if (typeof avatar === "string" && avatar.startsWith("./")) {
      avatar = new URL(avatar, PROFILES_BASE).href;
    }

    // --- Header ----------------------------------------------------------
    const header = document.createElement("div");
    header.className = "mie-author-header";

    const avatarEl = document.createElement("div");
    avatarEl.className = "mie-author-avatar";
    renderAvatar(avatarEl, avatar);
    header.appendChild(avatarEl);

    const textEl = document.createElement("div");
    textEl.className = "mie-author-text";
    const nameEl = document.createElement("div");
    nameEl.className = "mie-author-name";
    nameEl.textContent = name;
    textEl.appendChild(nameEl);
    if (handle) {
      const handleEl = document.createElement("div");
      handleEl.className = "mie-author-handle";
      handleEl.textContent = handle.startsWith("@") ? handle : "@" + handle;
      textEl.appendChild(handleEl);
    }
    if (tagline) {
      const taglineEl = document.createElement("div");
      taglineEl.className = "mie-author-tagline";
      taglineEl.textContent = tagline;
      textEl.appendChild(taglineEl);
    }
    header.appendChild(textEl);
    card.appendChild(header);

    // --- Links -----------------------------------------------------------
    if (Array.isArray(links) && links.length > 0) {
      const list = document.createElement("div");
      list.className = "mie-author-links";
      for (const link of links) {
        if (!link || !link.url) continue;
        const a = document.createElement("a");
        a.className = "mie-author-link";
        a.href = String(link.url);
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.title = String(link.url);
        const left = document.createElement("span");
        left.className = "mie-author-link-left";
        left.textContent = String(link.group ?? "");
        if (link.platform) {
          const plat = document.createElement("span");
          plat.className = "mie-author-link-platform";
          plat.textContent = `（${link.platform}）`;
          left.appendChild(plat);
        }
        a.appendChild(left);
        // No arrow icon — the whole button is the clickable affordance,
        // matching the Linktree / creator bio-link convention.
        list.appendChild(a);
      }
      card.appendChild(list);
    }

    // Nodes 2.0: the card has to sit in the normal flow with a
    // content-driven height, so the node box grows with the card and the
    // frontend's node badge row ends up BELOW it.
    if (isVueNodesMode()) {
      card.style.position = "relative";
      card.style.height = "auto";
    }
  };

  // Measure the natural height of the rendered card so draw() can grow the
  // node to fit. Mirrors RichText's _measureHeight pattern: force a sync
  // reflow via offsetHeight, then read scrollHeight. Returns 0 if the card
  // isn't built yet (draw will retry next frame).
  nodeType.prototype._measureAuthorHeight = function () {
    const card = this._mieAuthorCard || this._ensureAuthorCardElement();
    void card.offsetHeight;
    return card.scrollHeight;
  };

  nodeType.prototype.onPropertyChanged = function () {
    this._renderAuthorCard();
    this.setDirtyCanvas?.(true, true);
  };

  // Called when the user switches between the classic frontend and
  // Nodes 2.0: the card has to follow the renderer (in Nodes 2.0 it sits in
  // the normal flow so the node box grows with it).
  nodeType.prototype._onRendererModeChanged = function () {
    this._renderAuthorCard();
  };

  // Theme switcher via right-click menu. This is the only per-instance write.
  // We expose a SINGLE top-level menu item that opens a custom HTML modal
  // picker (openThemePicker). Returning one item (instead of 4 flattened
  // theme items) is deliberate: this LiteGraph build does not render native
  // submenus (has_submenu is ignored), and flattening 4 themes into 4
  // top-level items lets other extensions (rgthree) inject items between
  // them — observed pushing the 4th theme to the bottom of the menu. One
  // item sidesteps both problems.
  const _origGetMenuOptions_A = nodeType.prototype.getMenuOptions;
  nodeType.prototype.getMenuOptions = function (canvas) {
    const base = _origGetMenuOptions_A ? _origGetMenuOptions_A.call(this, canvas) : [];
    return [
      {
        content: "🎨  主题 (Theme)...",
        callback: async () => {
          const picked = await openThemePicker(this.properties.theme || "leaf");
          if (!picked) return;
          this.properties.theme = picked;
          this._renderAuthorCard();
          this.setDirtyCanvas?.(true, true);
        },
      },
      null,
      ...base,
    ];
  };

  nodeType.prototype.draw = function (ctx) {
    // Keep the LiteGraph shell invisible (mirrors rgthree Label.draw()).
    this.color = "transparent";
    this.bgcolor = "transparent";
    ctx.save();
    try {
      if (this.flags?.collapsed) return;

      const theme = this.properties.theme || "dark";
      // Leaf gets a softer, larger radius to match the Animal-Crossing vibe;
      // the others keep the crisp 10px card.
      const radius = theme === "leaf" ? 16 : 10;
      let bgColor = null;
      if (theme === "dark") bgColor = "rgba(31, 41, 55, 0.92)";
      else if (theme === "light") bgColor = "rgba(243, 244, 246, 0.95)";
      else if (theme === "leaf") bgColor = "rgba(254, 249, 231, 0.97)";  // warm cream
      // minimal: bgColor stays null -> fully transparent

      // Width: the node body grows to a fixed target width so the card has
      // room to render. The card element itself is left at width:100% (from
      // its CSS) so it tracks the .dom-widget host wrapper — NEVER set its
      // width in pixels here. Setting card.style.width = this.size[0] made
      // the card wider than its wrapper (LiteGraph computes the wrapper a
      // little narrower than the node to leave room for borders/scrollbars),
      // and since the card is position:absolute inside the wrapper, the
      // overflow spilled 20px to the right, producing the "left empty /
      // right crowded" misalignment.
      const cardW = 260;

      const card = this._mieAuthorCard;
      const vue = isVueNodesMode();
      if (card) {
        // Measure the natural height at the wrapper's current width, then
        // grow the node vertically to fit. Width stays driven by CSS (100%).
        card.style.height = "auto";
        void card.offsetWidth;              // force sync reflow at wrapper width
        const naturalH = card.scrollHeight;
        if (this.size[0] < cardW) this.size[0] = cardW;
        if (vue) {
          // Nodes 2.0: the card keeps its natural height so the node box
          // grows with it and the frontend's node badge row ends up BELOW
          // the card instead of on top of it. It also has to sit in the
          // normal flow — an absolutely positioned card would not make the
          // node box grow at all.
          card.style.position = "relative";
          card.style.height = "auto";
        } else {
          if (this.size[1] < naturalH) this.size[1] = naturalH;
          card.style.position = "absolute";
          card.style.height = `${this.size[1]}px`;
        }
      }

      // In the classic frontend the Canvas paints the themed backdrop under
      // the card; in Nodes 2.0 the card itself carries the themed
      // background (see .mie-author-card) because the node box there is
      // sized from the DOM content.
      if (!vue && bgColor) {
        ctx.fillStyle = bgColor;
        roundRect(ctx, 0, 0, this.size[0], this.size[1], radius);
        ctx.fill();
      }
    } finally {
      ctx.restore();
    }
  };
}

// ---------------------------------------------------------------------------
// Register the extension
// ---------------------------------------------------------------------------

// Nodes 2.0 never calls node.onDblClick, so the editor is opened through a
// document-level dblclick bridge instead (see the Nodes 2.0 section above).
installVueDblClickBridge();
// The frontend does not announce a switch between the classic frontend and
// Nodes 2.0, so the annotation nodes are re-rendered when the mode flips.
installRendererModeWatcher();

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
    } else if (nodeData.name === "AboutAuthorNode|Mie") {
      console.log("[MieText] installing AboutAuthor behavior (zh, author.json)");
      installAboutAuthorBehavior(nodeType, AUTHOR_PROFILE_URL);
      registerMieTextNodeType(nodeType);
      installMieDrawNodeHook();
    } else if (nodeData.name === "AboutAuthorNodeEn|Mie") {
      console.log("[MieText] installing AboutAuthor behavior (en, author_en.json)");
      installAboutAuthorBehavior(nodeType, AUTHOR_PROFILE_URL_EN);
      registerMieTextNodeType(nodeType);
      installMieDrawNodeHook();
    }
  },
});
