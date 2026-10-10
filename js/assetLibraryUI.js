import { app } from "../../scripts/app.js";

/**
 * 资产库前端辅助（docs/PLAN_ASSET_LIBRARY_V2.md）。
 *
 * 数据契约：
 * - 库位置每台机器固定（$MIE_ASSET_LIBRARY → ~/mie_asset_library），节点上
 *   无路径可编辑：library_dir widget 保留在输入里且永远隐藏（老工作流的
 *   widgets_values 下标不错位），执行与扫描完全忽略它的值。
 * - LoadAssetBatch 的隐藏 `names` STRING widget 是槽位顺序的唯一序列化
 *   记录（用户不可打开）；DOM 槽位条和选择器弹窗是它的两张脸。
 * - 本文件添加的一切 widget 都带 serialize:false 且永远追加在
 *   node.widgets 末尾——工作流 JSON 与加 UI 之前逐字节相同。
 * - 图片一律走 /thumb（256px JPEG）：弹窗/槽位条/预览绝不解码原图。
 * - 弹窗卡片 3:4 竖幅、object-fit: contain——肖像从头到脚完整可见。
 */

const LIST_ENDPOINT = "/mie_nodes/assets/list";
const META_ENDPOINT = "/mie_nodes/assets/meta";
const THUMB_ENDPOINT = "/mie_nodes/assets/thumb";
const OPEN_ENDPOINT = "/mie_nodes/assets/open";
const NEW_NAME = "（✏️ 新名字…）";
const PLACEHOLDER = "（下拉选择资产）";
const MAX_SLOTS = 9;
const MAX_AUDIO = 8;
const KIND_CHAR = "角色";
const KIND_SCENE = "场景";

// nodeData.name（含 |Mie 后缀）-> 行为配置
const NODES = {
    "LoadCharacterAsset|Mie": { kinds: ["characters"], preview: "character" },
    "LoadSceneAsset|Mie": { kinds: ["scenes"], preview: "scene" },
    "SaveCharacterAsset|Mie": { kinds: ["characters"], save: "character" },
    "SaveSceneAsset|Mie": { kinds: ["scenes"], save: "scene" },
    "ListCharacterAssets|Mie": {},
    "ListSceneAssets|Mie": {},
    "LoadAssetBatch|Mie": { slots: true },
};

// ---- 取数 ------------------------------------------------------------------
const cache = new Map(); // "" -> Promise<payload>（库固定，键只有一个）

function fetchAssets({ force = false } = {}) {
    if (!force && cache.has("")) return cache.get("");
    const p = fetch(LIST_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}), // 位置由服务端解析，客户端不传路径
    })
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error("HTTP " + r.status))))
        .then((payload) => {
            payload._ts = Date.now();
            return payload;
        })
        .catch((e) => ({ ok: false, characters: [], scenes: [], error: String(e), _ts: Date.now() }));
    cache.set("", p);
    return p;
}

function fetchMeta(name, kind) {
    const q = "name=" + encodeURIComponent(name) + "&kind=" + encodeURIComponent(kind);
    return fetch(META_ENDPOINT + "?" + q)
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error("HTTP " + r.status))))
        .catch((e) => ({ ok: false, error: String(e) }));
}

function thumbUrl(name, kind, ts) {
    return THUMB_ENDPOINT +
        "?name=" + encodeURIComponent(name) +
        "&kind=" + encodeURIComponent(kind) +
        (ts ? "&v=" + ts : "");
}

function reportScanError(payload) {
    if (payload && payload.ok === false && payload.error) {
        app.extensionManager?.toast?.add?.({
            severity: "error",
            summary: "资产库读取失败",
            detail: String(payload.error).slice(0, 200),
            life: 6000,
        });
    }
}

// ---- 选项与条目 ------------------------------------------------------------
// payload.characters / payload.scenes 是 {name, image_count, audio_count?} 对象。
function charEntries(payload) {
    return payload?.ok ? payload.characters : [];
}
function sceneEntries(payload) {
    return payload?.ok ? payload.scenes : [];
}
function bareEntries(payload, kinds) {
    const out = [];
    if (kinds.includes("characters")) for (const c of charEntries(payload)) out.push(c.name);
    if (kinds.includes("scenes")) for (const s of sceneEntries(payload)) out.push(s.name);
    return out;
}
function kindOfName(payload, name) {
    if (!payload?.ok) return null;
    if (charEntries(payload).some((c) => c.name === name)) return "character";
    if (sceneEntries(payload).some((s) => s.name === name)) return "scene";
    return null;
}
function entryOf(payload, name) {
    return charEntries(payload).find((c) => c.name === name)
        || sceneEntries(payload).find((s) => s.name === name)
        || null;
}
function setComboOptions(widget, values) {
    if (!widget) return;
    widget.options.values = values.slice();
}

// ---- 扩展注册 ---------------------------------------------------------------
app.registerExtension({
    name: "MieNodes.AssetLibraryUI",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const cfg = NODES[nodeData?.name];
        if (!cfg) return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origCreated?.apply(this, arguments);
            wireNode(this, cfg);
        };

        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            origConfigure?.apply(this, arguments);
            if (cfg.slots) syncStripFromNames(this);
            if (cfg.save) syncNewNameVisibility(this);
            applyLibrary(this);
            if (cfg.save) refreshSavePreview(this, { autofill: false });
            if (cfg.preview) refreshLoadPreview(this);
        };
    },
});

// 保存执行成功 -> 名单/缩略图作废并刷新画布上的资产节点（保存节点是
// OUTPUT_NODE，会发 'executed'）。
app.api?.addEventListener?.("executed", (e) => {
    const nodeId = e.detail?.node;
    const nodes = app.graph?._nodes || [];
    const node = nodes.find((n) => String(n.id) === String(nodeId));
    if (!node || !node.__mieAssets?.cfg?.save) return;
    cache.clear();
    for (const other of nodes) {
        if (other.__mieAssets) applyLibrary(other, { force: true });
    }
});

function wireNode(node, cfg) {
    const dirWidget = node.widgets?.find((w) => w.name === "library_dir");
    if (!dirWidget) return;
    node.__mieAssets = { cfg, payload: null, syncing: false, timer: null };

    // 库位置固定：widget 永久隐藏（不再有翻出来编辑的入口）。
    dirWidget.hidden = true;

    // 「📂 打开文件夹」请求固定端点，打开解析出的那一个目录。
    node.addWidget("button", "📂 打开文件夹", null, async () => {
        try {
            const r = await fetch(OPEN_ENDPOINT, { method: "POST" });
            const j = await r.json();
            if (!j.ok) throw new Error(j.error || "HTTP " + r.status);
            app.extensionManager?.toast?.add?.({
                severity: "success", summary: "已打开资产库", detail: j.library_dir, life: 4000,
            });
        } catch (err) {
            app.extensionManager?.toast?.add?.({
                severity: "error", summary: "打开失败", detail: String(err).slice(0, 160), life: 5000,
            });
        }
    }).serialize = false;

    node.addWidget("button", "🔄 刷新资产列表", null, () => {
        applyLibrary(node, { force: true });
    }).serialize = false;

    if (cfg.save) {
        const nameWidget = node.widgets.find((w) => w.name === "name");
        const origNameCallback = nameWidget?.callback;
        if (nameWidget) {
            nameWidget.callback = (...args) => {
                origNameCallback?.apply(nameWidget, args);
                syncNewNameVisibility(node);
                refreshSavePreview(node, { autofill: true });
            };
        }
        syncNewNameVisibility(node);
        patchSaveTextListeners(node);
        makePreviewWidget(node);
        if (cfg.save === "character") patchAudioAutogrow(node);
    } else if (cfg.preview) {
        const nameWidget = node.widgets.find((w) => w.name === "name");
        const origNameCallback = nameWidget?.callback;
        if (nameWidget) {
            nameWidget.callback = (...args) => {
                origNameCallback?.apply(nameWidget, args);
                refreshLoadPreview(node);
            };
        }
        makePreviewWidget(node);
    }

    if (cfg.slots) {
        const namesWidget = node.widgets.find((w) => w.name === "names");
        if (namesWidget) {
            // 名单不提供编辑入口（V2 第 1/2 条）：只作序列化记录，永久隐藏。
            namesWidget.hidden = true;
            const origChanged = node.onWidgetChanged;
            node.onWidgetChanged = function (widgetName, ...rest) {
                origChanged?.apply(this, [widgetName, ...rest]);
                if (widgetName === "names" && !node.__mieAssets?.syncing) {
                    syncStripFromNames(this);
                }
            };
        }
        if (typeof node.addDOMWidget === "function") {
            makeStripWidget(node);
            node.addWidget("button", "🖼 选择资产…", null, () => {
                openPicker(node);
            }).serialize = false;
        }
    }

    applyLibrary(node);
}

function applyLibrary(node, { force = false } = {}) {
    const cfg = node.__mieAssets?.cfg;
    if (!cfg) return;
    fetchAssets({ force }).then((payload) => {
        if (!node.__mieAssets) return;
        node.__mieAssets.payload = payload;
        reportScanError(payload);

        if (cfg.slots) {
            syncStripFromNames(node);
            return;
        }
        const nameWidget = node.widgets.find((w) => w.name === "name");
        if (!nameWidget) return;
        const values = bareEntries(payload, cfg.kinds);
        if (cfg.save) values.unshift(NEW_NAME);
        setComboOptions(nameWidget, values);
        if (cfg.save) syncNewNameVisibility(node);
        node.setDirtyCanvas?.(true, true);
    });
}

// ---- 保存节点：new_name 显隐 + 回填 + 将覆盖 ---------------------------------
function syncNewNameVisibility(node) {
    const nameWidget = node.widgets?.find((w) => w.name === "name");
    const newNameWidget = node.widgets?.find((w) => w.name === "new_name");
    if (!nameWidget || !newNameWidget) return;
    const isSentinel = (nameWidget.value || "").includes(NEW_NAME);
    newNameWidget.hidden = !isSentinel;
    node.setDirtyCanvas?.(true, true);
}

function saveTextFields(node, kind) {
    if (kind === "character") {
        return {
            description: node.widgets.find((w) => w.name === "description"),
            voice: node.widgets.find((w) => w.name === "voice"),
        };
    }
    return { description: node.widgets.find((w) => w.name === "description") };
}

async function refreshSavePreview(node, { autofill }) {
    const ui = node.__mieAssets;
    if (!ui) return;
    const kind = ui.cfg.save;
    const nameWidget = node.widgets.find((w) => w.name === "name");
    const name = (nameWidget?.value || "").trim();
    if (!name || name === PLACEHOLDER || name.includes(NEW_NAME)) {
        renderPreviewState(node, { mode: "empty-new" });
        return;
    }
    const meta = await fetchMeta(name, kind);
    if (!node.__mieAssets || (nameWidget?.value || "").trim() !== name) return;
    if (!meta.ok) {
        renderPreviewState(node, { mode: "missing", name, kind, error: meta.error });
        return;
    }
    if (autofill) {
        for (const [field, widget] of Object.entries(saveTextFields(node, kind))) {
            if (widget) widget.value = meta[field] || "";
        }
        node.setDirtyCanvas?.(true, true);
    }
    renderPreviewState(node, { mode: "existing", meta, kind });
}

async function refreshLoadPreview(node) {
    const ui = node.__mieAssets;
    if (!ui) return;
    const kind = ui.cfg.preview;
    const nameWidget = node.widgets.find((w) => w.name === "name");
    const name = (nameWidget?.value || "").trim();
    if (!name || name === PLACEHOLDER) {
        renderPreviewState(node, { mode: "empty-new" });
        return;
    }
    const meta = await fetchMeta(name, kind);
    if (!node.__mieAssets || (nameWidget?.value || "").trim() !== name) return;
    if (!meta.ok) {
        renderPreviewState(node, { mode: "missing", name, kind, error: meta.error });
        return;
    }
    renderPreviewState(node, { mode: "existing", meta, kind, readonly: true });
}

// 文本改动刷新将覆盖标记：onWidgetChanged 包装 + 输入框 input 事件双保险。
const patchedNodes = new WeakSet();
function patchSaveTextListeners(node) {
    if (patchedNodes.has(node)) return;
    patchedNodes.add(node);
    const wants = (name) => ["description", "voice"].includes(name);
    const orig = node.onWidgetChanged;
    node.onWidgetChanged = function (widgetName, ...rest) {
        orig?.apply(this, [widgetName, ...rest]);
        if (this.__mieAssets?.cfg?.save && wants(widgetName)) {
            refreshSavePreview(this, { autofill: false });
        }
    };
    for (const widget of node.widgets || []) {
        if (!wants(widget.name) || !widget.inputEl) continue;
        let timer = null;
        widget.inputEl.addEventListener("input", () => {
            widget.value = widget.inputEl.value;
            clearTimeout(timer);
            timer = setTimeout(() => refreshSavePreview(node, { autofill: false }), 300);
        });
    }
}

// ---- 预览面板 ----------------------------------------------------------------
function makePreviewWidget(node) {
    if (typeof node.addDOMWidget !== "function") return;
    const wrap = document.createElement("div");
    wrap.className = "mie-preview";
    const widget = node.addDOMWidget("预览", "mie-asset-preview", wrap, {
        serialize: false,
        hideOnZoom: false,
        getHeight: () => 96,
    });
    widget.serialize = false;
    node.__mieAssets.previewEl = wrap;
    renderPreviewState(node, { mode: "boot" });
}

function renderPreviewState(node, state) {
    const wrap = node.__mieAssets?.previewEl;
    if (!wrap) return;
    const ui = node.__mieAssets;
    const kind = state.kind || ui.cfg.save || ui.cfg.preview;
    const ts = ui.payload?._ts;
    wrap.innerHTML = "";
    if (state.mode === "boot" || state.mode === "empty-new") {
        wrap.innerHTML = '<div class="mie-pv-hint">（选择资产后显示预览；保存节点会先回填库里的正文）</div>';
        return;
    }
    if (state.mode === "missing") {
        wrap.innerHTML = '<div class="mie-pv-hint">（库里没有 ' + kindName(kind) + '「' +
            String(state.name).slice(0, 24) + '」：' + String(state.error || "").slice(0, 120) + '）</div>';
        return;
    }
    const meta = state.meta;
    const img = document.createElement("img");
    img.src = thumbUrl(meta.name, kind, ts);
    img.alt = meta.name;
    const body = document.createElement("div");
    body.className = "mie-pv-body";
    const title = document.createElement("div");
    title.className = "mie-pv-title";
    title.textContent = kindName(kind) + " " + meta.name +
        "（" + meta.image_count + " 张" +
        (meta.audio_count != null ? "，" + meta.audio_count + " 条音频" : "") + "）";
    body.append(title);

    if (state.readonly) {
        const line = document.createElement("div");
        line.className = "mie-pv-text";
        line.textContent = (meta.description || "").slice(0, 90);
        body.append(line);
        if (meta.voice) {
            const v = document.createElement("div");
            v.className = "mie-pv-text";
            v.textContent = "🔊 " + meta.voice.slice(0, 70);
            body.append(v);
        }
    } else {
        const fields = kind === "character"
            ? [["description", "描述"], ["voice", "音色"]]
            : [["description", "描述"]];
        for (const [field, label] of fields) {
            const widget = saveTextFields(node, kind)[field];
            const row = document.createElement("div");
            row.className = "mie-pv-row";
            const current = (widget?.value || "").trim();
            const disk = (meta[field] || "").trim();
            row.textContent = label + ": ";
            const badge = document.createElement("span");
            if (current === disk) {
                badge.className = "mie-badge ok";
                badge.textContent = "与库一致";
            } else {
                badge.className = "badge diff";
                badge.textContent = "将覆盖";
            }
            row.append(badge);
            body.append(row);
        }
    }
    wrap.append(img, body);
}

function kindName(kind) {
    return kind === "scene" ? KIND_SCENE : KIND_CHAR;
}

// ---- 保存角色：audio 插座自生长（stringFormat 同款）---------------------------
const audioGrowPatched = new WeakSet();
function patchAudioAutogrow(node) {
    if (audioGrowPatched.has(node)) return;
    audioGrowPatched.add(node);
    const PREFIX = "audio_";
    const socketIndices = () => {
        const out = [];
        (node.inputs || []).forEach((inp, i) => {
            if (inp && typeof inp.name === "string" && inp.name.startsWith(PREFIX)) out.push(i);
        });
        return out;
    };
    const trimTrailingTo = (keep) => {
        let indices = socketIndices();
        while (indices.length > keep) {
            const idx = indices.pop();
            node.removeInput(idx);
            indices = socketIndices();
        }
        node.setDirtyCanvas?.(true, true);
    };
    const appendNext = () => {
        const indices = socketIndices();
        if (indices.length >= MAX_AUDIO) return;
        node.addInput(PREFIX + (indices.length + 1), "AUDIO");
        node.setDirtyCanvas?.(true, true);
    };
    const sync = () => {
        const indices = socketIndices();
        let keep = 1; // 至少保留 audio_1
        for (let k = 0; k < indices.length; k++) {
            if (node.inputs[indices[k]].link != null) keep = k + 2;
        }
        keep = Math.min(keep, MAX_AUDIO);
        trimTrailingTo(keep);
        const now = socketIndices();
        if (now.length < keep) appendNext();
        else if (now.length < MAX_AUDIO
                 && node.inputs[now[now.length - 1]].link != null) {
            appendNext();
        }
    };
    const orig = node.onConnectionsChange;
    node.onConnectionsChange = function (type, index, connected, ...rest) {
        orig?.apply(this, [type, index, connected, ...rest]);
        if (type === 1 /*INPUT*/) sync();
    };
    // 新节点只露出 audio_1；configure 会按已存工作流还原更多。
    trimTrailingTo(1);
}

// ---- LoadAssetBatch 槽位条 ----------------------------------------------------
function currentSlotNames(node) {
    const namesWidget = node.widgets.find((w) => w.name === "names");
    return (namesWidget?.value || "").split("\n").map((s) => s.trim()).filter(Boolean);
}

function writeSlotNames(node, names) {
    const namesWidget = node.widgets.find((w) => w.name === "names");
    if (!namesWidget) return;
    const ui = node.__mieAssets;
    if (ui) ui.syncing = true;
    try {
        namesWidget.value = names.join("\n");
        namesWidget.callback?.(namesWidget.value);
    } finally {
        if (ui) ui.syncing = false;
    }
    syncStripFromNames(node);
    node.setDirtyCanvas?.(true, true);
}

function makeStripWidget(node) {
    const wrap = document.createElement("div");
    wrap.className = "mie-strip";
    const widget = node.addDOMWidget("槽位（顺序 = <Picture N>）", "mie-asset-strip", wrap, {
        serialize: false,
        hideOnZoom: false,
        getHeight: () => 64,
    });
    widget.serialize = false;
    node.__mieAssets.stripEl = wrap;
}

function syncStripFromNames(node) {
    const ui = node.__mieAssets;
    const wrap = ui?.stripEl;
    if (!wrap) return;
    const payload = ui.payload;
    const ts = payload?._ts;
    wrap.innerHTML = "";
    const picked = currentSlotNames(node);
    if (!picked.length) {
        wrap.innerHTML = '<span class="mie-strip-hint">（点「🖼 选择资产…」按顺序添加；顺序即 <Picture N>）</span>';
        return;
    }
    picked.forEach((name, i) => {
        const kind = kindOfName(payload, name) || "character";
        const chip = document.createElement("div");
        chip.className = "mie-slot";
        chip.draggable = true;
        chip.dataset.index = String(i);
        const img = document.createElement("img");
        img.src = thumbUrl(name, kind, ts);
        const label = document.createElement("span");
        label.className = "mie-slot-label";
        label.textContent = (i + 1) + ". " + name;
        const kindTag = document.createElement("span");
        kindTag.className = "mie-slot-kind";
        kindTag.textContent = kind === "scene" ? KIND_SCENE : KIND_CHAR;
        const entry = entryOf(payload, name);
        if (entry) {
            // 数量常驻：扫描回来后每个 chip 都标 N张（角色再加 N条音频）。
            let info = entry.image_count + "张";
            if (kind === "character" && entry.audio_count != null) {
                info += entry.audio_count + "音";
            }
            kindTag.textContent += " " + info;
        }
        const x = document.createElement("span");
        x.className = "mie-slot-x";
        x.textContent = "✕";
        x.title = "移出槽位 " + (i + 1);
        x.addEventListener("click", () => {
            const cur = currentSlotNames(node);
            cur.splice(i, 1);
            writeSlotNames(node, cur);
        });
        chip.append(img, label, kindTag, x);
        chip.addEventListener("dragstart", (e) => {
            e.dataTransfer.setData("text/mie-slot", String(i));
            chip.classList.add("dragging");
        });
        chip.addEventListener("dragend", () => chip.classList.remove("dragging"));
        chip.addEventListener("dragover", (e) => e.preventDefault());
        chip.addEventListener("drop", (e) => {
            e.preventDefault();
            const from = parseInt(e.dataTransfer.getData("text/mie-slot"), 10);
            if (Number.isNaN(from) || from === i) return;
            const cur = currentSlotNames(node);
            const [moved] = cur.splice(from, 1);
            cur.splice(i, 0, moved);
            writeSlotNames(node, cur);
        });
        wrap.append(chip);
    });
}

// ---- 图片选择器弹窗 -------------------------------------------------------------
let pickerStyleEl = null;

function ensurePickerStyles() {
    if (pickerStyleEl) return;
    pickerStyleEl = document.createElement("style");
    pickerStyleEl.textContent = [
        ".mie-picker-backdrop{position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:99999;",
        "display:flex;align-items:center;justify-content:center}",
        ".mie-picker{background:var(--comfy-menu-bg,#2a2a2a);color:var(--input-text,#ddd);",
        "border:1px solid var(--border-color,#444);border-radius:10px;width:min(880px,92vw);",
        "max-height:86vh;display:flex;flex-direction:column;box-shadow:0 12px 48px rgba(0,0,0,.5);",
        "font-family:inherit;font-size:12px}",
        ".mie-picker header{display:flex;align-items:center;gap:10px;padding:10px 14px;",
        "border-bottom:1px solid var(--border-color,#444)}",
        ".mie-picker header .t{font-size:14px;font-weight:600}",
        ".mie-picker header .d{font-size:11px;opacity:.65;flex:1;overflow:hidden;",
        "text-overflow:ellipsis;white-space:nowrap}",
        ".mie-picker .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));",
        "gap:10px;padding:12px;overflow-y:auto}",
        // 3:4 竖幅 + contain：肖像从头到脚完整可见（V2 第 7 条）。
        ".mie-card{border:1px solid var(--border-color,#444);border-radius:8px;overflow:hidden;",
        "cursor:pointer;background:rgba(255,255,255,.03);transition:border-color .12s}",
        ".mie-card:hover{border-color:#7f9cf5}",
        ".mie-card.disabled{opacity:.45;cursor:not-allowed}",
        ".mie-card img{width:100%;aspect-ratio:3/4;object-fit:contain;object-position:center;",
        "display:block;background:#151515}",
        ".mie-card .cap{display:flex;align-items:center;gap:6px;padding:6px 8px}",
        ".mie-card .kind{font-size:10px;padding:1px 6px;border-radius:6px;background:#3b4a6b;",
        "color:#cdd8f8;flex-shrink:0}",
        ".mie-card .nm{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}",
        ".mie-card .in{margin-left:auto;font-size:10px;opacity:.6;flex-shrink:0}",
        ".mie-empty{padding:26px;text-align:center;opacity:.6}",
        ".mie-picker footer{display:flex;flex-wrap:wrap;gap:6px;align-items:center;",
        "padding:10px 14px;border-top:1px solid var(--border-color,#444);min-height:40px}",
        ".mie-preview{display:flex;gap:8px;align-items:flex-start;padding:4px 2px}",
        ".mie-preview img{width:72px;height:96px;object-fit:contain;background:#151515;",
        "border-radius:8px}",
        ".mie-pv-body{flex:1;min-width:0}",
        ".mie-pv-title{font-weight:600;font-size:12px}",
        ".mie-pv-text{font-size:11px;opacity:.8;overflow:hidden;text-overflow:ellipsis;",
        "white-space:nowrap}",
        ".mie-pv-row{font-size:11px;margin-top:3px}",
        ".mie-badge{padding:0 6px;border-radius:6px;font-size:10px}",
        ".mie-badge.ok{background:#2e4d34;color:#b7e4c0}",
        ".badge.diff{background:#5a3535;color:#f3b6b6}",
        ".mie-pv-hint{font-size:11px;opacity:.55;padding:8px 2px}",
        ".mie-strip{display:flex;flex-wrap:wrap;gap:6px;padding:4px 2px;align-items:center}",
        ".mie-strip-hint{font-size:11px;opacity:.55}",
        ".mie-slot{display:flex;align-items:center;gap:6px;border:1px solid var(--border-color,#555);",
        "border-radius:8px;padding:3px 8px 3px 4px;background:rgba(255,255,255,.03)}",
        ".mie-slot.dragging{opacity:.4}",
        ".mie-slot img{width:38px;height:50px;object-fit:contain;background:#151515;",
        "border-radius:6px}",
        ".mie-slot-label{font-size:12px;max-width:120px;overflow:hidden;text-overflow:ellipsis;",
        "white-space:nowrap}",
        ".mie-slot-kind{font-size:10px;padding:1px 6px;border-radius:6px;background:#3b4a6b;",
        "color:#cdd8f8}",
        ".mie-slot-x{cursor:pointer;opacity:.6;font-weight:700}",
        ".mie-slot-x:hover{opacity:1;color:#f88}",
    ].join("");
    document.head.appendChild(pickerStyleEl);
}

function openPicker(node) {
    ensurePickerStyles();
    closePicker();
    const ui = node.__mieAssets;
    const payload = ui?.payload;
    const ts = payload?._ts;

    const backdrop = document.createElement("div");
    backdrop.className = "mie-picker-backdrop";
    const panel = document.createElement("div");
    panel.className = "mie-picker";

    const header = document.createElement("header");
    header.innerHTML = '<span class="t">🖼 选择资产（点卡片加入下一个槽位，上限 ' + MAX_SLOTS + '；拖动槽位条排序）</span><span class="d"></span>';
    header.querySelector(".d").textContent = payload?.library_dir ? "库：" + payload.library_dir : "库：（未扫描）";
    const refreshBtn = document.createElement("button");
    refreshBtn.textContent = "🔄";
    refreshBtn.title = "重新扫描";
    const closeBtn = document.createElement("button");
    closeBtn.textContent = "✕";
    header.append(refreshBtn, closeBtn);
    panel.append(header);

    const grid = document.createElement("div");
    grid.className = "grid";
    const footer = document.createElement("footer");
    panel.append(grid, footer);
    backdrop.append(panel);
    document.body.append(backdrop);

    function render() {
        grid.innerHTML = "";
        const conflict = new Set(
            payload?.ok
                ? charEntries(payload).map((c) => c.name).filter((n) => sceneEntries(payload).some((s) => s.name === n))
                : []
        );
        const items = [];
        if (payload?.ok) {
            for (const c of charEntries(payload)) items.push({ ...c, kind: "character" });
            for (const s of sceneEntries(payload)) items.push({ ...s, kind: "scene" });
        }
        if (!items.length) {
            const empty = document.createElement("div");
            empty.className = "mie-empty";
            empty.textContent = payload?.ok === false
                ? "库读取失败：" + (payload.error || "") + "（点 🔄 重试）"
                : "库里还没有可读资产，或尚未扫描——点右上 🔄。";
            grid.append(empty);
        }
        const picked = currentSlotNames(node);
        const inSlot = new Map(picked.map((n, i) => [n, i + 1]));
        for (const it of items) {
            const slot = inSlot.get(it.name);
            const isConflict = conflict.has(it.name);
            const card = document.createElement("div");
            card.className = "mie-card" + (isConflict ? " disabled" : "");
            card.title = isConflict
                ? "同名冲突：角色和场景同名，批量加载会报错；请在文件管理器里改名"
                : slot ? "已在槽位 " + slot : "点击加入下一个槽位";
            const img = document.createElement("img");
            img.loading = "lazy";
            img.src = thumbUrl(it.name, it.kind, ts);
            img.alt = it.name;
            const cap = document.createElement("div");
            cap.className = "cap";
            const kindTag = document.createElement("span");
            kindTag.className = "kind";
            kindTag.textContent = it.kind === "scene" ? KIND_SCENE : KIND_CHAR;
            const nm = document.createElement("span");
            nm.className = "nm";
            nm.textContent = it.name;
            cap.append(kindTag, nm);
            // 数量角标无条件常驻（用户要求"很清楚地显示有多少图多少语音"）：
            // 1 张也显示、0 条音频也显示；卡片图只用第一张，其余以计数告知。
            const counts = [it.image_count + " 张"];
            if (it.kind === "character") counts.push(it.audio_count + " 条音频");
            const badge = document.createElement("span");
            badge.className = "in";
            if (isConflict) {
                badge.textContent = "同名冲突";
            } else {
                // 槽位与数量并列常驻：进了槽位也看得见有几张图几条音频。
                const parts = [];
                if (slot) parts.push("槽位 " + slot);
                parts.push(...counts);
                badge.textContent = parts.join(" · ");
            }
            cap.append(badge);
            card.append(img, cap);
            card.addEventListener("click", () => {
                if (isConflict) return;
                const cur = currentSlotNames(node);
                if (cur.includes(it.name)) {
                    app.extensionManager?.toast?.add?.({
                        severity: "info", summary: "已在槽位 " + (cur.indexOf(it.name) + 1),
                        detail: it.name, life: 2500,
                    });
                    return;
                }
                if (cur.length >= MAX_SLOTS) {
                    app.extensionManager?.toast?.add?.({
                        severity: "warn", summary: "槽位已满",
                        detail: "最多 " + MAX_SLOTS + " 个（<Picture N> 上限）", life: 3000,
                    });
                    return;
                }
                writeSlotNames(node, cur.concat([it.name]));
                render();
            });
            grid.append(card);
        }
        footer.innerHTML = "";
        const pickedOrder = currentSlotNames(node);
        const hint = document.createElement("span");
        hint.style.cssText = "opacity:.6";
        hint.textContent = pickedOrder.length
            ? "已选 " + pickedOrder.length + "/" + MAX_SLOTS + "：顺序在节点的槽位条上调整（拖动）"
            : "（还没有选择任何资产）";
        footer.append(hint);
    }

    refreshBtn.addEventListener("click", async () => {
        refreshBtn.textContent = "⏳";
        const fresh = await fetchAssets({ force: true });
        if (node.__mieAssets) node.__mieAssets.payload = fresh;
        applyLibrary(node, { force: true });
        refreshBtn.textContent = "🔄";
        render();
    });
    closeBtn.addEventListener("click", closePicker);
    backdrop.addEventListener("click", (e) => {
        if (e.target === backdrop) closePicker();
    });
    document.addEventListener("keydown", pickerEscHandler);

    render();
    node.__miePicker = backdrop;
}

function pickerEscHandler(e) {
    if (e.key === "Escape") closePicker();
}

function closePicker() {
    document.querySelectorAll(".mie-picker-backdrop").forEach((el) => el.remove());
    document.removeEventListener("keydown", pickerEscHandler);
}
