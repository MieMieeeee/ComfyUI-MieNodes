import { app } from "../../scripts/app.js";

/**
 * 资产库前端辅助（docs/ASSET_LIBRARY_GUIDE.md；评审 REVIEW_ASSET_LIBRARY_UI）。
 *
 * 数据契约：
 * - 多行 `names` STRING widget（LoadAssetBatch）是唯一序列化真相；节点上的
 *   DOM 槽位条和图片选择器弹窗都是它的镜像。本文件添加的一切 widget 都带
 *   serialize:false 且永远追加在 node.widgets 末尾——litegraph 序列化与加载
 *   两侧只处理 serialize!==false 子集并按下标对齐，工作流 JSON 与加 UI 之前
 *   逐字节相同。
 * - 下拉选项按节点实例从 POST /mie_nodes/assets/list 取（类级 combo 拿不到
 *   另一个 widget 的值）；空 library_dir = 本机默认库，由服务端解析。
 * - 保存节点选中已有名字时，GET /mie_nodes/assets/meta 把磁盘上的正文回填
 *   进输入框——要覆盖的东西必须先被看见。
 * - 图片一律走 /thumb（256px JPEG，ETag=文件戳）：弹窗/槽位条/预览都绝不
 *   解码原图。
 */

const LIST_ENDPOINT = "/mie_nodes/assets/list";
const META_ENDPOINT = "/mie_nodes/assets/meta";
const THUMB_ENDPOINT = "/mie_nodes/assets/thumb";
const NEW_NAME = "（✏️ 新名字…）";
const PLACEHOLDER = "（下拉选择资产）";
const MAX_SLOTS = 9;
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

// ---- 取数：按目录缓存 + 去重 ---------------------------------------------
const cache = new Map(); // library_dir(原始串) -> Promise<payload>

function fetchAssets(libraryDir, { force = false } = {}) {
    // 空串 = 本机默认库，交给服务端解析（第六次修订后的语义）。
    const dir = (libraryDir || "").trim();
    if (!force && cache.has(dir)) return cache.get(dir);
    const p = fetch(LIST_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ library_dir: dir }),
    })
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error("HTTP " + r.status))))
        .then((payload) => {
            payload._ts = Date.now(); // 缩略图 URL 的新鲜度键（扫描时刻）
            return payload;
        })
        .catch((e) => ({ ok: false, characters: [], scenes: [], error: String(e), _ts: Date.now() }));
    cache.set(dir, p);
    return p;
}

function fetchMeta(dir, name, kind) {
    const q = "dir=" + encodeURIComponent(dir) + "&name=" + encodeURIComponent(name) +
        "&kind=" + encodeURIComponent(kind);
    return fetch(META_ENDPOINT + "?" + q)
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error("HTTP " + r.status))))
        .catch((e) => ({ ok: false, error: String(e) }));
}

function thumbUrl(dir, name, kind, ts) {
    return THUMB_ENDPOINT +
        "?dir=" + encodeURIComponent(dir) +
        "&name=" + encodeURIComponent(name) +
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

// ---- 选项列表 --------------------------------------------------------------
function bareEntries(payload, kinds) {
    const out = [];
    if (!payload || !payload.ok) return out;
    const wantChar = kinds.includes("characters");
    const wantScene = kinds.includes("scenes");
    if (wantChar) out.push(...payload.characters);
    if (wantScene) out.push(...payload.scenes);
    return out;
}

function kindOfName(payload, name) {
    if (!payload?.ok) return null;
    if (payload.characters.includes(name)) return "character";
    if (payload.scenes.includes(name)) return "scene";
    return null;
}

function setComboOptions(widget, values) {
    if (!widget) return;
    widget.options.values = values.slice();
    // 老工作流手打的值不在列表里也保留显示——combo 只约束菜单。
}

// ---- 扩展注册 -------------------------------------------------------------
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

        // 工作流加载/粘贴：widget 值应用完之后，由值重建 UI 状态。
        // onNodeCreated 时值还是默认（要等 configure 才应用），取数与
        // 回填判断也要在这里再触发一次。
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

// 保存执行成功 -> 该目录的名字列表与缩略图全部作废，刷新画布上的资产节点。
// （保存节点是 OUTPUT_NODE，会发 'executed' 事件。）
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

    // 库目录固定（第六次修订）：widget 默认隐藏（仍序列化、仍参与对齐）。
    // 按钮常驻显示解析后的库位置——不再只活在一条会消失的 toast 里。
    dirWidget.hidden = true;
    const dirBtn = node.addWidget("button", "📁 库…", null, () => {
        dirWidget.hidden = !dirWidget.hidden;
        const resolved = node.__mieAssets?.payload?.library_dir;
        app.extensionManager?.toast?.add?.({
            severity: "info",
            summary: dirWidget.hidden ? "库目录（已隐藏）" : "库目录（可直接编辑）",
            detail: resolved ? "当前生效: " + resolved
                : "默认: $MIE_ASSET_LIBRARY 或 ~/mie_asset_library",
            life: 5000,
        });
        node.setDirtyCanvas?.(true, true);
    });
    dirBtn.serialize = false;

    node.addWidget("button", "🔄 刷新资产列表", null, () => {
        applyLibrary(node, { force: true });
    }).serialize = false;

    // library_dir 手输/粘贴 -> 每节点一个防抖定时器（共享定时器会让
    // 节点 A 的输入吃掉节点 B 还没发出的刷新）。
    const origDirCallback = dirWidget.callback;
    dirWidget.callback = (...args) => {
        origDirCallback?.apply(dirWidget, args);
        const ui = node.__mieAssets;
        clearTimeout(ui?.timer);
        if (ui) ui.timer = setTimeout(() => applyLibrary(node), 500);
    };

    if (cfg.save) {
        const nameWidget = node.widgets.find((w) => w.name === "name");
        const origNameCallback = nameWidget?.callback;
        if (nameWidget) {
            nameWidget.callback = (...args) => {
                origNameCallback?.apply(nameWidget, args);
                syncNewNameVisibility(node);
                // 用户主动选了已有名字：把磁盘上的正文回填——要覆盖的
                // 东西必须先被看见。工作流加载（configure）不走这里。
                refreshSavePreview(node, { autofill: true });
            };
        }
        syncNewNameVisibility(node);
        patchSaveTextListeners(node);
        makePreviewWidget(node, cfg.save);
    } else if (cfg.preview) {
        const nameWidget = node.widgets.find((w) => w.name === "name");
        const origNameCallback = nameWidget?.callback;
        if (nameWidget) {
            nameWidget.callback = (...args) => {
                origNameCallback?.apply(nameWidget, args);
                refreshLoadPreview(node);
            };
        }
        makePreviewWidget(node, cfg.preview);
    }

    if (cfg.slots) {
        const namesWidget = node.widgets.find((w) => w.name === "names");
        if (namesWidget) {
            // names 默认收起（仍是唯一序列化真相）；「✎ 名单」给手改逃生口。
            namesWidget.hidden = true;
            node.addWidget("button", "✎ 名单", null, () => {
                namesWidget.hidden = !namesWidget.hidden;
                node.setDirtyCanvas?.(true, true);
            }).serialize = false;
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

function resolvedDir(node) {
    return node.__mieAssets?.payload?.library_dir ||
        (node.widgets.find((w) => w.name === "library_dir")?.value || "").trim();
}

function applyLibrary(node, { force = false } = {}) {
    const cfg = node.__mieAssets?.cfg;
    if (!cfg) return;
    // 空串 = 本机默认库，由服务端解析（第六次修订后的语义）。
    const dir = (node.widgets?.find((w) => w.name === "library_dir")?.value || "").trim();

    fetchAssets(dir, { force }).then((payload) => {
        if (!node.__mieAssets) return;
        node.__mieAssets.payload = payload;
        reportScanError(payload);

        // 库按钮常驻显示解析后的位置（basename + 完整路径在 tooltip）。
        const dirBtn = node.widgets.find((w) => w.name === "📁 库…");
        if (dirBtn && payload?.library_dir) {
            const full = String(payload.library_dir);
            dirBtn.name = "📁 " + full.split(/[\\/]/).filter(Boolean).pop();
            dirBtn.tooltip = full;
        }

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

// ---- Save 节点：new_name 显隐 + 选中回填 + 将覆盖标记 ----------------------
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
            appearance: node.widgets.find((w) => w.name === "appearance"),
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
    const pane = ui.previewEl;
    if (!pane) return;
    if (!name || name === PLACEHOLDER || name.includes(NEW_NAME)) {
        renderPreviewState(node, { mode: "empty-new" });
        return;
    }
    const meta = await fetchMeta(resolvedDir(node), name, kind);
    if (!node.__mieAssets || (nameWidget?.value || "").trim() !== name) return; // 已切走
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
    const meta = await fetchMeta(resolvedDir(node), name, kind);
    if (!node.__mieAssets || (nameWidget?.value || "").trim() !== name) return;
    if (!meta.ok) {
        renderPreviewState(node, { mode: "missing", name, kind, error: meta.error });
        return;
    }
    renderPreviewState(node, { mode: "existing", meta, kind, readonly: true });
}

// ---- 预览面板（Save / Load 共用的 DOM widget）-----------------------------
function makePreviewWidget(node, kind) {
    if (typeof node.addDOMWidget !== "function") return;
    const wrap = document.createElement("div");
    wrap.className = "mie-preview";
    const widget = node.addDOMWidget("预览", "mie-asset-preview", wrap, {
        serialize: false,
        hideOnZoom: false,
        getHeight: () => 86,
    });
    widget.serialize = false;
    node.__mieAssets.previewEl = wrap;
    wrap.dataset.kind = kind;
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
    img.src = thumbUrl(resolvedDir(node), meta.name, kind, ts);
    img.alt = meta.name;
    const body = document.createElement("div");
    body.className = "mie-pv-body";
    const title = document.createElement("div");
    title.className = "mie-pv-title";
    title.textContent = (kind === "scene" ? "场景 " : "角色 ") + meta.name;
    body.append(title);

    if (state.readonly) {
        const line = document.createElement("div");
        line.className = "mie-pv-text";
        line.textContent = (meta.description || meta.appearance || "").slice(0, 90);
        body.append(line);
        if (meta.voice) {
            const v = document.createElement("div");
            v.className = "mie-pv-text";
            v.textContent = "🔊 " + meta.voice.slice(0, 70);
            body.append(v);
        }
    } else {
        // Save：逐字段比较节点文本与库文本——“将覆盖”必须被看见。
        const fields = kind === "character"
            ? [["appearance", "人物"], ["voice", "音色"]]
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

// 文本框改动时刷新将覆盖标记。两条路都挂：onWidgetChanged 包装 + 直接
// 监听输入框的 input 事件——这个前端的 multiline 打字不保证走前者。
const origTextChangedPatched = new WeakSet();
function patchSaveTextListeners(node) {
    if (origTextChangedPatched.has(node)) return;
    origTextChangedPatched.add(node);
    const ui = node.__mieAssets;
    const wants = (name) => ["appearance", "voice", "description"].includes(name);
    const orig = node.onWidgetChanged;
    node.onWidgetChanged = function (widgetName, ...rest) {
        orig?.apply(this, [widgetName, ...rest]);
        if (this.__mieAssets?.cfg?.save && wants(widgetName)) {
            refreshSavePreview(this, { autofill: false });
        }
    };
    if (ui?.cfg?.save) {
        for (const widget of node.widgets || []) {
            if (!wants(widget.name) || !widget.inputEl) continue;
            let timer = null;
            widget.inputEl.addEventListener("input", () => {
                widget.value = widget.inputEl.value;
                clearTimeout(timer);
                timer = setTimeout(
                    () => refreshSavePreview(node, { autofill: false }), 300
                );
            });
        }
    }
}

// ---- LoadAssetBatch 槽位条（DOM widget，替代槽位 combo）--------------------
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
        // 直接赋 .value 不会触发 onWidgetChanged（那是 setValue 的路径），
        // 所以写完要显式重建槽位条。
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
        img.src = thumbUrl(resolvedDir(node), name, kind, ts);
        const label = document.createElement("span");
        label.className = "mie-slot-label";
        label.textContent = (i + 1) + ". " + name;
        const kindTag = document.createElement("span");
        kindTag.className = "mie-slot-kind";
        kindTag.textContent = kind === "scene" ? KIND_SCENE : KIND_CHAR;
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

        // 拖动排序：顺序就是 <Picture N>。
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

// ---- 图片选择器弹窗 --------------------------------------------------------
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
        ".mie-card{border:1px solid var(--border-color,#444);border-radius:8px;overflow:hidden;",
        "cursor:pointer;background:rgba(255,255,255,.03);transition:border-color .12s}",
        ".mie-card:hover{border-color:#7f9cf5}",
        ".mie-card.disabled{opacity:.45;cursor:not-allowed}",
        ".mie-card img{width:100%;height:104px;object-fit:cover;display:block;background:#111}",
        ".mie-card .cap{display:flex;align-items:center;gap:6px;padding:6px 8px}",
        ".mie-card .kind{font-size:10px;padding:1px 6px;border-radius:6px;background:#3b4a6b;",
        "color:#cdd8f8;flex-shrink:0}",
        ".mie-card .nm{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}",
        ".mie-card .in{margin-left:auto;font-size:10px;opacity:.6;flex-shrink:0}",
        ".mie-empty{padding:26px;text-align:center;opacity:.6}",
        ".mie-picker footer{display:flex;flex-wrap:wrap;gap:6px;align-items:center;",
        "padding:10px 14px;border-top:1px solid var(--border-color,#444);min-height:40px}",
        ".mie-preview{display:flex;gap:8px;align-items:flex-start;padding:4px 2px}",
        ".mie-preview img{width:76px;height:76px;object-fit:cover;border-radius:8px;background:#111}",
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
        ".mie-slot img{width:38px;height:38px;object-fit:cover;border-radius:6px;background:#111}",
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
    const dir = resolvedDir(node);

    const backdrop = document.createElement("div");
    backdrop.className = "mie-picker-backdrop";
    const panel = document.createElement("div");
    panel.className = "mie-picker";

    const header = document.createElement("header");
    header.innerHTML = '<span class="t">🖼 选择资产（点卡片加入下一个槽位，上限 ' + MAX_SLOTS + '；可拖动槽位条排序）</span><span class="d"></span>';
    header.querySelector(".d").textContent = dir ? "库：" + dir : "库：默认（未扫描）";
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
        // 同名角色/场景是执行期错误（跨类同名报错），在这里就不可选。
        const conflict = new Set(
            payload?.ok
                ? payload.characters.filter((n) => payload.scenes.includes(n))
                : []
        );
        const items = [];
        if (payload?.ok) {
            for (const n of payload.characters) items.push({ name: n, kind: "character" });
            for (const n of payload.scenes) items.push({ name: n, kind: "scene" });
        }
        if (!items.length) {
            const empty = document.createElement("div");
            empty.className = "mie-empty";
            empty.textContent = payload?.ok === false
                ? "库读取失败：" + (payload.error || "") + "（检查目录后点 🔄）"
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
            img.src = thumbUrl(dir || ".", it.name, it.kind, ts);
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
            if (isConflict) {
                const warn = document.createElement("span");
                warn.className = "in";
                warn.textContent = "同名冲突";
                cap.append(warn);
            } else if (slot) {
                const tag = document.createElement("span");
                tag.className = "in";
                tag.textContent = "槽位 " + slot;
                cap.append(tag);
            }
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
        const dirWidget = node.widgets.find((w) => w.name === "library_dir");
        const fresh = await fetchAssets(dirWidget?.value || "", { force: true });
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
