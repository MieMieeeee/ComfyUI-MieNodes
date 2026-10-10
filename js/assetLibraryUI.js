import { app } from "../../scripts/app.js";

/**
 * 资产库前端辅助（docs/PLAN_ASSET_LIBRARY_V2.md，2026-10-10 终版）。
 *
 * 数据契约：
 * - 库位置每台机器固定（$MIE_ASSET_LIBRARY → <插件>/asset_library），节点上
 *   无路径可编辑：library_dir widget 保留在输入里且永远隐藏，执行与扫描
 *   完全忽略它的值。
 * - 资产节点是简化模型（角色 = 一张图 + 描述 + 一段音频 + 音色；场景 =
 *   一张图 + 描述），`asset` 输出一根线携带完整资产包。
 * - 生成器 V2 的 character_N / scene_N 是 SOCKET（接 Load 节点的 asset
 *   输出，接上才露出下一个）；本文件不生成任何选择控件。
 * - 本文件添加的一切 widget 都带 serialize:false 且永远追加在
 *   node.widgets 末尾——工作流 JSON 与加 UI 之前逐字节相同。
 * - 预览缩略图一律走 /thumb（256px JPEG），绝不解码原图。
 */

const LIST_ENDPOINT = "/mie_nodes/assets/list";
const META_ENDPOINT = "/mie_nodes/assets/meta";
const THUMB_ENDPOINT = "/mie_nodes/assets/thumb";
const OPEN_ENDPOINT = "/mie_nodes/assets/open";
const NEW_NAME = "（✏️ 新名字…）";
const PLACEHOLDER = "（下拉选择资产）";
const KIND_CHAR = "角色";
const KIND_SCENE = "场景";

// nodeData.name（含 |Mie 后缀）-> 行为配置
const NODES = {
    "LoadCharacterAssetV2|Mie": { kinds: ["characters"], preview: "character" },
    "LoadSceneAssetV2|Mie": { kinds: ["scenes"], preview: "scene" },
    "SaveCharacterAssetV2|Mie": { kinds: ["characters"], save: "character" },
    "SaveSceneAssetV2|Mie": { kinds: ["scenes"], save: "scene" },
    // 生成器：角色/场景插座自动生长（无选择控件）。
    "MiniMaxH3LoopPromptGeneratorV2|Mie": { generatorSockets: true },
    // 槽位展开节点：吃 asset 包插座（同生成器），输出按接线数收放。
    "AssetToSlots|Mie": { assetSlots: true },
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

        if (cfg.assetSlots) {
            // 插座自动生长（同生成器），且可见输出跟随接线数量收放：
            // picture_1..N（角色+场景序）、audio_1..min(角色数,3)。
            const slotPrefixes = ["character", "scene"];
            const socketIdx = (node, prefix) => {
                const out = [];
                (node.inputs || []).forEach((inp, i) => {
                    if (inp && typeof inp.name === "string"
                        && inp.name.startsWith(prefix + "_")
                        && /^\d+$/.test(inp.name.slice(prefix.length + 1))) out.push(i);
                });
                return out;
            };
            const wiredCount = (node, prefix) => {
                let n = 0;
                socketIdx(node, prefix).forEach((i) => {
                    if (node.inputs[i].link != null) n += 1;
                });
                return n;
            };
            const syncOutputs = (node) => {
                const pics = wiredCount(node, "character") + wiredCount(node, "scene");
                const auds = Math.min(wiredCount(node, "character"), 3);
                for (const [prefix, want] of [["picture", Math.max(pics, 1)], ["audio", Math.max(auds, 1)]]) {
                    const isNamed = (o) => o && o.name && o.name.startsWith(prefix + "_")
                        && /^\d+$/.test(o.name.slice(prefix.length + 1));
                    let indices = [];
                    (node.outputs || []).forEach((o, i) => { if (isNamed(o)) indices.push(i); });
                    // 已接线的输出绝不裁掉。
                    let keep = want;
                    indices.forEach((i, k) => {
                        const o = node.outputs[i];
                        if (o.links != null && o.links.length) keep = Math.max(keep, k + 1);
                    });
                    while (indices.length > keep) {
                        node.removeOutput(indices.pop());
                        indices = [];
                        (node.outputs || []).forEach((o, i) => { if (isNamed(o)) indices.push(i); });
                    }
                    while (indices.length < want) {
                        node.addOutput(prefix + "_" + (indices.length + 1),
                                       prefix === "picture" ? "IMAGE" : "AUDIO");
                        indices = [];
                        (node.outputs || []).forEach((o, i) => { if (isNamed(o)) indices.push(i); });
                    }
                }
                node.setDirtyCanvas?.(true, true);
            };
            const syncSockets = (node) => {
                for (const prefix of slotPrefixes) {
                    const indices = socketIdx(node, prefix);
                    let keep = 1;
                    for (let k = 0; k < indices.length; k++) {
                        if (node.inputs[indices[k]].link != null) keep = k + 2;
                    }
                    keep = Math.min(keep, 9);
                    while (indices.length > keep) {
                        node.removeInput(indices.pop());
                    }
                    let now = socketIdx(node, prefix);
                    if (now.length < keep
                        || (now.length < 9 && node.inputs[now[now.length - 1]].link != null)) {
                        node.addInput(prefix + "_" + (now.length + 1),
                                      prefix === "character" ? "MIE_CHARACTER_ASSET" : "MIE_SCENE_ASSET");
                    }
                }
                syncOutputs(node);
            };
            const origConn = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (type, index, connected, ...rest) {
                origConn?.apply(this, [type, index, connected, ...rest]);
                if (type === 1 /*INPUT*/) syncSockets(this);
            };
            const origCreated2 = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                origCreated2?.apply(this, arguments);
                for (const prefix of slotPrefixes) {
                    const indices = socketIdx(this, prefix);
                    while (indices.length > 1) this.removeInput(indices.pop());
                }
                syncOutputs(this);
            };
            const origConfigure2 = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                origConfigure2?.apply(this, arguments);
                syncOutputs(this);
            };
            return;
        }

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origCreated?.apply(this, arguments);
            wireNode(this, cfg);
        };

        if (cfg.generatorSockets) {
            // Prototype-level wrap: the frontend may reassign the
            // INSTANCE's onConnectionsChange after our onNodeCreated, which
            // silently detached the instance-level wrap (live-caught).
            const origConn = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (type, index, connected, ...rest) {
                origConn?.apply(this, [type, index, connected, ...rest]);
                if (type === 1 /*INPUT*/ && this.__mieAssets?.cfg?.generatorSockets) {
                    this.__mieAssets.syncSlots?.();
                }
            };
        }

        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            origConfigure?.apply(this, arguments);
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
    if (cfg.generatorSockets) {
        node.__mieAssets = { cfg };
        patchGeneratorSocketGrow(node);
        return;
    }
    const dirWidget = node.widgets?.find((w) => w.name === "library_dir");
    if (!dirWidget) return;
    node.__mieAssets = { cfg, payload: null };

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

    applyLibrary(node);
}

function applyLibrary(node, { force = false } = {}) {
    const cfg = node.__mieAssets?.cfg;
    if (!cfg || cfg.generatorSockets) return;
    fetchAssets({ force }).then((payload) => {
        if (!node.__mieAssets) return;
        node.__mieAssets.payload = payload;
        reportScanError(payload);
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
let pickerStyleEl = null;

function ensureStyles() {
    if (pickerStyleEl) return;
    pickerStyleEl = document.createElement("style");
    pickerStyleEl.textContent = [
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
    ].join("");
    document.head.appendChild(pickerStyleEl);
}

function makePreviewWidget(node) {
    if (typeof node.addDOMWidget !== "function") return;
    ensureStyles();
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

// ---- 生成器 V2：character_N / scene_N 插座自动生长 -----------------------------
// 后端在 INPUT_TYPES 里声明满额 9 个；前端裁剪到「已接线的最高位 + 1」，
// 接上尾槽才露出下一个。configure() 会按已存工作流还原更多（同
// stringFormatAutogrow 的模式）。
const generatorGrowPatched = new WeakSet();

function socketIndices(node, prefix) {
    const out = [];
    (node.inputs || []).forEach((inp, i) => {
        if (inp && typeof inp.name === "string" && inp.name.startsWith(prefix + "_")
            && /^\d+$/.test(inp.name.slice(prefix.length + 1))) {
            out.push(i);
        }
    });
    return out;
}

function patchGeneratorSocketGrow(node) {
    if (generatorGrowPatched.has(node)) return;
    generatorGrowPatched.add(node);
    const TYPES = { character: "MIE_CHARACTER_ASSET", scene: "MIE_SCENE_ASSET" };
    const MAX = 9;

    const trimTo = (prefix, keep) => {
        let indices = socketIndices(node, prefix);
        while (indices.length > keep) {
            node.removeInput(indices.pop());
            indices = socketIndices(node, prefix);
        }
    };
    const appendNext = (prefix) => {
        const indices = socketIndices(node, prefix);
        if (indices.length >= MAX) return;
        node.addInput(prefix + "_" + (indices.length + 1), TYPES[prefix]);
    };
    const sync = () => {
        for (const prefix of ["character", "scene"]) {
            const indices = socketIndices(node, prefix);
            let keep = 1; // 至少保留 character_1 / scene_1
            for (let k = 0; k < indices.length; k++) {
                if (node.inputs[indices[k]].link != null) keep = k + 2;
            }
            keep = Math.min(keep, MAX);
            trimTo(prefix, keep);
            const now = socketIndices(node, prefix);
            if (now.length < keep) appendNext(prefix);
            else if (now.length < MAX
                     && node.inputs[now[now.length - 1]].link != null) {
                appendNext(prefix);
            }
        }
        node.setDirtyCanvas?.(true, true);
    };
    if (node.__mieAssets) node.__mieAssets.syncSlots = sync;
    // 新节点只露出 character_1 / scene_1；configure 会按已存工作流还原。
    trimTo("character", 1);
    trimTo("scene", 1);
}
