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
            // 生成器：恢复已存接线后刷新统计标签（socketGrow 的 sync 也会做）。
            if (cfg.generatorSockets && this.__mieAssets?.syncSlots) {
                this.__mieAssets.syncSlots();
            }
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
    app.extensionManager?.toast?.add?.({
        severity: "success", summary: "资产已保存",
        detail: "资产库写入完成；画布上的资产节点已刷新", life: 3500,
    });
    // 先重扫库（拿到新的名单和扫描戳），再刷新各节点的预览——顺序反了
    // 的话预览会拿旧 payload 渲染，看起来“保存没生效”。
    fetchAssets({ force: true }).then((payload) => {
        for (const other of nodes) {
            if (!other.__mieAssets) continue;
            other.__mieAssets.payload = payload;
            if (other.__mieAssets.cfg?.preview) refreshLoadPreview(other);
            if (other.__mieAssets.cfg?.save) refreshSavePreview(other, { autofill: false });
        }
    });
});

app.api?.addEventListener?.("execution_success", () => {
    // A finished queue may have written assets. Only spend a re-scan when
    // at least one save node is on the canvas.
    const nodes = app.graph?._nodes || [];
    const hasSave = nodes.some((n) => n.__mieAssets?.cfg?.save);
    if (!hasSave) return;
    cache.clear();
    fetchAssets({ force: true }).then((payload) => {
        for (const other of nodes) {
            if (!other.__mieAssets) continue;
            other.__mieAssets.payload = payload;
            if (other.__mieAssets.cfg?.preview) refreshLoadPreview(other);
            if (other.__mieAssets.cfg?.save) refreshSavePreview(other, { autofill: false });
        }
    });
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
    const openBtn = node.addWidget("button", "📂 打开文件夹", null, async () => {
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
    });
    openBtn.serialize = false;

    const refreshBtn = node.addWidget("button", "🔄 刷新资产列表", null, () => {
        applyLibrary(node, { force: true });
    });
    refreshBtn.serialize = false;

    // 库操作按钮置顶：addWidget 默认排在所有输入 widget 之后，这里移到
    // 最上面。两个按钮都 serialize:false，前端 litegraph 序列化/加载两侧
    // 都按 serialize 过滤后再顺序映射（1.53/1.55 已核实），位置不影响
    // 老工作流 widgets_values 的下标对应。原地重排，不换数组对象。
    const rest = node.widgets.filter((w) => w !== openBtn && w !== refreshBtn);
    node.widgets.length = 0;
    node.widgets.push(openBtn, refreshBtn, ...rest);

    if (cfg.save) {
        // 保存是 OUTPUT_NODE：点「执行/Queue」才真正落盘。这个按钮就是
        // 那一步——排队当前图（等价按 Queue），保存节点会随队列执行。
        if (typeof node.addDOMWidget === "function") {
            const wrap = document.createElement("div");
            wrap.className = "mie-save-apply";
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "mie-media-btn mie-save-btn";
            btn.innerHTML = "<span class='mie-media-btn-icon'>💾</span> 应用保存（执行）";
            btn.title = "把当前 widget/上传选择排入 ComfyUI 队列执行——保存节点是输出节点，队列跑完即写入资产库（等价于点 Queue）";
            btn.addEventListener("click", () => {
                btn.disabled = true;
                btn.classList.add("busy");
                try {
                    app.queuePrompt();
                    app.extensionManager?.toast?.add?.({
                        severity: "info", summary: "已加入队列",
                        detail: "保存节点将在本次执行中写入资产库",
                        life: 3000,
                    });
                } catch (err) {
                    app.extensionManager?.toast?.add?.({
                        severity: "error", summary: "排队失败",
                        detail: String(err).slice(0, 160), life: 5000,
                    });
                } finally {
                    setTimeout(() => {
                        btn.disabled = false;
                        btn.classList.remove("busy");
                    }, 600);
                }
            });
            wrap.append(btn);
            const w = node.addDOMWidget("应用保存", "mie-save-apply", wrap, {
                serialize: false, hideOnZoom: false,
                getHeight: () => Math.max(40, wrap.offsetHeight + 6),
            });
            w.serialize = false;
        }
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
        makeMediaPicker(
            node, "image_upload", "image",
            ".png,.jpg,.jpeg,.webp,image/png,image/jpeg,image/webp",
            "/upload/image", "image", "🖼", "定妆图"
        );
        if (cfg.save === "character") {
            makeMediaPicker(
                node, "audio_upload", "audio",
                ".wav,.mp3,audio/wav,audio/mpeg",
                "/mie_nodes/assets/upload_audio", "audio", "🎵", "音色样本"
            );
        }
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

// ---- 上传控件：选择文件 / 拖放 / 粘贴，全走我们自己的 DOM --------------------
// 每种媒体一个 DOM 行：[好看的文件下拉] + [醒目的上传按钮]。选中/上传的
// 文件名写进隐藏的 STRING widget（image_upload / audio_upload），
// Python 端照常读它；连线插座仍然优先。
function fetchUploadFiles(kind) {
    return fetch("/mie_nodes/assets/upload_files?kind=" + kind)
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error("HTTP " + r.status))))
        .then((j) => (j.ok ? j.files : []))
        .catch(() => []);
}

function makeMediaPicker(node, stateName, kind, accept, endpoint, field, icon, label) {
    if (typeof node.addDOMWidget !== "function") return;
    const stateWidget = node.widgets.find((w) => w.name === stateName);
    if (!stateWidget) return;
    stateWidget.hidden = true;  // state only — the DOM row is the UI

    const wrap = document.createElement("div");
    wrap.className = "mie-media";
    const head = document.createElement("div");
    head.className = "mie-media-head";
    const iconEl = document.createElement("span");
    iconEl.className = "mie-media-icon";
    iconEl.textContent = icon;
    const titleEl = document.createElement("span");
    titleEl.className = "mie-media-title";
    titleEl.textContent = label;
    head.append(iconEl, titleEl);
    const row = document.createElement("div");
    row.className = "mie-media-row";

    const select = document.createElement("select");
    select.className = "mie-media-select";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "（选择文件或点右侧上传）";
    select.append(placeholder);
    // 卡片里不放图片预览（2026-10-10 用户裁决）：选中/上传的图片由节点
    // 底部的预览面板显示，卡片只负责选文件和上传。

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "mie-media-btn";
    btn.innerHTML = "<span class='mie-media-btn-icon'>📤</span> 上传";
    btn.title = "点击选择文件；也可以把文件拖到这一行，或聚焦后 Ctrl+V 粘贴";

    const input = document.createElement("input");
    input.type = "file";
    input.accept = accept;
    input.style.display = "none";
    row.append(select, btn, input);
    wrap.append(head, row);

    const setSelection = (name) => {
        if (!name) {
            select.value = "";
        } else {
            if (![...select.options].some((o) => o.value === name)) {
                const opt = document.createElement("option");
                opt.value = name;
                opt.textContent = name;
                select.append(opt);
            }
            select.value = name;
        }
        stateWidget.value = name || "";
        stateWidget.callback?.(stateWidget.value);
        node.setDirtyCanvas?.(true, true);
    };
    select.addEventListener("change", () => setSelection(select.value));

    const upload = async (file) => {
        btn.disabled = true;
        btn.classList.add("busy");
        const old = btn.innerHTML;
        btn.innerHTML = "⏳ …";
        try {
            const form = new FormData();
            form.append(field, file, file.name);
            const r = await fetch(endpoint, { method: "POST", body: form });
            const j = await r.json();
            const name = j && (j.name || (j.data && j.data.name));
            if (!r.ok || !name) throw new Error((j && j.error) || ("HTTP " + r.status));
            const sub = j.subfolder ? j.subfolder + "/" + name : name;
            setSelection(sub);
            app.extensionManager?.toast?.add?.({
                severity: "success", summary: "已上传", detail: sub, life: 2500,
            });
        } catch (err) {
            app.extensionManager?.toast?.add?.({
                severity: "error", summary: "上传失败",
                detail: String(err).slice(0, 160), life: 5000,
            });
        } finally {
            btn.disabled = false;
            btn.classList.remove("busy");
            btn.innerHTML = old;
        }
    };
    btn.addEventListener("click", () => input.click());
    input.addEventListener("change", () => {
        if (input.files && input.files[0]) upload(input.files[0]);
        input.value = "";
    });
    row.addEventListener("dragover", (e) => { e.preventDefault(); wrap.classList.add("drag"); });
    row.addEventListener("dragleave", () => wrap.classList.remove("drag"));
    row.addEventListener("drop", (e) => {
        e.preventDefault();
        wrap.classList.remove("drag");
        const f = e.dataTransfer?.files?.[0];
        if (f) upload(f);
    });
    wrap.tabIndex = 0;
    wrap.addEventListener("paste", (e) => {
        const f = e.clipboardData?.files?.[0];
        if (f) { e.preventDefault(); upload(f); }
    });

    // Initial list + restore a saved selection.
    fetchUploadFiles(kind).then((files) => {
        for (const f of files) {
            const opt = document.createElement("option");
            opt.value = f;
            opt.textContent = f;
            select.append(opt);
        }
        if (stateWidget.value) setSelection(stateWidget.value);
    });

    const widget = node.addDOMWidget(label, "mie-media-picker", wrap, {
        serialize: false, hideOnZoom: false,
        // 缩略图出现/消失时卡片实际高度变化，静态高度会与下方 widget 重叠。
        getHeight: () => Math.max(58, wrap.offsetHeight + 6),
    });
    widget.serialize = false;
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
        const fields = kind === "character" ? ["description", "voice"] : ["description"];
        for (const field of fields) {
            const widget = node.widgets.find((w) => w.name === field);
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
        ".mie-media{padding:2px 0 4px;border:1px solid var(--border-color,#444);",
        "border-radius:10px;background:rgba(127,156,245,.06)}",
        ".mie-media.drag{border-color:#7f9cf5;background:rgba(127,156,245,.14)}",
        ".mie-media-head{display:flex;align-items:center;gap:6px;padding:4px 10px 0;",
        "font-size:12px;font-weight:600;color:var(--descrip-text,#c9d4e8)}",
        ".mie-media-icon{font-size:14px}",
        ".mie-media-row{display:flex;gap:6px;padding:6px 8px 8px;align-items:stretch}",
        ".mie-media-select{flex:1;min-width:0;font-size:12px;padding:4px 6px;",
        "border:1px solid var(--border-color,#555);border-radius:8px;",
        "background:var(--comfy-input-bg,#1e1e1e);color:var(--input-text,#ddd)}",
        ".mie-media-btn{cursor:pointer;font-size:12px;font-weight:700;white-space:nowrap;",
        "padding:5px 14px;border:none;border-radius:8px;color:#fff;",
        "background:linear-gradient(135deg,#4f7cff,#7f5cff);",
        "box-shadow:0 1px 6px rgba(79,124,255,.45);transition:filter .12s,transform .06s}",
        ".mie-media-btn:hover{filter:brightness(1.15)}",
        ".mie-media-btn:active{transform:translateY(1px)}",
        ".mie-media-btn.busy{filter:saturate(.4) brightness(.9);cursor:wait}",
        ".mie-save-apply{padding:2px 0}",
        ".mie-save-apply .mie-save-btn{width:100%;padding:7px 14px;font-size:13px;", 
        "background:linear-gradient(135deg,#2f9e6e,#4f7cff)}",
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
        // 预览内容异步填充（图片加载、文字回填），高度跟随实际内容。
        getHeight: () => Math.max(96, wrap.offsetHeight + 6),
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
    // 缓存键用图片自己的内容戳（meta.picture_*），不用扫描时间戳：保存
    // 换图后 stamp 必变、URL 必变，浏览器缓存里的旧图不可能再出现。
    const stamp =
        meta.picture_size != null
            ? meta.picture_size + "-" + meta.picture_mtime_ns
            : ts;
    img.src = thumbUrl(meta.name, kind, stamp);
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
        // 2026-10-10 ruling: NO consistency/overwrite judgments on the
        // save node — the preview shows what is IN THE LIBRARY, full stop.
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
    }
    wrap.append(img, body);
    // 内容异步变化后高度变了，让画布重排，避免预览面板压住下方 widget。
    img.addEventListener("load", () => relayoutNode(node));
    relayoutNode(node);
}

function kindName(kind) {
    return kind === "scene" ? KIND_SCENE : KIND_CHAR;
}

// 内容异步变化（图片加载、正文回填）后重排：画布重绘 + 节点外框只放大
// 不缩小。宽度永远保持用户当前值——computeSize 返回的是最小宽度
// （~240px），直接 setSize 会把节点压窄、挤坏卡片里的横排布局。
function relayoutNode(node) {
    node.setDirtyCanvas?.(true, true);
    try {
        const need = node.computeSize?.();
        if (need && need[1] > (node.size?.[1] || 0) + 1) {
            node.setSize?.([node.size[0], need[1]]);
        }
    } catch (_) { /* computeSize 在部分节点上不可用，忽略 */ }
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
    const updateCountLabel = () => {
        // 实时统计：接了几个角色/场景 = 多少 picture / 多少 audio。
        let chars = 0, scenes = 0;
        for (const prefix of ["character", "scene"]) {
            for (const i of socketIndices(node, prefix)) {
                if (node.inputs[i].link != null) {
                    if (prefix === "character") chars += 1; else scenes += 1;
                }
            }
        }
        if (!node.__mieCountLabel) {
            node.__mieCountLabel = node.addWidget(
                "button", "📊 0 图 · 0 音", null, () => {}
            );
            node.__mieCountLabel.serialize = false;
        }
        node.__mieCountLabel.name =
            "📊 " + (chars + scenes) + " 图 · " + chars + " 音" + (scenes ? "（含场景 " + scenes + "）" : "");
    };
    const sync = () => {
        updateCountLabel();
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
    updateCountLabel();
}
