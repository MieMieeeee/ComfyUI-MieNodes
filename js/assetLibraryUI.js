import { app } from "../../scripts/app.js";

/**
 * 资产库前端辅助（docs/ASSET_LIBRARY_GUIDE.md）。
 *
 * 资产下拉与 LoadAssetBatch 的槽位行按「节点实例」从
 * POST /mie_nodes/assets/list 取选项——类级 combo 选项拿不到另一个
 * widget 的值，所以真正的列表由这里按实例填充。本文件添加的所有
 * widget 都带 {serialize:false} 且永远追加在 node.widgets 末尾：
 * litegraph 序列化与加载两侧都只处理 serialize!==false 的子集并按
 * 下标对齐，因此工作流 JSON 与加 UI 之前逐字节相同。
 *
 * LoadAssetBatch 的数据流：多行 `names` STRING widget 是序列化真相；
 * 槽位行是它的实时镜像（加载/配置后由它重建，每次选择写回它）。
 */

const ENDPOINT = "/mie_nodes/assets/list";
const NEW_NAME = "（✏️ 新名字…）";
const EMPTY_SLOT = "（空）";
const MAX_SLOTS = 9;
const KIND_CHAR = "[角色] ";
const KIND_SCENE = "[场景] ";

// nodeData.name（含 |Mie 后缀）-> 行为配置
const NODES = {
    "LoadCharacterAsset|Mie": { kinds: ["characters"] },
    "LoadSceneAsset|Mie": { kinds: ["scenes"] },
    "SaveCharacterAsset|Mie": { kinds: ["both"], save: true },
    "SaveSceneAsset|Mie": { kinds: ["scenes"], save: true },
    "ListCharacterAssets|Mie": {},
    "ListSceneAssets|Mie": {},
    "LoadAssetBatch|Mie": { slots: true },
};

// ---- 共享取数：按目录缓存 + 去重 ---------------------------------------
const cache = new Map(); // library_dir -> Promise<payload>
let debounceTimer = null;

function fetchAssets(libraryDir, { force = false } = {}) {
    // 空串 = 本机默认库，交给服务端解析（第六次修订后的语义）。
    const dir = (libraryDir || "").trim();
    if (!force && cache.has(dir)) return cache.get(dir);
    const p = fetch(ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ library_dir: dir }),
    })
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error("HTTP " + r.status))))
        .catch((e) => ({ ok: false, characters: [], scenes: [], error: String(e) }));
    cache.set(dir, p);
    return p;
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

// ---- 选项列表 ------------------------------------------------------------
function entriesFor(payload, kinds, { bare = false } = {}) {
    const out = [];
    if (!payload || !payload.ok) return out;
    const wantChar = kinds.includes("characters") || kinds.includes("both");
    const wantScene = kinds.includes("scenes") || kinds.includes("both");
    if (wantChar) for (const n of payload.characters) out.push(bare ? n : KIND_CHAR + n);
    if (wantScene) for (const n of payload.scenes) out.push(bare ? n : KIND_SCENE + n);
    return out;
}

function stripKind(entry) {
    if (entry.startsWith(KIND_CHAR)) return entry.slice(KIND_CHAR.length);
    if (entry.startsWith(KIND_SCENE)) return entry.slice(KIND_SCENE.length);
    return entry;
}

function pictureUrl(dir, name, kind) {
    return (
        "/mie_nodes/assets/picture?dir=" + encodeURIComponent(dir) +
        "&name=" + encodeURIComponent(name) +
        "&kind=" + encodeURIComponent(kind)
    );
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
        // onNodeCreated 时 library_dir 还是空串（值要等 configure 才应用），
        // 所以取数也要在这里再触发一次。
        const origConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            origConfigure?.apply(this, arguments);
            if (cfg.slots) rebuildSlots(this);
            if (cfg.save) syncNewNameVisibility(this);
            applyLibrary(this);
        };
    },
});

function wireNode(node, cfg) {
    const dirWidget = node.widgets?.find((w) => w.name === "library_dir");
    if (!dirWidget) return;
    node.__mieAssets = { cfg, payload: null, syncing: false };

    // 库目录是每台机器固定的（第六次修订）：widget 默认隐藏（仍序列化，
    // 老工作流已填的值继续生效）。「📁 库目录」按钮显隐它并显示解析结果。
    dirWidget.hidden = true;
    node.addWidget("button", "📁 库目录…", null, () => {
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
    }).serialize = false;

    if (cfg.slots) {
        // 图片选择器按钮（serialize:false，位于 widgets 末尾）。
        node.addWidget("button", "🖼 选择资产…", null, () => {
            openPicker(node);
        }).serialize = false;
    }

    // 刷新按钮（serialize:false，位于 widgets 末尾）。
    node.addWidget("button", "🔄 刷新资产列表", null, () => {
        applyLibrary(node, { force: true });
    }).serialize = false;

    // library_dir 手输/粘贴 -> 防抖取数。
    const origDirCallback = dirWidget.callback;
    dirWidget.callback = (...args) => {
        origDirCallback?.apply(dirWidget, args);
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => applyLibrary(node), 500);
    };

    if (cfg.save) {
        const nameWidget = node.widgets.find((w) => w.name === "name");
        const origNameCallback = nameWidget?.callback;
        if (nameWidget) {
            nameWidget.callback = (...args) => {
                origNameCallback?.apply(nameWidget, args);
                syncNewNameVisibility(node);
            };
        }
        syncNewNameVisibility(node);
    }

    if (cfg.slots) {
        const namesWidget = node.widgets.find((w) => w.name === "names");
        // 手改 names 文本 -> 重建槽位行（我们写回时不触发）。
        const origChanged = node.onWidgetChanged;
        node.onWidgetChanged = function (widgetName, ...rest) {
            origChanged?.apply(this, [widgetName, ...rest]);
            if (widgetName === "names" && !node.__mieAssets?.syncing) {
                rebuildSlots(this);
            }
        };
        rebuildSlots(node);
    }

    applyLibrary(node);
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

        if (cfg.slots) {
            // 槽位行只吃选项；选中的值本身存在 names 文本里。
            rebuildSlots(node);
            return;
        }
        const nameWidget = node.widgets.find((w) => w.name === "name");
        if (!nameWidget) return;
        // 单资产节点按类型过滤，用裸名字：选中值直接进 Python，无人剥前缀。
        const values = entriesFor(payload, cfg.kinds, { bare: true });
        if (cfg.save) values.unshift(NEW_NAME);
        setComboOptions(nameWidget, values);
        if (cfg.save) syncNewNameVisibility(node);
        node.setDirtyCanvas?.(true, true);
    });
}

// ---- Save 节点的 new_name 显隐（照 mieForStart.js 的 w.hidden 模式）------
function syncNewNameVisibility(node) {
    const nameWidget = node.widgets?.find((w) => w.name === "name");
    const newNameWidget = node.widgets?.find((w) => w.name === "new_name");
    if (!nameWidget || !newNameWidget) return;
    const isSentinel = (nameWidget.value || "").includes(NEW_NAME);
    newNameWidget.hidden = !isSentinel;
    node.setDirtyCanvas?.(true, true);
}

// ---- LoadAssetBatch 槽位行 -------------------------------------------------
// 槽位 combo 是纯 UI 镜像（serialize:false、永远在 widgets 末尾）；
// 序列化真相是多行 `names` widget。
function currentSlotOptions(node) {
    const payload = node.__mieAssets?.payload;
    const values = [EMPTY_SLOT];
    if (payload?.ok) {
        values.push(...entriesFor(payload, ["both"]));
    } else {
        // 首次扫描未回来之前，names 里已有的名字也能显示。
        const namesWidget = node.widgets.find((w) => w.name === "names");
        for (const line of (namesWidget?.value || "").split("\n")) {
            const t = line.trim();
            if (t && !values.includes(KIND_CHAR + t)) values.push(KIND_CHAR + t);
        }
    }
    return values;
}

function rebuildSlots(node) {
    const ui = node.__mieAssets;
    if (!ui) return;
    // 移除旧槽位（widget.__mieSlot 标记）。
    for (let i = node.widgets.length - 1; i >= 0; i--) {
        if (node.widgets[i].__mieSlot != null) node.removeWidget(node.widgets[i]);
    }
    const namesWidget = node.widgets.find((w) => w.name === "names");
    const picked = (namesWidget?.value || "")
        .split("\n")
        .map((s) => s.trim())
        .filter(Boolean);
    // 已知 kind 的名字显示带前缀；未知 kind（老工作流）按角色显示。
    const known = ui.payload?.ok
        ? new Map([
              ...ui.payload.characters.map((n) => [n, KIND_CHAR + n]),
              ...ui.payload.scenes.map((n) => [n, KIND_SCENE + n]),
          ])
        : null;
    const values = currentSlotOptions(node);
    const addSlot = (index, value) => {
        const w = node.addWidget(
            "combo",
            `槽位 ${index + 1} = <Picture ${index + 1}>`,
            value,
            () => syncSlotsToNames(node),
            { values: values.slice(), serialize: false }
        );
        w.serialize = false;
        w.__mieSlot = index;
    };
    for (let i = 0; i < picked.length && i < MAX_SLOTS; i++) {
        addSlot(i, known ? known.get(picked[i]) || KIND_CHAR + picked[i] : KIND_CHAR + picked[i]);
    }
    if (picked.length < MAX_SLOTS) addSlot(picked.length, EMPTY_SLOT);
    node.setDirtyCanvas?.(true, true);
}

function syncSlotsToNames(node) {
    const ui = node.__mieAssets;
    if (!ui || ui.syncing) return;
    ui.syncing = true;
    try {
        const names = [];
        for (const w of node.widgets) {
            if (w.__mieSlot == null) continue;
            const v = w.value || "";
            if (v && v !== EMPTY_SLOT) names.push(stripKind(v));
        }
        const namesWidget = node.widgets.find((w) => w.name === "names");
        if (namesWidget) namesWidget.value = names.join("\n");
        rebuildSlots(node);
    } finally {
        ui.syncing = false;
    }
}

// ---- 图片选择器（LoadAssetBatch 专用）------------------------------------
// 固定定位的原生 DOM 弹窗：卡片网格（图 + 名字 + 类型徽章），点卡片追加到
// 下一个槽位，底部按槽位顺序显示可删 chip。所有改动都写回 names 文本——
// names 是唯一序列化真相，弹窗和槽位行只是它的两张脸。
let pickerStyleEl = null;

function ensurePickerStyles() {
    if (pickerStyleEl) return;
    pickerStyleEl = document.createElement("style");
    pickerStyleEl.textContent = `
.mie-picker-backdrop{position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:99999;
  display:flex;align-items:center;justify-content:center}
.mie-picker{background:var(--comfy-menu-bg,#2a2a2a);color:var(--input-text,#ddd);
  border:1px solid var(--border-color,#444);border-radius:10px;width:min(880px,92vw);
  max-height:86vh;display:flex;flex-direction:column;box-shadow:0 12px 48px rgba(0,0,0,.5);
  font-family:inherit}
.mie-picker header{display:flex;align-items:center;gap:10px;padding:10px 14px;
  border-bottom:1px solid var(--border-color,#444)}
.mie-picker header .t{font-size:15px;font-weight:600}
.mie-picker header .d{font-size:11px;opacity:.65;flex:1;overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
.mie-picker header button{cursor:pointer}
.mie-picker .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
  gap:10px;padding:12px;overflow-y:auto}
.mie-card{border:1px solid var(--border-color,#444);border-radius:8px;overflow:hidden;
  cursor:pointer;background:rgba(255,255,255,.03);transition:border-color .12s}
.mie-card:hover{border-color:#7f9cf5}
.mie-card img{width:100%;height:104px;object-fit:cover;display:block;background:#111}
.mie-card .cap{display:flex;align-items:center;gap:6px;padding:6px 8px;font-size:12px}
.mie-card .kind{font-size:10px;padding:1px 6px;border-radius:6px;background:#3b4a6b;
  color:#cdd8f8;flex-shrink:0}
.mie-card .nm{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mie-card .in{margin-left:auto;font-size:10px;opacity:.55;flex-shrink:0}
.mie-picker footer{display:flex;flex-wrap:wrap;gap:6px;align-items:center;
  padding:10px 14px;border-top:1px solid var(--border-color,#444);min-height:44px}
.mie-chip{display:flex;align-items:center;gap:6px;border:1px solid var(--border-color,#555);
  border-radius:14px;padding:3px 10px;font-size:12px}
.mie-chip img{width:26px;height:26px;object-fit:cover;border-radius:50%}
.mie-chip .x{cursor:pointer;opacity:.6;font-weight:700}
.mie-chip .x:hover{opacity:1;color:#f88}
.mie-empty{padding:26px;text-align:center;opacity:.6;font-size:13px}
`;
    document.head.appendChild(pickerStyleEl);
}

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
        // 所以写完要显式重建槽位行。
        namesWidget.value = names.join("\n");
        namesWidget.callback?.(namesWidget.value);
    } finally {
        if (ui) ui.syncing = false;
    }
    rebuildSlots(node);
    node.setDirtyCanvas?.(true, true);
}

function kindBadge(kind) {
    return kind === "scene" ? "场景" : "角色";
}

function openPicker(node) {
    ensurePickerStyles();
    closePicker();
    const ui = node.__mieAssets;
    const payload = ui?.payload;
    const dir = payload?.library_dir
        || (node.widgets.find((w) => w.name === "library_dir")?.value || "").trim();

    const backdrop = document.createElement("div");
    backdrop.className = "mie-picker-backdrop";
    const panel = document.createElement("div");
    panel.className = "mie-picker";

    const header = document.createElement("header");
    header.innerHTML = `<span class="t">🖼 选择资产（点卡片加入下一个槽位，上限 ${MAX_SLOTS}）</span>
        <span class="d"></span>`;
    const dirLabel = header.querySelector(".d");
    dirLabel.textContent = dir ? "库：" + dir : "库：默认（未扫描）";
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
            const card = document.createElement("div");
            card.className = "mie-card";
            card.title = slot ? "已在槽位 " + slot : "点击加入下一个槽位";
            const img = document.createElement("img");
            img.loading = "lazy";
            img.src = pictureUrl(dir || ".", it.name, it.kind);
            img.alt = it.name;
            const cap = document.createElement("div");
            cap.className = "cap";
            cap.innerHTML = `<span class="kind">${kindBadge(it.kind)}</span>
                <span class="nm"></span>`;
            cap.querySelector(".nm").textContent = it.name;
            if (slot) {
                const tag = document.createElement("span");
                tag.className = "in";
                tag.textContent = "槽位 " + slot;
                cap.append(tag);
            }
            card.append(img, cap);
            card.addEventListener("click", () => {
                const cur = currentSlotNames(node);
                if (cur.includes(it.name)) return; // 一个资产一个槽位
                if (cur.length >= MAX_SLOTS) {
                    app.extensionManager?.toast?.add?.({
                        severity: "warn", summary: "槽位已满",
                        detail: `最多 ${MAX_SLOTS} 个（<Picture N> 上限）`, life: 3000,
                    });
                    return;
                }
                writeSlotNames(node, cur.concat([it.name]));
                render();
            });
            grid.append(card);
        }
        // footer：当前槽位顺序 + 逐个删除
        footer.innerHTML = "";
        const pickedOrder = currentSlotNames(node);
        const kindOfName = (n) => {
            if (payload?.characters.includes(n)) return "character";
            if (payload?.scenes.includes(n)) return "scene";
            return "character";
        };
        if (!pickedOrder.length) {
            footer.innerHTML = '<span style="opacity:.55;font-size:12px">（还没有选择任何资产）</span>';
        }
        pickedOrder.forEach((n, i) => {
            const chip = document.createElement("span");
            chip.className = "mie-chip";
            const img = document.createElement("img");
            img.src = pictureUrl(dir || ".", n, kindOfName(n));
            const label = document.createElement("span");
            label.textContent = `${i + 1}. ${n}`;
            const x = document.createElement("span");
            x.className = "x";
            x.textContent = "✕";
            x.title = "移出槽位 " + (i + 1);
            x.addEventListener("click", () => {
                const cur = currentSlotNames(node);
                cur.splice(i, 1);
                writeSlotNames(node, cur);
                render();
            });
            chip.append(img, label, x);
            footer.append(chip);
        });
    }

    refreshBtn.addEventListener("click", async () => {
        refreshBtn.textContent = "⏳";
        const dirWidget = node.widgets.find((w) => w.name === "library_dir");
        const fresh = await fetchAssets(dirWidget?.value || "", { force: true });
        if (node.__mieAssets) {
            node.__mieAssets.payload = fresh;
            if (fresh) dirLabel.textContent = "库：" + (fresh.library_dir || dir);
        }
        applyLibrary(node, { force: true }); // 同步节点上的下拉选项
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
