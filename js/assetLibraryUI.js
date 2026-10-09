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
    const dir = (libraryDir || "").trim();
    if (!dir) return Promise.resolve(null);
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
    const dir = (node.widgets?.find((w) => w.name === "library_dir")?.value || "").trim();
    if (!dir) return;

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
