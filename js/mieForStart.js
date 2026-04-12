import { app } from "../../scripts/app.js";

/**
 * MieLoopStart 前端扩展
 *
 * 根据 params_mode 切换可见的参数输入框：
 *   - int_list → 显示 int_list
 *   - string_list → 显示 string_list
 *   - json_list → 显示 json_list
 *
 * 同时隐藏内部/高级字段（resume_loop_ctx, meta_json）以保持界面整洁。
 */
app.registerExtension({
    name: "MieLoopStart|Mie",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData) return;
        if (nodeData?.category !== "🐑 MieNodes/🐑 Loop") return;
        if (nodeData?.name !== "MieLoopStart|Mie") return;

        const getWidget = (node, name) => node.widgets?.find(w => w.name === name);

        const setWidgetVisible = (w, visible) => {
            if (!w) return;
            w.hidden = !visible;
        };

        const updateVisibility = (node) => {
            if (!node) return;
            const modeWidget = getWidget(node, "params_mode");
            if (!modeWidget) return;

            const mode = modeWidget.value;
            setWidgetVisible(getWidget(node, "int_list"), mode === "int_list");
            setWidgetVisible(getWidget(node, "string_list"), mode === "string_list");
            setWidgetVisible(getWidget(node, "json_list"), mode === "json_list");
            setWidgetVisible(getWidget(node, "resume_loop_ctx"), false);
            setWidgetVisible(getWidget(node, "meta_json"), false);

            node.setDirtyCanvas?.(true, true);
        };

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origOnNodeCreated?.apply(this, arguments);

            const modeWidget = getWidget(this, "params_mode");
            if (!modeWidget) return;

            updateVisibility(this);

            const origCallback = modeWidget.callback;
            modeWidget.callback = (v, ...args) => {
                origCallback?.(v, ...args);
                updateVisibility(this);
            };
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            origOnConfigure?.apply(this, arguments);

            updateVisibility(this);
            requestAnimationFrame(() => updateVisibility(this));
        };
    },
});