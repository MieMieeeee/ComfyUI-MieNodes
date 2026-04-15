import { app } from "../../scripts/app.js";

/**
 * MieLoopStart 前端扩展
 *
 * 根据 param_type + param_mode 切换可见的参数输入框：
 *   - int/list   → int_list
 *   - int/range  → int_range_*
 *   - float/list → float_list
 *   - float/range→ float_range_*
 *   - string     → string_list
 *   - json       → json_list
 *
 * 同时隐藏内部/高级字段（resume_loop_ctx, meta_json）以保持界面整洁。
 * 对不支持 range 的类型（string/json），会自动把 param_mode 拉回 list。
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

        const updateModeOptions = (typeWidget, modeWidget) => {
            if (!typeWidget || !modeWidget) return;
            const paramType = typeWidget.value;
            const rangeSupported = paramType === "int" || paramType === "float";

            setWidgetVisible(modeWidget, rangeSupported);
            if (!rangeSupported && modeWidget.value !== "list") {
                modeWidget.value = "list";
                modeWidget.callback?.("list");
            }
        };

        const updateVisibility = (node) => {
            if (!node) return;
            const typeWidget = getWidget(node, "param_type");
            const modeWidget = getWidget(node, "param_mode");
            if (!typeWidget || !modeWidget) return;

            updateModeOptions(typeWidget, modeWidget);

            const paramType = typeWidget.value;
            const paramMode = modeWidget.value;
            const showIntList = paramType === "int" && paramMode === "list";
            const showIntRange = paramType === "int" && paramMode === "range";
            const showFloatList = paramType === "float" && paramMode === "list";
            const showFloatRange = paramType === "float" && paramMode === "range";
            const showStringList = paramType === "string";
            const showJsonList = paramType === "json";
            const rangeSupported = paramType === "int" || paramType === "float";

            setWidgetVisible(getWidget(node, "int_list"), showIntList);
            setWidgetVisible(getWidget(node, "float_list"), showFloatList);
            setWidgetVisible(getWidget(node, "string_list"), showStringList);
            setWidgetVisible(getWidget(node, "json_list"), showJsonList);
            setWidgetVisible(getWidget(node, "int_range_start"), showIntRange);
            setWidgetVisible(getWidget(node, "int_range_end"), showIntRange);
            setWidgetVisible(getWidget(node, "int_range_step"), showIntRange);
            setWidgetVisible(getWidget(node, "float_range_start"), showFloatRange);
            setWidgetVisible(getWidget(node, "float_range_end"), showFloatRange);
            setWidgetVisible(getWidget(node, "float_range_step"), showFloatRange);
            setWidgetVisible(getWidget(node, "initial_state_json"), false);
            setWidgetVisible(getWidget(node, "resume_loop_ctx"), false);
            setWidgetVisible(getWidget(node, "meta_json"), false);

            node.setSize?.(node.computeSize?.());
            node.setDirtyCanvas?.(true, true);
        };

        const attachUpdateCallback = (node, widget) => {
            if (!widget) return;
            const origCallback = widget.callback;
            widget.callback = (v, ...args) => {
                origCallback?.(v, ...args);
                updateVisibility(node);
            };
        };

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origOnNodeCreated?.apply(this, arguments);

            const typeWidget = getWidget(this, "param_type");
            const modeWidget = getWidget(this, "param_mode");
            if (!typeWidget || !modeWidget) return;

            updateVisibility(this);

            attachUpdateCallback(this, typeWidget);
            attachUpdateCallback(this, modeWidget);
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            origOnConfigure?.apply(this, arguments);

            updateVisibility(this);
            requestAnimationFrame(() => updateVisibility(this));
        };
    },
});
