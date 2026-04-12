import {app} from "../../scripts/app.js";

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

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const getWidget = (name) => this.widgets?.find(w => w.name === name);
            const modeWidget = getWidget("params_mode");
            if (!modeWidget) return;

            const intListWidget = getWidget("int_list");
            const stringListWidget = getWidget("string_list");
            const jsonListWidget = getWidget("json_list");
            const resumeWidget = getWidget("resume_loop_ctx");
            const metaWidget = getWidget("meta_json");

            const setWidgetVisible = (w, visible) => {
                if (!w) return;
                w.hidden = !visible;
            };

            const updateVisibility = () => {
                const mode = modeWidget.value;
                setWidgetVisible(intListWidget, mode === "int_list");
                setWidgetVisible(stringListWidget, mode === "string_list");
                setWidgetVisible(jsonListWidget, mode === "json_list");
                // resume_loop_ctx 和 meta_json 是内部/高级字段，默认隐藏
                setWidgetVisible(resumeWidget, false);
                setWidgetVisible(metaWidget, false);
                this.setDirtyCanvas(true);
            };

            updateVisibility();
            const origCallback = modeWidget.callback;
            modeWidget.callback = (v) => {
                origCallback?.(v);
                updateVisibility();
            };
        };
    },
});
