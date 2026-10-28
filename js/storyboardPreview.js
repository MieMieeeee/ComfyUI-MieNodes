import {app} from "../../scripts/app.js";
import {ComfyWidgets} from "../../scripts/widgets.js";

// Paint the loaded script asset on the node itself: cast, every event,
// per-shot durations, the delivered total and the render-risk warnings.
// The node's third output carries the text (see LoadStoryboardAsset).
app.registerExtension({
    name: "LoadStoryboardAsset|Mie",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData || nodeData?.category !== "🐑 MieNodes/🐑 Assets") {
            return;
        }

        if (nodeData?.name === "LoadStoryboardAsset|Mie") {
            const PLACEHOLDER = "（运行后这里显示剧本内容：角色、逐条事件、每镜时长、交付总时长与渲染风险）\nRun to preview the loaded script asset.";

            // Create the read-only panel when the node appears, so an
            // un-run node explains itself instead of looking broken.
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                this.ensureStoryboardTextWidget?.(app);
            };

            nodeType.prototype.ensureStoryboardTextWidget = function (appRef) {
                if (!this.textWidget) {
                    this.textWidget = ComfyWidgets["STRING"](this, "displaytext", ["STRING", {multiline: true}], appRef).widget;
                    this.textWidget.inputEl.readOnly = true;
                    this.textWidget.inputEl.style.border = "none";
                    this.textWidget.inputEl.style.backgroundColor = "transparent";
                    this.textWidget.value = PLACEHOLDER;
                }
                return this.textWidget;
            };

            const onExecuted = nodeType.prototype.onExecuted;

            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                const widget = this.ensureStoryboardTextWidget(app);
                const text = message["text"];
                widget.value = Array.isArray(text) ? text.join("") : (text || "");
            };
        }
    },
});
