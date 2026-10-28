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
            const onExecuted = nodeType.prototype.onExecuted;

            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                if (!this.textWidget) {
                    this.textWidget = ComfyWidgets["STRING"](this, "displaytext", ["STRING", {multiline: true}], app).widget;
                    this.textWidget.inputEl.readOnly = true;
                    this.textWidget.inputEl.style.border = "none";
                    this.textWidget.inputEl.style.backgroundColor = "transparent";
                }

                const text = message["text"];
                this.textWidget.value = Array.isArray(text) ? text.join("") : (text || "");
            };
        }
    },
});
