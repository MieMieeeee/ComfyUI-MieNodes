import {app} from "../../scripts/app.js";
import {ComfyWidgets} from "../../scripts/widgets.js";

app.registerExtension({
    name: "ShowAnything|Mie",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData || nodeData?.category !== "🐑 MieNodes/🐑 Common") {
            return;
        }

        if (nodeData?.name === "ShowAnything|Mie") {
            const onExecuted = nodeType.prototype.onExecuted;

            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                if (this.widgets) {
                    for (let i = 1; i < this.widgets.length; i++) {
                        this.widgets[i].onRemove?.();
                    }
                    this.widgets.length = 1;
                }

                // Check if the "text" widget already exists.
                let textWidget = Array.isArray(this.widgets) && this.widgets.find(w => w.name === "displaytext");
                if (!textWidget) {
                    textWidget = ComfyWidgets["STRING"](this, "displaytext", ["STRING", {multiline: true}], app).widget;
                    textWidget.inputEl.readOnly = true;
                    textWidget.inputEl.style.border = "none";
                    textWidget.inputEl.style.backgroundColor = "transparent";
                }
                textWidget.value = message["text"].join("");
            };
        }
    },
});