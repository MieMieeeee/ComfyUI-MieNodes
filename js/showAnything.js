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

                // Ensure the "text" widget is created only once
                if (!this.textWidget) {
                    this.textWidget = ComfyWidgets["STRING"](this, "displaytext", ["STRING", {multiline: true}], app).widget;
                    this.textWidget.inputEl.readOnly = true;
                    this.textWidget.inputEl.style.border = "none";
                    this.textWidget.inputEl.style.backgroundColor = "transparent";
                }

                // Update the value of the text widget
                this.textWidget.value = message["text"].join("");
            };
        }
    },
});