import {app} from "../../scripts/app.js";
import {ComfyWidgets} from "../../scripts/widgets.js";

// Displays input text on a node
app.registerExtension({
    name: "ShowAnything|Mie",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData && nodeData.name === "ShowAnything|Mie") {
            console.log("ShowAnything|Mie");
            const onExecuted = nodeType.prototype.onExecuted;

            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                if (this.widgets) {
                    for (let i = 1; i < this.widgets.length; i++) {
                        this.widgets[i].onRemove?.();
                    }
                    this.widgets.length = 1;
                }

                console.log(message);

                // Check if the "text" widget already exists.
                let textWidget = this.widgets && this.widgets.find(w => w && w.name === "displaytext");

                console.log(textWidget);

                if (!textWidget) {
                    textWidget = ComfyWidgets["STRING"](this, "displaytext", ["STRING", {multiline: true}], app).widget;
                    textWidget.inputEl.readOnly = true;
                    textWidget.inputEl.style.border = "none";
                    textWidget.inputEl.style.backgroundColor = "transparent";
                }

                console.log(textWidget);

                textWidget.value = message["text"].join("");
            };
        }
    },
});