import {app} from "../../scripts/app.js";
import {ComfyWidgets} from "../../scripts/widgets.js";

app.registerExtension({
    name: "ClassicAspectRatio|Mie",
    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (!nodeData || nodeData?.category !== "🐑 MieNodes/🐑 Common") return;
        if (nodeData?.name !== "ClassicAspectRatio|Mie") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const ratioWidget = this.widgets?.find(w => w.name === "ratio");
            if (!ratioWidget) return;

            const RES_BY_RATIO = {
                "1:1": [
                    "512x512 ( 0.25MP )", "768x768 ( 0.56MP )", "1024x1024 ( 1MP )", "1280x1280 ( 1.56MP )", "1536x1536 ( 2.25MP )", "2048x2048 ( 4MP )",
                ],
                "2:3": [
                    "640x960 ( 0.59MP )", "768x1152 ( 0.84MP )", "832x1248 ( 0.99MP )", "1024x1536 ( 1.5MP )", "1248x1872 ( 2.23MP )", "1664x2496 ( 3.96MP )",
                ],
                "3:2": [
                    "960x640 ( 0.59MP )", "1152x768 ( 0.84MP )", "1248x832 ( 0.99MP )", "1536x1024 ( 1.5MP )", "1872x1248 ( 2.23MP )", "2496x1664 ( 3.96MP )",
                ],
                "3:4": [
                    "480x640 ( 0.29MP )", "720x960 ( 0.66MP )", "864x1152 ( 0.95MP )", "1104x1472 ( 1.55MP )", "1296x1728 ( 2.14MP )", "1728x2304 ( 3.8MP )",
                ],
                "4:3": [
                    "640x480 ( 0.29MP )", "960x720 ( 0.66MP )", "1152x864 ( 0.95MP )", "1472x1104 ( 1.55MP )", "1728x1296 ( 2.14MP )", "2304x1728 ( 3.8MP )",
                ],
                "7:9": [
                    "448x576 ( 0.25MP )", "560x720 ( 0.38MP )", "896x1152 ( 0.98MP )", "1120x1440 ( 1.54MP )", "1344x1728 ( 2.21MP )", "1792x2304 ( 3.94MP )",
                ],
                "9:7": [
                    "576x448 ( 0.25MP )", "720x560 ( 0.38MP )", "1152x896 ( 0.98MP )", "1440x1120 ( 1.54MP )", "1728x1344 ( 2.21MP )", "2304x1792 ( 3.94MP )",
                ],
                "9:16": [
                    "432x768 ( 0.32MP )", "576x1024 ( 0.56MP )", "720x1280 ( 0.88MP )", "864x1536 ( 1.27MP )", "1152x2048 ( 2.25MP )", "1512x2688 ( 3.88MP )",
                ],
                "16:9": [
                    "768x432 ( 0.32MP )", "1024x576 ( 0.56MP )", "1280x720 ( 0.88MP )", "1536x864 ( 1.27MP )", "2048x1152 ( 2.25MP )", "2688x1512 ( 3.88MP )",
                ],
                "9:21": [
                    "384x896 ( 0.33MP )", "432x1008 ( 0.42MP )", "576x1344 ( 0.74MP )", "720x1680 ( 1.15MP )", "864x2016 ( 1.66MP )", "1312x3072 ( 3.84MP )",
                ],
                "21:9": [
                    "896x384 ( 0.33MP )", "1008x432 ( 0.42MP )", "1344x576 ( 0.74MP )", "1680x720 ( 1.15MP )", "2016x864 ( 1.66MP )", "3072x1312 ( 3.84MP )",
                ],
            };

            const updateResolutionWidget = () => {
                const ratio = ratioWidget.value;
                const choices = RES_BY_RATIO[ratio] || RES_BY_RATIO["1:1"];

                const resWidget = this.widgets?.find(w => w.name === "resolution");
                if (!resWidget) return;

                if (!resWidget.options) {
                    resWidget.options = {};
                }
                
                resWidget.options.values = choices;

                if (!choices.includes(resWidget.value)) {
                    resWidget.value = choices[0];
                }
                
                this.setDirtyCanvas(true);
            };

            // Initialize
            // We need to wait a bit because sometimes widgets are not fully populated immediately in onNodeCreated if they are dynamic? 
            // Actually for standard nodes they should be there.
            updateResolutionWidget();

            const origCallback = ratioWidget.callback;
            ratioWidget.callback = (v) => {
                origCallback?.(v);
                updateResolutionWidget();
            };
        };
    },
});
