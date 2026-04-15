import { app } from "../../scripts/app.js";

const HIDE_CONFIG = {
    "MieLoopBodyOut|Mie": ["state_json"],
    "MieLoopEnd|Mie": ["state_json"],
    "MieLoopStateSet|Mie": ["base_state_json"],
    "MieLoopStateGetImage|Mie": ["fallback_image"],
};

app.registerExtension({
    name: "MieLoopAdvancedFields|Mie",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData) return;
        if (nodeData?.category !== "🐑 MieNodes/🐑 Loop") return;

        const hiddenNames = HIDE_CONFIG[nodeData?.name];
        if (!hiddenNames?.length) return;

        const getWidget = (node, name) => node.widgets?.find(w => w.name === name);
        const hideWidgets = (node) => {
            if (!node) return;
            for (const name of hiddenNames) {
                const widget = getWidget(node, name);
                if (!widget) continue;
                widget.hidden = true;
            }
            node.setSize?.(node.computeSize?.());
            node.setDirtyCanvas?.(true, true);
        };

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origOnNodeCreated?.apply(this, arguments);
            hideWidgets(this);
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            origOnConfigure?.apply(this, arguments);
            hideWidgets(this);
            requestAnimationFrame(() => hideWidgets(this));
        };
    },
});
