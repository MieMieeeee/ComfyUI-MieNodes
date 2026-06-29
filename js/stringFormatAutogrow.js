/**
 * StringFormat|Mie autogrow extension.
 *
 * Mirrors the Bernini Conditioning UX: the value_<N> slots grow one at a time
 * as the user wires connections into the last visible slot. The backend
 * (nodes/common/string_ops.py) declares MAX_FORMAT_VALUES = 16 optional
 * STRING inputs up front so the workflow JSON is well-defined; this
 * extension trims the visible set to DEFAULT_FORMAT_VALUES = 2 on a fresh
 * node and appends one more slot whenever the trailing slot gets a link.
 *
 * Workflow-load behaviour: on a saved workflow with N > DEFAULT slots,
 * the LiteGraph `configure()` call rewrites `this.inputs` from the saved
 * data, so any trim done in `onNodeCreated` is harmlessly replaced. On a
 * fresh drop only `onNodeCreated` runs and the node starts with the
 * trimmed view.
 *
 * The slot layout itself (name prefix `value_`, type `STRING`, link-null
 * defaults) matches what the Python class declares in INPUT_TYPES, so
 * addInput()/removeInput() round-trips through the saved workflow JSON
 * without any schema reconciliation on the backend.
 */

import { app } from "../../scripts/app.js";

const NODE_NAME = "StringFormat|Mie";
const MAX_VALUES = 16;
const DEFAULT_VALUES = 2;
const PREFIX = "value_";
const INPUT_TYPE = "STRING";
// LiteGraph.INPUT (avoid importing LiteGraph just for the constant).
const LT_INPUT = 1;

function valueInputIndices(node) {
    const out = [];
    if (!node || !Array.isArray(node.inputs)) return out;
    for (let i = 0; i < node.inputs.length; i++) {
        const inp = node.inputs[i];
        if (inp && typeof inp.name === "string" && inp.name.startsWith(PREFIX)) {
            out.push(i);
        }
    }
    return out;
}

function trimTrailingTo(node, keep) {
    // Remove value_* inputs from the highest index down until `keep` remain.
    // Iterating from the top keeps lower indices valid as we splice.
    let indices = valueInputIndices(node);
    while (indices.length > keep) {
        const idx = indices.pop();
        node.removeInput(idx);
        indices = valueInputIndices(node);
    }
    node.setDirtyCanvas?.(true, true);
}

function appendNextValueSlot(node) {
    const indices = valueInputIndices(node);
    if (indices.length >= MAX_VALUES) return false;
    const nextIndex = indices.length;
    node.addInput(PREFIX + nextIndex, INPUT_TYPE);
    node.setDirtyCanvas?.(true, true);
    return true;
}

function growIfLastConnected(node) {
    const indices = valueInputIndices(node);
    if (indices.length === 0) return;
    if (indices.length >= MAX_VALUES) return;
    const lastIdx = indices[indices.length - 1];
    const lastInp = node.inputs[lastIdx];
    if (lastInp && lastInp.link != null) {
        appendNextValueSlot(node);
    }
}

app.registerExtension({
    name: "MieStringFormatAutogrow",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!nodeData || nodeData.name !== NODE_NAME) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated?.apply(this, arguments);
            // Fresh drop: the schema declares MAX slots; trim to the
            // compact starting view. On workflow load, LiteGraph's
            // configure() rewrites this.inputs from saved data after
            // this hook, so the trim is harmless.
            trimTrailingTo(this, DEFAULT_VALUES);
            return r;
        };

        const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (slotType, slot, target, target_slot) {
            const r = origOnConnectionsChange?.apply(this, arguments);
            // Only react to input-side changes; output changes are never
            // relevant for this node.
            if (slotType === LT_INPUT) {
                growIfLastConnected(this);
            }
            return r;
        };
    },
});
