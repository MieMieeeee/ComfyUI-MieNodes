import copy
import json
import time
import uuid
from typing import Any

import torch

from .utils import any_typ, mie_log, add_suffix

try:
    from comfy_execution.graph_utils import GraphBuilder
except Exception:
    try:
        from comfy.graph_utils import GraphBuilder
    except Exception:
        GraphBuilder = None

MY_CATEGORY = "🐑 MieNodes/🐑 Loop"
EMPTY_IMAGE = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
EMPTY_IMAGES = torch.zeros((0, 1, 1, 3), dtype=torch.float32)
RUNTIME_STORE = {
    "collectors": {
        "image": {},
        "text": {},
        "json": {},
    },
    "state_objects": {
        "image": {},
    },
    "meta": {},
    "_detect_cache": {},
}
_MAX_RUNTIME_STORE_AGE = 3600  # seconds = 1 hour
_runtime_store_timestamps = {}


def _require_dict(value: Any, name: str):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _parse_json_object(value: str, name: str):
    try:
        parsed = json.loads(value)
    except Exception as e:
        raise ValueError(f"{name} is invalid JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must be a JSON object")
    return parsed


def _parse_json_array(value: str, name: str):
    try:
        parsed = json.loads(value)
    except Exception as e:
        raise ValueError(f"{name} is invalid JSON: {e}") from e
    if not isinstance(parsed, list):
        raise ValueError(f"{name} must be a JSON array")
    return parsed


def _parse_params_list(params_mode, int_list, string_list, json_list):
    if params_mode == "int_list":
        normalized = (
            str(int_list).replace("，", ",").replace("\n", ",").replace("\t", ",")
        )
        parts = [x.strip() for x in normalized.split(",") if x.strip()]
        if not parts:
            return []
        params = []
        for token in parts:
            try:
                params.append({"value": int(token)})
            except Exception as e:
                raise ValueError(
                    f"int_list contains invalid integer token '{token}'"
                ) from e
        return params
    if params_mode == "string_list":
        raw_lines = [x.strip() for x in str(string_list).splitlines() if x.strip()]
        if len(raw_lines) == 1 and "," in raw_lines[0]:
            raw_lines = [x.strip() for x in raw_lines[0].split(",") if x.strip()]
        if not raw_lines:
            return []
        return [{"value": x} for x in raw_lines]
    if params_mode == "json_list":
        parsed = _parse_json_array(json_list, "json_list")
        if not parsed:
            return []
        if not all(isinstance(x, dict) for x in parsed):
            raise ValueError("json_list must be an array of objects")
        return parsed
    raise ValueError(f"Unknown params_mode: {params_mode}")


def _validate_loop_ctx(loop_ctx):
    ctx = _require_dict(loop_ctx, "loop_ctx")
    if int(ctx.get("version", 0)) != 3:
        raise ValueError("loop_ctx.version must be 3")
    if ctx.get("mode") != "for_each":
        raise ValueError("loop_ctx.mode must be for_each")
    run_id = str(ctx.get("run_id", "")).strip()
    if run_id == "":
        raise ValueError("loop_ctx.run_id is required")
    params_list = ctx.get("params_list")
    if not isinstance(params_list, list):
        raise ValueError("loop_ctx.params_list must be an array")
    count = int(ctx.get("count", -1))
    index = int(ctx.get("index", -1))
    if count < 0:
        raise ValueError("loop_ctx.count must be >= 0")
    if count != len(params_list):
        raise ValueError("loop_ctx.count does not match params_list length")
    if count == 0:
        if index != 0:
            raise ValueError("loop_ctx.index must be 0 when count is 0")
    else:
        if index < 0 or index >= count:
            raise ValueError("loop_ctx.index out of range")
    current_params = ctx.get("current_params")
    if not isinstance(current_params, dict):
        raise ValueError("loop_ctx.current_params must be an object")
    state = ctx.get("state")
    if not isinstance(state, dict):
        raise ValueError("loop_ctx.state must be an object")
    collectors = ctx.get("collectors")
    if not isinstance(collectors, dict):
        raise ValueError("loop_ctx.collectors must be an object")
    meta = ctx.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("loop_ctx.meta must be an object")
    for meta_key in ("body_in_id", "body_out_id", "end_id"):
        if meta_key not in meta:
            raise ValueError(f"loop_ctx.meta.{meta_key} is required")
    return ctx


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", ""}:
            return False
    raise ValueError(f"cannot cast '{value}' to bool")


def _value_from_json_string(value):
    raw = (value or "").strip()
    if raw == "":
        return None, False
    try:
        return json.loads(raw), True
    except Exception:
        return raw, True


def _ensure_collectors(ctx):
    if "collectors" not in ctx or not isinstance(ctx["collectors"], dict):
        ctx["collectors"] = {}
    collectors = ctx["collectors"]
    legacy_image = collectors.get("images")
    image_collector = collectors.get("image")
    if not isinstance(image_collector, dict):
        image_collector = (
            copy.deepcopy(legacy_image)
            if isinstance(legacy_image, dict)
            else {"ref": None, "count": 0}
        )
        collectors["image"] = image_collector
    collectors.pop("images", None)
    if "text" not in collectors or not isinstance(collectors["text"], dict):
        collectors["text"] = {"ref": None, "count": 0}
    if "json" not in collectors or not isinstance(collectors["json"], dict):
        collectors["json"] = {"ref": None, "count": 0}


def _ensure_collector_slot(ctx, kind):
    _ensure_collectors(ctx)
    collectors = ctx["collectors"]
    if kind not in collectors or not isinstance(collectors[kind], dict):
        collectors[kind] = {"ref": None, "count": 0}
    return collectors[kind]


def _ensure_meta_fields(ctx):
    if "meta" not in ctx or not isinstance(ctx["meta"], dict):
        ctx["meta"] = {}
    if "body_in_id" not in ctx["meta"]:
        ctx["meta"]["body_in_id"] = None
    if "body_out_id" not in ctx["meta"]:
        ctx["meta"]["body_out_id"] = None
    if "end_id" not in ctx["meta"]:
        ctx["meta"]["end_id"] = None


def _ensure_runtime_meta(run_id):
    rid = str(run_id)
    if rid not in RUNTIME_STORE["meta"] or not isinstance(
        RUNTIME_STORE["meta"][rid], dict
    ):
        RUNTIME_STORE["meta"][rid] = {}
    meta = RUNTIME_STORE["meta"][rid]
    collector_refs = meta.get("collector_refs")
    if not isinstance(collector_refs, dict):
        collector_refs = {}
        meta["collector_refs"] = collector_refs
    legacy_image_refs = meta.get("image_refs")
    if isinstance(legacy_image_refs, list):
        collector_refs["image"] = legacy_image_refs
    if "image" not in collector_refs or not isinstance(collector_refs["image"], list):
        collector_refs["image"] = []
    meta["image_refs"] = collector_refs["image"]
    state_object_refs = meta.get("state_object_refs")
    if not isinstance(state_object_refs, dict):
        state_object_refs = {}
        meta["state_object_refs"] = state_object_refs
    if "image" not in state_object_refs or not isinstance(
        state_object_refs["image"], list
    ):
        state_object_refs["image"] = []
    _runtime_store_timestamps[rid] = time.time()
    return meta


def _ensure_runtime_collector_store(kind):
    if "collectors" not in RUNTIME_STORE or not isinstance(
        RUNTIME_STORE["collectors"], dict
    ):
        RUNTIME_STORE["collectors"] = {}
    if kind not in RUNTIME_STORE["collectors"] or not isinstance(
        RUNTIME_STORE["collectors"][kind], dict
    ):
        RUNTIME_STORE["collectors"][kind] = {}
    return RUNTIME_STORE["collectors"][kind]


def _ensure_state_object_store(kind):
    if "state_objects" not in RUNTIME_STORE or not isinstance(
        RUNTIME_STORE["state_objects"], dict
    ):
        RUNTIME_STORE["state_objects"] = {}
    if kind not in RUNTIME_STORE["state_objects"] or not isinstance(
        RUNTIME_STORE["state_objects"][kind], dict
    ):
        RUNTIME_STORE["state_objects"][kind] = {}
    return RUNTIME_STORE["state_objects"][kind]


def _ensure_runtime_collector_refs(run_meta, kind):
    collector_refs = run_meta.get("collector_refs")
    if not isinstance(collector_refs, dict):
        collector_refs = {}
        run_meta["collector_refs"] = collector_refs
    refs = collector_refs.get(kind)
    if not isinstance(refs, list):
        refs = []
        collector_refs[kind] = refs
    if kind == "image":
        run_meta["image_refs"] = refs
    return refs


def _remove_runtime_collector_ref(run_meta, kind, ref):
    refs = _ensure_runtime_collector_refs(run_meta, kind)
    if ref in refs:
        run_meta["collector_refs"][kind] = [x for x in refs if x != ref]
        if kind == "image":
            run_meta["image_refs"] = run_meta["collector_refs"][kind]


def _ensure_runtime_state_object_refs(run_meta, kind):
    state_object_refs = run_meta.get("state_object_refs")
    if not isinstance(state_object_refs, dict):
        state_object_refs = {}
        run_meta["state_object_refs"] = state_object_refs
    refs = state_object_refs.get(kind)
    if not isinstance(refs, list):
        refs = []
        state_object_refs[kind] = refs
    return refs


def _remove_runtime_state_object_ref(run_meta, kind, ref):
    refs = _ensure_runtime_state_object_refs(run_meta, kind)
    if ref in refs:
        run_meta["state_object_refs"][kind] = [x for x in refs if x != ref]


def _pop_collector_items(kind, ref):
    if not ref:
        return []
    return _ensure_runtime_collector_store(kind).pop(ref, [])


def _create_collector_ref(prefix, ctx):
    run_id = str(ctx.get("run_id", "norun"))
    loop_id = str(ctx.get("loop_id", "noloop"))
    return f"{prefix}_{run_id}_{loop_id}_{uuid.uuid4().hex[:8]}"


def _state_ref_key(key):
    return f"{str(key)}_ref"


def _make_state_object_ref(prefix, ctx, key):
    run_id = str(ctx.get("run_id", "norun"))
    loop_id = str(ctx.get("loop_id", "noloop"))
    return f"{prefix}_{run_id}_{loop_id}_{key}_{uuid.uuid4().hex[:8]}"


def _state_object_put_image(loop_ctx, key, image):
    ctx = copy.deepcopy(_validate_loop_ctx(loop_ctx))
    if not str(key).strip():
        raise ValueError("key must not be empty")
    if "state" not in ctx or not isinstance(ctx["state"], dict):
        ctx["state"] = {}
    state_key = _state_ref_key(key)
    old_ref = ctx["state"].get(state_key)
    store = _ensure_state_object_store("image")
    run_meta = _ensure_runtime_meta(ctx.get("run_id", ""))
    if old_ref:
        store.pop(old_ref, None)
        _remove_runtime_state_object_ref(run_meta, "image", old_ref)
    new_ref = _make_state_object_ref("stateimg", ctx, str(key))
    store[new_ref] = image.detach().clone()
    ctx["state"][state_key] = new_ref
    refs = _ensure_runtime_state_object_refs(run_meta, "image")
    if new_ref not in refs:
        refs.append(new_ref)
    return ctx


def _state_object_get_image(loop_ctx, key):
    ctx = _validate_loop_ctx(loop_ctx)
    state_key = _state_ref_key(key)
    ref = ctx.get("state", {}).get(state_key)
    if not ref:
        return None, False, None
    image = _ensure_state_object_store("image").get(ref)
    if image is None:
        return None, False, ref
    return image.detach().clone(), True, ref


def _state_object_remove_image(loop_ctx, key):
    ctx = copy.deepcopy(_validate_loop_ctx(loop_ctx))
    if "state" not in ctx or not isinstance(ctx["state"], dict):
        ctx["state"] = {}
    state_key = _state_ref_key(key)
    old_ref = ctx["state"].pop(state_key, None)
    if not old_ref:
        return ctx, False, None
    removed = _ensure_state_object_store("image").pop(old_ref, None)
    run_meta = _ensure_runtime_meta(ctx.get("run_id", ""))
    _remove_runtime_state_object_ref(run_meta, "image", old_ref)
    return ctx, removed is not None, old_ref


def _cleanup_runtime_for_run(run_id):
    run_meta = _ensure_runtime_meta(run_id)
    collector_refs = run_meta.get("collector_refs", {})
    if isinstance(collector_refs, dict):
        for kind, refs in list(collector_refs.items()):
            if not isinstance(refs, list):
                continue
            store = _ensure_runtime_collector_store(kind)
            for ref in list(refs):
                store.pop(ref, None)
            collector_refs[kind] = []
    state_object_refs = run_meta.get("state_object_refs", {})
    if isinstance(state_object_refs, dict):
        for kind, refs in list(state_object_refs.items()):
            if not isinstance(refs, list):
                continue
            store = _ensure_state_object_store(kind)
            for ref in list(refs):
                store.pop(ref, None)
            state_object_refs[kind] = []
    run_meta["image_refs"] = []
    RUNTIME_STORE["meta"].pop(str(run_id), None)
    _runtime_store_timestamps.pop(str(run_id), None)


def _prune_runtime_store():
    collector_stores = RUNTIME_STORE.get("collectors", {})
    live_refs_by_kind = {kind: set() for kind in collector_stores.keys()}
    state_object_stores = RUNTIME_STORE.get("state_objects", {})
    live_state_object_refs_by_kind = {kind: set() for kind in state_object_stores.keys()}
    for run_meta in RUNTIME_STORE["meta"].values():
        if isinstance(run_meta, dict):
            collector_refs = run_meta.get("collector_refs", {})
            if not isinstance(collector_refs, dict):
                collector_refs = {}
            legacy_image_refs = run_meta.get("image_refs")
            if isinstance(legacy_image_refs, list):
                collector_refs = dict(collector_refs)
                collector_refs.setdefault("image", legacy_image_refs)
            for kind, refs in collector_refs.items():
                if kind not in live_refs_by_kind:
                    live_refs_by_kind[kind] = set()
                if isinstance(refs, list):
                    for ref in refs:
                        live_refs_by_kind[kind].add(str(ref))
            state_object_refs = run_meta.get("state_object_refs", {})
            if isinstance(state_object_refs, dict):
                for kind, refs in state_object_refs.items():
                    if kind not in live_state_object_refs_by_kind:
                        live_state_object_refs_by_kind[kind] = set()
                    if isinstance(refs, list):
                        for ref in refs:
                            live_state_object_refs_by_kind[kind].add(str(ref))
    for kind, store in list(collector_stores.items()):
        live_refs = live_refs_by_kind.get(kind, set())
        for ref in list(store.keys()):
            if ref not in live_refs:
                store.pop(ref, None)
    for kind, store in list(state_object_stores.items()):
        live_refs = live_state_object_refs_by_kind.get(kind, set())
        for ref in list(store.keys()):
            if ref not in live_refs:
                store.pop(ref, None)
    now = time.time()
    stale_run_ids = [
        rid
        for rid, ts in _runtime_store_timestamps.items()
        if now - ts > _MAX_RUNTIME_STORE_AGE
    ]
    for rid in stale_run_ids:
        _cleanup_runtime_for_run(rid)
        _runtime_store_timestamps.pop(rid, None)


def _base_class_type(class_type: str):
    raw = str(class_type or "")
    if "|" in raw:
        return raw.split("|", 1)[0]
    return raw


def _is_protocol_class(base_class_type: str):
    return base_class_type in {"MieLoopBodyIn", "MieLoopBodyOut", "MieLoopEnd"}


def _is_collector_class(base_class_type: str):
    return base_class_type in {
        "MieLoopCollectImage",
        "MieLoopCollectText",
        "MieLoopCollectJSON",
    }


def _is_excluded_output_class(base_class_type: str):
    explicit = {
        "MieLoopStart",
        "MieLoopResume",
        "MieLoopFinalizeImages",
        "MieLoopCleanupImages",
        "MieLoopFinalizeTextList",
        "MieLoopCleanupText",
        "MieLoopFinalizeJSONList",
        "MieLoopCleanupJSON",
        "MieImageGrid",
        "SaveImage",
        "PreviewImage",
    }
    if base_class_type in explicit:
        return True
    lowered = base_class_type.lower()
    if "save" in lowered or "preview" in lowered:
        return True
    if "viewer" in lowered or "display" in lowered:
        return True
    return False


def is_link(value):
    if not isinstance(value, (list, tuple)):
        return False
    if len(value) < 2:
        return False
    src = value[0]
    out_idx = value[1]
    return isinstance(src, (str, int)) and isinstance(out_idx, int)


def _get_inputs(node):
    if isinstance(node, dict):
        inputs = node.get("inputs", {})
        return inputs if isinstance(inputs, dict) else {}
    if hasattr(node, "inputs"):
        inputs = getattr(node, "inputs")
        return inputs if isinstance(inputs, dict) else {}
    return {}


def _get_class_type(node):
    if isinstance(node, dict):
        return str(node.get("class_type", ""))
    if hasattr(node, "class_type"):
        return str(getattr(node, "class_type"))
    return ""


def _extract_nodes_from_dict(raw):
    out = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        if isinstance(v, dict):
            out[str(k)] = v
    return out


def _get_all_nodes(dynprompt):
    collected = {}
    if isinstance(dynprompt, dict):
        collected.update(_extract_nodes_from_dict(dynprompt))
    for attr_name in ("dynprompt", "prompt", "original_prompt"):
        if hasattr(dynprompt, attr_name):
            attr_value = getattr(dynprompt, attr_name)
            collected.update(_extract_nodes_from_dict(attr_value))
    if hasattr(dynprompt, "all_node_ids"):
        try:
            for node_id in getattr(dynprompt, "all_node_ids"):
                node = get_node(dynprompt, node_id)
                if isinstance(node, dict):
                    collected[str(node_id)] = node
        except Exception:
            pass
    return collected


def _resolve_current_node_id(unique_id=None, dynprompt=None):
    if unique_id is not None:
        return str(unique_id)
    if dynprompt is not None and hasattr(dynprompt, "get_current_node_id"):
        try:
            nid = dynprompt.get_current_node_id()
            if nid is not None:
                return str(nid)
        except Exception:
            pass
    return None


def _should_record_protocol_node_id(existing_id, current_node_id):
    current = str(current_node_id or "")
    existing = str(existing_id or "")
    if not current:
        return False
    # Expand rounds use cloned ids like "48.0.0.50". Keep the original
    # template protocol ids stable once they are known.
    if existing and "." in current:
        return False
    return True


def get_node(dynprompt, node_id):
    nid = str(node_id)
    if isinstance(dynprompt, dict):
        if nid in dynprompt:
            return dynprompt[nid]
        if node_id in dynprompt:
            return dynprompt[node_id]
    if hasattr(dynprompt, "get_node"):
        try:
            node = dynprompt.get_node(nid)
            if node is not None:
                return node
        except Exception:
            pass
    if hasattr(dynprompt, "get"):
        try:
            node = dynprompt.get(nid)
            if node is not None:
                return node
        except Exception:
            pass
    for attr_name in ("dynprompt", "prompt", "original_prompt"):
        if hasattr(dynprompt, attr_name):
            raw = getattr(dynprompt, attr_name)
            if isinstance(raw, dict):
                if nid in raw:
                    return raw[nid]
                if node_id in raw:
                    return raw[node_id]
    return None


def explore_backward_from_body_out(node_id, dynprompt, visited):
    nid = str(node_id)
    if nid in visited:
        return
    visited.add(nid)
    node = get_node(dynprompt, nid)
    if node is None:
        return
    inputs = _get_inputs(node)
    for input_value in inputs.values():
        if is_link(input_value):
            src_id = str(input_value[0])
            explore_backward_from_body_out(src_id, dynprompt, visited)


def explore_forward_from_body_in(node_id, dynprompt, visited, stop_ids=None):
    nid = str(node_id)
    if nid in visited:
        return
    visited.add(nid)
    normalized_stop_ids = {str(x) for x in (stop_ids or set())}
    if nid in normalized_stop_ids:
        return
    all_nodes = _get_all_nodes(dynprompt)
    for candidate_id, candidate_node in all_nodes.items():
        inputs = _get_inputs(candidate_node)
        for input_value in inputs.values():
            if is_link(input_value) and str(input_value[0]) == nid:
                explore_forward_from_body_in(
                    candidate_id, dynprompt, visited, normalized_stop_ids
                )
                break


def collect_loop_body(body_in_id, body_out_id, dynprompt, end_id=None):
    backward_set = set()
    forward_set = set()
    explore_backward_from_body_out(body_out_id, dynprompt, backward_set)
    stop_ids = {str(body_out_id)}
    if end_id:
        stop_ids.add(str(end_id))
    explore_forward_from_body_in(body_in_id, dynprompt, forward_set, stop_ids)
    body_nodes_raw = sorted(backward_set & forward_set)
    body_nodes_filtered = []
    body_nodes_business = []
    excluded_nodes = []
    protocol_nodes = []
    collector_nodes = []
    node_types = {}
    for node_id in body_nodes_raw:
        node = get_node(dynprompt, node_id)
        class_type = _get_class_type(node)
        base_class_type = _base_class_type(class_type)
        node_types[node_id] = base_class_type
        if _is_excluded_output_class(base_class_type):
            excluded_nodes.append(node_id)
            continue
        body_nodes_filtered.append(node_id)
        if _is_protocol_class(base_class_type):
            protocol_nodes.append(node_id)
            continue
        # Collector nodes inside the loop body MUST be cloned into the expand graph.
        # Each expand-round clone must re-execute the collector to append images
        # to the shared RUNTIME_STORE ref (run by run_id). If not cloned, the
        # collector reads stale cached outputs from round 0 and the loop body
        # cannot accumulate results across rounds.
        if _is_collector_class(base_class_type):
            collector_nodes.append(node_id)
        body_nodes_business.append(node_id)
    debug_info = {
        "forward_set": sorted(forward_set),
        "backward_set": sorted(backward_set),
        "body_nodes_raw": body_nodes_raw,
        "body_nodes_filtered": sorted(body_nodes_filtered),
        "body_nodes_business": sorted(body_nodes_business),
        "excluded_nodes": sorted(excluded_nodes),
        "protocol_nodes": sorted(protocol_nodes),
        "collector_nodes": sorted(collector_nodes),
        "node_types": node_types,
        "stop_ids": sorted(stop_ids),
    }
    nested_loop_nodes = []
    for node_id in body_nodes_raw:
        node = get_node(dynprompt, node_id)
        base_class_type = _base_class_type(_get_class_type(node))
        if base_class_type in {"MieLoopStart", "MieLoopResume"}:
            nested_loop_nodes.append(node_id)
    if nested_loop_nodes:
        debug_info["nested_loop_nodes"] = sorted(nested_loop_nodes)
        raise ValueError(
            f"Nested loop nodes detected inside loop body: {sorted(nested_loop_nodes)}"
        )
    if str(body_in_id) not in protocol_nodes or str(body_out_id) not in protocol_nodes:
        debug_info["protocol_guard_error"] = {
            "required_body_in_id": str(body_in_id),
            "required_body_out_id": str(body_out_id),
            "protocol_nodes": sorted(protocol_nodes),
        }
        raise ValueError(
            f"Loop body protocol guard failed: {debug_info['protocol_guard_error']}"
        )
    return sorted(body_nodes_filtered), debug_info


# =============================================================================
# body_in_id 语义说明 (v3 首版设计决策)
# ----------------------------------------------------------------------------
# v3 首版采用 "原模板入口锚点" 方案:
#   - body_in_id 固定指向原始模板中 BodyIn 节点的 id
#   - expand 时不 clone BodyIn (因为 loop_ctx 通过 Resume 注入)
#   - 后续轮次不更新 body_in_id
#   - collect_loop_body() 永远基于原模板图做节点发现
#
# 这样设计的好处是简单: 识别逻辑不依赖 expand 图结构,
# 而是用同一个 "模板蓝图" 来发现每轮的节点集合。
# =============================================================================


def _build_expand_graph_for_next_round(
    next_ctx, dynprompt, body_in_id, body_out_id, end_id, detect_result
):
    if GraphBuilder is None:
        raise ValueError("GraphBuilder is unavailable in current ComfyUI runtime")
    business_set = {str(x) for x in detect_result.get("body_nodes_business", [])}
    backward_set = {str(x) for x in detect_result.get("backward_set", [])}
    forward_set = {str(x) for x in detect_result.get("forward_set", [])}
    protocol_nodes = {str(body_out_id), str(end_id)}
    all_nodes = _get_all_nodes(dynprompt)
    graph = GraphBuilder()
    resume_node = graph.node(
        add_suffix("MieLoopResume"),
        loop_ctx_json=json.dumps(next_ctx, ensure_ascii=False),
    )
    built_nodes = {}
    visiting = set()
    walked = set()
    body_in_node = get_node(dynprompt, str(body_in_id))
    body_in_anchor_src = None
    if isinstance(body_in_node, dict):
        bi_inputs = _get_inputs(body_in_node)
        anchor_val = bi_inputs.get("anchor")
        if is_link(anchor_val):
            body_in_anchor_src = (str(anchor_val[0]), int(anchor_val[1]))

    def should_clone(node_id):
        nid = str(node_id)
        if nid in protocol_nodes:
            return True
        if nid in business_set:
            return True
        if nid not in backward_set or nid not in forward_set:
            return False
        node = get_node(dynprompt, nid)
        base_class_type = _base_class_type(_get_class_type(node))
        return not _is_excluded_output_class(base_class_type)

    def loop_ctx_chain_has_state_update(node_id, visited=None):
        nid = str(node_id)
        if visited is None:
            visited = set()
        if nid in visited:
            return False
        visited.add(nid)
        if nid == str(body_in_id):
            return False
        node = get_node(dynprompt, nid)
        if not isinstance(node, dict):
            return False
        base_class_type = _base_class_type(_get_class_type(node))
        if base_class_type.startswith("MieLoopStateSet"):
            return True
        loop_ctx_input = _get_inputs(node).get("loop_ctx")
        if is_link(loop_ctx_input):
            return loop_ctx_chain_has_state_update(loop_ctx_input[0], visited)
        return False

    def build_node(old_id):
        oid = str(old_id)
        if oid in built_nodes:
            return built_nodes[oid]
        if oid in visiting:
            raise ValueError(
                f"LoopEnd.expand failed: cyclic dependency detected at node {oid}"
            )
        old_node = get_node(dynprompt, oid)
        if not isinstance(old_node, dict):
            raise ValueError(f"LoopEnd.expand failed: cannot resolve node {oid}")
        visiting.add(oid)
        try:
            old_inputs = _get_inputs(old_node)
            new_inputs = copy.deepcopy(old_inputs)
            for in_name, in_val in list(old_inputs.items()):
                if not is_link(in_val):
                    continue
                src_id = str(in_val[0])
                src_idx = int(in_val[1])
                # body_in_id 不 clone，按输出索引分别处理
                if src_id == str(body_in_id):
                    if src_idx == 0:
                        # out(0) = loop_ctx → 从 Resume 获取
                        new_inputs[in_name] = resume_node.out(0)
                    elif src_idx == 1 and body_in_anchor_src is not None:
                        # out(1) = anchor → 追溯到 anchor 的原始来源
                        a_src_id, a_src_idx = body_in_anchor_src
                        if should_clone(a_src_id):
                            new_inputs[in_name] = build_node(a_src_id).out(a_src_idx)
                        else:
                            new_inputs[in_name] = [a_src_id, a_src_idx]
                    # else: anchor 未连接或 src_idx 超出范围 → 保留原值
                    continue
                if should_clone(src_id):
                    new_inputs[in_name] = build_node(src_id).out(src_idx)
                    continue
                new_inputs[in_name] = [src_id, src_idx]
            
            # 协议节点特殊重接：
            # 1) BodyOut 默认回退到 Resume，确保普通 collector workflow 沿用稳定的推进语义。
            #    只有当其真实 loop_ctx 上游链中存在 StateSet* 节点时，才保留真实链路，
            #    以便把跨轮 feedback / state 更新传进下一轮。
            # 2) End 的协议输入必须显式绑定到“当前层 cloned BodyOut”，
            #    不能依赖通用 rewiring，否则递归 expand 时可能错误连到上一层 BodyOut，
            #    导致 End 早于当前轮业务链执行，最终少收一张图。
            is_loop_end_node = oid == str(end_id)
            is_body_out_node = oid == str(body_out_id)

            if is_body_out_node:
                original_loop_ctx = old_inputs.get("loop_ctx")
                if is_link(original_loop_ctx):
                    original_src_id = str(original_loop_ctx[0])
                    if not loop_ctx_chain_has_state_update(original_src_id):
                        new_inputs["loop_ctx"] = resume_node.out(0)
                else:
                    new_inputs["loop_ctx"] = resume_node.out(0)

            if is_loop_end_node:
                # End 必须强依赖当前层 cloned BodyOut
                body_out_node = build_node(str(body_out_id))
                new_inputs["loop_ctx"] = body_out_node.out(0)
                new_inputs["state_json"] = body_out_node.out(1)
            if oid in business_set:
                new_inputs["__mie_loop_round_idx__"] = int(next_ctx.get("index", 0))
            new_node = graph.node(_get_class_type(old_node), oid, **new_inputs)
            new_node.set_override_display_id(oid)
            built_nodes[oid] = new_node
        finally:
            visiting.discard(oid)
        return built_nodes[oid]

    def walk_from_body_in(node_id):
        nid = str(node_id)
        if nid in walked:
            return
        walked.add(nid)
        for candidate_id, candidate_node in all_nodes.items():
            cid = str(candidate_id)
            inputs = _get_inputs(candidate_node)
            has_edge = False
            for input_value in inputs.values():
                if is_link(input_value) and str(input_value[0]) == nid:
                    has_edge = True
                    break
            if not has_edge:
                continue
            if should_clone(cid):
                build_node(cid)
            walk_from_body_in(cid)

    # walk_from_body_in(str(body_in_id))
    build_node(str(body_out_id))
    build_node(str(end_id))
    # 返回 end 节点的 Node 对象，用于在 MieLoopEnd.execute 中构造 is_link() 结果
    # 这是 ComfyUI expand 机制的关键：result 中必须包含 is_link() 值
    # （即 [node_id, output_index] 格式的 list），引擎才会为子图节点创建
    # strong_link 并将其加入执行队列。否则 expand 图注册但永远不会执行。
    # 参考：execution.py lines 572-576, EasyUse whileLoopEnd.
    end_built_node = built_nodes.get(str(end_id))
    finalize_result = graph.finalize()
    # Log detailed expand graph structure for debugging
    expand_node_ids = sorted(finalize_result.keys())
    mie_log(
        f"ExpandGraphBuild: body_nodes_business={sorted(detect_result.get('body_nodes_business', []))}, "
        f"collector_nodes={sorted(detect_result.get('collector_nodes', []))}, "
        f"built_node_ids={sorted(built_nodes.keys())}, "
        f"expand_node_ids={expand_node_ids}, "
        f"expand_nodes={len(finalize_result)}"
    )
    # Log each node's inputs in expand graph for debugging
    for nid in expand_node_ids:
        node_data = finalize_result[nid]
        inputs = node_data.get("inputs", {})
        mie_log(
            f"ExpandNode: id={nid}, class_type={node_data.get('class_type')}, inputs={inputs}"
        )
    return finalize_result, end_built_node


def _merge_images_for_ctx(loop_ctx):
    ctx = _validate_loop_ctx(loop_ctx)
    ref = _ensure_collector_slot(ctx, "image").get("ref")
    if not ref:
        return EMPTY_IMAGES
    batches = _pop_collector_items("image", ref)
    run_meta = _ensure_runtime_meta(ctx.get("run_id", ""))
    _remove_runtime_collector_ref(run_meta, "image", ref)
    if not batches:
        return EMPTY_IMAGES
    base_shape = tuple(batches[0].shape[1:])
    for idx, item in enumerate(batches):
        if tuple(item.shape[1:]) != base_shape:
            raise ValueError(
                f"MieLoopFinalizeImages: image shape mismatch at index {idx}: {tuple(item.shape[1:])} != {base_shape}"
            )
    return torch.cat(batches, dim=0)


def _grid_images(images, cols=2, pad=4):
    if images.ndim != 4:
        raise ValueError(
            f"MieImageGrid expects IMAGE tensor [B,H,W,C], got ndim={images.ndim}"
        )
    b, h, w, c = images.shape
    if b == 0:
        return images
    if cols <= 0:
        cols = 1
    if pad < 0:
        pad = 0
    for idx in range(1, b):
        if tuple(images[idx].shape) != (h, w, c):
            raise ValueError("MieImageGrid received mismatched image sizes")
    cols = min(cols, b)
    rows = (b + cols - 1) // cols
    grid_h = rows * h + (rows - 1) * pad
    grid_w = cols * w + (cols - 1) * pad
    grid = torch.zeros((1, grid_h, grid_w, c), dtype=images.dtype, device=images.device)
    for i in range(b):
        r = i // cols
        cc = i % cols
        y = r * (h + pad)
        x = cc * (w + pad)
        grid[0, y : y + h, x : x + w, :] = images[i]
    return grid


class MieLoopStart:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_id": ("STRING", {"default": "scan_steps"}),
                "params_mode": (
                    ["int_list", "string_list", "json_list"],
                    {"default": "int_list"},
                ),
                "int_list": ("STRING", {"default": "8,9,10"}),
                "string_list": (
                    "STRING",
                    {"default": "cat\ndog\ncar", "multiline": True},
                ),
                "json_list": (
                    "STRING",
                    {
                        "default": '[{"steps": 8}, {"steps": 9}, {"steps": 10}]',
                        "multiline": True,
                    },
                ),
                "initial_state_json": ("STRING", {"default": "{}"}),
            },
            "optional": {
                "meta_json": ("STRING", {"default": "{}"}),
                "resume_loop_ctx": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MIE_LOOP_CTX", "INT", "INT", "BOOLEAN")
    RETURN_NAMES = ("loop_ctx", "index", "count", "is_last")
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(
        self,
        loop_id,
        params_mode,
        int_list,
        string_list,
        json_list,
        initial_state_json,
        meta_json="{}",
        resume_loop_ctx="",
    ):
        _prune_runtime_store()
        resume_raw = (resume_loop_ctx or "").strip()
        if resume_raw:
            resumed = _parse_json_object(resume_raw, "resume_loop_ctx")
            ctx = copy.deepcopy(_validate_loop_ctx(resumed))
            if str(ctx.get("loop_id", "")) != str(loop_id):
                raise ValueError("resume_loop_ctx.loop_id does not match loop_id input")
            if int(ctx.get("index", 0)) >= int(ctx.get("count", 0)):
                raise ValueError("resume_loop_ctx.index is out of range")
            _ensure_collectors(ctx)
            _ensure_meta_fields(ctx)
            mie_log(
                f"LoopStart: resumed loop_id={ctx['loop_id']}, run_id={ctx['run_id']}, index={ctx['index']}, count={ctx['count']}"
            )
            return (ctx, int(ctx["index"]), int(ctx["count"]), bool(ctx["is_last"]))
        params_list = _parse_params_list(params_mode, int_list, string_list, json_list)
        initial_state = _parse_json_object(initial_state_json, "initial_state_json")
        meta = _parse_json_object(meta_json, "meta_json")
        count = len(params_list)
        if count == 0:
            run_id = uuid.uuid4().hex[:24]
            loop_ctx = {
                "version": 3,
                "loop_id": str(loop_id),
                "run_id": run_id,
                "mode": "for_each",
                "index": 0,
                "count": 0,
                "is_last": True,
                "params_list": [],
                "current_params": {},
                "state": initial_state,
                "collectors": {
                    "image": {"ref": None, "count": 0},
                    "text": {"ref": None, "count": 0},
                    "json": {"ref": None, "count": 0},
                },
                "meta": {
                    "user_label": meta.get("user_label", ""),
                    "created_by": "MieLoopStart",
                    "body_in_id": None,
                    "body_out_id": None,
                    "end_id": None,
                    **meta,
                },
            }
            _ensure_meta_fields(loop_ctx)
            _ensure_runtime_meta(run_id)
            RUNTIME_STORE["meta"][run_id]["loop_id"] = str(loop_id)
            RUNTIME_STORE["meta"][run_id]["count"] = 0
            RUNTIME_STORE["meta"][run_id]["status"] = "completed"
            mie_log(
                f"LoopStart: empty params, loop_id={loop_ctx['loop_id']}, run_id={run_id}"
            )
            return (loop_ctx, 0, 0, True)
        run_id = uuid.uuid4().hex[:24]
        loop_ctx = {
            "version": 3,
            "loop_id": str(loop_id),
            "run_id": run_id,
            "mode": "for_each",
            "index": 0,
            "count": count,
            "is_last": count == 1,
            "params_list": params_list,
            "current_params": params_list[0],
            "state": initial_state,
            "collectors": {
                "image": {"ref": None, "count": 0},
                "text": {"ref": None, "count": 0},
                "json": {"ref": None, "count": 0},
            },
            "meta": {
                "user_label": meta.get("user_label", ""),
                "created_by": "MieLoopStart",
                "body_in_id": None,
                "body_out_id": None,
                "end_id": None,
                **meta,
            },
        }
        _ensure_meta_fields(loop_ctx)
        _ensure_runtime_meta(run_id)
        RUNTIME_STORE["meta"][run_id]["loop_id"] = str(loop_id)
        RUNTIME_STORE["meta"][run_id]["count"] = count
        RUNTIME_STORE["meta"][run_id]["status"] = "running"
        mie_log(
            f"LoopStart: initialized loop_id={loop_ctx['loop_id']}, run_id={run_id}, count={count}, params_mode={params_mode}"
        )
        return (loop_ctx, 0, count, loop_ctx["is_last"])


# =============================================================================
# MieLoopResume - expand graph internal use only
# ----------------------------------------------------------------------------
# 这是 expand 图内部节点，用于注入下一轮的 loop_ctx。
# 用户不应在工作流中手动创建此节点。
# =============================================================================
class MieLoopResume:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_ctx_json": ("STRING", {"default": "{}"}),
            }
        }

    RETURN_TYPES = ("MIE_LOOP_CTX",)
    RETURN_NAMES = ("loop_ctx",)
    FUNCTION = "execute"
    CATEGORY = f"{MY_CATEGORY}/_internal"

    def execute(self, loop_ctx_json):
        ctx = _parse_json_object(loop_ctx_json, "loop_ctx_json")
        ctx = copy.deepcopy(_validate_loop_ctx(ctx))
        _ensure_meta_fields(ctx)
        return (ctx,)


class MieLoopBodyIn:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_ctx": ("MIE_LOOP_CTX",),
            },
            "optional": {
                "anchor": (any_typ,),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "dynprompt": "DYNPROMPT",
            },
        }

    RETURN_TYPES = ("MIE_LOOP_CTX", any_typ)
    RETURN_NAMES = ("loop_ctx", "anchor")
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, loop_ctx, anchor=None, unique_id=None, dynprompt=None):
        ctx = copy.deepcopy(_validate_loop_ctx(loop_ctx))
        _ensure_meta_fields(ctx)
        current_node_id = _resolve_current_node_id(
            unique_id=unique_id, dynprompt=dynprompt
        )
        if current_node_id is not None:
            ctx["meta"]["body_in_id"] = current_node_id
        return (ctx, anchor)


class MieLoopBodyOut:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_ctx": ("MIE_LOOP_CTX",),
            },
            "optional": {
                "state_json": ("STRING", {"default": "{}"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "dynprompt": "DYNPROMPT",
            },
        }

    RETURN_TYPES = ("MIE_LOOP_CTX", "STRING")
    RETURN_NAMES = ("loop_ctx", "state_json")
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(
        self,
        loop_ctx,
        state_json="{}",
        unique_id=None,
        dynprompt=None,
    ):
        ctx = copy.deepcopy(_validate_loop_ctx(loop_ctx))
        _ensure_meta_fields(ctx)
        current_node_id = _resolve_current_node_id(
            unique_id=unique_id, dynprompt=dynprompt
        )
        if _should_record_protocol_node_id(
            ctx["meta"].get("body_out_id"), current_node_id
        ):
            ctx["meta"]["body_out_id"] = current_node_id
        state_patch = _parse_json_object(state_json, "state_json")
        return (ctx, json.dumps(state_patch, ensure_ascii=False))


class MieLoopEnd:
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_ctx": ("MIE_LOOP_CTX",),
                "state_json": ("STRING", {"default": "{}"}),
            },
            "optional": {
                "debug": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("MIE_LOOP_CTX", "BOOLEAN")
    RETURN_NAMES = ("loop_ctx", "done")
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(
        self,
        loop_ctx,
        state_json,
        debug=False,
        dynprompt=None,
        unique_id=None,
        extra_pnginfo=None,
    ):
        _ = extra_pnginfo
        ctx = copy.deepcopy(_validate_loop_ctx(loop_ctx))
        _ensure_meta_fields(ctx)
        current_node_id = _resolve_current_node_id(
            unique_id=unique_id, dynprompt=dynprompt
        )
        if _should_record_protocol_node_id(ctx["meta"].get("end_id"), current_node_id):
            ctx["meta"]["end_id"] = current_node_id
        state_patch = _parse_json_object(state_json, "state_json")
        ctx["state"].update(state_patch)
        curr_index = int(ctx["index"])
        count = int(ctx["count"])
        done = False
        if count >= 30:
            mie_log(
                f"LoopEnd: warning large loop count={count}, loop_id={ctx['loop_id']}, run_id={ctx['run_id']}"
            )
        if curr_index + 1 < count:
            next_index = curr_index + 1
            ctx["index"] = next_index
            ctx["current_params"] = ctx["params_list"][next_index]
            ctx["is_last"] = next_index == count - 1
            mie_log(
                f"LoopEnd: loop continue: round={curr_index} -> {next_index}, count={count}, is_last={ctx['is_last']}, "
                f"loop_id={ctx['loop_id']}, run_id={ctx['run_id']}"
            )
        else:
            done = True
            ctx["index"] = count - 1
            ctx["is_last"] = True
            _ensure_runtime_meta(ctx["run_id"])["status"] = "completed"
            mie_log(
                f"LoopEnd: loop completed: final_round={curr_index}, count={count}, "
                f"loop_id={ctx['loop_id']}, run_id={ctx['run_id']}"
            )
        body_in_id = ctx.get("meta", {}).get("body_in_id")
        body_out_id = ctx.get("meta", {}).get("body_out_id")
        detect_result = {
            "forward_set": [],
            "backward_set": [],
            "body_nodes_raw": [],
            "body_nodes_filtered": [],
            "body_nodes_business": [],
            "excluded_nodes": [],
            "protocol_nodes": [],
            "collector_nodes": [],
            "node_types": {},
        }
        body_nodes_filtered = []
        # 缓存机制：第一轮成功检测后按 run_id 缓存到 RUNTIME_STORE，
        # 后续 expand 轮次直接复用。不能用 ctx 传递因为 expand 图中
        # BodyOut 的 loop_ctx 来自原始 CollectImage（无缓存）。
        run_id = ctx.get("run_id", "")
        cached_detect = RUNTIME_STORE.get("_detect_cache", {}).get(run_id)
        if cached_detect is not None:
            detect_result = cached_detect
            body_nodes_filtered = cached_detect.get("body_nodes_filtered", [])
        elif dynprompt is not None and body_in_id and body_out_id:
            body_nodes_filtered, detect_result = collect_loop_body(
                body_in_id,
                body_out_id,
                dynprompt,
                ctx.get("meta", {}).get("end_id"),
            )
            # 缓存供后续 expand 轮次使用
            if "_detect_cache" not in RUNTIME_STORE:
                RUNTIME_STORE["_detect_cache"] = {}
            RUNTIME_STORE["_detect_cache"][run_id] = detect_result
        if debug and dynprompt is not None and body_in_id and body_out_id:
            mie_log(
                f"LoopEndDetect: loop_id={ctx['loop_id']}, run_id={ctx['run_id']}, "
                f"body_in_id={body_in_id}, body_out_id={body_out_id}, end_id={ctx.get('meta', {}).get('end_id')}, "
                f"forward_set={detect_result['forward_set']}, backward_set={detect_result['backward_set']}, "
                f"body_nodes_raw={detect_result['body_nodes_raw']}, body_nodes_filtered={detect_result['body_nodes_filtered']}, "
                f"protocol_nodes={detect_result['protocol_nodes']}, collector_nodes={detect_result['collector_nodes']}, "
                f"body_nodes_business={detect_result['body_nodes_business']}, excluded_nodes={detect_result['excluded_nodes']}"
            )
        if (
            not done
            and dynprompt is not None
            and body_in_id
            and body_out_id
            and ctx.get("meta", {}).get("end_id")
        ):
            if debug and not detect_result.get("body_nodes_business"):
                raise ValueError(
                    f"LoopEnd.detect failed: body_nodes_business empty, debug={detect_result}"
                )
            expand_graph, end_built_node = _build_expand_graph_for_next_round(
                next_ctx=ctx,
                dynprompt=dynprompt,
                body_in_id=body_in_id,
                body_out_id=body_out_id,
                end_id=ctx.get("meta", {}).get("end_id"),
                detect_result=detect_result,
            )
            if end_built_node is None:
                raise ValueError(
                    "LoopEnd.expand failed: cloned end node not found in expand graph"
                )
            mie_log(
                f"LoopEndExpand: loop_id={ctx['loop_id']}, run_id={ctx['run_id']}, "
                f"next_index={ctx['index']}, expand_nodes={len(expand_graph)}, "
                f"end_node_id={end_built_node.id}"
            )
            # result 必须包含 is_link() 值（指向 expand 图中克隆的 LoopEnd 节点输出）
            # 这样 ComfyUI 引擎才会为子图创建 strong_link 并执行 expand 图中的所有节点
            # 参考 ComfyUI execution.py:572-576, EasyUse whileLoopEnd
            return {
                "result": tuple([end_built_node.out(i) for i in range(2)]),
                "expand": expand_graph,
            }
        if not done and dynprompt is None:
            raise ValueError(
                "LoopEnd.expand failed: dynprompt is required when done=False"
            )
        if not done and (not body_in_id or not body_out_id):
            raise ValueError(
                f"LoopEnd.expand failed: body_in_id/body_out_id missing, meta={ctx.get('meta', {})}"
            )
        if done:
            run_id_for_cleanup = ctx.get("run_id", "")
            if "_detect_cache" in RUNTIME_STORE:
                RUNTIME_STORE["_detect_cache"].pop(run_id_for_cleanup, None)
        mie_log(
            f"LoopEnd: loop_id={ctx['loop_id']}, run_id={ctx['run_id']}, next_index={ctx['index']}, done={done}"
        )
        return (ctx, done)


class MieLoopParamGetInt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_ctx": ("MIE_LOOP_CTX",),
                "key": ("STRING", {"default": "value"}),
                "default_value": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, loop_ctx, key, default_value):
        ctx = _validate_loop_ctx(loop_ctx)
        value = ctx.get("current_params", {}).get(key, default_value)
        try:
            return (int(value),)
        except Exception:
            mie_log(
                f"LoopParamGetInt: key='{key}' cast failed, using default={default_value}"
            )
            return (default_value,)


class MieLoopParamGetFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_ctx": ("MIE_LOOP_CTX",),
                "key": ("STRING", {"default": "value"}),
                "default_value": ("FLOAT", {"default": 0.0}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, loop_ctx, key, default_value):
        ctx = _validate_loop_ctx(loop_ctx)
        value = ctx.get("current_params", {}).get(key, default_value)
        try:
            return (float(value),)
        except Exception:
            mie_log(
                f"LoopParamGetFloat: key='{key}' cast failed, using default={default_value}"
            )
            return (default_value,)


class MieLoopParamGetString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_ctx": ("MIE_LOOP_CTX",),
                "key": ("STRING", {"default": "value"}),
                "default_value": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, loop_ctx, key, default_value):
        ctx = _validate_loop_ctx(loop_ctx)
        value = ctx.get("current_params", {}).get(key, default_value)
        try:
            return (str(value),)
        except Exception:
            mie_log(f"LoopParamGetString: key='{key}' cast failed, using default")
            return (default_value,)


class MieLoopParamGetBool:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_ctx": ("MIE_LOOP_CTX",),
                "key": ("STRING", {"default": "value"}),
                "default_value": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, loop_ctx, key, default_value):
        ctx = _validate_loop_ctx(loop_ctx)
        value = ctx.get("current_params", {}).get(key, default_value)
        try:
            return (_coerce_bool(value),)
        except Exception:
            mie_log(
                f"LoopParamGetBool: key='{key}' cast failed, using default={default_value}"
            )
            return (default_value,)


class MieLoopStateGetInt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_ctx": ("MIE_LOOP_CTX",),
                "key": ("STRING", {"default": "value"}),
                "default_value": ("INT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, loop_ctx, key, default_value):
        ctx = _validate_loop_ctx(loop_ctx)
        value = ctx.get("state", {}).get(key, default_value)
        try:
            return (int(value),)
        except Exception:
            mie_log(
                f"LoopStateGetInt: key='{key}' cast failed, using default={default_value}"
            )
            return (default_value,)


class MieLoopStateGetFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_ctx": ("MIE_LOOP_CTX",),
                "key": ("STRING", {"default": "value"}),
                "default_value": ("FLOAT", {"default": 0.0}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, loop_ctx, key, default_value):
        ctx = _validate_loop_ctx(loop_ctx)
        value = ctx.get("state", {}).get(key, default_value)
        try:
            return (float(value),)
        except Exception:
            mie_log(
                f"LoopStateGetFloat: key='{key}' cast failed, using default={default_value}"
            )
            return (default_value,)


class MieLoopStateGetString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_ctx": ("MIE_LOOP_CTX",),
                "key": ("STRING", {"default": "value"}),
                "default_value": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, loop_ctx, key, default_value):
        ctx = _validate_loop_ctx(loop_ctx)
        value = ctx.get("state", {}).get(key, default_value)
        try:
            return (str(value),)
        except Exception:
            mie_log(f"LoopStateGetString: key='{key}' cast failed, using default")
            return (default_value,)


class MieLoopStateGetBool:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_ctx": ("MIE_LOOP_CTX",),
                "key": ("STRING", {"default": "value"}),
                "default_value": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, loop_ctx, key, default_value):
        ctx = _validate_loop_ctx(loop_ctx)
        value = ctx.get("state", {}).get(key, default_value)
        try:
            return (_coerce_bool(value),)
        except Exception:
            mie_log(
                f"LoopStateGetBool: key='{key}' cast failed, using default={default_value}"
            )
            return (default_value,)


class MieLoopStateSet:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_state_json": ("STRING", {"default": "{}"}),
            },
            "optional": {
                "key1": ("STRING", {"default": ""}),
                "value1_json": ("STRING", {"default": ""}),
                "key2": ("STRING", {"default": ""}),
                "value2_json": ("STRING", {"default": ""}),
                "key3": ("STRING", {"default": ""}),
                "value3_json": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("state_json",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(
        self,
        base_state_json,
        key1="",
        value1_json="",
        key2="",
        value2_json="",
        key3="",
        value3_json="",
    ):
        state = _parse_json_object(base_state_json, "base_state_json")
        pairs = [(key1, value1_json), (key2, value2_json), (key3, value3_json)]
        for key, raw_value in pairs:
            if not key:
                continue
            parsed_value, ok = _value_from_json_string(raw_value)
            if ok:
                state[str(key)] = parsed_value
        return (json.dumps(state, ensure_ascii=False),)


class MieImageSelectFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "index": ("INT", {"default": 0, "min": -9999, "max": 9999}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, images, index):
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("images must be an IMAGE batch tensor [B,H,W,C]")
        batch = images.shape[0]
        if batch == 0:
            raise ValueError("images batch is empty")
        idx = int(index)
        if idx < 0:
            idx += batch
        if idx < 0 or idx >= batch:
            raise ValueError(f"index out of range: {index} for batch size {batch}")
        return (images[idx : idx + 1].detach().clone(),)


class MieLoopStateSetImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_ctx": ("MIE_LOOP_CTX",),
                "key": ("STRING", {"default": "feedback_image"}),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MIE_LOOP_CTX",)
    RETURN_NAMES = ("loop_ctx",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, loop_ctx, key, image):
        ctx = _state_object_put_image(loop_ctx, key, image)
        ref = ctx.get("state", {}).get(_state_ref_key(key))
        mie_log(
            f"LoopStateSetImage: loop_id={ctx['loop_id']}, run_id={ctx['run_id']}, key={key}, ref={ref}"
        )
        return (ctx,)


class MieLoopStateGetImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_ctx": ("MIE_LOOP_CTX",),
                "key": ("STRING", {"default": "feedback_image"}),
            },
            "optional": {
                "fallback_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("image", "found")
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, loop_ctx, key, fallback_image=None):
        image, found, ref = _state_object_get_image(loop_ctx, key)
        if found:
            return (image, True)
        if fallback_image is not None:
            mie_log(f"LoopStateGetImage: key={key}, ref={ref}, found=False, using=fallback")
            return (fallback_image, False)
        mie_log(f"LoopStateGetImage: key={key}, ref={ref}, found=False, using=empty")
        return (EMPTY_IMAGE, False)


class MieLoopStateCleanupImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_ctx": ("MIE_LOOP_CTX",),
                "key": ("STRING", {"default": "feedback_image"}),
            }
        }

    RETURN_TYPES = ("MIE_LOOP_CTX", "BOOLEAN")
    RETURN_NAMES = ("loop_ctx", "cleaned")
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, loop_ctx, key):
        ctx, cleaned, old_ref = _state_object_remove_image(loop_ctx, key)
        mie_log(
            f"LoopStateCleanupImage: loop_id={ctx['loop_id']}, run_id={ctx['run_id']}, key={key}, ref={old_ref}, cleaned={cleaned}"
        )
        return (ctx, cleaned)


class MieLoopCollectImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_ctx": ("MIE_LOOP_CTX",),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("MIE_LOOP_CTX",)
    RETURN_NAMES = ("loop_ctx",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, loop_ctx, image):
        ctx = copy.deepcopy(_validate_loop_ctx(loop_ctx))
        run_meta = _ensure_runtime_meta(ctx.get("run_id", ""))
        images_collector = _ensure_collector_slot(ctx, "image")
        ref = images_collector.get("ref")
        image_store = _ensure_runtime_collector_store("image")
        if not ref:
            ref = _create_collector_ref("imgcol", ctx)
            images_collector["ref"] = ref
            image_store[ref] = []
            image_refs = _ensure_runtime_collector_refs(run_meta, "image")
            if ref not in image_refs:
                image_refs.append(ref)
        if ref not in image_store:
            image_store[ref] = []
        image_store[ref].append(image.detach().clone())
        images_collector["count"] = len(image_store[ref])
        mie_log(
            f"LoopCollectImage: loop_id={ctx['loop_id']}, run_id={ctx['run_id']}, ref={ref}, count={images_collector['count']}"
        )
        return (ctx,)


class MieLoopFinalizeImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_ctx": ("MIE_LOOP_CTX",),
                "done": ("BOOLEAN", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, loop_ctx, done):
        ctx = copy.deepcopy(_validate_loop_ctx(loop_ctx))
        if not bool(done):
            return (EMPTY_IMAGES,)
        return (_merge_images_for_ctx(ctx),)


class MieLoopCleanupImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_ctx": ("MIE_LOOP_CTX",),
            }
        }

    RETURN_TYPES = ("MIE_LOOP_CTX", "BOOLEAN")
    RETURN_NAMES = ("loop_ctx", "cleaned")
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, loop_ctx):
        ctx = copy.deepcopy(_validate_loop_ctx(loop_ctx))
        ref = _ensure_collector_slot(ctx, "image").get("ref")
        if not ref:
            return (ctx, False)
        removed = _pop_collector_items("image", ref)
        run_meta = _ensure_runtime_meta(ctx.get("run_id", ""))
        _remove_runtime_collector_ref(run_meta, "image", ref)
        ctx["collectors"]["image"] = {"ref": None, "count": 0}
        return (ctx, len(removed) > 0)


class MieImageGrid:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "cols": ("INT", {"default": 2, "min": 1, "max": 64}),
                "pad": ("INT", {"default": 4, "min": 0, "max": 128}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("grid",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, images, cols=2, pad=4):
        return (_grid_images(images, cols, pad),)


class MieLoopCollectText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_ctx": ("MIE_LOOP_CTX",),
                "text": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("MIE_LOOP_CTX",)
    RETURN_NAMES = ("loop_ctx",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, loop_ctx, text):
        ctx = copy.deepcopy(_validate_loop_ctx(loop_ctx))
        run_meta = _ensure_runtime_meta(ctx.get("run_id", ""))
        text_collector = _ensure_collector_slot(ctx, "text")
        ref = text_collector.get("ref")
        text_store = _ensure_runtime_collector_store("text")
        if not ref:
            ref = _create_collector_ref("txtcol", ctx)
            text_collector["ref"] = ref
            text_store[ref] = []
            text_refs = _ensure_runtime_collector_refs(run_meta, "text")
            if ref not in text_refs:
                text_refs.append(ref)
        if ref not in text_store:
            text_store[ref] = []
        text_store[ref].append(str(text))
        text_collector["count"] = len(text_store[ref])
        mie_log(
            f"LoopCollectText: loop_id={ctx['loop_id']}, run_id={ctx['run_id']}, ref={ref}, count={text_collector['count']}"
        )
        return (ctx,)


class MieLoopFinalizeTextList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_ctx": ("MIE_LOOP_CTX",),
                "done": ("BOOLEAN", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_list_json",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, loop_ctx, done):
        ctx = copy.deepcopy(_validate_loop_ctx(loop_ctx))
        if not bool(done):
            return (json.dumps([], ensure_ascii=False),)
        ref = _ensure_collector_slot(ctx, "text").get("ref")
        if not ref:
            return (json.dumps([], ensure_ascii=False),)
        texts = _pop_collector_items("text", ref)
        run_meta = _ensure_runtime_meta(ctx.get("run_id", ""))
        _remove_runtime_collector_ref(run_meta, "text", ref)
        mie_log(
            f"LoopFinalizeTextList: loop_id={ctx['loop_id']}, run_id={ctx['run_id']}, ref={ref}, count={len(texts)}"
        )
        return (json.dumps(texts, ensure_ascii=False),)


class MieLoopCleanupText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_ctx": ("MIE_LOOP_CTX",),
            }
        }

    RETURN_TYPES = ("MIE_LOOP_CTX", "BOOLEAN")
    RETURN_NAMES = ("loop_ctx", "cleaned")
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, loop_ctx):
        ctx = copy.deepcopy(_validate_loop_ctx(loop_ctx))
        ref = _ensure_collector_slot(ctx, "text").get("ref")
        if not ref:
            return (ctx, False)
        removed = _pop_collector_items("text", ref)
        run_meta = _ensure_runtime_meta(ctx.get("run_id", ""))
        _remove_runtime_collector_ref(run_meta, "text", ref)
        cleaned = len(removed) > 0
        ctx["collectors"]["text"] = {"ref": None, "count": 0}
        mie_log(
            f"LoopCleanupText: loop_id={ctx['loop_id']}, run_id={ctx['run_id']}, ref={ref}, cleaned={cleaned}"
        )
        return (ctx, cleaned)


class MieLoopCollectJSON:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_ctx": ("MIE_LOOP_CTX",),
                "item_json": ("STRING", {"default": "{}", "multiline": True}),
            }
        }

    RETURN_TYPES = ("MIE_LOOP_CTX",)
    RETURN_NAMES = ("loop_ctx",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, loop_ctx, item_json):
        ctx = copy.deepcopy(_validate_loop_ctx(loop_ctx))
        run_meta = _ensure_runtime_meta(ctx.get("run_id", ""))
        json_collector = _ensure_collector_slot(ctx, "json")
        ref = json_collector.get("ref")
        json_store = _ensure_runtime_collector_store("json")
        try:
            parsed = json.loads(item_json)
        except Exception as e:
            raise ValueError(f"item_json is invalid JSON: {e}") from e
        if not ref:
            ref = _create_collector_ref("jsoncol", ctx)
            json_collector["ref"] = ref
            json_store[ref] = []
            json_refs = _ensure_runtime_collector_refs(run_meta, "json")
            if ref not in json_refs:
                json_refs.append(ref)
        if ref not in json_store:
            json_store[ref] = []
        json_store[ref].append(parsed)
        json_collector["count"] = len(json_store[ref])
        mie_log(
            f"LoopCollectJSON: loop_id={ctx['loop_id']}, run_id={ctx['run_id']}, ref={ref}, count={json_collector['count']}"
        )
        return (ctx,)


class MieLoopFinalizeJSONList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_ctx": ("MIE_LOOP_CTX",),
                "done": ("BOOLEAN", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_list",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, loop_ctx, done):
        ctx = copy.deepcopy(_validate_loop_ctx(loop_ctx))
        if not bool(done):
            return (json.dumps([], ensure_ascii=False),)
        ref = _ensure_collector_slot(ctx, "json").get("ref")
        if not ref:
            return (json.dumps([], ensure_ascii=False),)
        items = _pop_collector_items("json", ref)
        run_meta = _ensure_runtime_meta(ctx.get("run_id", ""))
        _remove_runtime_collector_ref(run_meta, "json", ref)
        mie_log(
            f"LoopFinalizeJSONList: loop_id={ctx['loop_id']}, run_id={ctx['run_id']}, ref={ref}, count={len(items)}"
        )
        return (json.dumps(items, ensure_ascii=False),)


class MieLoopCleanupJSON:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loop_ctx": ("MIE_LOOP_CTX",),
            }
        }

    RETURN_TYPES = ("MIE_LOOP_CTX", "BOOLEAN")
    RETURN_NAMES = ("loop_ctx", "cleaned")
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, loop_ctx):
        ctx = copy.deepcopy(_validate_loop_ctx(loop_ctx))
        ref = _ensure_collector_slot(ctx, "json").get("ref")
        if not ref:
            return (ctx, False)
        removed = _pop_collector_items("json", ref)
        run_meta = _ensure_runtime_meta(ctx.get("run_id", ""))
        _remove_runtime_collector_ref(run_meta, "json", ref)
        cleaned = len(removed) > 0
        ctx["collectors"]["json"] = {"ref": None, "count": 0}
        mie_log(
            f"LoopCleanupJSON: loop_id={ctx['loop_id']}, run_id={ctx['run_id']}, ref={ref}, cleaned={cleaned}"
        )
        return (ctx, cleaned)
