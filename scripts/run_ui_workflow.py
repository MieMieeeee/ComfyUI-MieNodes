"""Convert a saved UI workflow to API format and run it on a ComfyUI server.

This is the check that matters for a drag-in file: the frontend JSON has to
survive conversion into the API graph the server actually executes. Widget
values come from ``widgets_values_named`` where the frontend saved it (exact
names, no guessing) and fall back to positional order otherwise.
"""
import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1]
WORKFLOW = sys.argv[2]


def api(node_type):
    with urllib.request.urlopen(f"{BASE}/object_info/{urllib.parse.quote(node_type)}",
                                timeout=60) as r:
        d = json.loads(r.read())
    return d.get(node_type)


import urllib.parse  # noqa: E402


def widget_names(spec):
    """INPUT_TYPES entry names, in the order the frontend lists widgets."""
    names = []
    for sec in ("required", "optional"):
        for k in spec.get("input", {}).get(sec, {}):
            names.append(k)
    return names


def to_api(wf):
    spec_cache = {}
    out = {}
    nodes = {n["id"]: n for n in wf["nodes"]}
    links = {link[0]: link for link in wf["links"]}

    def source(link_id, visited=None):
        visited = set() if visited is None else visited
        if link_id in visited:
            raise ValueError("Cycle while resolving bypassed nodes")
        visited.add(link_id)
        link = links[link_id]
        upstream = nodes.get(link[1])
        if upstream and upstream.get("mode", 0) == 4:
            matching = [i for i in upstream.get("inputs", [])
                        if i.get("link") is not None and i.get("type") == link[5]]
            if len(matching) != 1:
                raise ValueError(f"Ambiguous bypass at node {upstream['id']}; export with ComfyUI")
            return source(matching[0]["link"], visited)
        return [str(link[1]), link[2]]

    for n in wf["nodes"]:
        if n.get("mode", 0) in (2, 4):
            continue
        nt = n["type"]
        if nt not in spec_cache:
            spec_cache[nt] = api(nt)
        spec = spec_cache[nt]
        named = n.get("widgets_values_named")
        vals = n.get("widgets_values") or []
        order = widget_names(spec) if spec else []

        inputs = {}
        # socket inputs that carry a link
        for i in n.get("inputs", []):
            if i.get("link") is not None:
                inputs[i["name"]] = source(i["link"])

        # widget values
        if named:
            for k, v in named.items():
                if k == "control_after_generate":
                    continue
                if k in inputs:
                    continue
                if spec and k not in order:
                    continue
                # Socketless widgets (BOUNDING_BOXES etc.) must arrive as the
                # JSON string the API expects, not as a raw list -- a list is
                # read as a link and fails validation.
                if isinstance(v, (list, dict)):
                    v = json.dumps(v, ensure_ascii=False)
                inputs[k] = v
            # required widgets the frontend may not have saved (e.g. an enum
            # with no stored value) get their schema default.
            if spec:
                for section in ("required", "optional"):
                    for k, entry in spec["input"].get(section, {}).items():
                        if k in inputs:
                            continue
                        if section != "required":
                            continue
                        t = entry[0] if isinstance(entry, (list, tuple)) else entry
                        if isinstance(t, list) and t and t[0] == "COMBO":
                            inputs[k] = t[1][0] if t[1] else ""
                        elif isinstance(t, list) and t and t[0] in (
                                "COMBO_DYNAMICCOMBO_V3", "COMBO_V3"):
                            # A dynamic combo's own sub-inputs are separate
                            # widgets; fill them from the saved values below.
                            continue
                        elif isinstance(t, str) and t in (
                                "IMAGE", "LATENT", "MASK", "CONDITIONING", "MODEL",
                                "CLIP", "VAE", "AUDIO", "VIDEO", "BOUNDING_BOXES"):
                            continue

            # DynamicCombo sub-inputs (TextGenerate's sampling_mode.*) are
            # saved as their own widgets but never appear in the parent's
            # INPUT_TYPES, so fill them from widgets_values_named directly.
            if named:
                for k, v in named.items():
                    if k in inputs:
                        continue
                    if "." in k and k.split(".", 1)[0] in inputs:
                        inputs[k] = v
        else:
            # positional fallback: widget entries in INPUT_TYPES order
            wi = 0
            for k in order:
                if spec:
                    entry = spec["input"].get("required", {}).get(k) or \
                            spec["input"].get("optional", {}).get(k)
                    t = entry[0] if isinstance(entry, (list, tuple)) else entry
                    if isinstance(t, str) and t in ("IMAGE", "LATENT", "MASK", "CONDITIONING",
                                                    "MODEL", "CLIP", "VAE", "AUDIO", "VIDEO"):
                        continue
                if wi < len(vals):
                    if k not in inputs:
                        inputs[k] = vals[wi]
                    wi += 1
        out[str(n["id"])] = {"class_type": nt, "inputs": inputs}
    return out


def post(path, payload):
    req = urllib.request.Request(
        f"{BASE}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())


def wait(pid, tries=900):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(f"{BASE}/history/{pid}", timeout=30) as r:
                d = json.loads(r.read())
            if pid in d:
                h = d[pid]
                if h.get("status", {}).get("completed"):
                    return h
                for m in h.get("status", {}).get("messages", []):
                    if m[0] == "execution_error":
                        return h
        except Exception:
            pass
        time.sleep(3)
    return None


if __name__ == "__main__":
    wf = json.load(open(WORKFLOW, encoding="utf-8"))
    apiwf = to_api(wf)
    print("converted nodes:", len(apiwf))
    resp = post("/prompt", {"prompt": apiwf})
    if "error" in resp:
        print("REJECTED:", json.dumps(resp.get("error", {}), ensure_ascii=False)[:600])
        for d in resp.get("node_errors", {}).values():
            for e in d.get("errors", []):
                print("  *", e.get("type"), "|", (e.get("message") or "")[:300])
        sys.exit(1)
    print("submitted:", resp["prompt_id"])
    t0 = time.time()
    h = wait(resp["prompt_id"])
    print(f"status={(h or {}).get('status', {}).get('status_str')} elapsed={time.time()-t0:.0f}s")
    for m in (h or {}).get("status", {}).get("messages", []):
        if m[0] == "execution_error":
            d = m[1]
            print("ERROR node", d.get("node_id"), d.get("node_type"))
            print("  ", (d.get("exception_message") or "")[:400])
    for v in (h or {}).get("outputs", {}).values():
        for im in v.get("images") or []:
            print("  image:", im.get("filename"), "|", im.get("subfolder"))
    if not h or h.get("status", {}).get("status_str") != "success":
        sys.exit(1)
