"""One-shot probe for a single (label, url, model, key) tuple.

Usage: python tests/_probe_one.py <label> <url> <model> <key>
Prints: [<label>] <model> HTTP <code> | <body-snippet>
"""
import sys
import requests

label, url, model, key = sys.argv[1:5]
body = {
    "model": model,
    "messages": [{"role": "user", "content": "OK"}],
    "stream": False,
    "max_completion_tokens": 8,
    "temperature": 1.0,
    "top_p": 0.95,
}
headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
try:
    r = requests.post(url, json=body, headers=headers, timeout=12)
    print(f"[{label}] {model} HTTP {r.status_code} | {(r.text or '')[:300]!r}")
except Exception as e:
    print(f"[{label}] {model} EXC {type(e).__name__}: {e}")
