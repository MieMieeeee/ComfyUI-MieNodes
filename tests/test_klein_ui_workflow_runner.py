"""The UI verifier must execute connected inputs and honor frontend bypasses."""
import importlib.util
from pathlib import Path

def load_runner(monkeypatch):
    monkeypatch.setattr("sys.argv", ["runner", "unused", "unused"])
    spec = importlib.util.spec_from_file_location("klein_ui_runner", Path(__file__).parents[1] / "_run_ui_workflow.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, "api", lambda _: {"input": {"required": {"text": ["STRING", {}]}}})
    return runner

def test_connected_prompt_wins_over_stale_saved_widget(monkeypatch):
    runner = load_runner(monkeypatch)
    wf = {"nodes": [{"id": 1, "type": "CLIPTextEncode", "inputs": [{"name": "text", "link": 5}], "widgets_values_named": {"text": "stale prompt"}}], "links": [[5, 2, 0, 1, 0, "STRING"]]}
    assert runner.to_api(wf)["1"]["inputs"]["text"] == ["2", 0]

def test_bypassed_image_scaler_resolves_to_original_reference(monkeypatch):
    runner = load_runner(monkeypatch)
    wf = {"nodes": [
        {"id": 1, "type": "LoadImage", "inputs": []},
        {"id": 2, "type": "ImageScaleToTotalPixels", "mode": 4, "inputs": [{"name": "image", "type": "IMAGE", "link": 5}], "outputs": [{"type": "IMAGE"}]},
        {"id": 3, "type": "ReferenceMultiLatent", "inputs": [{"name": "images.image0", "link": 6}]},
    ], "links": [[5, 1, 0, 2, 0, "IMAGE"], [6, 2, 0, 3, 0, "IMAGE"]]}
    graph = runner.to_api(wf)
    assert graph["3"]["inputs"]["images.image0"] == ["1", 0]
    assert "2" not in graph