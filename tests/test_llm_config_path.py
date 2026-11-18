"""The node shell resolves configuration paths before entering the pip client."""
import importlib.util
from pathlib import Path
import sys
import types

def test_connector_default_config_is_relative_to_plugin(monkeypatch):
    source = Path(__file__).resolve().parents[1] / "services" / "llm.py"
    stub = types.ModuleType("_mienodes_internal.core.utils")
    stub.mie_log = lambda *args: None
    monkeypatch.setitem(sys.modules, "_mienodes_internal.core.utils", stub)
    spec = importlib.util.spec_from_file_location("_mie_config_shell", source)
    shell = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shell)
    monkeypatch.setattr(shell, "MiniMaxTokenPlanConnectorGeneral", lambda *args, **kwargs: kwargs)
    result = shell.SetMiniMaxTokenPlanLLMServiceConnector().execute("", "MiniMax-M3")[0]
    assert Path(result["config_file"]) == source.parents[1] / "mie_llm_keys.json"

def test_explicit_config_path_is_preserved(monkeypatch, tmp_path):
    source = Path(__file__).resolve().parents[1] / "services" / "llm.py"
    stub = types.ModuleType("_mienodes_internal.core.utils")
    stub.mie_log = lambda *args: None
    monkeypatch.setitem(sys.modules, "_mienodes_internal.core.utils", stub)
    spec = importlib.util.spec_from_file_location("_mie_config_shell_absolute", source)
    shell = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shell)
    monkeypatch.setattr(shell, "MiniMaxTokenPlanConnectorGeneral", lambda *args, **kwargs: kwargs)
    target = tmp_path / "keys.json"
    result = shell.SetMiniMaxTokenPlanLLMServiceConnector().execute("", "MiniMax-M3", config_file=str(target))[0]
    assert Path(result["config_file"]) == target
