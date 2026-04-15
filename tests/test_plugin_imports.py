import importlib.util
import importlib
from importlib.machinery import SourceFileLoader
from pathlib import Path
import sys
import types


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLUGIN_INIT = PROJECT_ROOT / "__init__.py"
PLUGIN_INIT_BACKUP = PROJECT_ROOT / "__init__.py.bak_test"
PACKAGE_NAME = "comfyui_mienodes"
COMFY_STYLE_PACKAGE_NAME = r"E:\FF\ComfyUI_Mie_2026_V8_x_0\ComfyUI\custom_nodes\ComfyUI-MieNodes"
INTERNAL_PACKAGE = "_mienodes_internal"


def install_runtime_stubs():
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.models_dir = str(PROJECT_ROOT / "models")
    folder_paths.output_directory = str(PROJECT_ROOT / "output")
    folder_paths.input_directory = str(PROJECT_ROOT / "input")
    folder_paths.get_filename_list = lambda *_args, **_kwargs: []
    folder_paths.get_full_path = lambda *_args, **_kwargs: ""
    folder_paths.get_output_directory = lambda: folder_paths.output_directory
    folder_paths.get_input_directory = lambda: folder_paths.input_directory
    sys.modules["folder_paths"] = folder_paths

    torchaudio = types.ModuleType("torchaudio")
    torchaudio.load = lambda *_args, **_kwargs: None
    torchaudio.save = lambda *_args, **_kwargs: None
    sys.modules["torchaudio"] = torchaudio

    deepdiff = types.ModuleType("deepdiff")
    deepdiff.DeepDiff = dict
    sys.modules["deepdiff"] = deepdiff

    imagehash = types.ModuleType("imagehash")
    imagehash.average_hash = lambda *_args, **_kwargs: 0
    sys.modules["imagehash"] = imagehash


def clear_plugin_modules(package_name=PACKAGE_NAME):
    for module_name in list(sys.modules):
        if module_name == package_name or module_name.startswith(f"{package_name}."):
            del sys.modules[module_name]
        if module_name == INTERNAL_PACKAGE or module_name.startswith(f"{INTERNAL_PACKAGE}."):
            del sys.modules[module_name]


def load_plugin_module(package_name=PACKAGE_NAME, register_before_exec=True):
    install_runtime_stubs()
    clear_plugin_modules(package_name)
    plugin_entry = PLUGIN_INIT if PLUGIN_INIT.exists() else PLUGIN_INIT_BACKUP
    assert plugin_entry.exists(), "plugin entry module should exist for import regression tests"
    loader = SourceFileLoader(package_name, str(plugin_entry))
    spec = importlib.util.spec_from_loader(
        package_name,
        loader,
    )
    spec.submodule_search_locations = [str(PROJECT_ROOT)]
    module = importlib.util.module_from_spec(spec)
    if register_before_exec:
        sys.modules[package_name] = module
    loader.exec_module(module)
    return module


def test_plugin_root_import_exposes_node_mappings():
    plugin = load_plugin_module()

    assert hasattr(plugin, "NODE_CLASS_MAPPINGS")
    assert hasattr(plugin, "NODE_DISPLAY_NAME_MAPPINGS")
    assert isinstance(plugin.NODE_CLASS_MAPPINGS, dict)
    assert isinstance(plugin.NODE_DISPLAY_NAME_MAPPINGS, dict)


def test_plugin_root_import_keeps_representative_nodes_available():
    plugin = load_plugin_module()

    expected_nodes = {
        "BatchRenameFiles|Mie",
        "ShowAnything|Mie",
        "PromptGenerator|Mie",
        "SetGeneralLLMServiceConnector|Mie",
        "WavConcat|Mie",
        "QwenTTSNode|Mie",
        "SingleImageToVideo|Mie",
        "MieLoopStart|Mie",
    }

    assert expected_nodes.issubset(plugin.NODE_CLASS_MAPPINGS.keys())
    for node_name in expected_nodes:
        assert isinstance(plugin.NODE_CLASS_MAPPINGS[node_name], type)


def test_refactored_packages_are_importable():
    load_plugin_module()

    expected_modules = {
        f"{PACKAGE_NAME}.core.utils",
        f"{PACKAGE_NAME}.nodes.common",
        f"{PACKAGE_NAME}.nodes.files",
        f"{PACKAGE_NAME}.nodes.llm",
        f"{PACKAGE_NAME}.nodes.media",
        f"{PACKAGE_NAME}.nodes.loop",
        f"{PACKAGE_NAME}.nodes.loop.loop",
        f"{PACKAGE_NAME}.services.llm",
        f"{PACKAGE_NAME}.services.tts",
    }

    for module_name in expected_modules:
        imported = importlib.import_module(module_name)
        assert imported is not None


def test_root_directory_keeps_only_plugin_entry_for_python_modules():
    legacy_root_modules = {
        "audio_operator.py",
        "caption_file_operator.py",
        "common.py",
        "downloader.py",
        "image_operator.py",
        "llm_service_connector.py",
        "loop.py",
        "prompt_generator.py",
        "string_operator.py",
        "translator.py",
        "tts_generator.py",
        "tts_service_connector.py",
        "utils.py",
    }

    for filename in legacy_root_modules:
        assert not (PROJECT_ROOT / filename).exists()


def test_plugin_root_import_works_with_comfy_style_module_name():
    plugin = load_plugin_module(COMFY_STYLE_PACKAGE_NAME)

    assert hasattr(plugin, "NODE_CLASS_MAPPINGS")
    assert "MieLoopStart|Mie" in plugin.NODE_CLASS_MAPPINGS


def test_plugin_root_import_works_without_pre_registering_module():
    plugin = load_plugin_module(
        COMFY_STYLE_PACKAGE_NAME,
        register_before_exec=False,
    )

    assert hasattr(plugin, "NODE_CLASS_MAPPINGS")
    assert "MieLoopStart|Mie" in plugin.NODE_CLASS_MAPPINGS
