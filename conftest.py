# Root conftest.py — prevents pytest from importing __init__.py at the root.
# The project root has __init__.py (ComfyUI plugin) with relative imports that fail outside ComfyUI.
import os

# Rename __init__.py temporarily so pytest doesn't try to import it as a package
_root_init = os.path.join(os.path.dirname(os.path.abspath(__file__)), "__init__.py")
_backup_init = _root_init + ".bak_test"


def pytest_configure(config):
    """Temporarily hide __init__.py from pytest."""
    if os.path.exists(_root_init) and not os.path.exists(_backup_init):
        os.rename(_root_init, _backup_init)


def pytest_unconfigure(config):
    """Restore __init__.py after pytest finishes."""
    if os.path.exists(_backup_init):
        os.rename(_backup_init, _root_init)
