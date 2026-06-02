"""Top-level package for chapter and figure code.

Allows imports such as ``code.chapters.ch23_vectorized_and_event_based_backtesting``.

Because this package has the same name as the Python standard library
``code`` module, we re-export every public name from the stdlib so that
``code.InteractiveConsole`` (and similar) continue to work when this
package is on ``sys.path``.
"""

import importlib.util as _ilu
import sys as _sys
from pathlib import Path as _Path


def _load_stdlib_code():
    """Locate and load the stdlib code.py by file path.

    Tries multiple strategies so the lookup works across regular
    CPython installs, virtual environments, and Docker containers.
    """
    # Candidate paths for the stdlib code.py
    _candidates: list[_Path] = []

    # Strategy 1: sysconfig paths
    try:
        import sysconfig as _sc  # noqa: F811
        for _key in ("stdlib", "platstdlib"):
            _candidates.append(_Path(_sc.get_path(_key)) / "code.py")
    except Exception:
        pass

    # Strategy 2: sys.prefix / sys.base_prefix with version subdir
    _ver = f"python{_sys.version_info.major}.{_sys.version_info.minor}"
    for _prefix in (
        getattr(_sys, "base_prefix", None),
        _sys.prefix,
        getattr(_sys, "base_exec_prefix", None),
        _sys.exec_prefix,
    ):
        if _prefix:
            _candidates.extend(
                [
                    _Path(_prefix) / "lib" / _ver / "code.py",
                    _Path(_prefix) / "Lib" / "code.py",
                    _Path(_prefix) / _ver / "code.py",
                ]
            )

    # Strategy 3: Walk sys.path entries (skip the directory that
    # contains this local code/ package to avoid loading ourselves).
    try:
        _this_parent = str(_Path(__path__[0]).parent)
    except Exception:
        _this_parent = ""
    for _entry in _sys.path:
        if _entry and str(_entry) != _this_parent:
            _p = _Path(_entry) / "code.py"
            if _p.is_file():
                _candidates.append(_p)

    for _path in _candidates:
        if not _path.is_file():
            continue
        try:
            _spec = _ilu.spec_from_file_location(
                "_stdlib_code", str(_path)
            )
            if _spec and _spec.loader:
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                # Verify this is actually the stdlib module
                if hasattr(_mod, "InteractiveConsole"):
                    return _mod
        except Exception:
            continue
    return None


_stdlib_code = _load_stdlib_code()

if _stdlib_code is not None:
    for _attr in dir(_stdlib_code):
        if not _attr.startswith("__") and _attr not in globals():
            globals()[_attr] = getattr(_stdlib_code, _attr)
    InteractiveConsole = _stdlib_code.InteractiveConsole  # noqa: F401
    InteractiveInterpreter = _stdlib_code.InteractiveInterpreter  # noqa: F401
    compile_command = _stdlib_code.compile_command  # noqa: F401
    del _stdlib_code

