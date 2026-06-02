"""Top-level package for chapter and figure code.

Allows imports such as ``code.chapters.ch23_vectorized_and_event_based_backtesting``.

Because this package has the same name as the Python standard library
``code`` module, we re-export every public name from the stdlib so that
``code.InteractiveConsole`` (and similar) continue to work when this
package is on ``sys.path``.
"""

import importlib.util
import sysconfig as _sysconfig
from pathlib import Path as _Path

_stdlib_code_path = _Path(_sysconfig.get_path("stdlib")) / "code.py"

if _stdlib_code_path.exists():
    _spec = importlib.util.spec_from_file_location(
        "_stdlib_code", str(_stdlib_code_path)
    )
    if _spec and _spec.loader:
        _stdlib_code = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_stdlib_code)
        for _attr in dir(_stdlib_code):
            if not _attr.startswith("__") and _attr not in globals():
                globals()[_attr] = getattr(_stdlib_code, _attr)
        InteractiveConsole = _stdlib_code.InteractiveConsole  # noqa: F401
        InteractiveInterpreter = _stdlib_code.InteractiveInterpreter  # noqa: F401
        compile_command = _stdlib_code.compile_command  # noqa: F401

    del _spec, _stdlib_code

