"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 3 - Python Infrastructure.

This module does not replace tooling such as ``make`` or shell scripts,
but it shows how you might:

- detect the active interpreter and environment,
- print concise diagnostics for bug reports,
- and provide a small entry point that checks basic prerequisites.

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass


@dataclass
class EnvInfo:
    """Lightweight snapshot of the current Python environment."""

    python_executable: str
    python_version: str
    platform: str


def collect_env_info() -> EnvInfo:
    """Return basic information about the current interpreter."""

    return EnvInfo(
        python_executable=sys.executable,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
    )


def main() -> None:
    """Print a short diagnostics block similar to bug-report templates."""

    info = collect_env_info()
    print("Python executable:", info.python_executable)
    print("Python version:   ", info.python_version)
    print("Platform:         ", info.platform)

    assert info.python_version, "version must not be empty"
    assert info.python_executable, "executable path must not be empty"


if __name__ == "__main__":
    main()

