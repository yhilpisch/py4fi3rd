"""Validate the companion repository in a Colab-style runtime.

This is intentionally separate from ``validate_notebooks.py``.  The regular
validator checks notebook structure and local-kernel execution.  This helper
first exercises the repository's Colab/local bootstrap, then optionally runs
selected scripts and notebooks from a fresh kernel.

Usage from the repository root::

    python tools/validate_colab.py
    python tools/validate_colab.py --scripts
    python tools/validate_colab.py --notebook ch20_*.ipynb
    python tools/validate_colab.py --all-notebooks --timeout 300
    python tools/validate_colab.py --all-notebooks --log-file /content/validation.log
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks"
DATA_ROOT = ROOT / "data"

if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))

from _book_notebook_support import setup_notebook  # noqa: E402


REQUIRED_DATA = (
    "eod_data.csv",
    "hf_data.csv",
    "spx_options_snapshot.csv",
)

DEFAULT_SCRIPTS = (
    "code/chapters/ch20_asset_management_systems_and_reporting.py",
    "code/chapters/ch21_asset_management_library_in_python.py",
    "code/chapters/ch24_building_a_market_and_broker_for_trading.py",
    "code/chapters/ch25_automated_deployment_of_trading_strategies.py",
    "code/figures/ch31_atm_term_structure.py",
)


class _Tee:
    """Write validation output to the console and a shareable log file."""

    def __init__(self, console, log_handle):
        self.console = console
        self.log_handle = log_handle

    def write(self, value: str) -> int:
        self.console.write(value)
        self.log_handle.write(value)
        return len(value)

    def flush(self) -> None:
        self.console.flush()
        self.log_handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Colab-style companion-repository checks."
    )
    parser.add_argument(
        "--scripts",
        action="store_true",
        help="Run the representative chapter scripts.",
    )
    parser.add_argument(
        "--script",
        action="append",
        default=[],
        help="Run one script path; may be repeated.",
    )
    parser.add_argument(
        "--notebook",
        action="append",
        default=[],
        help="Run one notebook path or notebooks/ glob; may be repeated.",
    )
    parser.add_argument(
        "--all-notebooks",
        action="store_true",
        help="Run all non-checkpoint chapter and lab notebooks.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-cell notebook timeout in seconds.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help=(
            "Write a copy of console output and errors to this file. "
            "Parent directories are created automatically."
        ),
    )
    return parser.parse_args()


def expand_notebooks(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.is_file():
            paths.append(candidate.resolve())
            continue
        paths.extend(sorted(NOTEBOOK_ROOT.glob(pattern)))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen or ".ipynb_checkpoints" in resolved.parts:
            continue
        unique.append(resolved)
        seen.add(resolved)
    return unique


def bootstrap() -> None:
    context = setup_notebook(notebook_subdir="notebooks")
    if context["PROJECT_ROOT"] != ROOT:
        raise RuntimeError(
            f"Bootstrap found {context['PROJECT_ROOT']}, expected {ROOT}."
        )

    for module_name in ("assetlib", "engine", "dxlib"):
        module = importlib.import_module(module_name)
        print(f"  import {module_name}: {module.__file__}")

    for filename in REQUIRED_DATA:
        path = DATA_ROOT / filename
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"  data {filename}: OK")


def run_script(path: Path, timeout: int) -> None:
    env = os.environ.copy()
    code_root = str(ROOT / "code")
    current_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (code_root, current_pythonpath) if part
    )
    env.setdefault("MPLBACKEND", "Agg")
    mplconfig_dir = ROOT / "_tmp" / "mplconfig"
    mplconfig_dir.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(mplconfig_dir)

    print(f"  script {path.relative_to(ROOT)}")
    try:
        completed = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT,
            env=env,
            check=True,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, end="")
        raise
    except subprocess.TimeoutExpired as exc:
        if exc.stdout:
            output = exc.stdout
            if isinstance(output, bytes):
                output = output.decode(errors="replace")
            print(output, end="")
        raise
    else:
        if completed.stdout:
            print(completed.stdout, end="")


def run_notebook(path: Path, timeout: int) -> None:
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Notebook validation requires nbformat and nbclient."
        ) from exc

    with path.open("r", encoding="utf8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    print(f"  notebook {path.relative_to(ROOT)}")
    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()


def run_validation(args: argparse.Namespace) -> None:
    print(f"Colab-style validation root: {ROOT}")
    bootstrap()

    script_paths = [ROOT / path for path in args.script]
    if args.scripts:
        script_paths.extend(ROOT / path for path in DEFAULT_SCRIPTS)
    for path in script_paths:
        run_script(path, args.timeout)

    notebook_patterns = list(args.notebook)
    if args.all_notebooks:
        notebook_patterns.extend(("*.ipynb", "labs/*.ipynb"))
    for path in expand_notebooks(notebook_patterns):
        run_notebook(path, args.timeout)

    print("Colab-style validation: OK")


if __name__ == "__main__":
    args = parse_args()
    if args.log_file is None:
        run_validation(args)
    else:
        log_path = args.log_file.expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_handle:
            with redirect_stdout(_Tee(sys.stdout, log_handle)):
                with redirect_stderr(_Tee(sys.stderr, log_handle)):
                    print(f"Colab-style validation log: {log_path}")
                    try:
                        run_validation(args)
                    except BaseException:
                        traceback.print_exc()
                        raise SystemExit(1)
