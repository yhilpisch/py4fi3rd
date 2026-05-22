"""
Python for Finance, 3rd ed., O'Reilly (2026).
Notebook Validation Helper

Validate Jupyter notebooks in the ``notebooks/`` directory by checking JSON
readability, basic notebook structure, and optional execution. This version is
tailored to the book project and, by default, validates all chapter notebooks.

Usage
  python tools/validate_notebooks.py
  python tools/validate_notebooks.py ch15*.ipynb
  python tools/validate_notebooks.py --skip-execute
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = ROOT / "notebooks"
DEFAULT_PATTERNS = ["*.ipynb"]
BULLET = "  •"


def format_duration(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f}µs"
    if seconds < 1.0:
        return f"{seconds * 1_000:.1f}ms"
    return f"{seconds:.2f}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Jupyter notebooks in notebooks/ with "
            "JSON/structure/execute checks"
        )
    )
    parser.add_argument(
        "files",
        nargs="*",
        help=(
            "Notebook paths or glob patterns relative to notebooks/. "
            "Defaults to *.ipynb."
        ),
    )
    parser.add_argument(
        "--skip-execute",
        action="store_true",
        help="Skip notebook execution step",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-notebook execution timeout (seconds)",
    )
    return parser.parse_args()


def expand_files(patterns: Sequence[str]) -> list[Path]:
    if not patterns:
        patterns = DEFAULT_PATTERNS
    paths: list[Path] = []
    for pat in patterns:
        candidate = Path(pat)
        if candidate.is_file():
            paths.append(candidate.resolve())
            continue
        candidate_under_root = NOTEBOOK_ROOT / pat
        if candidate_under_root.is_file():
            paths.append(candidate_under_root.resolve())
            continue
        paths.extend(sorted(NOTEBOOK_ROOT.glob(pat)))
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            ordered.append(resolved)
            seen.add(resolved)
    return ordered


def check_json_and_structure(path: Path) -> tuple[bool, str | Exception, float]:
    t0 = time.perf_counter()
    try:
        payload = json.loads(path.read_text(encoding="utf8"))
    except Exception as exc:
        return False, exc, time.perf_counter() - t0

    try:
        validate_structure(payload)
    except Exception as exc:
        return False, exc, time.perf_counter() - t0
    return True, "OK", time.perf_counter() - t0


def validate_structure(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise TypeError("Notebook root must be a JSON object.")

    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("Notebook must contain a non-empty 'cells' list.")

    nbformat = payload.get("nbformat")
    if not isinstance(nbformat, int):
        raise ValueError("Notebook must define integer 'nbformat'.")

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Notebook must define object 'metadata'.")

    for idx, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise TypeError(f"Cell {idx} must be a JSON object.")
        cell_type = cell.get("cell_type")
        if cell_type not in {"markdown", "code", "raw"}:
            raise ValueError(f"Cell {idx} has invalid cell_type {cell_type!r}.")
        source = cell.get("source")
        if not isinstance(source, list):
            raise ValueError(f"Cell {idx} must use list-form 'source'.")
        if cell_type == "code":
            outputs = cell.get("outputs")
            if not isinstance(outputs, list):
                raise ValueError(f"Code cell {idx} must define list-form 'outputs'.")
            if "execution_count" not in cell:
                raise ValueError(f"Code cell {idx} must define 'execution_count'.")


def execute_notebook(path: Path, *, timeout: int) -> tuple[bool, str | Exception, float]:
    t0 = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("PYTHONWARNINGS", "ignore")
    env.setdefault("MPLBACKEND", "Agg")
    runner = (
        "import sys\n"
        "from pathlib import Path\n"
        "try:\n"
        "    import nbformat\n"
        "    from nbclient import NotebookClient\n"
        "except Exception as exc:\n"
        "    raise SystemExit(f'Missing notebook execution dependency: {exc}')\n"
        "path = Path(sys.argv[1])\n"
        "with path.open('r', encoding='utf8') as fh:\n"
        "    nb = nbformat.read(fh, as_version=4)\n"
        "client = NotebookClient(\n"
        "    nb,\n"
        "    timeout=None,\n"
        "    kernel_name='python3',\n"
        "    resources={'metadata': {'path': str(path.parent)}},\n"
        ")\n"
        "client.execute()\n"
    )
    try:
        subprocess.run(
            [sys.executable, "-c", runner, str(path)],
            cwd=str(ROOT),
            timeout=timeout,
            check=True,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        return True, "OK", time.perf_counter() - t0
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or str(exc)
        return False, detail, time.perf_counter() - t0
    except subprocess.TimeoutExpired as exc:
        return False, exc, time.perf_counter() - t0


def main() -> None:
    args = parse_args()
    notebook_paths = expand_files(args.files)
    if not notebook_paths:
        print("No notebook files matched.", file=sys.stderr)
        sys.exit(1)

    total = len(notebook_paths)
    failures: list[tuple[Path, str, str | Exception]] = []

    for idx, path in enumerate(notebook_paths, start=1):
        rel = path.relative_to(ROOT)
        print(f"[{idx}/{total}] Validating {rel}")
        total_start = time.perf_counter()

        ok_json, reason, t_json = check_json_and_structure(path)
        status = "OK" if ok_json else f"FAIL ({reason})"
        print(f"{BULLET} JSON+Struct: {status:<8} ({format_duration(t_json)})")
        if not ok_json:
            failures.append((rel, "json+structure", reason))
            print(f"{BULLET} Execute: SKIP")
            print(
                f"{BULLET} Total: {format_duration(time.perf_counter() - total_start)}\n"
            )
            continue

        if args.skip_execute:
            print(f"{BULLET} Execute: SKIP")
            print(
                f"{BULLET} Total: {format_duration(time.perf_counter() - total_start)}\n"
            )
            continue

        ok_exec, exec_reason, t_exec = execute_notebook(path, timeout=args.timeout)
        status = "OK" if ok_exec else f"FAIL ({exec_reason})"
        print(f"{BULLET} Execute: {status:<8} ({format_duration(t_exec)})")
        if not ok_exec:
            failures.append((rel, "execute", exec_reason))

        print(f"{BULLET} Total: {format_duration(time.perf_counter() - total_start)}\n")

    if failures:
        print("Summary: FAIL\n")
        for rel, stage, err in failures:
            print(f" - {rel}: {stage} failed ({err})")
        sys.exit(1)

    print("Summary: OK")


if __name__ == "__main__":
    main()
