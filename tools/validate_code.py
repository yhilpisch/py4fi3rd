"""
Python for Finance, 3rd ed., O'Reilly (2026).
Code Validation Helper

Validate Python scripts in the ``code/`` directory by checking syntax,
import availability, and optional execution. This version is tailored
to the book project and, by default, validates both chapter examples
and figure generators.

Usage
  python tools/validate_code.py
  python tools/validate_code.py chapters/ch05*.py
  python tools/validate_code.py --skip-execute
"""
from __future__ import annotations

import argparse
import ast
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = ROOT / "code"
DEFAULT_PATTERNS = ["chapters/*.py", "figures/*.py"]
BULLET = "  •"

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", "/tmp/mplconfig")).expanduser()
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))


def format_duration(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.1f}µs"
    if seconds < 1.0:
        return f"{seconds * 1_000:.1f}ms"
    return f"{seconds:.2f}s"


def collect_modules_from_source(source: str) -> List[str]:
    modules = set()
    try:
        node = ast.parse(source)
    except SyntaxError:
        return []
    for stmt in ast.walk(node):
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(stmt, ast.ImportFrom):
            if stmt.level == 0 and stmt.module:
                modules.add(stmt.module.split(".")[0])
    return sorted(m for m in modules if m and not m.startswith("."))


def check_syntax(path: Path) -> Tuple[bool, Exception | None, float, str, list[str]]:
    t0 = time.perf_counter()
    try:
        source = path.read_text(encoding="utf8")
    except Exception as exc:  # pragma: no cover - I/O error path
        return False, exc, time.perf_counter() - t0, "", []
    try:
        ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return False, exc, time.perf_counter() - t0, source, []
    modules = collect_modules_from_source(source)
    return True, None, time.perf_counter() - t0, source, modules


def check_imports(
    modules: Sequence[str], *, probe: bool = True
) -> Tuple[bool, list[Tuple[str, Exception]], float]:
    missing: list[Tuple[str, Exception]] = []
    t0 = time.perf_counter()
    if probe:
        for mod in modules:
            try:
                importlib.import_module(mod)
            except Exception as exc:  # pragma: no cover - depends on local env
                missing.append((mod, exc))
    duration = time.perf_counter() - t0
    return (len(missing) == 0), missing, duration


def execute_script(
    path: Path,
    *,
    timeout: int,
) -> Tuple[bool, Exception | None, float]:
    t0 = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("PYTHONWARNINGS", "ignore")
    env.setdefault("MPLBACKEND", "Agg")
    old_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(CODE_ROOT) if not old_path else f"{CODE_ROOT}{os.pathsep}{old_path}"
    )
    try:
        subprocess.run(
            [sys.executable, str(path)],
            cwd=str(ROOT),
            timeout=timeout,
            check=True,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return True, None, time.perf_counter() - t0
    except subprocess.CalledProcessError as exc:
        return False, exc, time.perf_counter() - t0
    except subprocess.TimeoutExpired as exc:
        return False, exc, time.perf_counter() - t0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Python scripts in code/ with syntax/import/execute checks"
        )
    )
    parser.add_argument(
        "files",
        nargs="*",
        help=(
            "Code paths or glob patterns relative to code/. "
            "Defaults to chapters/*.py and figures/*.py."
        ),
    )
    parser.add_argument(
        "--skip-execute", action="store_true", help="Skip execution step"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-script execution timeout (seconds)",
    )
    parser.add_argument(
        "--import-probe/--no-import-probe",
        dest="import_probe",
        default=True,
        action=argparse.BooleanOptionalAction,
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
        else:
            paths.extend(sorted(CODE_ROOT.glob(pat)))
    seen: set[Path] = set()
    ordered: list[Path] = []
    for p in paths:
        if p not in seen:
            ordered.append(p)
            seen.add(p)
    return ordered


def main() -> None:
    args = parse_args()
    code_paths = expand_files(args.files)
    if not code_paths:
        print("No code files matched.", file=sys.stderr)
        sys.exit(1)

    total = len(code_paths)
    failures: list[tuple[Path, str, Exception | str]] = []

    for idx, path in enumerate(code_paths, start=1):
        path = path.resolve()
        rel = path.relative_to(ROOT)
        header = f"[{idx}/{total}] Validating {rel}"
        print(header)

        total_start = time.perf_counter()

        ok_syntax, exc, t_syntax, _, modules = check_syntax(path)
        status = "OK" if ok_syntax else f"FAIL ({exc})"
        print(f"{BULLET} Syntax: {status:<8} ({format_duration(t_syntax)})")
        if not ok_syntax:
            failures.append((rel, "syntax", exc or "syntax error"))
            print(f"{BULLET} Imports: SKIP")
            print(f"{BULLET} Execute: SKIP")
            total_duration = time.perf_counter() - total_start
            print(f"{BULLET} Total: {format_duration(total_duration)}\n")
            continue

        ok_imports, missing, t_imports = check_imports(
            modules, probe=args.import_probe
        )
        if ok_imports:
            print(
                f"{BULLET} Imports: OK   ({format_duration(t_imports)} probe)"
            )
        else:
            reason = ", ".join(f"{mod}: {exc!s}" for mod, exc in missing)
            print(
                f"{BULLET} Imports: FAIL ({reason}) "
                f"({format_duration(t_imports)} probe)"
            )
            failures.append((rel, "imports", reason))

        if args.skip_execute:
            print(f"{BULLET} Execute: SKIP")
            total_duration = time.perf_counter() - total_start
            print(f"{BULLET} Total: {format_duration(total_duration)}\n")
            continue

        ok_exec, exec_exc, t_exec = execute_script(path, timeout=args.timeout)
        exec_status = "OK" if ok_exec else f"FAIL ({exec_exc})"
        print(f"{BULLET} Execute: {exec_status:<8} ({format_duration(t_exec)})")
        if not ok_exec:
            failures.append((rel, "execute", exec_exc or "unknown error"))

        total_duration = time.perf_counter() - total_start
        print(f"{BULLET} Total: {format_duration(total_duration)}\n")

    if failures:
        print("Summary: FAIL\n")
        for rel, stage, err in failures:
            print(f" - {rel}: {stage} failed ({err})")
        sys.exit(1)
    else:
        print("Summary: OK")


if __name__ == "__main__":
    main()
