from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping

REPO_NAME = "py4fi3rd"
REPO_URL = "https://github.com/yhilpisch/py4fi3rd.git"

DEFAULT_COLAB_PACKAGES: dict[str, str] = {
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "statsmodels": "statsmodels",
    "sympy": "sympy",
}


def running_in_colab() -> bool:
    return "google.colab" in sys.modules


def _has_project_layout(path: Path) -> bool:
    return (path / "code").is_dir() and (path / "notebooks").is_dir()


def find_project_root(start: Path | None = None) -> Path | None:
    origin = (start or Path.cwd()).resolve()
    for candidate in (origin, *origin.parents):
        if _has_project_layout(candidate):
            return candidate
    return None


def _clone_repo(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(destination)],
        check=True,
    )


def ensure_project_root() -> Path:
    root = find_project_root()
    if root is not None:
        return root
    if not running_in_colab():
        msg = "Could not locate the project root. Run inside the repo or Colab."
        raise RuntimeError(msg)
    root = Path("/content") / REPO_NAME
    if not root.exists():
        _clone_repo(root)
    return root


def ensure_packages(package_map: Mapping[str, str] | None = None) -> None:
    if not package_map:
        return
    missing: list[str] = []
    for module_name, package_name in package_map.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)
    if not missing:
        return
    unique_missing = sorted(set(missing))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *unique_missing],
        check=True,
    )


def configure_book_style() -> None:
    import matplotlib as mpl
    from matplotlib import style as mpl_style

    available_styles = set(mpl_style.available)
    if "seaborn-v0_8" in available_styles:
        mpl_style.use("seaborn-v0_8")
    elif "seaborn" in available_styles:
        mpl_style.use("seaborn")
    mpl.rcParams.update({"font.family": "serif", "figure.dpi": 300})


def setup_notebook(
    *,
    notebook_subdir: str,
    colab_packages: Mapping[str, str] | None = None,
    apply_style: bool = True,
) -> dict[str, Path]:
    if running_in_colab():
        package_map = dict(DEFAULT_COLAB_PACKAGES)
        if colab_packages:
            package_map.update(colab_packages)
        ensure_packages(package_map)

    project_root = ensure_project_root()
    notebook_dir = project_root / notebook_subdir
    code_dir = project_root / "code"
    chapters_dir = code_dir / "chapters"
    figures_dir = code_dir / "figures"
    data_dir = project_root / "data"

    loaded_code = sys.modules.get("code")
    if loaded_code is not None and not hasattr(loaded_code, "__path__"):
        del sys.modules["code"]

    os.chdir(notebook_dir)
    for path in (project_root, code_dir, chapters_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    if apply_style:
        configure_book_style()

    return {
        "PROJECT_ROOT": project_root,
        "NOTEBOOK_DIR": notebook_dir,
        "CODE_DIR": code_dir,
        "CHAPTERS_DIR": chapters_dir,
        "FIGURES_DIR": figures_dir,
        "DATA_DIR": data_dir,
    }
