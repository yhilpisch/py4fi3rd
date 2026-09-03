"""Build the Chapter 11 OpenMP Cython extension in place."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup


HERE = Path(__file__).resolve().parent
compile_args = ["-O3"]
link_args: list[str] = []
include_dirs: list[str] = []

if sys.platform == "darwin":
    prefix = os.environ.get("LIBOMP_PREFIX")
    if prefix is None:
        raise SystemExit(
            "Install an OpenMP runtime and set LIBOMP_PREFIX to its prefix "
            "(for Homebrew libomp: export LIBOMP_PREFIX=$(brew --prefix libomp))."
        )
    include_dirs.append(str(Path(prefix) / "include"))
    library_dir = Path(prefix) / "lib"
    compile_args.extend(["-Xpreprocessor", "-fopenmp"])
    link_args.extend(
        [f"-L{library_dir}", "-lomp", f"-Wl,-rpath,{library_dir}"]
    )
elif sys.platform == "win32":
    compile_args.extend(["/O2", "/openmp"])
else:
    compile_args.append("-fopenmp")
    link_args.append("-fopenmp")

extension = Extension(
    "ch11_mc_cython_optimized",
    [str(HERE / "ch11_mc_cython_optimized.pyx")],
    include_dirs=include_dirs,
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

setup(
    name="ch11-mc-cython-optimized",
    ext_modules=cythonize(
        [extension],
        compiler_directives={"language_level": 3},
    ),
)
