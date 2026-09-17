<img src="https://hilpisch.com/tpq_logo_bic.png" width="25%" align="right">
<br clear="all">

# Python for Finance, Third Edition · Companion Code and Notebooks

This repository contains the companion code, notebooks, and data sets for
*Python for Finance, Third Edition*. It is designed to give readers one place
to run the chapter examples, explore the notebooks interactively, and adapt the
reusable packages and scripts for their own study and experiments.

<img src="https://hilpisch.com/py4fi_3rd_cover_bw.png" width="30%" align="left">
<br clear="all">

## Layout

- `code/`
  - `assetlib/`, `dxlib/`, and `engine/` package-style modules
  - `chapters/` and `figures/` scripts used in the manuscript
  - `labs/` scripts for the companion lab material
- `notebooks/`
  - chapter notebooks
  - appendix notebooks
  - lab notebooks
- `data/`
  - CSV and JSON datasets used by the examples
- `tools/`
  - validation helpers for checking scripts and notebooks

## Usage

Set `PYTHONPATH` from the repository root when you want to import the local packages:

```bash
export PYTHONPATH="$PWD/code"
```

Run scripts and notebooks from the repository root so relative paths to `data/`
resolve correctly. Some figure scripts may write local output under `assets/`;
that directory is intentionally excluded from version control in this companion
repo.

## Validation

Use the validation helpers from the repository root:

```bash
python tools/validate_code.py
python tools/validate_code.py --skip-execute
python tools/validate_code.py chapters/ch08*.py
python tools/validate_notebooks.py
python tools/validate_notebooks.py --skip-execute
python tools/validate_notebooks.py ch15*.ipynb
```

The code validator checks syntax, imported modules, and optionally executes scripts with a safe Matplotlib backend.
The notebook validator checks JSON structure and optionally executes notebooks in place.

For a Colab-style check from a fresh runtime, run the repository directly from its
checkout; no project package installation is required:

```bash
python -u tools/validate_colab.py --scripts --all-notebooks --timeout 300
python -u tools/validate_code.py 'chapters/ch*.py' 'figures/*.py' 'labs/*.py' --timeout 300
```

The Colab validator runs the repository bootstrap, representative scripts, and all
non-checkpoint notebooks in fresh kernels. The code validator then executes all
runnable chapter, figure, and lab scripts; the `ch*.py` pattern excludes the Chapter
11 build helper. For a fast static/import check that includes setup helpers, use
`python -u tools/validate_code.py 'chapters/*.py' 'figures/*.py' 'labs/*.py' --skip-execute`.
Pass `--log-file /content/py4fi3rd-colab-validation.log` to
`validate_colab.py` to mirror its output and errors to a downloadable text file.
The broader script-validation output can be appended with `tee -a`; in Colab,
download the resulting file with:

```python
from google.colab import files

files.download("/content/py4fi3rd-colab-validation.log")
```

The optional TsTables workflow is not part of this base check and should be tested
separately after its external dependencies have been deliberately provisioned.

For the complete command-line run with live progress and one combined log, execute
from the companion-repository root:

```bash
bash execute_validation.sh
```

The default log is `colab-validation.log`. Each script and notebook reports its
elapsed time, as do the two phases and the complete run. Override
`VALIDATION_TIMEOUT`, `VALIDATION_LOG`, or `PYTHON_BIN` when needed.

## Disclaimer

This material is provided for educational and personal research use only.
It is not investment advice and does not constitute an offer to buy or sell any financial instrument.
Past performance, whether simulated or real, is not indicative of future results.
Trading financial instruments involves risk, including the risk of loss of capital.

You are solely responsible for any decisions you make based on this code or the accompanying notebooks and documentation.

## Contact

- The Python Quants GmbH: <https://tpq.io>
- Yves J. Hilpisch: <https://hilpisch.com>
- Links: <https://linktr.ee/dyjh>
