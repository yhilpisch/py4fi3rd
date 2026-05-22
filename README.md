<img src="https://hilpisch.com/tpq_logo_bic.png" width="25%" align="right">
<br clear="all">

# Python for Finance, Third Edition · Companion Code and Notebooks

This repository contains the companion code, notebooks, data sets, and figure
assets for *Python for Finance, Third Edition*. It is designed to give readers
one place to run the chapter examples, explore the notebooks interactively, and
adapt the reusable packages and scripts for their own study and experiments.

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
- `assets/`
  - figure exports and other shared assets referenced by code and notebooks
- `tools/`
  - validation helpers for checking scripts and notebooks

## Usage

Set `PYTHONPATH` from the repository root when you want to import the local packages:

```bash
export PYTHONPATH="$PWD/code"
```

Run scripts and notebooks from the repository root so the relative paths to `data/` and `assets/` resolve correctly.

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
