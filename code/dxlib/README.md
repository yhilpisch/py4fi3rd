# `dxlib` — Minimal Derivatives Analytics Library

`dxlib` is a compact derivatives analytics library used in *Python for Finance,
3rd Edition* (Part VI, Chapters 27–31). It is designed as a readable companion
package for the book examples and includes:

- deterministic discount curves and time helpers,
- a small market-environment container,
- random numbers with basic variance reduction,
- risk-factor simulation models (GBM, jump diffusion, Heston, CIR),
- Monte Carlo valuation helpers (European terminal payoffs and American put LSM),
- simple Black-76 forward-form pricing and implied-vol inversion,
- helpers to load the book’s options snapshot and build a small implied-vol
  surface,
- simple Monte Carlo-based calibration routines for Heston and jump diffusion
  (no SciPy dependency),
- a tiny portfolio container for aggregating pricers and computing deltas via
  bump-and-revalue.

The focus is clarity and reproducibility, not exhaustive product coverage.

## Quick Start

### 1) Make the package importable

This repo keeps the library under `code/dxlib/`. To use `import dxlib` from the
project root, add `code/` to `PYTHONPATH`.

On macOS and Linux (bash/zsh):

```bash
export PYTHONPATH="$PWD/code"
```

In PowerShell on Windows:

```powershell
$env:PYTHONPATH = "$PWD\\code"
```

### 2) Build a curve and a market environment

```python
import datetime as dt

from dxlib import ConstantShortRateCurve, MarketEnvironment

pricing_date = dt.date(2027, 1, 2)
disc = ConstantShortRateCurve(
    name="disc",
    reference_date=pricing_date,
    rate=0.03,
)

env = MarketEnvironment(name="base", pricing_date=pricing_date)
env.add_curve("discount", disc)
env.add_constant("spot", 100.0)
env.add_constant("volatility", 0.2)
```

### 3) Simulate a GBM risk factor on a time grid

```python
from dxlib import GeometricBrownianMotion, build_time_grid

grid = build_time_grid(maturity=1.0, steps=12)  # 13 points incl. 0
gbm = GeometricBrownianMotion(drift=0.05, volatility=0.2, seed=7)
paths = gbm.simulate_paths(spot=100.0, time_grid=grid, paths=10_000)
terminal = paths[:, -1]
print(float(terminal.mean()))
```

### 4) Price a European payoff with a Monte Carlo pricer

```python
from dxlib import (
    EuropeanCall,
    EuropeanMCPricer,
    FlatDiscounting,
    GeometricBrownianMotion,
)

process = GeometricBrownianMotion(drift=0.03, volatility=0.2, seed=7)
pricer = EuropeanMCPricer(
    process=process,
    payoff=EuropeanCall(strike=100.0),
    discounting=FlatDiscounting(rate=0.03),
    maturity=1.0,
    steps=52,
    paths=50_000,
)
price, stderr = pricer.value(spot=100.0)
print(price, stderr)
```

## Public API

The package root (`dxlib/__init__.py`) re-exports the primary entry points:

- Distributions and Black-76:
  - `norm_cdf()`, `norm_pdf()`
  - `bs_price_forward()`, `implied_vol_forward()`
- Curves (dated, ACT/day_count time):
  - `DiscountCurve`, `ConstantShortRateCurve`, `InterpolatedZeroCurve`
- Environment: `MarketEnvironment`
- Random numbers: `standard_normals()`
- Time utilities: `ensure_datetime_array()`, `year_fractions()`, `time_to_maturity()`
- Processes and grids:
  - `PathSimulator`
  - `build_time_grid()`
  - `GeometricBrownianMotion`, `JumpDiffusion`, `HestonModel`, `CIRShortRate`
- Discounting (year-fraction time):
  - `DiscountingModel`, `FlatDiscounting`, `discount_factors()`
- Payoffs:
  - `TerminalPayoff`, `EuropeanCall`, `EuropeanPut`, `AmericanPut`
- Monte Carlo valuation:
  - `EuropeanMCPricer`
  - `AmericanPutLSM`, `lsm_american_put_from_paths()`
- Portfolio:
  - `Position`, `Portfolio`
  - `price_and_stderr()`, `delta_central()`
- Market-based:
  - `OptionsSnapshot`, `load_spx_snapshot()`
  - `parity_table()`, `estimate_forward()`, `select_small_surface()`
  - `ImpliedVolSurface`
- Heston Monte Carlo helpers:
  - `HestonParams`, `simulate_heston_spot_paths()`, `mc_call_prices()`
- Calibration:
  - `CalibrationBounds`, `CalibrationDiagnostics`
  - `JumpDiffusionBounds`, `JumpDiffusionParams`
  - `calibrate_jump_diffusion_single_expiry()`,
    `evaluate_jump_diffusion_fit_table()`
  - `calibrate_heston_global()`, `calibrate_heston_local_v0()`,
    `calibrate_heston_local_v0_rho()`,
    `calibrate_heston_local_theta_v0_rho()`
  - `spot_from_forwards()`

## Package Layout

- `dxlib/curves.py`
  - `DiscountCurve` protocol: minimal `.discount_factor()` interface
  - `ConstantShortRateCurve`: constant continuously-compounded short rate
  - `InterpolatedZeroCurve`: linear interpolation on zero rates
- `dxlib/env.py`
  - `MarketEnvironment`: container for constants, lists, and curves
- `dxlib/time.py`
  - `ensure_datetime_array()`: normalizes `date`/`datetime` sequences
  - `year_fractions()`: ACT/day_count year fractions relative to origin
  - `time_to_maturity()`: scalar ACT/day_count time-to-maturity helper
- `dxlib/random.py`
  - `standard_normals()`: optional antithetic sampling and moment matching
- `dxlib/processes.py`
  - `PathSimulator`: protocol for single-factor path simulators
  - `build_time_grid()`: uniform grid in year fractions
  - `GeometricBrownianMotion`: exact log-normal step simulation
  - `JumpDiffusion`: Merton-style jump diffusion with log-normal jumps
  - `HestonModel`: stochastic volatility (Euler-style discretization)
  - `CIRShortRate`: CIR short-rate process
- `dxlib/discounting.py`
  - `DiscountingModel` protocol: `.discount_factor(ttm)` interface
  - `FlatDiscounting`: constant continuously-compounded short rate
  - `discount_factors()`: vectorized discount factors on a time grid
- `dxlib/payoffs.py`
  - `EuropeanCall`, `EuropeanPut`, `AmericanPut`: terminal/intrinsic payoffs
- `dxlib/valuation.py`
  - `EuropeanMCPricer`: terminal-payoff Monte Carlo valuation
- `dxlib/lsm.py`
  - `AmericanPutLSM`: Longstaff-Schwartz least-squares Monte Carlo valuation
  - `lsm_american_put_from_paths()`: LSM valuation from pre-simulated paths
- `dxlib/portfolio.py`
  - `Position`, `Portfolio`: aggregate multiple pricers
  - `price_and_stderr()`: normalize pricer return types
  - `delta_central()`: bump-and-revalue delta helper
- `dxlib/black_scholes.py`
  - `norm_cdf()`, `norm_pdf()`: standard normal helpers
  - `bs_price_forward()`, `implied_vol_forward()`: forward-form pricing/IV
- `dxlib/marketdata.py`
  - `load_spx_snapshot()`: load and clean the snapshot CSV with spreads
  - `parity_table()`: merge calls/puts by strike for one expiry
  - `estimate_forward()`: forward estimate from put-call parity
  - `select_small_surface()`: select a small quotes surface and compute IVs
- `dxlib/volsurface.py`
  - `ImpliedVolSurface`: interpolation on a small surface with explicit
    extrapolation policy
- `dxlib/heston.py`
  - `simulate_heston_spot_paths()`: MC path simulation wrapper
  - `mc_call_prices()`: vectorized call prices from simulated paths
- `dxlib/calibration.py`
  - self-contained calibration routines for Heston and jump diffusion
  - `CalibrationDiagnostics`: metadata stored in fit table `.attrs`

## Usage Patterns

### Black-76 pricing and implied volatility

```python
from dxlib import bs_price_forward, implied_vol_forward

forward = 100.0
strike = 105.0
ttm = 0.5
df = 0.99
vol = 0.2

price = bs_price_forward(
    forward,
    strike,
    ttm,
    vol,
    option_type="call",
    discount_factor=df,
)
iv = implied_vol_forward(
    price,
    forward,
    strike,
    ttm,
    option_type="call",
    discount_factor=df,
)
print(price, iv)
```

### Interpolated zero curve

```python
import datetime as dt

from dxlib import InterpolatedZeroCurve

ref = dt.date(2027, 1, 2)
curve = InterpolatedZeroCurve(
    name="zero",
    reference_date=ref,
    nodes=[
        (dt.date(2027, 7, 2), 0.02),
        (dt.date(2028, 1, 2), 0.025),
        (dt.date(2029, 1, 2), 0.03),
    ],
)

df = curve.discount_factor(dt.date(2028, 1, 2))
print(df)
```

### Variance-reduced standard normals

```python
from dxlib import standard_normals

z = standard_normals((252, 10_000), seed=7, antithetic=True, moment_matching=True)
```

### Build a small implied-vol surface from the book snapshot

Chapter 31 uses the local snapshot `data/spx_options_snapshot.csv`.

```python
from dxlib import ImpliedVolSurface, load_spx_snapshot, select_small_surface

snap = load_spx_snapshot("data/spx_options_snapshot.csv")
expiries = sorted(snap.raw["EXPIR_DATE"].unique())[:3]

rate = 0.03
surface_df = select_small_surface(
    snap,
    expiries=expiries,
    rate=rate,
    calibrate_to="CALL",
    moneyness_targets=[0.8, 0.9, 1.0, 1.05, 1.1],
)
surf = ImpliedVolSurface.from_frame(surface_df, extrapolate="flat")
print(surf.vol(maturity=float(surface_df["TTM"].min()), moneyness=1.0))
```

### Heston Monte Carlo calibration and local smile refinement

```python
from dxlib import calibrate_heston_global, calibrate_heston_local_theta_v0_rho

params, table = calibrate_heston_global(surface_df, rate=rate)
local_map, fit_local = calibrate_heston_local_theta_v0_rho(
    surface_df,
    global_params=params,
    rate=rate,
)
print(params)
print(float(table["IV_ERR2"].mean()))
print(table.attrs["diagnostics"])
print(local_map)
```

### Jump diffusion calibration for a short-horizon smile

```python
from dxlib import calibrate_jump_diffusion_single_expiry

short_expiry = [min(expiries)]
short_surface = select_small_surface(
    snap,
    expiries=short_expiry,
    rate=rate,
    calibrate_to="CALL",
    moneyness_targets=[0.8, 0.9, 1.0, 1.05, 1.1],
)
jd_params, jd_fit = calibrate_jump_diffusion_single_expiry(
    short_surface,
    rate=rate,
)
print(jd_params)
print(float(jd_fit["IV_ERR2"].mean()))
print(jd_fit.attrs["diagnostics"])
```

### American put via least-squares Monte Carlo (LSM)

```python
from dxlib import AmericanPut, AmericanPutLSM
from dxlib import FlatDiscounting, GeometricBrownianMotion

lsm = AmericanPutLSM(
    process=GeometricBrownianMotion(drift=0.03, volatility=0.2, seed=7),
    payoff=AmericanPut(strike=100.0),
    discounting=FlatDiscounting(rate=0.03),
    maturity=1.0,
    steps=50,
    paths=50_000,
    basis_degree=2,
    min_itm=200,
)
out = lsm.value(spot=100.0)
print(out["price"], out["stderr"])
```

### Portfolio aggregation and bump-and-revalue delta

```python
from dxlib import EuropeanCall, EuropeanMCPricer
from dxlib import FlatDiscounting, GeometricBrownianMotion
from dxlib import Portfolio, Position

process = GeometricBrownianMotion(drift=0.03, volatility=0.2, seed=7)
discounting = FlatDiscounting(rate=0.03)

call_100 = EuropeanMCPricer(
    process=process,
    payoff=EuropeanCall(strike=100.0),
    discounting=discounting,
    maturity=1.0,
    steps=52,
    paths=25_000,
)
call_110 = EuropeanMCPricer(
    process=process,
    payoff=EuropeanCall(strike=110.0),
    discounting=discounting,
    maturity=1.0,
    steps=52,
    paths=25_000,
)

portfolio = Portfolio(
    name="demo",
    positions=(
        Position(name="C100", quantity=1.0, pricer=call_100),
        Position(name="C110", quantity=-0.5, pricer=call_110),
    ),
)

report = portfolio.value(spot=100.0)
print(report["total_value"], report["stderr_method"])
print(portfolio.deltas(spot=100.0)["portfolio"])
```

## Notes and Limitations

- `dxlib` is a book companion library. It aims to be readable and stable for
  the examples in Part VI rather than to cover every real-world pricing need.
- Some models use simplified discretizations to keep the implementation short.
- Curve and surface extrapolation defaults to `"flat"` and can be switched to
  `"error"` when out-of-range queries should fail explicitly.
- Portfolio-level standard errors use an independent root-sum-of-squares
  approximation unless shared-scenario valuation is implemented separately.
- There are two discounting concepts:
  - `dxlib.curves` uses dated curves (`reference_date`, `target date`), and
  - `dxlib.discounting` uses year-fraction time (`ttm`) for Monte Carlo pricers.
