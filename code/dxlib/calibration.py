"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 31 - Market-Based Valuation.

Simple Monte Carlo based calibration routines (no SciPy dependency).

(c) Dr. Yves J. Hilpisch
AI-supported by various LLMs
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    package_dir = Path(__file__).resolve().parent
    sys.path = [
        path
        for path in sys.path
        if Path(path or ".").resolve() != package_dir
    ]
    sys.path.insert(0, str(package_dir.parent))
    __package__ = "dxlib"

import numpy as np
import pandas as pd

from .black_scholes import implied_vol_forward
from .heston import HestonParams, mc_call_prices, simulate_heston_spot_paths
from .processes import JumpDiffusion, build_time_grid

__all__ = [
    "CalibrationBounds",
    "CalibrationDiagnostics",
    "JumpDiffusionBounds",
    "JumpDiffusionParams",
    "evaluate_jump_diffusion_fit_table",
    "calibrate_jump_diffusion_single_expiry",
    "evaluate_heston_fit_table",
    "calibrate_heston_global",
    "calibrate_heston_local_v0",
    "calibrate_heston_local_theta_v0_rho",
    "calibrate_heston_local_v0_rho",
    "spot_from_forwards",
]


@dataclass(frozen=True, slots=True)
class CalibrationBounds:
    kappa: tuple[float, float] = (0.5, 8.0)
    theta: tuple[float, float] = (0.005, 0.20)
    vol_of_vol: tuple[float, float] = (0.05, 1.25)
    rho: tuple[float, float] = (-0.95, -0.05)
    v0: tuple[float, float] = (0.005, 0.20)


@dataclass(frozen=True, slots=True)
class CalibrationDiagnostics:
    seed: int
    n_candidates: int
    n_refine: int
    evaluations: int
    failures: int
    best_loss: float
    best_stage: str
    loss_path: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class JumpDiffusionParams:
    volatility: float
    jump_intensity: float
    jump_mean: float
    jump_std: float

    def __post_init__(self) -> None:
        if self.volatility < 0:
            raise ValueError("volatility must be non-negative")
        if self.jump_intensity < 0:
            raise ValueError("jump_intensity must be non-negative")
        if self.jump_std < 0:
            raise ValueError("jump_std must be non-negative")


@dataclass(frozen=True, slots=True)
class JumpDiffusionBounds:
    volatility: tuple[float, float] = (0.05, 0.60)
    jump_intensity: tuple[float, float] = (0.0, 8.0)
    # Widened from (-0.25, 0.05).
    jump_mean: tuple[float, float] = (-0.40, 0.10)
    jump_std: tuple[float, float] = (0.01, 0.80)  # widened from (0.05, 0.60)


MIN_CALIBRATION_STEPS = 24


def _require_surface_columns(surface_df: pd.DataFrame) -> None:
    required = {"EXPIR_DATE", "TTM", "FWD", "DF", "STRIKE", "IMPL_VOL", "MNY"}
    missing = required - set(surface_df.columns)
    if missing:
        msg = f"surface_df missing required columns: {sorted(missing)}"
        raise ValueError(msg)


def spot_from_forwards(surface_df: pd.DataFrame) -> float:
    """
    Estimate a single spot from per-expiry forward levels under q=0.
    """

    _require_surface_columns(surface_df)
    spot = surface_df["FWD"].to_numpy(dtype=float) * surface_df["DF"].to_numpy(
        dtype=float
    )
    return float(np.median(spot))


def _calc_model_fit_table(
    surface_df: pd.DataFrame,
    *,
    rate: float,
    params: HestonParams | JumpDiffusionParams,
    paths: int,
    steps_per_year: int,
    seed: int,
) -> pd.DataFrame:
    """
    Unified evaluation logic for Heston and Jump Diffusion models.
    """

    _require_surface_columns(surface_df)
    tables: list[pd.DataFrame] = []
    implied: list[float] = []

    for _, sub in surface_df.groupby("EXPIR_DATE", sort=True):
        ttm = float(sub["TTM"].iloc[0])
        spot0 = float(sub["FWD"].iloc[0]) * float(sub["DF"].iloc[0])
        steps = _steps_for_ttm(ttm, steps_per_year)
        iter_seed = seed + int(round(ttm * 10_000))

        if isinstance(params, HestonParams):
            spot_paths, _ = simulate_heston_spot_paths(
                spot=spot0,
                rate=rate,
                maturity=ttm,
                steps=steps,
                paths=paths,
                params=params,
                seed=iter_seed,
                antithetic=True,
                moment_matching=False,
            )
        else:
            grid = build_time_grid(maturity=ttm, steps=steps)
            model = JumpDiffusion(
                drift=rate,
                volatility=float(params.volatility),
                jump_intensity=float(params.jump_intensity),
                jump_mean=float(params.jump_mean),
                jump_std=float(params.jump_std),
                seed=iter_seed,
                antithetic=True,
                moment_matching=False,
            )
            spot_paths = model.simulate_paths(
                spot=spot0,
                time_grid=grid,
                paths=int(paths),
            )

        work = sub.sort_values("STRIKE").copy()
        strikes = work["STRIKE"].to_numpy(dtype=float)
        fwd = float(sub["FWD"].iloc[0])
        df = float(sub["DF"].iloc[0])

        prices = mc_call_prices(
            spot_paths,
            strikes,
            discount_factor=df,
            forward=fwd,
        )

        work["MODEL_PRICE_RAW"] = prices
        intrinsic = df * np.maximum(fwd - strikes, 0.0)
        work["INTRINSIC"] = intrinsic
        work["FLOORED_TO_INTRINSIC"] = prices < intrinsic
        work["MODEL_PRICE"] = np.maximum(prices, intrinsic)
        work["TV"] = work["MODEL_PRICE"] - work["INTRINSIC"]

        for _, row in work.iterrows():
            iv = implied_vol_forward(
                price=float(row["MODEL_PRICE"]),
                forward=float(row["FWD"]),
                strike=float(row["STRIKE"]),
                maturity=float(row["TTM"]),
                option_type="call"
                if str(row.get("CALIBRATE_TO", "CALL")).upper() == "CALL"
                else "put",
                discount_factor=float(row["DF"]),
            )
            implied.append(iv)
        tables.append(work)

    out = pd.concat(tables, axis=0, ignore_index=True)
    out["MODEL_IV"] = np.array(implied, dtype=float)
    out["IV_ERR"] = out["MODEL_IV"] - out["IMPL_VOL"]
    out["IV_ERR2"] = out["IV_ERR"] ** 2
    return out


def _clip(x: float, low: float, high: float) -> float:
    return float(min(max(float(x), float(low)), float(high)))


def _steps_for_ttm(ttm: float, steps_per_year: int) -> int:
    steps = int(round(float(steps_per_year) * float(ttm)))
    return max(steps, MIN_CALIBRATION_STEPS)


def _with_diagnostics(
    table: pd.DataFrame,
    diagnostics: CalibrationDiagnostics | dict[object, CalibrationDiagnostics],
) -> pd.DataFrame:
    out = table.copy()
    out.attrs["diagnostics"] = diagnostics
    return out


def evaluate_heston_fit_table(
    surface_df: pd.DataFrame,
    *,
    rate: float,
    params: HestonParams,
    paths: int,
    steps_per_year: int,
    seed: int,
) -> pd.DataFrame:
    """
    Evaluate model prices and implied vols for a forward-based small surface.
    """

    return _calc_model_fit_table(
        surface_df,
        rate=rate,
        params=params,
        paths=paths,
        steps_per_year=steps_per_year,
        seed=seed,
    )


def evaluate_jump_diffusion_fit_table(
    surface_df: pd.DataFrame,
    *,
    rate: float,
    params: JumpDiffusionParams,
    paths: int,
    steps_per_year: int,
    seed: int,
) -> pd.DataFrame:
    """
    Evaluate model prices and implied vols under a jump diffusion model.
    """

    return _calc_model_fit_table(
        surface_df,
        rate=rate,
        params=params,
        paths=paths,
        steps_per_year=steps_per_year,
        seed=seed,
    )


def calibrate_jump_diffusion_single_expiry(
    surface_df: pd.DataFrame,
    *,
    rate: float,
    bounds: JumpDiffusionBounds | None = None,
    paths: int = 25_000,
    steps_per_year: int = 240,
    n_candidates: int = 80,
    n_refine: int = 40,
    seed: int = 29,
) -> tuple[JumpDiffusionParams, pd.DataFrame]:
    """
    Calibrate a jump diffusion model to one expiry surface slice.
    """

    _require_surface_columns(surface_df)
    expiries = surface_df["EXPIR_DATE"].drop_duplicates().to_list()
    if len(expiries) != 1:
        raise ValueError("surface_df must contain exactly one expiry")

    if bounds is None:
        bounds = JumpDiffusionBounds()

    rng = np.random.default_rng(seed)

    def sample(low: float, high: float, n: int) -> np.ndarray:
        return low + (high - low) * rng.random(n)

    candidates = pd.DataFrame(
        {
            "volatility": sample(*bounds.volatility, n_candidates),
            "jump_intensity": sample(*bounds.jump_intensity, n_candidates),
            "jump_mean": sample(*bounds.jump_mean, n_candidates),
            "jump_std": sample(*bounds.jump_std, n_candidates),
        }
    )

    best_params: JumpDiffusionParams | None = None
    best_loss = float("inf")
    best_table: pd.DataFrame | None = None
    evaluations = 0
    failures = 0
    best_stage = "global"
    loss_path: list[float] = []

    for _, row in candidates.iterrows():
        params = JumpDiffusionParams(
            volatility=float(row["volatility"]),
            jump_intensity=float(row["jump_intensity"]),
            jump_mean=float(row["jump_mean"]),
            jump_std=float(row["jump_std"]),
        )
        try:
            table = _calc_model_fit_table(
                surface_df,
                rate=rate,
                params=params,
                paths=paths,
                steps_per_year=steps_per_year,
                seed=seed,
            )
            loss = float(table["IV_ERR2"].mean())
        except Exception:
            failures += 1
            continue
        evaluations += 1
        loss_path.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_params = params
            best_table = table

    if best_params is None or best_table is None:
        raise RuntimeError(
            "JD calibration failed: no candidate produced a result"
        )

    for step in range(int(n_refine)):
        scale = 0.35 * (0.98**step)
        vol = best_params.volatility * float(
            rng.lognormal(mean=0.0, sigma=scale)
        )
        lam = best_params.jump_intensity * float(
            rng.lognormal(mean=0.0, sigma=scale)
        )
        m_jump = best_params.jump_mean + float(rng.normal(0.0, 0.08 * scale))
        s_jump = best_params.jump_std * float(
            rng.lognormal(mean=0.0, sigma=scale)
        )
        params = JumpDiffusionParams(
            volatility=_clip(vol, *bounds.volatility),
            jump_intensity=_clip(lam, *bounds.jump_intensity),
            jump_mean=_clip(m_jump, *bounds.jump_mean),
            jump_std=_clip(s_jump, *bounds.jump_std),
        )
        try:
            table = _calc_model_fit_table(
                surface_df,
                rate=rate,
                params=params,
                paths=paths,
                steps_per_year=steps_per_year,
                seed=seed,
            )
            loss = float(table["IV_ERR2"].mean())
        except Exception:
            failures += 1
            continue
        evaluations += 1
        loss_path.append(loss)
        if loss < best_loss:
            best_loss = loss
            best_params = params
            best_table = table
            best_stage = "refine"

    diagnostics = CalibrationDiagnostics(
        seed=int(seed),
        n_candidates=int(n_candidates),
        n_refine=int(n_refine),
        evaluations=evaluations,
        failures=failures,
        best_loss=best_loss,
        best_stage=best_stage,
        loss_path=tuple(loss_path),
    )
    return best_params, _with_diagnostics(best_table, diagnostics)


def calibrate_heston_global(
    surface_df: pd.DataFrame,
    *,
    rate: float,
    bounds: CalibrationBounds | None = None,
    paths: int = 25_000,
    steps_per_year: int = 200,
    n_candidates: int = 80,
    n_refine: int = 40,
    seed: int = 17,
) -> tuple[HestonParams, pd.DataFrame]:
    """
    Global calibration to the full 3x5 surface via random search.
    """

    if bounds is None:
        bounds = CalibrationBounds()

    rng = np.random.default_rng(seed)

    def sample(low: float, high: float, n: int) -> np.ndarray:
        return low + (high - low) * rng.random(n)

    candidates = pd.DataFrame(
        {
            "kappa": sample(*bounds.kappa, n_candidates),
            "theta": sample(*bounds.theta, n_candidates),
            "vol_of_vol": sample(*bounds.vol_of_vol, n_candidates),
            "rho": sample(*bounds.rho, n_candidates),
            "v0": sample(*bounds.v0, n_candidates),
        }
    )

    best_params: HestonParams | None = None
    best_loss = float("inf")
    best_table: pd.DataFrame | None = None
    evaluations = 0
    failures = 0
    best_stage = "global"
    loss_path: list[float] = []

    for _, row in candidates.iterrows():
        params = HestonParams(
            kappa=float(row["kappa"]),
            theta=float(row["theta"]),
            vol_of_vol=float(row["vol_of_vol"]),
            rho=float(row["rho"]),
            v0=float(row["v0"]),
        )
        try:
            table = _calc_model_fit_table(
                surface_df,
                rate=rate,
                params=params,
                paths=paths,
                steps_per_year=steps_per_year,
                seed=seed,
            )
            loss = float(table["IV_ERR2"].mean())
        except Exception:
            failures += 1
            continue
        evaluations += 1
        loss_path.append(loss)

        if loss < best_loss:
            best_loss = loss
            best_params = params
            best_table = table

    if best_params is None or best_table is None:
        msg = "Calibration failed: no candidate produced a result"
        raise RuntimeError(msg)

    for step in range(int(n_refine)):
        scale = 0.35 * (0.98**step)
        kappa = best_params.kappa * float(rng.lognormal(mean=0.0, sigma=scale))
        theta = best_params.theta * float(rng.lognormal(mean=0.0, sigma=scale))
        vol_of_vol = best_params.vol_of_vol * float(
            rng.lognormal(mean=0.0, sigma=scale)
        )
        v0 = best_params.v0 * float(rng.lognormal(mean=0.0, sigma=scale))
        rho = best_params.rho + float(rng.normal(0.0, 0.15 * scale))

        params = HestonParams(
            kappa=_clip(kappa, *bounds.kappa),
            theta=_clip(theta, *bounds.theta),
            vol_of_vol=_clip(vol_of_vol, *bounds.vol_of_vol),
            rho=_clip(rho, *bounds.rho),
            v0=_clip(v0, *bounds.v0),
        )
        try:
            table = _calc_model_fit_table(
                surface_df,
                rate=rate,
                params=params,
                paths=paths,
                steps_per_year=steps_per_year,
                seed=seed,
            )
            loss = float(table["IV_ERR2"].mean())
        except Exception:
            failures += 1
            continue
        evaluations += 1
        loss_path.append(loss)

        if loss < best_loss:
            best_loss = loss
            best_params = params
            best_table = table
            best_stage = "refine"

    diagnostics = CalibrationDiagnostics(
        seed=int(seed),
        n_candidates=int(n_candidates),
        n_refine=int(n_refine),
        evaluations=evaluations,
        failures=failures,
        best_loss=best_loss,
        best_stage=best_stage,
        loss_path=tuple(loss_path),
    )
    return best_params, _with_diagnostics(best_table, diagnostics)


def calibrate_heston_local_v0(
    surface_df: pd.DataFrame,
    *,
    global_params: HestonParams,
    rate: float,
    bounds: CalibrationBounds | None = None,
    paths: int = 25_000,
    steps_per_year: int = 200,
    seed: int = 19,
    v0_grid: np.ndarray | None = None,
    mode: str = "atm",
) -> tuple[dict[object, float], pd.DataFrame]:
    """
    Re-fit v0 per expiry while keeping other parameters fixed.
    """

    mode = str(mode).strip().lower()
    if mode not in {"atm", "smile"}:
        raise ValueError("mode must be 'atm' or 'smile'")

    if bounds is None:
        bounds = CalibrationBounds()

    if v0_grid is None:
        base = float(global_params.v0)
        v0_grid = np.array(
            [0.25 * base, 0.5 * base, base, 2.0 * base, 4.0 * base]
        )
        v0_grid = np.clip(v0_grid, *bounds.v0)

    tables: list[pd.DataFrame] = []
    v0_map: dict[object, float] = {}
    diagnostics_map: dict[object, CalibrationDiagnostics] = {}

    for expiry, sub in surface_df.groupby("EXPIR_DATE", sort=True):
        best_v0 = float(v0_grid[0])
        best_loss = float("inf")
        best_table: pd.DataFrame | None = None
        evaluations = 0
        failures = 0
        loss_path: list[float] = []

        target = sub.copy()
        if mode == "atm":
            target = target.assign(abs_mny=(target["MNY"] - 1.0).abs())
            target = target.sort_values("abs_mny").head(1)

        for v0 in v0_grid:
            params = HestonParams(
                kappa=global_params.kappa,
                theta=global_params.theta,
                vol_of_vol=global_params.vol_of_vol,
                rho=global_params.rho,
                v0=float(v0),
            )
            try:
                table = _calc_model_fit_table(
                    target,
                    rate=rate,
                    params=params,
                    paths=paths,
                    steps_per_year=steps_per_year,
                    seed=seed,
                )
                loss = float(table["IV_ERR2"].mean())
            except Exception:
                failures += 1
                continue
            evaluations += 1
            loss_path.append(loss)

            if loss < best_loss:
                best_loss = loss
                best_v0 = float(v0)
                best_table = table

        if best_table is None:
            raise RuntimeError(f"Local calibration failed for expiry {expiry}")

        v0_map[expiry] = best_v0
        full_params = HestonParams(
            kappa=global_params.kappa,
            theta=global_params.theta,
            vol_of_vol=global_params.vol_of_vol,
            rho=global_params.rho,
            v0=best_v0,
        )
        full_table = _calc_model_fit_table(
            sub,
            rate=rate,
            params=full_params,
            paths=paths,
            steps_per_year=steps_per_year,
            seed=seed,
        )
        tables.append(full_table)
        diagnostics_map[expiry] = CalibrationDiagnostics(
            seed=int(seed),
            n_candidates=int(len(v0_grid)),
            n_refine=0,
            evaluations=evaluations,
            failures=failures,
            best_loss=best_loss,
            best_stage=mode,
            loss_path=tuple(loss_path),
        )

    out = pd.concat(tables, axis=0, ignore_index=True)
    return v0_map, _with_diagnostics(out, diagnostics_map)


def calibrate_heston_local_v0_rho(
    surface_df: pd.DataFrame,
    *,
    global_params: HestonParams,
    rate: float,
    bounds: CalibrationBounds | None = None,
    paths: int = 25_000,
    steps_per_year: int = 200,
    seed: int = 23,
    v0_grid: np.ndarray | None = None,
    rho_grid: np.ndarray | None = None,
    n_candidates: int = 40,
    n_refine: int = 25,
) -> tuple[dict[object, tuple[float, float]], pd.DataFrame]:
    """
    Local refinement: re-fit v0 and rho per expiry (smile fit).
    """

    if bounds is None:
        bounds = CalibrationBounds()

    param_map: dict[object, tuple[float, float]] = {}
    tables: list[pd.DataFrame] = []
    diagnostics_map: dict[object, CalibrationDiagnostics] = {}

    rng = np.random.default_rng(seed)

    for expiry, sub in surface_df.groupby("EXPIR_DATE", sort=True):
        best_v0 = float(global_params.v0)
        best_rho = float(global_params.rho)
        best_loss = float("inf")
        evaluations = 0
        failures = 0
        if v0_grid is not None and rho_grid is not None:
            best_stage = "grid"
        else:
            best_stage = "global"
        loss_path: list[float] = []

        def loss_for(v0: float, rho: float) -> float:
            params = HestonParams(
                kappa=global_params.kappa,
                theta=global_params.theta,
                vol_of_vol=global_params.vol_of_vol,
                rho=float(rho),
                v0=float(v0),
            )
            table = _calc_model_fit_table(
                sub,
                rate=rate,
                params=params,
                paths=paths,
                steps_per_year=steps_per_year,
                seed=seed,
            )
            return float(table["IV_ERR2"].mean())

        if v0_grid is not None and rho_grid is not None:
            for v0 in v0_grid:
                for rho in rho_grid:
                    try:
                        loss = loss_for(float(v0), float(rho))
                    except Exception:
                        failures += 1
                        continue
                    evaluations += 1
                    loss_path.append(loss)
                    if loss < best_loss:
                        best_loss = loss
                        best_v0 = float(v0)
                        best_rho = float(rho)
        else:
            base_v0 = float(global_params.v0)
            base_rho = float(global_params.rho)
            for _ in range(int(n_candidates)):
                v0 = base_v0 * float(rng.lognormal(mean=0.0, sigma=0.50))
                v0 = _clip(v0, *bounds.v0)
                rho = base_rho + float(rng.normal(0.0, 0.25))
                rho = _clip(rho, *bounds.rho)
                try:
                    loss = loss_for(v0, rho)
                except Exception:
                    failures += 1
                    continue
                evaluations += 1
                loss_path.append(loss)
                if loss < best_loss:
                    best_loss = loss
                    best_v0 = v0
                    best_rho = rho

            for step in range(int(n_refine)):
                scale = 0.35 * (0.98**step)
                v0 = best_v0 * float(rng.lognormal(mean=0.0, sigma=scale))
                v0 = _clip(v0, *bounds.v0)
                rho = best_rho + float(rng.normal(0.0, 0.15 * scale))
                rho = _clip(rho, *bounds.rho)
                try:
                    loss = loss_for(v0, rho)
                except Exception:
                    failures += 1
                    continue
                evaluations += 1
                loss_path.append(loss)
                if loss < best_loss:
                    best_loss = loss
                    best_v0 = v0
                    best_rho = rho
                    best_stage = "refine"

        if evaluations == 0:
            raise RuntimeError(f"Local calibration failed for expiry {expiry}")

        param_map[expiry] = (best_v0, best_rho)
        best_params = HestonParams(
            kappa=global_params.kappa,
            theta=global_params.theta,
            vol_of_vol=global_params.vol_of_vol,
            rho=best_rho,
            v0=best_v0,
        )
        table = _calc_model_fit_table(
            sub,
            rate=rate,
            params=best_params,
            paths=paths,
            steps_per_year=steps_per_year,
            seed=seed,
        )
        tables.append(table)
        diagnostics_map[expiry] = CalibrationDiagnostics(
            seed=int(seed),
            n_candidates=int(n_candidates)
            if v0_grid is None or rho_grid is None
            else int(len(v0_grid) * len(rho_grid)),
            n_refine=int(n_refine),
            evaluations=evaluations,
            failures=failures,
            best_loss=best_loss,
            best_stage=best_stage,
            loss_path=tuple(loss_path),
        )

    out = pd.concat(tables, axis=0, ignore_index=True)
    return param_map, _with_diagnostics(out, diagnostics_map)


def calibrate_heston_local_theta_v0_rho(
    surface_df: pd.DataFrame,
    *,
    global_params: HestonParams,
    rate: float,
    bounds: CalibrationBounds | None = None,
    paths: int = 25_000,
    steps_per_year: int = 200,
    seed: int = 23,
    n_candidates: int = 40,
    n_refine: int = 25,
) -> tuple[dict[object, tuple[float, float, float]], pd.DataFrame]:
    """
    Local refinement: re-fit theta, v0, and rho per expiry.
    """

    if bounds is None:
        bounds = CalibrationBounds()

    param_map: dict[object, tuple[float, float, float]] = {}
    tables: list[pd.DataFrame] = []
    diagnostics_map: dict[object, CalibrationDiagnostics] = {}

    rng = np.random.default_rng(seed)

    for expiry, sub in surface_df.groupby("EXPIR_DATE", sort=True):
        best_theta = float(global_params.theta)
        best_v0 = float(global_params.v0)
        best_rho = float(global_params.rho)
        best_loss = float("inf")
        evaluations = 0
        failures = 0
        best_stage = "global"
        loss_path: list[float] = []

        def loss_for(theta: float, v0: float, rho: float) -> float:
            params = HestonParams(
                kappa=global_params.kappa,
                theta=float(theta),
                vol_of_vol=global_params.vol_of_vol,
                rho=float(rho),
                v0=float(v0),
            )
            table = _calc_model_fit_table(
                sub,
                rate=rate,
                params=params,
                paths=paths,
                steps_per_year=steps_per_year,
                seed=seed,
            )
            return float(table["IV_ERR2"].mean())

        for _ in range(int(n_candidates)):
            theta = global_params.theta * float(
                rng.lognormal(mean=0.0, sigma=0.45)
            )
            theta = _clip(theta, *bounds.theta)
            v0 = global_params.v0 * float(rng.lognormal(mean=0.0, sigma=0.50))
            v0 = _clip(v0, *bounds.v0)
            rho = global_params.rho + float(rng.normal(0.0, 0.25))
            rho = _clip(rho, *bounds.rho)
            try:
                loss = loss_for(theta, v0, rho)
            except Exception:
                failures += 1
                continue
            evaluations += 1
            loss_path.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_theta = theta
                best_v0 = v0
                best_rho = rho

        for step in range(int(n_refine)):
            scale = 0.35 * (0.98**step)
            theta = best_theta * float(rng.lognormal(mean=0.0, sigma=scale))
            theta = _clip(theta, *bounds.theta)
            v0 = best_v0 * float(rng.lognormal(mean=0.0, sigma=scale))
            v0 = _clip(v0, *bounds.v0)
            rho = best_rho + float(rng.normal(0.0, 0.15 * scale))
            rho = _clip(rho, *bounds.rho)
            try:
                loss = loss_for(theta, v0, rho)
            except Exception:
                failures += 1
                continue
            evaluations += 1
            loss_path.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_theta = theta
                best_v0 = v0
                best_rho = rho
                best_stage = "refine"

        if evaluations == 0:
            raise RuntimeError(f"Local calibration failed for expiry {expiry}")

        param_map[expiry] = (best_theta, best_v0, best_rho)
        best_params = HestonParams(
            kappa=global_params.kappa,
            theta=best_theta,
            vol_of_vol=global_params.vol_of_vol,
            rho=best_rho,
            v0=best_v0,
        )
        table = _calc_model_fit_table(
            sub,
            rate=rate,
            params=best_params,
            paths=paths,
            steps_per_year=steps_per_year,
            seed=seed,
        )
        tables.append(table)
        diagnostics_map[expiry] = CalibrationDiagnostics(
            seed=int(seed),
            n_candidates=int(n_candidates),
            n_refine=int(n_refine),
            evaluations=evaluations,
            failures=failures,
            best_loss=best_loss,
            best_stage=best_stage,
            loss_path=tuple(loss_path),
        )

    out = pd.concat(tables, axis=0, ignore_index=True)
    return param_map, _with_diagnostics(out, diagnostics_map)
