"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 27 - Valuation Framework.

Minimal derivatives analytics library used throughout Part VI.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path

    package_dir = Path(__file__).resolve().parent
    sys.path = [
        path
        for path in sys.path
        if Path(path or ".").resolve() != package_dir
    ]
    sys.path.insert(0, str(package_dir.parent))
    __package__ = "dxlib"

from .curves import (
    ConstantShortRateCurve,
    DiscountCurve,
    InterpolatedZeroCurve,
)
from .black_scholes import (
    bs_price_forward,
    implied_vol_forward,
    norm_cdf,
    norm_pdf,
)
from .discounting import DiscountingModel, FlatDiscounting, discount_factors
from .env import MarketEnvironment
from .calibration import (
    CalibrationBounds,
    CalibrationDiagnostics,
    JumpDiffusionBounds,
    JumpDiffusionParams,
    calibrate_jump_diffusion_single_expiry,
    calibrate_heston_global,
    calibrate_heston_local_v0,
    calibrate_heston_local_theta_v0_rho,
    calibrate_heston_local_v0_rho,
    evaluate_jump_diffusion_fit_table,
    evaluate_heston_fit_table,
    spot_from_forwards,
)
from .heston import HestonParams, mc_call_prices, simulate_heston_spot_paths
from .lsm import AmericanPutLSM, lsm_american_put_from_paths
from .marketdata import (
    OptionsSnapshot,
    estimate_forward,
    load_spx_snapshot,
    parity_table,
    select_small_surface,
)
from .payoffs import AmericanPut, EuropeanCall, EuropeanPut, TerminalPayoff
from .portfolio import Portfolio, Position, delta_central, price_and_stderr
from .processes import (
    CIRShortRate,
    GeometricBrownianMotion,
    HestonModel,
    JumpDiffusion,
    PathSimulator,
    build_time_grid,
)
from .random import standard_normals
from .time import ensure_datetime_array, time_to_maturity, year_fractions
from .valuation import EuropeanMCPricer
from .volsurface import ImpliedVolSurface

__all__ = [
    "norm_cdf",
    "norm_pdf",
    "bs_price_forward",
    "implied_vol_forward",
    "DiscountCurve",
    "ConstantShortRateCurve",
    "InterpolatedZeroCurve",
    "DiscountingModel",
    "FlatDiscounting",
    "discount_factors",
    "MarketEnvironment",
    "HestonParams",
    "simulate_heston_spot_paths",
    "mc_call_prices",
    "CalibrationBounds",
    "CalibrationDiagnostics",
    "JumpDiffusionBounds",
    "JumpDiffusionParams",
    "calibrate_jump_diffusion_single_expiry",
    "calibrate_heston_global",
    "calibrate_heston_local_v0",
    "calibrate_heston_local_theta_v0_rho",
    "calibrate_heston_local_v0_rho",
    "evaluate_jump_diffusion_fit_table",
    "evaluate_heston_fit_table",
    "spot_from_forwards",
    "TerminalPayoff",
    "EuropeanCall",
    "EuropeanPut",
    "AmericanPut",
    "OptionsSnapshot",
    "load_spx_snapshot",
    "parity_table",
    "estimate_forward",
    "select_small_surface",
    "ImpliedVolSurface",
    "standard_normals",
    "PathSimulator",
    "build_time_grid",
    "GeometricBrownianMotion",
    "JumpDiffusion",
    "HestonModel",
    "CIRShortRate",
    "EuropeanMCPricer",
    "AmericanPutLSM",
    "lsm_american_put_from_paths",
    "Position",
    "Portfolio",
    "price_and_stderr",
    "delta_central",
    "ensure_datetime_array",
    "year_fractions",
    "time_to_maturity",
]
