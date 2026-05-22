"""Python for Finance, 3rd ed., O'Reilly (2026).
Chapter 11 - Performance Python.

This companion module keeps a few benchmarkable kernels in plain Python
form and, when available, exposes optional accelerated variants based on
multiprocessing, Numba, and vectorisation.

(c) Dr. Yves J. Hilpisch
AI-supported by GPT 5.x
The Python Quants GmbH | https://tpq.io
https://hilpisch.com | https://linktr.ee/dyjh
"""

from __future__ import annotations

import math
import multiprocessing as mp
import sys
import timeit
from typing import Callable

import numpy as np
import numpy.typing as npt

try:
    import numba as nb
except ImportError:  # pragma: no cover - optional dependency
    nb = None


Array = npt.NDArray[np.float64]


def average_py(x: Array) -> float:
    """Compute an arithmetic mean with an explicit Python loop."""

    total = 0.0
    for value in x:
        total += float(value)
    return total / float(x.shape[0])


def average_np(x: Array) -> float:
    """Compute an arithmetic mean via NumPy."""

    return float(np.mean(x))


def average_chunk(x: Array) -> float:
    """Worker-friendly chunk average used in multiprocessing examples."""

    return average_np(x)


def average_mp(x: Array, workers: int | None = None) -> float:
    """Average an array by splitting it into chunks across processes."""

    if workers is None:
        workers = max(1, min(4, mp.cpu_count()))
    chunks = [chunk for chunk in np.array_split(x, workers) if chunk.size > 0]
    if sys.platform != 'win32':
        ctx = mp.get_context('fork')
    else:
        ctx = mp.get_context('spawn')
    with ctx.Pool(processes=len(chunks)) as pool:
        means = pool.map(average_chunk, chunks)
    weights = np.array([chunk.size for chunk in chunks], dtype=float)
    return float(np.average(np.array(means, dtype=float), weights=weights))


if nb is not None:

    @nb.njit(cache=True)
    def average_nb(x: Array) -> float:
        total = 0.0
        for i in range(x.shape[0]):
            total += x[i]
        return total / x.shape[0]


    @nb.njit(cache=True)
    def mc_euro_call_nb(
        s0: float,
        K: float,
        r: float,
        sigma: float,
        T: float,
        n_paths: int,
        seed: int,
    ) -> float:
        np.random.seed(seed)
        drift = (r - 0.5 * sigma * sigma) * T
        diff = sigma * math.sqrt(T)
        payoff_sum = 0.0
        for _ in range(n_paths):
            z = np.random.random()
            z = math.sqrt(-2.0 * math.log(max(z, 1e-12))) * math.cos(
                2.0 * math.pi * np.random.random()
            )
            s_T = s0 * math.exp(drift + diff * z)
            payoff_sum += max(s_T - K, 0.0)
        return math.exp(-r * T) * payoff_sum / n_paths


    @nb.njit(cache=True)
    def ewma_nb(x: Array, lam: float) -> Array:
        out = np.empty_like(x)
        var = x[0] * x[0]
        out[0] = math.sqrt(var)
        for t in range(1, x.shape[0]):
            var = lam * var + (1.0 - lam) * x[t] * x[t]
            out[t] = math.sqrt(var)
        return out

else:
    average_nb = None
    mc_euro_call_nb = None
    ewma_nb = None


def mc_euro_call_py(
    s0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    seed: int = 2027,
) -> float:
    """Estimate a Black-Scholes call price by Monte Carlo."""

    rng = np.random.default_rng(seed=seed)
    z = rng.standard_normal(n_paths)
    s_T = s0 * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z)
    return float(math.exp(-r * T) * np.maximum(s_T - K, 0.0).mean())


def ewma_py(x: Array, lam: float) -> Array:
    """Compute the EWMA volatility recursion in pure Python/NumPy."""

    out = np.empty_like(x)
    var = float(x[0] ** 2)
    out[0] = math.sqrt(var)
    for t in range(1, x.shape[0]):
        var = lam * var + (1.0 - lam) * float(x[t] ** 2)
        out[t] = math.sqrt(var)
    return out


def benchmark(
    label: str,
    func: Callable[[], object],
    repeat: int = 3,
    number: int = 1,
) -> float:
    """Return a simple wall-clock benchmark in seconds."""

    timer = timeit.Timer(func)
    result = min(timer.repeat(repeat=repeat, number=number)) / number
    print(f'{label}: {result:.6f}s')
    return result


def main() -> None:
    """Run a compact set of chapter-style performance checks."""

    rng = np.random.default_rng(seed=42)
    x = rng.standard_normal(500_000).astype(float)

    benchmark('average_py', lambda: average_py(x))
    benchmark('average_np', lambda: average_np(x))

    if average_nb is not None:
        average_nb(x)
        benchmark('average_nb', lambda: average_nb(x))

    benchmark('average_mp', lambda: average_mp(x, workers=2))

    mc_params = dict(
        s0=100.0,
        K=100.0,
        r=0.02,
        sigma=0.2,
        T=1.0,
        n_paths=50_000,
    )
    benchmark('mc_euro_call_py', lambda: mc_euro_call_py(**mc_params))
    if mc_euro_call_nb is not None:
        mc_euro_call_nb(seed=2027, **mc_params)
        benchmark(
            'mc_euro_call_nb',
            lambda: mc_euro_call_nb(seed=2027, **mc_params),
        )

    rets = rng.standard_normal(5_000).astype(float) * 0.01
    lam = 0.94
    benchmark('ewma_py', lambda: ewma_py(rets, lam))
    if ewma_nb is not None:
        ewma_nb(rets, lam)
        benchmark('ewma_nb', lambda: ewma_nb(rets, lam))


if __name__ == '__main__':
    main()
