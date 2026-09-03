# cython: language_level=3
"""Streaming, parallel Cython Monte Carlo kernel used in Chapter 11."""

cimport cython
from cython.parallel cimport prange
from libc.math cimport cos, exp, log, sin, sqrt
from libc.stdint cimport uint64_t


cdef inline uint64_t splitmix64_next(uint64_t* state) noexcept nogil:
    """Seed one xoshiro256** state word."""
    cdef uint64_t value
    state[0] += <uint64_t>0x9E3779B97F4A7C15
    value = state[0]
    value = (value ^ (value >> 30)) * <uint64_t>0xBF58476D1CE4E5B9
    value = (value ^ (value >> 27)) * <uint64_t>0x94D049BB133111EB
    return value ^ (value >> 31)


cdef inline uint64_t rotate_left(uint64_t value, int shift) noexcept nogil:
    return (value << shift) | (value >> (64 - shift))


cdef inline uint64_t xoshiro256ss(
    uint64_t* s0,
    uint64_t* s1,
    uint64_t* s2,
    uint64_t* s3,
) noexcept nogil:
    """Return the next xoshiro256** unsigned integer."""
    cdef uint64_t result = rotate_left(s1[0] * <uint64_t>5, 7) * <uint64_t>9
    cdef uint64_t temporary = s1[0] << 17
    s2[0] ^= s0[0]
    s3[0] ^= s1[0]
    s1[0] ^= s2[0]
    s0[0] ^= s3[0]
    s2[0] ^= temporary
    s3[0] = rotate_left(s3[0], 45)
    return result


cdef inline double uniform_open(
    uint64_t* s0,
    uint64_t* s1,
    uint64_t* s2,
    uint64_t* s3,
) noexcept nogil:
    """Map 53 random bits to an open-interval uniform variate."""
    return (
        (xoshiro256ss(s0, s1, s2, s3) >> 11) + 0.5
    ) * (1.0 / 9007199254740992.0)


cdef inline double path_payoff(
    uint64_t seed,
    Py_ssize_t n_steps,
    double s0,
    double k,
    double total_drift,
    double vol,
) noexcept nogil:
    """Generate one path's shocks and return its terminal payoff."""
    cdef uint64_t seed_state = seed
    cdef uint64_t s0_rng = splitmix64_next(&seed_state)
    cdef uint64_t s1_rng = splitmix64_next(&seed_state)
    cdef uint64_t s2_rng = splitmix64_next(&seed_state)
    cdef uint64_t s3_rng = splitmix64_next(&seed_state)
    cdef Py_ssize_t step = 0
    cdef double u1, u2, radius, angle
    cdef double shock_sum = 0.0
    cdef double price

    # Box-Muller produces two independent standard normals per pair.
    while step + 1 < n_steps:
        u1 = uniform_open(&s0_rng, &s1_rng, &s2_rng, &s3_rng)
        u2 = uniform_open(&s0_rng, &s1_rng, &s2_rng, &s3_rng)
        radius = sqrt(-2.0 * log(u1))
        angle = 6.2831853071795864769 * u2
        shock_sum += radius * cos(angle) + radius * sin(angle)
        step += 2

    if step < n_steps:
        u1 = uniform_open(&s0_rng, &s1_rng, &s2_rng, &s3_rng)
        u2 = uniform_open(&s0_rng, &s1_rng, &s2_rng, &s3_rng)
        shock_sum += sqrt(-2.0 * log(u1)) * cos(
            6.2831853071795864769 * u2
        )

    price = s0 * exp(total_drift + vol * shock_sum)
    if price > k:
        return price - k
    return 0.0


@cython.cdivision(True)
def mc_euro_call_cy_optimized(
    double s0,
    double k,
    double r,
    double sigma,
    double t,
    Py_ssize_t n_paths,
    Py_ssize_t n_steps,
    unsigned long long seed=42,
):
    """Price a call with fused RNG/path evolution and OpenMP parallelism."""
    cdef Py_ssize_t path
    cdef double dt = t / n_steps
    cdef double total_drift = n_steps * (
        r - 0.5 * sigma * sigma
    ) * dt
    cdef double vol = sigma * sqrt(dt)
    cdef double payoff_sum = 0.0

    for path in prange(n_paths, nogil=True, schedule="static"):
        payoff_sum += path_payoff(
            <uint64_t>seed
            + <uint64_t>path * <uint64_t>0x9E3779B97F4A7C15,
            n_steps,
            s0,
            k,
            total_drift,
            vol,
        )

    return exp(-r * t) * payoff_sum / n_paths
