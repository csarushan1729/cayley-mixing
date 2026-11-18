"""Computational additive combinatorics on Z/N (the setting of Freiman's theorem).

Normalisation (as in Tao-Vu and Zhao's MIT 18.225 notes):  1_A^(r) = (1/N) sum_{a in A} e^{-2 pi i a r / N},
so 1_A^(0) = alpha = |A|/N.

Main objects
------------
* sumsets and the doubling constant K = |A + A| / |A|
* additive energy E(A) = #{(a, b, c, d) in A^4 : a + b = c + d}
* the large spectrum  R = {r != 0 : |1_A^(r)| > alpha^{3/2}}
* Bohr sets           Bohr(R, eps) = {x : || r x / N ||_{R/Z} <= eps  for all r in R}
* Bogolyubov's lemma  2A - 2A contains Bohr(R, 1/4) and |R| < 1/alpha^2

The bridge to random walks
--------------------------
For a symmetric set S = -S in Z/N, let X_t be the walk with uniform steps
from S.  Then

    P(X_4 = 0) = E(S) / |S|^4                                         (1)
    chi2(t)    = N P(X_{2t} = 0) - 1                                    (2)

so  chi2(2) = N E(S) / |S|^4 - 1  >=  N / (K |S|) - 1,                 (3)

using E(S) >= |S|^3 / K (Cauchy-Schwarz).  In words: additive structure
(small doubling, large energy) forces the walk to keep returning home, i.e.
to mix slowly.  ``experiments/e08_additive_bridge.py`` checks (1)-(3) and
measures how far the effect persists to the full mixing time.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import fftconvolve


# --------------------------------------------------------------------------
# basic objects
# --------------------------------------------------------------------------
def indicator(A, N: int) -> np.ndarray:
    v = np.zeros(N)
    v[np.asarray(list(A), dtype=np.int64) % N] = 1.0
    return v


def fourier_coefficients(A, N: int) -> np.ndarray:
    """1_A^(r) for r = 0..N-1 with the 1/N normalisation."""
    return np.fft.fft(indicator(A, N)) / N


def _support(v: np.ndarray) -> np.ndarray:
    return np.nonzero(v > 0.5)[0]


def sumset_mod(A, B, N: int) -> np.ndarray:
    conv = np.fft.ifft(np.fft.fft(indicator(A, N)) * np.fft.fft(indicator(B, N))).real
    return _support(conv)


def iterated_sumset_mod(A, N: int, plus: int, minus: int) -> np.ndarray:
    """Support of plus*A - minus*A in Z/N."""
    F = np.fft.fft(indicator(A, N))
    G = F**plus * np.conj(F) ** minus
    return _support(np.fft.ifft(G).real)


def sumset_Z(A, B) -> np.ndarray:
    A = np.asarray(sorted(set(A)), dtype=np.int64)
    B = np.asarray(sorted(set(B)), dtype=np.int64)
    a0, b0 = A.min(), B.min()
    ia = np.zeros(A.max() - a0 + 1)
    ib = np.zeros(B.max() - b0 + 1)
    ia[A - a0] = 1
    ib[B - b0] = 1
    conv = fftconvolve(ia, ib)
    return _support(conv) + a0 + b0


def doubling_Z(A) -> float:
    A = set(A)
    return len(sumset_Z(A, A)) / len(A)


def doubling_mod(A, N: int) -> float:
    A = set(int(a) % N for a in A)
    return len(sumset_mod(A, A, N)) / len(A)


def additive_energy_mod(A, N: int) -> int:
    """E(A) = sum_x r_{A+A}(x)^2, computed exactly by FFT + rounding."""
    F = np.fft.fft(indicator(A, N))
    r = np.rint(np.fft.ifft(F * F).real)
    return int(np.sum(r**2))


# --------------------------------------------------------------------------
# Bohr sets and Bogolyubov
# --------------------------------------------------------------------------
def large_spectrum(A, N: int, threshold: float | None = None) -> np.ndarray:
    f = np.abs(fourier_coefficients(A, N))
    alpha = f[0]
    thr = alpha**1.5 if threshold is None else threshold
    r = np.nonzero(f > thr)[0]
    return r[r != 0]


def bohr_set(R, N: int, eps: float) -> np.ndarray:
    x = np.arange(N, dtype=np.int64)
    ok = np.ones(N, dtype=bool)
    for r in np.asarray(R, dtype=np.int64):
        frac = (r * x % N) / N
        dist = np.minimum(frac, 1 - frac)
        ok &= dist <= eps + 1e-12
    return np.nonzero(ok)[0]


def bogolyubov_check(A, N: int) -> dict:
    """Verify Bogolyubov's lemma (Zhao, MIT 18.225, Lecture 23 form) on one set."""
    A = sorted(set(int(a) % N for a in A))
    alpha = len(A) / N
    R = large_spectrum(A, N)
    bohr = bohr_set(R, N, 0.25)
    S = iterated_sumset_mod(A, N, 2, 2)
    s_set = set(S.tolist())
    contained = all(int(x) in s_set for x in bohr)
    return {
        "N": N,
        "size": len(A),
        "alpha": alpha,
        "spectrum_size": int(len(R)),
        "spectrum_bound": 1 / alpha**2,
        "bound_holds": len(R) < 1 / alpha**2,
        "bohr_size": int(len(bohr)),
        "two_A_minus_two_A": int(len(S)),
        "contained": bool(contained),
    }


# --------------------------------------------------------------------------
# energy / mixing bridge
# --------------------------------------------------------------------------
def symmetrise(S, N: int) -> list[int]:
    s = {int(x) % N for x in S}
    s |= {(-x) % N for x in s}
    return sorted(s)


def return_probability_4(S, N: int) -> float:
    """P(X_4 = 0) for the walk with uniform steps from S (computed directly)."""
    mu = indicator(S, N) / len(set(int(x) % N for x in S))
    F = np.fft.fft(mu)
    p4 = np.fft.ifft(F**4).real
    return float(p4[0])


def energy_bridge(S, N: int) -> dict:
    """Numbers appearing in identities (1)-(3) of the module docstring."""
    S = symmetrise(S, N)
    k = len(S)
    E = additive_energy_mod(S, N)
    K = doubling_mod(S, N)
    p4 = return_probability_4(S, N)
    return {
        "size": k,
        "doubling": K,
        "energy": E,
        "p4_direct": p4,
        "p4_from_energy": E / k**4,
        "chi2_2": N * p4 - 1,
        "chi2_2_lower_bound": N / (K * k) - 1,
    }


# --------------------------------------------------------------------------
# families of sets with controlled additive structure
# --------------------------------------------------------------------------
def symmetric_gap(N: int, steps, L: int) -> list[int]:
    """{l_1 x_1 + ... + l_d x_d : |l_i| <= L} \\ {0}, reduced mod N."""
    pts = {0}
    for x in steps:
        pts = {(p + l * x) % N for p in pts for l in range(-L, L + 1)}
    pts.discard(0)
    return sorted(pts)


def random_symmetric(N: int, k: int, rng: np.random.Generator) -> list[int]:
    """Uniformly random symmetric set of size k (k even, N odd)."""
    half = rng.choice(np.arange(1, (N - 1) // 2 + 1), size=k // 2, replace=False)
    return symmetrise(half, N)


def sparse_ap_subset(N: int, k: int, spread: int, rng: np.random.Generator) -> list[int]:
    """Random symmetric k-subset of the interval [-spread*k/2, spread*k/2].

    spread = 1 is an AP (doubling ~ 2); larger spread interpolates towards a
    generic set (doubling ~ k / 2).
    """
    width = max(k // 2, spread * k // 2)
    half = rng.choice(np.arange(1, width + 1), size=k // 2, replace=False)
    return symmetrise(half, N)


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for q in range(2, int(math.isqrt(n)) + 1):
        if n % q == 0:
            return False
    return True
