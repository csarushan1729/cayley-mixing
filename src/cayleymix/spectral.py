"""Spectral / Fourier analysis of random walks on finite abelian groups.

For a walk on an abelian group G driven by mu, every character chi_r is an
eigenfunction of the transition operator:

    (P chi_r)(x) = sum_s mu(s) chi_r(x + s) = mu_hat(r) chi_r(x),
    mu_hat(r)   = sum_s mu(s) chi_r(s).

So the whole spectrum is {mu_hat(r) : r in G} and it can be read off without
diagonalising anything.  Plancherel then gives an *identity* (not merely a
bound) for the chi-square distance after t steps from any start:

    chi2(t) = N * sum_x (p_t(x) - 1/N)^2 = sum_{r != 0} |mu_hat(r)|^{2t}.     (*)

Since TV <= sqrt(chi2)/2 by Cauchy-Schwarz, (*) yields Diaconis' "upper bound
lemma".  The classical beta-bound (Terras, *Fourier Analysis on Finite
Groups and Applications*, ch. 1),

    ||T^m p - u||_1 <= sqrt(N) beta^m,   beta = max_{r != 0} |mu_hat(r)|,

is the crudest version of the same argument: it replaces every |mu_hat(r)| by
the worst one.  ``experiments/e02_bound_tightness.py`` measures what that
costs.
"""
from __future__ import annotations

import math

import numpy as np

from . import distances
from .groups import AbelianGroup


def _measure_array(group: AbelianGroup, measure: dict, laziness: float) -> np.ndarray:
    mu = np.zeros(group.order)
    total = float(sum(measure.values()))
    for s, w in measure.items():
        mu[group.index(s)] += (1 - laziness) * w / total
    mu[group.identity()] += laziness
    return mu


def abelian_eigenvalues(group: AbelianGroup, measure: dict, laziness: float = 0.0) -> np.ndarray:
    """mu_hat(r) for every character r (flat C-order, r = 0 first).

    Computed by FFT: numpy's fftn uses exp(-2 pi i ...), so mu_hat(r) is the
    complex conjugate of fftn(mu)[r]; we return it in the eigenvalue
    convention above.  For symmetric measures it is real.
    """
    mu = _measure_array(group, measure, laziness).reshape(group.moduli)
    return np.conj(np.fft.fftn(mu)).ravel()


def abelian_distribution(group: AbelianGroup, measure: dict, t: int, laziness: float = 0.0) -> np.ndarray:
    """Exact law of X_t started at 0, via p_t = IFFT(FFT(mu)^t).

    Cost O(N log N) regardless of t, which is what makes t ~ n^2 on a cycle of
    length 10^4 cheap.
    """
    mu = _measure_array(group, measure, laziness).reshape(group.moduli)
    pt = np.fft.ifftn(np.fft.fftn(mu) ** t).real.ravel()
    return pt


def second_eigenvalue_modulus(eigs: np.ndarray) -> float:
    """beta = max |lambda| over nontrivial characters (the beta of the classical bound)."""
    return float(np.max(np.abs(eigs[1:])))


def spectral_gap(eigs: np.ndarray) -> float:
    """Absolute spectral gap 1 - beta."""
    return 1.0 - second_eigenvalue_modulus(eigs)


def relaxation_time(eigs: np.ndarray) -> float:
    gap = spectral_gap(eigs)
    return math.inf if gap <= 0 else 1.0 / gap


def chi2_exact(eigs: np.ndarray, t: int) -> float:
    """Identity (*): chi-square distance after t steps."""
    a = np.abs(eigs[1:])
    return float(np.sum(a ** (2 * t)))


def tv_upper_bound_lemma(eigs: np.ndarray, t: int) -> float:
    """Diaconis' upper bound lemma: TV <= sqrt(sum_{r!=0} |mu_hat|^{2t}) / 2."""
    return 0.5 * math.sqrt(chi2_exact(eigs, t))


def tv_beta_bound(eigs: np.ndarray, t: int) -> float:
    """Classical beta-bound ||P^t p - u||_1 <= sqrt(N) beta^t, in TV units."""
    n = eigs.size
    return 0.5 * math.sqrt(n) * second_eigenvalue_modulus(eigs) ** t


def beta_bound_mixing_time(eigs: np.ndarray, eps: float = 0.25) -> int:
    """Smallest t with sqrt(N) beta^t / 2 <= eps."""
    beta = second_eigenvalue_modulus(eigs)
    if beta == 0:
        return 0
    if beta >= 1:
        return 10**18
    return max(0, math.ceil(math.log(2 * eps / math.sqrt(eigs.size)) / math.log(beta)))


def upper_bound_lemma_mixing_time(eigs: np.ndarray, eps: float = 0.25) -> int:
    """Smallest t with the upper-bound-lemma bound <= eps (doubling + bisection)."""
    hi = 1
    while tv_upper_bound_lemma(eigs, hi) > eps:
        hi *= 2
        if hi > 2**50:
            raise RuntimeError("walk does not mix")
    lo = hi // 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if tv_upper_bound_lemma(eigs, mid) <= eps:
            hi = mid
        else:
            lo = mid
    return hi if tv_upper_bound_lemma(eigs, lo) > eps else lo


def relaxation_bounds(eigs: np.ndarray, eps: float = 0.25) -> tuple[float, float]:
    """Classical sandwich for reversible chains (LPW Thm 12.4 & 12.5):

        (t_rel - 1) log(1/(2 eps))  <=  t_mix(eps)  <=  t_rel log(N / eps).
    """
    trel = relaxation_time(eigs)
    n = eigs.size
    return (trel - 1) * math.log(1 / (2 * eps)), trel * math.log(n / eps)


def mixing_time_fft(group: AbelianGroup, measure: dict, eps: float = 0.25, laziness: float = 0.0) -> int:
    """Exact t_mix(eps) using FFT distributions and bisection.

    Valid because d(t) = TV(p_t, u) is non-increasing in t for any Markov
    chain (LPW Exercise 4.2).
    """
    def d(t: int) -> float:
        return distances.tv(abelian_distribution(group, measure, t, laziness))

    if d(0) <= eps:
        return 0
    hi = 1
    while d(hi) > eps:
        hi *= 2
        if hi > 2**40:
            raise RuntimeError("walk does not mix (periodic or disconnected?)")
    lo = hi // 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if d(mid) <= eps:
            hi = mid
        else:
            lo = mid
    return hi


def dense_spectrum(P) -> np.ndarray:
    """Eigenvalues of a (small) transition matrix, sorted by modulus, descending."""
    P = P.toarray() if hasattr(P, "toarray") else np.asarray(P)
    ev = np.linalg.eigvals(P)
    return ev[np.argsort(-np.abs(ev))]
