"""Distances from a probability vector to the uniform distribution.

Conventions (all standard, cf. Levin-Peres-Wilmer, *Markov Chains and Mixing
Times*, ch. 4):

* total variation  ``tv(p) = 1/2 sum |p(x) - 1/N|``  -- lies in [0, 1]
* L1               ``l1(p) = sum |p(x) - 1/N| = 2 tv(p)`` (Terras' normalisation)
* chi-square / L2(u) ``chi2(p) = N sum (p(x) - 1/N)^2``; ``tv <= sqrt(chi2)/2``
* separation        ``sep(p) = max_x (1 - N p(x))``;      ``tv <= sep``
"""
from __future__ import annotations

import numpy as np


def tv(p: np.ndarray, q: np.ndarray | None = None) -> float:
    p = np.asarray(p, dtype=float)
    if q is None:
        return 0.5 * float(np.abs(p - 1.0 / p.size).sum())
    return 0.5 * float(np.abs(p - np.asarray(q, dtype=float)).sum())


def l1(p: np.ndarray) -> float:
    return 2.0 * tv(p)


def l2(p: np.ndarray) -> float:
    """Plain Euclidean norm ||p - u||_2 (Terras' L2 norm)."""
    p = np.asarray(p, dtype=float)
    return float(np.linalg.norm(p - 1.0 / p.size))


def chi2(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    n = p.size
    return float(n * ((p - 1.0 / n) ** 2).sum())


def separation(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    return float(np.max(1.0 - p.size * p))


def tv_weighted(p_classes: np.ndarray, pi_classes: np.ndarray) -> float:
    """TV between two distributions given on classes (lumped states).

    When both distributions are constant on each class (e.g. Hamming weight on
    the hypercube, cycle type on S_n), TV on the big space equals TV between
    the class masses.  This is how exact distances on spaces of size 2^1000 or
    40! are computed.
    """
    return 0.5 * float(np.abs(np.asarray(p_classes) - np.asarray(pi_classes)).sum())
