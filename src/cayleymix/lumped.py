"""Exact mixing profiles on state spaces far too large to store.

The trick is *lumping*: if the walk starts at the identity and the measure is
invariant under a symmetry group, then P^t(id, .) is constant on orbits of
that symmetry, and so is the uniform distribution.  TV distance on the huge
space then equals TV between the orbit masses, and the orbit masses
themselves evolve as a small Markov chain.

* Hypercube Z_2^d, symmetric under coordinate permutations -> orbits are
  Hamming weights 0..d -> the Ehrenfest urn (d + 1 states instead of 2^d).
* Random transpositions on S_n, invariant under conjugation -> orbits are
  cycle types -> a chain on the p(n) integer partitions of n (37,338 states
  for n = 40 instead of 40! ~ 8e47).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
import scipy.sparse as sp
from scipy.stats import binom

from .distances import tv_weighted


# --------------------------------------------------------------------------
# Hypercube / Ehrenfest urn
# --------------------------------------------------------------------------
def hypercube_profile(d: int, t_max: int, hold: float = 0.5) -> np.ndarray:
    """Exact TV(t), t = 0..t_max, for the walk on Z_2^d that holds with
    probability ``hold`` and otherwise flips a uniformly random coordinate.

    hold = 1/2 is the standard lazy walk (cutoff at (1/2) d log d);
    hold = 1/(d+1) is the walk driven by the uniform measure on
    {0, e_1, ..., e_d} (cutoff at (1/4) d log d, Diaconis-Shahshahani).
    """
    w = np.arange(d + 1)
    pi = binom.pmf(w, d, 0.5)
    q = np.zeros(d + 1)
    q[0] = 1.0
    down = (1 - hold) * w / d          # prob weight w -> w-1
    up = (1 - hold) * (d - w) / d      # prob weight w -> w+1
    out = np.empty(t_max + 1)
    out[0] = tv_weighted(q, pi)
    for t in range(1, t_max + 1):
        new = hold * q
        new[:-1] += down[1:] * q[1:]
        new[1:] += up[:-1] * q[:-1]
        q = new
        out[t] = tv_weighted(q, pi)
    return out


# --------------------------------------------------------------------------
# Random transpositions / cycle-type chain
# --------------------------------------------------------------------------
def partitions(n: int) -> list[tuple[int, ...]]:
    """All partitions of n as non-increasing tuples."""
    out: list[tuple[int, ...]] = []

    def rec(remaining: int, max_part: int, prefix: list[int]) -> None:
        if remaining == 0:
            out.append(tuple(prefix))
            return
        for part in range(min(remaining, max_part), 0, -1):
            prefix.append(part)
            rec(remaining - part, part, prefix)
            prefix.pop()

    rec(n, n, [])
    return out


def z_lambda(lam: tuple[int, ...]) -> int:
    """n! / |conjugacy class of cycle type lam| = prod_v v^{m_v} m_v!."""
    z = 1
    for v, m in Counter(lam).items():
        z *= v**m * math.factorial(m)
    return z


def _replace(lam_counter: Counter, remove: list[int], add: list[int]) -> tuple[int, ...]:
    c = lam_counter.copy()
    for v in remove:
        c[v] -= 1
    for v in add:
        c[v] += 1
    return tuple(sorted(c.elements(), reverse=True))


class TranspositionChain:
    """Cycle-type process of the random-transposition shuffle on S_n.

    One step: pick i, j uniformly and independently from {1..n} and swap
    cards i and j (nothing happens if i = j, probability 1/n).  Composing a
    permutation with a transposition either merges the two cycles containing
    i and j (if different) or splits the common cycle; which split depends
    only on the distance from i to j along the cycle.  That gives the exact
    transition law on cycle types:

      stay                      1/n
      merge cycles of sizes u,v 2 u v / n^2 per unordered pair of cycles
      split a u-cycle into
        (m, u - m), 1<=m<u      u / n^2 per cycle and per m
    """

    def __init__(self, n: int):
        self.n = n
        self.states = partitions(n)
        self.index = {lam: i for i, lam in enumerate(self.states)}
        self.pi = np.array([1.0 / z_lambda(lam) for lam in self.states])
        self.P = self._build()
        self.start = self.index[(1,) * n]

    def _build(self) -> sp.csr_matrix:
        n2 = float(self.n * self.n)
        rows, cols, vals = [], [], []
        for i, lam in enumerate(self.states):
            acc: dict[int, float] = defaultdict(float)
            acc[i] += 1.0 / self.n
            cnt = Counter(lam)
            values = sorted(cnt)
            # merges
            for a_idx, u in enumerate(values):
                mu_ = cnt[u]
                if mu_ >= 2:
                    j = self.index[_replace(cnt, [u, u], [2 * u])]
                    acc[j] += u * u * mu_ * (mu_ - 1) / n2
                for v in values[a_idx + 1:]:
                    j = self.index[_replace(cnt, [u, v], [u + v])]
                    acc[j] += 2.0 * u * v * mu_ * cnt[v] / n2
            # splits
            for u in values:
                for m in range(1, u):
                    j = self.index[_replace(cnt, [u], [m, u - m])]
                    acc[j] += u * cnt[u] / n2
            for j, w in acc.items():
                rows.append(i)
                cols.append(j)
                vals.append(w)
        k = len(self.states)
        return sp.csr_matrix((vals, (rows, cols)), shape=(k, k))

    def profile(self, t_max: int) -> np.ndarray:
        q = np.zeros(len(self.states))
        q[self.start] = 1.0
        PT = self.P.T.tocsr()
        out = np.empty(t_max + 1)
        out[0] = tv_weighted(q, self.pi)
        for t in range(1, t_max + 1):
            q = PT @ q
            out[t] = tv_weighted(q, self.pi)
        return out


def first_below(profile: np.ndarray, eps: float) -> int:
    """First index t with profile[t] <= eps (profiles are non-increasing)."""
    hits = np.nonzero(profile <= eps)[0]
    if hits.size == 0:
        raise ValueError("profile never drops below eps; extend t_max")
    return int(hits[0])
