"""Random walks on Cayley graphs, computed exactly.

A walk on a group G driven by a probability measure mu is the Markov chain

    X_{t+1} = X_t * s_{t+1},   s_1, s_2, ... iid ~ mu.

When mu is uniform on a symmetric set S this is the simple random walk on the
Cayley graph Cay(G, S) (transition matrix T = A / k, A the adjacency matrix).  Its
transition operator is P = sum_s mu(s) R_s, where R_s is the permutation
x -> x s, so one step costs O(|G| |supp mu|) and the full |G| x |G| matrix is
never needed unless we want its eigenvalues.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from . import distances
from .groups import FiniteGroup


@dataclass
class CayleyWalk:
    group: FiniteGroup
    measure: dict  # group element -> weight (normalised on construction)
    laziness: float = 0.0  # extra holding probability, applied on top of mu

    def __post_init__(self) -> None:
        if not 0.0 <= self.laziness < 1.0:
            raise ValueError("laziness must be in [0, 1)")
        total = float(sum(self.measure.values()))
        if total <= 0:
            raise ValueError("measure must have positive mass")
        self._steps: list[tuple[np.ndarray, float]] = []
        for s, w in self.measure.items():
            if w < 0:
                raise ValueError("weights must be non-negative")
            if w == 0:
                continue
            self._steps.append((self.group.right_action(s), (1 - self.laziness) * w / total))
        self.n = self.group.order

    # ------------------------------------------------------------------ exact
    def step(self, p: np.ndarray) -> np.ndarray:
        out = self.laziness * p
        for act, w in self._steps:
            out[act] += w * p  # act is a permutation, so no index collisions
        return out

    def point_mass(self, start: int | None = None) -> np.ndarray:
        p = np.zeros(self.n)
        p[self.group.identity() if start is None else start] = 1.0
        return p

    def distribution(self, t: int, start: int | None = None) -> np.ndarray:
        p = self.point_mass(start)
        for _ in range(t):
            p = self.step(p)
        return p

    def profile(self, t_max: int, metric=distances.tv, start: int | None = None) -> np.ndarray:
        """metric(P^t delta_start) for t = 0..t_max."""
        p = self.point_mass(start)
        out = np.empty(t_max + 1)
        out[0] = metric(p)
        for t in range(1, t_max + 1):
            p = self.step(p)
            out[t] = metric(p)
        return out

    def mixing_time(self, eps: float = 0.25, t_max: int = 10**6, start: int | None = None) -> int:
        """Smallest t with TV(P^t delta, uniform) <= eps.

        On a group the walk is vertex-transitive, so the start does not matter
        and this is exactly t_mix(eps) of Levin-Peres-Wilmer.
        """
        p = self.point_mass(start)
        for t in range(t_max + 1):
            if distances.tv(p) <= eps:
                return t
            p = self.step(p)
        raise RuntimeError(f"did not mix to {eps} within {t_max} steps")

    # ----------------------------------------------------------------- matrix
    def transition_matrix(self, sparse: bool = True):
        rows, cols, vals = [], [], []
        idx = np.arange(self.n)
        if self.laziness:
            rows.append(idx), cols.append(idx), vals.append(np.full(self.n, self.laziness))
        for act, w in self._steps:
            rows.append(idx), cols.append(act), vals.append(np.full(self.n, w))
        P = sp.coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.n, self.n),
        ).tocsr()
        return P if sparse else P.toarray()

    # ------------------------------------------------------------ Monte Carlo
    def simulate(self, t: int, walkers: int, rng: np.random.Generator, start: int | None = None) -> np.ndarray:
        """Positions of ``walkers`` independent copies after t steps."""
        acts = np.stack([a for a, _ in self._steps])
        w = np.array([w for _, w in self._steps])
        if self.laziness:
            acts = np.vstack([np.arange(self.n), acts])
            w = np.concatenate([[self.laziness], w])
        w = w / w.sum()
        pos = np.full(walkers, self.group.identity() if start is None else start)
        for _ in range(t):
            choice = rng.choice(len(w), size=walkers, p=w)
            pos = acts[choice, pos]
        return pos


def uniform_measure(generators) -> dict:
    """Uniform measure on a list of generators (repeats count as multiplicity)."""
    m: dict = {}
    for g in generators:
        m[g] = m.get(g, 0) + 1
    return m
