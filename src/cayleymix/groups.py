"""Finite groups with elements indexed 0..N-1.

Every group exposes the one operation a random walk needs: for a group
element ``s``, the permutation of element indices ``x -> x * s``.  With that,
a random walk driven by a probability measure ``mu`` on the group is just a
weighted sum of permutations of a probability vector (see ``walk.py``).

Two families are implemented:

* ``AbelianGroup(moduli)`` -- Z/n1 x ... x Z/nk.  Cycles, tori and the
  hypercube are special cases.  These carry an explicit character table, so
  their spectra are available in closed form (``spectral.py``).
* ``SymmetricGroup(n)`` -- all n! permutations, for brute-force checks on small
  n.  (Large symmetric groups are handled exactly by lumping, see
  ``shuffles.py``.)
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np


class FiniteGroup:
    """Minimal interface used by the walk engine."""

    order: int

    def identity(self) -> int:  # pragma: no cover - interface
        raise NotImplementedError

    def right_action(self, s) -> np.ndarray:  # pragma: no cover - interface
        """Index array ``a`` with ``a[x] = index(x * s)`` for every element x."""
        raise NotImplementedError

    def inverse(self, s):  # pragma: no cover - interface
        raise NotImplementedError


# --------------------------------------------------------------------------
# Abelian groups
# --------------------------------------------------------------------------
@dataclass
class AbelianGroup(FiniteGroup):
    """The group Z/n1 x Z/n2 x ... x Z/nk, written additively.

    Elements are tuples of residues; they are indexed in C (row-major) order,
    which is also the layout ``numpy.fft.fftn`` expects.  That alignment is
    what lets ``spectral.abelian_distribution`` compute exact distributions
    by FFT.
    """

    moduli: tuple[int, ...]
    _coords: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.moduli = tuple(int(m) for m in self.moduli)
        if any(m < 1 for m in self.moduli):
            raise ValueError("moduli must be positive")
        self.order = int(np.prod(self.moduli))
        grids = np.indices(self.moduli).reshape(len(self.moduli), -1)
        self._coords = grids.T.copy()  # (order, k)

    # element <-> index -----------------------------------------------------
    def element(self, idx: int) -> tuple[int, ...]:
        return tuple(int(v) for v in self._coords[idx])

    def index(self, g) -> int:
        g = self._normalise(g)
        return int(np.ravel_multi_index(g, self.moduli))

    def _normalise(self, g) -> tuple[int, ...]:
        if np.isscalar(g):
            g = (g,)
        if len(g) != len(self.moduli):
            raise ValueError(f"element {g} has wrong length for {self.moduli}")
        return tuple(int(v) % m for v, m in zip(g, self.moduli))

    # group structure ---------------------------------------------------------
    def identity(self) -> int:
        return 0

    def inverse(self, g):
        g = self._normalise(g)
        return tuple((-v) % m for v, m in zip(g, self.moduli))

    def right_action(self, s) -> np.ndarray:
        s = np.array(self._normalise(s))
        moved = (self._coords + s) % np.array(self.moduli)
        return np.ravel_multi_index(moved.T, self.moduli)

    def character_values(self, s) -> np.ndarray:
        """chi_r(s) for every character r, in the same flat order as elements.

        Characters of Z/n1 x ... x Z/nk are indexed by r in the same group:
        chi_r(x) = exp(2 pi i * sum_j r_j x_j / n_j).
        """
        s = np.array(self._normalise(s), dtype=float)
        phase = (self._coords * (s / np.array(self.moduli))).sum(axis=1)
        return np.exp(2j * np.pi * phase)

    def __repr__(self) -> str:
        return "Z/" + " x Z/".join(map(str, self.moduli))


def cyclic(n: int) -> AbelianGroup:
    return AbelianGroup((n,))


def torus(n: int, d: int) -> AbelianGroup:
    return AbelianGroup((n,) * d)


def hypercube(d: int) -> AbelianGroup:
    return AbelianGroup((2,) * d)


def standard_generators(group: AbelianGroup, symmetric: bool = True) -> list[tuple[int, ...]]:
    """+-e_j for each coordinate (just e_j when the modulus is 2)."""
    k = len(group.moduli)
    gens: list[tuple[int, ...]] = []
    for j, m in enumerate(group.moduli):
        e = [0] * k
        e[j] = 1
        gens.append(tuple(e))
        if symmetric and m > 2:
            e[j] = m - 1
            gens.append(tuple(e))
    return gens


# --------------------------------------------------------------------------
# Symmetric groups
# --------------------------------------------------------------------------
class SymmetricGroup(FiniteGroup):
    """All permutations of {0..n-1}; practical for n <= 8 or so.

    A permutation is stored as a tuple ``p`` with ``p[i]`` = the card in
    position i.  The product is composition ``(x * s)[i] = x[s[i]]``, i.e.
    applying the rearrangement ``s`` to the deck ``x``.
    """

    def __init__(self, n: int):
        if n > 9:
            raise ValueError("SymmetricGroup is brute force; use lumped chains for n > 9")
        self.n = n
        self.order = math.factorial(n)
        self.perms = np.array(list(itertools.permutations(range(n))), dtype=np.int16)
        self._fact = np.array([math.factorial(n - 1 - i) for i in range(n)], dtype=np.int64)

    def rank(self, perms: np.ndarray) -> np.ndarray:
        """Lexicographic rank of each row (matches itertools order)."""
        perms = np.atleast_2d(perms)
        n = self.n
        lehmer = np.zeros(perms.shape, dtype=np.int64)
        for i in range(n):
            lehmer[:, i] = (perms[:, i + 1:] < perms[:, [i]]).sum(axis=1)
        return lehmer @ self._fact

    def index(self, p) -> int:
        return int(self.rank(np.array(p))[0])

    def element(self, idx: int) -> tuple[int, ...]:
        return tuple(int(v) for v in self.perms[idx])

    def identity(self) -> int:
        return 0

    def inverse(self, p):
        p = np.asarray(p)
        inv = np.empty_like(p)
        inv[p] = np.arange(len(p))
        return tuple(int(v) for v in inv)

    def right_action(self, s) -> np.ndarray:
        s = np.asarray(s)
        return self.rank(self.perms[:, s])

    def __repr__(self) -> str:
        return f"S_{self.n}"


def transposition(n: int, i: int, j: int) -> tuple[int, ...]:
    p = list(range(n))
    p[i], p[j] = p[j], p[i]
    return tuple(p)
