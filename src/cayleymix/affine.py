"""The Chung-Diaconis-Graham process: a linear congruential generator with noise.

Random-number generators are often introduced through the LCG  X_{n+1} = a X_n + b (mod m).
Replace the constant increment b by a small random kick:

    X_{t+1} = a X_t + eps_t  (mod p),   eps_t uniform on {-1, 0, +1}.

* a = 1 is the lazy-ish random walk on the cycle Z/p: it needs ~ p^2 steps.
* a = 2 multiplies before every kick, which spreads mass exponentially fast.
  Chung, Diaconis & Graham (1987) proved O(log p log log p) steps suffice for
  every odd p, and ~c log p for almost all p; Hildebrand later showed the
  log log p factor is really needed for p = 2^m - 1 (Mersenne-type moduli),
  where multiplication by 2 is just a cyclic bit-rotation.

This module evolves the exact law (O(p) per step).  For a = 1 use the FFT
route in ``spectral.py`` since t ~ p^2 is large.
"""
from __future__ import annotations

import numpy as np

from .distances import tv


class AffineProcess:
    def __init__(self, p: int, a: int, kicks=(-1, 0, 1), weights=None):
        if np.gcd(a, p) != 1:
            raise ValueError("a must be invertible mod p for a bijective step")
        self.p, self.a = p, a
        kicks = np.asarray(kicks)
        w = np.ones(len(kicks)) if weights is None else np.asarray(weights, dtype=float)
        self.w = w / w.sum()
        x = np.arange(p)
        self.targets = [(a * x + e) % p for e in kicks]

    def step(self, q: np.ndarray) -> np.ndarray:
        out = np.zeros_like(q)
        for tgt, w in zip(self.targets, self.w):
            out[tgt] += w * q
        return out

    def profile(self, t_max: int, start: int = 0) -> np.ndarray:
        q = np.zeros(self.p)
        q[start] = 1.0
        out = np.empty(t_max + 1)
        out[0] = tv(q)
        for t in range(1, t_max + 1):
            q = self.step(q)
            out[t] = tv(q)
        return out

    def mixing_time(self, eps: float = 0.25, t_max: int = 100_000, start: int = 0) -> int:
        """Worst start is not guaranteed to be 0, but the chain is
        translation-covariant up to the affine map, and 0 is the conventional
        start used by CDG; ``worst_start_mixing_time`` checks all starts on
        small p."""
        q = np.zeros(self.p)
        q[start] = 1.0
        for t in range(t_max + 1):
            if tv(q) <= eps:
                return t
            q = self.step(q)
        raise RuntimeError("did not mix")

    def worst_start_mixing_time(self, eps: float = 0.25, t_max: int = 10_000) -> int:
        """Max over all starts, by evolving the full p x p matrix (small p only)."""
        Q = np.eye(self.p)
        for t in range(t_max + 1):
            d = 0.5 * np.abs(Q - 1.0 / self.p).sum(axis=1).max()
            if d <= eps:
                return t
            new = np.zeros_like(Q)
            for tgt, w in zip(self.targets, self.w):
                new[:, tgt] += w * Q
            Q = new
        raise RuntimeError("did not mix")
