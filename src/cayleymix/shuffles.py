"""Riffle shuffles: exact total variation via rising sequences.

The Gilbert-Shannon-Reeds (GSR) model: cut the deck Binomial(n, 1/2), then
drop cards from the two packets with probability proportional to packet size.
It is a random walk on S_n, and Bayer & Diaconis (1992, "Trailing the
dovetail shuffle to its lair") showed that after k shuffles

    P(sigma) = C(2^k + n - R(sigma), n) / 2^{kn},

where R(sigma) is the number of *rising sequences* of the arrangement.  The
law depends on sigma only through R, and #{sigma : R(sigma) = r} is the
Eulerian number A(n, r - 1) (permutations with r - 1 descents).  So the
exact TV over all 52! ~ 8e67 arrangements is a sum of 52 terms, computed
here in exact rational arithmetic.
"""
from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def eulerian_row(n: int) -> tuple[int, ...]:
    """A(n, k) for k = 0..n-1: permutations of n with exactly k descents."""
    row = [1]
    for m in range(2, n + 1):
        new = [0] * m
        for k in range(m):
            left = (k + 1) * row[k] if k < len(row) else 0
            right = (m - k) * row[k - 1] if k >= 1 else 0
            new[k] = left + right
        row = new
    return tuple(row)


def a_shuffle_rising_law(n: int, a: int) -> list[Fraction]:
    """P(R = r) for r = 1..n after one a-shuffle (= k GSR shuffles, a = 2^k)."""
    A = eulerian_row(n)
    denom = a**n
    return [Fraction(A[r - 1] * math.comb(a + n - r, n), denom) for r in range(1, n + 1)]


def uniform_rising_law(n: int) -> list[Fraction]:
    A = eulerian_row(n)
    f = math.factorial(n)
    return [Fraction(A[r - 1], f) for r in range(1, n + 1)]


def riffle_tv_exact(n: int, k: int) -> Fraction:
    """Exact TV distance to uniform after k GSR shuffles of n cards."""
    a = 2**k
    p = a_shuffle_rising_law(n, a)
    u = uniform_rising_law(n)
    return sum((abs(x - y) for x, y in zip(p, u)), Fraction(0)) / 2


def riffle_separation_exact(n: int, k: int) -> Fraction:
    """Separation distance max(1 - n! P(sigma)); the max is at R = n (reversal)."""
    a = 2**k
    p_min = Fraction(math.comb(a, n), a**n)  # C(a + n - n, n) / a^n
    return 1 - math.factorial(n) * p_min


def top_card_position_law(n: int, k: int) -> np.ndarray:
    """Exact law of the final position of the original top card after k shuffles.

    In the inverse-shuffle picture each card gets an iid uniform label in
    {0..a-1} (a = 2^k) and the deck is stably sorted by label; the forward
    shuffle sends the top card to position c exactly when card c ends up on
    top of the inverse-shuffled deck, i.e. when its label is strictly below
    those of cards 0..c-1 and at most those of cards c+1..n-1.
    """
    a = 2**k
    v = np.arange(a, dtype=float)
    probs = np.array([
        np.sum((1 / a) * ((a - 1 - v) / a) ** c * ((a - v) / a) ** (n - 1 - c))
        for c in range(n)
    ])
    return probs


def simulate_rising_sequences(n: int, k: int, trials: int, rng: np.random.Generator) -> np.ndarray:
    """Sample R after k GSR shuffles, via k inverse shuffles = sort by k random bits.

    Returns an int array of R values.  (R of the forward arrangement equals
    1 + #descents of the inverse-shuffled deck read top to bottom.)
    """
    out = np.empty(trials, dtype=np.int64)
    chunk = 20000
    for start in range(0, trials, chunk):
        m = min(chunk, trials - start)
        keys = rng.integers(0, 2**k, size=(m, n), dtype=np.int64)
        deck_inv = np.argsort(keys, axis=1, kind="stable")
        out[start:start + m] = 1 + (deck_inv[:, 1:] < deck_inv[:, :-1]).sum(axis=1)
    return out


def gsr_measure_bruteforce(n: int) -> dict[tuple[int, ...], float]:
    """One GSR shuffle as an explicit measure on S_n (for tests, small n).

    Enumerates the 2^n inverse-shuffle labelings; the forward shuffle is the
    inverse permutation.  Keys are decks: ``deck[i]`` = card at position i.
    """
    meas: dict[tuple[int, ...], float] = {}
    for bits in range(2**n):
        labels = [(bits >> i) & 1 for i in range(n)]
        deck_inv = sorted(range(n), key=lambda c: labels[c])  # stable
        deck = [0] * n  # forward deck = inverse permutation of deck_inv
        for pos, card in enumerate(deck_inv):
            deck[card] = pos
        key = tuple(deck)
        meas[key] = meas.get(key, 0.0) + 2.0**-n
    return meas


# --------------------------------------------------------------------------
# floating-point version for large decks (n in the thousands)
# --------------------------------------------------------------------------
def log_eulerian_row(n: int) -> np.ndarray:
    """log A(n, k), k = 0..n-1, by the Eulerian recurrence in log space."""
    row = np.array([0.0])
    for m in range(2, n + 1):
        k = np.arange(m)
        left = np.full(m, -np.inf)
        left[: m - 1] = np.log(k[: m - 1] + 1) + row
        right = np.full(m, -np.inf)
        right[1:] = np.log(m - k[1:]) + row
        row = np.logaddexp(left, right)
    return row


def riffle_tv_float(n: int, k: int | None = None, log_A: np.ndarray | None = None,
                    a: int | None = None) -> float:
    """TV after k GSR shuffles (or one a-shuffle), accurate for n up to ~10^4.

    Writes the exact law as  P(R = r) = w_r * rho_r  with uniform weights
    w_r = A(n, r-1) / n!  and likelihood ratio

        rho_r = n! C(a + n - r, n) / a^n = prod_{i=1}^{n} (1 + (i - r)/a),

    evaluated as a difference of cumulative sums of log1p(j / a), which keeps
    full precision even when a = 2^k is huge.
    """
    if log_A is None:
        log_A = log_eulerian_row(n)
    a = float(2**k) if a is None else float(a)
    log_w = log_A - np.logaddexp.reduce(log_A)  # uniform R-law, renormalised
    w = np.exp(log_w)
    r = np.arange(1, n + 1)
    j = np.arange(-n, n + 1)  # need j in [1 - r, n - r]
    alive = 1 + j / a > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(alive, np.log1p(np.where(alive, j / a, 0.0)), 0.0)
    G = np.concatenate([[0.0], np.cumsum(g)])
    dead = np.concatenate([[0], np.cumsum(~alive)])
    lo = (1 - r) + n          # index of j = 1 - r in the j array
    hi = (n - r) + n + 1      # one past j = n - r
    log_rho = G[hi] - G[lo]
    zero = (dead[hi] - dead[lo]) > 0
    # P(R = r) = w_r rho_r <= 1, so exponentiate the product in log space
    p = np.where(zero, 0.0, np.exp(np.where(zero, -np.inf, log_w + log_rho)))
    return 0.5 * float(np.sum(np.abs(p - w)))
