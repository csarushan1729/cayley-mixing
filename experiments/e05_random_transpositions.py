"""E5 - Random transpositions on S_n: exact cutoff for |S_n| up to 50! ~ 3e64.

The Diaconis-Shahshahani (1981) shuffle: swap two uniformly chosen cards (the
same card twice = do nothing).  They proved cutoff at (1/2) n log n with
window O(n), using the character theory of S_n.  Teyssier (2020) found the
exact limit shape: with t = (n/2)(log n + c),

    TV(t) -> TV( Poisson(1 + e^{-c}), Poisson(1) ).

(Intuition: e^{-c} cards are still untouched and sit at their original
places, on top of the Poisson(1) fixed points of a uniform permutation.)

Because the walk is conjugation-invariant, the law at every time is a class
function, so TV is computed *exactly* on the chain of cycle types
(lumped.TranspositionChain).
"""
import math
import time

import numpy as np
from scipy.stats import poisson

from _common import dump, plt, save
from cayleymix import lumped


def teyssier(c):
    k = np.arange(0, 80)
    return np.array([0.5 * np.abs(poisson.pmf(k, 1 + math.exp(-ci)) - poisson.pmf(k, 1)).sum() for ci in np.atleast_1d(c)])


fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
rows = []
for n in [10, 20, 30, 40, 50]:
    t0 = time.time()
    chain = lumped.TranspositionChain(n)
    T = int(0.5 * n * math.log(n) + 3 * n)
    prof = chain.profile(T)
    t = np.arange(T + 1)
    t_c = 0.5 * n * math.log(n)
    c = 2 * t / n - math.log(n)
    axes[0].plot(t / t_c, prof, label=f"n = {n}  ({len(chain.states):,} cycle types)")
    axes[1].plot(c, prof, label=f"n = {n}")
    mask = (c >= -3) & (c <= 5)
    rows.append({
        "n": n,
        "cycle_types": len(chain.states),
        "log10_group_order": math.lgamma(n + 1) / math.log(10),
        "t_mix_quarter": lumped.first_below(prof, 0.25),
        "half_n_log_n": t_c,
        "sup_err_vs_teyssier": float(np.max(np.abs(prof[mask] - teyssier(c[mask])))),
        "seconds": time.time() - t0,
    })
    print(f"  n={n}: {len(chain.states)} states, t_mix={rows[-1]['t_mix_quarter']}, "
          f"(1/2)n log n={t_c:.1f}, sup err vs limit={rows[-1]['sup_err_vs_teyssier']:.4f}, {rows[-1]['seconds']:.1f}s")
cg = np.linspace(-3, 5, 200)
axes[1].plot(cg, teyssier(cg), "k--", lw=2.5, label="Teyssier limit")
axes[0].set_xlim(0, 2.2)
axes[0].set_xlabel(r"$t\,/\,(\frac{1}{2} n\log n)$")
axes[0].set_ylabel("TV to uniform on $S_n$")
axes[0].set_title("random transpositions: exact TV on $S_n$")
axes[0].legend(fontsize=7)
axes[1].set_xlim(-3, 5)
axes[1].set_xlabel(r"window coordinate $c = 2t/n - \log n$")
axes[1].set_title(r"limit profile $TV(\mathrm{Poi}(1+e^{-c}),\mathrm{Poi}(1))$")
axes[1].legend(fontsize=8)
save(fig, "e05_random_transpositions")
dump("e05_random_transpositions", rows)
