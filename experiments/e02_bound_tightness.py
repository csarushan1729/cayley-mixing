"""E2 - How good is the classical beta-bound as a *mixing-time* estimate?

The classical bound  ||T^m p - u||_1 <= sqrt(N) beta^m.  Solving for m gives a
mixing-time bound ~ log(sqrt N) / (1 - beta).  We compare four answers to
"how many steps until TV <= 1/4?":

  exact            -- computed exactly (FFT on the cycle, Ehrenfest lumping on
                      the hypercube)
  upper-bound lemma-- TV <= (1/2) sqrt(sum_{r != 0} |lambda_r|^{2t})  (uses the
                      whole spectrum, Diaconis 1988)
  beta-bound       -- sqrt(N) beta^t, uses only the second eigenvalue
  relaxation sandwich -- (t_rel - 1) log 2 <= t_mix <= t_rel log(4N)

Punchline: on the cycle every spectral method is right up to log factors,
because one eigenvalue dominates.  On the hypercube N = 2^d, the
sqrt(N) factor costs a whole factor d / log d: the beta-bound predicts ~ (ln 2 / 2) d^2
steps, the truth is ~ (1/2) d log d.  Using all eigenvalues *with their
multiplicities* recovers the right answer.
"""
import math

import numpy as np
from scipy.special import gammaln, logsumexp

from _common import dump, plt, save
from cayleymix import lumped, spectral
from cayleymix.groups import cyclic
from cayleymix.walk import uniform_measure

EPS = 0.25


def first_t(f, eps=EPS):
    hi = 1
    while f(hi) > eps:
        hi *= 2
    lo = hi // 2
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if f(mid) <= eps:
            hi = mid
        else:
            lo = mid
    return hi


# ------------------------------------------------------------- hypercube
hyper = []
for d in [8, 16, 32, 64, 128, 256, 512, 1024]:
    j = np.arange(1, d + 1)
    log_mult = gammaln(d + 1) - gammaln(j + 1) - gammaln(d - j + 1)
    lam = 1 - j / d  # lazy walk: eigenvalue 1 - j/d with multiplicity C(d, j)

    def ubl(t):
        with np.errstate(divide="ignore"):
            terms = log_mult + 2 * t * np.log(np.abs(lam))
        return 0.5 * math.exp(0.5 * logsumexp(terms))

    beta = 1 - 1 / d
    log_n = d * math.log(2)
    bb = math.ceil(math.log(2 * EPS / math.exp(0.5 * log_n)) / math.log(beta))
    trel = d
    prof = lumped.hypercube_profile(d, int(0.5 * d * math.log(d) + 3 * d + 10))
    hyper.append({
        "d": d,
        "exact": lumped.first_below(prof, EPS),
        "upper_bound_lemma": first_t(ubl),
        "beta_bound": bb,
        "relax_upper": trel * (log_n + math.log(1 / EPS)),
        "relax_lower": (trel - 1) * math.log(1 / (2 * EPS)),
        "half_d_log_d": 0.5 * d * math.log(d),
    })

# ------------------------------------------------------------- cycle
cyc = []
meas = uniform_measure([(1,), (-1,)])
for n in [16, 32, 64, 128, 256, 512, 1024, 2048]:
    G = cyclic(n)
    eigs = spectral.abelian_eigenvalues(G, meas, 0.5).real
    trel = spectral.relaxation_time(eigs)
    lo, hi = spectral.relaxation_bounds(eigs, EPS)
    cyc.append({
        "n": n,
        "exact": spectral.mixing_time_fft(G, meas, EPS, 0.5),
        "upper_bound_lemma": spectral.upper_bound_lemma_mixing_time(eigs, EPS),
        "beta_bound": spectral.beta_bound_mixing_time(eigs, EPS),
        "relax_upper": hi,
        "relax_lower": lo,
        "t_rel": trel,
    })

# ------------------------------------------------------------- plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
style = {
    "exact": dict(marker="o", lw=2.5, color="k", label="exact $t_{mix}(1/4)$"),
    "upper_bound_lemma": dict(marker="s", label="upper-bound lemma (all eigenvalues)"),
    "beta_bound": dict(marker="^", label=r"$\beta$-bound $\sqrt{N}\beta^t/2$"),
    "relax_upper": dict(ls="--", label=r"$t_{rel}\log(4N)$"),
    "relax_lower": dict(ls=":", label=r"$(t_{rel}-1)\log 2$ (lower)"),
}
for ax, rows, key, title in [(axes[0], hyper, "d", r"lazy walk on hypercube $\{0,1\}^d$, $N=2^d$"),
                             (axes[1], cyc, "n", r"lazy walk on cycle $\mathbb{Z}/n$")]:
    x = [r[key] for r in rows]
    for k, st in style.items():
        ax.loglog(x, [r[k] for r in rows], **st)
    if key == "d":
        ax.loglog(x, [r["half_d_log_d"] for r in rows], color="gray", lw=1, alpha=0.7,
                  label=r"$\frac{1}{2} d\log d$")
    ax.set_xlabel(key)
    ax.set_ylabel("steps")
    ax.set_title(title)
    ax.legend(fontsize=8)
save(fig, "e02_bound_tightness")
dump("e02_bound_tightness", {"hypercube": hyper, "cycle": cyc})

print("  hypercube   d  exact   UBL   beta    beta/exact")
for r in hyper:
    print(f"           {r['d']:5d} {r['exact']:6d} {r['upper_bound_lemma']:6d} {r['beta_bound']:7d}  {r['beta_bound']/r['exact']:6.1f}x")
print("  cycle       n  exact     UBL    beta   beta/exact")
for r in cyc:
    print(f"           {r['n']:5d} {r['exact']:7d} {r['upper_bound_lemma']:7d} {r['beta_bound']:7d}  {r['beta_bound']/r['exact']:6.2f}x")
