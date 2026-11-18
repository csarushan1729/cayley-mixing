"""E6 - How many riffle shuffles mix a deck?  Exact answers, and a lesson about
measuring randomness from samples.

(a) Exact TV and separation for 52 cards, k = 1..12 (Bayer-Diaconis Table 1).
(b) Cutoff at (3/2) log2 n for decks of up to 13,312 cards, against the
    Bayer-Diaconis limit  TV -> 1 - 2 Phi(-2^{-theta} / (4 sqrt 3)),
    k = (3/2) log2 n + theta.
(c) Estimating TV from samples.  The empirical distribution of a sample over
    52! arrangements is useless (every sample is distinct), but the law of
    the deck depends only on the number of rising sequences R, so R is a
    *sufficient statistic*: TV on S_52 equals TV of the R-distribution, a
    52-cell problem that Monte Carlo handles well.
(d) The flip side: a statistic that is NOT sufficient can look random long
    before the deck is.  The position of the original top card is close to
    uniform after a few shuffles while the full deck is still far from it.
"""
import math

import numpy as np
from scipy.stats import norm

from _common import dump, plt, save
from cayleymix import shuffles

RNG = np.random.default_rng(2026)
N = 52

# (a) ---------------------------------------------------------------------
table = []
for k in range(1, 13):
    tv = float(shuffles.riffle_tv_exact(N, k))
    sep = float(shuffles.riffle_separation_exact(N, k))
    top = shuffles.top_card_position_law(N, k)
    table.append({"k": k, "tv": tv, "separation": sep, "top_card_tv": 0.5 * float(np.abs(top - 1 / N).sum())})
print("  k    TV       sep     top-card TV")
for r in table:
    print(f"  {r['k']:2d}  {r['tv']:.4f}  {r['separation']:.4f}  {r['top_card_tv']:.4f}")

# (b) ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
cut = []
theta_grid = np.linspace(-4, 4, 200)
limit = 1 - 2 * norm.cdf(-(2.0 ** -theta_grid) / (4 * math.sqrt(3)))
for n in [13, 52, 208, 832, 3328, 13312]:
    L = shuffles.log_eulerian_row(n)
    lo, hi = 1.5 * math.log2(n) - 4.5, 1.5 * math.log2(n) + 4.5
    a_vals = np.unique(np.round(2 ** np.linspace(max(lo, 0.0), hi, 160)).astype(np.int64))
    tvs = np.array([shuffles.riffle_tv_float(n, log_A=L, a=int(a)) for a in a_vals])
    theta = np.log2(a_vals) - 1.5 * math.log2(n)
    axes[0].plot(theta, tvs, label=f"n = {n:,}")
    lim_at = 1 - 2 * norm.cdf(-(2.0 ** -theta) / (4 * math.sqrt(3)))
    k_int = np.arange(1, int(hi) + 2)
    tv_int = np.array([shuffles.riffle_tv_float(n, int(k), log_A=L) for k in k_int])
    cut.append({
        "n": n,
        "shuffles_to_tv_quarter": int(k_int[np.argmax(tv_int <= 0.25)]),
        "three_halves_log2_n": 1.5 * math.log2(n),
        "sup_err_vs_limit": float(np.max(np.abs(tvs - lim_at)[np.abs(theta) <= 4])),
    })
axes[0].plot(theta_grid, limit, "k--", lw=2.5, label="Bayer-Diaconis limit")
axes[0].set_xlabel(r"$\theta = \log_2 a - \frac{3}{2}\log_2 n$   ($a = 2^k$ for $k$ shuffles)")
axes[0].set_ylabel("TV to uniform")
axes[0].set_title("riffle shuffle cutoff at $\\frac{3}{2}\\log_2 n$")
axes[0].legend(fontsize=7)
print("  n       shuffles for TV<=1/4   (3/2)log2 n   sup|TV - limit|")
for r in cut:
    print(f"  {r['n']:6d}  {r['shuffles_to_tv_quarter']:4d}               {r['three_halves_log2_n']:6.2f}        {r['sup_err_vs_limit']:.4f}")

# (c) ---------------------------------------------------------------------
TRIALS = 200_000
mc = []
for k in range(4, 12):
    sample = shuffles.simulate_rising_sequences(N, k, TRIALS, RNG)
    emp = np.bincount(sample, minlength=N + 1)[1:] / TRIALS
    unif = np.array([float(x) for x in shuffles.uniform_rising_law(N)])
    exact_law = np.array([float(x) for x in shuffles.a_shuffle_rising_law(N, 2**k)])
    # plug-in estimate and its bias at this sample size (estimated by resampling the exact law)
    plug = 0.5 * np.abs(emp - unif).sum()
    sims = RNG.multinomial(TRIALS, exact_law, size=200) / TRIALS
    bias = float(np.mean(0.5 * np.abs(sims - unif).sum(axis=1)) - 0.5 * np.abs(exact_law - unif).sum())
    mc.append({"k": k, "exact": table[k - 1]["tv"], "plug_in": float(plug), "plug_in_bias": bias,
               "bias_corrected": float(plug - bias)})
ks = [r["k"] for r in mc]
axes[1].semilogy(ks, [r["exact"] for r in mc], "ko-", lw=2, label="exact TV on $S_{52}$")
axes[1].semilogy(ks, [r["plug_in"] for r in mc], "s--", label=f"Monte Carlo via $R$, plug-in ({TRIALS:,} decks)")
axes[1].semilogy(ks, [max(r["bias_corrected"], 1e-4) for r in mc], "^:", label="bias-corrected")
axes[1].set_xlabel("number of riffle shuffles k (52 cards)")
axes[1].set_title("sufficient statistic: rising sequences")
axes[1].legend(fontsize=8)

# (d) ---------------------------------------------------------------------
kk = [r["k"] for r in table]
axes[2].plot(kk, [r["tv"] for r in table], "ko-", lw=2, label="whole deck")
axes[2].plot(kk, [r["top_card_tv"] for r in table], "s-", label="position of original top card")
axes[2].plot(kk, [r["separation"] for r in table], "^:", label="separation distance")
axes[2].axhline(0.25, color="gray", lw=0.8)
axes[2].set_xlabel("number of riffle shuffles k (52 cards)")
axes[2].set_ylabel("distance to uniform")
axes[2].set_title("a single statistic can look random too early")
axes[2].legend(fontsize=8)
save(fig, "e06_riffle_shuffles")
dump("e06_riffle_shuffles", {"table_52": table, "cutoff": cut, "monte_carlo": mc})
print("  MC: k  exact   plug-in  bias   corrected")
for r in mc:
    print(f"      {r['k']:2d} {r['exact']:.4f}  {r['plug_in']:.4f}  {r['plug_in_bias']:.4f} {r['bias_corrected']:.4f}")
