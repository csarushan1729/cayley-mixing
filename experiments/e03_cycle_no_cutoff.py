"""E3 - The cycle: diffusive mixing, t_mix ~ c n^2, and *no cutoff*.

For the lazy walk on Z/n the eigenvalues are lambda_k = (1 + cos(2 pi k / n)) / 2,
so at time t = s n^2,  lambda_k^t -> exp(-pi^2 k^2 s).  The law of X_t / n
therefore converges to a wrapped Gaussian (the heat kernel on the circle)

    f_s(x) = 1 + 2 sum_{k>=1} exp(-pi^2 k^2 s) cos(2 pi k x),

and TV(s n^2) -> TV_inf(s) = (1/2) int_0^1 |f_s - 1|.  The limit is a smooth
function of s: the walk forgets its start gradually.  Contrast with E4/E5/E6,
where TV falls off a cliff.  We check the convergence numerically and read
off the constant c = t_mix(1/4) / n^2.
"""
import numpy as np
from scipy.optimize import brentq

from _common import dump, plt, save
from cayleymix import distances, spectral
from cayleymix.groups import cyclic
from cayleymix.walk import uniform_measure

meas = uniform_measure([(1,), (-1,)])
s_grid = np.linspace(0.002, 0.25, 70)
x = (np.arange(20000) + 0.5) / 20000


def tv_limit(s: float) -> float:
    k = np.arange(1, 200)[:, None]
    f = 1 + 2 * np.sum(np.exp(-np.pi**2 * k**2 * s) * np.cos(2 * np.pi * k * x), axis=0)
    return 0.5 * float(np.mean(np.abs(f - 1)))


limit = np.array([tv_limit(s) for s in s_grid])
c_star = brentq(lambda s: tv_limit(s) - 0.25, 0.01, 0.3)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
rows = []
for n in [8, 16, 32, 64, 256, 1024, 4096]:
    G = cyclic(n)
    prof = [distances.tv(spectral.abelian_distribution(G, meas, int(round(s * n * n)), 0.5)) for s in s_grid]
    axes[0].plot(s_grid, prof, label=f"n = {n}", lw=1.2)
    eigs = spectral.abelian_eigenvalues(G, meas, 0.5).real
    tmix = spectral.mixing_time_fft(G, meas, 0.25, 0.5)
    rows.append({
        "n": n,
        "t_mix": tmix,
        "t_mix_over_n2": tmix / n**2,
        "t_rel": spectral.relaxation_time(eigs),
        "t_mix_over_t_rel": tmix / spectral.relaxation_time(eigs),
        "sup_gap_to_limit": float(np.max(np.abs(np.array(prof) - limit))),
    })
axes[0].plot(s_grid, limit, "k--", lw=2.5, label="heat-kernel limit")
axes[0].axhline(0.25, color="gray", lw=0.8)
axes[0].set_xlabel(r"rescaled time $s = t / n^2$")
axes[0].set_ylabel("TV to uniform")
axes[0].set_title("lazy walk on the n-cycle: profiles collapse onto a smooth curve")
axes[0].legend(fontsize=8)

ns = np.array([r["n"] for r in rows])
axes[1].loglog(ns, [r["t_mix"] for r in rows], "o-", label=r"exact $t_{mix}(1/4)$")
axes[1].loglog(ns, c_star * ns**2, "k--", label=rf"$c^* n^2$, $c^*={c_star:.4f}$ (from limit)")
axes[1].loglog(ns, [r["t_rel"] for r in rows], "s:", label=r"$t_{rel} \approx n^2/\pi^2$")
axes[1].set_xlabel("n")
axes[1].set_ylabel("steps")
axes[1].set_title(r"$t_{mix}/t_{rel}$ stays bounded $\Rightarrow$ no cutoff")
axes[1].legend(fontsize=8)
save(fig, "e03_cycle_no_cutoff")

slope = np.polyfit(np.log(ns[2:]), np.log([r["t_mix"] for r in rows][2:]), 1)[0]
dump("e03_cycle_no_cutoff", {"c_star": c_star, "loglog_slope": slope, "rows": rows})
print(f"  c* = {c_star:.5f}   fitted exponent = {slope:.4f}")
for r in rows:
    print(f"  n={r['n']:5d}  t_mix/n^2={r['t_mix_over_n2']:.5f}  t_mix/t_rel={r['t_mix_over_t_rel']:.3f}  "
          f"sup|TV - limit|={r['sup_gap_to_limit']:.4f}")
