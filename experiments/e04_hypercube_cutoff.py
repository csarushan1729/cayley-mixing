"""E4 - Cutoff on the hypercube {0,1}^d, computed exactly for N = 2^4096.

Lazy walk: hold w.p. 1/2, else flip a uniform coordinate.  By lumping onto the
Hamming weight (the Ehrenfest urn, see lumped.py) the TV distance is exact
for any d.

Cutoff phenomenon (Aldous-Diaconis): TV stays near 1 and then drops to 0 in a
window much shorter than the mixing time itself.  Here
t_mix = (1/2) d log d + O(d), window O(d), so on the scale t / ((1/2) d log d)
the curves converge to a step function.

Heuristic limit profile.  At t = (1/2) d log d + c d the mean Hamming weight
sits (sqrt(d)/2) e^{-c} below d/2, i.e. e^{-c} standard deviations, while the
fluctuations are already Gaussian with the stationary variance d/4.  TV
between two normals with that mean shift gives

    TV -> erf( e^{-c} / (2 sqrt 2) ).

We check this against the exact computation.
"""
import math

import numpy as np
from scipy.special import erf

from _common import dump, plt, save
from cayleymix import lumped

fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
rows = []
c_grid = np.linspace(-2.5, 3.5, 200)
for d in [16, 64, 256, 1024, 4096]:
    T = int(0.5 * d * math.log(d) + 4 * d)
    prof = lumped.hypercube_profile(d, T)
    t = np.arange(T + 1)
    t_c = 0.5 * d * math.log(d)
    axes[0].plot(t / t_c, prof, label=f"d = {d}")
    axes[1].plot((t - t_c) / d, prof, label=f"d = {d}")
    t25, t75 = lumped.first_below(prof, 0.25), lumped.first_below(prof, 0.75)
    limit_at = erf(np.exp(-(t - t_c) / d) / (2 * math.sqrt(2)))
    mask = np.abs((t - t_c) / d) <= 2.5
    rows.append({
        "d": d,
        "t_mix_quarter": t25,
        "t_mix_over_half_dlogd": t25 / t_c,
        "window_t25_minus_t75_over_d": (t25 - t75) / d,
        "t_rel": d,
        "t_mix_over_t_rel": t25 / d,
        "sup_err_vs_erf_limit": float(np.max(np.abs(prof[mask] - limit_at[mask]))),
    })
axes[0].set_xlim(0, 2)
axes[0].set_xlabel(r"$t \,/\, (\frac{1}{2} d \log d)$")
axes[0].set_ylabel("TV to uniform")
axes[0].set_title("lazy hypercube: the profile sharpens to a step")
axes[0].legend(fontsize=8)
axes[1].plot(c_grid, erf(np.exp(-c_grid) / (2 * math.sqrt(2))), "k--", lw=2.5,
             label=r"$\mathrm{erf}(e^{-c}/2\sqrt{2})$")
axes[1].set_xlim(-2.5, 3.5)
axes[1].set_xlabel(r"window coordinate $c = (t - \frac{1}{2} d \log d)/d$")
axes[1].set_title("inside the cutoff window: convergence to a limit shape")
axes[1].legend(fontsize=8)
save(fig, "e04_hypercube_cutoff")
dump("e04_hypercube_cutoff", rows)
for r in rows:
    print(f"  d={r['d']:5d} t_mix={r['t_mix_quarter']:6d}  ratio to (1/2)d log d={r['t_mix_over_half_dlogd']:.3f}  "
          f"window/d={r['window_t25_minus_t75_over_d']:.3f}  t_mix/t_rel={r['t_mix_over_t_rel']:.2f}  "
          f"sup|TV-erf limit|={r['sup_err_vs_erf_limit']:.4f}")
