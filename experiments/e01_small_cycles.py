"""E1 - Two small cycles, computed exactly: periodicity and the lazy fix.

Take the walk on the 5-cycle and the 6-cycle.  The convergence theorem needs
"connected and non-bipartite": the 5-cycle qualifies, the 6-cycle does not
(it is bipartite, -1 is an eigenvalue) and the walk never converges - the mass
alternates between even and odd vertices forever, so TV is stuck at 1/2.
Adding laziness (hold with probability 1/2) shifts the spectrum into [0, 1]
and restores convergence.  We also check the beta-bound on both.
"""
import numpy as np

from _common import dump, plt, save
from cayleymix import distances, spectral
from cayleymix.groups import cyclic
from cayleymix.walk import CayleyWalk, uniform_measure

T = 24
cases = [("5-cycle", 5, 0.0), ("6-cycle", 6, 0.0), ("6-cycle, lazy", 6, 0.5)]
fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), gridspec_kw={"height_ratios": [1.25, 1]})
summary = {}
for col, (name, n, lazy) in enumerate(cases):
    G = cyclic(n)
    meas = uniform_measure([(1,), (-1,)])
    W = CayleyWalk(G, meas, laziness=lazy)
    dist = np.array([W.distribution(t) for t in range(T + 1)])
    eigs = spectral.abelian_eigenvalues(G, meas, lazy).real
    tvs = np.array([distances.tv(p) for p in dist])
    bb = np.array([spectral.tv_beta_bound(eigs, t) for t in range(T + 1)])

    ax = axes[0, col]
    im = ax.imshow(dist.T, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=0.6,
                   extent=[-0.5, T + 0.5, -0.5, n - 0.5])
    ax.set_title(f"{name}: P(X_t = x), start 0")
    ax.set_xlabel("step t")
    ax.set_ylabel("vertex x")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = axes[1, col]
    ax.semilogy(tvs, "o-", ms=3, label="exact TV")
    ax.semilogy(np.minimum(bb, 1.5), "--", label=r"$\beta$-bound: $\sqrt{n}\beta^t/2$")
    ax.set_ylim(1e-6, 2)
    ax.set_xlabel("step t")
    ax.set_title(f"spectrum: {np.round(np.sort(eigs)[::-1], 3).tolist()}", fontsize=8)
    ax.legend(fontsize=8)
    summary[name] = {
        "eigenvalues": sorted(eigs.tolist(), reverse=True),
        "beta": spectral.second_eigenvalue_modulus(eigs),
        "tv_t10": float(tvs[10]),
        "tv_t24": float(tvs[24]),
        "bipartite": bool(np.isclose(eigs.min(), -1)),
        "t_mix_quarter": int(np.argmax(tvs <= 0.25)) if (tvs <= 0.25).any() else None,
    }
save(fig, "e01_small_cycles")
dump("e01_small_cycles", summary)
for k, v in summary.items():
    print(f"  {k:14s} beta={v['beta']:.3f} bipartite={v['bipartite']} TV(24)={v['tv_t24']:.2e} t_mix={v['t_mix_quarter']}")
