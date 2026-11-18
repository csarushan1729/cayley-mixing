"""E8 - Additive structure slows mixing: linking Freiman to random walks.

Random walks and Freiman-type additive combinatorics are usually taught separately.  They meet in one
identity.  For a symmetric set S = -S in Z/N and the walk with uniform steps
from S:

    P(X_4 = 0) = E(S) / |S|^4,          chi2(t) = N P(X_{2t} = 0) - 1,

so  chi2(2) = N E(S)/|S|^4 - 1 >= N/(K|S|) - 1  with K the doubling constant.
Small doubling (the hypothesis of Freiman's theorem) forces large additive
energy, which forces the walk to keep returning to its start.

Part (a) checks the identity and the bound on 400 sets.
Part (b) asks whether the effect persists all the way to t_mix: sets of the
same size (~24) in Z/10007 ranging from an arithmetic progression (K ~ 2) to a
random set (K ~ 12) are compared on doubling, spectral gap and exact t_mix.
Part (d) explains the outliers in (b): what controls the mixing time is the
*dimension* of the progression S lives in (Freiman's d(K)), not K alone.  A
d-dimensional GAP with generic steps behaves like a walk on a d-dimensional
torus of side ~ p^{1/d}, so t_mix ~ p^{2/d}; we fit the exponent.
Part (c) verifies Bogolyubov's lemma (2A - 2A contains Bohr(R, 1/4), with
|R| < 1/alpha^2), the Fourier engine of the proof of Freiman's theorem, on
1,200 sets.
"""
import math

import numpy as np

from _common import dump, plt, save
from cayleymix import additive, spectral
from cayleymix.groups import cyclic
from cayleymix.walk import uniform_measure

RNG = np.random.default_rng(678)
P = 10007
K_SIZE = 24


def families(rng):
    """(name, set) pairs of symmetric sets of size ~24 in Z/P."""
    out = [("AP", additive.symmetrise(range(1, K_SIZE // 2 + 1), P))]
    for spread in (2, 4, 8, 32, 128):
        out.append((f"sparse AP x{spread}", additive.sparse_ap_subset(P, K_SIZE, spread, rng)))
    steps2 = rng.integers(1, P, size=2)
    out.append(("GAP d=2", additive.symmetric_gap(P, steps2, 2)))
    steps3 = rng.integers(1, P, size=3)
    out.append(("GAP d=3", additive.symmetric_gap(P, steps3, 1)))
    out.append(("random", additive.random_symmetric(P, K_SIZE, rng)))
    return out


# (a) + (b) ------------------------------------------------------------------
rows = []
for seed in range(12):
    rng = np.random.default_rng(seed)
    for name, S in families(rng):
        b = additive.energy_bridge(S, P)
        meas = uniform_measure([(s,) for s in S])
        eigs = spectral.abelian_eigenvalues(cyclic(P), meas).real
        rows.append({
            "family": name, "seed": seed, **b,
            "gap": spectral.spectral_gap(eigs),
            "t_mix": spectral.mixing_time_fft(cyclic(P), meas, 0.25),
        })
ident_err = max(abs(r["p4_direct"] - r["p4_from_energy"]) / r["p4_from_energy"] for r in rows)
bound_ok = all(r["chi2_2"] >= r["chi2_2_lower_bound"] - 1e-9 for r in rows)

# extra identity checks on varied sizes
extra = 0.0
for _ in range(300):
    k = int(RNG.integers(2, 40)) * 2
    S = additive.random_symmetric(P, k, RNG) if RNG.random() < 0.5 else additive.sparse_ap_subset(P, k, int(RNG.integers(1, 20)), RNG)
    b = additive.energy_bridge(S, P)
    extra = max(extra, abs(b["p4_direct"] - b["p4_from_energy"]) / b["p4_from_energy"])
    bound_ok &= b["chi2_2"] >= b["chi2_2_lower_bound"] - 1e-9
ident_err = max(ident_err, extra)

K = np.array([r["doubling"] for r in rows])
T = np.array([r["t_mix"] for r in rows])
rho = float(np.corrcoef(np.argsort(np.argsort(K)), np.argsort(np.argsort(T)))[0, 1])  # Spearman
slope = float(np.polyfit(np.log(K), np.log(T), 1)[0])

# (c) Bogolyubov ---------------------------------------------------------------
bog = []
for N in (101, 211, 1009, 4001):
    if not additive.is_prime(N):
        continue
    for _ in range(300):
        kind = RNG.choice(["random", "interval", "sparse interval", "bohr"])
        alpha = float(RNG.uniform(0.03, 0.45))
        size = max(3, int(alpha * N))
        if kind == "random":
            A = RNG.choice(N, size=size, replace=False)
        elif kind == "interval":
            A = np.arange(size)
        elif kind == "sparse interval":
            A = RNG.choice(min(N, 3 * size), size=size, replace=False)
        else:
            R0 = RNG.integers(1, N, size=int(RNG.integers(1, 3)))
            A = additive.bohr_set(R0, N, float(RNG.uniform(0.05, 0.3)))
            if len(A) < 3:
                continue
        res = additive.bogolyubov_check(A, N)
        res["kind"] = str(kind)
        bog.append(res)
all_contained = all(r["contained"] for r in bog)
all_bound = all(r["bound_holds"] for r in bog)

# (d) dimension of the GAP sets the exponent: t_mix ~ p^{2/d} ---------------------
def next_prime(n: int) -> int:
    while not additive.is_prime(n):
        n += 1
    return n


dim_rows = []
primes = [next_prime(int(x)) for x in np.geomspace(1000, 64000, 7)]
for d, L in [(1, 4), (2, 1), (3, 1)]:
    med = []
    for p in primes:
        ts = []
        for seed in range(5):
            rng = np.random.default_rng(100 * d + seed)
            steps = [1] + list(rng.integers(2, p, size=d - 1))
            S = additive.symmetric_gap(p, steps, L)
            ts.append(spectral.mixing_time_fft(cyclic(p), uniform_measure([(x,) for x in S]), 0.25))
        med.append(float(np.median(ts)))
    expo = float(np.polyfit(np.log(primes), np.log(med), 1)[0])
    dim_rows.append({"d": d, "L": L, "size": (2 * L + 1) ** d - 1, "primes": primes, "median_t_mix": med,
                     "fitted_exponent": expo, "predicted": 2 / d})

# plot -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
fams = list(dict.fromkeys(r["family"] for r in rows))
cmap = plt.get_cmap("viridis", len(fams))
for i, f in enumerate(fams):
    sel = [r for r in rows if r["family"] == f]
    axes[0].scatter([r["chi2_2_lower_bound"] for r in sel], [r["chi2_2"] for r in sel], s=18, color=cmap(i), label=f)
    axes[1].scatter([r["doubling"] for r in sel], [r["t_mix"] for r in sel], s=18, color=cmap(i), label=f)
lims = [min(r["chi2_2_lower_bound"] for r in rows) * 0.8, max(r["chi2_2"] for r in rows) * 1.2]
axes[0].plot(lims, lims, "k--", lw=1, label="bound = value")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel(r"lower bound $N/(K|S|) - 1$")
axes[0].set_ylabel(r"exact $\chi^2$ distance after 2 steps $= N E(S)/|S|^4 - 1$")
axes[0].set_title("energy identity: structure keeps the walk near home")
axes[0].legend(fontsize=6, ncol=2)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("doubling constant K = |S+S|/|S|")
axes[1].set_ylabel(r"exact $t_{mix}(1/4)$ on $\mathbb{Z}/10007$")
axes[1].set_title(rf"same |S|: less structure $\Rightarrow$ faster (Spearman $\rho$={rho:.2f})")
axes[1].legend(fontsize=6, ncol=2)
alphas = np.array([r["alpha"] for r in bog])
specs = np.array([r["spectrum_size"] for r in bog])
kinds = [r["kind"] for r in bog]
for j, kd in enumerate(["random", "interval", "sparse interval", "bohr"]):
    m = np.array([k == kd for k in kinds])
    axes[2].scatter(1 / alphas[m] ** 2, specs[m] + 0.5, s=10, alpha=0.6, label=kd, color=f"C{j}")
xx = np.linspace(1, (1 / alphas**2).max(), 50)
axes[2].plot(xx, xx, "k--", lw=1, label=r"$|R| = 1/\alpha^2$")
axes[2].set_xscale("log")
axes[2].set_yscale("log")
axes[2].set_xlabel(r"$1/\alpha^2$")
axes[2].set_ylabel(r"$|R| + \frac{1}{2}$, large spectrum $\{r: |\hat 1_A(r)| > \alpha^{3/2}\}$")
axes[2].set_title(f"Bogolyubov: Bohr(R,1/4) in 2A-2A for {sum(r['contained'] for r in bog)}/{len(bog)} sets")
axes[2].legend(fontsize=7)
for r in dim_rows:
    axes[3].loglog(r["primes"], r["median_t_mix"], "o-", label=f"GAP dim {r['d']}: slope {r['fitted_exponent']:.2f} (2/d = {r['predicted']:.2f})")
axes[3].set_xlabel("prime p")
axes[3].set_ylabel(r"median $t_{mix}(1/4)$")
axes[3].set_title(r"Freiman's dimension sets the exponent: $t_{mix} \sim p^{2/d}$")
axes[3].legend(fontsize=7)
save(fig, "e08_additive_bridge")

summary = {
    "identity_max_rel_error": ident_err,
    "chi2_bound_always_holds": bool(bound_ok),
    "spearman_doubling_vs_tmix": rho,
    "loglog_slope_tmix_vs_doubling": slope,
    "by_family": {
        f: {
            "doubling_median": float(np.median([r["doubling"] for r in rows if r["family"] == f])),
            "t_mix_median": float(np.median([r["t_mix"] for r in rows if r["family"] == f])),
            "gap_median": float(np.median([r["gap"] for r in rows if r["family"] == f])),
            "size": int(np.median([r["size"] for r in rows if r["family"] == f])),
        } for f in fams
    },
    "bogolyubov": {
        "sets": len(bog), "all_contained": all_contained, "all_spectrum_bounds": all_bound,
        "max_ratio_R_to_bound": float(max(r["spectrum_size"] * r["alpha"] ** 2 for r in bog)),
        "fraction_empty_spectrum": float(np.mean(specs == 0)),
    },
}
summary["gap_dimension_scaling"] = [{k: v for k, v in r.items() if k in ("d", "size", "fitted_exponent", "predicted")} for r in dim_rows]
dump("e08_additive_bridge", {"summary": summary, "rows": rows, "bogolyubov": bog, "dimension": dim_rows})
print(f"  identity max rel error = {ident_err:.2e}; chi2 bound always holds = {bound_ok}")
print(f"  Spearman(doubling, t_mix) = {rho:.3f}; log-log slope = {slope:.2f}")
for f, v in summary["by_family"].items():
    print(f"  {f:16s} |S|={v['size']:3d}  K={v['doubling_median']:6.2f}  gap={v['gap_median']:.2e}  t_mix={v['t_mix_median']:10.0f}")
print(f"  Bogolyubov: {summary['bogolyubov']}")
for r in dim_rows:
    print(f"  GAP dim {r['d']} (|S|={r['size']}): fitted exponent {r['fitted_exponent']:.3f} vs 2/d = {r['predicted']:.3f}")
