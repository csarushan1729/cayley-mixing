"""E7 - From the LCG to the Chung-Diaconis-Graham process.

X_{t+1} = a X_t + eps_t (mod p), eps uniform on {-1, 0, 1}.

Findings reproduced here:
1. a = 1 (a Cayley walk on Z/p): t_mix ~ p^2.  a = 2: t_mix(1/4) ~ log2 p.
   Doubling before each kick is exponentially faster.
2. The *modulus matters*.  For p = 2^m - 1 (Mersenne primes) multiplication by
   2 is a cyclic rotation of m-bit strings, 2^m = 1 (mod p).  The Fourier
   transform of the law after t steps is

       p_hat_t(r) = prod_{i<t} phi(2^i r / p),   phi(x) = (1 + 2 cos 2 pi x) / 3,

   and for r = 1 the angles 2^i / p repeat with period m.  One full period
   multiplies |p_hat(1)| by  c_m = prod_{i<m} |phi(2^i / p)|  ->  c_inf ~ 0.07,
   so the tail of TV decays only by a constant factor every log2 p steps.
   For generic p the orbit {2^i r mod p} does not cycle quickly and the decay
   is super-exponential once t > log2 p.  At TV <= 1/4 the two look alike; at
   TV <= 10^-3 Mersenne moduli need about twice as many steps, and the gap
   grows with log2 p (Hildebrand's lower bound is c log p log log p).
3. The real culprit is the multiplicative order of 2 mod p.  Any prime with
   a short orbit shows the same heavy tail: the Fermat primes 257 and 65537
   (ord_p(2) = 2 log2 p) and 73, which divides 2^9 - 1 (ord_73(2) = 9).
"""
import math

import numpy as np

from _common import dump, plt, save
from cayleymix import spectral
from cayleymix.additive import is_prime
from cayleymix.affine import AffineProcess
from cayleymix.groups import cyclic
from cayleymix.walk import uniform_measure

EPS_LIST = [0.25, 0.05, 0.01, 0.001]
mersenne = [p for p in (2**m - 1 for m in range(3, 20)) if is_prime(p)]
generic = []
for m in range(5, 20):
    q, found = 2**m + 1, 0
    while found < 3:
        if is_prime(q):
            generic.append(q)
            found += 1
        q += 2

def order_of_2(p: int) -> int:
    k, x = 1, 2 % p
    while x != 1:
        x = 2 * x % p
        k += 1
    return k


rows = []
for p in sorted(set(mersenne + generic)):
    if p < 31:
        continue
    proc = AffineProcess(p, 2)
    ordp = order_of_2(p)
    kind = "Mersenne" if p in mersenne else ("short-orbit" if ordp <= 2 * math.log2(p) + 1 else "generic")
    rows.append({"p": p, "log2p": math.log2(p), "mersenne": p in mersenne, "kind": kind, "ord2": ordp,
                 **{f"t_{e}": proc.mixing_time(e) for e in EPS_LIST}})

# a = 1 for comparison (Cayley walk; FFT bisection)
a1 = []
for p in [31, 127, 521, 2053, 8191, 32771, 131071]:
    a1.append({"p": p, "t_mix": spectral.mixing_time_fft(cyclic(p), uniform_measure([(-1,), (0,), (1,)]), 0.25)})

# worst-start sanity check on small primes
worst = [{"p": p, "start0": AffineProcess(p, 2).mixing_time(), "worst": AffineProcess(p, 2).worst_start_mixing_time()}
         for p in (31, 127, 131, 257, 509, 521)]

# Fourier staircase for one Mersenne / one generic prime of the same size
def max_char(p: int, T: int) -> np.ndarray:
    proc = AffineProcess(p, 2)
    q = np.zeros(p)
    q[0] = 1
    out = []
    for _ in range(T + 1):
        out.append(np.max(np.abs(np.fft.fft(q)[1:])))
        q = proc.step(q)
    return np.array(out)


def c_period(m: int) -> float:
    p = 2**m - 1
    x = 2.0 ** np.arange(m) / p
    return float(np.prod(np.abs((1 + 2 * np.cos(2 * np.pi * x)) / 3)))


c_inf = c_period(60)
stair_m, stair_g = max_char(8191, 70), max_char(8209, 70)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
for e, ls in [(0.25, ":"), (0.001, "-")]:
    for kind, mk, col in [("generic", "o", "C0"), ("Mersenne", "s", "C3"), ("short-orbit", "D", "C2")]:
        sel = [r for r in rows if r["kind"] == kind]
        axes[0].plot([r["log2p"] for r in sel], [r[f"t_{e}"] for r in sel], ls if kind != "short-orbit" else "none",
                     marker=mk, ms=5, color=col, mfc=col if e == 0.001 else "white",
                     label=f"{kind} p, TV<={e}")
axes[0].set_xlabel(r"$\log_2 p$")
axes[0].set_ylabel(r"steps to TV $\leq \varepsilon$ (start 0)")
axes[0].set_title(r"$X \to 2X+\varepsilon$: short orbits of $\times 2$ give heavy tails")
axes[0].legend(fontsize=7, ncol=2)

axes[1].semilogy(stair_m, label="p = 8191 = 2^13 - 1")
axes[1].semilogy(stair_g, label="p = 8209 (generic)")
for j in range(1, 6):
    axes[1].axvline(13 * j, color="gray", lw=0.5)
axes[1].set_ylim(1e-17, 2)
axes[1].set_xlabel("step t")
axes[1].set_ylabel(r"$\max_{r\neq0}|\hat p_t(r)|$")
axes[1].set_title(rf"Fourier staircase: factor $\approx${c_inf:.3f} per $m=13$ steps")
axes[1].legend(fontsize=8)

pa = np.array([r["p"] for r in a1], float)
axes[2].loglog(pa, [r["t_mix"] for r in a1], "o-", label="a = 1 (walk on the cycle)")
sel = [r for r in rows if r["kind"] == "generic"]
axes[2].loglog([r["p"] for r in sel], [r["t_0.25"] for r in sel], "s-", label="a = 2")
axes[2].loglog(pa, 0.14 * pa**2, "k--", lw=0.8, label=r"$\propto p^2$")
axes[2].set_xlabel("p")
axes[2].set_ylabel(r"$t_{mix}(1/4)$")
axes[2].set_title("multiplying before the kick: $p^2 \\to \\log p$")
axes[2].legend(fontsize=8)
save(fig, "e07_affine_lcg")
dump("e07_affine_lcg", {"rows": rows, "a1": a1, "worst_start_check": worst, "c_inf": c_inf,
                         "c_period": {m: c_period(m) for m in (5, 7, 13, 17, 19)}})

print(f"  c_inf = {c_inf:.4f}")
print("  worst-start check:", [(w["p"], w["start0"], w["worst"]) for w in worst])
print("  p        type   " + "  ".join(f"t({e})" for e in EPS_LIST))
for r in rows:
    if r["kind"] != "generic" or r["p"] in (generic[::3]):
        print(f"  {r['p']:7d}  {r['kind']:8s} ord2={r['ord2']:7d}  " + "  ".join(f"{r[f't_{e}']:6d}" for e in EPS_LIST))
print("  a=1:", [(r["p"], r["t_mix"]) for r in a1])
