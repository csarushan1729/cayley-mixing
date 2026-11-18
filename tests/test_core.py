"""Every fast/exact method is checked against a brute-force computation."""
import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from cayleymix import additive, affine, distances, lumped, shuffles, spectral
from cayleymix.groups import (
    AbelianGroup, SymmetricGroup, cyclic, hypercube, standard_generators, torus, transposition,
)
from cayleymix.walk import CayleyWalk, uniform_measure

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- groups/walk
def test_group_indexing_roundtrip():
    G = AbelianGroup((3, 4, 5))
    for i in range(G.order):
        assert G.index(G.element(i)) == i
    S = SymmetricGroup(5)
    assert np.array_equal(S.rank(S.perms), np.arange(120))


def test_transition_matrix_is_doubly_stochastic_and_symmetric():
    G = torus(5, 2)
    W = CayleyWalk(G, uniform_measure(standard_generators(G)), laziness=0.3)
    P = W.transition_matrix(sparse=False)
    assert np.allclose(P.sum(axis=1), 1) and np.allclose(P.sum(axis=0), 1)
    assert np.allclose(P, P.T)  # symmetric generating set -> reversible w.r.t. uniform


def test_step_matches_matrix():
    G = AbelianGroup((4, 6))
    W = CayleyWalk(G, {(1, 0): 0.2, (0, 1): 0.5, (3, 5): 0.3}, laziness=0.1)
    p = RNG.random(G.order)
    p /= p.sum()
    assert np.allclose(W.step(p), p @ W.transition_matrix(sparse=False))


# ---------------------------------------------------------------- spectral
def test_cycle_eigenvalues_closed_form():
    n = 11
    G = cyclic(n)
    eigs = spectral.abelian_eigenvalues(G, uniform_measure([(1,), (-1,)]))
    assert np.allclose(eigs, np.cos(2 * np.pi * np.arange(n) / n))


def test_character_eigenvalues_match_dense_spectrum_nonsymmetric():
    G = AbelianGroup((5, 3))
    meas = {(1, 0): 0.5, (0, 1): 0.3, (2, 2): 0.2}
    W = CayleyWalk(G, meas, laziness=0.2)
    fast = np.sort_complex(np.round(spectral.abelian_eigenvalues(G, meas, 0.2), 10))
    slow = np.sort_complex(np.round(np.linalg.eigvals(W.transition_matrix(sparse=False)), 10))
    assert np.allclose(fast, slow)


def test_characters_are_eigenfunctions():
    G = AbelianGroup((4, 5))
    meas = {(1, 2): 1.0, (3, 0): 2.0}
    W = CayleyWalk(G, meas)
    P = W.transition_matrix(sparse=False)
    lam = spectral.abelian_eigenvalues(G, meas)
    for r in range(G.order):
        chi = np.array([np.exp(2j * np.pi * sum(a * b / m for a, b, m in zip(G.element(r), G.element(x), G.moduli)))
                        for x in range(G.order)])
        assert np.allclose(P @ chi, lam[r] * chi)


def test_bipartite_and_connectivity_spectral_facts():
    even = spectral.abelian_eigenvalues(cyclic(8), uniform_measure([(1,), (-1,)]))
    assert np.isclose(np.min(even.real), -1)            # bipartite <=> -1 eigenvalue
    disconnected = spectral.abelian_eigenvalues(cyclic(8), uniform_measure([(2,), (-2,)]))
    assert np.sum(np.isclose(disconnected, 1)) == 2      # disconnected <=> 1 repeated


def test_fft_distribution_matches_direct_evolution():
    G = AbelianGroup((7, 4))
    meas = uniform_measure(standard_generators(G))
    W = CayleyWalk(G, meas, laziness=0.25)
    for t in (0, 1, 5, 17):
        assert np.allclose(W.distribution(t), spectral.abelian_distribution(G, meas, t, 0.25), atol=1e-12)


def test_chi2_identity_is_exact_and_bounds_hold():
    G = torus(6, 2)
    meas = uniform_measure(standard_generators(G))
    W = CayleyWalk(G, meas, laziness=0.5)
    eigs = spectral.abelian_eigenvalues(G, meas, 0.5)
    p = W.point_mass()
    for t in range(40):
        assert np.isclose(distances.chi2(p), spectral.chi2_exact(eigs, t), rtol=1e-9, atol=1e-12)
        tvt = distances.tv(p)
        assert tvt <= spectral.tv_upper_bound_lemma(eigs, t) + 1e-12
        assert tvt <= spectral.tv_beta_bound(eigs, t) + 1e-12
        p = W.step(p)


def test_mixing_time_fft_matches_direct():
    G = cyclic(31)
    meas = uniform_measure([(1,), (-1,)])
    W = CayleyWalk(G, meas, laziness=0.5)
    assert spectral.mixing_time_fft(G, meas, 0.25, 0.5) == W.mixing_time(0.25)


# ---------------------------------------------------------------- lumping
@pytest.mark.parametrize("hold", [0.5, None])
def test_hypercube_lumping_matches_full_walk(hold):
    d = 7
    G = hypercube(d)
    gens = standard_generators(G)
    if hold is None:  # uniform on {0, e_1..e_d}
        hold = 1 / (d + 1)
    W = CayleyWalk(G, uniform_measure(gens), laziness=hold)
    assert np.allclose(W.profile(30), lumped.hypercube_profile(d, 30, hold))


def test_partitions_count():
    assert [len(lumped.partitions(n)) for n in (1, 5, 10, 20)] == [1, 7, 42, 627]


@pytest.mark.parametrize("n", [4, 5, 6])
def test_transposition_chain_matches_full_symmetric_group(n):
    G = SymmetricGroup(n)
    meas = {tuple(range(n)): float(n)}  # i == j  (n of the n^2 ordered pairs)
    for i, j in itertools.permutations(range(n), 2):
        t = transposition(n, i, j)
        meas[t] = meas.get(t, 0) + 1.0
    W = CayleyWalk(G, meas)
    chain = lumped.TranspositionChain(n)
    assert np.allclose(chain.P.sum(axis=1), 1)
    assert np.isclose(chain.pi.sum(), 1)
    assert np.allclose(chain.P.T @ chain.pi, chain.pi)
    assert np.allclose(W.profile(25), chain.profile(25))


# ---------------------------------------------------------------- shuffles
def test_eulerian_numbers():
    for n in range(1, 10):
        assert sum(shuffles.eulerian_row(n)) == math.factorial(n)
    assert shuffles.eulerian_row(4) == (1, 11, 11, 1)


def test_rising_laws_are_probabilities():
    for k in range(0, 8):
        assert sum(shuffles.a_shuffle_rising_law(13, 2**k)) == 1


@pytest.mark.parametrize("n", [3, 4, 5])
def test_bayer_diaconis_formula_matches_bruteforce_convolution(n):
    G = SymmetricGroup(n)
    W = CayleyWalk(G, shuffles.gsr_measure_bruteforce(n))
    prof = W.profile(6)
    for k in range(7):
        assert np.isclose(prof[k], float(shuffles.riffle_tv_exact(n, k)), atol=1e-12)


def test_known_52_card_values():
    # Bayer & Diaconis (1992), Table 1.
    table = {5: 0.924, 6: 0.614, 7: 0.334, 8: 0.167, 9: 0.085, 10: 0.043}
    for k, v in table.items():
        assert round(float(shuffles.riffle_tv_exact(52, k)), 3) == v


def test_rising_sequence_simulation_agrees_with_exact():
    n, k = 20, 3
    sample = shuffles.simulate_rising_sequences(n, k, 60000, RNG)
    exact = np.array([float(x) for x in shuffles.a_shuffle_rising_law(n, 2**k)])
    mean_exact = float(np.dot(np.arange(1, n + 1), exact))
    assert abs(sample.mean() - mean_exact) < 0.03


def test_top_card_law_against_bruteforce():
    n = 5
    G = SymmetricGroup(n)
    W = CayleyWalk(G, shuffles.gsr_measure_bruteforce(n))
    for k in range(1, 4):
        p = W.distribution(k)
        # elements are decks (deck[i] = card at position i); find card 0
        law = np.zeros(n)
        for idx, mass in enumerate(p):
            law[G.element(idx).index(0)] += mass
        assert np.allclose(law, shuffles.top_card_position_law(n, k))


# ---------------------------------------------------------------- affine
def test_affine_a1_is_cycle_walk():
    p = 23
    proc = affine.AffineProcess(p, 1)
    meas = uniform_measure([(-1,), (0,), (1,)])
    for t in (1, 4, 30):
        q = np.zeros(p)
        q[0] = 1
        for _ in range(t):
            q = proc.step(q)
        assert np.allclose(q, spectral.abelian_distribution(cyclic(p), meas, t))


def test_affine_worst_start_equals_zero_start_for_a1():
    proc = affine.AffineProcess(17, 1)
    assert proc.mixing_time() == proc.worst_start_mixing_time()


# ---------------------------------------------------------------- additive
def test_energy_fft_matches_bruteforce():
    N = 31
    A = [0, 1, 3, 7, 12, 13]
    brute = sum(1 for a, b, c, d in itertools.product(A, repeat=4) if (a + b - c - d) % N == 0)
    assert additive.additive_energy_mod(A, N) == brute


def test_sumsets_and_doubling():
    assert additive.doubling_Z(range(10)) == pytest.approx(19 / 10)
    assert set(additive.sumset_Z([0, 2], [0, 5]).tolist()) == {0, 2, 5, 7}
    assert set(additive.sumset_mod([1, 2], [3], 5).tolist()) == {4, 0}


def test_return_probability_identity():
    N = 101
    S = additive.random_symmetric(N, 10, RNG)
    b = additive.energy_bridge(S, N)
    assert b["p4_direct"] == pytest.approx(b["p4_from_energy"], rel=1e-9)
    assert b["chi2_2"] >= b["chi2_2_lower_bound"] - 1e-9


def test_chi2_equals_return_probability():
    N = 61
    S = additive.symmetric_gap(N, [1, 9], 1)
    meas = uniform_measure([(s,) for s in S])
    for t in (1, 2, 3, 6):
        pt = spectral.abelian_distribution(cyclic(N), meas, t)
        p2t = spectral.abelian_distribution(cyclic(N), meas, 2 * t)
        assert distances.chi2(pt) == pytest.approx(N * p2t[0] - 1, abs=1e-9)


def test_bogolyubov_on_random_and_structured_sets():
    N = 211
    for _ in range(20):
        A = RNG.choice(N, size=int(RNG.integers(15, 80)), replace=False)
        res = additive.bogolyubov_check(A, N)
        assert res["contained"] and res["bound_holds"]
    res = additive.bogolyubov_check(range(30), N)
    assert res["contained"] and res["bound_holds"]


def test_float_riffle_matches_exact_rationals():
    for n in (5, 13, 52, 100):
        for k in range(0, 14):
            assert shuffles.riffle_tv_float(n, k) == pytest.approx(float(shuffles.riffle_tv_exact(n, k)), abs=1e-9)
