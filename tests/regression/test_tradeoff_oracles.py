"""Exact-mode trade-off elicitation against three independent oracles, and the truthful invariant.

1. **Order constraints only** reproduce SMAA-2's ``ranking`` kind: starting from the uniform
   (unknown) prior, answers ``w_a >= w_b`` that spell out each group's order must give the
   distribution of the engine's own ranking sampler (sorted normalised exponentials), on a flat
   four-leaf tree and on case 2's nested tree (16 varying leaves). Two-sample KS on every
   marginal.
2. **No constraints** reproduce the existing unknown path: prior draws moved by the rejuvenation
   kernel (no answer to filter by) must keep SMAA-2's Dirichlet(1, ..., 1) distribution on case 2.
3. **Brute-force rejection** on a four-leaf toy: a Dirichlet(10 * w0) prior restricted by three
   answers, against 10^6 prior draws kept when they satisfy every answer. KS on every marginal.

Invariant: with a truthful exact respondent whose weights are a prior draw, no answer ever leaves
the particles empty, the planted weights satisfy every kept answer, and every particle satisfies
every kept answer, on every answer of every run.

Every seed below derives from the base seed 2026112811, unused anywhere before this test was
written; they were fixed before the first run. KS at alpha = 0.001 per marginal, as the ranking
reference tests use. Particles after resampling are correlated, which makes KS stricter than its
nominal level, not looser. Design: docs/design/tradeoff-elicitation.md.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.stats import ks_2samp

from mcda_studio.domain.preferences import AnalysisConfiguration
from mcda_studio.domain.uncertainty import GroupUncertainty, SmaaModel
from mcda_studio.methods import tradeoff as tm
from mcda_studio.methods.smaa import engine
from mcda_studio.services import project_service as ps
from mcda_studio.services import tradeoff_service as ts
from mcda_studio.storage.json_repository import JsonProjectRepository

from builders import flat_project

BASE_SEED = 2026112811
SEED_RANKING_FLAT = BASE_SEED + 1
SEED_RANKING_NESTED = BASE_SEED + 2
SEED_UNKNOWN = BASE_SEED + 3
SEED_TOY = BASE_SEED + 4
SEED_TOY_REJECTION = BASE_SEED + 5
SEED_REFERENCE = BASE_SEED + 6  # the engine's own uniforms for the reference draws
SEED_INVARIANT = BASE_SEED + 100  # runs use SEED_INVARIANT + run
ALPHA = 0.001
STUDY = Path(__file__).parents[2] / "examples" / "nz_venture_screening" / "project.json"
PREFERENCE = AnalysisConfiguration(method="smaa2", parameters={"uncertainty": "preference"})


def engine_global(project, n: int, seed: int) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Global leaf weights from SMAA-2's own group sampler, ``(n, leaves)``."""
    plan = engine._plan(project, PREFERENCE, "preference")
    width = sum(len(g.varying) for g in plan.groups)
    uniforms = np.random.Generator(np.random.PCG64(seed)).random((n, width))
    local, used = engine._group_locals(plan, uniforms)
    assert used == width
    return engine._weights(plan, local, n)


def assert_marginals_match(got: np.ndarray, want: np.ndarray, columns: list[int]) -> None:
    worst = 1.0
    for j in columns:
        p = ks_2samp(got[:, j], want[:, j]).pvalue
        worst = min(worst, p)
        assert p > ALPHA, f"leaf column {j}: KS p = {p:.2e}"
    assert worst > ALPHA


def order_contrasts(prior: tm.ConstrainedPrior, orders: dict[str | None, list[str]]) -> np.ndarray:
    """``e_a - e_b`` for each adjacent pair of each group's order, on the global leaves of the
    subtree under each child (the sum of a child's leaves is its share of the parent)."""
    leaves = prior.leaf_ids
    rows = []
    for order in orders.values():
        for upper, lower in zip(order, order[1:], strict=False):
            row = np.zeros(len(leaves))
            for j, leaf in enumerate(leaves):
                if upper in prior.paths[leaf]:
                    row[j] += 1.0
                if lower in prior.paths[leaf]:
                    row[j] -= 1.0
            rows.append(row)
    return np.array(rows)


# ---- 1. order constraints reproduce the ranking kind ------------------------------------------


def test_order_answers_on_a_flat_tree_reproduce_the_ranking_kind() -> None:
    order = ["c2", "c0", "c3", "c1"]
    values = [[0.1 * (i + j) % 1 for j in range(4)] for i in range(3)]
    base = flat_project(values, [10.0, 20.0, 30.0, 40.0])
    ctx = ts.context(base, None, "unknown")
    contrasts = order_contrasts(ctx.prior, {None: order})
    result = tm.smc_fit(ctx.prior, contrasts, tm.SmcSettings(seed=SEED_RANKING_FLAT))
    assert result.conflicts == []
    ranked = ps.set_smaa_model(
        base, SmaaModel(groups=[GroupUncertainty(parent_id=None, kind="ranking", order=order)])
    )
    want = engine_global(ranked, 200_000, SEED_REFERENCE)
    assert_marginals_match(result.weights, want, list(range(4)))


def test_order_answers_on_case_2_reproduce_the_ranking_kind_in_every_group() -> None:
    project = JsonProjectRepository().load(STUDY)
    ctx = ts.context(project, "technical_ip", "unknown")
    orders = {g.parent_id: list(g.children) for g in ctx.prior.groups}
    # Reverse the stated order in two groups so the answers are not the swing order.
    orders[None] = ["traction", "market", "team", "problem_solution"]
    orders["team"] = list(reversed(orders["team"]))
    contrasts = order_contrasts(ctx.prior, orders)
    result = tm.smc_fit(ctx.prior, contrasts, tm.SmcSettings(seed=SEED_RANKING_NESTED))
    assert result.conflicts == []
    groups = [
        GroupUncertainty(
            parent_id=parent,
            kind="ranking",
            order=order,
            held=["mandate"] if parent is None else [],
        )
        for parent, order in orders.items()
    ]
    ranked = project.model_copy(
        update={
            "smaa": project.smaa.model_copy(update={"groups": groups}),  # type: ignore[union-attr]
            "analysis": project.analysis.model_copy(update={"weight_profile_id": "technical_ip"}),
        }
    )
    want = engine_global(ranked, 200_000, SEED_REFERENCE)
    varying = [j for j in range(len(ctx.prior.leaf_ids)) if ctx.prior.varying[j]]
    assert len(varying) == 16
    assert_marginals_match(result.weights, want, varying)


# ---- 2. no constraints reproduce the unknown path ---------------------------------------------


def test_moves_without_answers_keep_the_unknown_dirichlet_on_case_2() -> None:
    project = JsonProjectRepository().load(STUDY)
    ctx = ts.context(project, "technical_ip", "unknown")
    state = tm.SmcState.start(ctx.prior, tm.SmcSettings(seed=SEED_UNKNOWN))
    for _ in range(3):  # as many sweeps as three answers would give, with nothing to filter
        state.move(tm.SmcSettings().moves)
    groups = [
        GroupUncertainty(parent_id=g.parent_id, kind="unknown", held=list(g.held))
        for g in ctx.prior.groups
    ]
    unknown = project.model_copy(
        update={
            "smaa": project.smaa.model_copy(update={"groups": groups}),  # type: ignore[union-attr]
            "analysis": project.analysis.model_copy(update={"weight_profile_id": "technical_ip"}),
        }
    )
    want = engine_global(unknown, 200_000, SEED_REFERENCE)
    varying = [j for j in range(len(ctx.prior.leaf_ids)) if ctx.prior.varying[j]]
    assert_marginals_match(state.weights, want, varying)


# ---- 3. brute-force rejection on a four-leaf toy ----------------------------------------------


def toy() -> tuple[tm.ConstrainedPrior, np.ndarray, np.ndarray]:
    alpha = 10.0 * np.array([0.4, 0.3, 0.2, 0.1])
    leaves = ["l0", "l1", "l2", "l3"]
    prior = tm.ConstrainedPrior(
        leaves, leaves, {leaf: [leaf] for leaf in leaves}, dict.fromkeys(leaves, 0.25),
        [tm.LatentGroup(None, "dirichlet", tuple(leaves), 1.0, (), tuple(alpha))],
    )  # fmt: skip
    # Three answers: l3 at full swing beats l1 at 40% of its swing; l2 beats l0 at half its swing;
    # l1 + l2 together beat l0 at 90%.
    contrasts = np.array([[0.0, -0.4, 0.0, 1.0], [-0.5, 0.0, 1.0, 0.0], [-0.9, 1.0, 1.0, 0.0]])
    return prior, alpha, contrasts


def test_exact_mode_matches_brute_force_rejection_on_a_four_leaf_toy() -> None:
    prior, alpha, contrasts = toy()
    result = tm.smc_fit(prior, contrasts, tm.SmcSettings(seed=SEED_TOY))
    assert result.conflicts == []
    rng = np.random.Generator(np.random.PCG64(SEED_TOY_REJECTION))
    draws = rng.dirichlet(alpha, size=1_000_000)
    kept = draws[np.all(draws @ contrasts.T >= 0.0, axis=1)]
    assert 10_000 < kept.shape[0] < 500_000  # a region neither trivial nor empty
    assert_marginals_match(result.weights, kept, list(range(4)))
    # Means within six standard errors of the rejection mean (the SMC's own error included at
    # the nominal particle count, which understates it).
    se = np.sqrt(kept.var(axis=0) / kept.shape[0] + kept.var(axis=0) / result.weights.shape[0])
    assert np.all(np.abs(result.weights.mean(axis=0) - kept.mean(axis=0)) < 6.0 * se)


# ---- the truthful invariant -------------------------------------------------------------------


@pytest.mark.parametrize("run", range(8))
def test_a_truthful_respondent_never_leaves_the_region(run: int) -> None:
    project = JsonProjectRepository().load(STUDY)
    ctx = ts.context(project, "technical_ip", "ranking")
    seed = SEED_INVARIANT + run
    planted = ctx.prior.global_weights(
        ctx.prior.sample(np.random.Generator(np.random.PCG64(seed).jumped(5)), 1)
    )[0]
    settings = tm.SmcSettings(particles=2000, seed=seed)
    state = tm.SmcState.start(ctx.prior, settings)
    pool = ts.candidates(ctx)
    contrasts = np.array([c.contrast for c in pool])
    asked: set[int] = set()
    for _ in range(tm.MAX_QUESTIONS):
        probs = tm.answer_probabilities(state.weights, contrasts)
        pick = tm.select_question(probs, asked)
        if pick is None:
            break
        asked.add(pick)
        answer = contrasts[pick] if planted @ contrasts[pick] >= 0.0 else -contrasts[pick]
        assert state.add(answer, settings.moves) > 0  # never empty
        assert np.all(state.constraints @ planted >= 0.0)  # the truth satisfies every answer
        assert np.all(state.weights @ state.constraints.T >= 0.0)  # so does every particle
