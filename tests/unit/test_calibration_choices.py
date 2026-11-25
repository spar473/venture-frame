"""Revealed preference (Layer 2) on synthetic decisions: the choice likelihood against hand-computed
values, the prior returned with no decisions (KS against exact Beta marginals, means within six
standard errors), the convergence gate, refusals, read-only behaviour and the command.

Design: docs/design/revealed-preference.md. Every claim here is about the method on synthetic data.
"""

import json
import math

import numpy as np
import pytest
from scipy import stats
from typer.testing import CliRunner

import reference_case_builders as b
from mcda_studio.calibration import choices as ch
from mcda_studio.calibration import diagnostics as dg
from mcda_studio.calibration import report
from mcda_studio.calibration import synthetic as sy
from mcda_studio.calibration.sampler import SamplerSettings, log_posterior, run
from mcda_studio.cli import app
from mcda_studio.domain.errors import McdaError
from mcda_studio.domain.uncertainty import GroupUncertainty, SmaaModel
from mcda_studio.services import project_service as ps

KAPPA = {None: 8.0, "obj_a": 6.0, "obj_b": 12.0}


def column(*xs: float) -> np.ndarray:
    return np.array(xs, dtype=float)


# ---- the likelihood ----------------------------------------------------------------------------


def tiny() -> ch.ChoiceData:
    """Two leaves; an accepted alternative rating both, a declined one rating only the first."""
    return ch.ChoiceData.build(
        np.array([[1.0, 0.0], [0.5, 0.9]]),
        np.array([[1.0, 1.0], [1.0, 0.0]]),
        np.array([1.0, 0.0]),
        [0, 1],
    )


def test_each_decision_matches_a_hand_computed_value() -> None:
    weights = np.array([[0.25], [0.75]])  # global weights, one chain
    alpha, beta = column(0.4), column(2.0)
    # Accepted: s = 0.25 * 1 + 0.75 * 0 = 0.25, eta = 0.4 + 2 (0.25 - 0.5) = -0.1,
    # log sigmoid(-0.1) = -log(1 + e^0.1).
    # Declined: only the first leaf is rated, so s = 0.5 (missing is not zero: 0.9 is ignored),
    # eta = 0.4, log(1 - sigmoid(0.4)) = -log(1 + e^0.4).
    got = ch.case_log_likelihood(tiny(), weights, alpha, beta)[:, 0]
    assert got[0] == pytest.approx(-0.744396660073571, abs=1e-12)
    assert got[1] == pytest.approx(-0.9130152523999526, abs=1e-12)
    total = ch.log_likelihood(tiny(), weights, alpha, beta)
    assert total[0] == pytest.approx(-0.744396660073571 - 0.9130152523999526, abs=1e-12)


def test_the_intercept_is_the_log_odds_at_the_reference_score() -> None:
    data = ch.ChoiceData.build(np.array([[0.5, 0.5]]), np.ones((1, 2)), np.array([1.0]), [0, 1])
    weights = np.array([[0.3], [0.7]])
    for beta in (0.5, 8.0, 40.0):
        got = ch.case_log_likelihood(data, weights, column(-0.8), column(beta))[0, 0]
        assert got == pytest.approx(math.log(1 / (1 + math.exp(0.8))), abs=1e-12)


def test_accept_and_decline_are_complements() -> None:
    weights = np.array([[0.4], [0.6]])
    values = np.array([[0.2, 0.9]])
    accept = ch.ChoiceData.build(values, np.ones((1, 2)), np.array([1.0]), [0, 1])
    decline = ch.ChoiceData.build(values, np.ones((1, 2)), np.array([0.0]), [0, 1])
    a = ch.case_log_likelihood(accept, weights, column(0.0), column(6.0))[0, 0]
    d = ch.case_log_likelihood(decline, weights, column(0.0), column(6.0))[0, 0]
    assert math.exp(a) + math.exp(d) == pytest.approx(1.0, abs=1e-12)
    assert a > d  # s = 0.62 is above the reference score: acceptance is the likelier call


def test_extreme_scores_stay_finite() -> None:
    data = ch.ChoiceData.build(
        np.array([[1.0], [1.0], [0.0], [0.0]]),
        np.ones((4, 1)),
        np.array([1.0, 0.0, 1.0, 0.0]),
        [0],
    )
    got = ch.case_log_likelihood(data, np.array([[1.0]]), column(0.0), column(2000.0))[:, 0]
    assert np.all(np.isfinite(got))
    assert got[0] == pytest.approx(0.0, abs=1e-300)
    assert got[1] == pytest.approx(-1000.0)
    assert got[2] == pytest.approx(-1000.0)
    assert got[3] == pytest.approx(0.0, abs=1e-300)


def test_no_decisions_give_zero_log_likelihood() -> None:
    assert ch.log_likelihood(
        None, np.ones((1, 3)), column(0.0, 1.0), column(1.0, 2.0)
    ).tolist() == [
        0.0,
        0.0,
    ]


def test_the_data_are_checked() -> None:
    with pytest.raises(ValueError, match="accept"):
        ch.ChoiceData.build(np.ones((1, 1)), np.ones((1, 1)), np.array([0.5]), [0])
    with pytest.raises(ValueError, match="at least one"):
        ch.ChoiceData.build(np.ones((1, 2)), np.zeros((1, 2)), np.array([1.0]), [0, 1])
    with pytest.raises(ValueError, match="same decisions"):
        ch.ChoiceData.build(np.ones((2, 1)), np.ones((2, 1)), np.array([1.0]), [0])
    with pytest.raises(ValueError, match="base accept rate"):
        ch.alpha_prior_mean(1.0)
    assert ch.alpha_prior_mean(0.5) == 0.0
    assert ch.alpha_prior_mean(0.3) == pytest.approx(math.log(0.3 / 0.7))


def test_beta_enters_the_posterior_positive_and_the_priors_are_the_stated_ones() -> None:
    prior = sy.bench_prior(20.0)
    dset = sy.simulate_decisions(prior, sy.planted_choice(), 40, np.random.default_rng(4))
    problem = ch.choice_problem(prior, dset.data, 0.3)
    assert problem.alpha_mean == pytest.approx(math.log(0.3 / 0.7))
    assert problem.alpha_sd == 1.5 and problem.log_beta_sd == 1.0
    assert problem.log_beta_mean == pytest.approx(math.log(8.0))
    rng = np.random.default_rng(0)
    ys = [y[:2] for y in prior.sample_ys(rng, 2)]
    theta = np.array(
        [[-0.5, math.log(9.0)], [0.2, math.log(3.0)]]
    )  # (alpha, log beta), no centring
    got = log_posterior(problem, ys, theta)
    weights = prior.global_weights(ys, 2)
    alpha, beta = theta[:, 0], np.exp(theta[:, 1])
    expected = (
        prior.log_density(ys, 2)
        + stats.norm(problem.alpha_mean, 1.5).logpdf(alpha)
        + stats.norm(math.log(8.0), 1.0).logpdf(theta[:, 1])
        + ch.log_likelihood(dset.data, weights, alpha, beta)
    )
    assert got == pytest.approx(expected, rel=1e-10)


# ---- with no decisions the posterior is the prior ---------------------------------------------


@pytest.fixture(scope="module")
def prior_only():
    prior = sy.bench_prior(KAPPA)
    problem = ch.choice_problem(prior, None, sy.BASE_ACCEPT_RATE)
    settings = SamplerSettings(chains=4, warmup=1000, draws=6000, samples=4000, seed=11)
    return prior, problem, run(problem, settings)


def test_with_no_decisions_every_share_has_its_exact_beta_marginal(prior_only) -> None:
    prior, _problem, chains = prior_only
    w = {leaf: chains.weights[:, :, i] for i, leaf in enumerate(prior.leaf_ids)}
    obj_a = w["a1"] + w["a2"] + w["a3"]
    obj_b = w["b1"] + w["b2"]
    shares = {
        "obj_a": obj_a, "obj_b": obj_b,
        "a1": w["a1"] / obj_a, "a2": w["a2"] / obj_a, "a3": w["a3"] / obj_a,
        "b1": w["b1"] / obj_b, "b2": w["b2"] / obj_b,
    }  # fmt: skip
    for group in prior.groups:
        for child, centre in zip(group.children, group.centre, strict=True):
            a, b_ = group.kappa * centre, group.kappa * (1 - centre)
            draws = shares[child]
            thinned = draws[:, ::30].ravel()
            p_value = stats.kstest(thinned, stats.beta(a, b_).cdf).pvalue
            assert p_value > 0.01, (child, p_value)
            mean_se = math.sqrt(a * b_ / ((a + b_) ** 2 * (a + b_ + 1))) / math.sqrt(
                dg.ess_bulk(draws)
            )
            assert abs(draws.mean() - a / (a + b_)) < 6 * mean_se, child


def test_with_no_decisions_alpha_and_beta_keep_their_priors(prior_only) -> None:
    _prior, problem, chains = prior_only
    alpha = chains.alpha[:, ::30].ravel()
    assert stats.kstest(alpha, stats.norm(problem.alpha_mean, 1.5).cdf).pvalue > 0.01
    beta = chains.beta[:, ::30].ravel()
    assert stats.kstest(beta, stats.lognorm(1.0, scale=8.0).cdf).pvalue > 0.01


def test_a_prior_only_run_converges_and_is_prior_dominated(prior_only) -> None:
    prior, problem, chains = prior_only
    assert report.diagnose(prior, chains).converged
    assert report.prior_dominated(report.leaf_summaries(prior, chains))
    beta = {p.name: p for p in ch.parameter_summaries(chains, problem)}["beta"]
    assert beta.prior_mean == pytest.approx(8.0 * math.exp(0.5))


# ---- the convergence gate ----------------------------------------------------------------------


def test_a_fifty_iteration_run_is_labelled_not_converged() -> None:
    settings = SamplerSettings(warmup=25, draws=25, samples=100, seed=0)
    check = sy.choices_synthetic_check(200, seed=0, settings=settings)
    assert not check.diagnostics.converged


def test_the_synthetic_check_repeats_for_a_seed_and_carries_its_label() -> None:
    settings = SamplerSettings(warmup=200, draws=200, samples=400, seed=3)
    one = sy.choices_synthetic_check(100, seed=3, settings=settings)
    two = sy.choices_synthetic_check(100, seed=3, settings=settings)
    assert one == two
    assert one.label == ch.SYNTHETIC_LABEL
    assert one.decisions == 100 and 0 < one.accepts < 100


def test_flipping_flips_the_stated_share() -> None:
    prior = sy.bench_prior()
    clean = sy.simulate_decisions(prior, sy.planted_choice(), 4000, np.random.default_rng(8))
    noisy = sy.simulate_decisions(
        prior, sy.planted_choice(), 4000, np.random.default_rng(8), flip_share=0.2
    )
    assert clean.flipped == 0
    assert noisy.flipped == int((clean.data.accept != noisy.data.accept).sum())
    low, high = stats.binomtest(noisy.flipped, 4000).proportion_ci(0.99)
    assert low <= 0.2 <= high


# ---- from a project: refusals and read-only ----------------------------------------------------

LABELS = {"o4": b.FOUR, "o5": b.FIVE, "cat": b.COLOURS, "o4b": b.FOUR, "o5b": b.FIVE}
FAST = SamplerSettings(warmup=1000, draws=1500, samples=1000, seed=0)


def decisions(n: int = 120, seed: int = 2, decider: str | None = "d1") -> list[ch.Decision]:
    """Synthetic decisions on the builder project's labels: accept when the label sum is high."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        labels = {leaf: str(rng.choice(options)) for leaf, options in LABELS.items()}
        score = sum(
            options.index(labels[leaf]) / (len(options) - 1) for leaf, options in LABELS.items()
        )
        accepted = bool(rng.random() < 1 / (1 + math.exp(-(score - 2.5) * 3)))
        out.append(ch.Decision(labels=labels, accepted=accepted, decider=decider))
    return out


@pytest.fixture(scope="module")
def project_fit():
    project = b.project()
    before = (project.model_dump_json(), project.fingerprint())
    fit = ch.fit_decisions(
        project, None, decisions(), base_rate=0.3, conviction="medium", settings=FAST
    )
    return project, before, fit


def test_fitting_decisions_never_changes_the_project(project_fit) -> None:
    project, before, fit = project_fit
    assert (project.model_dump_json(), project.fingerprint()) == before
    assert fit.report.project_fingerprint == before[1]


def test_a_project_fit_reports_what_entered(project_fit) -> None:
    _project, _before, fit = project_fit
    r = fit.report
    assert r.status == "converged", r.diagnostics
    assert fit.chains is not None
    assert r.compared_criteria == ["o4", "o5", "cat", "o4b", "o5b"]
    assert r.decisions == 120 and r.accepts + r.declines == 120
    assert r.deciders == ["d1"] and not r.pooled
    assert r.base_accept_rate == 0.3 and r.s_ref == 0.5 and r.link == "logit"
    assert r.conviction == "medium" and r.conviction_source == "stated by the caller"
    assert r.rated_decisions == dict.fromkeys(r.compared_criteria, 120)
    zero = {leaf.criterion_id: leaf for leaf in r.leaves}["zero"]
    assert zero.posterior_mean == 0.0  # a held zero-weight leaf stays at 0


def test_refusals() -> None:
    project = b.project()
    with pytest.raises(McdaError, match="no decisions"):
        ch.fit_decisions(project, None, [], base_rate=0.3, conviction="low")
    all_accept = [ch.Decision(labels={"o4": b.FOUR[0]}, accepted=True)] * 3
    with pytest.raises(McdaError, match="both accepted and declined"):
        ch.fit_decisions(project, None, all_accept, base_rate=0.3, conviction="low")
    two = decisions(10, decider="d1") + decisions(10, seed=3, decider="d2")
    with pytest.raises(McdaError, match="never averaged"):
        ch.fit_decisions(project, None, two, base_rate=0.3, conviction="low")
    unrated = [
        ch.Decision(labels={"num": "x"}, accepted=True),
        ch.Decision(labels={"o4": b.FOUR[0]}, accepted=False),
    ]
    with pytest.raises(McdaError, match="rates no compared criterion"):
        ch.fit_decisions(project, None, unrated, base_rate=0.3, conviction="low")
    with pytest.raises(ValueError, match="base accept rate"):
        ch.fit_decisions(project, None, decisions(10), base_rate=0.0, conviction="low")


def test_pooling_two_deciders_is_a_stated_choice() -> None:
    two = decisions(40, decider="d1") + decisions(40, seed=3, decider="d2")
    fit = ch.fit_decisions(
        b.project(),
        None,
        two,
        base_rate=0.3,
        conviction="low",
        pool_deciders=True,
        settings=SamplerSettings(warmup=100, draws=100, samples=200, seed=0),
    )
    assert fit.report.pooled and fit.report.deciders == ["d1", "d2"]


# ---- the command -------------------------------------------------------------------------------


def test_the_command_prints_the_check_and_exits_1_when_not_converged() -> None:
    result = CliRunner().invoke(
        app,
        [
            "calibrate",
            "choices-synthetic-check",
            "--decisions",
            "60",
            "--warmup",
            "25",
            "--draws",
            "25",
        ],
    )
    assert result.exit_code == 1
    printed = json.loads(result.stdout[: result.stdout.rindex("}") + 1])
    assert printed["decisions"] == 60 and printed["label"] == ch.SYNTHETIC_LABEL
    assert printed["diagnostics"]["converged"] is False


def test_the_command_checks_its_options() -> None:
    runner = CliRunner()
    assert runner.invoke(
        app, ["calibrate", "choices-synthetic-check", "--decisions", "0"]
    ).exit_code
    bad = runner.invoke(app, ["calibrate", "choices-synthetic-check", "--flip-share", "0.6"])
    assert bad.exit_code


def test_children_held_in_the_projects_smaa_section_stay_at_their_stated_weight() -> None:
    project = ps.set_smaa_model(
        b.project(),
        SmaaModel(groups=[GroupUncertainty(parent_id=None, kind="unknown", held=["cat"])]),
    )
    fit = ch.fit_decisions(
        project,
        None,
        decisions(60),
        base_rate=0.3,
        conviction="low",
        settings=SamplerSettings(warmup=100, draws=100, samples=200, seed=0),
    )
    cat = {leaf.criterion_id: leaf for leaf in fit.report.leaves}["cat"]
    assert not cat.varies
    assert cat.posterior_mean == pytest.approx(0.25, abs=1e-12)
    assert cat.posterior_sd == pytest.approx(0.0, abs=1e-12)


def test_the_choice_fit_uses_the_calibration_gate() -> None:
    check = sy.choices_synthetic_check(
        50, seed=0, settings=SamplerSettings(warmup=25, draws=25, samples=100, seed=0)
    )
    assert check.diagnostics.rhat_max == 1.01 and check.diagnostics.ess_min == 400.0
    assert not dg.converged([1.0101], [10_000.0])
    assert not dg.converged([1.0], [399.0])
    assert dg.converged([1.0099], [400.0])
