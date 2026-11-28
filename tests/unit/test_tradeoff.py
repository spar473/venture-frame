"""Trade-off elicitation: the method's parts, the service and the command.

The likelihood against hand-computed values, the pairwise conflict check against SciPy's LP
solver, the question rules (never an implied answer, the even split first), the stop rule, the
prior blocks (intervals and ordered) under the rejuvenation kernel, noisy mode's convergence and
direction, the tradeoff-answers and weight-samples (basis tradeoff) sidecars, conflicts and the
fallback to noisy mode, and read-only commands. Design: docs/design/tradeoff-elicitation.md.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.optimize import linprog
from scipy.stats import ks_2samp
from typer.testing import CliRunner

from mcda_studio.calibration import sampler
from mcda_studio.calibration import tradeoff as ct
from mcda_studio.cli import app
from mcda_studio.domain import tradeoff_answers as ta
from mcda_studio.domain import weight_samples as ws
from mcda_studio.domain.errors import McdaError
from mcda_studio.methods import tradeoff as tm
from mcda_studio.services import project_service as ps
from mcda_studio.services import tradeoff_service as ts
from mcda_studio.storage.json_repository import JsonProjectRepository

from builders import flat_project

STUDY = Path(__file__).parents[2] / "examples" / "nz_venture_screening" / "project.json"
SEED = 2026112811 + 200  # the DD11 base seed, offset for the unit tests


def flat3():  # type: ignore[no-untyped-def]
    values = [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.5, 1.0]]
    return flat_project(values, [50.0, 30.0, 20.0])


def noisy_with(settings: sampler.SamplerSettings):  # type: ignore[no-untyped-def]
    return lambda prior, contrasts, seed: ct.fit_noisy(prior, contrasts, seed, settings)


def answers_for(project, profile_id: str, items: list[ta.TradeoffAnswer]) -> ta.TradeoffAnswers:  # type: ignore[no-untyped-def]
    return ta.TradeoffAnswers(
        kind="tradeoff-answers", version=1, project_id=project.id,
        project_fingerprint=project.fingerprint(), weight_profile_id=profile_id,
        answered_by="test", source="person", answers=items,
    )  # fmt: skip


# ---- the likelihood -----------------------------------------------------------------------------


def test_the_bradley_terry_likelihood_matches_a_hand_computed_value() -> None:
    nested = tm.ordered_nested_prior(ts.context(flat3(), None, "unknown").prior)
    contrasts = np.array([[1.0, -0.5, 0.0], [0.0, 1.0, -1.0]])
    problem = ct.tradeoff_problem(nested, contrasts)
    w = np.array([[0.5], [0.3], [0.2]])
    # s = (0.5 - 0.15, 0.3 - 0.2) = (0.35, 0.1); beta = 4: log sigmoid(1.4) + log sigmoid(0.4)
    got = problem.log_likelihood(w, np.zeros(1), np.array([4.0]))
    want = -np.log1p(np.exp(-1.4)) - np.log1p(np.exp(-0.4))
    assert got[0] == pytest.approx(want, abs=1e-12)
    assert problem.score_centre() == 0.0
    assert problem.log_beta_mean == pytest.approx(np.log(20.0)) and problem.log_beta_sd == 1.0
    empty = ct.tradeoff_problem(nested, np.zeros((0, 3)))
    assert empty.log_likelihood(w, np.zeros(1), np.array([4.0]))[0] == 0.0


# ---- the conflict check -------------------------------------------------------------------------


def lp_infeasible(a: np.ndarray, b: np.ndarray) -> bool:
    n = a.size
    res = linprog(
        np.zeros(n), A_ub=-np.vstack([a, b]), b_ub=np.zeros(2), A_eq=np.ones((1, n)),
        b_eq=[1.0], bounds=[(0, None)] * n, method="highs",
    )  # fmt: skip
    return res.status == 2


def test_the_pair_check_agrees_with_an_lp_solver() -> None:
    rng = np.random.Generator(np.random.PCG64(SEED))
    infeasible = 0
    for _ in range(400):
        n = int(rng.integers(2, 7))
        a = rng.normal(size=n) * (rng.random(n) < 0.8)
        b = -a * rng.random() + rng.normal(size=n) * 0.3
        got = tm.pair_infeasible(a, b)
        assert got == lp_infeasible(a, b)
        infeasible += got
    assert 20 < infeasible < 380  # both outcomes are exercised


def test_the_pair_check_respects_its_support() -> None:
    a = np.array([1.0, -1.0, 0.0])
    b = np.array([-1.0, 1.0, 0.0])
    assert not tm.pair_infeasible(a, b)  # w = e_2 satisfies both with equality
    assert tm.pair_infeasible(a, b - np.array([0, 0, 1.0]), np.array([True, True, False])) is False
    # Feasible only through the third leaf (w = e_2): leaving it out makes the pair infeasible.
    a2, b2 = np.array([1.0, -2.0, 1.0]), np.array([-1.0, 1.0, 1.0])
    assert not tm.pair_infeasible(a2, b2) and not lp_infeasible(a2, b2)
    assert tm.pair_infeasible(a2, b2, np.array([True, True, False]))
    assert lp_infeasible(a2[:2], b2[:2])


# ---- questions and stopping ---------------------------------------------------------------------


def test_an_implied_answer_is_never_asked_and_the_even_split_comes_first() -> None:
    probs = np.array([0.99, 0.01, 0.875, 0.75, 0.25, 0.019, 0.981])
    assert tm.select_question(probs, set()) == 3  # 0.75 and 0.25 tie exactly; the first wins
    assert tm.select_question(probs, {3}) == 4
    assert tm.select_question(probs, {3, 4}) == 2
    assert tm.select_question(probs, {2, 3, 4}) is None
    edge = np.array([0.02, 0.98])
    assert tm.select_question(edge, set()) == 0
    rng = np.random.Generator(np.random.PCG64(SEED))
    picks = {tm.select_question(probs, set(), mode="random", rng=rng) for _ in range(200)}
    assert picks == {2, 3, 4}
    with pytest.raises(ValueError):
        tm.select_question(probs, set(), mode="random")


def test_the_stop_rule() -> None:
    flat = [np.array([0.5, 0.5])] * 4
    assert not tm.should_stop(flat[:3], 4000, 2)  # needs more than the window
    assert tm.should_stop(flat, 4000, 3)
    moving = [np.array([0.5, 0.5])] * 3 + [np.array([0.6, 0.4])]
    assert not tm.should_stop(moving, 4000, 3)
    # 99% noise of a difference at p = 0.5 from 4000 draws: 2.576 * sqrt(0.5 / 4000) = 0.0288
    small = [np.array([0.5, 0.5])] * 3 + [np.array([0.528, 0.472])]
    assert tm.should_stop(small, 4000, 3)
    assert tm.should_stop(moving, 4000, 30) and not tm.should_stop(moving, 4000, 29)
    certain = [np.array([1.0, 0.0])] * 4
    assert tm.should_stop(certain, 4000, 3)


def test_p_first_counts_each_draws_leader() -> None:
    partial = np.array([[1.0, 0.0], [0.0, 1.0], [0.4, 0.4]])
    weights = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.45, 0.55]])
    assert tm.first_probabilities(weights, partial).tolist() == [0.5, 0.5, 0.0]


def test_every_question_changes_two_criteria_at_their_own_levels() -> None:
    project = JsonProjectRepository().load(STUDY)
    ctx = ts.context(project, None)
    pool = ts.candidates(ctx)
    assert len(pool) > 1000
    for c in pool:
        nonzero = np.flatnonzero(c.contrast)
        assert len(nonzero) == 2 and set(c.a) == set(c.b) and len(c.a) == 2
        assert (c.contrast[nonzero] > 0).sum() == 1  # better on one, worse on the other
        for leaf in c.a:
            shown = {_ for lv in ctx.levels[leaf] for _ in [ts._norm(lv.raw)]}
            assert ts._norm(c.a[leaf]) in shown and ts._norm(c.b[leaf]) in shown
    mandate = {"hq_country", "stage", "sector"}
    assert not any(mandate & set(c.a) for c in pool)  # screened-only leaves never asked
    real = ts.candidates(ctx, real_pairs=True)
    pairs = [c for c in real if c.alternatives is not None]
    assert len(pairs) == 28  # every pair of eight ventures


# ---- the prior blocks under the kernel ----------------------------------------------------------


def test_the_kernel_keeps_an_intervals_block_and_an_ordered_block() -> None:
    leaves = ["l0", "l1", "l2", "m0", "m1", "m2"]
    nodes = ["g", "h", *leaves]
    paths = {leaf: ["g" if leaf[0] == "l" else "h", leaf] for leaf in leaves}
    local = {"g": 0.5, "h": 0.5, **dict.fromkeys(leaves, 1 / 3)}
    groups = [
        tm.LatentGroup(None, "intervals", ("g", "h"), 1.0, lower=(20.0, 10.0), upper=(80.0, 40.0)),
        tm.LatentGroup("g", "ordered", ("l2", "l0", "l1"), 1.0, alpha=(1.0, 1.0, 1.0)),
    ]
    prior = tm.ConstrainedPrior(nodes, leaves, paths, local, groups)
    state = tm.SmcState.start(prior, tm.SmcSettings(particles=4000, seed=SEED))
    for _ in range(3):
        state.move(20)
    fresh = prior.global_weights(
        prior.sample(np.random.Generator(np.random.PCG64(SEED + 1)), 100_000)
    )
    for j in range(len(leaves)):
        assert ks_2samp(state.weights[:, j], fresh[:, j]).pvalue > 0.001
    order = [leaves.index(x) for x in ("l2", "l0", "l1")]
    assert np.all(np.diff(state.weights[:, order], axis=1) <= 1e-15)
    assert np.allclose(state.weights.sum(axis=1), 1.0)


def test_the_ordered_nested_prior_refuses_a_broken_order() -> None:
    ctx = ts.context(flat3(), None, "ranking")
    assert ctx.prior.groups[0].children == ("c0", "c1", "c2")  # ratings 50, 30, 20
    draws = ctx.prior.global_weights(
        ctx.prior.sample(np.random.Generator(np.random.PCG64(SEED)), 500)
    )
    assert np.all(np.diff(draws, axis=1) <= 0.0)  # prior draws keep the order
    nested = tm.ordered_nested_prior(ctx.prior)
    rng = np.random.Generator(np.random.PCG64(SEED))
    for _ in range(20):
        start = nested.initial_ys(rng)
        assert np.isfinite(nested.log_density([s[None, :] for s in start]))[0]
    broken = [np.array([[-1.0, 0.5]])]  # p = softmax(-1, 0.5, 0): not decreasing
    assert nested.log_density(broken)[0] == -np.inf
    with pytest.raises(ValueError):
        tm.ordered_nested_prior(
            ts.context(JsonProjectRepository().load(STUDY), None, "stated").prior
        )


def test_the_stated_prior_reads_case_2s_intervals_and_holds_the_mandate() -> None:
    ctx = ts.context(JsonProjectRepository().load(STUDY), None, "stated")
    (group,) = ctx.prior.groups
    assert group.kind == "intervals" and "mandate" in group.held and len(group.children) == 4
    assert group.lower == (80.0, 60.0, 60.0, 50.0) and group.upper == (90.0, 100.0, 80.0, 100.0)


# ---- noisy mode ---------------------------------------------------------------------------------


def test_noisy_mode_converges_and_moves_toward_the_answers() -> None:
    project = flat3()
    ctx = ts.context(project, None, "unknown")
    # Ten answers that c1 at full swing beats c0 at 60%: w1 > 0.6 w0, against the stated 0.3/0.5.
    items = [
        ta.TradeoffAnswer(id=f"q{i}", a={"c0": 0.4, "c1": 1.0}, b={"c0": 1.0, "c1": 0.0},
                          choice="A")
        for i in range(10)
    ]  # fmt: skip
    answers = answers_for(project, ctx.profile.id, items)
    settings = sampler.SamplerSettings(chains=4, warmup=1000, draws=1500, samples=2000, seed=SEED)
    result = ts.fit(ctx, answers, mode="noisy", noisy=noisy_with(settings))
    assert result.report.status == "noisy"
    assert result.report.diagnostics is not None and result.report.diagnostics.converged
    assert result.draws is not None
    ratio = result.draws[:, 1] / result.draws[:, 0]
    assert np.median(ratio) > 0.6
    prior_ratio = result.report.leaves[1].prior_median / result.report.leaves[0].prior_median
    assert np.median(ratio) > prior_ratio


# ---- the answers sidecar ------------------------------------------------------------------------


def good_answers() -> dict:  # type: ignore[type-arg]
    return {
        "kind": "tradeoff-answers", "version": 1, "project_id": "p1",
        "project_fingerprint": "0" * 64, "weight_profile_id": "w", "answered_by": "K",
        "source": "person",
        "answers": [{"id": "q1", "a": {"x": "High", "y": 0.0}, "b": {"x": "Low", "y": 10.0},
                     "choice": "A"}],
    }  # fmt: skip


def test_the_answers_sidecar_is_strict() -> None:
    ta.TradeoffAnswers.model_validate_json(json.dumps(good_answers()))
    for change, words in [
        ({"kind": "weight-samples"}, "not a tradeoff-answers file"),
        ({"version": 2}, "version 2 is not supported"),
        ({"version": "1"}, "explicit integers"),
        ({"extra": 1}, "Extra inputs"),
        ({"source": "model"}, "source"),
    ]:
        with pytest.raises(ValidationError, match=words):
            ta.TradeoffAnswers.model_validate_json(json.dumps({**good_answers(), **change}))
    bad_items = [
        {"id": "q1", "a": {"x": "High"}, "b": {"y": "Low"}, "choice": "A"},
        {"id": "q1", "a": {"x": "High"}, "b": {"x": "High"}, "choice": "A"},
        {"id": "q1", "alternatives": ["a", "a"], "choice": "B"},
        {"id": "q1", "a": {"x": "H"}, "b": {"x": "L"}, "alternatives": ["a", "b"], "choice": "B"},
        {"id": "q1", "a": {"x": "H"}, "b": {"x": "L"}, "choice": "maybe"},
    ]
    for item in bad_items:
        with pytest.raises(ValidationError):
            ta.TradeoffAnswers.model_validate_json(
                json.dumps({**good_answers(), "answers": [item]})
            )
    twice = {**good_answers(), "answers": good_answers()["answers"] * 2}
    with pytest.raises(ValidationError, match="own id"):
        ta.TradeoffAnswers.model_validate_json(json.dumps(twice))


def test_answers_are_checked_against_the_project() -> None:
    project = JsonProjectRepository().load(STUDY)
    ctx = ts.context(project, "technical_ip")
    item = ta.TradeoffAnswer(
        id="q1", a={"defensibility": "None stated", "market_growth": 40.0},
        b={"defensibility": "Granted patent or exclusive licence", "market_growth": 0.0},
        choice="B",
    )  # fmt: skip
    good = answers_for(project, "technical_ip", [item])
    assert ts.check_answers(ctx, good) == []
    d = ctx.contrast(item)
    j_def, j_growth = ctx.leaf_ids.index("defensibility"), ctx.leaf_ids.index("market_growth")
    assert d[j_def] == pytest.approx(0.75) and d[j_growth] == pytest.approx(-1.0)  # B chosen
    skip = item.model_copy(update={"choice": "skip"})
    assert not ctx.contrast(skip).any()
    with pytest.raises(McdaError, match="profile"):
        ts.check_answers(ctx, good.model_copy(update={"weight_profile_id": "commercial_traction"}))
    with pytest.raises(McdaError, match="project"):
        ts.check_answers(ctx, good.model_copy(update={"project_id": "other"}))
    stale = good.model_copy(update={"project_fingerprint": "1" * 64})
    assert "changed" in ts.check_answers(ctx, stale)[0]
    wrong = item.model_copy(update={"a": {"defensibility": "Great", "market_growth": 40.0}})
    with pytest.raises(McdaError, match="not a level"):
        ts.check_answers(ctx, answers_for(project, "technical_ip", [wrong]))
    outside = item.model_copy(update={"a": {"defensibility": "None stated", "market_growth": 90.0}})
    with pytest.raises(McdaError):
        ts.check_answers(ctx, answers_for(project, "technical_ip", [outside]))


# ---- conflicts ----------------------------------------------------------------------------------


def test_a_contradiction_is_reported_and_auto_falls_back_to_noisy_mode() -> None:
    project = flat3()  # a0 = (1, 0, 0.5), a1 = (0, 1, 0.5), a2 = (0.5, 0.5, 1)
    ctx = ts.context(project, None, "unknown")
    items = [
        # a0 over a2: 0.5 w0 - 0.5 w1 - 0.5 w2 >= 0, so w0 >= 1/2.
        ta.TradeoffAnswer(id="q1", alternatives=["a0", "a2"], choice="A"),
        ta.TradeoffAnswer(id="q2", a={"c0": 1.0, "c2": 0.0}, b={"c0": 0.0, "c2": 1.0}, choice="A"),
        ta.TradeoffAnswer(id="q3", a={"c0": 1.0, "c1": 0.0}, b={"c0": 0.0, "c1": 1.0}, choice="A"),
        # B: 0.5 w1 >= w0. With q1 that needs w1 >= 1: impossible. With q3 alone it leaves only
        # w = e_2 (w0 = w1 = 0), which is feasible, so q3 is not listed.
        ta.TradeoffAnswer(id="q4", a={"c0": 1.0, "c1": 0.5}, b={"c0": 0.0, "c1": 1.0}, choice="B"),
    ]
    answers = answers_for(project, ctx.profile.id, items)
    exact = ts.fit(ctx, answers, mode="exact", settings=tm.SmcSettings(seed=SEED))
    assert exact.report.status == "conflict" and exact.draws is None
    (conflict,) = exact.report.conflicts
    assert conflict.answer_id == "q4" and conflict.contradicts == ["q1"]
    nxt = ts.next_question(ctx, answers, tm.SmcSettings(seed=SEED))
    assert [c.answer_id for c in nxt.conflicts] == ["q4"]
    settings = sampler.SamplerSettings(chains=4, warmup=1000, draws=1500, samples=2000, seed=SEED)
    auto = ts.fit(ctx, answers, settings=tm.SmcSettings(seed=SEED), noisy=noisy_with(settings))
    assert auto.report.status == "noisy" and auto.report.conflicts
    with pytest.raises(McdaError, match="command line"):
        ts.fit(ctx, answers, settings=tm.SmcSettings(seed=SEED))  # the app never fits noisy mode
    with pytest.raises(McdaError, match="noisy mode needs"):
        stated = ts.context(JsonProjectRepository().load(STUDY), None, "stated")
        ts.fit(stated, answers_for(stated.project, stated.profile.id, [
            ta.TradeoffAnswer(id="q1", a={"defensibility": "None stated", "market_growth": 40.0},
                              b={"defensibility": "Granted patent or exclusive licence",
                                 "market_growth": 0.0}, choice="A")]),
               mode="noisy", noisy=ct.fit_noisy)  # fmt: skip


# ---- weight samples with basis tradeoff ---------------------------------------------------------


def test_a_fit_becomes_a_tradeoff_weight_samples_sidecar() -> None:
    project = flat3()
    ctx = ts.context(project, None)
    items = [ta.TradeoffAnswer(id="q1", a={"c0": 1.0, "c1": 0.0}, b={"c0": 0.0, "c1": 1.0},
                               choice="A")]  # fmt: skip
    answers = answers_for(project, ctx.profile.id, items)
    result = ts.fit(ctx, answers, settings=tm.SmcSettings(particles=500, seed=SEED))
    samples = ts.weight_samples(ctx, answers, result)
    assert samples.basis == "tradeoff" and samples.label == ws.TRADEOFF_LABEL
    again = ws.WeightSamples.model_validate_json(samples.model_dump_json())
    assert again == samples
    raw = json.loads(samples.model_dump_json())
    for change, words in [
        ({"label": ws.PRIOR_LABEL}, "trade-off label"),
        ({"usable": False}, "never diagnostic"),
        ({"reference_set_id": "r"}, "uses no reference set"),
    ]:
        with pytest.raises(ValidationError, match=words):
            ws.WeightSamples.model_validate_json(json.dumps({**raw, **change}))
    simulated = answers.model_copy(update={"source": "simulated"})
    with pytest.raises(McdaError, match="simulated"):
        ts.weight_samples(ctx, simulated, ts.fit(ctx, simulated, settings=tm.SmcSettings(
            particles=500, seed=SEED)))  # fmt: skip


# ---- the command --------------------------------------------------------------------------------


def test_next_sheet_and_fit_are_read_only(tmp_path: Path) -> None:
    project = flat3()
    path = tmp_path / "project.json"
    JsonProjectRepository().save(project, path)
    before = path.read_bytes()
    runner = CliRunner()
    nxt = runner.invoke(app, ["elicit", "tradeoff", "next", str(path), "--particles", "500"])
    assert nxt.exit_code == 0, nxt.output
    question = json.loads(nxt.stdout)
    assert question["status"] == "ask" and question["asked"] == 0
    assert 0.02 <= question["question"]["p_a_over_b"] <= 0.98
    sheet_dir = tmp_path / "sheet"
    made = runner.invoke(app, ["elicit", "tradeoff", "sheet", str(path), "-o", str(sheet_dir),
                               "--questions", "4", "--particles", "500"])  # fmt: skip
    assert made.exit_code == 0, made.output
    text = (sheet_dir / "tradeoff.csv").read_bytes()
    assert text.startswith(b"\xef\xbb\xbf")
    html_page = (sheet_dir / "tradeoff.html").read_text(encoding="utf-8")
    assert "How the questions were chosen" in html_page and "probability 1/2" in html_page
    again = runner.invoke(app, ["elicit", "tradeoff", "sheet", str(path), "-o", str(sheet_dir),
                                "--questions", "4", "--particles", "500"])  # fmt: skip
    assert again.exit_code == 1 and "Refusing to overwrite" in again.output
    lines = text.decode("utf-8-sig").splitlines()
    filled = [lines[0]] + [line[:-1] + ("A," if i % 2 else "B,") for i, line in
                           enumerate(lines[1:])]  # fmt: skip
    sheet = tmp_path / "filled.csv"
    sheet.write_text("\r\n".join(filled) + "\r\n", encoding="utf-8-sig")
    saved = tmp_path / "answers.json"
    out = tmp_path / "samples.json"
    fit = runner.invoke(app, ["elicit", "tradeoff", "fit", str(path), "--answers", str(sheet),
                              "--particles", "500", "--answered-by", "Tester",
                              "--save-answers", str(saved), "--out", str(out)])  # fmt: skip
    assert fit.exit_code == 0, fit.output
    report = json.loads(fit.stdout)
    rows = len(lines) - 1  # the ordered three-leaf toy leaves fewer than four open questions
    assert 1 <= rows <= 4
    assert report["status"] == "exact" and report["kept"] == rows
    assert report["answered_by"] == "Tester"
    stored = ta.TradeoffAnswers.model_validate_json(saved.read_text(encoding="utf-8"))
    assert [a.choice for a in stored.answers] == ["B", "A", "B", "A"][:rows]
    assert ws.WeightSamples.model_validate_json(out.read_text(encoding="utf-8")).basis == "tradeoff"
    refit = runner.invoke(app, ["elicit", "tradeoff", "fit", str(path), "--answers", str(saved),
                                "--particles", "500"])  # fmt: skip
    assert refit.exit_code == 0 and json.loads(refit.stdout)["leaves"] == report["leaves"]
    # SMAA-2 reads the tradeoff sidecar as its weight source.
    evaluated = runner.invoke(app, ["evaluate", str(path), "--method", "smaa2", "--uncertainty",
                                    "preference", "--samples", "1000", "--resample",
                                    "--weight-samples", str(out), "--json"])  # fmt: skip
    assert evaluated.exit_code == 0, evaluated.output
    assert json.loads(evaluated.stdout)["weight_samples"]["basis"] == "tradeoff"
    assert path.read_bytes() == before


def test_the_command_refuses_bad_options(tmp_path: Path) -> None:
    path = tmp_path / "project.json"
    JsonProjectRepository().save(flat3(), path)
    runner = CliRunner()
    bad = runner.invoke(app, ["elicit", "tradeoff", "next", str(path), "--prior", "strong"])
    assert bad.exit_code == 1 and "--prior" in bad.output
    few = runner.invoke(app, ["elicit", "tradeoff", "next", str(path), "--particles", "10"])
    assert few.exit_code == 1 and "100 particles" in few.output
    nothing = ps.set_judgement(flat_project([[0.0], [1.0]], [1.0]), "c0", 1.0)
    single = tmp_path / "single.json"
    JsonProjectRepository().save(nothing, single)
    none = runner.invoke(app, ["elicit", "tradeoff", "next", str(single)])
    assert none.exit_code == 1 and "two or more weights" in none.output
