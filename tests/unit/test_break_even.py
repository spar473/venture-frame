"""Break-even weights: an analytic two-alternative case, a brute-force grid on a three-leaf toy,
the nearest shift against SciPy's LP solver, held leaves, class boundaries and the command.

Design: docs/design/tradeoff-elicitation.md (Part 3).
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linprog
from typer.testing import CliRunner

from mcda_studio.cli import app
from mcda_studio.domain.errors import McdaError
from mcda_studio.domain.preferences import AnalysisConfiguration
from mcda_studio.methods.mavt.break_even import nearest_shift
from mcda_studio.services import break_even_service as be
from mcda_studio.services import project_service as ps
from mcda_studio.storage.json_repository import JsonProjectRepository

from builders import flat_project

STUDY = Path(__file__).parents[2] / "examples" / "nz_venture_screening" / "project.json"
SEED = 2026112811 + 300


def test_two_alternatives_break_even_where_the_analysis_says() -> None:
    # w = (0.6, 0.4); a0 = (1, 0) scores 0.6, a1 = (0, 1) scores 0.4. a0 leads until w0 < 1/2.
    project = flat_project([[1.0, 0.0], [0.0, 1.0]], [60.0, 40.0])
    result = be.break_even(project)
    assert result.leader == "a0"
    c0, c1 = result.criteria
    assert c0.falls_below is not None and c0.rises_above is None
    assert c0.falls_below.local_weight == pytest.approx(0.5, abs=1e-12)
    assert c0.falls_below.share == pytest.approx(0.5, abs=1e-12)
    assert c0.falls_below.new_leader == "a1"
    assert c1.rises_above is not None and c1.rises_above.share == pytest.approx(0.5, abs=1e-12)
    assert c0.sentence == "alt 0 stays first unless crit 0 falls below 50%"
    assert c1.sentence == "alt 0 stays first unless crit 1 rises above 50%"
    nearest = result.nearest
    assert nearest is not None and nearest.alternative_id == "a1"
    assert nearest.distance == pytest.approx(0.2, abs=1e-12)  # 0.1 moves from c0 to c1
    assert [(c.criterion_id, c.changed) for c in nearest.changes] == [
        ("c0", pytest.approx(0.5)),
        ("c1", pytest.approx(0.5)),
    ]


def simplex_grid(step: float) -> np.ndarray:
    n = int(round(1 / step))
    pts = [(i, j, n - i - j) for i in range(n + 1) for j in range(n + 1 - i)]
    return np.array(pts, dtype=float) / n


def test_a_brute_force_grid_agrees_on_a_three_leaf_toy() -> None:
    values = [[0.9, 0.2, 0.4], [0.3, 0.8, 0.5], [0.5, 0.5, 0.9], [0.1, 0.3, 0.2]]
    project = with_classes(flat_project(values, [50.0, 30.0, 20.0]))
    cfg = AnalysisConfiguration(method="mavt_sorting")
    result = be.break_even(project, cfg)
    partial = np.array(values)
    w0 = np.array([0.5, 0.3, 0.2])
    lead = int(np.argmax(partial @ w0))
    assert result.leader == f"a{lead}"
    step = 0.001
    grid = simplex_grid(step)
    scores = grid @ partial.T
    changed = np.any(np.delete(scores, lead, axis=1) >= scores[:, [lead]], axis=1)
    brute = np.abs(grid[changed] - w0).sum(axis=1).min()
    assert result.nearest is not None
    assert result.nearest.distance <= brute + 1e-12  # exact is never worse than the grid
    assert brute - result.nearest.distance < 4 * step  # and the grid gets within its resolution
    # Per criterion: vary leaf k's weight on a fine line, siblings rescaled in proportion.
    for k, row in enumerate(result.criteria):
        ts = np.linspace(0.0, 1.0, 100_001)
        rest = np.delete(w0, k) / np.delete(w0, k).sum()
        line = np.zeros((ts.size, 3))
        line[:, k] = ts
        line[:, [j for j in range(3) if j != k]] = (1 - ts)[:, None] * rest[None, :]
        leaders = np.argmax(line @ partial.T, axis=1)
        other = leaders != lead
        below = ts[other & (ts < w0[k])]
        above = ts[other & (ts > w0[k])]
        if row.falls_below is None:
            assert below.size == 0
        else:
            assert abs(below.max() - row.falls_below.local_weight) < 2e-5
        if row.rises_above is None:
            assert above.size == 0
        else:
            assert abs(above.min() - row.rises_above.local_weight) < 2e-5
    # Class boundaries (0.6 and 0.4): the nearest crossing on the grid.
    bounds = [0.6, 0.4]
    for i, move in enumerate(result.classes):
        v = scores[:, i]
        now = partial[i] @ w0
        k = sum(1 for b in bounds if now < b)
        if move.up is not None:
            hit = v >= bounds[k - 1]
            assert abs(np.abs(grid[hit] - w0).sum(axis=1).min() - move.up.distance) < 4 * step
        if move.down is not None:
            hit = v <= bounds[k]
            assert abs(np.abs(grid[hit] - w0).sum(axis=1).min() - move.down.distance) < 4 * step
    assert any(m.up for m in result.classes) and any(m.down for m in result.classes)


def with_classes(project):  # type: ignore[no-untyped-def]
    from mcda_studio.domain.sorting import SortingClass, SortingModel

    model = SortingModel(
        classes=[
            SortingClass(id="top", name="Top", lower_boundary=0.6),
            SortingClass(id="mid", name="Middle", lower_boundary=0.4),
            SortingClass(id="low", name="Low", lower_boundary=None),
        ]
    )
    return ps.set_sorting_model(project, model)


def lp_distance(w0: np.ndarray, d: np.ndarray, eligible: np.ndarray) -> float | None:
    n = w0.size
    # variables: w (n), u (n) with u >= |w - w0|; minimise sum u
    c = np.r_[np.zeros(n), np.ones(n)]
    a_ub = np.vstack(
        [
            np.c_[np.eye(n), -np.eye(n)],
            np.c_[-np.eye(n), -np.eye(n)],
            np.r_[-d, np.zeros(n)][None, :],
        ]
    )
    b_ub = np.r_[w0, -w0, 0.0]
    a_eq = np.r_[np.ones(n), np.zeros(n)][None, :]
    bounds = [(0, None) if eligible[j] else (w0[j], w0[j]) for j in range(n)] + [(0, None)] * n
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=[1.0], bounds=bounds, method="highs")
    return float(res.fun) if res.status == 0 else None


def test_the_nearest_shift_agrees_with_an_lp_solver() -> None:
    rng = np.random.Generator(np.random.PCG64(SEED))
    solved = 0
    for _ in range(300):
        n = int(rng.integers(2, 8))
        w0 = rng.dirichlet(np.ones(n))
        d = rng.normal(size=n) - 0.3
        eligible = rng.random(n) < 0.8
        got = nearest_shift(w0, d, eligible)
        want = lp_distance(w0, d, eligible)
        if want is None:
            assert got is None
            continue
        assert got is not None
        assert got.distance == pytest.approx(want, abs=1e-9)
        assert got.weights @ d >= -1e-12 and got.weights.sum() == pytest.approx(1.0)
        assert np.all(got.weights[~eligible] == w0[~eligible])
        solved += got.distance > 0
    assert solved > 100


def test_a_leaf_rated_zero_never_moves_on_case_2() -> None:
    result = be.break_even(JsonProjectRepository().load(STUDY))
    assert result.leader_name == "Upstock"
    mandate = {"hq_country", "stage", "sector"}
    for change in [result.nearest, *result.challengers]:
        assert change is not None and not mandate & {c.criterion_id for c in change.changes}
    team = next(c for c in result.criteria if c.criterion_id == "team")
    assert team.sentence.startswith("Upstock stays first unless Team falls below")
    assert not any(c.criterion_id in mandate | {"mandate"} for c in result.criteria)


def test_break_even_refuses_a_method_without_the_value_model() -> None:
    project = flat_project([[1.0, 0.0], [0.0, 1.0]], [60.0, 40.0])
    with pytest.raises(McdaError, match="additive value model"):
        be.break_even(project, AnalysisConfiguration(method="promethee2"))


def test_evaluate_prints_break_even(tmp_path: Path) -> None:
    runner = CliRunner()
    out = runner.invoke(app, ["evaluate", str(STUDY), "--break-even", "--json"])
    assert out.exit_code == 0, out.output
    report = json.loads(out.stdout)
    assert report["leader"] == "upstock" and report["nearest"]["alternative_id"] == "wellumio"
    table = runner.invoke(app, ["evaluate", str(STUDY), "--break-even"])
    assert table.exit_code == 0 and "Break-even weights" in table.stdout
    assert "Nearest change" in table.stdout
    plain = runner.invoke(app, ["evaluate", str(STUDY)])
    assert "Break-even" not in plain.stdout
