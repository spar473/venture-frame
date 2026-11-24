"""Decision confidence against independent answers.

The shares are recounted from SMAA-2's own rank and class counts, the sampled intervals are
recomputed from SciPy's beta quantiles (not the engine's), and the exact-enumeration path is
checked against ``smaa_oracle``'s exact probabilities. Design: ``docs/design/robustness-bands.md``,
"Decision confidence"."""

from fractions import Fraction
from pathlib import Path

import pytest
import smaa_oracle
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import beta
from test_smaa_reference import cell, qualification_with_discrete_uncertainty, tie_case

from mcda_studio.domain.preferences import AnalysisConfiguration
from mcda_studio.domain.results import SmaaResult
from mcda_studio.domain.uncertainty import DiscreteDistribution, SmaaModel
from mcda_studio.services import analysis_service
from mcda_studio.services import project_service as ps
from mcda_studio.services import screening_robustness_service as svc
from mcda_studio.storage.json_repository import JsonProjectRepository, to_json_text

from builders import scored_project

STUDY = Path(__file__).parents[2] / "examples" / "nz_venture_screening" / "project.json"
PROFILES = ["technical_ip", "commercial_traction"]
#: SciPy and the engine compute the same exact interval by different routes; they agree to
#: far better than this, which is well below anything the report or the app displays.
INTERVAL_AGREEMENT = 1e-9


def scipy_clopper_pearson(count: int, n: int, confidence: float) -> tuple[float, float]:
    alpha = 1 - confidence
    lower = 0.0 if count == 0 else float(beta.ppf(alpha / 2, count, n - count + 1))
    upper = 1.0 if count == n else float(beta.ppf(1 - alpha / 2, count + 1, n - count))
    return lower, upper


def _study():
    return JsonProjectRepository().load(STUDY)


def _smaa(project, configuration: AnalysisConfiguration) -> SmaaResult:  # type: ignore[no-untyped-def]
    result = analysis_service.evaluate_project(
        project, configuration.model_copy(update={"method": "smaa2", "parameters": {}})
    )
    assert isinstance(result, SmaaResult)
    return result


@pytest.mark.parametrize("profile", PROFILES)
def test_case_2_confidence_is_recounted_from_smaa_counts(profile: str) -> None:
    project = _study()
    configuration = project.analysis.model_copy(update={"weight_profile_id": profile})
    bands = svc.screening_robustness(project, configuration)
    smaa = _smaa(project, configuration)
    n = smaa.samples
    assert n == 10_000 and bands.top_k == 3 and bands.confidence == smaa.confidence == 0.95
    by = {a.alternative_id: a for a in smaa.acceptability}
    class_order = [c.class_id for c in smaa.classes]
    assert class_order == ["prioritise_for_diligence", "monitor", "not_now"]
    for row in bands.rows:
        counts = [e.count for e in by[row.alternative_id].ranks]
        assert all(c is not None for c in counts)
        # first: rank 1, ties included, the same share as rank1_share
        assert row.first.share == row.rank1_share == counts[0] / n
        assert (row.first.lower, row.first.upper) == pytest.approx(
            scipy_clopper_pearson(counts[0], n, 0.95), abs=INTERVAL_AGREEMENT
        )
        # top 3: each draw gives one rank, so the first three counts add
        top = sum(counts[:3])
        assert row.top_ranks.share == top / n
        assert (row.top_ranks.lower, row.top_ranks.upper) == pytest.approx(
            scipy_clopper_pearson(top, n, 0.95), abs=INTERVAL_AGREEMENT
        )
        assert row.first.share <= row.top_ranks.share
        # classes: best first, every draw in exactly one class
        assert [c.class_id for c in row.classes] == class_order
        class_counts = {k: e.count for k, e in by[row.alternative_id].classes.items()}
        assert sum(class_counts.values()) == n  # type: ignore[arg-type]
        for share in row.classes:
            count = class_counts[share.class_id]
            assert count is not None
            assert share.share == count / n
            assert (share.lower, share.upper) == pytest.approx(
                scipy_clopper_pearson(count, n, 0.95), abs=INTERVAL_AGREEMENT
            )


def test_case_2_reads_as_the_design_note_says() -> None:
    project = _study()
    technical = svc.screening_robustness(
        project, project.analysis.model_copy(update={"weight_profile_id": "technical_ip"})
    )
    by = {r.alternative_id: r for r in technical.rows}
    assert by["upstock"].first.share == 0.9302
    assert by["upstock"].top_ranks.share == 0.9991
    assert by["upstock"].classes[0].share == 0.9744
    assert by["wych"].classes[0].share == 0.357
    assert by["kea_aerospace"].classes[2].share == 0.719
    assert by["kea_aerospace"].top_ranks.share == 0.0


def test_exact_enumeration_reports_probabilities_without_intervals() -> None:
    project = qualification_with_discrete_uncertainty()
    bands = svc.screening_robustness(project)
    assert bands.available and bands.evaluation == "exact_enumeration"
    assert bands.draws is None and bands.confidence is None
    ranks, _, _, classes = smaa_oracle.exact_acceptability(project.model_dump(mode="json"))
    k = bands.top_k
    assert k == min(3, len(project.alternatives))
    for row in bands.rows:
        a = row.alternative_id
        expected_top = sum(ranks[a][:k], Fraction(0))
        # fsum of the engine's float probabilities against the oracle's exact sum
        assert row.top_ranks.share == pytest.approx(float(expected_top), abs=1e-12)
        assert row.first.share == float(ranks[a][0])
        assert {c.class_id: c.share for c in row.classes} == {
            cid: float(p) for cid, p in classes[a].items()
        }
        for estimate in [row.first, row.top_ranks, *row.classes]:
            assert estimate.lower is None and estimate.upper is None


rank_counts = st.lists(st.integers(min_value=0, max_value=50), min_size=1, max_size=8).filter(any)


@settings(max_examples=200, deadline=None)
@given(rank_counts, st.integers(min_value=1, max_value=8))
def test_top_k_counts_add_the_first_k_ranks(counts: list[int], k: int) -> None:
    n = sum(counts)
    top = svc.top_k_count(counts, k)
    assert top == sum(c for rank, c in enumerate(counts, start=1) if rank <= k)
    estimate = svc.share_from_counts(top, n, 0.95)
    assert estimate.share == top / n
    assert estimate.lower is not None and estimate.upper is not None
    assert 0.0 <= estimate.lower <= estimate.share <= estimate.upper <= 1.0


def test_fewer_alternatives_than_top_k_count_every_rank() -> None:
    project = ps.set_smaa_model(
        scored_project({"a": 0.5, "b": 0.6}),
        SmaaModel(
            cells=[
                cell(
                    "b", "score", DiscreteDistribution(values=[0.4, 0.7], probabilities=[0.5, 0.5])
                )
            ]
        ),
    )
    bands = svc.screening_robustness(project)
    assert bands.available and bands.top_k == 2 < svc.TOP_K
    assert [r.top_ranks.share for r in bands.rows] == [1.0, 1.0]
    assert sorted(r.first.share for r in bands.rows) == [0.5, 0.5]


def test_first_counts_tied_first_places() -> None:
    """In tie_case, a is first only when b ties it: tied first is 0.5, sole first is 0."""
    project = tie_case()
    bands = svc.screening_robustness(project)
    by = {r.alternative_id: r for r in bands.rows}
    smaa = {a.alternative_id: a for a in _smaa(project, project.analysis).acceptability}
    assert smaa["a"].sole_first.frequency == 0.0 and smaa["a"].tied_first.frequency == 0.5
    assert by["a"].first.share == by["a"].rank1_share == 0.5


def test_looking_changes_nothing() -> None:
    project = _study()
    before, fingerprint = to_json_text(project), project.fingerprint()
    svc.screening_robustness(project)
    assert to_json_text(project) == before and project.fingerprint() == fingerprint


def test_the_label_never_speaks_of_success_as_a_probability_it_gives() -> None:
    label = svc.CONFIDENCE_LABEL
    assert "not a probability of success" in label
    assert "share of draws" in label
