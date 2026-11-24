"""Robustness bands against the independent oracle: hand-built acceptabilities, generated ones and
case 2's real SMAA-2 output. The service must match the oracle exactly."""

import subprocess
import sys
from pathlib import Path

import pytest
import robustness_oracle as oracle
from hypothesis import given, settings
from hypothesis import strategies as st

from mcda_studio.domain.preferences import AnalysisConfiguration
from mcda_studio.domain.results import SmaaResult
from mcda_studio.services import analysis_service
from mcda_studio.services import screening_robustness_service as svc
from mcda_studio.storage.json_repository import JsonProjectRepository

STUDY = Path(__file__).parents[2] / "examples" / "nz_venture_screening" / "project.json"
PROFILES = ["technical_ip", "commercial_traction"]


def _range(counts: list[int], preferred: int) -> tuple[int, int]:
    return svc.rank_range_from_counts(counts, sum(counts), preferred)


# Hand-built acceptabilities --------------------------------------------------------------------


def test_a_one_hot_matrix_separates_every_alternative() -> None:
    ranges = [(f"a{r}", _range([100 if i == r else 0 for i in range(4)], r + 1)) for r in range(4)]
    assert [interval for _, interval in ranges] == [(1, 1), (2, 2), (3, 3), (4, 4)]
    assert svc.overlap_groups(ranges) == [["a0"], ["a1"], ["a2"], ["a3"]]


def test_a_uniform_matrix_needs_every_rank_and_joins_everyone() -> None:
    # 75 % fits in three ranks, short of 80 %: the whole run is needed.
    ranges = [(f"a{i}", _range([25, 25, 25, 25], i + 1)) for i in range(4)]
    assert {interval for _, interval in ranges} == {(1, 4)}
    assert svc.overlap_groups(ranges) == [["a0", "a1", "a2", "a3"]]


@pytest.mark.parametrize(
    ("preferred", "expected"),
    [(1, (1, 2)), (2, (1, 2)), (3, (2, 3)), (4, (1, 2))],
)
def test_ties_go_to_the_run_holding_the_mavt_rank_then_the_lower_ranks(
    preferred: int, expected: tuple[int, int]
) -> None:
    # [1, 2] and [2, 3] both hold exactly 80 %.
    assert _range([10, 70, 10, 10], preferred) == expected
    assert oracle.rank_range([10, 70, 10, 10], preferred, 100) == expected


def test_exactly_the_coverage_is_enough_and_one_draw_less_is_not() -> None:
    assert _range([80, 20], 1) == (1, 1)
    assert _range([79, 21], 1) == (1, 2)
    assert _range([8000, 2000], 2) == (1, 1)


def test_enumerated_probabilities_reach_the_coverage_despite_binary_rounding() -> None:
    # 0.1 + 0.7 sums to 0.7999999999999999 in binary; it is 80 % as stated.
    assert svc.rank_range_from_probabilities([0.1, 0.7, 0.1, 0.1], 2) == (1, 2)
    assert oracle.rank_range([0.1, 0.7, 0.1, 0.1], 2) == (1, 2)
    assert svc.rank_range_from_probabilities([0.1, 0.69, 0.1, 0.11], 2) == (1, 3)


def test_groups_are_interval_components_not_a_sweep_in_mavt_order() -> None:
    # In MAVT order A [3-4], B [1-1], C [2-3]: B overlaps nothing; A and C share rank 3.
    ranges = [("A", (3, 4)), ("B", (1, 1)), ("C", (2, 3))]
    assert svc.overlap_groups(ranges) == [["A", "C"], ["B"]]
    assert oracle.groups([(a, lo, hi) for a, (lo, hi) in ranges]) == [["A", "C"], ["B"]]


def test_touching_ranges_overlap_and_adjacent_ones_do_not() -> None:
    assert svc.overlap_groups([("a", (2, 4)), ("b", (4, 7))]) == [["a", "b"]]
    assert svc.overlap_groups([("a", (2, 3)), ("b", (4, 5))]) == [["a"], ["b"]]


# Generated acceptabilities ---------------------------------------------------------------------

rows = st.integers(min_value=1, max_value=8).flatmap(
    lambda m: st.lists(
        st.lists(st.integers(min_value=0, max_value=50), min_size=m, max_size=m).filter(any),
        min_size=m,
        max_size=m,
    )
)


@settings(max_examples=300, deadline=None)
@given(rows)
def test_generated_matrices_match_the_oracle_and_hold_the_properties(
    matrix: list[list[int]],
) -> None:
    m = len(matrix)
    ranges = []
    for i, counts in enumerate(matrix):
        preferred = i + 1
        n = sum(counts)
        low, high = svc.rank_range_from_counts(counts, n, preferred)
        assert (low, high) == oracle.rank_range(counts, preferred, n)
        assert 1 <= low <= high <= m
        assert sum(counts[low - 1 : high]) * 5 >= 4 * n
        width = high - low + 1
        for start in range(m - width + 2):  # no shorter run reaches the coverage
            assert sum(counts[start : start + width - 1]) * 5 < 4 * n
        ranges.append((f"a{i}", (low, high)))
    found = svc.overlap_groups(ranges)
    assert found == oracle.groups([(a, lo, hi) for a, (lo, hi) in ranges])
    members = [a for group in found for a in group]
    assert sorted(members) == sorted(a for a, _ in ranges) and len(members) == m


# Case 2's real SMAA-2 output -------------------------------------------------------------------


def _study():
    return JsonProjectRepository().load(STUDY)


@pytest.mark.parametrize("profile", PROFILES)
def test_case_2_bands_match_the_oracle_on_real_smaa_output(profile: str) -> None:
    project = _study()
    configuration = project.analysis.model_copy(update={"weight_profile_id": profile})
    bands = svc.screening_robustness(project, configuration)
    smaa = analysis_service.evaluate_project(
        project, configuration.model_copy(update={"method": "smaa2", "parameters": {}})
    )
    assert isinstance(smaa, SmaaResult) and smaa.samples is not None
    counts = {a.alternative_id: [e.count for e in a.ranks] for a in smaa.acceptability}
    expected = []
    for ranked in smaa.ranking:
        low, high = oracle.rank_range(counts[ranked.alternative_id], ranked.rank, smaa.samples)
        expected.append((ranked.alternative_id, ranked.rank, low, high))
    assert bands.available and bands.draws == 10_000 and bands.seed == 0
    assert [
        (r.alternative_id, r.mavt_rank, r.rank_low, r.rank_high) for r in bands.rows
    ] == expected
    assert bands.groups == oracle.groups([(a, lo, hi) for a, _, lo, hi in expected])
    for row in bands.rows:
        held = sum(counts[row.alternative_id][row.rank_low - 1 : row.rank_high])
        assert row.band_share == held / smaa.samples >= 0.8
        assert row.rank1_share == counts[row.alternative_id][0] / smaa.samples


def test_case_2_reads_as_the_design_note_says() -> None:
    project = _study()
    technical = svc.screening_robustness(
        project, project.analysis.model_copy(update={"weight_profile_id": "technical_ip"})
    )
    middle = ["wellumio", "hectre", "wych", "captivate", "scentian_bio", "hnry"]
    assert technical.groups == [["upstock"], middle, ["kea_aerospace"]]
    assert [(r.rank_low, r.rank_high) for r in technical.rows] == [
        (1, 1), (2, 6), (2, 4), (2, 4), (4, 7), (3, 6), (5, 7), (8, 8),
    ]  # fmt: skip
    assert technical.rows[0].rank1_share == 0.9302
    assert technical.basis == (
        "Varied: 87 stated measurement ranges; the swing ratings at the top level within stated "
        "intervals, in place of this profile's. Fixed: other weights (this profile's), value "
        "functions, class boundaries. 10,000 draws, seed 0, SMAA-2 defaults (no saved SMAA-2 "
        "parameters)."
    )


def test_the_same_seed_gives_the_same_bands() -> None:
    project = _study()
    assert svc.screening_robustness(project) == svc.screening_robustness(project)


def test_the_oracle_imports_nothing_from_mcda_studio() -> None:
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import robustness_oracle; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'mcda_studio'))"
    )
    run = subprocess.run(
        [sys.executable, "-I", "-c", code, str(Path(__file__).parent)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert run.stdout.strip() == "[]"


def test_saved_smaa_parameters_are_used_and_named() -> None:
    project = _study()
    saved = project.model_copy(
        update={
            "analysis": AnalysisConfiguration(
                method="smaa2",
                parameters={"uncertainty": "performance", "samples": 2000, "seed": 7},
            )
        }
    )
    bands = svc.screening_robustness(saved)
    assert (bands.run, bands.draws, bands.seed, bands.parameters) == (
        "performance",
        2000,
        7,
        "saved",
    )
    assert bands.basis.startswith("Varied: 87 stated measurement ranges. Fixed: every weight")
    assert "2,000 draws, seed 7, saved SMAA-2 parameters." in bands.basis
