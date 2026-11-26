"""Reference sets version 2: partial dates, window arithmetic, every outcome validation rule, the
snapshot guard on coarse source dates, the observation-aware summary and rating coverage.

Oracles are independent of the code under test: partial dates and calendar months are checked by
walking every day of a calendar, and summary numbers are hand counts from the fixture below.
"""

import copy
import datetime as dt
import json
from typing import Any

import pytest
from pydantic import ValidationError
from scipy import stats

import reference_case_builders as b
from mcda_studio.domain.reference_cases import ReferenceCases, ReferenceCasesInvalid
from mcda_studio.domain.reference_cases_v2 import ReferenceCasesV2, add_months, date_bounds
from mcda_studio.services import reference_case_service as v1
from mcda_studio.services import reference_outcome_service as svc

SNAPSHOT = b.SNAPSHOT  # 2021-10-27; a 36-month window ends 2024-10-27
END = dt.date(2024, 10, 27)
FOLLOW_UP = dt.date(2026, 11, 20)


def outcome(status: str, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"definition": "next_36m", "status": status, **extra}
    if status in ("event", "competing_event"):
        out.setdefault("event_date", "2023-05-02")
        out.setdefault("claim_ids", ["c2"])
    if status == "no_event":
        out.setdefault("observed_through", FOLLOW_UP.isoformat())
    if status == "censored":
        out.setdefault("observed_through", "2023-01-31")
    return out


def case(case_id: str, labels: dict[str, str], *outcomes: dict[str, Any], **kw: Any) -> dict:
    review = kw.pop("review", "reviewed")
    raw = b.case(case_id, labels, **kw)
    del raw["outcome"]
    for rating in raw["ratings"]:
        rating["review"] = review
    raw["outcomes"] = list(outcomes)
    return raw


def reference_set(*cases: dict[str, Any], **sources: dict[str, Any]) -> dict[str, Any]:
    raw = b.reference_set(*cases, **sources)
    raw["version"] = 2
    raw["observed_through"] = FOLLOW_UP.isoformat()
    raw["outcome_definitions"] = {
        "next_36m": {
            "label": "Next event within 36 months",
            "window_months": 36,
            "description": "A synthetic event for tests.",
        }
    }
    return raw


def load(raw: dict[str, Any]) -> ReferenceCasesV2:
    return ReferenceCasesV2.model_validate_json(json.dumps(raw))


def issues(raw: dict[str, Any]) -> list[tuple[str, str]]:
    return [(i.code, i.path) for i in svc.validate(b.project(), load(raw)).issues]


HAND = {"o4": "D", "o5": "v5", "cat": "red"}


# ---- partial dates and calendar months (oracle: walk the calendar) ------------------------------


def _days(year_from: int, year_to: int) -> list[dt.date]:
    day, out = dt.date(year_from, 1, 1), []
    while day.year <= year_to:
        out.append(day)
        day += dt.timedelta(days=1)
    return out


@pytest.mark.parametrize("year", [2019, 2020, 2100, 2000])
def test_partial_dates_cover_exactly_the_days_they_name(year: int) -> None:
    days = _days(year, year)
    assert date_bounds(str(year)) == (days[0], days[-1])
    for q in range(1, 5):
        inside = [d for d in days if (d.month - 1) // 3 + 1 == q]
        assert date_bounds(f"{year}-Q{q}") == (inside[0], inside[-1])
    for m in range(1, 13):
        inside = [d for d in days if d.month == m]
        assert date_bounds(f"{year}-{m:02d}") == (inside[0], inside[-1])
    for d in days[::17]:
        assert date_bounds(d.isoformat()) == (d, d)


@pytest.mark.parametrize("text", ["2019-13", "2019-Q5", "2019-02-30", "19", "2019/06", "2019-6"])
def test_malformed_partial_dates_are_refused(text: str) -> None:
    with pytest.raises(ValueError):
        date_bounds(text)


def test_add_months_matches_counting_month_ends_on_a_calendar() -> None:
    for start in _days(2019, 2021)[::5]:
        for months in (1, 11, 12, 13, 24, 36, 60):
            index = start.month - 1 + months
            y, m = start.year + index // 12, index % 12 + 1
            month_days = [d for d in _days(y, y) if d.month == m]
            expected = month_days[min(start.day, len(month_days)) - 1]
            assert add_months(start, months) == expected


def test_window_end_is_the_snapshot_plus_whole_months() -> None:
    assert svc.window_end(SNAPSHOT, 36) == END
    assert svc.window_end(dt.date(2020, 1, 31), 1) == dt.date(2020, 2, 29)


# ---- the format ---------------------------------------------------------------------------------


def test_a_valid_v2_set_has_no_issues() -> None:
    raw = reference_set(case("a", HAND, outcome("event")), case("b", HAND, outcome("unknown")))
    report = svc.validate(b.project(), load(raw))
    assert report.valid, report.issues
    assert (report.case_count, report.rating_count) == (2, 6)


def test_versions_are_kept_apart() -> None:
    raw = reference_set(case("a", HAND))
    with pytest.raises(ValidationError, match="version 1 reader"):
        load({**raw, "version": 1})
    with pytest.raises(ValidationError, match="version 2 is not supported"):
        ReferenceCases.model_validate_json(json.dumps(raw))
    with pytest.raises(ValidationError, match="explicit integers"):
        load({**raw, "version": "2"})


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["cases"][0]["ratings"][0].pop("review"),
        lambda r: r["cases"][0]["ratings"][0].update(review="accepted"),
        lambda r: r["cases"][0]["outcomes"][0].update(status="failed"),
        lambda r: r["cases"][0]["outcomes"][0].update(event_date="2023-6"),
        lambda r: r["outcome_definitions"]["next_36m"].update(window_months=0),
        lambda r: r["outcome_definitions"]["next_36m"].update(window_months=241),
        lambda r: r.update(outcome_definitions={}),
        lambda r: r.pop("observed_through"),
        lambda r: r["cases"][0].update(outcome={"label": "x", "as_of": "2024-01-01"}),
        lambda r: r["sources"]["before"].update(date="Sept 2021"),
    ],
)
def test_structural_errors_are_refused(change: Any) -> None:
    raw = reference_set(case("a", HAND, outcome("event")))
    change(raw)
    with pytest.raises(ValidationError):
        load(raw)


# ---- outcome rules --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("observation", "expected"),
    [
        (outcome("event", definition="other"), "OUTCOME_UNKNOWN_DEFINITION"),
        (outcome("event", event_date=None), "OUTCOME_EVENT_DATE_REQUIRED"),
        (outcome("event", claim_ids=[]), "OUTCOME_EVENT_UNSUPPORTED"),
        (outcome("event", observed_through="2023-06-01"), "OUTCOME_UNEXPECTED_OBSERVED_THROUGH"),
        (outcome("event", event_date="2021-09-30"), "OUTCOME_EVENT_BEFORE_SNAPSHOT"),
        (outcome("event", event_date="2024-11"), "OUTCOME_EVENT_AFTER_WINDOW"),
        (outcome("event", event_date="2021-10"), "OUTCOME_EVENT_DATE_TOO_COARSE"),
        (outcome("event", event_date="2024"), "OUTCOME_EVENT_DATE_TOO_COARSE"),
        (outcome("event", claim_ids=["c9"]), "REFERENCE_UNKNOWN_CLAIM"),
        (outcome("no_event", event_date="2023"), "OUTCOME_UNEXPECTED_EVENT_DATE"),
        (outcome("no_event", observed_through=None), "OUTCOME_OBSERVED_THROUGH_REQUIRED"),
        (outcome("no_event", observed_through="2024-10-26"), "OUTCOME_WINDOW_NOT_COMPLETE"),
        (outcome("censored", observed_through=None), "OUTCOME_OBSERVED_THROUGH_REQUIRED"),
        (outcome("censored", observed_through="2024-10-27"), "OUTCOME_CENSORED_AFTER_WINDOW"),
        (outcome("censored", observed_through="2021-10-26"), "OUTCOME_OBSERVED_BEFORE_SNAPSHOT"),
        (outcome("censored", observed_through="2026-11-21"), "OUTCOME_CENSORED_AFTER_WINDOW"),
        (outcome("unknown", observed_through="2023-01-01"), "OUTCOME_UNEXPECTED_OBSERVED_THROUGH"),
        (outcome("competing_event", event_date=None), "OUTCOME_EVENT_DATE_REQUIRED"),
        (outcome("competing_event", claim_ids=[]), "OUTCOME_EVENT_UNSUPPORTED"),
        (outcome("competing_event", event_date="2025-01"), "OUTCOME_EVENT_AFTER_WINDOW"),
        (outcome("competing_event", event_date="2021-Q4"), "OUTCOME_EVENT_DATE_TOO_COARSE"),
        (
            outcome("competing_event", observed_through="2023-06-01"),
            "OUTCOME_UNEXPECTED_OBSERVED_THROUGH",
        ),
    ],
)
def test_each_outcome_rule_has_its_code(observation: dict[str, Any], expected: str) -> None:
    codes = [code for code, _ in issues(reference_set(case("a", HAND, observation)))]
    assert expected in codes, codes


def test_boundaries_are_inclusive_where_the_window_says_so() -> None:
    edges = [
        outcome("event", event_date=SNAPSHOT.isoformat()),
        outcome("event", event_date=END.isoformat()),
        outcome("no_event", observed_through=END.isoformat()),
        outcome("censored", observed_through=SNAPSHOT.isoformat()),
        outcome("censored", observed_through=(END - dt.timedelta(days=1)).isoformat()),
        outcome("event", event_date="2022"),  # a whole year inside the window
        outcome("competing_event", event_date=END.isoformat()),
    ]
    for observation in edges:
        assert issues(reference_set(case("a", HAND, observation))) == [], observation


def test_a_definition_is_recorded_once_per_case() -> None:
    raw = reference_set(case("a", HAND, outcome("unknown"), outcome("event")))
    assert ("OUTCOME_DUPLICATE_DEFINITION", "cases[0].outcomes[1].definition") in issues(raw)


def test_nothing_can_be_seen_after_the_follow_up_date() -> None:
    raw = reference_set(case("a", HAND, outcome("event")))
    raw["observed_through"] = "2023-01-01"
    assert ("OUTCOME_AFTER_FOLLOW_UP", "cases[0].outcomes[0].event_date") in issues(raw)
    raw = reference_set(case("a", HAND, outcome("unknown"), snapshot=dt.date(2027, 1, 1)))
    assert ("REFERENCE_SNAPSHOT_AFTER_FOLLOW_UP", "cases[0].snapshot_date") in issues(raw)
    raw = reference_set(case("a", HAND), late={**b.source(None), "date": "2027"})
    assert ("REFERENCE_SOURCE_AFTER_FOLLOW_UP", "sources.late.date") in issues(raw)


# ---- the snapshot guard on coarse dates -------------------------------------------------------


@pytest.mark.parametrize(
    ("date", "allowed"),
    [
        ("2021-10-27", True),  # the snapshot day itself
        ("2021-10-28", False),
        ("2021-09", True),
        ("2021-10", False),  # could be after the 27th, so it cannot support a rating
        ("2021-Q3", True),
        ("2021-Q4", False),
        ("2020", True),
        ("2021", False),
    ],
)
def test_a_coarse_source_date_counts_as_late_as_it_can_be(date: str, allowed: bool) -> None:
    raw = reference_set(
        case("a", HAND, source_key="coarse"), coarse={**b.source(None), "date": date}
    )
    codes = [code for code, _ in issues(raw)]
    assert ("REFERENCE_SOURCE_AFTER_SNAPSHOT" not in codes) is allowed, codes


def test_version_1_rating_rules_still_apply() -> None:
    raw = reference_set(case("a", {"num": "5", "o4": "E"}, source_key="undated"))
    codes = {code for code, _ in issues(raw)}
    assert {"REFERENCE_NUMERIC_CRITERION", "REFERENCE_UNKNOWN_LABEL"} <= codes
    assert "REFERENCE_UNDATED_SOURCE" in codes
    message = next(i.message for i in svc.validate(b.project(), load(raw)).issues)
    assert "version 1" not in message


def test_version_1_messages_are_unchanged() -> None:
    raw = b.reference_set(b.case("a", {"num": "5"}))
    message = v1.validate(b.project(), b.load(raw)).issues[0].message
    assert message == (
        "criterion 'num' is numeric; version 1 of the reference set rates only labelled "
        "(ordinal or categorical) criteria"
    )


# ---- the summary (hand counts) ---------------------------------------------------------------

LOW = {"o4": "A"}
SUMMARY_SET = reference_set(
    case("e1", HAND, outcome("event")),
    case("e2", LOW, outcome("event", event_date="2024-Q1"), review="proposed"),
    case("n1", HAND, outcome("no_event")),
    case("n2", HAND, outcome("no_event")),
    case("n3", LOW, outcome("no_event"), review="proposed"),
    case("c1", HAND, outcome("censored")),
    case("u1", {}, outcome("unknown")),
    case("r1", HAND),  # nothing recorded for the definition
    case("k1", HAND, outcome("competing_event", event_date="2022-08")),
)


def test_summary_counts_keep_unresolved_cases_apart() -> None:
    result = svc.reference_outcomes(b.project(), load(SUMMARY_SET))
    (summary,) = result.outcomes
    assert summary.counts.model_dump() == {
        "event": 2,
        "no_event": 3,
        "competing_event": 1,
        "censored": 1,
        "unknown": 1,
        "not_recorded": 1,
    }
    assert summary.case_count == 9 and summary.fully_observed == 6
    assert summary.observed_share is not None and summary.observed_share.exact == "1/3"  # 2 of 6
    assert summary.lowest_share is not None and summary.lowest_share.exact == "2/9"
    assert summary.highest_share is not None and summary.highest_share.exact == "5/9"  # 9 - 3 - 1
    lo, hi = summary.observed_interval or (0.0, 0.0)
    assert lo == pytest.approx(stats.beta.ppf(0.025, 2, 5), abs=1e-12)
    assert hi == pytest.approx(stats.beta.ppf(0.975, 3, 4), abs=1e-12)


def test_the_range_always_contains_the_observed_share() -> None:
    (summary,) = svc.reference_outcomes(b.project(), load(SUMMARY_SET)).outcomes
    assert summary.lowest_share and summary.observed_share and summary.highest_share
    assert summary.lowest_share.value <= summary.observed_share.value
    assert summary.observed_share.value <= summary.highest_share.value


def test_no_fully_observed_case_gives_no_share_and_no_interval() -> None:
    raw = reference_set(case("c1", HAND, outcome("censored")), case("u1", HAND, outcome("unknown")))
    (summary,) = svc.reference_outcomes(b.project(), load(raw)).outcomes
    assert summary.observed_share is None and summary.observed_interval is None
    assert summary.lowest_share and summary.lowest_share.exact == "0/1"
    assert summary.highest_share and summary.highest_share.exact == "1/1"


def test_coverage_counts_ratings_by_review_state() -> None:
    result = svc.reference_outcomes(b.project(), load(SUMMARY_SET))
    rows = {row.criterion_id: (row.reviewed, row.proposed, row.missing) for row in result.coverage}
    # reviewed: e1 n1 n2 c1 r1 k1; proposed: e2 n3; missing: u1
    assert rows["o4"] == (6, 2, 1)
    assert rows["cat"] == (6, 0, 3)
    assert rows["o4b"] == (0, 0, 9)
    assert "num" not in rows  # numeric leaves are not rated in a reference set
    assert (result.reviewed_rating_count, result.proposed_rating_count) == (18, 2)


def test_the_summary_refuses_an_invalid_set_and_changes_nothing() -> None:
    project = b.project()
    before = project.model_dump_json()
    raw = copy.deepcopy(SUMMARY_SET)
    raw["cases"][0]["outcomes"][0]["event_date"] = "2020-01-01"
    with pytest.raises(ReferenceCasesInvalid, match="OUTCOME_EVENT_BEFORE_SNAPSHOT"):
        svc.reference_outcomes(project, load(raw))
    svc.reference_outcomes(project, load(SUMMARY_SET))
    assert project.model_dump_json() == before


def test_the_label_never_calls_the_share_a_probability_of_success() -> None:
    text = (svc.LABEL + " ".join(svc.LIMITATIONS)).lower()
    assert "not a chance of success" in text
    for word in ("funding", "series", "startup", "venture", "investor"):
        assert word not in text


# ---- the command line ---------------------------------------------------------------------------


def test_the_commands_read_version_2_and_print_the_services_json(tmp_path: Any) -> None:
    from typer.testing import CliRunner

    from mcda_studio.cli import app as cli_app
    from mcda_studio.storage.json_repository import JsonProjectRepository

    project_path = tmp_path / "p.json"
    JsonProjectRepository().save(b.project(), project_path)
    before = project_path.read_bytes()
    set_path = tmp_path / "r.json"
    set_path.write_text(json.dumps(SUMMARY_SET), encoding="utf-8")

    def run(*args: str) -> Any:
        return CliRunner().invoke(cli_app, ["reference-cases", *args])

    valid = run("validate", str(set_path), "--project", str(project_path))
    assert valid.exit_code == 0, valid.output
    assert json.loads(valid.stdout)["valid"] is True
    outcomes = run("outcomes", str(set_path), "--project", str(project_path))
    assert outcomes.exit_code == 0, outcomes.output
    expected = svc.reference_outcomes(b.project(), load(SUMMARY_SET))
    assert json.loads(outcomes.stdout) == json.loads(expected.model_dump_json())

    bad = copy.deepcopy(SUMMARY_SET)
    bad["cases"][0]["outcomes"][0]["event_date"] = "2020-01-01"
    set_path.write_text(json.dumps(bad), encoding="utf-8")
    invalid = run("validate", str(set_path), "--project", str(project_path))
    assert invalid.exit_code == 1
    assert json.loads(invalid.stdout)["issues"][0]["code"] == "OUTCOME_EVENT_BEFORE_SNAPSHOT"
    assert run("outcomes", str(set_path), "--project", str(project_path)).exit_code != 0

    v1_path = tmp_path / "v1.json"
    v1_path.write_text(json.dumps(b.reference_set(b.case("a", HAND))), encoding="utf-8")
    assert run("outcomes", str(v1_path), "--project", str(project_path)).exit_code != 0
    assert project_path.read_bytes() == before
