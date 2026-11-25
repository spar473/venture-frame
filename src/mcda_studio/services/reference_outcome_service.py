"""Version 2 reference sets: validation and an observation-aware outcome summary.

See ``docs/design/reference-cases.md`` (version 2). The summary is bookkeeping, not a model: it
counts what was recorded, keeps censored and unknown cases apart from cases with no event, and
gives the range the event share could take if every case were followed to the end of its window.
It predicts nothing for any alternative. Ratings are counted by review state, so a proposed rating
is never reported as reviewed.

Read-only and opt-in: nothing here changes a score, rank, class or stored project byte.
"""

import datetime as dt
from collections import Counter
from fractions import Fraction
from typing import Any

from mcda_studio.domain.common import DomainModel, Id, Issue
from mcda_studio.domain.errors import McdaError
from mcda_studio.domain.project import Project
from mcda_studio.domain.reference_cases import ReferenceCasesInvalid
from mcda_studio.domain.reference_cases_v2 import (
    OutcomeStatus,
    ReferenceCasesV2,
    add_months,
    date_bounds,
)
from mcda_studio.domain.validation import ValidationReport
from mcda_studio.methods.smaa.intervals import clopper_pearson
from mcda_studio.services.reference_case_service import (
    ReferenceCasesValidation,
    _issue,
    check_claims_and_ratings,
    fraction_text,
)

CONFIDENCE = 0.95

LABEL = (
    "These are counts of recorded outcomes in one reference set, not a chance of success for any "
    "alternative. Censored and unknown cases are never counted as having no event. The share among "
    "fully observed cases leaves out every censored and unknown case, so it can be biased either "
    "way; the range shows every share the set allows if each case were followed to the end of its "
    "window. A small reference set is not representative, and public sources under-report cases "
    "that closed."
)
LIMITATIONS = [
    "The interval on the observed share treats fully observed cases as independent draws from "
    "one population; a reference set is chosen by its author, not sampled.",
    "Cases followed for longer are more likely to show an event, so the observed share is not "
    "corrected for follow-up time. A time-to-event model belongs to the calibration work, not to "
    "this summary.",
    "A competing event (the case ended another way first) is counted as fully observed without "
    "the event; it is reported separately and is not read as better or worse than no event.",
    "Coverage counts ratings by review state; a proposed rating is not a reviewed one.",
]


def _v2_source_date(source: Any) -> tuple[dt.date, str] | None:
    """The latest day a partial source date can mean, so the snapshot guard stays conservative."""
    if source.date is None:
        return None
    return date_bounds(source.date)[1], source.date


def window_end(snapshot: dt.date, window_months: int) -> dt.date:
    """The last day of the observation window that opens on the snapshot day."""
    return add_months(snapshot, window_months)


def validate(project: Project, reference_set: ReferenceCasesV2) -> ReferenceCasesValidation:
    """Check a version 2 reference set against the project. Read-only."""
    issues: list[Issue] = []
    tree = project.tree()
    known = set(tree.all_ids())
    if reference_set.criteria_from != project.id:
        issues.append(
            _issue(
                "REFERENCE_PROJECT_MISMATCH",
                f"the reference set uses the criteria of project {reference_set.criteria_from!r}, "
                f"not {project.id!r}",
                "criteria_from",
            )
        )
    for key, source in reference_set.sources.items():
        if source.date is not None and date_bounds(source.date)[0] > reference_set.observed_through:
            issues.append(
                _issue(
                    "REFERENCE_SOURCE_AFTER_FOLLOW_UP",
                    f"source {key!r} is dated {source.date}, after the follow-up date "
                    f"{reference_set.observed_through.isoformat()}",
                    f"sources.{key}.date",
                )
            )
    seen: set[str] = set()
    for c, case in enumerate(reference_set.cases):
        at = f"cases[{c}]"
        if case.id in seen:
            issues.append(
                _issue("REFERENCE_DUPLICATE_CASE", f"case id {case.id!r} is repeated", f"{at}.id")
            )
        seen.add(case.id)
        if case.snapshot_date > reference_set.observed_through:
            issues.append(
                _issue(
                    "REFERENCE_SNAPSHOT_AFTER_FOLLOW_UP",
                    f"the snapshot {case.snapshot_date.isoformat()} is after the follow-up date "
                    f"{reference_set.observed_through.isoformat()}",
                    f"{at}.snapshot_date",
                )
            )
        claims = check_claims_and_ratings(
            tree,
            known,
            reference_set.sources,
            case,
            at,
            issues,
            _v2_source_date,
            scope="the reference set",
        )
        recorded: set[str] = set()
        for o, outcome in enumerate(case.outcomes):
            _check_outcome(reference_set, case, at, o, outcome, claims, recorded, issues)
    return ReferenceCasesValidation(
        reference_set_id=reference_set.id,
        valid=not issues,
        issues=issues,
        case_count=len(reference_set.cases),
        rating_count=sum(len(case.ratings) for case in reference_set.cases),
    )


def _check_outcome(
    reference_set: ReferenceCasesV2,
    case: Any,
    at: str,
    o: int,
    outcome: Any,
    claims: dict[str, Any],
    recorded: set[str],
    issues: list[Issue],
) -> None:
    where = f"{at}.outcomes[{o}]"
    definition = reference_set.outcome_definitions.get(outcome.definition)
    if definition is None:
        issues.append(
            _issue(
                "OUTCOME_UNKNOWN_DEFINITION",
                f"outcome definition {outcome.definition!r} is not in the reference set",
                f"{where}.definition",
            )
        )
    if outcome.definition in recorded:
        issues.append(
            _issue(
                "OUTCOME_DUPLICATE_DEFINITION",
                f"outcome definition {outcome.definition!r} is recorded more than once for "
                "this case",
                f"{where}.definition",
            )
        )
    recorded.add(outcome.definition)
    for k, claim_id in enumerate(outcome.claim_ids):
        if claim_id not in claims:
            issues.append(
                _issue(
                    "REFERENCE_UNKNOWN_CLAIM",
                    f"claim {claim_id!r} is not among this case's claims",
                    f"{where}.claim_ids[{k}]",
                )
            )
    status: OutcomeStatus = outcome.status
    snapshot = case.snapshot_date
    follow_up = reference_set.observed_through
    dated = status in ("event", "competing_event")
    if dated:
        if outcome.event_date is None:
            issues.append(
                _issue(
                    "OUTCOME_EVENT_DATE_REQUIRED",
                    f"{status} needs the date it happened",
                    f"{where}.event_date",
                )
            )
        if outcome.observed_through is not None:
            issues.append(
                _issue(
                    "OUTCOME_UNEXPECTED_OBSERVED_THROUGH",
                    f"{status} is dated by event_date; observed_through is for no_event and "
                    "censored",
                    f"{where}.observed_through",
                )
            )
        if not outcome.claim_ids:
            issues.append(
                _issue(
                    "OUTCOME_EVENT_UNSUPPORTED",
                    f"{status} needs at least one claim behind it",
                    f"{where}.claim_ids",
                )
            )
    else:
        if outcome.event_date is not None:
            issues.append(
                _issue(
                    "OUTCOME_UNEXPECTED_EVENT_DATE",
                    f"only an event or a competing event has an event date; this outcome is "
                    f"{status}",
                    f"{where}.event_date",
                )
            )
        if status == "unknown" and outcome.observed_through is not None:
            issues.append(
                _issue(
                    "OUTCOME_UNEXPECTED_OBSERVED_THROUGH",
                    "unknown means no follow-up; record censored with the date it was watched to",
                    f"{where}.observed_through",
                )
            )
        if status in ("no_event", "censored") and outcome.observed_through is None:
            issues.append(
                _issue(
                    "OUTCOME_OBSERVED_THROUGH_REQUIRED",
                    f"{status} needs the last day the case was watched",
                    f"{where}.observed_through",
                )
            )
    if definition is None:
        return
    end = window_end(snapshot, definition.window_months)
    if dated and outcome.event_date is not None:
        first, last = date_bounds(outcome.event_date)
        path = f"{where}.event_date"
        if last < snapshot:
            issues.append(
                _issue(
                    "OUTCOME_EVENT_BEFORE_SNAPSHOT",
                    f"the {status} ({outcome.event_date}) is before the snapshot "
                    f"{snapshot.isoformat()}",
                    path,
                )
            )
        elif first > end:
            issues.append(
                _issue(
                    "OUTCOME_EVENT_AFTER_WINDOW",
                    f"the {status} ({outcome.event_date}) is after the window ends on "
                    f"{end.isoformat()}; for this definition record no_event",
                    path,
                )
            )
        elif first < snapshot or last > end:
            issues.append(
                _issue(
                    "OUTCOME_EVENT_DATE_TOO_COARSE",
                    f"the {status} date {outcome.event_date} could fall outside the window "
                    f"{snapshot.isoformat()} to {end.isoformat()}; record a finer date or "
                    "record the outcome as censored",
                    path,
                )
            )
        if first > follow_up:
            issues.append(
                _issue(
                    "OUTCOME_AFTER_FOLLOW_UP",
                    f"the {status} ({outcome.event_date}) is after the follow-up date "
                    f"{follow_up.isoformat()}",
                    path,
                )
            )
    seen_to = outcome.observed_through
    if status in ("no_event", "censored") and seen_to is not None:
        path = f"{where}.observed_through"
        if seen_to < snapshot:
            issues.append(
                _issue(
                    "OUTCOME_OBSERVED_BEFORE_SNAPSHOT",
                    f"watched only to {seen_to.isoformat()}, before the snapshot "
                    f"{snapshot.isoformat()}; record unknown",
                    path,
                )
            )
        elif status == "no_event" and seen_to < end:
            issues.append(
                _issue(
                    "OUTCOME_WINDOW_NOT_COMPLETE",
                    f"watched to {seen_to.isoformat()}, before the window ends on "
                    f"{end.isoformat()}; record censored",
                    path,
                )
            )
        elif status == "censored" and seen_to >= end:
            issues.append(
                _issue(
                    "OUTCOME_CENSORED_AFTER_WINDOW",
                    f"watched to {seen_to.isoformat()}, through the end of the window on "
                    f"{end.isoformat()}; record no_event",
                    path,
                )
            )
        if seen_to > follow_up:
            issues.append(
                _issue(
                    "OUTCOME_AFTER_FOLLOW_UP",
                    f"watched to {seen_to.isoformat()}, after the follow-up date "
                    f"{follow_up.isoformat()}",
                    path,
                )
            )


def check(project: Project, reference_set: ReferenceCasesV2) -> None:
    """Raise ``ReferenceCasesInvalid`` with every issue when the reference set does not fit."""
    report = validate(project, reference_set)
    if not report.valid:
        raise ReferenceCasesInvalid(ValidationReport(issues=report.issues))


# ---- the summary -----------------------------------------------------------------------------


class ShareText(DomainModel):
    value: float
    exact: str  # "p/q"


class OutcomeCounts(DomainModel):
    event: int
    no_event: int
    competing_event: int
    censored: int
    unknown: int
    not_recorded: int  # the case has no observation for this definition


class OutcomeSummary(DomainModel):
    definition_id: Id
    label: str
    window_months: int
    case_count: int
    counts: OutcomeCounts
    fully_observed: int  # event + no_event + competing_event
    observed_share: ShareText | None = None  # events among fully observed cases
    observed_interval: tuple[float, float] | None = None  # Clopper-Pearson, CONFIDENCE
    lowest_share: ShareText | None = None  # every unresolved case ends without the event
    highest_share: ShareText | None = None  # every unresolved case ends with the event


class LeafCoverage(DomainModel):
    criterion_id: Id
    criterion_name: str
    reviewed: int
    proposed: int
    missing: int


class ReferenceOutcomesResult(DomainModel):
    project_fingerprint: str
    reference_set_id: Id
    reference_set_title: str
    observed_through: dt.date
    case_count: int
    confidence: float = CONFIDENCE
    outcomes: list[OutcomeSummary]
    coverage: list[LeafCoverage]
    reviewed_rating_count: int
    proposed_rating_count: int
    label: str = LABEL
    limitations: list[str] = LIMITATIONS


def _share(numerator: int, denominator: int) -> ShareText | None:
    if denominator == 0:
        return None
    value = Fraction(numerator, denominator)
    return ShareText(value=float(value), exact=fraction_text(value))


def summarise_definition(reference_set: ReferenceCasesV2, definition_id: str) -> OutcomeSummary:
    definition = reference_set.outcome_definitions.get(definition_id)
    if definition is None:
        raise McdaError(f"outcome definition {definition_id!r} is not in the reference set")
    tally: Counter[str] = Counter()
    for case in reference_set.cases:
        statuses = [o.status for o in case.outcomes if o.definition == definition_id]
        tally[statuses[0] if statuses else "not_recorded"] += 1
    n = len(reference_set.cases)
    resolved_without = tally["no_event"] + tally["competing_event"]
    complete = tally["event"] + resolved_without
    return OutcomeSummary(
        definition_id=definition_id,
        label=definition.label,
        window_months=definition.window_months,
        case_count=n,
        counts=OutcomeCounts(
            event=tally["event"],
            no_event=tally["no_event"],
            competing_event=tally["competing_event"],
            censored=tally["censored"],
            unknown=tally["unknown"],
            not_recorded=tally["not_recorded"],
        ),
        fully_observed=complete,
        observed_share=_share(tally["event"], complete),
        observed_interval=(
            clopper_pearson(tally["event"], complete, CONFIDENCE) if complete else None
        ),
        lowest_share=_share(tally["event"], n),
        highest_share=_share(n - resolved_without, n),
    )


def coverage(project: Project, reference_set: ReferenceCasesV2) -> list[LeafCoverage]:
    """For every labelled leaf, how many cases rate it, by review state, and how many do not."""
    tree = project.tree()
    rows: list[LeafCoverage] = []
    for leaf in tree.leaves():
        scale = tree.node(leaf).scale
        if scale is None or not scale.is_labelled:
            continue
        reviewed = proposed = 0
        for case in reference_set.cases:
            states = [r.review for r in case.ratings if r.criterion_id == leaf]
            if states and states[0] == "reviewed":
                reviewed += 1
            elif states:
                proposed += 1
        rows.append(
            LeafCoverage(
                criterion_id=leaf,
                criterion_name=tree.node(leaf).name,
                reviewed=reviewed,
                proposed=proposed,
                missing=len(reference_set.cases) - reviewed - proposed,
            )
        )
    return rows


def reference_outcomes(
    project: Project, reference_set: ReferenceCasesV2
) -> ReferenceOutcomesResult:
    """Validate the set, then summarise every outcome definition and the rating coverage."""
    check(project, reference_set)
    ratings = [r for case in reference_set.cases for r in case.ratings]
    return ReferenceOutcomesResult(
        project_fingerprint=project.fingerprint(),
        reference_set_id=reference_set.id,
        reference_set_title=reference_set.title,
        observed_through=reference_set.observed_through,
        case_count=len(reference_set.cases),
        outcomes=[
            summarise_definition(reference_set, key) for key in reference_set.outcome_definitions
        ],
        coverage=coverage(project, reference_set),
        reviewed_rating_count=sum(r.review == "reviewed" for r in ratings),
        proposed_rating_count=sum(r.review == "proposed" for r in ratings),
    )
