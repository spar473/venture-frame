"""The ``reference-cases`` sidecar, version 2 (``docs/design/reference-cases.md``, version 2).

Version 2 keeps the version 1 envelope, claims, context and snapshot guard and adds what an
outcome model needs and version 1 cannot express:

- **Outcome definitions** with an observation window, named once in the envelope.
- **Outcome observations** per case and definition, with a status that keeps what was not seen
  apart from what did not happen: ``event``, ``no_event`` (watched through the whole window),
  ``competing_event`` (a different terminal event ended the case first, so the event can no
  longer happen), ``censored`` (watched for part of the window, no event seen) and ``unknown``
  (no follow-up).
- **Partial dates** (``YYYY``, ``YYYY-Qn``, ``YYYY-MM`` or ``YYYY-MM-DD``) for sources and event
  dates, so a coarse public date is recorded as coarse instead of being invented as a day.
- **A review state** on every rating: ``proposed`` (for example by a tool) or ``reviewed`` by a
  person. Nothing downstream may treat a proposed rating as reviewed.

The structural rules live here. Rules that need the project, the window arithmetic and the
snapshot guard live in ``services/reference_outcome_service.py``. Version 1 files are read by
``domain/reference_cases.py`` exactly as before.
"""

import calendar
import datetime as dt
import re
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, ConfigDict, Field, model_validator

from mcda_studio.domain.common import Id
from mcda_studio.domain.reference_cases import (
    KIND,
    ContextRow,
    IsoDate,
    Note,
    ReferenceClaim,
    ReferenceSource,
    ShortText,
    Text,
)
from mcda_studio.domain.sidecar_common import SidecarModel

VERSION = 2
WINDOW_MONTHS_MAX = 240

_PARTIAL = re.compile(r"^(\d{4})(?:-(?:Q([1-4])|(\d{2})(?:-(\d{2}))?))?$")


def date_bounds(text: str) -> tuple[dt.date, dt.date]:
    """The first and last calendar day a partial date can mean."""
    match = _PARTIAL.match(text)
    if match is None:
        raise ValueError(
            "partial dates are YYYY, YYYY-Qn, YYYY-MM or YYYY-MM-DD (for example 2019, 2019-Q2, "
            "2019-06 or 2019-06-30)"
        )
    year = int(match.group(1))
    quarter, month, day = match.group(2), match.group(3), match.group(4)
    if day is not None:
        exact = dt.date.fromisoformat(text)
        return exact, exact
    if month is not None:
        m = int(month)
        if not 1 <= m <= 12:
            raise ValueError(f"{text!r} has no month {m}")
        return dt.date(year, m, 1), dt.date(year, m, calendar.monthrange(year, m)[1])
    if quarter is not None:
        first = 3 * (int(quarter) - 1) + 1
        last = first + 2
        return dt.date(year, first, 1), dt.date(year, last, calendar.monthrange(year, last)[1])
    return dt.date(year, 1, 1), dt.date(year, 12, 31)


def _partial_date(value: str) -> str:
    date_bounds(value)  # raises with the accepted forms
    return value


PartialDate = Annotated[str, AfterValidator(_partial_date)]


def add_months(day: dt.date, months: int) -> dt.date:
    """Calendar months later; a day past the end of the target month moves to its last day."""
    index = day.month - 1 + months
    year, month = day.year + index // 12, index % 12 + 1
    return dt.date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


ReviewState = Literal["proposed", "reviewed"]
OutcomeStatus = Literal["event", "no_event", "competing_event", "censored", "unknown"]


class ReferenceSourceV2(ReferenceSource):
    """The version 1 source, with a partial publication or capture date."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
        json_schema_serialization_defaults_required=True,
    )

    date: PartialDate | None = None  # type: ignore[assignment]


class ReferenceRatingV2(SidecarModel):
    """A scale label for one leaf criterion as of the snapshot, with who stands behind it."""

    criterion_id: Id
    label: Annotated[str, Field(min_length=1, max_length=300)]
    claim_ids: Annotated[list[Id], Field(min_length=1, max_length=50)]
    review: ReviewState
    note: Note = ""


class OutcomeDefinition(SidecarModel):
    """What counts as the event, and how long after the snapshot a case is watched for it."""

    label: ShortText
    window_months: Annotated[int, Field(ge=1, le=WINDOW_MONTHS_MAX)]
    description: Text


class OutcomeObservation(SidecarModel):
    """What was seen of one outcome definition for one case.

    ``event``: the event happened inside the window, on ``event_date``. ``no_event``: the case was
    watched through the end of the window (``observed_through``) and no event was seen.
    ``competing_event``: a different terminal event, dated ``event_date`` inside the window, ended
    the case before any event was seen, so none can follow; the note says what it was.
    ``censored``: watched up to ``observed_through``, before the window ended, and no event was
    seen. ``unknown``: no follow-up. Censored and unknown are never counted as no event.
    """

    definition: Id  # a key of ``ReferenceCasesV2.outcome_definitions``
    status: OutcomeStatus
    event_date: PartialDate | None = None
    observed_through: IsoDate | None = None
    claim_ids: Annotated[list[Id], Field(max_length=50)] = []
    note: Note = ""


class ReferenceCaseV2(SidecarModel):
    id: Id
    name: ShortText
    snapshot_date: IsoDate
    context: Annotated[list[ContextRow], Field(max_length=50)] = []
    claims: Annotated[list[ReferenceClaim], Field(max_length=500)] = []
    ratings: Annotated[list[ReferenceRatingV2], Field(max_length=500)] = []
    outcomes: Annotated[list[OutcomeObservation], Field(max_length=20)] = []


class ReferenceCasesV2(SidecarModel):
    """A version 2 reference set: rated past alternatives with observation-aware outcomes."""

    kind: Literal["reference-cases"]
    version: Literal[2]
    id: Id
    title: ShortText
    description: Annotated[str, Field(max_length=4000)] = ""
    criteria_from: Id
    observed_through: IsoDate  # the day the follow-up behind every outcome was compiled
    outcome_definitions: Annotated[dict[Id, OutcomeDefinition], Field(min_length=1, max_length=20)]
    cases: Annotated[list[ReferenceCaseV2], Field(max_length=500)] = []
    sources: Annotated[dict[Id, ReferenceSourceV2], Field(max_length=5000)] = {}

    @model_validator(mode="before")
    @classmethod
    def known_kind_and_version(cls, value: Any) -> Any:
        if isinstance(value, dict):
            if "kind" in value and value["kind"] != KIND:
                raise ValueError(f"this is not a reference set (kind must be {KIND!r})")
            version = value.get("version")
            if type(version) is int and version != VERSION:
                raise ValueError(
                    f"this reader takes reference-cases version {VERSION}; version {version} "
                    "is read by the version 1 reader"
                    if version == 1
                    else f"reference-cases version {version} is not supported"
                )
        return value
