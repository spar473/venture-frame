"""The ``rating-proposals`` sidecar, version 1 (``docs/design/reference-rating.md``).

Proposed ratings for the cases of a version 2 reference set, each resting on one dated public
source, an excerpt and a locator. A proposal is never a rating: it becomes one only when a person
accepts or edits it in a review sheet and the importer writes it with ``review: "reviewed"``.
"Not enough evidence" is a proposal of its own and leaves the criterion unrated (missing, not zero).

Strict sidecar conventions: unknown fields are refused, ``version`` must be an explicit integer,
and any version other than 1 is refused naming the supported one. Rules that need the project or the
reference set live in ``services/reference_rating_service.py``.
"""

import re
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, model_validator

from mcda_studio.domain.common import Id
from mcda_studio.domain.evidence import Origin
from mcda_studio.domain.reference_cases import IsoDate, Note
from mcda_studio.domain.reference_cases_v2 import PartialDate
from mcda_studio.domain.sidecar_common import SidecarModel

KIND = "rating-proposals"
VERSION = 1
EXCERPT_WORDS_MAX = 25

#: Short enough that the claim id ``rp_<id>`` is still an engine id.
ProposalId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,59}$")]
Confidence = Literal["high", "medium", "low"]
_URL = re.compile(r"^https?://\S+$")


def excerpt_words(text: str) -> int:
    return len(text.split())


class RatingProposal(SidecarModel):
    """One proposed label (or "not enough evidence") for one criterion of one reference case."""

    id: ProposalId
    case_id: Id
    criterion_id: Id
    label: Annotated[str, Field(min_length=1, max_length=300)] | None = None
    not_enough_evidence: bool = False
    claim: Annotated[str, Field(min_length=1, max_length=600, pattern=r"\S")]
    source_url: Annotated[str, Field(max_length=2000)] | None = None
    source_title: Annotated[str, Field(max_length=300)] = ""
    source_date: PartialDate | None = None
    locator: Annotated[str, Field(max_length=300)] = ""
    excerpt: Annotated[str, Field(max_length=400)] = ""
    confidence: Confidence
    note: Note = ""

    @model_validator(mode="after")
    def evidence(self) -> "RatingProposal":
        if (self.label is None) == (not self.not_enough_evidence):
            raise ValueError("give exactly one of a label and not_enough_evidence: true")
        if self.source_url is not None and not _URL.match(self.source_url):
            raise ValueError("source_url must be an http(s) URL")
        if excerpt_words(self.excerpt) > EXCERPT_WORDS_MAX:
            raise ValueError(f"the excerpt has more than {EXCERPT_WORDS_MAX} words")
        if self.label is not None:
            missing = [
                name
                for name, value in (
                    ("source_url", self.source_url),
                    ("source_date", self.source_date),
                    ("locator", self.locator.strip()),
                    ("excerpt", self.excerpt.strip()),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    "a proposed label needs a dated source, a locator and an excerpt; missing: "
                    + ", ".join(missing)
                )
        return self


class RatingProposals(SidecarModel):
    kind: Literal["rating-proposals"]
    version: Literal[1]
    reference_set_id: Id
    criteria_from: Id  # the project whose criteria the proposals rate
    proposer: Origin
    created_on: IsoDate
    proposals: Annotated[list[RatingProposal], Field(max_length=5000)] = []

    @model_validator(mode="before")
    @classmethod
    def known_kind_and_version(cls, value: Any) -> Any:
        if isinstance(value, dict):
            if "kind" in value and value["kind"] != KIND:
                raise ValueError(f"this is not a rating-proposals file (kind must be {KIND!r})")
            version = value.get("version")
            if type(version) is int and version != VERSION:
                raise ValueError(
                    f"rating-proposals version {version} is not supported; this reader takes "
                    f"version {VERSION}"
                )
        return value

    @model_validator(mode="after")
    def unique_ids(self) -> "RatingProposals":
        seen: set[str] = set()
        for proposal in self.proposals:
            if proposal.id in seen:
                raise ValueError(f"proposal id {proposal.id!r} is repeated")
            seen.add(proposal.id)
        return self
