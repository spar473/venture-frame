"""The ``tradeoff-answers`` sidecar, version 1: a person's answers to pairwise trade-off questions.

A separate local JSON file, kept outside the repository (in ``$MCDA_DATA``), never stored inside a
project. It stores judgements only: for each question, the two alternatives as levels of the
criteria they differ on (everything else equal), or two of the project's alternatives by id, and
which one the person preferred. Value differences, implied weights and every other number are
derived from the project when the answers are read, never stored.

Strict sidecar conventions: unknown fields are refused, ``version`` must be an explicit integer,
and any version other than 1 is refused naming the supported one. Design:
``docs/design/tradeoff-elicitation.md``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from mcda_studio.domain.common import Id
from mcda_studio.domain.sidecar_common import SidecarModel

__all__ = ["KIND", "VERSION", "Choice", "Level", "TradeoffAnswer", "TradeoffAnswers"]

KIND = "tradeoff-answers"
VERSION = 1
Choice = Literal["A", "B", "skip"]
#: A level is a label of a labelled scale or a number on a numeric one.
Level = str | float


class TradeoffAnswer(SidecarModel):
    """One question and its answer.

    Either ``a`` and ``b`` give the two hypothetical alternatives' levels on the same criteria
    (every other criterion equal), or ``alternatives`` names two of the project's alternatives.
    """

    id: Id
    a: dict[Id, Level] = Field(default_factory=dict)
    b: dict[Id, Level] = Field(default_factory=dict)
    alternatives: list[Id] | None = Field(default=None, exclude_if=lambda v: v is None)
    choice: Choice
    note: str = Field(default="", exclude_if=lambda v: not v)

    @model_validator(mode="after")
    def one_form(self) -> TradeoffAnswer:
        if self.alternatives is not None:
            if self.a or self.b:
                raise ValueError("give either levels for A and B or two alternatives, not both")
            if len(self.alternatives) != 2 or self.alternatives[0] == self.alternatives[1]:
                raise ValueError("alternatives names two different alternatives")
            return self
        if not self.a or set(self.a) != set(self.b):
            raise ValueError("A and B give levels on the same criteria")
        if all(self.a[k] == self.b[k] for k in self.a):
            raise ValueError("A and B must differ on at least one criterion")
        return self


class TradeoffAnswers(SidecarModel):
    kind: Literal["tradeoff-answers"]
    version: Literal[1]
    project_id: Id
    # The project as it was when the questions were asked; a later project is read with a warning.
    project_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    weight_profile_id: Id
    answered_by: str
    answered_on: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")] | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    # "person": a person's answers. "simulated": a synthetic respondent's, for checks only.
    source: Literal["person", "simulated"]
    answers: list[TradeoffAnswer]

    @model_validator(mode="before")
    @classmethod
    def known_kind_and_version(cls, value: Any) -> Any:
        if isinstance(value, dict):
            if "kind" in value and value["kind"] != KIND:
                raise ValueError(f"this is not a tradeoff-answers file (kind must be {KIND!r})")
            version = value.get("version")
            if type(version) is int and version != VERSION:
                raise ValueError(
                    f"tradeoff-answers version {version} is not supported; this reader takes "
                    f"version {VERSION}"
                )
        return value

    @model_validator(mode="after")
    def unique_ids(self) -> TradeoffAnswers:
        ids = [a.id for a in self.answers]
        if len(ids) != len(set(ids)):
            raise ValueError("every answer has its own id")
        return self
