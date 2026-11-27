"""Aids for stating weights: swings anchored in the project's own alternatives, break-even weights,
and a reference profile set beside the person's order.

Read-only and derived; nothing here is stored or used as a default. Python supplies every number
and sentence; the interface only shows them.

- **Anchors**: for each criterion leaf with weight, the alternative worst and best on it (by partial
  value; ties go to the first in the project's order) with their levels, so a swing can be put as
  "from Kea Aerospace's level to Upstock's".
- **Break-even**: ``break_even_service`` for the configured profile.
- **Reference profile**: an ordinal ``reference-profile`` document (statements "these criteria
  above those", per variant and optionally per horizon, with citations). Each statement is compared
  with the profile's local weights in the group it names: agree, disagree or tie. Statements naming
  criteria the project lacks, or criteria in different sibling groups, are reported as not
  comparable. It is never a default and never weights.

Design: ``docs/design/tradeoff-elicitation.md`` (Part 4).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from mcda_studio.domain.common import DomainModel, Id
from mcda_studio.domain.errors import McdaError, ValidationFailed
from mcda_studio.domain.preferences import AnalysisConfiguration
from mcda_studio.domain.project import Project
from mcda_studio.domain.sidecar_common import SidecarModel
from mcda_studio.methods.mavt.aggregation import prepare_mavt
from mcda_studio.methods.mavt.weighting import compute_weights
from mcda_studio.services import break_even_service

# ---- the reference profile document --------------------------------------------------------------


class Citation(SidecarModel):
    authors: str
    year: int
    title: str = Field(default="", exclude_if=lambda v: not v)
    where: str
    url: str


class Statement(SidecarModel):
    horizon: str | None
    above: Annotated[list[Id], Field(min_length=1)]
    below: Annotated[list[Id], Field(min_length=1)]


class Variant(SidecarModel):
    id: Id
    name: str
    summary: str
    statements: Annotated[list[Statement], Field(min_length=1)]
    citations: Annotated[list[Citation], Field(min_length=1)]


class ReferenceProfile(SidecarModel):
    """Ordinal statements from cited sources; no weights, no percentages."""

    kind: Literal["reference-profile"]
    version: Literal[1]
    id: Id
    title: str
    purpose: str
    criteria_note: str
    variants: Annotated[list[Variant], Field(min_length=1)]


def load_reference_profiles(directory: Path | None) -> list[ReferenceProfile]:
    """Every ``*.json`` reference profile in ``directory`` (an examples folder), sorted by name.
    A file that does not parse is skipped, never half-read."""
    if directory is None or not directory.is_dir():
        return []
    out = []
    for path in sorted(directory.glob("*.json")):
        try:
            out.append(ReferenceProfile.model_validate_json(path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError, ValueError):
            continue
    return out


# ---- the aids ------------------------------------------------------------------------------------


class AnchorEnd(DomainModel):
    alternative_id: str
    name: str
    level: str  # the measurement as entered
    value: float  # its partial value


class Anchor(DomainModel):
    criterion_id: str
    name: str
    worst: AnchorEnd
    best: AnchorEnd
    tied_worst: int  # how many alternatives share the worst value
    tied_best: int
    swing: str


class PairCheck(DomainModel):
    above: str
    below: str
    above_name: str
    below_name: str
    partner: Literal["agrees", "disagrees", "tie", "not comparable"]


class StatementCheck(DomainModel):
    horizon: str | None
    text: str
    pairs: list[PairCheck]
    disagreements: int


class VariantCheck(DomainModel):
    id: str
    name: str
    summary: str
    citations: list[Citation]
    statements: list[StatementCheck]
    disagreements: int


class ReferenceComparison(DomainModel):
    reference_id: str
    title: str
    purpose: str
    criteria_note: str
    partner_order: list[str]  # the compared criteria, best first by the profile's local weights
    partner_order_names: list[str]
    variants: list[VariantCheck]


class JudgementAids(DomainModel):
    project_fingerprint: str
    weight_profile_id: str
    available: bool
    reason: str = ""
    anchors: list[Anchor] = []
    break_even: break_even_service.BreakEven | None = None
    references: list[ReferenceComparison] = []
    label: str = (
        "Derived from the project and the profile; nothing here changes the project. A reference "
        "profile is for comparison only: it is never a default and gives no weights."
    )


def _level(raw: object) -> str:
    if isinstance(raw, float) and raw.is_integer():
        return f"{raw:g}"
    return str(raw)


def anchors(project: Project, configuration: AnalysisConfiguration) -> list[Anchor]:
    cfg = configuration.model_copy(update={"method": "mavt", "parameters": {}})
    prep = prepare_mavt(project, cfg)
    names = {n.id: n.name for n in project.criteria}
    alt_names = {a.id: a.name for a in project.alternatives}
    out = []
    for j, leaf in enumerate(prep.leaf_ids):
        if prep.global_[leaf] <= 0.0:
            continue
        column = prep.partial[:, j]
        lo, hi = float(column.min()), float(column.max())
        worst = int(column.argmin())
        best = int(column.argmax())

        def end(i: int, leaf: str = leaf, j: int = j) -> AnchorEnd:
            alt = prep.alt_ids[i]
            return AnchorEnd(
                alternative_id=alt,
                name=alt_names[alt],
                level=_level(prep.raw[(alt, leaf)]),
                value=float(prep.partial[i, j]),
            )

        w, b = end(worst), end(best)
        if lo == hi:
            swing = f"Every alternative is at the same level on {names[leaf]} ({w.level})"
        else:
            swing = f"From {w.name}'s level ({w.level}) to {b.name}'s ({b.level}) on {names[leaf]}"
        out.append(
            Anchor(
                criterion_id=leaf,
                name=names[leaf],
                worst=w,
                best=b,
                tied_worst=int((column == lo).sum()),
                tied_best=int((column == hi).sum()),
                swing=swing,
            )
        )
    return out


def compare_reference(
    project: Project, configuration: AnalysisConfiguration, reference: ReferenceProfile
) -> ReferenceComparison | None:
    """The reference's statements against the profile's local weights; ``None`` when the project
    has none of the criteria the reference names."""
    profile = project.resolve_profile(configuration)
    if profile is None:
        raise McdaError("the project has no weight profile")
    tree = project.tree()
    local = compute_weights(tree, profile.judgements).local
    names = {n.id: n.name for n in project.criteria}
    named = {c for v in reference.variants for s in v.statements for c in (*s.above, *s.below)}
    present = [c for c in tree.all_ids() if c in named]
    if not present:
        return None
    variants = []
    for variant in reference.variants:
        checks = []
        for s in variant.statements:
            pairs = []
            for a in s.above:
                for b in s.below:
                    if a not in tree or b not in tree:
                        verdict: Literal["agrees", "disagrees", "tie", "not comparable"] = (
                            "not comparable"
                        )
                    elif tree.node(a).parent_id != tree.node(b).parent_id:
                        verdict = "not comparable"
                    elif local[a] > local[b]:
                        verdict = "agrees"
                    elif local[a] < local[b]:
                        verdict = "disagrees"
                    else:
                        verdict = "tie"
                    pairs.append(
                        PairCheck(above=a, below=b, above_name=names.get(a, a),
                                  below_name=names.get(b, b), partner=verdict)
                    )  # fmt: skip
            text = (
                " and ".join(names.get(a, a) for a in s.above)
                + " above "
                + " and ".join(names.get(b, b) for b in s.below)
            )
            bad = sum(1 for p in pairs if p.partner == "disagrees")
            checks.append(
                StatementCheck(horizon=s.horizon, text=text, pairs=pairs, disagreements=bad)
            )
        variants.append(
            VariantCheck(
                id=variant.id,
                name=variant.name,
                summary=variant.summary,
                citations=variant.citations,
                statements=checks,
                disagreements=sum(c.disagreements for c in checks),
            )
        )
    position = {c: i for i, c in enumerate(tree.all_ids())}
    order = sorted(present, key=lambda c: (-local[c], position[c]))
    return ReferenceComparison(
        reference_id=reference.id,
        title=reference.title,
        purpose=reference.purpose,
        criteria_note=reference.criteria_note,
        partner_order=order,
        partner_order_names=[names[c] for c in order],
        variants=variants,
    )


def judgement_aids(
    project: Project,
    configuration: AnalysisConfiguration,
    references: list[ReferenceProfile] | None = None,
) -> JudgementAids:
    profile = project.resolve_profile(configuration)
    if profile is None:
        raise McdaError("the project has no weight profile")
    try:
        found = anchors(project, configuration)
        cfg = configuration
        if cfg.method not in break_even_service.MAVT_METHODS:
            cfg = cfg.model_copy(update={"method": "mavt", "parameters": {}})
        breakeven = break_even_service.break_even(project, cfg)
    except (ValidationFailed, McdaError) as exc:
        return JudgementAids(
            project_fingerprint=project.fingerprint(),
            weight_profile_id=profile.id,
            available=False,
            reason=f"the model cannot be evaluated yet: {exc}",
        )
    compared = [
        c
        for c in (compare_reference(project, configuration, r) for r in references or [])
        if c is not None
    ]
    return JudgementAids(
        project_fingerprint=project.fingerprint(),
        weight_profile_id=profile.id,
        available=True,
        anchors=found,
        break_even=breakeven,
        references=compared,
    )


def dump(reference: ReferenceProfile) -> str:
    return json.dumps(reference.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"
