"""Reviewed ratings for reference cases: packets, a review sheet and an importer.

See ``docs/design/reference-rating.md``. The flow is: packets (no outcome information) -> proposals
made outside the application (``rating-proposals`` v1) -> a review sheet a person fills in ->
``import_reviews``, which writes only the accepted and edited proposals into a new version 2
reference set as ``review: "reviewed"`` ratings, each citing its dated source, and refuses the whole
import on any issue. Nothing here calls a model, and nothing changes the project or the input set.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import html
import io
import json
from collections import Counter
from typing import Any, Literal

from mcda_studio.domain.common import DomainModel, Id, Issue
from mcda_studio.domain.errors import McdaError
from mcda_studio.domain.project import Project
from mcda_studio.domain.rating_proposals import RatingProposal, RatingProposals
from mcda_studio.domain.reference_cases import ReferenceCasesInvalid
from mcda_studio.domain.reference_cases_v2 import ReferenceCasesV2, date_bounds
from mcda_studio.domain.validation import ValidationReport
from mcda_studio.services import reference_outcome_service
from mcda_studio.services.reference_case_service import _issue

Decision = Literal["accept", "edit", "reject", "skip"]
DECISIONS: tuple[Decision, ...] = ("accept", "edit", "reject", "skip")
SHEET_COLUMNS = [
    "proposal_id", "proposal_sha256", "case_id", "case_name", "snapshot_date", "criterion_id",
    "criterion_name", "proposed_label", "not_enough_evidence", "allowed_labels", "claim",
    "source_url", "source_date", "locator", "excerpt", "confidence", "proposer", "decision",
    "final_label", "reviewer_note",
]  # fmt: skip
REVIEWER_NOTE_MAX = 500
PACKET_LABEL = (
    "A packet holds what was known about a case at its snapshot date and nothing about how it "
    "turned out. Rate each criterion only from sources whose latest possible date is on or before "
    "the snapshot date; when the evidence is not there, say so: not enough evidence is a valid "
    "answer and leaves the criterion unrated."
)
_FORMULA = ("=", "+", "-", "@", "\t", "\r")


class RatingRefusedError(McdaError):
    """The proposals or the review sheet cannot be used; every issue is listed."""

    def __init__(self, issues: list[Issue]) -> None:
        self.issues = issues
        super().__init__("; ".join(f"{i.path}: {i.message}" for i in issues))


def proposal_digest(proposal: RatingProposal) -> str:
    """SHA-256 of the proposal's canonical JSON (sorted keys, UTF-8, no spaces)."""
    text = json.dumps(
        proposal.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _labelled_leaves(project: Project) -> dict[str, tuple[str, list[str]]]:
    tree = project.tree()
    out: dict[str, tuple[str, list[str]]] = {}
    for leaf in tree.leaves():
        node = tree.node(leaf)
        if node.scale is not None and node.scale.is_labelled:
            out[leaf] = (node.name, list(node.scale.labels or []))
    return out


def supports_rating(source_date: str | None, snapshot: dt.date) -> bool:
    """The version 2 snapshot guard: the latest day the date can mean is on or before the
    snapshot."""
    return source_date is not None and date_bounds(source_date)[1] <= snapshot


def check_proposals(
    project: Project, reference_set: ReferenceCasesV2, proposals: RatingProposals
) -> list[Issue]:
    """Every issue that stops the proposals being reviewed or imported."""
    issues: list[Issue] = []
    if proposals.criteria_from != project.id:
        issues.append(
            _issue(
                "RATING_PROJECT_MISMATCH",
                f"the proposals rate project {proposals.criteria_from!r}, not {project.id!r}",
                "criteria_from",
            )
        )
    if proposals.reference_set_id != reference_set.id:
        issues.append(
            _issue(
                "RATING_SET_MISMATCH",
                f"the proposals are for reference set {proposals.reference_set_id!r}, not "
                f"{reference_set.id!r}",
                "reference_set_id",
            )
        )
    cases = {case.id: case for case in reference_set.cases}
    leaves = _labelled_leaves(project)
    for p, proposal in enumerate(proposals.proposals):
        at = f"proposals[{p}]"
        case = cases.get(proposal.case_id)
        if case is None:
            issues.append(
                _issue("RATING_UNKNOWN_CASE", f"no case {proposal.case_id!r}", f"{at}.case_id")
            )
        if proposal.criterion_id not in leaves:
            issues.append(
                _issue(
                    "RATING_UNKNOWN_CRITERION",
                    f"{proposal.criterion_id!r} is not a labelled leaf criterion of the project",
                    f"{at}.criterion_id",
                )
            )
        elif proposal.label is not None and proposal.label not in leaves[proposal.criterion_id][1]:
            issues.append(
                _issue(
                    "RATING_UNKNOWN_LABEL",
                    f"{proposal.label!r} is not one of the labels of {proposal.criterion_id!r}",
                    f"{at}.label",
                )
            )
        if (
            case is not None
            and proposal.label is not None
            and not supports_rating(proposal.source_date, case.snapshot_date)
        ):
            issues.append(
                _issue(
                    "RATING_SOURCE_AFTER_SNAPSHOT",
                    f"the source dated {proposal.source_date} could be after the snapshot "
                    f"{case.snapshot_date.isoformat()}, so it cannot support a rating",
                    f"{at}.source_date",
                )
            )
    return issues


def _refuse_if(issues: list[Issue]) -> None:
    if issues:
        raise RatingRefusedError(issues)


# ---- packets --------------------------------------------------------------------------------


class PacketLeaf(DomainModel):
    criterion_id: Id
    name: str
    labels: list[str]
    rubric: str = ""


class PacketContext(DomainModel):
    label: str
    value: str


class Packet(DomainModel):
    case_id: Id
    name: str
    snapshot_date: dt.date
    context: list[PacketContext]
    leaves: list[PacketLeaf]
    instructions: str = PACKET_LABEL


def packets(
    project: Project,
    reference_set: ReferenceCasesV2,
    case_ids: list[str] | None = None,
    leaf_ids: list[str] | None = None,
    context_labels: list[str] | None = None,
    rubric_text: dict[str, str] | None = None,
) -> list[Packet]:
    """One packet per case: no outcome definition, observation, outcome claim or follow-up date.

    Context rows are free text and may describe outcomes, so a row enters only when its label is
    in ``context_labels`` (none by default)."""
    leaves = _labelled_leaves(project)
    chosen = leaf_ids if leaf_ids is not None else list(leaves)
    unknown = [leaf for leaf in chosen if leaf not in leaves]
    if unknown:
        raise McdaError(f"not labelled leaf criteria of the project: {', '.join(unknown)}")
    by_id = {case.id: case for case in reference_set.cases}
    wanted = case_ids if case_ids is not None else list(by_id)
    missing = [c for c in wanted if c not in by_id]
    if missing:
        raise McdaError(f"no such cases in the reference set: {', '.join(missing)}")
    allowed = set(context_labels or [])
    rubric_text = rubric_text or {}
    return [
        Packet(
            case_id=case.id,
            name=case.name,
            snapshot_date=case.snapshot_date,
            context=[
                PacketContext(label=row.label, value=row.value)
                for row in case.context
                if row.label in allowed
            ],
            leaves=[
                PacketLeaf(
                    criterion_id=leaf,
                    name=leaves[leaf][0],
                    labels=leaves[leaf][1],
                    rubric=rubric_text.get(leaf, ""),
                )
                for leaf in chosen
            ],
        )
        for case in (by_id[c] for c in wanted)
    ]


# ---- the review sheet ---------------------------------------------------------------------


def _safe(value: str) -> str:
    """A cell Excel shows as text and never evaluates."""
    return "'" + value if value.startswith(_FORMULA) else value


def _unsafe(value: str) -> str:
    return value[1:] if value.startswith("'") and value[1:].startswith(_FORMULA) else value


def review_rows(
    project: Project, reference_set: ReferenceCasesV2, proposals: RatingProposals
) -> list[dict[str, str]]:
    _refuse_if(check_proposals(project, reference_set, proposals))
    leaves = _labelled_leaves(project)
    cases = {case.id: case for case in reference_set.cases}
    proposer = proposals.proposer.proposed_by
    rows = []
    for proposal in proposals.proposals:
        case = cases[proposal.case_id]
        name, labels = leaves[proposal.criterion_id]
        rows.append(
            {
                "proposal_id": proposal.id,
                "proposal_sha256": proposal_digest(proposal),
                "case_id": case.id,
                "case_name": case.name,
                "snapshot_date": case.snapshot_date.isoformat(),
                "criterion_id": proposal.criterion_id,
                "criterion_name": name,
                "proposed_label": proposal.label or "",
                "not_enough_evidence": "yes" if proposal.not_enough_evidence else "",
                "allowed_labels": " | ".join(labels),
                "claim": proposal.claim,
                "source_url": proposal.source_url or "",
                "source_date": proposal.source_date or "",
                "locator": proposal.locator,
                "excerpt": proposal.excerpt,
                "confidence": proposal.confidence,
                "proposer": proposer,
                "decision": "skip",
                "final_label": "",
                "reviewer_note": "",
            }
        )
    return rows


def review_csv(rows: list[dict[str, str]]) -> str:
    """The sheet as CSV text, every cell spreadsheet-safe. Write it with ``utf-8-sig``."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=SHEET_COLUMNS, lineterminator="\r\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _safe(v) for k, v in row.items()})
    return buffer.getvalue()


def review_html(
    project: Project,
    reference_set: ReferenceCasesV2,
    proposals: RatingProposals,
    rubric_text: dict[str, str] | None = None,
) -> str:
    """A page to keep open beside the sheet: each criterion's labels and rubric, then the
    proposals by case with their sources."""
    rows = review_rows(project, reference_set, proposals)
    leaves = _labelled_leaves(project)
    rubric_text = rubric_text or {}
    used = list(dict.fromkeys(row["criterion_id"] for row in rows))
    e = html.escape
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>Rating review</title>",
        "<style>body{font:15px/1.5 system-ui,sans-serif;max-width:60rem;margin:2rem auto;"
        "padding:0 1rem;color:#1b2430}h2{margin-top:2.5rem}table{border-collapse:collapse;"
        "width:100%}td,th{border:1px solid #c8d2de;padding:.35rem .5rem;vertical-align:top;"
        "text-align:left}.rubric{color:#3d4a5a}.q{font-style:italic}.nee{color:#8a4b00}</style>",
        "</head><body>",
        f"<h1>Rating review: {e(reference_set.title)}</h1>",
        f"<p>{e(str(len(rows)))} proposals by {e(proposals.proposer.proposed_by)}. Fill in "
        "<code>decision</code> (accept, edit, reject or skip), <code>final_label</code> (for "
        "edit) and <code>reviewer_note</code> in the sheet, and save it as CSV UTF-8. Open each "
        "source and check that its excerpt is there and that it predates the snapshot.</p>",
        "<h2>Criteria</h2>",
    ]
    for leaf in used:
        name, labels = leaves[leaf]
        parts.append(f"<h3>{e(name)} <code>{e(leaf)}</code></h3>")
        if rubric_text.get(leaf):
            parts.append(f'<p class="rubric">{e(rubric_text[leaf])}</p>')
        parts.append("<ol>" + "".join(f"<li>{e(label)}</li>" for label in labels) + "</ol>")
    parts.append("<h2>Proposals by case</h2>")
    for case_id in dict.fromkeys(row["case_id"] for row in rows):
        mine = [row for row in rows if row["case_id"] == case_id]
        parts.append(
            f"<h3>{e(mine[0]['case_name'])} <code>{e(case_id)}</code>, snapshot "
            f"{e(mine[0]['snapshot_date'])}</h3>"
        )
        parts.append(
            "<table><tr><th>Proposal</th><th>Criterion</th><th>Proposed</th><th>Claim and "
            "excerpt</th><th>Source</th><th>Confidence</th></tr>"
        )
        for row in mine:
            label = (
                '<span class="nee">not enough evidence</span>'
                if row["not_enough_evidence"]
                else e(row["proposed_label"])
            )
            source = (
                f'<a href="{e(row["source_url"], quote=True)}" rel="noopener noreferrer" '
                f'target="_blank">{e(row["source_url"])}</a><br>dated {e(row["source_date"])}; '
                f"{e(row['locator'])}"
                if row["source_url"]
                else "none"
            )
            excerpt = f'<br><span class="q">"{e(row["excerpt"])}"</span>' if row["excerpt"] else ""
            parts.append(
                f"<tr><td><code>{e(row['proposal_id'])}</code></td><td>{e(row['criterion_name'])}"
                f"</td><td>{label}</td><td>{e(row['claim'])}{excerpt}</td><td>{source}</td>"
                f"<td>{e(row['confidence'])}</td></tr>"
            )
        parts.append("</table>")
    parts.append("</body></html>")
    return "\n".join(parts) + "\n"


# ---- the importer ---------------------------------------------------------------------------


class ImportReport(DomainModel):
    reference_set_id: Id
    reviewer: str
    reviewed_on: dt.date
    rows: int
    decisions: dict[str, int]
    imported: list[str]  # proposal ids that became reviewed ratings (new or replaced)
    unchanged: list[str]  # already present, identical
    replaced: list[str]  # an existing reviewed rating replaced under --replace
    not_enough_evidence: list[str]  # accepted "not enough evidence": the leaf stays unrated
    reviewed_rating_count: int


def parse_sheet(text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text))
    header = reader.fieldnames or []
    needed = {"proposal_id", "proposal_sha256", "decision", "final_label", "reviewer_note"}
    absent = sorted(needed - set(header))
    if absent:
        raise RatingRefusedError(
            [_issue("RATING_SHEET_COLUMNS", f"missing columns: {', '.join(absent)}", "header")]
        )
    return [{k: (v or "") for k, v in row.items() if k is not None} for row in reader]


def decode_sheet(raw: bytes, encoding: str = "utf-8-sig") -> str:
    try:
        return raw.decode(encoding)
    except UnicodeDecodeError as exc:
        raise McdaError(
            f"the review sheet is not valid {encoding} ({exc.reason}); save it from Excel as "
            "'CSV UTF-8', or name its encoding with --sheet-encoding"
        ) from None


def source_key(url: str, date: str) -> str:
    return "rp_" + hashlib.sha256(f"{url}|{date}".encode()).hexdigest()[:12]


def import_reviews(
    project: Project,
    reference_set: ReferenceCasesV2,
    proposals: RatingProposals,
    sheet_text: str,
    *,
    reviewer: str,
    reviewed_on: dt.date,
    replace: bool = False,
) -> tuple[ReferenceCasesV2, ImportReport]:
    """A new reference set with the accepted and edited proposals as reviewed ratings.

    Refuses everything (``RatingRefusedError``) on any issue: nothing partial is ever returned."""
    if not reviewer.strip():
        raise McdaError("name the reviewer")
    issues = check_proposals(project, reference_set, proposals)
    leaves = _labelled_leaves(project)
    by_id = {p.id: p for p in proposals.proposals}
    rows = parse_sheet(sheet_text)
    decisions: Counter[str] = Counter()
    chosen: list[tuple[RatingProposal, str, str]] = []  # proposal, label, reviewer note
    nee: list[str] = []
    seen_rows: set[str] = set()
    seen_leaf: dict[tuple[str, str], str] = {}
    for r, row in enumerate(rows, start=2):  # physical line 1 is the header
        at = f"line {r}"
        pid = row["proposal_id"].strip()
        decision = _unsafe(row["decision"]).strip().lower()
        if decision not in DECISIONS:
            issues.append(
                _issue(
                    "RATING_UNKNOWN_DECISION",
                    f"decision {row['decision']!r} must be accept, edit, reject or skip",
                    at,
                )
            )
            continue
        decisions[decision] += 1
        proposal = by_id.get(pid)
        if proposal is None:
            issues.append(_issue("RATING_UNKNOWN_PROPOSAL", f"no proposal {pid!r}", at))
            continue
        if pid in seen_rows:
            issues.append(_issue("RATING_DUPLICATE_ROW", f"proposal {pid!r} is listed twice", at))
            continue
        seen_rows.add(pid)
        if row["proposal_sha256"].strip() != proposal_digest(proposal):
            issues.append(
                _issue(
                    "RATING_DIGEST_MISMATCH",
                    f"proposal {pid!r} changed since the sheet was made; make a new sheet",
                    at,
                )
            )
            continue
        if decision in ("reject", "skip"):
            continue
        note = _unsafe(row["reviewer_note"]).strip()
        if len(note) > REVIEWER_NOTE_MAX:
            issues.append(
                _issue(
                    "RATING_NOTE_TOO_LONG",
                    f"keep the note under {REVIEWER_NOTE_MAX} characters",
                    at,
                )
            )
            continue
        if decision == "accept":
            if proposal.not_enough_evidence:
                nee.append(pid)
                continue
            label = proposal.label or ""
        else:
            label = _unsafe(row["final_label"]).strip()
            allowed = leaves.get(proposal.criterion_id, ("", []))[1]
            if label not in allowed:
                issues.append(
                    _issue(
                        "RATING_EDIT_UNKNOWN_LABEL",
                        f"final_label {label!r} is not one of the labels of "
                        f"{proposal.criterion_id!r}",
                        at,
                    )
                )
                continue
        if not (proposal.source_url and proposal.source_date and proposal.excerpt.strip()):
            issues.append(
                _issue(
                    "RATING_NO_SOURCE",
                    f"proposal {pid!r} cites no dated source and excerpt, so it cannot become a "
                    "rating; propose again with a source",
                    at,
                )
            )
            continue
        case = next((c for c in reference_set.cases if c.id == proposal.case_id), None)
        if case is not None and not supports_rating(proposal.source_date, case.snapshot_date):
            issues.append(
                _issue(
                    "RATING_SOURCE_AFTER_SNAPSHOT",
                    f"proposal {pid!r}: the source dated {proposal.source_date} could be after the "
                    f"snapshot {case.snapshot_date.isoformat()}",
                    at,
                )
            )
            continue
        key = (proposal.case_id, proposal.criterion_id)
        if key in seen_leaf:
            issues.append(
                _issue(
                    "RATING_TWO_FOR_ONE_LEAF",
                    f"proposals {seen_leaf[key]!r} and {pid!r} both rate {key[1]!r} for case "
                    f"{key[0]!r}; accept one",
                    at,
                )
            )
            continue
        seen_leaf[key] = pid
        chosen.append((proposal, label, note))
    _refuse_if(issues)

    raw = reference_set.model_dump(mode="json")
    cases = {case["id"]: case for case in raw["cases"]}
    imported: list[str] = []
    unchanged: list[str] = []
    replaced: list[str] = []
    origin = proposals.proposer.model_dump(mode="json")
    for proposal, label, note in chosen:
        assert proposal.source_url is not None and proposal.source_date is not None
        case = cases[proposal.case_id]
        skey = source_key(proposal.source_url, proposal.source_date)
        raw["sources"].setdefault(
            skey,
            {
                "name": proposal.source_title or proposal.source_url,
                "reference": proposal.source_url,
                "location": "",
                "date": proposal.source_date,
                "description": (
                    f"Public source cited in rating proposals by {proposals.proposer.proposed_by}."
                ),
                "classification": "unknown",
            },
        )
        claim_id = f"rp_{proposal.id}"
        claim = {
            "id": claim_id,
            "text": f'{proposal.claim} Excerpt ({proposal.locator}): "{proposal.excerpt}"',
            "source": skey,
            "origin": origin,
        }
        rating = {
            "criterion_id": proposal.criterion_id,
            "label": label,
            "claim_ids": [claim_id],
            "review": "reviewed",
            "note": (
                f"Reviewed by {reviewer.strip()} on {reviewed_on.isoformat()}."
                + (f" {note}" if note else "")
                + f" (proposal {proposal.id}, {proposal.confidence} confidence)"
            ),
        }
        current = next(
            (x for x in case["ratings"] if x["criterion_id"] == proposal.criterion_id), None
        )
        existing_claim = next((c for c in case["claims"] if c["id"] == claim_id), None)
        if current == rating and existing_claim == claim:
            unchanged.append(proposal.id)
            continue
        if current is not None and current["review"] == "reviewed":
            if not replace:
                issues.append(
                    _issue(
                        "RATING_REVIEWED_EXISTS",
                        f"case {proposal.case_id!r} already has a reviewed rating for "
                        f"{proposal.criterion_id!r}; pass --replace to change it",
                        f"proposal {proposal.id}",
                    )
                )
                continue
            replaced.append(proposal.id)
        if existing_claim is not None and existing_claim != claim and not replace:
            issues.append(
                _issue(
                    "RATING_CLAIM_EXISTS",
                    f"claim {claim_id!r} already exists with other content; pass --replace",
                    f"proposal {proposal.id}",
                )
            )
            continue
        case["claims"] = [c for c in case["claims"] if c["id"] != claim_id] + [claim]
        case["ratings"] = [
            x for x in case["ratings"] if x["criterion_id"] != proposal.criterion_id
        ] + [rating]
        imported.append(proposal.id)
    _refuse_if(issues)
    updated = ReferenceCasesV2.model_validate_json(json.dumps(raw))
    report = reference_outcome_service.validate(project, updated)
    if not report.valid:
        raise ReferenceCasesInvalid(ValidationReport(issues=report.issues))
    return updated, ImportReport(
        reference_set_id=updated.id,
        reviewer=reviewer.strip(),
        reviewed_on=reviewed_on,
        rows=len(rows),
        decisions={d: decisions.get(d, 0) for d in DECISIONS},
        imported=imported,
        unchanged=unchanged,
        replaced=replaced,
        not_enough_evidence=nee,
        reviewed_rating_count=sum(r.review == "reviewed" for c in updated.cases for r in c.ratings),
    )


def dump_reference_set(reference_set: ReferenceCasesV2) -> str:
    """The set as written by the importer: indented JSON, UTF-8, one trailing newline."""
    return json.dumps(reference_set.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"


def dump_packets(items: list[Packet]) -> list[dict[str, Any]]:
    return [p.model_dump(mode="json") for p in items]
