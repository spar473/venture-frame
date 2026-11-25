"""Reference rating (deep dive 6): proposals, the review sheet and the importer, on synthetic data.

Every decision type; edits to unknown labels, sources dated after the snapshot and proposals
without a source or excerpt are refused; re-import is idempotent; an existing reviewed rating is
protected without ``--replace``; the byte-order-mark CSV round-trips under a cp949 or ASCII locale
and under UTF-8; packets carry no outcome information; the project and the input set never change.
"""

import copy
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import reference_case_builders as b
from mcda_studio.domain.rating_proposals import RatingProposals
from mcda_studio.domain.reference_cases_v2 import ReferenceCasesV2
from mcda_studio.services import reference_outcome_service
from mcda_studio.services import reference_rating_service as rr

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = b.SNAPSHOT  # 2021-10-27
FOLLOW_UP = "2026-11-20"
DAY = dt.date(2026, 11, 26)
OUTCOME_WORDS = "acquired for a large sum in 2023"


def v2_set(**ratings: dict[str, Any]) -> dict[str, Any]:
    """Two cases; ``c1`` carries an outcome, a context row describing it, and any ratings given."""
    cases = []
    for case_id in ("c1", "c2"):
        raw = b.case(case_id, {})
        del raw["outcome"]
        raw["context"] = [
            {"label": "Entry round", "value": "first round"},
            {"label": "Outcome status", "value": OUTCOME_WORDS},
        ]
        raw["ratings"] = [
            {"criterion_id": leaf, "label": spec["label"], "claim_ids": ["c1"],
             "review": spec["review"], "note": ""}
            for leaf, spec in (ratings.get(case_id) or {}).items()
        ]  # fmt: skip
        raw["outcomes"] = [
            {"definition": "next_36m", "status": "event", "event_date": "2023-05-02",
             "claim_ids": ["c2"]}
        ]  # fmt: skip
        cases.append(raw)
    raw_set = b.reference_set(*cases)
    raw_set.update(
        version=2,
        observed_through=FOLLOW_UP,
        outcome_definitions={
            "next_36m": {"label": "Next event", "window_months": 36, "description": "Synthetic."}
        },
    )
    return raw_set


def load_set(raw: dict[str, Any]) -> ReferenceCasesV2:
    return ReferenceCasesV2.model_validate_json(json.dumps(raw))


def proposal(pid: str, case: str = "c1", leaf: str = "o4", label: str | None = "C", **kw: Any):
    out: dict[str, Any] = {
        "id": pid,
        "case_id": case,
        "criterion_id": leaf,
        "claim": kw.pop("claim", f"Claim for {pid} – Māori ≥ 3."),
        "confidence": "medium",
    }
    if label is None:
        out["not_enough_evidence"] = True
    else:
        out.update(
            label=label,
            source_url="https://example.org/page",
            source_title="An example page",
            source_date="2021-09",
            locator="second paragraph",
            excerpt="The team shipped its first prototype in September.",
        )
    out.update(kw)
    return out


def proposals(*items: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "rating-proposals",
        "version": 1,
        "reference_set_id": "fixture_set",
        "criteria_from": "fixture_project",
        "proposer": {"kind": "tool", "proposed_by": "Test proposer", "tool_id": "test"},
        "created_on": "2026-11-26",
        "proposals": list(items),
    }


def load_proposals(raw: dict[str, Any]) -> RatingProposals:
    return RatingProposals.model_validate_json(json.dumps(raw))


def sheet(rows: list[dict[str, str]], decisions: dict[str, tuple[str, str, str]]) -> str:
    """The generated sheet with some rows decided: id -> (decision, final_label, note)."""
    filled = []
    for row in rows:
        row = dict(row)
        if row["proposal_id"] in decisions:
            row["decision"], row["final_label"], row["reviewer_note"] = decisions[
                row["proposal_id"]
            ]
        filled.append(row)
    return rr.review_csv(filled)


def run_import(raw_set, raw_props, decisions, **kw):
    project = b.project()
    reference_set = load_set(raw_set)
    props = load_proposals(raw_props)
    rows = rr.review_rows(project, reference_set, props)
    return rr.import_reviews(
        project, reference_set, props, sheet(rows, decisions),
        reviewer="Test Reviewer", reviewed_on=DAY, **kw,
    )  # fmt: skip


def ratings_of(reference_set: ReferenceCasesV2, case_id: str) -> dict[str, Any]:
    case = next(c for c in reference_set.cases if c.id == case_id)
    return {r.criterion_id: r for r in case.ratings}


# ---- decisions ------------------------------------------------------------------------------


def test_each_decision_type() -> None:
    raw_props = proposals(
        proposal("acc", leaf="o4", label="C"),
        proposal("edi", leaf="o5", label="v2"),
        proposal("rej", leaf="o4b", label="B"),
        proposal("ski", leaf="o5b", label="v4"),
        proposal("nee", leaf="cat", label=None),
    )
    updated, report = run_import(
        v2_set(),
        raw_props,
        {
            "acc": ("accept", "", "Checked the page."),
            "edi": ("edit", "v3", "Second paragraph supports v3."),
            "rej": ("reject", "", ""),
            "nee": ("accept", "", ""),
        },
    )
    got = ratings_of(updated, "c1")
    assert set(got) == {"o4", "o5"}  # reject, skip and not-enough-evidence leave leaves unrated
    assert got["o4"].label == "C" and got["o4"].review == "reviewed"
    assert got["o5"].label == "v3" and got["o5"].review == "reviewed"
    assert got["o4"].note == (
        "Reviewed by Test Reviewer on 2026-11-26. Checked the page. (proposal acc, medium "
        "confidence)"
    )
    case = next(c for c in updated.cases if c.id == "c1")
    claim = next(c for c in case.claims if c.id == "rp_acc")
    assert claim.origin.proposed_by == "Test proposer" and "first prototype" in claim.text
    source = updated.sources[claim.source]
    assert source.reference == "https://example.org/page" and source.date == "2021-09"
    assert report.decisions == {"accept": 2, "edit": 1, "reject": 1, "skip": 1}
    assert report.imported == ["acc", "edi"] and report.not_enough_evidence == ["nee"]
    assert reference_outcome_service.validate(b.project(), updated).valid


def test_an_edit_to_an_unknown_label_is_refused() -> None:
    with pytest.raises(
        rr.RatingRefusedError, match="RATING_EDIT_UNKNOWN_LABEL|not one of the labels"
    ):
        run_import(v2_set(), proposals(proposal("edi")), {"edi": ("edit", "Z", "")})
    with pytest.raises(rr.RatingRefusedError, match="not one of the labels"):
        run_import(v2_set(), proposals(proposal("edi")), {"edi": ("edit", "", "")})


def test_a_source_dated_after_the_snapshot_is_refused() -> None:
    late = proposal("late", source_date="2021-10")  # October 2021 may end after 27 October
    project, reference_set = b.project(), load_set(v2_set())
    issues = rr.check_proposals(project, reference_set, load_proposals(proposals(late)))
    assert [i.code for i in issues] == ["RATING_SOURCE_AFTER_SNAPSHOT"]
    with pytest.raises(
        rr.RatingRefusedError, match="RATING_SOURCE_AFTER_SNAPSHOT|after the snapshot"
    ):
        rr.review_rows(project, reference_set, load_proposals(proposals(late)))
    ok = rr.check_proposals(
        project, reference_set, load_proposals(proposals(proposal("ok", source_date="2021-10-27")))
    )
    assert ok == []  # the snapshot day itself is allowed


def test_the_importer_rechecks_the_snapshot_guard_itself(monkeypatch: Any) -> None:
    raw = proposals(proposal("late", source_date="2022"))
    monkeypatch.setattr(rr, "check_proposals", lambda *a: [])  # as if the sheet came from elsewhere
    project, reference_set = b.project(), load_set(v2_set())
    props = load_proposals(raw)
    rows = rr.review_rows(project, reference_set, props)
    with pytest.raises(rr.RatingRefusedError, match="after the snapshot"):
        rr.import_reviews(project, reference_set, props, sheet(rows, {"late": ("accept", "", "")}),
                          reviewer="R", reviewed_on=DAY)  # fmt: skip


def test_a_proposal_without_a_source_or_excerpt_is_refused() -> None:
    for drop in ("source_url", "source_date", "locator", "excerpt"):
        raw = proposal("p")
        raw.pop(drop)
        with pytest.raises(ValidationError, match="needs a dated source"):
            load_proposals(proposals(raw))
    with pytest.raises(ValidationError, match="more than 25 words"):
        load_proposals(proposals(proposal("p", excerpt=" ".join(["word"] * 26))))
    # Editing a not-enough-evidence row that cites no source cannot create a rating.
    with pytest.raises(rr.RatingRefusedError, match="RATING_NO_SOURCE|cites no dated source"):
        run_import(v2_set(), proposals(proposal("nee", label=None)), {"nee": ("edit", "B", "")})


def test_reimport_is_idempotent() -> None:
    raw_props = proposals(proposal("acc"), proposal("edi", leaf="o5", label="v2"))
    decisions = {"acc": ("accept", "", "ok"), "edi": ("edit", "v3", "")}
    first, _ = run_import(v2_set(), raw_props, decisions)
    second, report = run_import(json.loads(rr.dump_reference_set(first)), raw_props, decisions)
    assert rr.dump_reference_set(second) == rr.dump_reference_set(first)
    assert report.imported == [] and report.unchanged == ["acc", "edi"]


def test_an_existing_reviewed_rating_is_protected_without_replace() -> None:
    raw_set = v2_set(c1={"o4": {"label": "A", "review": "reviewed"}})
    with pytest.raises(rr.RatingRefusedError, match="RATING_REVIEWED_EXISTS|pass --replace"):
        run_import(raw_set, proposals(proposal("acc")), {"acc": ("accept", "", "")})
    updated, report = run_import(
        raw_set, proposals(proposal("acc")), {"acc": ("accept", "", "")}, replace=True
    )
    assert ratings_of(updated, "c1")["o4"].label == "C" and report.replaced == ["acc"]


def test_an_existing_proposed_rating_is_replaced_by_the_reviewed_one() -> None:
    raw_set = v2_set(c1={"o4": {"label": "A", "review": "proposed"}})
    updated, report = run_import(raw_set, proposals(proposal("acc")), {"acc": ("accept", "", "")})
    assert ratings_of(updated, "c1")["o4"].review == "reviewed" and report.replaced == []


def test_sheet_integrity_is_checked() -> None:
    project, reference_set = b.project(), load_set(v2_set())
    props = load_proposals(proposals(proposal("a"), proposal("b", label="B")))
    rows = rr.review_rows(project, reference_set, props)

    def refused(text: str, pattern: str) -> None:
        with pytest.raises(rr.RatingRefusedError, match=pattern):
            rr.import_reviews(project, reference_set, props, text, reviewer="R", reviewed_on=DAY)

    refused(sheet(rows, {"a": ("maybe", "", "")}), "must be accept, edit, reject or skip")
    tampered = [dict(rows[0], proposal_sha256="0" * 64), rows[1]]
    refused(sheet(tampered, {"a": ("accept", "", "")}), "changed since the sheet was made")
    refused(rr.review_csv([rows[0], rows[0]]), "listed twice")
    refused(sheet(rows, {"a": ("accept", "", ""), "b": ("accept", "", "")}), "accept one")
    ghost = [dict(rows[0], proposal_id="ghost")]
    refused(rr.review_csv(ghost), "no proposal 'ghost'")
    with pytest.raises(rr.RatingRefusedError, match="missing columns"):
        rr.parse_sheet("proposal_id,decision\r\na,accept\r\n")


def test_the_sheet_starts_every_row_as_skip_and_is_spreadsheet_safe() -> None:
    project, reference_set = b.project(), load_set(v2_set())
    props = load_proposals(proposals(proposal("a", claim="=HYPERLINK(1)")))
    rows = rr.review_rows(project, reference_set, props)
    assert [r["decision"] for r in rows] == ["skip"]
    text = rr.review_csv(rows)
    assert text.splitlines()[0] == ",".join(rr.SHEET_COLUMNS)
    assert "'=HYPERLINK(1)" in text
    # Unchanged, the sheet imports nothing.
    updated, report = rr.import_reviews(project, reference_set, props, text, reviewer="R",
                                        reviewed_on=DAY)  # fmt: skip
    assert report.imported == [] and rr.dump_reference_set(updated) == rr.dump_reference_set(
        reference_set
    )


def test_the_page_lists_labels_rubrics_and_sources() -> None:
    props = load_proposals(proposals(proposal("a")))
    page = rr.review_html(b.project(), load_set(v2_set()), props, {"o4": "Rubric for O4."})
    assert "Rubric for O4." in page and "https://example.org/page" in page
    assert page.count("<li>") == 4  # O4's four labels


# ---- packets ----------------------------------------------------------------------------------

OUTCOME_KEYS = {
    "outcomes", "outcome", "outcome_definitions", "observed_through", "event_date", "status",
    "definition", "claims", "sources", "ratings",
}  # fmt: skip


def keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(keys(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(keys(v) for v in value)) if value else set()
    return set()


def test_packets_carry_no_outcome_information() -> None:
    items = rr.dump_packets(
        rr.packets(
            b.project(),
            load_set(v2_set()),
            ["c1"],
            ["o4", "o5"],
            ["Entry round"],
            {"o4": "Rubric."},
        )  # fmt: skip
    )
    assert not keys(items) & OUTCOME_KEYS
    text = json.dumps(items)
    assert OUTCOME_WORDS not in text and "2023-05-02" not in text and FOLLOW_UP not in text
    (packet,) = items
    assert set(packet) == {"case_id", "name", "snapshot_date", "context", "leaves", "instructions"}
    assert packet["context"] == [{"label": "Entry round", "value": "first round"}]
    assert [leaf["criterion_id"] for leaf in packet["leaves"]] == ["o4", "o5"]
    assert packet["leaves"][0]["labels"] == ["A", "B", "C", "D"]
    # Without a whitelist, no context row enters at all.
    bare = rr.dump_packets(rr.packets(b.project(), load_set(v2_set()), ["c1"]))
    assert bare[0]["context"] == []


def test_packets_refuse_unknown_cases_and_leaves() -> None:
    with pytest.raises(Exception, match="no such cases"):
        rr.packets(b.project(), load_set(v2_set()), ["nope"])
    with pytest.raises(Exception, match="not labelled leaf"):
        rr.packets(b.project(), load_set(v2_set()), None, ["num"])


# ---- read-only and the sidecar ---------------------------------------------------------------


def test_the_project_and_the_input_set_never_change() -> None:
    project = b.project()
    before, fingerprint = project.model_dump_json(), project.fingerprint()
    reference_set = load_set(v2_set())
    set_before = reference_set.model_dump_json()
    props = load_proposals(proposals(proposal("a")))
    rows = rr.review_rows(project, reference_set, props)
    rr.import_reviews(project, reference_set, props, sheet(rows, {"a": ("accept", "", "")}),
                      reviewer="R", reviewed_on=DAY)  # fmt: skip
    rr.packets(project, reference_set)
    assert project.model_dump_json() == before and project.fingerprint() == fingerprint
    assert reference_set.model_dump_json() == set_before


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r.update(version="1"), "explicit integers"),
        (lambda r: r.update(version=2), "version 2 is not supported"),
        (lambda r: r.update(kind="reference-cases"), "not a rating-proposals file"),
        (lambda r: r.update(extra=True), "Extra inputs"),
        (lambda r: r["proposals"][0].update(extra=True), "Extra inputs"),
        (lambda r: r["proposals"].append(copy.deepcopy(r["proposals"][0])), "is repeated"),
        (lambda r: r["proposals"][0].update(not_enough_evidence=True), "exactly one of"),
        (lambda r: r["proposals"][0].update(source_url="ftp://x"), "http"),
        (lambda r: r["proposals"][0].update(source_date="2021-13"), "no month 13"),
        (lambda r: r["proposals"][0].update(confidence="certain"), "high"),
    ],
)
def test_the_proposals_sidecar_is_strict(change: Any, message: str) -> None:
    raw = proposals(proposal("a"))
    change(raw)
    with pytest.raises(ValidationError, match=message):
        load_proposals(raw)


def test_proposals_must_match_the_project_set_case_and_leaf() -> None:
    raw = proposals(proposal("a", case="zz"), proposal("b", leaf="num"), proposal("c", label="Q"))
    raw["criteria_from"], raw["reference_set_id"] = "other", "other_set"
    codes = {
        i.code for i in rr.check_proposals(b.project(), load_set(v2_set()), load_proposals(raw))
    }
    assert codes == {
        "RATING_PROJECT_MISMATCH", "RATING_SET_MISMATCH", "RATING_UNKNOWN_CASE",
        "RATING_UNKNOWN_CRITERION", "RATING_UNKNOWN_LABEL",
    }  # fmt: skip


# ---- the commands, under narrow locales -------------------------------------------------------


def _files(tmp: Path, decisions: dict[str, tuple[str, str, str]] | None = None) -> dict[str, Path]:
    from mcda_studio.storage.json_repository import JsonProjectRepository

    paths = {"project": tmp / "project.json", "set": tmp / "set.json", "props": tmp / "props.json"}
    JsonProjectRepository().save(b.project(), paths["project"])
    paths["set"].write_text(json.dumps(v2_set(), ensure_ascii=False), encoding="utf-8")
    paths["props"].write_text(
        json.dumps(proposals(proposal("acc"), proposal("rej", leaf="o5", label="v2")),
                   ensure_ascii=False), encoding="utf-8",
    )  # fmt: skip
    return paths


def _cli(args: list[str], env_extra: dict[str, str]) -> subprocess.CompletedProcess[bytes]:
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONIOENCODING", "PYTHONUTF8"}}
    env.update(PYTHONPATH=str(ROOT / "src"), **env_extra)
    return subprocess.run([sys.executable, "-m", "mcda_studio", *args], capture_output=True,
                          env=env, cwd=ROOT)  # fmt: skip


NARROW = {"PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0", "LC_ALL": "C", "LANG": "C"}


@pytest.mark.parametrize("mode", ["narrow", "utf8"])
def test_the_bom_sheet_round_trips_under_any_locale(tmp_path: Path, mode: str) -> None:
    env = NARROW if mode == "narrow" else {"PYTHONUTF8": "1"}
    paths = _files(tmp_path)
    out_dir = tmp_path / "review"
    made = _cli(["reference-cases", "review-sheet", str(paths["props"]), "--reference-cases",
                 str(paths["set"]), "--project", str(paths["project"]), "--out-dir", str(out_dir)],
                env)  # fmt: skip
    assert made.returncode == 0, made.stderr.decode("utf-8", "replace")
    raw = (out_dir / "review.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    assert "Māori ≥ 3" in text
    # A reviewer accepts one row and writes a note with Korean text, then saves as CSV UTF-8.
    lines = text.splitlines()
    rows = list(csv.DictReader(lines))
    filled = sheet(
        [{k: rr._unsafe(v) for k, v in r.items()} for r in rows],
        {"acc": ("accept", "", "검토 완료 – Māori"), "rej": ("reject", "", "")},
    )
    (tmp_path / "filled.csv").write_bytes(filled.encode("utf-8-sig"))
    out = tmp_path / "updated.json"
    done = _cli(["reference-cases", "import-reviews", str(tmp_path / "filled.csv"), "--proposals",
                 str(paths["props"]), "--reference-cases", str(paths["set"]), "--project",
                 str(paths["project"]), "--reviewer", "Test Reviewer", "--reviewed-on",
                 "2026-11-26", "--out", str(out)], env)  # fmt: skip
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")
    report = json.loads(done.stdout.decode("utf-8"))
    assert report["imported"] == ["acc"]
    updated = ReferenceCasesV2.model_validate_json(out.read_text(encoding="utf-8"))
    note = ratings_of(updated, "c1")["o4"].note
    assert "검토 완료 – Māori" in note
    # The same sheet saved in Excel's Korean ANSI encoding imports with --sheet-encoding cp949
    # (Korean text only, which cp949 can hold) and is refused, with advice, without it.
    korean = (
        sheet(
            [{k: rr._unsafe(v) for k, v in r.items()} for r in rows],
            {"acc": ("accept", "", "검토 완료"), "rej": ("reject", "", "")},
        )
        .replace("Māori ≥ 3", "x")
        .replace("–", "-")
    )
    (tmp_path / "ansi.csv").write_bytes(korean.encode("cp949"))
    refused = _cli(["reference-cases", "import-reviews", str(tmp_path / "ansi.csv"), "--proposals",
                    str(paths["props"]), "--reference-cases", str(paths["set"]), "--project",
                    str(paths["project"]), "--reviewer", "R", "--reviewed-on", "2026-11-26",
                    "--out", str(tmp_path / "x.json")], env)  # fmt: skip
    assert refused.returncode == 1 and b"CSV UTF-8" in refused.stderr
    assert not (tmp_path / "x.json").exists()
    named = tmp_path / "ansi-updated.json"
    accepted = _cli(
        ["reference-cases", "import-reviews", str(tmp_path / "ansi.csv"), "--proposals",
         str(paths["props"]), "--reference-cases", str(paths["set"]), "--project",
         str(paths["project"]), "--reviewer", "R", "--reviewed-on", "2026-11-26",
         "--sheet-encoding", "cp949", "--out", str(named)], env,
    )  # fmt: skip
    assert accepted.returncode == 0, accepted.stderr.decode("utf-8", "replace")
    korean_set = ReferenceCasesV2.model_validate_json(named.read_text(encoding="utf-8"))
    assert "검토 완료" in ratings_of(korean_set, "c1")["o4"].note


def test_the_import_command_refuses_and_writes_nothing(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from mcda_studio.cli import app

    paths = _files(tmp_path)
    project, reference_set = b.project(), load_set(v2_set())
    props = load_proposals(json.loads(paths["props"].read_text(encoding="utf-8")))
    rows = rr.review_rows(project, reference_set, props)
    bad = tmp_path / "bad.csv"
    bad.write_bytes(sheet(rows, {"acc": ("edit", "Nope", "")}).encode("utf-8-sig"))
    out = tmp_path / "updated.json"
    result = CliRunner().invoke(app, [
        "reference-cases", "import-reviews", str(bad), "--proposals", str(paths["props"]),
        "--reference-cases", str(paths["set"]), "--project", str(paths["project"]),
        "--reviewer", "R", "--reviewed-on", "2026-11-26", "--out", str(out),
    ])  # fmt: skip
    assert result.exit_code == 1 and "RATING_EDIT_UNKNOWN_LABEL" in result.output
    assert not out.exists()
    packets_out = tmp_path / "packets.json"
    made = CliRunner().invoke(app, [
        "reference-cases", "packets", str(paths["set"]), "--project", str(paths["project"]),
        "--case", "c1", "--leaf", "o4", "--out", str(packets_out),
    ])  # fmt: skip
    assert made.exit_code == 0, made.output
    assert not keys(json.loads(packets_out.read_text(encoding="utf-8"))) & OUTCOME_KEYS


def test_a_second_sheet_is_written_beside_the_first_and_never_over_it(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from mcda_studio.cli import app

    paths = _files(tmp_path)
    out_dir = tmp_path / "review"
    base = ["reference-cases", "review-sheet", str(paths["props"]), "--reference-cases"]
    base += [str(paths["set"]), "--project", str(paths["project"]), "--out-dir", str(out_dir)]
    assert CliRunner().invoke(app, base).exit_code == 0
    first = (out_dir / "review.csv").read_bytes()
    second = CliRunner().invoke(app, [*base, "--stem", "review-2"])
    assert second.exit_code == 0, second.output
    assert (out_dir / "review-2.csv").exists() and (out_dir / "review-2.html").exists()
    assert (out_dir / "review.csv").read_bytes() == first
    again = CliRunner().invoke(app, [*base, "--stem", "review-2"])
    assert again.exit_code == 1 and "Refusing to overwrite" in again.output
    for stem in ("../elsewhere", "a\\b", ".."):
        bad = CliRunner().invoke(app, [*base, "--stem", stem])
        assert bad.exit_code == 1 and "plain file name" in bad.output, stem
