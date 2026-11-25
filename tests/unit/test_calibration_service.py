"""Calibration ``fit`` on a real project and a version 2 reference set: read-only behaviour, which
ratings and outcomes enter, the sufficiency refusal, diagnostic runs, the convergence gate, the
``weight-samples`` sidecar's strictness, and the commands.
"""

import copy
import datetime as dt
import hashlib
import json
import math
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

import reference_case_builders as b
from mcda_studio.calibration import service as svc
from mcda_studio.calibration import weight_samples as ws
from mcda_studio.calibration.sampler import SamplerSettings
from mcda_studio.domain.errors import McdaError
from mcda_studio.domain.reference_cases_v2 import ReferenceCasesV2
from mcda_studio.domain.uncertainty import GroupUncertainty, SmaaModel
from mcda_studio.services import project_service as ps

SNAPSHOT = b.SNAPSHOT  # 2021-10-27; the 36-month window ends 2024-10-27
END = dt.date(2024, 10, 27)
LENGTH = (END - SNAPSHOT).days
FOLLOW_UP = dt.date(2026, 11, 20)
LABELS = {"o4": b.FOUR, "o5": b.FIVE, "cat": b.COLOURS, "o4b": b.FOUR, "o5b": b.FIVE}
VALUES = {
    "o4": {label: i / 3 for i, label in enumerate(b.FOUR)},
    "o5": {label: i / 4 for i, label in enumerate(b.FIVE)},
    "cat": {"red": 0.0, "green": 0.5, "blue": 1.0},
    "o4b": {label: i / 3 for i, label in enumerate(b.FOUR)},
    "o5b": {label: i / 4 for i, label in enumerate(b.FIVE)},
}
WEIGHTS = {"o4": 1 / 8, "o5": 1 / 8, "cat": 2 / 8, "o4b": 2 / 8, "o5b": 1 / 8}
FAST = SamplerSettings(warmup=2000, draws=3000, samples=2000, seed=0)


def outcome(status: str, **extra: Any) -> dict[str, Any]:
    return {"definition": "next_36m", "status": status, **extra}


def case(case_id: str, labels: dict[str, str], obs: dict[str, Any], review="reviewed") -> dict:
    raw = b.case(case_id, labels)
    del raw["outcome"]
    for rating in raw["ratings"]:
        rating["review"] = review
    raw["outcomes"] = [obs]
    return raw


def reference_set(cases: list[dict[str, Any]]) -> dict[str, Any]:
    raw = b.reference_set(*cases)
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


def simulated(n: int = 160, seed: int = 1, review: str = "reviewed") -> dict[str, Any]:
    """Cases simulated from the model with the profile's weights, alpha -1.5 and beta 3."""
    rng = np.random.Generator(np.random.PCG64(seed))
    cases = []
    for i in range(n):
        labels = {leaf: str(rng.choice(options)) for leaf, options in LABELS.items()}
        s = sum(WEIGHTS[leaf] * VALUES[leaf][label] for leaf, label in labels.items()) / sum(
            WEIGHTS.values()
        )
        t = rng.exponential(1.0 / math.exp(-1.5 + 3.0 * s))
        f = rng.random() if rng.random() < 0.3 else 2.0
        if t <= min(f, 1.0):
            day = SNAPSHOT + dt.timedelta(days=int(t * LENGTH))
            obs = outcome("event", event_date=day.isoformat(), claim_ids=["c2"])
        elif f >= 1.0:
            obs = outcome("no_event", observed_through=FOLLOW_UP.isoformat())
        else:
            seen = SNAPSHOT + dt.timedelta(days=int(f * LENGTH))
            obs = outcome("censored", observed_through=seen.isoformat())
        cases.append(case(f"k{i}", labels, obs, review))
    return reference_set(cases)


def raw_bytes(raw: dict[str, Any]) -> bytes:
    return json.dumps(raw).encode("utf-8")


@pytest.fixture(scope="module")
def converged() -> tuple[Any, bytes, svc.FitOutcome]:
    project = b.project()
    data = raw_bytes(simulated())
    return project, data, svc.fit(project, None, data, conviction="medium", settings=FAST)


# ---- the main path ------------------------------------------------------------------------------


def test_a_fit_converges_and_writes_a_valid_sidecar(converged) -> None:
    _project, data, outcome_ = converged
    report = outcome_.report
    assert report.status == "converged", report.diagnostics
    assert report.samples_available and outcome_.samples is not None
    samples = outcome_.samples
    assert samples.usable and not samples.diagnostic and samples.review_states == ["reviewed"]
    assert samples.reference_set_sha256 == hashlib.sha256(data).hexdigest()
    assert samples.reference_set_digest_basis == "file bytes"
    assert len(samples.samples) == FAST.samples == len(samples.alpha) == len(samples.beta)
    assert samples.leaf_ids == ["o4", "o5", "cat", "o4b", "o5b", "num", "zero"]
    assert all(row[-1] == 0.0 for row in samples.samples)  # the zero-weight leaf stays at 0
    assert samples.label == ws.LABEL
    assert report.data.compared_criteria == ["o4", "o5", "cat", "o4b", "o5b"]


def test_the_prior_is_centred_on_the_profile_with_the_conviction_recorded(converged) -> None:
    _project, _data, outcome_ = converged
    prior = outcome_.report.prior
    assert prior is not None
    assert prior.conviction == "medium" and prior.conviction_source == "stated at the command line"
    assert prior.half_width == 0.30 and prior.base_rate == 0.15
    (top,) = prior.groups
    assert top.parent_id is None and top.held == ["zero"]
    assert top.children == ["o4", "o5", "cat", "o4b", "o5b", "num"]
    assert top.centre == pytest.approx([1 / 8, 1 / 8, 2 / 8, 2 / 8, 1 / 8, 1 / 8])
    assert top.kappa_method.startswith("Gauss-Legendre")
    assert prior.centre_global["cat"] == pytest.approx(0.25)


def test_fitting_never_changes_the_project_or_the_reference_set() -> None:
    project = b.project()
    before_json, before_print = project.model_dump_json(), project.fingerprint()
    raw = simulated(60)
    data = raw_bytes(raw)
    model = ReferenceCasesV2.model_validate_json(data)
    model_before = model.model_dump_json()
    svc.fit(project, None, model, conviction="low",
            settings=SamplerSettings(warmup=50, draws=50, samples=100))  # fmt: skip
    assert project.model_dump_json() == before_json and project.fingerprint() == before_print
    assert model.model_dump_json() == model_before


def test_a_model_input_is_digested_as_canonical_json() -> None:
    project = b.project()
    model = ReferenceCasesV2.model_validate_json(raw_bytes(simulated(40)))
    outcome_ = svc.fit(project, None, model, conviction="low",
                       settings=SamplerSettings(warmup=50, draws=50, samples=100))  # fmt: skip
    assert outcome_.report.reference_set_digest_basis == "canonical JSON"
    expected = hashlib.sha256(model.model_dump_json().encode("utf-8")).hexdigest()
    assert outcome_.report.reference_set_sha256 == expected


# ---- what enters ---------------------------------------------------------------------------------


def test_unknown_and_unrecorded_outcomes_and_thin_cases_are_excluded_and_counted() -> None:
    full = {leaf: options[0] for leaf, options in LABELS.items()}
    half = dict(list(full.items())[:3])  # 3 of 5: at least half
    thin = dict(list(full.items())[:2])  # 2 of 5: under half
    cases = [
        case("a", full, outcome("event", event_date="2022-03-04", claim_ids=["c2"])),
        case("b", half, outcome("no_event", observed_through=FOLLOW_UP.isoformat())),
        case("c", thin, outcome("no_event", observed_through=FOLLOW_UP.isoformat())),
        case("d", full, outcome("unknown")),
    ]
    raw = reference_set(cases)
    raw["cases"].append({**case("e", full, outcome("unknown")), "outcomes": []})
    report = svc.fit(b.project(), None, raw_bytes(raw), conviction="low",
                     settings=SamplerSettings(warmup=10, draws=10, samples=20)).report  # fmt: skip
    assert report.data.entering_cases == 2
    assert report.data.excluded == {
        "outcome not recorded": 1,
        "outcome unknown": 1,
        "too few rated compared criteria": 1,
    }
    assert report.data.statuses == {"event": 1, "no_event": 1}
    assert report.data.rated_cases == {"o4": 2, "o5": 2, "cat": 2, "o4b": 1, "o5b": 1}


def test_outcomes_are_encoded_in_window_fractions_with_partial_dates_as_intervals() -> None:
    full = {leaf: options[0] for leaf, options in LABELS.items()}
    cases = [
        case("ev", full, outcome("event", event_date="2022-03-04", claim_ids=["c2"])),
        case("pm", full, outcome("event", event_date="2022-03", claim_ids=["c2"])),
        case("ce", full, outcome("censored", observed_through="2023-01-31")),
        case("co", full, outcome("competing_event", event_date="2022-Q2", claim_ids=["c2"])),
        case("no", full, outcome("no_event", observed_through=FOLLOW_UP.isoformat())),
    ]
    rs = ReferenceCasesV2.model_validate_json(raw_bytes(reference_set(cases)))
    leaves = list(LABELS)
    got = svc._cases(b.project(), rs, "next_36m", leaves, False, svc.parse_fraction("1/2"))
    ev, pm, ce, co, no = got.observations
    days = lambda d: (d - SNAPSHOT).days / LENGTH  # noqa: E731
    assert (ev.density, ev.exposure, ev.width) == (1.0, days(dt.date(2022, 3, 4)), 0.0)
    assert pm.density == 0.0 and pm.exposure == days(dt.date(2022, 3, 1))
    assert pm.width == pytest.approx(days(dt.date(2022, 4, 1)) - days(dt.date(2022, 3, 1)))
    assert (ce.density, ce.exposure, ce.width) == (0.0, days(dt.date(2023, 1, 31)), 0.0)
    assert (co.density, co.exposure, co.width) == (0.0, days(dt.date(2022, 4, 1)), 0.0)
    assert (no.density, no.exposure, no.width) == (0.0, 1.0, 0.0)
    assert got.values[0] == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_only_reviewed_ratings_count_unless_the_run_is_diagnostic() -> None:
    raw = simulated(60, review="proposed")
    raw["cases"][0]["ratings"][0]["review"] = "reviewed"  # one reviewed rating is not enough
    report = svc.fit(b.project(), None, raw_bytes(raw)).report
    assert report.status == "insufficient" and report.data.entering_cases == 0
    raw = simulated(60, review="proposed")
    outcome_ = svc.fit(b.project(), None, raw_bytes(raw))
    assert outcome_.samples is None
    assert outcome_.report.insufficient[0] == "0 reviewed ratings on the compared criteria"


def test_a_diagnostic_run_says_so_everywhere_and_is_never_usable() -> None:
    raw = simulated(review="proposed")
    outcome_ = svc.fit(b.project(), None, raw_bytes(raw), conviction="medium",
                       include_proposed=True, settings=FAST)  # fmt: skip
    report = outcome_.report
    assert report.diagnostic and report.diagnostic_label == ws.DIAGNOSTIC_LABEL
    assert report.status == "converged" and outcome_.samples is not None
    assert outcome_.samples.diagnostic and not outcome_.samples.usable
    assert outcome_.samples.review_states == ["proposed", "reviewed"]
    assert ws.DIAGNOSTIC_LABEL in outcome_.samples.label


def test_a_diagnostic_run_with_no_entering_case_returns_the_prior() -> None:
    raw = simulated(30, review="proposed")
    for c in raw["cases"]:
        c["outcomes"] = [outcome("unknown")]
    outcome_ = svc.fit(b.project(), None, raw_bytes(raw), conviction="medium",
                       include_proposed=True, settings=FAST)  # fmt: skip
    report = outcome_.report
    assert report.data.entering_cases == 0 and report.insufficient
    assert report.prior_dominated is True
    assert outcome_.samples is not None and not outcome_.samples.usable
    for leaf in report.leaves:
        if leaf.varies:
            assert leaf.posterior_mean == pytest.approx(leaf.prior_mean, abs=6 * leaf.prior_sd / 30)


def test_a_conviction_is_required_when_the_profile_has_none() -> None:
    with pytest.raises(McdaError, match="no conviction"):
        svc.fit(b.project(), None, raw_bytes(simulated(40)), settings=FAST)


def test_held_children_named_in_the_projects_smaa_section_are_held() -> None:
    project = ps.set_smaa_model(
        b.project(),
        SmaaModel(groups=[GroupUncertainty(parent_id=None, kind="unknown", held=["cat"])]),
    )
    outcome_ = svc.fit(project, None, raw_bytes(simulated(60)), conviction="low",
                       settings=SamplerSettings(warmup=100, draws=100, samples=200))  # fmt: skip
    assert outcome_.report.prior is not None
    (top,) = outcome_.report.prior.groups
    assert "cat" in top.held and "cat" not in top.children
    columns = outcome_.report.leaves
    cat = next(row for row in columns if row.criterion_id == "cat")
    assert cat.posterior_sd == 0.0 and cat.posterior_mean == pytest.approx(0.25)


def test_a_short_run_is_not_converged_and_gives_no_samples() -> None:
    outcome_ = svc.fit(b.project(), None, raw_bytes(simulated(80)), conviction="medium",
                       settings=SamplerSettings(warmup=25, draws=25, samples=100))  # fmt: skip
    assert outcome_.report.status == "not_converged"
    assert outcome_.samples is None and not outcome_.report.samples_available


def test_inputs_are_checked() -> None:
    project, data = b.project(), raw_bytes(simulated(20))
    with pytest.raises(McdaError, match="unknown weight profile"):
        svc.fit(project, "nope", data)
    with pytest.raises(McdaError, match="unknown outcome definition"):
        svc.fit(project, None, data, definition="nope")
    with pytest.raises(McdaError, match="minimum rated fraction"):
        svc.fit(project, None, data, min_rated_fraction="0")
    raw = simulated(20)
    raw["cases"][0]["outcomes"][0]["status"] = "censored"
    raw["cases"][0]["outcomes"][0].pop("event_date", None)
    raw["cases"][0]["outcomes"][0].pop("claim_ids", None)
    raw["cases"][0]["outcomes"][0]["observed_through"] = FOLLOW_UP.isoformat()
    with pytest.raises(McdaError, match="OUTCOME_CENSORED_AFTER_WINDOW"):
        svc.fit(project, None, raw_bytes(raw))


def test_the_label_and_limitations_use_no_domain_words() -> None:
    text = " ".join([ws.LABEL, ws.DIAGNOSTIC_LABEL, svc.MISSING_AT_RANDOM, *ws.LIMITATIONS])
    assert "not a forecast" in ws.LABEL
    for word in ("funding", "series", "startup", "venture", "investor", "company", "founder"):
        assert word not in text.lower()


# ---- the sidecar ---------------------------------------------------------------------------


def _sidecar(converged) -> dict[str, Any]:
    return json.loads(converged[2].samples.model_dump_json())


def test_a_calibrated_sidecar_written_before_basis_existed_reads_as_calibrated(converged) -> None:
    raw = _sidecar(converged)
    assert raw["basis"] == "calibrated"
    del raw["basis"]
    assert ws.WeightSamples.model_validate_json(json.dumps(raw)).basis == "calibrated"


def test_the_sidecar_round_trips(converged) -> None:
    samples = converged[2].samples
    again = ws.WeightSamples.model_validate_json(samples.model_dump_json())
    assert again == samples
    assert again.model_dump_json() == samples.model_dump_json()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r.update(extra=1), "Extra inputs"),
        (lambda r: r.update(version="1"), "explicit integers"),
        (lambda r: r.update(version=1.0), "explicit integers"),
        (lambda r: r.update(version=2), "version 2 is not supported"),
        (lambda r: r.update(kind="reference-cases"), "not a weight-samples file"),
        (lambda r: r["diagnostics"].update(converged=False), "did not converge"),
        (lambda r: r.update(usable=False), "usable must be true"),
        (lambda r: r.update(diagnostic=True), "usable must be true"),
        (lambda r: r.update(review_states=["proposed", "reviewed"]), "review_states"),
        (lambda r: r.update(label="Forecast weights."), "fixed calibration label"),
        (lambda r: r["samples"].pop(), "expected 2000 samples"),
        (lambda r: r["samples"][0].__setitem__(0, r["samples"][0][0] + 0.01), "sum to 1"),
        (lambda r: r["samples"][0].append(0.0), "one weight per leaf"),
        (lambda r: r["beta"].__setitem__(0, -1.0), "beta is positive"),
        (lambda r: r["data"].update(extra=1), "Extra inputs"),
    ],
)
def test_the_sidecar_is_strict(converged, change, message: str) -> None:
    raw = copy.deepcopy(_sidecar(converged))
    change(raw)
    with pytest.raises(ValidationError, match=message):
        ws.WeightSamples.model_validate_json(json.dumps(raw))


# ---- the commands --------------------------------------------------------------------------


def _cli(tmp_path: Any, raw: dict[str, Any], *extra: str) -> Any:
    from typer.testing import CliRunner

    from mcda_studio.cli import app
    from mcda_studio.storage.json_repository import JsonProjectRepository

    project_file = tmp_path / "project.json"
    JsonProjectRepository().save(b.project(), project_file)
    rs_file = tmp_path / "set.json"
    rs_file.write_text(json.dumps(raw), encoding="utf-8")
    return CliRunner().invoke(
        app, ["calibrate", "fit", str(project_file), "--reference-cases", str(rs_file), *extra]
    )


def test_the_command_refuses_insufficient_data(tmp_path: Any) -> None:
    result = _cli(tmp_path, simulated(30, review="proposed"))
    assert result.exit_code == 1
    assert "insufficient: 0 reviewed ratings" in result.output
    assert json.loads(result.stdout[: result.stdout.rindex("}") + 1])["status"] == "insufficient"


def test_the_command_writes_a_converged_sidecar_once(tmp_path: Any) -> None:
    out = tmp_path / "samples.json"
    args = ("--conviction", "medium", "--samples", "2000",
            "--out", str(out))  # fmt: skip
    result = _cli(tmp_path, simulated(), *args)
    assert result.exit_code == 0, result.output
    samples = ws.WeightSamples.model_validate_json(out.read_text(encoding="utf-8"))
    assert samples.usable
    again = _cli(tmp_path, simulated(), *args)
    assert again.exit_code == 1 and "Refusing to overwrite" in again.output


def test_the_command_writes_nothing_when_not_converged(tmp_path: Any) -> None:
    out = tmp_path / "samples.json"
    result = _cli(tmp_path, simulated(80), "--conviction", "medium", "--warmup", "25",
                  "--draws", "25", "--samples", "100", "--out", str(out))  # fmt: skip
    assert result.exit_code == 1 and "not converged" in result.output
    assert not out.exists()


def test_the_synthetic_check_command(tmp_path: Any) -> None:
    from typer.testing import CliRunner

    from mcda_studio.cli import app

    result = CliRunner().invoke(app, ["calibrate", "synthetic-check", "--cases", "300"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["diagnostics"]["converged"] and data["cases_entering"] == 300
    assert "says nothing about any real outcome" in data["label"]


def test_a_case_rating_exactly_half_the_compared_leaves_enters() -> None:
    # Profile "four" rates o5b 0, so four leaves are compared: two rated is exactly half.
    full = {leaf: options[0] for leaf, options in LABELS.items() if leaf != "o5b"}
    two = dict(list(full.items())[:2])
    one = dict(list(full.items())[:1])
    cases = [
        case("half", two, outcome("event", event_date="2022-03-04", claim_ids=["c2"])),
        case("under", one, outcome("no_event", observed_through=FOLLOW_UP.isoformat())),
        case("all", full, outcome("no_event", observed_through=FOLLOW_UP.isoformat())),
    ]
    report = svc.fit(b.project(), "four", raw_bytes(reference_set(cases)), conviction="low",
                     settings=SamplerSettings(warmup=10, draws=10, samples=20)).report  # fmt: skip
    assert report.data.compared_criteria == ["o4", "o5", "cat", "o4b"]
    assert report.data.entering_cases == 2
    assert report.data.excluded == {"too few rated compared criteria": 1}
    assert report.data.min_rated_fraction == "1/2"
