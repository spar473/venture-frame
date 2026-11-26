"""SMAA-2's samples weight source and schema 8 (``docs/design/sampled-weights-in-smaa.md``).

Every refusal code; S <= N draws distinct stored vectors and S > N needs resample; the index
stream leaves the main stream alone; a performance run ignores the file; the record names the
source; schema 8 only for a saved samples configuration, a version 7 file keeps its bytes, and a
version 7 reader refuses version 8.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mcda_studio.domain.errors import ProjectLoadError, ValidationFailed
from mcda_studio.domain.preferences import AnalysisConfiguration
from mcda_studio.domain.results import SmaaResult
from mcda_studio.domain.uncertainty import (
    CellUncertainty,
    GroupUncertainty,
    SmaaModel,
    UniformDistribution,
)
from mcda_studio.domain.weight_samples import WeightSamples
from mcda_studio.methods.smaa import samples as wsm
from mcda_studio.services import analysis_service as svc
from mcda_studio.services import conviction_prior_service as cps
from mcda_studio.services import project_service as ps
from mcda_studio.storage import json_repository as jr

from builders import tiny_project

ROOT = Path(__file__).resolve().parents[2]
VERSION_7 = ROOT / "examples" / "nz_first_round_retrospective" / "project.json"


def with_cells(project: Any) -> Any:
    cell = CellUncertainty(
        alternative_id="a",
        criterion_id="quality",
        distribution=UniformDistribution(lower=4, upper=8),
    )
    return ps.set_smaa_model(project, SmaaModel(cells=[cell]))


@pytest.fixture(scope="module")
def tiny() -> tuple[Any, WeightSamples]:
    project = with_cells(tiny_project())
    return project, cps.conviction_prior(project, None, kappa=4.0, draws=3000, seed=1)


def configuration(sidecar: WeightSamples | None, **extra: Any) -> AnalysisConfiguration:
    parameters: dict[str, Any] = {"weight_source": "samples", "uncertainty": "combined",
                                  "samples": 2000, "seed": 3}  # fmt: skip
    if sidecar is not None:
        parameters["weight_samples_sha256"] = wsm.digest(sidecar)
    parameters.update(extra)
    return AnalysisConfiguration(method="smaa2", parameters=parameters)


def run(project: Any, cfg: AnalysisConfiguration, samples: WeightSamples | None) -> SmaaResult:
    with wsm.supplied(samples):
        result = svc.evaluate_project(project, cfg)
    assert isinstance(result, SmaaResult)
    return result


def refused(project: Any, cfg: AnalysisConfiguration, samples: WeightSamples | None) -> set[str]:
    with pytest.raises(ValidationFailed) as caught, wsm.supplied(samples):
        svc.evaluate_project(project, cfg)
    return {i.code for i in caught.value.report.errors}


# ---- the run ------------------------------------------------------------------------------


def test_a_samples_run_records_its_source(tiny: Any) -> None:
    project, samples = tiny
    result = run(project, configuration(samples), samples)
    assert result.algorithm == "smaa2.pcg64.v1+samples"
    use = result.weight_samples
    assert use is not None
    assert (use.sha256, use.basis, use.stored, use.used) == (
        wsm.digest(samples),
        "prior",
        3000,
        2000,
    )
    assert use.index_method == "without replacement" and use.note is None
    assert use.index_stream == "PCG64(3).jumped(2)"
    assert "WEIGHT_SAMPLES" in [d.code for d in result.diagnostics]
    raw = json.loads(result.model_dump_json())
    assert raw["weight_samples"]["stored"] == 3000


def test_a_stated_run_keeps_its_shape(tiny: Any) -> None:
    project = ps.set_smaa_model(
        tiny_project(), SmaaModel(groups=[GroupUncertainty(parent_id=None, kind="unknown")])
    )
    result = run(project, AnalysisConfiguration(method="smaa2", parameters={"samples": 1000}), None)
    raw = json.loads(result.model_dump_json())
    assert "weight_samples" not in raw and raw["algorithm"] == "smaa2.pcg64.v1"


def test_the_same_seed_repeats_and_another_differs(tiny: Any) -> None:
    project, samples = tiny
    one = run(project, configuration(samples), samples)
    two = run(project, configuration(samples), samples)
    other = run(project, configuration(samples, seed=4), samples)
    assert one.model_dump_json() == two.model_dump_json()
    assert one.acceptability[0].ranks[0].count != other.acceptability[0].ranks[0].count


def test_indices_are_distinct_without_resample_and_reused_only_with_it() -> None:
    picked = wsm.indices(5, 1000, 3000, False)
    assert len(set(picked.tolist())) == 1000 and picked.min() >= 0 and picked.max() < 3000
    assert np.array_equal(picked, wsm.indices(5, 1000, 3000, False))
    full = wsm.indices(5, 3000, 3000, False)
    assert sorted(full.tolist()) == list(range(3000))  # every stored vector exactly once
    reused = wsm.indices(5, 5000, 3000, True)
    assert len(reused) == 5000 and len(set(reused.tolist())) < 5000
    with pytest.raises(ValueError, match="needs resample"):
        wsm.indices(5, 5000, 3000, False)


def test_more_draws_than_vectors_needs_resample_and_says_so(tiny: Any) -> None:
    project, samples = tiny
    assert "WEIGHT_SAMPLES_TOO_FEW" in refused(
        project, configuration(samples, samples=5000), samples
    )
    combined = run(project, configuration(samples, samples=5000, resample=True), samples)
    use = combined.weight_samples
    assert use is not None and use.index_method == "with replacement"
    assert use.note == wsm.RESAMPLE_SENTENCE.format(S=5000, N=3000)
    assert use.interval_n is None  # a combined run keeps n = S


def test_a_resampled_preference_run_bases_its_intervals_on_n(tiny: Any) -> None:
    from mcda_studio.methods.smaa.intervals import clopper_pearson

    project, samples = tiny
    cfg = configuration(samples, samples=6000, resample=True, uncertainty="preference")
    result = run(project, cfg, samples)
    assert result.weight_samples is not None and result.weight_samples.interval_n == 3000
    first = result.acceptability[0].ranks[0]
    assert first.count is not None and result.samples == 6000
    scaled = int(first.count * 3000 / 6000 + 0.5)
    assert (first.lower, first.upper) == clopper_pearson(scaled, 3000, result.confidence)
    assert first.frequency == first.count / 6000


def test_the_index_stream_leaves_the_measurement_draws_alone(tiny: Any) -> None:
    """With every stored vector equal to the stated weights, a combined samples run must count
    exactly what a performance run (the same cells, the same seed, fixed weights) counts."""
    project, samples = tiny
    stated = samples.prior.centre_global
    flat = WeightSamples.model_validate(
        {**samples.model_dump(mode="json"),
         "samples": [[stated[leaf] for leaf in samples.leaf_ids]] * len(samples.samples)}
    )  # fmt: skip
    from_samples = run(project, configuration(flat), flat)
    performance = run(
        project,
        AnalysisConfiguration(method="smaa2", parameters={"uncertainty": "performance",
                                                          "samples": 2000, "seed": 3}),
        None,
    )  # fmt: skip
    for a, b in zip(from_samples.acceptability, performance.acceptability, strict=True):
        assert [r.count for r in a.ranks] == [r.count for r in b.ranks]


def test_a_performance_run_ignores_the_file(tiny: Any) -> None:
    project, _samples = tiny
    cfg = configuration(None, uncertainty="performance")  # no digest and no sidecar
    result = run(project, cfg, None)
    assert result.weight_samples is None and result.algorithm == "smaa2.pcg64.v1"


# ---- refusals ------------------------------------------------------------------------------


def test_every_refusal(tiny: Any) -> None:
    project, samples = tiny
    assert "WEIGHT_SAMPLES_MISSING" in refused(project, configuration(samples), None)
    cfg = configuration(samples)
    kept = {k: v for k, v in cfg.parameters.items() if k != "weight_samples_sha256"}
    no_digest = cfg.model_copy(update={"parameters": kept})
    assert "WEIGHT_SAMPLES_DIGEST_REQUIRED" in refused(project, no_digest, samples)
    other = cps.conviction_prior(project, None, kappa=4.0, draws=3000, seed=2)
    assert "WEIGHT_SAMPLES_DIGEST_MISMATCH" in refused(project, cfg, other)

    def variant(**update: Any) -> tuple[WeightSamples, AnalysisConfiguration]:
        changed = samples.model_copy(update=update)
        return changed, configuration(changed)

    s, c = variant(usable=False)
    assert "WEIGHT_SAMPLES_NOT_USABLE" in refused(project, c, s)
    s, c = variant(leaf_ids=list(reversed(samples.leaf_ids)))
    assert "WEIGHT_SAMPLES_TREE_MISMATCH" in refused(project, c, s)
    s, c = variant(weight_profile_id="nobody")
    assert "WEIGHT_SAMPLES_UNKNOWN_PROFILE" in refused(project, c, s)
    renamed = ps.rename_alternative(project, "a", "A renamed")
    assert "WEIGHT_SAMPLES_STALE" in refused(renamed, cfg, samples)
    grouped = ps.set_smaa_model(
        project, SmaaModel(cells=project.smaa.cells,
                           groups=[GroupUncertainty(parent_id=None, kind="unknown")])
    )  # fmt: skip
    assert "WEIGHT_SOURCE_CONFLICT" in refused(grouped, cfg, samples)


def test_the_run_profile_must_be_the_sidecars(tiny: Any) -> None:
    from mcda_studio.domain.preferences import WeightProfile

    project, _ = tiny
    two = ps.add_weight_profile(project, WeightProfile(id="second", name="Second"))
    samples = cps.conviction_prior(two, None, kappa=4.0, draws=3000, seed=1)
    cfg = configuration(samples).model_copy(update={"weight_profile_id": "second"})
    assert "WEIGHT_SAMPLES_PROFILE_MISMATCH" in refused(two, cfg, samples)


def test_saving_the_source_does_not_make_the_samples_stale(tiny: Any) -> None:
    project, samples = tiny
    cfg = configuration(samples)
    saved = project.model_copy(update={"analysis": cfg})
    # The sidecar was drawn before the method and parameters were saved: still stale, because the
    # method changed; drawn after saving the method, it stays valid once the source is added.
    with_method = project.model_copy(
        update={"analysis": project.analysis.model_copy(update={"method": "smaa2",
                "parameters": {"uncertainty": "combined", "samples": 2000, "seed": 3}})}
    )  # fmt: skip
    after = cps.conviction_prior(with_method, None, kappa=4.0, draws=3000, seed=1)
    pinned = with_method.model_copy(
        update={"analysis": with_method.analysis.model_copy(update={"parameters": {
            **with_method.analysis.parameters, "weight_source": "samples",
            "weight_samples_sha256": wsm.digest(after)}})}
    )  # fmt: skip
    assert wsm.fitted_fingerprint(pinned) == after.project_fingerprint
    assert run(pinned, pinned.analysis, after).weight_samples is not None
    assert wsm.fitted_fingerprint(saved) != samples.project_fingerprint


# ---- schema 8 ------------------------------------------------------------------------------


def test_version_7_and_older_files_keep_their_bytes(tmp_path: Path) -> None:
    from mcda_studio.domain.preferences import Elicitation

    repo = jr.JsonProjectRepository()
    base = tiny_project()
    profile = base.weight_profiles[0].model_copy(
        update={"elicitation": Elicitation(conviction="low")}
    )
    v7 = base.model_copy(update={"weight_profiles": [profile]})
    assert v7.required_schema_version() == 7
    first = repo.save(v7, tmp_path / "v7.json")
    second = repo.save(repo.load(first), tmp_path / "v7-again.json")
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text(encoding="utf-8"))["schema_version"] == 7
    case_2 = ROOT / "examples" / "nz_venture_screening" / "project.json"
    text = case_2.read_text(encoding="utf-8")
    again = repo.save(repo.load(case_2), tmp_path / "case2.json")
    assert again.read_text(encoding="utf-8") == text


def test_a_saved_samples_configuration_is_version_8(tiny: Any, tmp_path: Path) -> None:
    project, samples = tiny
    saved = project.model_copy(update={"analysis": configuration(samples)})
    assert saved.required_schema_version() == 8
    path = jr.JsonProjectRepository().save(saved, tmp_path / "v8.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 8
    assert jr.JsonProjectRepository().load(path).analysis.parameters["weight_source"] == "samples"
    stated = project.model_copy(update={"analysis": configuration(samples, weight_source="stated")})
    assert stated.required_schema_version() < 8


def test_a_version_7_file_cannot_hold_the_samples_source(tiny: Any, tmp_path: Path) -> None:
    project, samples = tiny
    raw = json.loads(
        project.model_copy(update={"analysis": configuration(samples)}).model_dump_json()
    )
    raw["schema_version"] = 7
    with pytest.raises(ProjectLoadError, match="schema version 7 cannot contain"):
        jr.JsonProjectRepository().from_dict(raw)


def test_a_version_7_reader_refuses_version_8(tiny: Any, monkeypatch: Any) -> None:
    project, samples = tiny
    raw = json.loads(
        project.model_copy(update={"analysis": configuration(samples)}).model_dump_json()
    )
    monkeypatch.setattr(jr, "SCHEMA_VERSION", 7)
    with pytest.raises(ProjectLoadError, match="version 8 is newer than this program supports"):
        jr.JsonProjectRepository().from_dict(raw)


# ---- the command line and the API -----------------------------------------------------------


def test_the_command_line_reads_a_sidecar(tiny: Any, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from mcda_studio.cli import app

    project, _ = tiny
    path = jr.JsonProjectRepository().save(project, tmp_path / "p.json")
    prior = tmp_path / "prior.json"
    made = CliRunner().invoke(app, ["calibrate", "prior", str(path), "--kappa", "4", "--draws",
                                    "2000", "--out", str(prior)])  # fmt: skip
    assert made.exit_code == 0, made.output
    done = CliRunner().invoke(app, ["evaluate", str(path), "--method", "smaa2", "--samples",
                                    "1000", "--weight-samples", str(prior), "--json"])  # fmt: skip
    assert done.exit_code == 0, done.output
    result = json.loads(done.stdout)
    assert result["weight_samples"]["stored"] == 2000 and result["weight_samples"]["used"] == 1000
    too_many = CliRunner().invoke(app, ["evaluate", str(path), "--method", "smaa2", "--samples",
                                        "3000", "--weight-samples", str(prior)])  # fmt: skip
    assert too_many.exit_code == 1 and "WEIGHT_SAMPLES_TOO_FEW" in too_many.output
    lonely = CliRunner().invoke(app, ["evaluate", str(path), "--resample"])
    assert lonely.exit_code == 1 and "--resample needs --weight-samples" in lonely.output


def test_the_api_evaluates_with_a_sidecar(tiny: Any, tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from mcda_studio.api import create_app

    project, samples = tiny
    client = TestClient(create_app(store_dir=tmp_path / "store", frontend_dir=False))
    body = {"project": json.loads(project.model_dump_json()),
            "configuration": configuration(samples).model_dump(mode="json")}  # fmt: skip
    missing = client.post("/api/evaluate", json=body)
    assert missing.status_code == 422 and "WEIGHT_SAMPLES_MISSING" in missing.text
    body["weight_samples"] = json.loads(samples.model_dump_json())
    ok = client.post("/api/evaluate", json=body)
    assert ok.status_code == 200, ok.text
    assert ok.json()["weight_samples"]["sha256"] == wsm.digest(samples)


def test_indices_come_from_the_jumped_stream_by_partial_fisher_yates() -> None:
    rng = np.random.Generator(np.random.PCG64(9).jumped(2))
    order = list(range(50))
    for i in range(20):
        j = int(rng.integers(i, 50))
        order[i], order[j] = order[j], order[i]
    assert wsm.indices(9, 20, 50, False).tolist() == order[:20]
    again = np.random.Generator(np.random.PCG64(9).jumped(2)).integers(0, 50, size=80)
    assert wsm.indices(9, 80, 50, True).tolist() == again.tolist()
