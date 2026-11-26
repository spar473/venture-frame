"""SMAA-2's ``samples`` weight source (``docs/design/sampled-weights-in-smaa.md``).

With ``weight_source: "samples"`` a preference or combined run takes each draw's global weights
from a ``weight-samples`` sidecar instead of drawing them from stated weight groups. The sidecar is
never stored in the project; the caller supplies it for one evaluation (``supplied``), and the
saved parameter ``weight_samples_sha256`` pins which file a configuration means.

Draw matching never recycles silently. ``S`` SMAA-2 draws and ``N`` stored vectors:

- ``S <= N``: ``S`` distinct indices, a partial Fisher-Yates shuffle on ``PCG64(seed).jumped(2)``;
- ``S > N``: refused (``WEIGHT_SAMPLES_TOO_FEW``) unless ``resample`` is true, and then ``S``
  indices with replacement from the same stream.

The index stream is separate from the main stream, so measurement draws are untouched, and it
uses integers only, so the indices are the same on every platform.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mcda_studio.domain.common import Issue, Severity
from mcda_studio.domain.preferences import AnalysisConfiguration
from mcda_studio.domain.project import Project
from mcda_studio.domain.weight_samples import WeightSamples

#: The SMAA-2 parameters this source adds; a project that saves ``weight_source: "samples"`` is
#: written as schema version 8.
SOURCE_PARAMETERS = ("weight_source", "weight_samples_sha256", "resample")
STREAM = "smaa2.pcg64.v1+samples"
INDEX_STREAM_JUMP = 2
RESAMPLE_SENTENCE = (
    "{S} draws reuse {N} stored weight vectors; interval widths for weight-driven frequencies "
    "reflect {S} draws but the weights carry the information of {N}."
)

_SUPPLIED: ContextVar[WeightSamples | None] = ContextVar("weight_samples", default=None)


@contextmanager
def supplied(samples: WeightSamples | None) -> Iterator[None]:
    """Make ``samples`` the sidecar for the evaluations inside the block."""
    token = _SUPPLIED.set(samples)
    try:
        yield
    finally:
        _SUPPLIED.reset(token)


def supplied_samples() -> WeightSamples | None:
    return _SUPPLIED.get()


def digest(samples: WeightSamples) -> str:
    """SHA-256 of the sidecar's canonical JSON (sorted keys, UTF-8, ``(',', ':')``)."""
    text = json.dumps(
        samples.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fitted_fingerprint(project: Project) -> str:
    """The project's fingerprint with this source's parameters removed: the project as it was
    when the samples were drawn, if nothing else has changed since."""
    parameters = {
        k: v for k, v in project.analysis.parameters.items() if k not in SOURCE_PARAMETERS
    }
    stripped = project.model_copy(
        update={"analysis": project.analysis.model_copy(update={"parameters": parameters})}
    )
    return stripped.fingerprint()


def uses_samples(params: Any) -> bool:
    """Whether this run takes its weights from samples (a performance run never does)."""
    return (
        params is not None
        and params.weight_source == "samples"
        and params.uncertainty in ("preference", "combined")
    )


def check(project: Project, configuration: AnalysisConfiguration, params: Any) -> list[Issue]:
    """Every refusal for a run that uses the samples source; none otherwise."""
    if not uses_samples(params):
        return []
    issues: list[Issue] = []

    def add(code: str, message: str, path: str) -> None:
        issues.append(Issue(severity=Severity.ERROR, code=code, message=message, path=path))

    at = "configuration.parameters"
    pinned = params.weight_samples_sha256
    if pinned is None:
        add(
            "WEIGHT_SAMPLES_DIGEST_REQUIRED",
            "weight_source 'samples' needs weight_samples_sha256, the digest of the sidecar it "
            "means",
            f"{at}.weight_samples_sha256",
        )
    if project.smaa is not None and project.smaa.groups:
        add(
            "WEIGHT_SOURCE_CONFLICT",
            "the project states weight uncertainty in sibling groups; a run takes its weights "
            "either from those groups or from samples, never both",
            "smaa.groups",
        )
    samples = supplied_samples()
    if samples is None:
        add(
            "WEIGHT_SAMPLES_MISSING",
            "weight_source 'samples' needs a weight-samples sidecar supplied with the request",
            f"{at}.weight_source",
        )
        return issues
    if pinned is not None and digest(samples) != pinned:
        add(
            "WEIGHT_SAMPLES_DIGEST_MISMATCH",
            "the supplied sidecar's digest is not the weight_samples_sha256 this configuration "
            "names",
            f"{at}.weight_samples_sha256",
        )
    if not samples.usable:
        add(
            "WEIGHT_SAMPLES_NOT_USABLE",
            "the sidecar is diagnostic or rests on insufficient data, so it is not usable",
            f"{at}.weight_source",
        )
    if samples.leaf_ids != project.tree().leaves():
        add(
            "WEIGHT_SAMPLES_TREE_MISMATCH",
            "the sidecar's leaves are not this project's leaves in tree order",
            f"{at}.weight_source",
        )
    profiles = {p.id for p in project.weight_profiles}
    if samples.weight_profile_id not in profiles:
        add(
            "WEIGHT_SAMPLES_UNKNOWN_PROFILE",
            f"the sidecar is centred on profile {samples.weight_profile_id!r}, which this "
            "project does not have",
            f"{at}.weight_source",
        )
    else:
        run_profile = project.resolve_profile(configuration)
        if run_profile is not None and run_profile.id != samples.weight_profile_id:
            add(
                "WEIGHT_SAMPLES_PROFILE_MISMATCH",
                f"the run uses profile {run_profile.id!r} but the sidecar is centred on "
                f"{samples.weight_profile_id!r}",
                "configuration.weight_profile_id",
            )
    if samples.project_fingerprint != fitted_fingerprint(project):
        add(
            "WEIGHT_SAMPLES_STALE",
            "the project has changed since the samples were drawn (beyond choosing this weight "
            "source); draw them again",
            f"{at}.weight_source",
        )
    stored = len(samples.samples)
    if params.samples > stored and not params.resample:
        add(
            "WEIGHT_SAMPLES_TOO_FEW",
            f"{params.samples} draws need {params.samples} distinct stored vectors and the "
            f"sidecar has {stored}; lower the samples, or set resample to reuse them",
            f"{at}.samples",
        )
    return issues


def indices(seed: int, draws: int, stored: int, resample: bool) -> NDArray[np.intp]:
    """Which stored vector each draw uses, from ``PCG64(seed).jumped(2)``, integers only."""
    rng = np.random.Generator(np.random.PCG64(seed).jumped(INDEX_STREAM_JUMP))
    if draws <= stored:
        order = np.arange(stored, dtype=np.intp)
        for i in range(draws):  # a partial Fisher-Yates shuffle
            j = int(rng.integers(i, stored))
            order[i], order[j] = order[j], order[i]
        return order[:draws].copy()
    if not resample:
        raise ValueError("more draws than stored vectors needs resample")
    out: NDArray[np.intp] = rng.integers(0, stored, size=draws).astype(np.intp)
    return out
