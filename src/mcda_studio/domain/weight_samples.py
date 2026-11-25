"""The ``weight-samples`` sidecar, version 1: global weight vectors with their provenance.

``basis`` says what the vectors are: ``"calibrated"``, posterior draws fitted to a reference set's
outcomes (``mcda calibrate fit``, ``docs/design/calibration.md``), or ``"prior"``, exact draws from
the nested Dirichlet conviction prior with no outcome data (``mcda calibrate prior``,
``docs/design/sampled-weights-in-smaa.md``). The reference-set and outcome-model fields are
required only when the basis is calibrated, and absent otherwise. A file without ``basis`` was
written before the field existed and is calibrated.

A separate local JSON file, never stored inside a project. Strict sidecar conventions: unknown
fields are refused, ``version`` must be an explicit integer, and any version other than 1 is
refused naming the supported one. Only a converged run can be written: the model refuses
``converged: false``. A run under ``--include-proposed`` is marked ``diagnostic`` and is never
``usable``. Design: ``docs/design/calibration.md`` section 6.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from mcda_studio.domain.common import Id
from mcda_studio.domain.sidecar_common import SidecarModel

__all__ = [
    "DIAGNOSTIC_LABEL", "KIND", "LABEL", "LIMITATIONS", "PRIOR_LABEL", "PRIOR_LIMITATIONS",
    "SUM_TOLERANCE", "VERSION", "Basis", "DataSpec", "DiagnosticsSpec", "FixedGroupPrior",
    "GroupPrior", "PriorSpec", "QuantityDiagnosticsSpec", "SamplerSpec", "WeightSamples",
]  # fmt: skip

KIND = "weight-samples"
VERSION = 1
LABEL = (
    "Calibrated weights describe what fitted the reference set's outcomes under the stated model; "
    "they are not a forecast for any alternative."
)
DIAGNOSTIC_LABEL = (
    "Diagnostic run: proposed ratings were counted as if reviewed. These samples are for "
    "inspecting the method only and must not be used as weights."
)
LIMITATIONS = [
    "The outcome model is a constant hazard within the window on a linear score; a fit to a small "
    "reference set says nothing about other alternatives.",
    "Cases with no follow-up are left out, which assumes follow-up does not depend on the outcome "
    "given the ratings.",
    "Weights are identified only along criteria whose rated values vary across the entering "
    "cases; elsewhere the prior decides.",
    "The reference set's author knew the outcomes when rating the cases.",
]
PRIOR_LABEL = (
    "Prior weights describe the stated conviction about this profile's weights as a Dirichlet "
    "distribution around them; they use no outcome data and are not a forecast for any "
    "alternative."
)
PRIOR_LIMITATIONS = [
    "The concentration is a convention matched to the conviction bands (or stated), not an "
    "estimate from data.",
    "Groups are independent; children rated 0 and listed held children keep their stated weight.",
    "The draws are exact and independent (no Markov chain), so there is nothing to converge; "
    "they repeat for a seed on one platform and NumPy version.",
]
SUM_TOLERANCE = 1e-9
Basis = Literal["prior", "calibrated"]


class GroupPrior(SidecarModel):
    parent_id: Id | None
    children: list[Id]  # the free children, in tree order
    centre: list[float]
    kappa: float
    kappa_method: str
    share: float  # what the held children leave
    held: list[Id] = []


class FixedGroupPrior(SidecarModel):
    parent_id: Id | None
    children: list[Id]
    reason: str


class PriorSpec(SidecarModel):
    conviction: Literal["high", "medium", "low"] | None
    conviction_source: str
    half_width: float | None
    groups: list[GroupPrior]
    fixed_groups: list[FixedGroupPrior]
    # The outcome model's priors: present only when the basis is calibrated.
    base_rate: float | None = Field(default=None, exclude_if=lambda v: v is None)
    alpha_mean: float | None = Field(default=None, exclude_if=lambda v: v is None)
    alpha_sd: float | None = Field(default=None, exclude_if=lambda v: v is None)
    log_beta_mean: float | None = Field(default=None, exclude_if=lambda v: v is None)
    log_beta_sd: float | None = Field(default=None, exclude_if=lambda v: v is None)
    centre_global: dict[Id, float]


class SamplerSpec(SidecarModel):
    algorithm: str
    chains: int
    warmup: int
    draws: int  # kept iterations per chain
    samples: int  # stored vectors
    numpy_version: str
    python_version: str


class QuantityDiagnosticsSpec(SidecarModel):
    name: str
    rhat: float | None
    ess_bulk: float | None


class DiagnosticsSpec(SidecarModel):
    converged: bool
    rhat_max: float
    ess_min: float
    quantities: list[QuantityDiagnosticsSpec]
    acceptance: dict[str, list[float]]


class DataSpec(SidecarModel):
    outcome_definition: Id
    min_rated_fraction: str
    compared_criteria: list[Id]
    entering_cases: int
    excluded: dict[str, int]
    statuses: dict[str, int]
    rated_cases: dict[Id, int]
    insufficient: list[str]


class WeightSamples(SidecarModel):
    kind: Literal["weight-samples"]
    version: Literal[1]
    project_id: Id
    project_fingerprint: Annotated[str, Field(min_length=64, max_length=64)]
    weight_profile_id: Id
    basis: Basis = "calibrated"
    reference_set_id: Id | None = Field(default=None, exclude_if=lambda v: v is None)
    reference_set_sha256: Annotated[str | None, Field(pattern=r"^[0-9a-f]{64}$")] = Field(
        default=None, exclude_if=lambda v: v is None
    )
    reference_set_digest_basis: Literal["file bytes", "canonical JSON"] | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    diagnostic: bool
    usable: bool
    review_states: list[Literal["proposed", "reviewed"]]
    data: DataSpec | None = Field(default=None, exclude_if=lambda v: v is None)
    prior: PriorSpec
    seed: Annotated[int, Field(ge=0)]
    sampler: SamplerSpec
    diagnostics: DiagnosticsSpec
    leaf_ids: Annotated[list[Id], Field(min_length=1)]
    samples: Annotated[list[list[float]], Field(min_length=1)]
    alpha: list[float] | None = Field(default=None, exclude_if=lambda v: v is None)
    beta: list[float] | None = Field(default=None, exclude_if=lambda v: v is None)
    label: str
    limitations: list[str]

    @model_validator(mode="before")
    @classmethod
    def known_kind_and_version(cls, value: Any) -> Any:
        if isinstance(value, dict):
            if "kind" in value and value["kind"] != KIND:
                raise ValueError(f"this is not a weight-samples file (kind must be {KIND!r})")
            version = value.get("version")
            if type(version) is int and version != VERSION:
                raise ValueError(
                    f"weight-samples version {version} is not supported; this reader takes "
                    f"version {VERSION}"
                )
        return value

    @model_validator(mode="after")
    def consistent(self) -> WeightSamples:
        if not self.diagnostics.converged:
            raise ValueError("samples from a run that did not converge are never written")
        n = self.sampler.samples
        if self.basis == "prior":
            self._prior_basis()
        else:
            self._calibrated_basis(n)
        if len(self.samples) != n:
            raise ValueError(f"expected {n} samples of the weights")
        width = len(self.leaf_ids)
        for row in self.samples:
            if len(row) != width:
                raise ValueError("every sample has one weight per leaf")
            if any(not math.isfinite(x) or x < 0.0 for x in row):
                raise ValueError("weights are finite and non-negative")
            if abs(math.fsum(row) - 1.0) > SUM_TOLERANCE:
                raise ValueError("every sample's weights sum to 1")
        if any(not math.isfinite(b) or b <= 0.0 for b in self.beta or []):
            raise ValueError("beta is positive")
        return self

    def _calibrated_basis(self, n: int) -> None:
        missing = [name for name in _REFERENCE_FIELDS if getattr(self, name) is None] + [
            f"prior.{name}" for name in _OUTCOME_PRIORS if getattr(self.prior, name) is None
        ]
        if missing:
            raise ValueError("a calibrated basis needs " + ", ".join(missing))
        assert self.data is not None and self.alpha is not None and self.beta is not None
        expected_usable = not self.diagnostic and not self.data.insufficient
        if self.usable != expected_usable:
            raise ValueError(
                "usable must be true exactly when the run is neither diagnostic nor insufficient"
            )
        states = sorted(self.review_states)
        if states != (["proposed", "reviewed"] if self.diagnostic else ["reviewed"]):
            raise ValueError("review_states must be ['reviewed'], or both under a diagnostic run")
        if self.label != (LABEL if not self.diagnostic else f"{LABEL} {DIAGNOSTIC_LABEL}"):
            raise ValueError("the label must be the fixed calibration label")
        if not (len(self.alpha) == len(self.beta) == n):
            raise ValueError(f"expected {n} samples of the weights, alpha and beta")

    def _prior_basis(self) -> None:
        present = [name for name in _REFERENCE_FIELDS if getattr(self, name) is not None] + [
            f"prior.{name}" for name in _OUTCOME_PRIORS if getattr(self.prior, name) is not None
        ]
        if present:
            raise ValueError(
                "a prior basis uses no reference set or outcome model; remove " + ", ".join(present)
            )
        if self.diagnostic or not self.usable or self.review_states:
            raise ValueError("a prior basis is never diagnostic, is usable and counts no ratings")
        if self.label != PRIOR_LABEL:
            raise ValueError("the label must be the fixed prior label")


_REFERENCE_FIELDS = (
    "reference_set_id", "reference_set_sha256", "reference_set_digest_basis", "data", "alpha",
    "beta",
)  # fmt: skip
_OUTCOME_PRIORS = ("base_rate", "alpha_mean", "alpha_sd", "log_beta_mean", "log_beta_sd")
