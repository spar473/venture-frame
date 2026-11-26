"""The ``weight-samples`` sidecar, version 1: posterior global weight vectors with their provenance.

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
SUM_TOLERANCE = 1e-9


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
    base_rate: float
    alpha_mean: float
    alpha_sd: float
    log_beta_mean: float = 0.0
    log_beta_sd: float
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
    reference_set_id: Id
    reference_set_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    reference_set_digest_basis: Literal["file bytes", "canonical JSON"]
    diagnostic: bool
    usable: bool
    review_states: list[Literal["proposed", "reviewed"]]
    data: DataSpec
    prior: PriorSpec
    seed: Annotated[int, Field(ge=0)]
    sampler: SamplerSpec
    diagnostics: DiagnosticsSpec
    leaf_ids: Annotated[list[Id], Field(min_length=1)]
    samples: Annotated[list[list[float]], Field(min_length=1)]
    alpha: list[float]
    beta: list[float]
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
        n = self.sampler.samples
        if not (len(self.samples) == len(self.alpha) == len(self.beta) == n):
            raise ValueError(f"expected {n} samples of the weights, alpha and beta")
        width = len(self.leaf_ids)
        for row in self.samples:
            if len(row) != width:
                raise ValueError("every sample has one weight per leaf")
            if any(not math.isfinite(x) or x < 0.0 for x in row):
                raise ValueError("weights are finite and non-negative")
            if abs(math.fsum(row) - 1.0) > SUM_TOLERANCE:
                raise ValueError("every sample's weights sum to 1")
        if any(not math.isfinite(b) or b <= 0.0 for b in self.beta):
            raise ValueError("beta is positive")
        return self
