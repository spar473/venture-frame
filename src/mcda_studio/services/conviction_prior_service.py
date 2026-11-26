"""The conviction prior: exact draws of a profile's weights from a nested Dirichlet around them.

DD3 turned a conviction (high, medium, low) into rating intervals of +/-15/30/50%, which behave
like a Dirichlet concentration of about 555 / 136 / 47 on case 2's top level. This service draws
the Dirichlet itself, ``Dirichlet(kappa * w0)`` per sibling group, and writes the draws as a
``weight-samples`` sidecar with ``basis: "prior"`` that SMAA-2 can read as its weight source
(``docs/design/sampled-weights-in-smaa.md``). It uses no outcome data and changes no project.
"""

from __future__ import annotations

import hashlib
import json
import platform
from typing import Literal

import numpy as np

from mcda_studio.domain import weight_samples as ws
from mcda_studio.domain.errors import McdaError, UnknownReferenceError
from mcda_studio.domain.preferences import WeightProfile
from mcda_studio.domain.project import Project
from mcda_studio.methods.mavt.weighting import compute_weights
from mcda_studio.methods.nested_dirichlet import NestedDirichletPrior, prior_for_profile
from mcda_studio.services.elicitation_service import CONVICTION_HALF_WIDTH

ALGORITHM = "conviction.dirichlet.v1"
DEFAULT_DRAWS = 10_000  # SMAA-2's default sample count, so S <= N by default
MAX_DRAWS = 100_000  # SMAA-2's largest sample count

Conviction = Literal["high", "medium", "low"]


def sidecar_digest(samples: ws.WeightSamples) -> str:
    """SHA-256 of the sidecar's canonical JSON: sorted keys, UTF-8, ``(',', ':')`` separators.

    A file and the same sidecar sent in a request have the same digest whatever their layout."""
    text = json.dumps(
        samples.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _profile(project: Project, profile_id: str | None) -> WeightProfile:
    if profile_id is None:
        profile = project.resolve_profile(project.analysis)
        if profile is None:
            raise McdaError("the project has no weight profile")
        return profile
    for candidate in project.weight_profiles:
        if candidate.id == profile_id:
            return candidate
    raise UnknownReferenceError(f"unknown weight profile id {profile_id!r}")


def build_prior(
    project: Project,
    profile: WeightProfile,
    conviction: Conviction | None = None,
    kappa: float | None = None,
) -> tuple[NestedDirichletPrior, Conviction | None, str, float | None]:
    """The prior, the conviction used, where it came from, and its half-width."""
    if conviction is not None and kappa is not None:
        raise McdaError("give a conviction or a concentration, not both")
    held: dict[str | None, list[str]] = {}
    if project.smaa is not None:
        for group in project.smaa.groups:
            held[group.parent_id] = list(group.held)
    tree = project.tree()
    local = compute_weights(tree, profile.judgements).local
    if kappa is not None:
        prior = prior_for_profile(tree, local, profile.judgements, held, kappa=kappa)
        return prior, None, "concentration stated directly", None
    source = "stated with the request"
    if conviction is None and profile.elicitation is not None:
        conviction = profile.elicitation.conviction
        source = "the profile's elicitation record"
    if conviction is None:
        raise McdaError(
            "no conviction: the profile has no elicitation record with a conviction; state a "
            "conviction (high, medium or low) or a concentration"
        )
    half_width = CONVICTION_HALF_WIDTH[conviction]
    prior = prior_for_profile(tree, local, profile.judgements, held, half_width=half_width)
    return prior, conviction, source, half_width


def conviction_prior(
    project: Project,
    profile_id: str | None,
    *,
    conviction: Conviction | None = None,
    kappa: float | None = None,
    draws: int = DEFAULT_DRAWS,
    seed: int = 0,
) -> ws.WeightSamples:
    """``draws`` exact draws of the profile's global weights as a ``weight-samples`` sidecar."""
    if not 1 <= draws <= MAX_DRAWS:
        raise McdaError(f"draws must be between 1 and {MAX_DRAWS}")
    if seed < 0:
        raise McdaError("the seed must be a non-negative integer")
    profile = _profile(project, profile_id)
    prior, used, source, half_width = build_prior(project, profile, conviction, kappa)
    if not prior.groups:
        raise McdaError("no sibling group of this profile has two or more weights to vary")
    weights = prior.sample_global(np.random.Generator(np.random.PCG64(seed)), draws)
    return ws.WeightSamples(
        kind="weight-samples",
        version=1,
        project_id=project.id,
        project_fingerprint=project.fingerprint(),
        weight_profile_id=profile.id,
        basis="prior",
        diagnostic=False,
        usable=True,
        review_states=[],
        prior=ws.PriorSpec(
            conviction=used,
            conviction_source=source,
            half_width=half_width,
            groups=[
                ws.GroupPrior(
                    parent_id=g.parent_id,
                    children=list(g.children),
                    centre=list(g.centre),
                    kappa=g.kappa,
                    kappa_method=g.kappa_method,
                    share=g.share,
                    held=list(g.held),
                )
                for g in prior.groups
            ],
            fixed_groups=[
                ws.FixedGroupPrior(
                    parent_id=f.parent_id, children=list(f.children), reason=f.reason
                )
                for f in prior.fixed
            ],
            centre_global=prior.centre_global(),
        ),
        seed=seed,
        sampler=ws.SamplerSpec(
            algorithm=ALGORITHM,
            chains=1,
            warmup=0,
            draws=draws,
            samples=draws,
            numpy_version=np.__version__,
            python_version=platform.python_version(),
        ),
        diagnostics=ws.DiagnosticsSpec(
            converged=True, rhat_max=1.01, ess_min=400.0, quantities=[], acceptance={}
        ),
        leaf_ids=list(prior.leaf_ids),
        samples=[[float(x) for x in weights[:, k]] for k in range(draws)],
        label=ws.PRIOR_LABEL,
        limitations=list(ws.PRIOR_LIMITATIONS),
    )
