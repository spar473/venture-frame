"""Trade-off elicitation: weights implied by pairwise choices between two alternatives.

Domain-neutral. An answer "A over B" is the linear constraint ``w . (v(A) - v(B)) >= 0`` on the
**global leaf weights** ``w``, with ``v`` the existing value functions; working in global leaf
space keeps constraints across sibling groups linear. The prior is a stated weight profile's
uncertainty, one block per varying sibling group:

- ``dirichlet``: the local weights of the free children are ``Dirichlet(alpha)`` (the conviction
  prior's ``kappa * centre``, or all ones for an unknown group);
- ``ordered``: uniform on the part of the simplex that keeps a stated order (Dirichlet(1, ..., 1)
  restricted to it), as the ``ranking`` kind of SMAA-2 weight uncertainty;
- ``intervals``: each free child's rating uniform on a stated interval, then normalised, as the
  ``rating_intervals`` kind.

Held children keep their stated local weight and the free ones share the rest; every other group
is fixed at the stated local weights; global weights are products of local weights along each
leaf's path (``methods/mavt/weighting.py``).

Two posteriors:

- **exact** (``smc``): the prior restricted to the region the answers allow. Sequential Monte
  Carlo over prior draws: after each answer, keep the particles that satisfy it, resample them
  systematically, then rejuvenate with random-walk Metropolis moves in each block's latent
  coordinates that target the prior density times the indicator of every constraint so far. Each
  move keeps the restricted prior invariant, so no constraint is ever left once satisfied.
- **noisy**: ``P(A over B) = sigmoid(beta * w . (v(A) - v(B)))`` through the calibration
  sampler's ``Problem.log_likelihood`` hook, with its convergence gate. That part lives in
  ``calibration/tradeoff.py`` (only the command line imports calibration); this module supplies
  its prior (``ordered_nested_prior``) and the ``NoisyFit`` it returns.

Latent coordinates: a Dirichlet or ordered block moves the additive log-ratio ``y`` with
``p = softmax(y, 0)``; its density includes the Jacobian (``group_log_density``). An intervals
block moves the ratings themselves; their density is constant on the box.

Design: ``docs/design/tradeoff-elicitation.md``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from mcda_studio.methods.nested_dirichlet import (
    DirichletGroup,
    NestedDirichletPrior,
    alr,
    alr_inverse,
    dirichlet_log_normaliser,
    log_simplex,
)

__all__ = [
    "ALGORITHM", "IMPLIED_HIGH", "IMPLIED_LOW", "LOG_BETA_PRIOR_MEAN", "LOG_BETA_PRIOR_SD",
    "MAX_QUESTIONS", "NOISE_Z", "STOP_WINDOW", "Array", "BlockKind", "ConstrainedPrior",
    "LatentGroup", "NoisyFit", "NoisyFitter", "OrderedNestedPrior", "SmcResult", "SmcSettings",
    "SmcState", "answer_probabilities", "first_probabilities", "ordered_nested_prior",
    "pair_infeasible", "select_question", "smc_fit", "should_stop",
]  # fmt: skip

Array = NDArray[np.float64]
BlockKind = Literal["dirichlet", "ordered", "intervals"]

ALGORITHM = "tradeoff.smc.v1"
#: A question whose answer the current draws already imply is never asked.
IMPLIED_LOW = 0.02
IMPLIED_HIGH = 0.98
#: Stop after this many questions at most.
MAX_QUESTIONS = 30
#: Stop when P(first) moved less than its 99% sampling noise over this many answers.
STOP_WINDOW = 3
NOISE_Z = 2.5758293035489004  # the standard normal's 0.995 quantile
#: Noisy mode: beta ~ LogNormal(log 20, 1). A full swing on one leaf against none on another is a
#: score difference of about one leaf's weight (0.05 to 0.1 on a 16-leaf tree), so beta = 20 puts
#: such a pair at sigmoid(1 to 2) = 0.73 to 0.88; the prior spans beta 2.8 to 140 (95%).
LOG_BETA_PRIOR_MEAN = math.log(20.0)
LOG_BETA_PRIOR_SD = 1.0
_TINY = 1e-300
_JITTER = 1e-9


# ---- the prior over global weights --------------------------------------------------------------


@dataclass(frozen=True)
class LatentGroup:
    """One varying sibling group. ``children`` are the free children; for an ``ordered`` block
    they are listed best first, and every draw keeps them in that order."""

    parent_id: str | None
    kind: BlockKind
    children: tuple[str, ...]
    share: float  # what the held children leave
    held: tuple[str, ...] = ()
    alpha: tuple[float, ...] = ()  # dirichlet and ordered (all ones)
    lower: tuple[float, ...] = ()  # intervals
    upper: tuple[float, ...] = ()

    @property
    def dim(self) -> int:
        return len(self.children) - 1 if self.kind != "intervals" else len(self.children)

    def check(self) -> None:
        k = len(self.children)
        if k < 2:
            raise ValueError("a varying group needs two or more free children")
        if not 0.0 < self.share <= 1.0 + 1e-12:
            raise ValueError("the free children's share must lie in (0, 1]")
        if self.kind in ("dirichlet", "ordered"):
            if len(self.alpha) != k or any(not a > 0.0 for a in self.alpha):
                raise ValueError("a Dirichlet block needs one positive concentration per child")
            if self.kind == "ordered" and any(a != 1.0 for a in self.alpha):
                raise ValueError("an ordered block is uniform: every concentration is 1")
        else:
            if len(self.lower) != k or len(self.upper) != k:
                raise ValueError("an intervals block needs one interval per child")
            if any(not 0.0 <= lo < hi for lo, hi in zip(self.lower, self.upper, strict=True)):
                raise ValueError("every interval needs 0 <= lower < upper")


class ConstrainedPrior:
    """The prior over global leaf weights, as latent blocks; groups top level first."""

    def __init__(
        self,
        node_ids: Sequence[str],
        leaf_ids: Sequence[str],
        paths: Mapping[str, Sequence[str]],
        base_local: Mapping[str, float],
        groups: Sequence[LatentGroup],
    ) -> None:
        for g in groups:
            g.check()
        self.node_ids = tuple(node_ids)
        self.leaf_ids = tuple(leaf_ids)
        self.paths = {leaf: tuple(paths[leaf]) for leaf in self.leaf_ids}
        self.base_local = {n: float(base_local[n]) for n in self.node_ids}
        self.groups = tuple(groups)
        index = {n: i for i, n in enumerate(self.node_ids)}
        one = len(self.node_ids)
        depth = max(len(p) for p in self.paths.values())
        self._path_index = np.full((len(self.leaf_ids), depth), one, dtype=np.intp)
        for row, leaf in enumerate(self.leaf_ids):
            for col, node in enumerate(self.paths[leaf]):
                self._path_index[row, col] = index[node]
        self._base = np.array([*self.base_local.values(), 1.0])
        self._group_index = [np.array([index[c] for c in g.children]) for g in self.groups]
        self._normalisers = [
            dirichlet_log_normaliser(list(g.alpha)) if g.kind != "intervals" else 0.0
            for g in self.groups
        ]
        varying = np.zeros(len(self.leaf_ids), dtype=bool)
        for g in self.groups:
            for row, leaf in enumerate(self.leaf_ids):
                if any(c in self.paths[leaf] for c in g.children):
                    varying[row] = True
        self.varying = varying

    @property
    def dims(self) -> list[int]:
        return [g.dim for g in self.groups]

    # The local weights of a group's free children from its latent block, (n, k).
    def _shares(self, g: LatentGroup, z: Array) -> Array:
        if g.kind == "intervals":
            total = z.sum(axis=1, keepdims=True)
            out: Array = z / total
            return out
        return alr_inverse(z)

    def global_weights(self, latents: Sequence[Array]) -> Array:
        """Global weights, ``(n, leaves)`` in leaf order."""
        n = latents[0].shape[0] if latents else 1
        local = np.repeat(self._base[:, None], n, axis=1)
        for g, idx, z in zip(self.groups, self._group_index, latents, strict=True):
            local[idx, :] = g.share * self._shares(g, z).T
        out = local[self._path_index[:, 0], :]
        for col in range(1, self._path_index.shape[1]):
            out = out * local[self._path_index[:, col], :]
        result: Array = out.T
        return result

    def block_log_density(self, b: int, z: Array) -> Array:
        """One block's log density in its latent coordinates, ``(n,)``; ``-inf`` off support."""
        g = self.groups[b]
        if g.kind == "intervals":
            lo = np.asarray(g.lower)
            hi = np.asarray(g.upper)
            inside = np.all((z >= lo) & (z <= hi), axis=1)
            out: Array = np.where(inside, 0.0, -np.inf)
            return out
        logp = log_simplex(z)
        density: Array = self._normalisers[b] + (np.asarray(g.alpha) * logp).sum(axis=1)
        if g.kind == "ordered":
            ordered = np.all(np.diff(logp, axis=1) <= 0.0, axis=1)
            density = np.where(ordered, density, -np.inf)
        return density

    def sample(self, rng: np.random.Generator, n: int) -> list[Array]:
        """``n`` exact prior draws, one latent array per block, each ``(n, dim)``.

        Blocks are drawn in order: ``Generator.dirichlet`` for a Dirichlet block, the same then
        sorted largest first for an ordered block (uniform on the ordered region), and
        ``Generator.random`` scaled to the intervals for an intervals block."""
        out: list[Array] = []
        for g in self.groups:
            if g.kind == "intervals":
                lo = np.asarray(g.lower)
                hi = np.asarray(g.upper)
                out.append(lo + rng.random((n, len(g.children))) * (hi - lo))
                continue
            p = rng.dirichlet(np.asarray(g.alpha), size=n)
            if g.kind == "ordered":
                p = -np.sort(-p, axis=1)
            p = np.clip(p, _TINY, None)
            out.append(alr(p / p.sum(axis=1, keepdims=True)))
        return out


# ---- the noisy mode: the calibration sampler with a Bradley-Terry likelihood ---------------------


class OrderedNestedPrior(NestedDirichletPrior):
    """A nested Dirichlet whose listed groups are uniform on their ordered region.

    For the calibration sampler: a group named in ``ordered`` must have all concentrations 1
    (``kappa * centre`` = ones) and lists its children best first; its density is ``-inf`` where
    the order is broken, and chains start inside the region."""

    def __init__(self, *args: object, ordered: Sequence[int] = (), **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.ordered = tuple(ordered)
        for b in self.ordered:
            g = self.groups[b]
            if any(abs(g.kappa * c - 1.0) > 1e-12 for c in g.centre):
                raise ValueError("an ordered group is uniform: kappa * centre must be all ones")

    def log_density(self, ys: Sequence[Array], chains: int | None = None) -> Array:
        total = super().log_density(ys, chains)
        for b in self.ordered:
            logp = log_simplex(ys[b])
            ordered = np.all(np.diff(logp, axis=1) <= 0.0, axis=1)
            total = np.where(ordered, total, -np.inf)
        return total

    def initial_ys(self, rng: np.random.Generator) -> list[Array]:
        start = super().initial_ys(rng)
        for b in self.ordered:
            p = alr_inverse(start[b][None, :])[0]
            start[b] = alr(-np.sort(-p))
        return start


@dataclass(frozen=True)
class NoisyFit:
    """What noisy mode returns: thinned draws, beta, and the sampler's convergence record."""

    draws: Array  # (samples, leaves)
    beta: Array  # (samples,)
    converged: bool
    quantities: list[tuple[str, float | None, float | None]]  # (name, R-hat, bulk ESS)
    acceptance: dict[str, list[float]]
    algorithm: str
    chains: int
    warmup: int
    draws_per_chain: int
    samples: int


#: Noisy mode lives with the calibration sampler (``calibration/tradeoff.py``); only the command
#: line, which may import calibration, passes it in. The application never fits it.
NoisyFitter = Callable[[NestedDirichletPrior, Array, int], NoisyFit]


def ordered_nested_prior(prior: ConstrainedPrior) -> OrderedNestedPrior:
    """The noisy mode's prior from a constrained prior with Dirichlet and ordered blocks only."""
    groups: list[DirichletGroup] = []
    ordered: list[int] = []
    for b, g in enumerate(prior.groups):
        if g.kind == "intervals":
            raise ValueError(
                "noisy mode needs Dirichlet or ordered groups; rating intervals have no density "
                "the sampler can use"
            )
        kappa = math.fsum(g.alpha)
        groups.append(
            DirichletGroup(
                parent_id=g.parent_id,
                children=g.children,
                centre=tuple(a / kappa for a in g.alpha),
                kappa=kappa,
                share=g.share,
                held=g.held,
                kappa_method="ordered" if g.kind == "ordered" else "stated",
            )
        )
        if g.kind == "ordered":
            ordered.append(b)
    return OrderedNestedPrior(
        prior.node_ids, prior.leaf_ids, prior.paths, prior.base_local, groups, [],
        ordered=ordered,
    )  # fmt: skip


# ---- the exact mode: sequential Monte Carlo ----------------------------------------------------


@dataclass(frozen=True)
class SmcSettings:
    particles: int = 4000
    moves: int = 20  # Metropolis sweeps over every block after each answer
    seed: int = 0

    def check(self) -> None:
        if self.particles < 100:
            raise ValueError("at least 100 particles")
        if self.moves < 1:
            raise ValueError("at least one move per answer")
        if self.seed < 0:
            raise ValueError("the seed must be a non-negative integer")


@dataclass
class SmcState:
    """Particles: latent blocks, their global weights ``(n, leaves)`` and the kept constraints."""

    prior: ConstrainedPrior
    latents: list[Array]
    weights: Array
    log_density: list[Array]
    constraints: Array  # (kept answers, leaves)
    rng: np.random.Generator
    acceptance: list[float] = field(default_factory=list)

    @staticmethod
    def start(prior: ConstrainedPrior, settings: SmcSettings) -> SmcState:
        settings.check()
        rng = np.random.Generator(np.random.PCG64(settings.seed))
        latents = prior.sample(rng, settings.particles)
        return SmcState(
            prior=prior,
            latents=latents,
            weights=prior.global_weights(latents),
            log_density=[prior.block_log_density(b, z) for b, z in enumerate(latents)],
            constraints=np.zeros((0, len(prior.leaf_ids))),
            rng=rng,
        )

    @property
    def n(self) -> int:
        return int(self.weights.shape[0])

    def feasible(self, weights: Array) -> NDArray[np.bool_]:
        if self.constraints.shape[0] == 0:
            return np.ones(weights.shape[0], dtype=bool)
        out: NDArray[np.bool_] = np.all(weights @ self.constraints.T >= 0.0, axis=1)
        return out

    def resample(self, keep: NDArray[np.intp]) -> None:
        """Systematic resampling of the kept particles (equal weights) back to ``n``."""
        n = self.n
        u = (self.rng.random() + np.arange(n)) / n
        picks = keep[np.minimum((u * keep.size).astype(np.intp), keep.size - 1)]
        self.latents = [z[picks].copy() for z in self.latents]
        self.weights = self.weights[picks].copy()
        self.log_density = [d[picks].copy() for d in self.log_density]

    def move(self, sweeps: int) -> None:
        """Random-walk Metropolis per block; target prior density x every constraint so far.

        The proposal covariance of a block is the particle cloud's at the start of the move
        (scaled by ``2.38^2 / dim``) and stays fixed during it, so each step is a valid
        Metropolis kernel for the restricted prior."""
        prior = self.prior
        chols = []
        for z in self.latents:
            d = z.shape[1]
            cov = np.atleast_2d(np.cov(z, rowvar=False)) + _JITTER * np.eye(d)
            try:
                chol = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                chol = np.diag(np.sqrt(np.maximum(np.diag(cov), _JITTER)))
            chols.append(chol * (2.38 / math.sqrt(d)))
        accepted = 0
        proposed_total = 0
        for _ in range(sweeps):
            for b, z in enumerate(self.latents):
                step = self.rng.standard_normal(z.shape) @ chols[b].T
                trial = z + step
                trial_density = prior.block_log_density(b, trial)
                trial_latents = list(self.latents)
                trial_latents[b] = trial
                trial_weights = prior.global_weights(trial_latents)
                log_u = np.log(self.rng.random(self.n))
                with np.errstate(invalid="ignore"):
                    accept = log_u < trial_density - self.log_density[b]
                accept &= np.isfinite(trial_density) & self.feasible(trial_weights)
                self.latents[b] = np.where(accept[:, None], trial, z)
                self.log_density[b] = np.where(accept, trial_density, self.log_density[b])
                self.weights = np.where(accept[:, None], trial_weights, self.weights)
                accepted += int(accept.sum())
                proposed_total += self.n
        self.acceptance.append(accepted / max(1, proposed_total))

    def add(self, contrast: Array, sweeps: int) -> int:
        """Filter by one answer, resample and move. Returns the survivors; on 0 nothing changes
        and the answer is not kept."""
        g = self.weights @ contrast
        keep = np.flatnonzero(g >= 0.0)
        if keep.size == 0:
            return 0
        self.constraints = np.vstack([self.constraints, contrast[None, :]])
        self.resample(keep)
        self.move(sweeps)
        return int(keep.size)


@dataclass
class SmcResult:
    weights: Array  # (n, leaves)
    kept: list[int]  # indexes of the answers that were kept
    conflicts: list[int]  # indexes of the answers that left no particle
    survivors: list[int]  # per answer, before resampling (0 for a conflict)
    acceptance: list[float]


def smc_fit(prior: ConstrainedPrior, contrasts: Array, settings: SmcSettings) -> SmcResult:
    """The exact posterior given every answer in turn; an answer that leaves no particle is
    reported as a conflict and not kept (the caller falls back to the noisy mode)."""
    state = SmcState.start(prior, settings)
    kept: list[int] = []
    conflicts: list[int] = []
    survivors: list[int] = []
    for k, row in enumerate(np.atleast_2d(np.asarray(contrasts, dtype=np.float64))):
        if row.size == 0:
            continue
        alive = state.add(row, settings.moves)
        survivors.append(alive)
        (kept if alive else conflicts).append(k)
    return SmcResult(state.weights, kept, conflicts, survivors, state.acceptance)


# ---- questions -----------------------------------------------------------------------------------


def answer_probabilities(weights: Array, contrasts: Array) -> Array:
    """``P(first over second)`` for every candidate under the draws, ``(candidates,)``."""
    if contrasts.shape[0] == 0:
        return np.zeros(0)
    out: Array = (weights @ contrasts.T > 0.0).mean(axis=0)
    return out


def select_question(
    probabilities: Array,
    asked: set[int],
    *,
    mode: Literal["adaptive", "random"] = "adaptive",
    rng: np.random.Generator | None = None,
) -> int | None:
    """The candidate to ask next, or ``None`` when every candidate is asked or implied.

    Adaptive: P closest to 0.5 (the first such candidate on a tie). Random: uniform over the
    candidates that are neither asked nor implied (P below 0.02 or above 0.98)."""
    eligible = [
        c
        for c in range(probabilities.size)
        if c not in asked and IMPLIED_LOW <= probabilities[c] <= IMPLIED_HIGH
    ]
    if not eligible:
        return None
    if mode == "random":
        if rng is None:
            raise ValueError("random questions need a generator")
        return eligible[int(rng.integers(len(eligible)))]
    gap = np.abs(probabilities[eligible] - 0.5)
    return eligible[int(np.argmin(gap))]


def first_probabilities(weights: Array, partial: Array) -> Array:
    """P(first) per alternative under the draws; ``partial`` is ``(alternatives, leaves)``.

    A draw's leader is the alternative with the highest value, the first in order on an exact
    tie (ties have probability zero under a continuous prior)."""
    values = weights @ partial.T  # (n, alternatives)
    leaders = np.argmax(values, axis=1)
    counts = np.bincount(leaders, minlength=partial.shape[0])
    out: Array = counts / weights.shape[0]
    return out


def should_stop(history: Sequence[Array], particles: int, asked: int) -> bool:
    """Stop at ``MAX_QUESTIONS``, or when P(first) of every alternative moved less than its 99%
    sampling noise over the last ``STOP_WINDOW`` answers.

    The noise of a difference of two estimates from ``particles`` draws each is
    ``z * sqrt(2 p (1 - p) / particles)`` with ``p`` their mean. Particles after rejuvenation are
    correlated, so the real noise is larger: this threshold errs toward asking more."""
    if asked >= MAX_QUESTIONS:
        return True
    if len(history) <= STOP_WINDOW:
        return False
    now, then = history[-1], history[-1 - STOP_WINDOW]
    p = (now + then) / 2.0
    noise = NOISE_Z * np.sqrt(2.0 * p * (1.0 - p) / particles)
    return bool(np.all(np.abs(now - then) <= noise))


# ---- conflicts -----------------------------------------------------------------------------------


def pair_infeasible(a: Array, b: Array, support: NDArray[np.bool_] | None = None) -> bool:
    """Whether ``w . a >= 0`` and ``w . b >= 0`` have no common point on the simplex.

    Exact, by vertex enumeration: the set ``{w in simplex: w . a >= 0}`` is a polytope whose
    vertices are the simplex vertices ``e_j`` with ``a_j >= 0`` and the points on the edges
    ``e_i e_j`` where ``w . a = 0``; ``w . b`` is linear, so the pair is feasible exactly when its
    maximum over those vertices is ``>= 0``. ``support`` restricts the simplex to some leaves (the
    ones the prior can move); it is a relaxation of a prior whose support is smaller still."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if support is not None:
        a, b = a[support], b[support]
    best = -math.inf
    for j in range(a.size):
        if a[j] >= 0.0:
            best = max(best, float(b[j]))
    pos = np.flatnonzero(a > 0.0)
    neg = np.flatnonzero(a < 0.0)
    for i in pos.tolist():
        for k in neg.tolist():
            denom = a[i] - a[k]
            value = (-a[k] * b[i] + a[i] * b[k]) / denom
            best = max(best, float(value))
    return best < 0.0
