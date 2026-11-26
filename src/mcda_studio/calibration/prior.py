"""Layer 1: a nested Dirichlet prior on the weights, centred on a profile's stated local weights.

Every sibling group with two or more *free* children gets ``p ~ Dirichlet(kappa * centre)`` on its
free children; held children keep their stated local weight and the free ones share what is left;
every other group is fixed at the stated local weights. Groups are independent. Global weights
are products of local weights along each leaf's path, as in ``methods/mavt/weighting.py``.

The sampler works on an unconstrained vector per varying group: the additive log-ratio (ALR)
``y in R^(k-1)`` with ``p = softmax(y, 0)``. The density of ``y`` includes the Jacobian
``prod_i p_i``.

Design: ``docs/design/calibration.md`` section 3.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mcda_studio.methods import portable

Array = NDArray[np.float64]

#: Tensor Gauss-Legendre quadrature for the moment-matched concentration: at most this many
#: points in all, and at most ``QUADRATURE_NODES_MAX`` nodes per free child.
QUADRATURE_POINTS = 1_000_000
QUADRATURE_NODES_MAX = 24
#: Above this many free children (fewer than three nodes per axis) the delta method is used.
QUADRATURE_CHILDREN_MAX = 12


# ---- the additive log-ratio transform ---------------------------------------------------------


def _with_reference(y: Array) -> Array:
    return np.concatenate([y, np.zeros((*y.shape[:-1], 1))], axis=-1)


def log_simplex(y: Array) -> Array:
    """``log p`` for ``p = softmax(y, 0)``, computed stably; shape ``(..., k)``."""
    z = _with_reference(np.asarray(y, dtype=np.float64))
    top = z.max(axis=-1, keepdims=True)
    return z - (top + np.log(np.exp(z - top).sum(axis=-1, keepdims=True)))


def alr_inverse(y: Array) -> Array:
    """The simplex point ``p = softmax(y, 0)``; ``y`` has shape ``(..., k - 1)``."""
    return np.exp(log_simplex(y))


def alr(p: Array) -> Array:
    """The additive log-ratio ``log(p_i / p_k)`` for ``i < k``."""
    q = np.asarray(p, dtype=np.float64)
    return np.log(q[..., :-1]) - np.log(q[..., -1:])


def log_jacobian(y: Array) -> Array:
    """``log |det d(p_1..p_{k-1}) / dy|`` = ``sum_i log p_i`` over all ``k`` components."""
    jac: Array = log_simplex(y).sum(axis=-1)
    return jac


def dirichlet_log_normaliser(concentration: Sequence[float]) -> float:
    """``log Gamma(sum a) - sum log Gamma(a_i)`` with the portable log-gamma."""
    total = math.fsum(concentration)
    return portable.lgamma(total) - math.fsum(portable.lgamma(a) for a in concentration)


def dirichlet_log_density(p: Array, concentration: Sequence[float]) -> Array:
    """The Dirichlet log density on the simplex (no Jacobian)."""
    a = np.asarray(concentration, dtype=np.float64)
    density: Array = dirichlet_log_normaliser(list(a)) + ((a - 1.0) * np.log(p)).sum(axis=-1)
    return density


def group_log_density(y: Array, centre: Sequence[float], kappa: float) -> Array:
    """The density of the ALR coordinates of ``Dirichlet(kappa * centre)``, Jacobian included."""
    a = kappa * np.asarray(centre, dtype=np.float64)
    density: Array = dirichlet_log_normaliser(list(a)) + (a * log_simplex(y)).sum(axis=-1)
    return density


# ---- the concentration from a conviction ------------------------------------------------------


@dataclass(frozen=True)
class Concentration:
    kappa: int
    method: str


def moment_matched_kappa(ratings: Sequence[float], half_width: float) -> Concentration:
    """The Dirichlet concentration that matches the conviction proposal's local-weight spread.

    Each rating ``r_i`` is drawn uniformly from ``[r_i (1 - h), r_i (1 + h)]`` and normalised;
    ``kappa = round(mean_i [w_i (1 - w_i) / Var(q_i)] - 1)`` with ``w_i`` the stated local weights.
    """
    r = np.asarray(ratings, dtype=np.float64)
    k = r.size
    if k < 2 or np.any(r <= 0.0) or not 0.0 < half_width < 1.0:
        raise ValueError("needs two or more positive ratings and a half-width in (0, 1)")
    w = r / r.sum()
    if k > QUADRATURE_CHILDREN_MAX:
        v = (r * half_width) ** 2 / 3.0
        var = ((1.0 - w) ** 2 * v + w**2 * (v.sum() - v)) / r.sum() ** 2
        method = "first-order delta method"
    else:
        nodes = min(QUADRATURE_NODES_MAX, int(math.floor(QUADRATURE_POINTS ** (1.0 / k) + 1e-9)))
        x, weights = np.polynomial.legendre.leggauss(nodes)
        axes = [
            (r[i] * (1.0 - half_width) + r[i] * half_width * (x + 1.0)).reshape(
                [-1 if j == i else 1 for j in range(k)]
            )
            for i in range(k)
        ]
        mass = np.ones([1] * k)
        for i in range(k):
            mass = mass * (weights / 2.0).reshape([-1 if j == i else 1 for j in range(k)])
        total = sum(axes, np.zeros([1] * k))
        var = np.empty(k)
        for i in range(k):
            q = axes[i] / total
            mean = float((mass * q).sum())
            var[i] = float((mass * q * q).sum()) - mean * mean
        method = f"Gauss-Legendre quadrature, {nodes} nodes per child"
    value = float(np.mean(w * (1.0 - w) / var)) - 1.0
    return Concentration(kappa=max(1, int(math.floor(value + 0.5))), method=method)


# ---- the prior ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class DirichletGroup:
    """A varying sibling group: free children share ``share`` as ``Dirichlet(kappa * centre)``."""

    parent_id: str | None
    children: tuple[str, ...]
    centre: tuple[float, ...]
    kappa: float
    share: float
    held: tuple[str, ...]
    kappa_method: str

    @property
    def dim(self) -> int:
        return len(self.children) - 1


@dataclass(frozen=True)
class FixedGroup:
    parent_id: str | None
    children: tuple[str, ...]
    reason: str


class NestedDirichletPrior:
    """The prior over global weights; groups are visited top level first, then in tree order."""

    def __init__(
        self,
        node_ids: Sequence[str],
        leaf_ids: Sequence[str],
        paths: Mapping[str, Sequence[str]],
        base_local: Mapping[str, float],
        groups: Sequence[DirichletGroup],
        fixed: Sequence[FixedGroup],
    ) -> None:
        self.node_ids = tuple(node_ids)
        self.leaf_ids = tuple(leaf_ids)
        self.base_local = {n: float(base_local[n]) for n in self.node_ids}
        self.groups = tuple(groups)
        self.fixed = tuple(fixed)
        self.paths = {leaf: tuple(paths[leaf]) for leaf in self.leaf_ids}
        index = {n: i for i, n in enumerate(self.node_ids)}
        one = len(self.node_ids)  # an extra slot that always holds 1.0
        depth = max(len(p) for p in self.paths.values())
        self._path_index = np.full((len(self.leaf_ids), depth), one, dtype=np.intp)
        for row, leaf in enumerate(self.leaf_ids):
            for col, node in enumerate(self.paths[leaf]):
                self._path_index[row, col] = index[node]
        self._base = np.array([*self.base_local.values(), 1.0])
        self._group_index = [np.array([index[c] for c in g.children]) for g in self.groups]
        self._normalisers = [
            dirichlet_log_normaliser([g.kappa * c for c in g.centre]) for g in self.groups
        ]
        _mean, sd = self.moments()
        self.varying_leaves = tuple(leaf for leaf in self.leaf_ids if sd[leaf] > 0.0)

    @property
    def dims(self) -> list[int]:
        return [g.dim for g in self.groups]

    def local(self, ys: Sequence[Array], chains: int | None = None) -> Array:
        """Local weights of every node (plus a trailing 1.0), shape ``(nodes + 1, chains)``."""
        chains = chains or (ys[0].shape[0] if ys else 1)
        out = np.repeat(self._base[:, None], chains, axis=1)
        for g, idx, y in zip(self.groups, self._group_index, ys, strict=True):
            out[idx, :] = g.share * alr_inverse(y).T
        return out

    def global_weights(self, ys: Sequence[Array], chains: int | None = None) -> Array:
        """Global weights of every leaf in leaf order, shape ``(leaves, chains)``."""
        local = self.local(ys, chains)
        out = local[self._path_index[:, 0], :]
        for col in range(1, self._path_index.shape[1]):
            out = out * local[self._path_index[:, col], :]
        return out

    def log_density(self, ys: Sequence[Array], chains: int | None = None) -> Array:
        """Sum of the groups' ALR densities, shape ``(chains,)``."""
        chains = chains or (ys[0].shape[0] if ys else 1)
        total = np.zeros(chains)
        for g, norm, y in zip(self.groups, self._normalisers, ys, strict=True):
            a = g.kappa * np.asarray(g.centre)
            total = total + norm + (a * log_simplex(y)).sum(axis=-1)
        return total

    def centre_global(self) -> dict[str, float]:
        weights = self.global_weights([alr(np.array([g.centre])) for g in self.groups])[:, 0]
        return dict(zip(self.leaf_ids, (float(w) for w in weights), strict=True))

    def moments(self) -> tuple[dict[str, float], dict[str, float]]:
        """The exact prior mean and standard deviation of every global weight."""
        first = {n: v for n, v in self.base_local.items()}
        second = {n: v * v for n, v in self.base_local.items()}
        for g in self.groups:
            for child, c in zip(g.children, g.centre, strict=True):
                first[child] = g.share * c
                second[child] = g.share**2 * c * (g.kappa * c + 1.0) / (g.kappa + 1.0)
        mean: dict[str, float] = {}
        sd: dict[str, float] = {}
        for leaf in self.leaf_ids:
            m = math.prod(first[n] for n in self.paths[leaf])
            s = math.prod(second[n] for n in self.paths[leaf])
            mean[leaf] = m
            sd[leaf] = math.sqrt(max(0.0, s - m * m))
        return mean, sd

    def sample_ys(self, rng: np.random.Generator, n: int) -> list[Array]:
        """Exact prior draws in ALR coordinates, one list entry per group, each ``(n, dim)``."""
        out = []
        for g in self.groups:
            p = rng.dirichlet(g.kappa * np.asarray(g.centre), size=n)
            p = np.clip(p, 1e-300, None)
            out.append(alr(p / p.sum(axis=1, keepdims=True)))
        return out

    def initial_ys(self, rng: np.random.Generator) -> list[Array]:
        """A dispersed starting point for one chain: ``N(alr(centre), 1)`` per coordinate."""
        return [alr(np.asarray(g.centre)) + rng.standard_normal(g.dim) for g in self.groups]


def make_prior(
    node_ids: Sequence[str],
    children: Mapping[str | None, Sequence[str]],
    local: Mapping[str, float],
    kappa: Mapping[str | None, float],
    held: Mapping[str | None, Sequence[str]] | None = None,
    kappa_method: Mapping[str | None, str] | None = None,
) -> NestedDirichletPrior:
    """Build the prior from a tree (``children`` by parent, ``None`` the top level).

    ``node_ids`` lists every node depth-first in sibling order (``CriterionTree.all_ids``).
    ``kappa`` must name every group that varies. A group is fixed when its parent's path share is
    0 or when fewer than two children are free; a child is held when its local weight is 0 or it is
    listed in ``held``.
    """
    held = held or {}
    kappa_method = kappa_method or {}
    parent_of: dict[str, str | None] = {}
    for parent, kids in children.items():
        for kid in kids:
            parent_of[kid] = parent
    paths: dict[str, list[str]] = {}
    for node in node_ids:
        path = [node]
        while parent_of[path[-1]] is not None:
            path.append(parent_of[path[-1]])  # type: ignore[arg-type]
        paths[node] = list(reversed(path))
    leaf_ids = [n for n in node_ids if not children.get(n)]
    groups: list[DirichletGroup] = []
    fixed: list[FixedGroup] = []
    parents: list[str | None] = [None, *[n for n in node_ids if children.get(n)]]
    for parent in parents:
        kids = tuple(children.get(parent, ()))
        if not kids:
            continue
        share_of_parent = 1.0 if parent is None else math.prod(local[n] for n in paths[parent])
        if share_of_parent == 0.0:
            fixed.append(FixedGroup(parent, kids, "zero-weight branch"))
            continue
        listed = set(held.get(parent, ()))
        held_kids = tuple(k for k in kids if local[k] == 0.0 or k in listed)
        free = tuple(k for k in kids if k not in held_kids)
        if len(free) < 2:
            fixed.append(FixedGroup(parent, kids, "fewer than two free children"))
            continue
        share = math.fsum(local[k] for k in free)
        if parent not in kappa:
            raise ValueError(f"no concentration for the group under {parent!r}")
        groups.append(
            DirichletGroup(
                parent_id=parent,
                children=free,
                centre=tuple(local[k] / share for k in free),
                kappa=float(kappa[parent]),
                share=share,
                held=held_kids,
                kappa_method=kappa_method.get(parent, "stated"),
            )
        )
    return NestedDirichletPrior(node_ids, leaf_ids, paths, local, groups, fixed)
