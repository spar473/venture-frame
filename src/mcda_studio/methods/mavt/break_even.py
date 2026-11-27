"""The nearest weight vector (L1) at which one linear score difference reaches zero.

For global weights ``w0`` on the simplex and a contrast ``d`` (for example ``v(b) - v(leader)``,
or ``v(a) - boundary``), find the smallest ``||w - w0||_1`` over ``w`` on the simplex (restricted
to ``eligible`` leaves) with ``w . d >= 0``. Moving mass ``m`` from leaf ``i`` to leaf ``j`` costs
``2m`` in L1 and gains ``m (d_j - d_i)``, so the optimum puts every unit taken on the leaf with the
largest ``d`` and takes from the leaves with the smallest ``d`` first (a fractional knapsack whose
greedy order is optimal: any feasible move that takes from a leaf with larger ``d`` first, or
gives to any other leaf, gains less per unit of L1). Exact, no iteration; the tests cross-check it
against a brute-force grid and an LP solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Shift:
    distance: float  # L1 distance from w0
    weights: Array  # the nearest weight vector
    to_leaf: int  # the leaf that gains
    from_leaves: list[int]  # the leaves that lose, in the order mass was taken


def nearest_shift(w0: Array, d: Array, eligible: NDArray[np.bool_]) -> Shift | None:
    """The nearest ``w`` with ``w . d >= 0`` moving only ``eligible`` leaves; ``None`` if none.

    ``distance`` is 0 when ``w0 . d >= 0`` already."""
    w0 = np.asarray(w0, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    need = -math.fsum(float(x) for x in w0 * d)
    if need <= 0.0:
        return Shift(0.0, w0.copy(), -1, [])
    idx = np.flatnonzero(eligible)
    if idx.size < 2:
        return None
    to = int(idx[np.argmax(d[idx])])
    top = d[to]
    donors = sorted((int(i) for i in idx if i != to and d[i] < top), key=lambda i: (d[i], i))
    w = w0.copy()
    moved = 0.0
    taken: list[int] = []
    for i in donors:
        if need <= 0.0:
            break
        rate = top - d[i]
        amount = min(w0[i], need / rate)
        if amount <= 0.0:
            continue
        w[i] -= amount
        w[to] += amount
        moved += amount
        need -= amount * rate
        taken.append(i)
    if need > 1e-12:
        return None
    return Shift(2.0 * moved, w, to, taken)
