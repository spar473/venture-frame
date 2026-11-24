"""Independent oracle for robustness bands (``docs/design/robustness-bands.md``).

Plain Python, no ``mcda_studio`` import, and different algorithms from the service: every
interval of ranks is listed and the best is chosen by one sort key, and groups come from a
union-find over every overlapping pair rather than a sweep.
"""

from fractions import Fraction

COVERAGE = Fraction(4, 5)
TOLERANCE = 1e-12


def rank_range(shares, preferred_rank, exact_total=None):
    """``shares`` are counts (with ``exact_total`` draws) or probabilities (``exact_total=None``).

    Returns ``(low, high)``: the shortest run of ranks holding at least COVERAGE of the draws;
    ties go to the run containing ``preferred_rank``, then to the lowest ranks.
    """
    m = len(shares)
    reaching = []
    for low in range(1, m + 1):
        for high in range(low, m + 1):
            part = shares[low - 1 : high]
            if exact_total is not None:
                ok = Fraction(sum(part), exact_total) >= COVERAGE
            else:
                ok = float(sum(Fraction(p) for p in part)) >= float(COVERAGE) - TOLERANCE
            if ok:
                contains = low <= preferred_rank <= high
                reaching.append((high - low, 0 if contains else 1, low, high))
    _, _, low, high = min(reaching)
    return low, high


def groups(ranges):
    """``ranges``: list of ``(id, low, high)`` in MAVT rank order. Overlap = at least one shared
    rank. Returns the connected components, members and groups in that order."""
    parent = list(range(len(ranges)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i, (_, lo_i, hi_i) in enumerate(ranges):
        for j, (_, lo_j, hi_j) in enumerate(ranges):
            if i < j and lo_i <= hi_j and lo_j <= hi_i:
                parent[find(j)] = find(i)
    components = {}
    for i, (alt, _, _) in enumerate(ranges):
        components.setdefault(find(i), []).append((i, alt))
    ordered = sorted(components.values(), key=lambda members: min(i for i, _ in members))
    return [[alt for _, alt in sorted(members)] for members in ordered]
