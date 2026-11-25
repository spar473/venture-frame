# Structured judgement

**Status:** accepted by the owner, 25 November 2026 (deep dive 3 of the evidence-backed weighting
plan). This includes stored schema version 7, the SMAA-2 `ranking` kind and `held` children, and
the conviction rule. Earlier method outputs are unchanged: 969 of 969 byte-identical against the
previous commit (`scripts/byte_identity.py`).

## Problem

A weight profile used to be a set of swing ratings someone typed in, with no record of who gave
them, how sure they were or why. Deep dive 1 showed what that costs. Decision confidence on case 2
moves from 100% to 42% for the same company depending only on how the weight uncertainty is
written down. So that uncertainty has to be a recorded judgement, not an author's guess.

## What is stored (schema 7)

A weight profile may carry an **elicitation record** (`WeightProfile.elicitation`):

| Field | Meaning |
| --- | --- |
| `elicited_by`, `elicited_on`, `note` | Who gave the judgement, when, and anything to remember |
| `conviction` | `high`, `medium` or `low`: how far the swing ratings could reasonably move |
| `groups[].parent_id` | One entry per sibling group judged (`null`: the top level) |
| `groups[].order` | The children best first, each once, no ties |
| `groups[].pairwise` | Optional AHP answers for the same children, a cross-check |
| `groups[].reasons` | One line per child, in the person's words |

The swing ratings stay in `WeightProfile.judgements`. The record stores judgements only;
everything below is derived and never stored.

SMAA-2 gains two things:

- **`ranking`** weight groups (`kind: "ranking"`, `order`): local weights uniform on the part of the
  simplex that keeps the order.
- **`held`** children on any weight group: held at the profile's local weight in every draw, while
  the other children share what is left. This fixes the mandate problem from deep dive 1. A
  screened-only group the profile rates 0 can now be held at 0 while its siblings vary.

A project that uses none of these keeps its bytes, fingerprint and schema version (the
lowest-sufficient-version rule). One that uses any of them is version 7. A version-6 file that
contains them is refused on load (`_v6_to_v7`). Removing a criterion removes it from every order,
held list and elicitation record. Moving one leaves stale references for validation and the check
to report, as intervals already do.

## The ranking kind

For a ranked group with varying children *c₁ … cₖ* (best first), each draw:

1. takes *k* uniforms and turns them into exponentials *eᵢ = −log(1 − uᵢ)* (portable log1p, as for
   `unknown`);
2. normalises them, *dᵢ = eᵢ / Σe*, which is uniform on the simplex (Dirichlet(1, …, 1));
3. sorts the shares in descending order and gives the largest to *c₁*, the next to *c₂*, and so on.

Sorting maps each of the *k!* equal-volume ordered cells of the simplex onto the one cell that
keeps the stated order. So the result is uniform on that cell. Its mean is the rank-order centroid
*wⱼ = (1/k) Σᵢ₌ⱼᵏ 1/i* (Barron and Barrett, 1996). With held children, the shares are multiplied
by *1 − Σ held local weight*. "Inputs to check" judges a ranked child by its local weight against
the median, as for `unknown`.

## The check (`elicitation_service.elicitation_check`)

For every sibling group with two or more children:

- **Swing weights** from the profile's ratings. Children rated 0 are marked **held**.
- **Rank-order weights** (ROC) for the stated order, on the same footing as the swing weights
  (scaled by what held children leave).
- **Order against ratings**: every pair where the child ranked higher has a strictly lower swing
  rating. Equal ratings never conflict.
- **AHP cross-check** over the ranked children, using the engine's own AHP matrix code: priorities,
  consistency ratio (CR > 0.10 is flagged, Saaty's convention) and the most inconsistent triad.
  Incomplete or invalid answers are reported, never filled in.
- **Swing-AHP gap**: the largest |swing − AHP| local weight. Above **0.10** it asks for a second
  look (a stated convention).
- **Missing reasons** for ranked children.
- **Suggested order**: children rated above 0 by swing rating (ties in tree order). The app offers
  it as a starting point the person confirms or changes.
- **Status** per group and overall: `consistent`, `check` or `incomplete`. A group under a
  criterion rated 0 is marked `scored: false` and is never varied.

## Proposals

A proposal is a complete SMAA-2 section: the project's stated measurement ranges plus new weight
groups. It changes the project only when a person applies it (`set_smaa_model`, undoable).

- **Conviction.** In every scored sibling group with two or more children rated above 0, each
  rating *r* becomes the interval [*r*(1 − h), *r*(1 + h)], with **h = 0.15 / 0.30 / 0.50** for
  high / medium / low. Children rated 0 are held.
- **Ranking.** In every scored group with an order, a `ranking` group with that order. Children
  rated 0 are held, and groups without an order keep the profile's ratings.

**How wide the conviction bands really are.** Rating intervals are normalised within a group, so
the local weights move less than the ratings do. Matching moments against a Dirichlet(κ·w)
distribution on case 2's top level (ratings 90 / 100 / 60 / 50):

| Conviction | Rating band | Local-weight s.d. | Dirichlet-equivalent κ |
| --- | --- | --- | --- |
| High | ±15% | 0.014 to 0.022 | about 550 |
| Medium | ±30% | 0.028 to 0.045 | about 140 |
| Low | ±50% | 0.048 to 0.076 | about 47 |

Even low conviction is a confident statement about the weights. A partner who is genuinely unsure
of magnitudes but sure of the order should use the ranking proposal. Under the ranking proposal,
case 2's Upstock is first in about 40% of draws, against 93% under the stated intervals.

**Preview** (`elicitation_preview`): decision confidence (deep dive 1) under the current stated
uncertainty and under each proposal, with the same SMAA-2 settings, so the person sees what their
conviction does to the decision before applying anything.

## Surfaces

- **App:** a *Judgement* tab in Build, after Weights. It holds the record fields, one editor per
  sibling group (rank with ↑/↓, the engine's swing, rank-order and AHP weights, findings, reasons,
  an optional AHP cross-check), the proposals with *Use as the weight uncertainty*, and
  *Compare decision confidence*. The Uncertainty tab can also state a ranking and hold children
  directly. The browser computes no weight, rank or check.
- **API:** `POST /api/elicitation/check`, `POST /api/elicitation/preview` and their stored-project
  forms; the `set_elicitation` edit operation.
- **CLI:** `mcda elicitation-check FILE --profile ID`; `mcda uncertainty group FILE --rank CHILD …
  --hold CHILD`.

## Verification

- `tests/regression/test_ranking_weights_reference.py`:
  - every ranked draw keeps the order (exact);
  - the mean draw equals the ROC weights to six standard errors (200,000 draws);
  - with two children the larger weight is uniform on [½, 1] (KS, α = 0.001);
  - three children match an independent rejection sampler on each coordinate and on a joint
    feature (two-sample KS);
  - held children keep the profile's weight exactly under every kind;
  - on case 2, SMAA-2 with a ranking and the mandate held matches a NumPy replica within its
    exact 99.9% intervals for first place and *Prioritise* for every company.
- `tests/unit/test_structured_judgement_schema.py`: version 7 only when used, byte-identical
  otherwise; round trip; v6 files refused; the edit operation; pruning on removal; every new
  validation code.
- `tests/unit/test_elicitation_service.py`: ROC values; order conflicts; AHP agreement with the
  engine's own matrix code; inconsistent and incomplete answers; the gap; missing reasons;
  incomplete ratings; both proposals validate and run; preview; looking changes nothing.
- API, edit-operation and CLI tests; Vitest for the Judgement and Uncertainty editors; a
  Playwright test that ranks, states conviction, compares decision confidence and applies a
  proposal against the real engine.
- 15 mutants under `--structured_judgement-only` (`scripts/mutation_check.py`): 15 of 15 killed.

## Limits

- The conviction half-widths and the 0.10 gap threshold are conventions, stated with every
  proposal. They are not estimates.
- A ranking sets the ratings' magnitudes aside entirely, and ties are not allowed. A partner with
  a genuine tie breaks it or uses intervals.
- A true Dirichlet(κ·w) conviction needs a portable gamma quantile. It is deferred to the
  calibration layer (deep dive 5), which will store sampled weights.
- The record says who gave a judgement. It does not prove that they did, and in case 2 it is the
  study author's practice elicitation on public information, not a partner's.
