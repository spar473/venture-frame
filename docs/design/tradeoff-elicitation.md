# Trade-off elicitation: weights implied by a person's own choices (DD11)

**Status:** approved by the owner in the DD11 session prompt of 28 November 2026: new modules
`src/mcda_studio/methods/tradeoff.py`, `src/mcda_studio/services/tradeoff_service.py`,
`src/mcda_studio/domain/tradeoff_answers.py` (the `tradeoff-answers` sidecar, A1),
`src/mcda_studio/calibration/tradeoff.py` (noisy mode on the calibration sampler), the CLI group
`mcda elicit tradeoff`, one additive `weight-samples` basis value `"tradeoff"` with its readers
(A2), tests, the mutant group `tradeoff`, and a synthetic study notebook kept in `$MCDA_DATA`. No
existing method maths, prior, kappa rule, seed, test or threshold changes; no stored project schema
change (trade-off answers never go inside a project file); no new dependency.

## Why

Layer 1 of the weight checks asks whether a judgement agrees with itself (swing against ROC and
AHP, consistency ratio, order conflicts). Nothing checked it against anything outside it, and
Layers 2 and 3 need a fund's decision log or outcomes. Trade-off questions are a first-person check
that needs no fund data: the partner answers choices between two ventures, and the weights those
answers imply are set beside the weights the partner stated. A gap is a prompt to talk, not a
verdict.

## The method

**Answers are linear constraints on the global leaf weights.** Each question shows two ventures
that differ on exactly two criteria, one better on the first and worse on the second, at the levels
of each criterion's own scale (PAPRIKA style). With `v` the project's value functions and `w` the
global leaf weights, the answer "A over B" says `w . (v(A) - v(B)) >= 0`. Working in global leaf
space keeps constraints that cross sibling groups linear.

**Levels.** The labels of a labelled scale; the points of a piecewise-linear value function; the
ends and middle of a linear or direct-rating one (on the resolved domain). Two labels with the same
value count once. Leaves the prior cannot move (a screened-only mandate, a branch rated 0) are
never asked. Optionally (`--real-pairs`), pairs of the project's own ventures are added.

**Prior: the partner's stated profile**, one block per varying sibling group:

| `--prior` | block | source |
| --- | --- | --- |
| `auto` (default) | the profile's elicitation record: its conviction if it has one, otherwise `ranking` | |
| `ranking` | uniform on the ordered region (Dirichlet(1, ..., 1) restricted to the order) | the record's order, else the swing ratings' order (ties: tree order) |
| `unknown` | Dirichlet(1, ..., 1) | |
| `low` / `medium` / `high` | Dirichlet(kappa * stated local weights), kappa moment-matched to the conviction band | `nested_dirichlet.prior_for_profile`, unchanged |
| `stated` | the project's SMAA-2 groups: `unknown`, `ranking` or `rating_intervals` (ratings uniform in their intervals, normalised) | `project.smaa.groups` |

Held children keep their stated local weight; fixed groups keep their stated weights; global
weights are products down each path, as everywhere else in the engine.

**Exact mode (default): the prior restricted to the region the answers allow.** Sequential Monte
Carlo, own NumPy, `PCG64(seed)`: draw `N = 4000` particles from the prior; after each answer keep
the particles that satisfy it, resample them systematically back to `N`, then rejuvenate with 20
sweeps of random-walk Metropolis per block. Moves are made in each block's latent coordinates (the
additive log-ratio for a Dirichlet or ordered block, with the Jacobian in the density; the ratings
themselves for an intervals block), target the prior density times the indicator of every answer
so far, and use the particle cloud's covariance (scaled by `2.38^2 / d`) fixed for the duration of
the move. Each step is a valid Metropolis kernel for the restricted prior, so no particle ever
leaves a constraint once inside it.

**Noisy mode:** `P(A over B) = sigmoid(beta * w . (v(A) - v(B)))`, `beta ~ LogNormal(log 20, 1)`,
plugged into the calibration sampler through `Problem.log_likelihood` (the hook revealed
preference added), with the sampler and its gate (R-hat below 1.01, bulk ESS at least 400)
unchanged. `beta = 20` puts a full swing on one leaf against nothing on another (a score difference
of about one leaf's weight, 0.05 to 0.1 on case 2's 16 leaves) at 0.73 to 0.88. An ordered block
becomes a Dirichlet(1, ..., 1) group whose density is `-inf` off the order. Rating intervals have no
density the sampler can use, so noisy mode refuses a `stated` prior with interval groups. Noisy
mode lives in the calibration package, which only `cli.py` may import (the boundary test): the API
and the app never fit it.

**Questions.** Default: the candidate whose `P(A over B)` under the current particles is closest
to 0.5 (the first on a tie). Never a question whose answer is already implied (`P < 0.02` or
`P > 0.98`), never one already asked in either orientation. `--random` picks uniformly among the
open questions (a control). **Stop** after 30 questions, or when P(first) of every venture has moved
less than its 99% sampling noise, `2.576 * sqrt(2 p (1 - p) / N)`, over the last 3 answers.
Particles after rejuvenation are correlated, so the real noise is larger and this rule errs toward
asking more.

**Conflicts (exact mode).** An answer that leaves no particle is not dropped silently: it is
reported with the earlier answers it contradicts one to one. That pairwise check is exact, by
vertex enumeration on the simplex of the leaves the prior moves (a relaxation when the prior's
support is smaller, such as an order). No LP solver is a runtime dependency (SciPy is dev-only), so
a minimal set of three or more answers is not searched for; when no single earlier answer
contradicts it, the report says the answers may conflict jointly, or leave a region smaller than the
particles resolve. The person is asked to re-ask the listed questions. `fit --mode auto` (default)
then falls back to noisy mode; `--mode exact` stops with status `conflict` and writes nothing.

**Report** (`mcda elicit tradeoff fit`): per leaf, the stated weight, the prior's median and 90%
interval, the implied median and 90% interval, and the gap (implied median minus stated); P(first)
and class shares per venture under the prior and under the implied weights; conflicts; noisy-mode
diagnostics and beta. With `--out`, a `weight-samples` sidecar with `basis: "tradeoff"` (A2) that
SMAA-2 reads as its weight source; simulated answers never become one.

## Stored formats

- **`tradeoff-answers` v1** (A1): a strict sidecar kept in `$MCDA_DATA`, never inside a project.
  Judgements only: per question, the two ventures as levels on the criteria they differ on (or two
  of the project's ventures by id), and the choice `A`, `B` or `skip`; who answered, when, against
  which project fingerprint and profile, and whether the respondent was a person or simulated.
  Value differences and weights are derived when the file is read. A project changed since the
  answers were given is read with a warning; a level the project no longer has is refused.
- **`weight-samples` v1, basis `"tradeoff"`** (A2): the same rules as basis `"prior"` (no
  reference set, no outcome-model priors, never diagnostic, usable, no review states) with its own
  fixed label and limitations. The prior groups are recorded; for an ordered group, `kappa_method`
  names the order. Provenance (the answers' SHA-256, who answered, exact or noisy) is in
  `prior.conviction_source` and `sampler.algorithm`. Readers: SMAA-2's samples source (unchanged
  checks), `WeightSamplesUse.basis`, and the decision-confidence wording.
- The filled sheet (CSV) can be read directly by `fit`; `--save-answers` writes it as
  `tradeoff-answers`.

## Commands

```sh
mcda elicit tradeoff next  <project> [--answers answers.json|sheet.csv] [--prior auto] [--real-pairs]
mcda elicit tradeoff fit   <project> --answers answers.json|sheet.csv [--mode auto|exact|noisy] \
    [--answered-by NAME --answered-on YYYY-MM-DD --save-answers answers.json] [--out samples.json]
mcda elicit tradeoff sheet <project> -o <dir> [--questions 20]
```

**The fixed sheet.** `sheet` chooses its questions adaptively against a simulated respondent who
answers each question A or B with probability 1/2 (`PCG64(seed).jumped(2)`); the simulated answers
are discarded and only the questions kept. The stop rule is not applied, so the person answers all
of them. The page beside the CSV explains this in plain words.

## Proof

All seeds derive from the base seed 2026112811, unused before DD11, fixed before the first run.

- **Oracles** (`tests/regression/test_tradeoff_oracles.py`), KS at alpha = 0.001 per marginal:
  order answers only reproduce SMAA-2's `ranking` kind (flat tree; case 2's nested tree, 16
  leaves); moves without answers keep SMAA-2's `unknown` Dirichlet(1, ..., 1) on case 2; on a
  four-leaf toy, exact mode matches brute-force rejection from 10^6 prior draws (marginals and
  means within six standard errors).
- **Invariant:** with a truthful respondent whose weights are a prior draw, no answer ever empties
  the particles, and the planted weights and every particle satisfy every kept answer, on every
  answer of eight 30-question runs.
- **Unit tests** (`tests/unit/test_tradeoff.py`): the likelihood by hand, the pair check against
  SciPy's LP solver on 400 random pairs, the question and stop rules, the intervals and ordered
  blocks under the kernel, noisy mode's convergence and direction, the sidecars' refusals,
  conflicts and the fallback, and read-only commands (the project file's bytes are unchanged).
- **Mutants:** the group `tradeoff` in `scripts/mutation_check.py` (`--tradeoff-only`).
- **Synthetic study** on case 2: `$MCDA_DATA/notebooks/dd11_tradeoff.ipynb`, with its design,
  seeds and success threshold written before it ran; results are in the handoff.

## Part 3: break-even weights

`services/break_even_service.py`, `methods/mavt/break_even.py`, `mcda evaluate --break-even`
(`--json` for the report alone), and the `break_even` field of the judgement aids (A3).

- **Per criterion** (every node with siblings and weight): the local weight at which the leader
  first changes, down and up, with its siblings rescaled in proportion. This is the existing
  one-way sensitivity (`weight_sensitivity`, exact crossovers), reused and only exposed. Each
  threshold is also given as the node's share of the whole goal (local weight times its parent's
  share, which does not move), and as a sentence: "Upstock stays first unless Team falls below
  4.0% or rises above 58%". For a top-level criterion this is the same as rescaling every other
  weight in proportion; deeper down, only the siblings move (the engine's one rule).
- **Nearest change**: the global weight vector nearest the stated one in L1 at which another
  alternative ties the leader, and which one. Moving mass from leaf i to leaf j costs 2 per unit
  and gains `d_j - d_i` with `d = v(challenger) - v(leader)`, so the optimum gives everything to the
  leaf with the largest `d` and takes from the smallest `d` first: exact, no solver. Leaves rated 0
  (a screened-only mandate) never move. The distance is reported in L1; the sentence also gives
  half of it, the weight that moves.
- **Class boundaries** (MAVT sorting): per alternative, the nearest weight vector that takes it to
  the boundary of the class above and of the class below (`w . v - b = w . (v - b)` on the simplex).
- Tests: the analytic two-alternative case; a brute-force simplex grid and a fine line search on
  a three-leaf toy; the nearest shift against SciPy's LP solver on 300 random cases; mutants in the
  group `break_even`.

## Part 4: anchored swings and the reference profile (A3, A4)

`services/judgement_aids_service.py`, `POST /api/judgement-aids` and
`POST /api/projects/{id}/judgement-aids` (additive), and a section of the Judgement tab. Python
supplies every number, sentence and verdict; the tab only shows them.

- **Anchors**: for each leaf with weight, the worst and best alternative in the project on it (by
  partial value; ties to the first in project order, with the tie counts), with their levels as
  entered: "From Kea Aerospace's level (...) to Upstock's (...) on Team completeness".
- **Reference profile** (`examples/reference_profiles/vc_literature.json`, a strict
  `reference-profile` v1 document): ordinal statements only ("these criteria above those"), per
  variant and optionally per horizon, with citations. No weights, no percentages, never a default:
  the app shows the partner's order beside it and marks each pair the order agrees with, disagrees
  with, or ties. Pairs in different sibling groups, or naming criteria the project lacks, are "not
  comparable". Variants:
  - *Stated by VCs* (Gompers, Gornall, Kaplan and Strebulaev, NBER w22587, 2016; JFE 2020): team
    above business;
  - *Outcome-predictive* (Jang and Kaplan, NBER w33483, 2025): for early funding (at least $1M)
    team above business; for later rounds and long-term outcomes market and product above team.
  Mapping onto case 2's top level: business = Problem and solution (product) and Market; Traction is
  left out because neither source names it. The API reads reference profiles from the examples
  folder it serves; an installed wheel does not bundle this folder (the build configuration is
  unchanged), so there the section is simply absent.

## What the synthetic study found (28 November 2026)

Case 2, profile Technical/IP-led, ranking prior, 40 planted truths drawn from it, 30 questions;
full tables in the DD11 handoff and notebook (kept in `$MCDA_DATA`). The success threshold was
fixed before the run and not tuned afterwards.

- **Threshold not met.** Adaptive exact mode reached a median L1 error of 0.199 after 15 questions
  against 0.353 for the ranking prior alone (ratio 0.56; the bar was 0.5). It reached half between
  20 and 25 questions (0.105 after 30). Adaptive beat random questions at every count.
- With a truthful respondent, coverage of the 90% intervals stayed at or above nominal.
- **Exact mode does not catch random slips.** An adaptive question is only asked when both answers
  keep at least 2% of the particles, so a flipped answer never empties them and the conflict check
  never fires; with 10% or 20% of answers flipped, coverage fell to 0.63 and 0.46 after 30
  questions. Noisy mode on the same answers kept coverage at 0.92 to 0.96, though at the sampler's
  defaults 3 to 5 of 20 fits did not pass the convergence gate. For a real person's answers, fit
  noisy mode as well as exact mode and compare.
- The stop rule would have stopped most runs after about 11 or 12 questions, before the later
  questions' gains.
- **Against a wrong conviction** (truths 0.1 to 0.3 TV from the stated weights, low-conviction
  prior), 30 answers roughly halved the L1 error but did not restore coverage: the answers do not
  overrule a confident prior that far within 30 questions.

## Known limits

- The additive model is assumed: each answer reads the model's trade-off at two levels with
  everything else equal.
- Answers inform weights only along the contrasts asked; elsewhere the prior decides.
- Exact mode treats every answer as considered; one slip removes the truth from the region and
  nothing in exact mode brings it back. That is what noisy mode and the conflict report are for.
- SMC particles are correlated; intervals carry more Monte Carlo error than N suggests.
- Draws repeat for a seed on one platform and NumPy version; matrix products are not bit-identical
  across platforms, so results are compared with tolerances, never by bytes.
- One person's answers: partners are never averaged.
