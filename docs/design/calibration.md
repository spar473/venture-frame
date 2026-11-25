# Calibration: weights that fit recorded outcomes (deep dive 5)

**Status:** approved in scope by the owner in the DD5 session prompt of 25 November 2026: a new,
domain-neutral, offline module `src/mcda_studio/calibration/`, a new opt-in sidecar format
`weight-samples` version 1, a new CLI group `mcda calibrate`, tests and docs. It changes no method
maths, no stored project schema (still version 7), no reference fixture, golden report, pinned
digest or byte-identity manifest, and neither `reference-cases` format. SMAA-2 does not read the
sidecar; that is a separate proposal, [sampled weights in SMAA-2](sampled-weights-in-smaa.md).

## What this is, and what it is not

Calibration asks: given a weight profile someone stated, and a version 2 reference set of past
cases rated on the same labelled criteria with observation-aware outcomes, which global weights fit
those outcomes under a stated time-to-event model, and how far does the evidence move them from
what was stated? The answer is a set of posterior weight samples with diagnostics.

**Claims ladder.**

- *Allowed after this deep dive:* "the method recovers planted weights on synthetic data", with the
  recovery, coverage and simulation-based calibration numbers of the tests below.
- *Not allowed:* anything about real outcomes. On the real reference set available today (55
  cases, 0 reviewed ratings, 2 of 17 labelled leaves rated even with proposed ratings) the method
  must report that the data are insufficient and the posterior is the prior.

Every sidecar carries this fixed label:

> Calibrated weights describe what fitted the reference set's outcomes under the stated model; they
> are not a forecast for any alternative.

The app never calls a model and the API and the interface never import this module (a test
enforces it). It runs from the command line only.

## 1. Data

Inputs: a project, a weight profile id (default: the profile the project's analysis
configuration resolves), a version 2 reference set (validated with
`reference_outcome_service.check` first, so nothing runs on an invalid set or one that breaks the
snapshot guard), and an outcome definition key (default: the set's only definition; required when
it has several).

- **Compared leaves `L`.** The labelled (ordinal or categorical) leaves whose global weight in the
  profile is positive, from the engine's own `compute_weights`, in tree order. This is
  `reference_case_service.compared_criteria`, reused unchanged.
- **Values.** For case `i` and leaf `j ∈ L`, `v_ij` is the project's own value function for `j`
  (the base model's qualitative mapping; scenarios are not applied) evaluated at the case's label,
  through `build_evaluator`. So `v_ij ∈ [0, 1]` and higher is better, as everywhere in the engine.
- **Which ratings count.** Only ratings with `review: "reviewed"`. `--include-proposed` also counts
  `proposed` ratings. It exists for diagnostics only: the run is then marked `diagnostic: true`,
  every output says so in a fixed sentence, it never meets the usable bar, and a future consumer
  must refuse it.
- **Missing is missing.** A leaf a case does not rate is never zero. Let `R_i ⊆ L` be the leaves case
  `i` rates (under the review rule). A case **enters** only if `2·|R_i| ≥ |L|` (it rates at least
  half the compared leaves; exactly half enters, the same convention as reference cases). The
  threshold is the `--min-rated-fraction` input, default `1/2`, an exact fraction greater than 0
  and at most 1.
- **Marginalising missing leaves by renormalising.** For an entering case, the weights are
  renormalised over the leaves it rates (next section). *Alternative not chosen:* integrating the
  missing labels over a distribution of labels (for example the empirical label frequencies of the
  other cases). That would invent a rating distribution the reference set does not state, and it
  makes every case's likelihood depend on every other case's labels. Renormalising states plainly
  that a case is scored only on what it rates.
- **Which outcomes count.** For the chosen definition: `event`, `no_event`, `censored` and
  `competing_event` enter; `unknown` and "not recorded" are excluded. **Assumption (missing at
  random):** whether a case has follow-up does not depend on its outcome given its ratings. If
  cases are unknown because they quietly closed, this assumption fails and the fit is biased
  toward cases that were followed; the output says so.
- **Time.** Time is measured in window fractions: `t = (day − snapshot) / D`, with `D` the window
  length in days (`window_end(snapshot, window_months) − snapshot`, from the reference-outcome
  service). So the window is `[0, 1]`.
- **Sufficiency.** The run is **insufficient** when any of these holds, with each reason listed:
  `0 reviewed ratings` on the compared leaves (or `0 ratings` under `--include-proposed`); no case
  enters; no event among entering cases; no entering case without an event. An insufficient run
  that is not diagnostic is refused before any sampling: the command prints the reasons and exits
  1, and nothing is written. A diagnostic run still samples (with no entering case the posterior is
  exactly the prior) so that prior and posterior can be compared, and is labelled
  `prior-dominated / insufficient`.

## 2. Score

For an entering case:

    s_i = Σ_{j ∈ R_i} w_j v_ij / Σ_{j ∈ R_i} w_j

where `w` are the global weights: the product of local shares along each leaf's path (objective
share × within-objective share), exactly as `global_weights` computes them. With every leaf rated
this is `Σ_j w_j v_ij`; renormalising only removes the missing leaves. `s_i ∈ [0, 1]`.

Leaves outside `L` (numeric, zero-weight) never enter a score, so their weights are moved only by
the prior and by the sum-to-one constraint.

## 3. Prior (layer 1): a nested Dirichlet centred on the stated weights

The tree's sibling groups are visited in the engine's order: the top level, then every non-leaf in
tree order. In each group:

- **Fixed groups.** A group is fixed at the profile's local weights when its parent's path share is
  0 (a zero-weight branch: its leaves have global weight 0 whatever their local shares), or when it
  has fewer than two free children.
- **Held children.** A child is held at its profile local weight in every draw when the profile
  rates it 0 (DD3: "children rated 0 are held") or when the project's SMAA-2 section names it in
  that group's `held` list (DD3's held children). Held children take no Dirichlet component.
- **Free children.** The others. Their shares are `(1 − Σ held local weight) · p`, with
  `p ~ Dirichlet(κ_g · w0_g)`, where `w0_g` are the free children's profile local weights
  renormalised among themselves. So the prior mean of every local weight is exactly the profile's.
- **κ from conviction.** The conviction comes from the profile's elicitation record (`high`,
  `medium` or `low`); when the profile has none, `--conviction` must state one (recorded as stated
  at the command line); with neither, the fit is refused. Conviction maps to DD3's rating
  half-width `h` (0.15, 0.30, 0.50, read from `elicitation_service.CONVICTION_HALF_WIDTH`). For each
  varying group, `κ_g` is **moment-matched** to DD3's conviction proposal on that group: draw each
  free rating `x_i` uniformly from `[r_i(1 − h), r_i(1 + h)]`, form `q_i = x_i / Σ x`, and set

      κ_g = round( mean_i [ w0_i (1 − w0_i) / Var(q_i) ] − 1 )

  rounded to the nearest integer. (A Dirichlet has `Var(p_i) = w0_i(1 − w0_i)/(κ + 1)`; the mean
  over components is the matching DD3's table used.) `Var(q_i)` is computed by tensor
  Gauss–Legendre quadrature over the rating box with `n = min(24, ⌊10⁶^{1/k}⌋)` nodes per free
  child (`k` free children), which is exact to far below the rounding for the smooth ratio. For
  `k > 12` (fewer than 3 nodes per axis) the first-order delta method is used instead and the prior
  record says so. On case 2's top level (ratings 90 / 100 / 60 / 50) this gives **555, 136 and 47**
  for high, medium and low, the defaults the DD5 prompt names.
- **Independence.** Groups are independent a priori, so the prior mean of a global weight is the
  product of the means along its path and its prior variance follows from
  `E[p²] = w0(κw0 + 1)/(κ + 1)` per factor. These exact moments are reported.

**Unconstrained parametrisation.** Each varying group with `k` free children has `y ∈ ℝ^{k−1}`,
`p = softmax(y, 0)` (the additive log-ratio, last free child as reference). The density of `y`
includes the Jacobian `Π_i p_i`:

    log π(y) = log Γ(κ) − Σ_i log Γ(κ w0_i) + Σ_i κ w0_i log p_i

(the Dirichlet's `Σ (κ w0_i − 1) log p_i` plus `Σ log p_i`).

## 4. Outcome likelihood (censoring-aware)

Constant hazard within the window, `λ_i = exp(α + β s_i)`, in events per window. For entering
case `i` with the chosen definition:

| Status | Contribution to the log-likelihood |
| --- | --- |
| `event` on an exact day, `t` | `log λ − λ t` (density `λ e^{−λt}`) |
| `event` on a partial date | `log(e^{−λa} − e^{−λb}) = −λa + log(1 − e^{−λ(b − a)})`, with `a = (first − snapshot)/D` and `b = (last + 1 day − snapshot)/D`, both clipped to `[0, 1]` |
| `no_event` | `−λ` (watched through the window) |
| `censored`, watched to `c` | `−λ c`, `c = (observed_through − snapshot)/D` |
| `competing_event` at `t` | `−λ t`: censored at `t` (cause-specific hazard); on a partial date, censored at `a`, the earliest day it can mean (it uses no follow-up that may not have happened) |
| `unknown`, not recorded | excluded (above) |

An exact-day `event` at `t = 0` has density `λ`. `log(1 − e^{−x})` is evaluated as
`log(−expm1(−x))` for `x ≤ log 2` and `log1p(−e^{−x})` above (Mächler, 2012), accurate at both
ends.

**Other priors.**

- `α ~ Normal(log(−log(1 − p0)), 1.5²)`. `p0` is a stated base rate of the event over the window
  (`--base-rate`, default 0.15). It is an input recorded in the output, never fitted. With `s = 0`
  and `β` small, `p0` is the prior median chance of an event in the window.
- `β ~ LogNormal(0, 1)`, sampled as `b = log β ~ Normal(0, 1)` (this is the density of `b`, so no
  further Jacobian).

**Why β must be positive.** The value functions already say which direction is better on every
leaf. The likelihood depends on the weights only through `β·s_i`. If `β` could be negative, a fit
with `β < 0` would read "lower value is better" on every leaf at once, which contradicts the value
functions, and near `β = 0` the weights would not be identified at all. A positive `β` encodes
"higher value is better" and removes that reflected mode.

**What is identified.** Even with `β > 0`, the weights are identified only up to the information in
`v`. Only contrasts along leaves whose values vary across entering cases inform them; a leaf no
entering case rates is moved only by the prior; and a leaf with the same value in every entering
case trades off against `α` and `β` and is informed mainly by the prior. Two leaves whose values
move together across cases are identified only in their sum. The report gives, per leaf, how many
entering cases rate it and how far the posterior moved from the prior, so this is visible.

## 5. Sampler

Our own sampler, NumPy only: no PyMC, no SciPy at runtime (the engine's rule; SciPy serves the
tests as an oracle). The Dirichlet normalising constants use the engine's portable `lgamma`. The
per-draw arithmetic uses NumPy's `exp`, `log`, `expm1` and `log1p`: the sampler's output is
promised to repeat on one platform and NumPy version, not bit for bit across platforms (see
*Chains and seeds*), and the gate below is statistical, so the portable functions would add cost
without adding a guarantee. Normal scores for the diagnostics use the standard library's
`statistics.NormalDist`.

- **Blocks (Metropolis within Gibbs).** One block per varying Dirichlet group (its `y`), in tree
  order, then one block `(α, log β)`. Each iteration updates every block once with a random-walk
  Metropolis step on the full log posterior.
- **Adaptation.** During warm-up only: each block's proposal is `N(0, σ² Σ)`. `Σ` starts as the
  identity times 0.1 and is replaced at warm-up iterations 200, 400, 800, … by the block's
  empirical covariance over the second half of warm-up so far (plus `10⁻⁶ I`); `log σ` follows a
  Robbins–Monro step toward the acceptance rate 0.44 for one dimension and 0.30 otherwise, with
  step size `iteration^{−0.6}`. Adaptation stops at the end of warm-up, so the kept draws come from
  a fixed Markov kernel whose stationary distribution is the posterior.
- **Chains and seeds.** 4 chains by default. The engine's convention is
  `numpy.random.Generator(PCG64(seed))`; chain `c` uses `PCG64(seed).jumped(c)`. Each chain starts
  from its own draw: every `y` from `N(ALR(w0), 1)` coordinatewise, `α` and `b` from their priors.
  Defaults: 2 000 warm-up and 3 000 kept iterations per chain, thinned evenly to **4 000** stored
  vectors. The algorithm is named `calibration.rwm-gibbs.v1`. The same seed, settings, inputs,
  platform and NumPy version give the same samples; other platforms are not promised the same bits
  (NumPy's normal draws use the platform's maths library).
- **Diagnostics.** For every global weight that varies, and for `α` and `β`:
  - **split-R̂**, rank-normalised (Vehtari, Gelman, Simpson, Carpenter and Bürkner, 2021): each
    chain is split in half, draws are replaced by normal scores of their pooled ranks, and the
    classic potential scale reduction is computed on the halves; the maximum of that and the same
    on folded draws (`|x − median|`) is reported;
  - **bulk ESS**, from the same rank-normalised split chains, with autocorrelations by FFT and
    Geyer's initial monotone sequence.
- **The convergence gate.** A run is **converged** only if every reported R̂ is below **1.01** and
  every bulk ESS is at least **400**. A run that is not converged is labelled `not converged`, its
  diagnostics are printed, and no sidecar is written (the sidecar model refuses `converged: false`).
  The thresholds are constants, not options.

## 6. Output: the `weight-samples` sidecar, version 1

A separate JSON file, never stored inside a project. Strict sidecar conventions (unknown fields
refused, `version` an explicit integer, any version but 1 refused naming the supported version).

| Field | Content |
| --- | --- |
| `kind`, `version` | `"weight-samples"`, `1` |
| `project_id`, `project_fingerprint` | the project the weights belong to |
| `weight_profile_id` | the stated profile the prior is centred on |
| `reference_set_id`, `reference_set_sha256`, `reference_set_digest_basis` | the set; the digest is of the file's bytes (`"file bytes"`) when read from a file, else of its canonical JSON (`"canonical JSON"`) |
| `outcome_definition` | the definition key used |
| `diagnostic`, `review_states` | `true` and `["proposed", "reviewed"]` under `--include-proposed`; else `false`, `["reviewed"]` |
| `data` | entering cases, cases excluded by reason, events, no events, censored, competing events, rated counts per leaf, the threshold |
| `prior` | conviction and its source, every group (parent, free, held and fixed children, centre, κ, how κ was computed), `base_rate`, `alpha_mean`, `alpha_sd`, the `beta` prior, the centre global weights |
| `seed`, `sampler` | algorithm, chains, warm-up, kept iterations, thinning, NumPy version, Python version |
| `diagnostics` | `converged`, the thresholds, R̂ and bulk ESS per reported quantity, acceptance rate per block and chain |
| `leaf_ids`, `samples` | every leaf of the tree in tree order, and N global weight vectors (non-negative, each summing to 1 within 10⁻⁹) |
| `alpha`, `beta` | the matching N draws of the outcome-model parameters |
| `label`, `limitations` | the fixed label above, and the limitations below |

The `fit` result (printed by the CLI) also gives, per leaf, the exact prior mean and standard
deviation, the posterior mean, standard deviation and 90% interval, and the **shrinkage ratio**
`posterior SD / prior SD`; a run where every varying leaf keeps a ratio above 0.9 is labelled
`prior-dominated`.

**Limitations (in every output).** The model is a constant hazard in a linear score; its fit to a
small set says nothing about other alternatives; unknown follow-up is assumed missing at random;
weights are identified only where the rated values vary; the reference set's author knew the
outcomes when rating.

## 7. Command line

- `mcda calibrate fit PROJECT --reference-cases FILE [--profile ID] [--definition KEY]
  [--base-rate P] [--conviction high|medium|low] [--include-proposed] [--min-rated-fraction F]
  [--seed S] [--chains N] [--warmup N] [--draws N] [--samples N] [--out FILE]` prints the fit
  report as JSON and writes the sidecar to `--out` only when the run is converged. It exits 1 when
  the run is refused (insufficient and not diagnostic, invalid input) or not converged.
- `mcda calibrate synthetic-check [--cases N] [--seed S] [--censoring independent|score]` plants
  weights on a two-objective, five-leaf synthetic tree, fits, and prints the planted values, the
  posterior summaries, whether each 90% interval covers the truth, and the diagnostics.

## 8. Verification

Each test can fail; long ones carry the `slow` marker (registered in `pyproject.toml`; they run in
the full suite and can be skipped with `-m "not slow"`).

1. **Transforms:** ALR round trip; log-Jacobian against central finite differences of the map.
2. **Prior-only run returns the prior:** with no entering case, each free share's marginal passes a
   KS test against its exact `Beta(κ w0_i, κ(1 − w0_i))` (SciPy) and its mean is within 6 standard
   errors (using the sampler's ESS).
3. **Likelihood:** every status against a hand-computed value; the interval form for a partial
   date; competing equals censored at the same time.
4. **Recovery:** planted weights on a two-objective, five-leaf tree at N ∈ {50, 150, 500, 2000}; the
   error falls with N; over repeated simulations with the truth drawn from the prior, the 90%
   intervals cover the truth at a rate whose 99% Clopper–Pearson interval contains 0.90.
5. **Simulation-based calibration** (Talts, Betancourt, Simpson, Vehtari and Gelman, 2018): at least
   100 prior draws; the rank of the truth among thinned posterior draws is uniform by a chi-square
   test at α = 0.01 for one weight, for α and for β (`slow`).
6. **Censoring matters:** with heavy censoring that depends on the score (later snapshots, higher
   scores, shorter follow-up), dropping censored cases and treating them as no event both bias `β`
   beyond their own posterior intervals, while the censoring-aware likelihood covers the truth.
7. **Convergence gate:** a 50-iteration run is labelled not converged and writes nothing usable.
8. **Read-only and the sidecar:** project bytes and fingerprint unchanged; the sidecar round-trips;
   unknown fields refused; the version must be an explicit integer; `converged: false` refused.
9. **Boundaries:** no domain words in the module or its fixed text; neither `mcda_studio.api` nor the
   frontend imports it.
10. **Mutants:** the `calibration` group in `scripts/mutation_check.py`, at least 12 mutants, every
    one killed.
