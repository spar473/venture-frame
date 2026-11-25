# Revealed preference: weights that fit a log of accept/decline decisions (Layer 2, DD8)

**Status:** approved by the owner in the DD8 session prompt of 26 November 2026, as a design
note and a synthetic-only demonstration: a domain-neutral, offline module
`src/mcda_studio/calibration/choices.py`, a generator in `calibration/synthetic.py`, the CLI command
`mcda calibrate choices-synthetic-check`, tests, mutants and a notebook. It changes no method maths,
no stored project schema, no stored format (no new `weight-samples` basis, no decision-log format),
no reference fixture, golden report, pinned digest or byte-identity output. It uses no real fund
data of any kind. Neither the API nor the interface imports it (the calibration boundary test is
extended to say so).

**Why now.** The DD7 reachability diagnostic found that, even accepting every proposed rating, 1
of 12 pilot cases reaches the 7-of-14 entry bar for outcome calibration (Layer 3) on public New
Zealand data: 0 to 22 of the 55-case cohort. Layer 3 stays prior-dominated. The informative
evidence is a fund's own log of the calls it made, which is what this layer reads. It is built
and proved on synthetic decisions first, so that a request for an anonymised decision log comes
with a working, tested method.

## What this is, and what it is not

Revealed preference asks: given a stated weight profile and a list of past decisions (each an
alternative rated on the same labelled criteria, and whether the decider accepted it), which
global weights make those decisions most plausible under a stated choice model, and how far do they
move from what was stated? The answer is posterior weight samples with diagnostics, exactly as in
[calibration](calibration.md), from the same prior and the same sampler, with a different
likelihood.

**Claims ladder.**

- *Allowed after this deep dive:* "the method recovers planted weights from synthetic decisions",
  with the recovery, coverage, simulation-based calibration and label-noise numbers of the tests
  below.
- *Not allowed:* anything about a real fund, a real decider or a real decision log. No real log
  exists in this repository or in `$MCDA_DATA`, and none was used.

Every output carries this fixed label:

> Synthetic decisions: planted weights and simulated accept/decline calls. This checks the method;
> it says nothing about any real decider.

The engine's words are domain-neutral: an *alternative*, a *decision*, *accept* or *decline*, a
*decider*. "Advance/pass" appears only in examples and the notebook's wording.

## 1. Data

A decision log is a list of decisions. Decision `i` has:

- **Values** `v_ij` for the compared leaves `j ∈ L`: the project's own value function for leaf `j`
  evaluated at the alternative's label, exactly as calibration builds them (`build_evaluator` on
  the base model; scenarios not applied). So `v_ij ∈ [0, 1]`, higher is better.
- **Rated** `r_ij ∈ {0, 1}`: whether the decision record has a label for leaf `j`. **Missing is
  missing**, never zero: the score renormalises over the rated leaves (section 2), and a decision
  that rates no compared leaf is refused.
- **The decision** `y_i ∈ {1 (accept), 0 (decline)}`.
- Optionally a **decision date** and a **decider id**. Dates are carried for provenance (and for a
  later out-of-time check); they do not enter the likelihood in this version.

**Compared leaves** are the labelled leaves with positive global weight in the profile
(`reference_case_service.compared_criteria`, reused unchanged), as in calibration.

**One decider per fit.** Partners are never averaged. A log with more than one decider id is
refused unless the caller states a pooling choice (`pool_deciders=True`), which assumes the deciders
share one weight vector and one `(α, β)`; the output then records the pooling and the decider ids.
Fitting each decider separately and comparing their posteriors is the default way to look at a
partnership. A hierarchical model (deciders' weights drawn around a shared centre) is a possible
later step, not built here.

## 2. Score

The same score as calibration (section 2 there), reused from `likelihood.scores`:

    s_i = Σ_{j ∈ R_i} w_j v_ij / Σ_{j ∈ R_i} w_j,   R_i = { j : r_ij = 1 }

with `w` the global weights (products of local shares along each path). With every leaf rated it
is `Σ_j w_j v_ij`, and `s_i ∈ [0, 1]`.

## 3. Model

    P(accept_i) = σ(α + β · (s_i − s_ref)),   σ(x) = 1 / (1 + e^{−x}),   s_ref = 1/2

This is the stated model `σ(α′ + β s_i)` with `α′ = α − β s_ref`: the same family, with the
intercept placed at a fixed reference score. `s_ref` is the midpoint of the value scale, a
constant, not a property of the data, so the prior on `α` does not depend on the log. Each
decision contributes

    y_i log σ(η_i) + (1 − y_i) log σ(−η_i),   η_i = α + β (s_i − s_ref)

evaluated stably as `−log(1 + e^{−η})` and `−log(1 + e^{η})` (`numpy.logaddexp(0, ∓η)`).

**Priors.**

- **Weights:** the nested Dirichlet of calibration section 3, reused from
  `methods/nested_dirichlet.py` unchanged: centred on the stated profile, `κ` per group
  moment-matched from conviction (`high`, `medium`, `low` → DD3 half-widths 0.15, 0.30, 0.50),
  held children (rated 0, or listed in the SMAA-2 `held` lists) fixed, zero-weight branches fixed.
- **`α ~ Normal(logit p0, 1.5²)`.** `p0` is a **stated base accept rate**: the share of
  alternatives at the reference score that the decider accepts. It is a required input, recorded
  in every output and **never fitted silently**: the posterior of `α` is reported beside its prior.
  The synthetic bench states `p0 = 0.3`.
- **`β ~ LogNormal(log 8, 1²)`**, sampled as `log β`. `β` is the change in log-odds from the worst
  possible alternative (`s = 0`) to the best (`s = 1`). The median 8 means a decider who, at
  `p0 = 0.3`, accepts about 1% of the worst and 96% of the best alternatives; the central 90% of
  the prior is `β ∈ [1.5, 41]`. Calibration's outcome model uses `LogNormal(0, 1)` for a hazard
  ratio, a different quantity; a screening decision is far more sensitive to the score than an
  outcome hazard, and a prior centred at 1 would claim that deciders barely look at the criteria.

**Why `β` must be positive.** The value functions already fix which direction is better on every
leaf, and the likelihood depends on the weights only through `β s_i`. With `β < 0` a fit would read
"lower value is better" on every leaf at once, contradicting the value functions, and it would
create a reflected mode. Near `β = 0` the decisions do not depend on the score at all and the
weights are not identified: the posterior is then the prior, which is the honest answer. A
positive `β` states the one assumption the method needs: the decider prefers higher value.

**Links considered and not chosen.**

- *Probit* (`Φ` instead of `σ`): nearly the same fit after rescaling `β` by about 1.7; the logistic
  link gives `β` a log-odds meaning and has heavier tails, so a few inconsistent calls pull less on
  the weights.
- *A threshold model* (accept when `s_i + ε_i > τ`): the logistic model **is** a threshold model
  with logistic `ε` (`α′ = −τ/scale`, `β = 1/scale`). A hard threshold (no `ε`) cannot explain a
  single inconsistent call and has no usable likelihood.
- *Conditional logit over choice sets:* the right model if the log recorded which alternatives
  competed for one slot. Screening logs record independent accept/decline calls, so the binary
  model is the natural one.
- *A lapse (label-noise) mixture* `ε + (1 − 2ε) σ(η)`: one more weakly identified parameter. This
  version measures what label noise does instead (section 6, test 6) and leaves the mixture for a
  real log that shows the need.

## 4. Sampler and the convergence gate

The calibration sampler (`calibration.sampler.run`), reused: adaptive random-walk Metropolis within
Gibbs blocks, one block per varying Dirichlet group and one for `(a, log β)` with the centred
intercept `a = α + β · c`, `c` the mean of `s_i − s_ref` at the prior's centre weights (Jacobian 1,
so the posterior is unchanged). 4 chains, chain `c` on `PCG64(seed).jumped(c)`. The same gate: a
run is **converged** only if every split-R̂ (rank-normalised, folded) is below **1.01** and every
bulk ESS is at least **400**, for every varying weight, `α` and `β`; otherwise it is labelled
**not converged** and returns no usable samples.

**What reuse needs, and why it changes no output.** The sampler gets two additive hooks:

- `Problem.log_beta_mean` (default `0.0`): the outcome model's `β` prior is `LogNormal(0, 1)`, so
  `0.0` reproduces it exactly (`x − 0.0 == x` and `0.0 + x == x` in floating point, and no random
  draw is added or removed).
- `Problem.log_likelihood(weights, alpha, beta)`: the outcome likelihood by default; the choice
  model overrides it in a subclass, `ChoiceProblem`, together with the centring constant.

The calibration mutant "beta allowed negative in the likelihood" follows the one line that moved;
it mutates the same argument in the same way. The byte-identity manifest, the calibration tests and
the `calibration` mutant group are run to show nothing else moved.

## 5. What it cannot identify

- **Selection.** A fund only logs alternatives it saw. That is selection on `v` (which deals
  arrived), not on the decision given `v`, so the conditional model `P(accept | v)` and the weights
  are still identified; what changes is *where* they are informative: only along value
  contrasts that the incoming flow actually contains. A log that records only accepted
  alternatives (selection on the decision) identifies nothing, and the method refuses a log with no
  declines or no accepts.
- **Inconsistent deciders.** Calls that depend on things outside the criteria act as label noise.
  Noise attenuates `β` toward zero; with symmetric noise the direction of the score, and so the
  weight ratios, are roughly kept, but the intervals widen (test 6 measures both).
- **Partners are never averaged.** One fit per decider, or a stated pooling choice (section 1).
- **Weights are identified only up to the information in `v`.** A leaf no decision rates is moved
  only by the prior; a leaf with nearly constant values across the log trades off against `α`;
  leaves whose values move together are identified only in their sum. Real deal flow has
  correlated criteria, so it is less informative per decision than the bench's independent values.
- **Omitted criteria.** If the decider weighs something that is not a leaf, the fitted weights
  describe the best linear stand-in on the leaves that exist, not the decider's full preference.

## 6. Output and verification

**Output in this session.** Samples stay in memory: in the test process, the CLI's printed summary
and the notebook. **No new stored format.** `mcda calibrate choices-synthetic-check` prints planted
values, posterior summaries, interval coverage and diagnostics as JSON and writes nothing.

**Proposal for storing a fitted result (not implemented; needs the owner's approval).** Reuse the
`weight-samples` sidecar with a new `basis: "revealed"`, as `basis: "prior"` was added in #11:

| Field | Content for `basis: "revealed"` |
| --- | --- |
| `decision_log_id`, `decision_log_sha256`, `decision_log_digest_basis` | in place of the reference-set fields, which a revealed basis would refuse |
| `deciders`, `pooled` | the decider id(s) and whether they were pooled |
| `data` | decisions entering, accepts, declines, rated counts per leaf, date range |
| `prior` | as today, plus `base_accept_rate`, `s_ref`, `link: "logit"` and the `β` prior's mean |
| `alpha`, `beta` | the choice model's draws |

It would also need a decision-log input format (versioned, outside the repository, with a decider
id and a date per decision). Both are stored formats and both wait for approval.

**Tests** (fixed seeds; long ones marked `slow`; each can fail):

1. The likelihood against hand-computed values (accept and decline, a missing leaf renormalised,
   extreme `η` stable).
2. With no decisions the posterior is the prior: each free share's marginal passes a KS test
   against its exact Beta, means within 6 standard errors; `α` and `β` keep their priors.
3. **Recovery** on the DD5 two-objective, five-leaf bench at `K ∈ {50, 100, 200, 500, 2000}`
   decisions (four replicates each): the mean absolute error of the posterior-mean global weights
   falls with `K`. The error at `K = 200` is **reported** against the notebook's "about 5%"; it is
   not a pass threshold.
4. **Coverage:** truths drawn from the prior; the 90% intervals of every weight, `α` and `β` cover
   the truth at a rate whose 99% Clopper–Pearson interval contains 0.90.
5. **Simulation-based calibration** for one weight, `α` and `β`: ranks uniform by a chi-square test
   at the 0.01 level (`slow`).
6. **Label noise:** a stated share of decisions flipped. `β` attenuates (its noisy posterior lies
   below the clean fit's) while the weights' noisy point estimates stay within the clean fit's
   intervals; both are reported.
7. **Convergence gate:** a 50-iteration run is labelled not converged.
8. **Read-only:** fitting synthetic decisions against a real example project leaves the project's
   bytes and fingerprint unchanged.
9. **Refusals:** no decisions with both outcomes, a decision with no rated compared leaf, several
   deciders without a pooling choice, a base rate outside (0, 1).
10. **Boundary:** neither the API nor the interface imports `calibration.choices`.

Mutant group `revealed` (`python scripts/mutation_check.py --revealed-only --skip-control`):
at least 10 mutants, all of which must be killed.
