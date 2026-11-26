"""Calibration: weight samples that fit a reference set's recorded outcomes under a stated model.

Offline and opt-in, run from the command line only. Neither the API nor the interface imports this
package (a test enforces it). It reads a project, a weight profile and a version 2 reference set and
never changes any of them. Design: ``docs/design/calibration.md``.

Modules:

- ``prior``: the nested Dirichlet centred on a profile's stated weights, its unconstrained
  (additive log-ratio) parametrisation and the moment-matched concentration.
- ``likelihood``: the censoring-aware constant-hazard outcome model.
- ``sampler``: adaptive random-walk Metropolis within Gibbs blocks, several chains.
- ``diagnostics``: rank-normalised split-R-hat and bulk effective sample size.
- ``synthetic``: planted weights and simulated outcomes for method validation.
- ``weight_samples``: the ``weight-samples`` sidecar, version 1.
- ``service``: ``fit``, from a project and a reference set to a report and, when the run converges,
  a sidecar.
"""
