# Venture Frame: agent rules

Venture Frame is a decision-support app for early-stage venture screening, built on the
domain-neutral `mcda_studio` engine (CLI `mcda`). Python engine, thin FastAPI layer, React UI.

Never broken:
1. Python is the only source of maths. React never computes values, ranks, classes or groups.
2. Store judgements, not results. Looking never changes the project. Missing is not zero.
3. Ask the owner before changing method maths, the stored schema, reference fixtures, pinned
   digests or historical outputs.
4. Never weaken a test, oracle, tolerance, mutant or timeout to get a pass.
5. The engine stays domain-neutral. Venture content lives in examples, product configuration
   and wording.
6. The app never calls a model. Model output is a proposal until a person reviews it.
7. Real data stays outside the repository (in `$MCDA_DATA`).
8. Stay free: no accounts, trials, keys or paid services without the owner's say-so.
9. Work on a short-lived feature branch and merge into `main` by pull request. Never push, tag
   or release unless the owner says so in that session. Commit email ends in
   `@users.noreply.github.com`.
10. Report only what was run in the session, and say what was not verified.

Supported platforms: macOS, Linux and native Windows (Python 3.12, Node 20.19+).
