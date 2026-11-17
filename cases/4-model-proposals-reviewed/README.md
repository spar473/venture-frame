# Case 4: Model suggestions, reviewed by a person

**The question.** A classifier could read a company's public pages and suggest ratings. How can
that help an analyst without the model ever deciding anything?

**Open:** [the worked example](../../examples/external_proposals/README.md) (invented data) ·
[the runner's design](../../docs/design/typed-proposal-runner.md).

> Every company, source, model output and review in the worked example is **invented**. No model
> runs inside the application, and no accuracy is claimed for any model.

## The rules

1. **The application never calls a model.** A separate script, kept outside the application, runs
   the model and writes its output as *proposals* in a published file format.
2. **A proposal is never a rating** until a person accepts it. The analyst sees each suggestion
   beside the current value and its cited source, then accepts, corrects or rejects it, one at a
   time or as an explicitly selected batch. There is no "accept all".
3. **Confidence is information, never a weight.** A model's confidence is shown but never enters
   any score.
4. **The model sees source text only,** never the analyst's ratings or notes.
5. **Accepted proposals go into a separate copy** of the project, with a record of every original
   suggestion, review and source. The original project is untouched, and disagreement stays
   visible after a correction.

## The worked example

Two invented companies, Option A and Option B, with a tool's suggested labels (hardware or
software, early or ready):

- the first suggestion is **corrected**;
- a readiness claim and an annotation are **accepted**;
- a numeric score of 0.91 and a tag are **rejected**;
- a second tool's suggestion and a source assertion are **left pending**, and they appear as open
  questions.

The example README gives the clicks in the app (Evidence → External proposals) and the same steps
from the command line.

## On the real companies

The runner has been tried on case 1: local models (Qwen 2.5 1.5B and Laya) classified the cited
source text for all 112 rubric cells. Those outputs stay outside this repository, and **nothing has
been compared with any rating**. Before any model is scored, a second person must rate the same
cells blind, and the agreement between the two people decides whether a reference exists at all.
That second rating has not happened yet.
