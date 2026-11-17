# Venture screening: a six-minute demo

A click path through the [venture screening study](../examples/nz_venture_screening/README.md) for
a coffee conversation with an investor: what to click, what to say, and where the honest limits
are. Numbers are the study's as built and re-checked on 7 November 2026 in the default Analyst view;
if the study is rebuilt they may move.

**Before you start.** Run `mcda serve` (or `python -m mcda_studio serve`), open the start page at
1440 × 900 or larger, and close other projects. Have one line ready: *"This is a screening queue
for a hypothetical seed/Series A fund, built from public information on eight real New Zealand
companies. It is not investment advice and it does not predict success."*

**Two views.** **Open worked example** opens the study in **Analyst view**: results, sensitivity, comparison,
evidence, questions and the decision record, with nothing to edit by accident. **Builder view**
(top right) adds the model editors. Everything below is read-only: looking never changes the saved
study, its fingerprint or the undo history, and the what-if trial in step 6:15 is never saved and
cannot save a profile.

## The click path

| Time | Click | Say |
| --- | --- | --- |
| 0:00 | Start page → **Venture screening** card → **Open worked example — Venture screening** | Eight real NZ startups across six sectors, as of 4 November 2026. Every rating is a judgement against a written rubric, and every fact behind it is a sourced claim. Companies were not contacted. |
| 0:30 | **Results** (it opens on the screening summary) | One row per company: rank, class, whether it meets the mandate, what drives its value, how close it is to changing place, and how many questions are open. The value is this partner's preference, on a 0–1 scale, not a probability. Optional: **Download screening report** saves one self-contained HTML file (it prints to A4) that carries the whole reading, with the study's own description and statements at the top, for a partner who has not installed anything. |
| 1:15 | Click **Upstock** | **Why the leader leads:** the company detail shows its requirements, every scored contribution (largest first: product readiness 0.0920, then market size 0.0648) and, under *Ratings and sources*, the rubric level behind each rating. It is the only company in *Prioritise for diligence* under the Technical/IP-led profile. Close it with Escape. |
| 2:00 | Upstock's **Closest change** cell, or **Show in Sensitivity** in its detail; then **Sensitivity** | **How fragile:** adding 0.0615 of the total weight to problem severity would drop Upstock to *Monitor*. Elsewhere in the queue, neighbours swap for 0.01–0.04 of the weight: the middle of the queue is close. Sensitivity shows every exact crossover; nothing is sampled. |
| 2:45 | **Weight profile** → **Commercial/traction-led**, then **Results** | **Partners disagree:** under the traction-led partner, Hectre jumps to second and sits 0.0009 below *Prioritise*; a 0.0010 change of weight would promote it. Upstock's margin widens from 0.0414 to 0.1028. Two profiles, two readings, nothing averaged. |
| 3:15 | Tick **Compare partners** (then untick it) | **Both partners at once:** the two profiles side by side, each its own ranking and classes, with the places each company moves. There is no combined ranking. Click a company on the right to open its detail under the right-hand profile. |
| 3:30 | Hnry's row ("Fails: Stage at most Series A") | **Mandate beside the ranking:** Hnry is fourth for this partner (seventh under the Technical/IP-led profile) but is Series C, outside a seed/Series A mandate. The failure is shown beside the ranking and in the feasibility table below; no score can offset it, and Hnry is never hidden. |
| 3:45 | **Weight profile** → back to the first profile, **Technical/IP-led** | The remaining numbers use the Technical/IP-led profile again. |
| 4:00 | **Method** → **SMAA-2** | **Stated uncertainty:** market estimates differ between publishers by up to three orders of magnitude, and judgement-only ratings vary by one level. Under those stated assumptions Upstock is first in 93.0% of draws. That is acceptability under assumptions, not a chance of success. |
| 4:30 | **Compare** | **Cross-check:** PROMETHEE II, which does not let a strong criterion fully offset a weak one, also puts Upstock first. AHP is plainly *not set up*: nobody made the 448 pairwise judgements it would need, and we do not invent them. Agreement between methods that share inputs is not independent validation. |
| 5:00 | **Evidence** → **Open the study's evidence**, then **Open Questions** | **Open diligence questions:** 154 claims; two disputes left open (Kea Aerospace's stage, Scentian Bio's market size) and 33 claims older than a year, 12 of them behind Upstock, more than any other company. Its first diligence task is to refresh that evidence. Without the dossier, the base model has 65 questions under Technical/IP-led or 64 under Commercial/traction-led; with it, 104 or 105. Deliberate zero-weight mandate changes are excluded. The list is grouped by company (then a *Model-wide* group); within a company, disputes come first, then outdated claims, then weight questions, then the rest. Optional: tick **Sort by decision impact** (off by default) to put first the cells where trying each other rubric level, one cell at a time, would change a rank, the leader or a class; each says why the cell is open (a judgement alone, disputed, or out of date). Untick it to return to the list above. |
| 5:45 | **Decision** → **Open a decision record** → `examples/nz_venture_screening/decision.json` | **Decision record:** an *illustrative* record selecting Upstock for the next slot, with its rationale, unresolved assumptions and review triggers, sealed with the exact analyses it relied on. Revising it makes a new linked record; the old one never changes. |
| 6:15 | Optional: **Results** → **What-if weights** → drag **Team** to 0 | **A temporary trial:** the four top-level raw ratings become sliders. Under the Technical/IP-led profile, with Team at zero, Wych moves ahead of Upstock (0.6607 against 0.6443). Only the summary and ranking chart re-rank, using the engine's weights, and the saved tables below say they use the saved weights. Counts of open questions are hidden because they reflect the saved weights. **Reset** returns the saved reading. |
| 6:45 | Optional: **Builder view** → **Model** → **Present** | The same summary above the value tree, for discussing weights with a partner. The Model and Present tabs exist only in Builder view. |

**Reset before you leave Results.** While a what-if trial is active the company detail,
**Sensitivity** and **Compare partners** each show *Reset what-if to use this*, and every download
(analysis JSON or CSV, SVG, factor-run records) shows *Reset what-if to download the saved model*.
This is deliberate: those views and files always describe the saved model. Reset (or choosing a
different profile, scenario or revision) discards the trial.

## Looking back: five first rounds

A two-minute add-on using the
[first-round retrospective](../examples/nz_first_round_retrospective/README.md). It asks what the
public evidence could support around the first rounds of Vertus Energy, Croptide, RosterLab,
Argo Navis Aerospace and Helico, and sets the method's ranking beside what happened next. It
demonstrates how the method behaves on old cases, not predictive accuracy.

| Click | Say |
| --- | --- |
| Start page → **Examples…** → **Add to my projects** on *Looking back: five NZ first rounds (public information)* → **Analyst view** | Five past companies, each with its own first-round anchor. Sources range from 12 months before that anchor through 180 days after it, plus earlier own-site captures. Later-in-window progress is not an announcement-day fact. Helico's first round was labelled seed; Argo's exact first-round date is unconfirmed. |
| **Results**, then switch between **Technical/IP-led** and **Commercial/traction-led** | Both profiles give the same order: Vertus Energy, Croptide, Argo Navis Aerospace, RosterLab, Helico. All five are in the lowest class: the classes were set for a seed/Series A fund's current pipeline, and these are first-round snapshots. Croptide and Argo Navis are almost tied. The venture study's criteria, profiles, classes and mandate are unchanged. |
| **Evidence** → open `examples/nz_first_round_retrospective/evidence.json` | One unreviewed claim for each of the 95 cells, with a dated source and the rubric step. 39 claims say they follow the owner's rule: where the sources were silent, the lowest "nothing public" label, as in the venture study. Argo Navis has seven of them, so its thin record pulls it down. The market estimates were available around each company's own date; some segments are broad. |
| **Reference cases** → open `examples/nz_first_round_retrospective/reference-cases.json` → select **Vertus Energy** → tick **Leave out this alternative's own case** | First Vertus matches its own case exactly, a consistency check. With its own case left out it is compared with the other four on all 14 labelled criteria. Numeric market criteria are not part of this similarity. |
| **Show what happened next** | Outcomes are off by default and appear only on request. They are dated public facts, kept out of every input, and say nothing about the people involved. In the README's results table: the top two raised larger rounds, the 3rd and 5th were liquidated and the 4th raised. That is five cases, not evidence either way. |

**Honesty line:** the ratings were drafted by Claude Code, checked and extended by Codex, and
completed by Claude Code under the owner's gap rule, all with the outcomes known; author review is
pending. The gap rule penalises thin public records. Source-date checks cannot remove hindsight bias.
Public sources under-report companies that closed. Five selected companies cannot show whether
the method predicts anything. The next research step is a blind re-rating packet; it has not been
produced. Review the run report (historical record, not included) for every gap,
source qualification and owner decision. The previous two-current-company draft was deleted at
the owner's request.

## Honest limits to state

- **Screening, not prediction.** The model ranks companies for diligence under stated preferences.
  It says nothing about returns or the chance of success, and it has not been validated against
  what later happened to these companies.
- **Public information only.** Ratings rest on what the public record showed on 4 November 2026,
  much of it company-published; a quiet company scores lower than a loud one with the same
  substance. 71 of the 112 rubric ratings are the author's judgements.
- **Market sizes are contested.** Each is the median of publishers' public estimates for the
  narrowest matching segment; the segment choice is itself a judgement.
- **One weight at a time.** Sensitivity and the closest change vary a single weight; they are not
  robustness over every combination of weights.
- **A trial is not evidence.** What-if varies raw swing ratings only. It never projects evidence,
  never changes a rating or source, and cannot be saved as a profile.
- **Hypothetical fund.** The mandate, classes and partner profiles are illustrations, not any
  firm's process.

## If something goes wrong

- *The card asks for a title:* that is **Create my project** (a fresh copy). Use **Open worked
  example** for the study as shipped.
- *Evidence offers no "Open the study's evidence" button:* the project was edited, so it is no
  longer exactly the shipped study; open `examples/nz_venture_screening/evidence.json` with the file
  picker, or reopen the worked example.
- *SMAA-2 numbers differ slightly from these:* the screen uses the default seed and run; the
  decision record keeps its own seed (2026) and environment.
- *A control says "Reset what-if to use this" or a download is disabled:* a what-if trial is
  active; use **Reset** under **What-if weights** in Results.
- *There is no Model or Present tab:* switch to **Builder view**.
- *The study opened in Builder view:* only the gallery's **Open worked example** opens it in Analyst
  view automatically. Reopening it from the Projects table, or after you chose a view in the same
  session, uses that choice; press **Analyst view**.
