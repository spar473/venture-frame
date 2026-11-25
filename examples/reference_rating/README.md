# Reference rating: reviewed ratings for the VEE cohort

> **Real companies, public sources.** Everything this example produces is about real New Zealand
> companies. It is **never** committed here: write it to `$MCDA_DATA`, and `pilot.py` refuses
> paths inside the repository. Nothing in it is investment advice or a prediction.

The calibration ([docs/design/calibration.md](../../docs/design/calibration.md)) refuses to fit the
VEE cohort, because none of its ratings is reviewed and it rates only two zero-weight mandate gates.
This example is the path to reviewed ratings: a proposer suggests labels from dated public sources,
a person reviews every one, and the engine imports only what that person accepted. The design is in
[docs/design/reference-rating.md](../../docs/design/reference-rating.md).

## 1. Cases, packets and the review page (no outcome information)

```sh
python examples/reference_rating/pilot.py sample \
  --reference-cases "$MCDA_DATA/vee-reference-cases.json" --size 12 --seed 20261126 \
  --out "$MCDA_DATA/dd6/pilot-cases.json"
python examples/reference_rating/pilot.py rubric-text \
  --rubrics examples/nz_venture_screening/rubrics.json --out "$MCDA_DATA/dd6/rubric-text.json"
python examples/reference_rating/pilot.py packets \
  --reference-cases "$MCDA_DATA/vee-reference-cases.json" --audit "$MCDA_DATA/vee-cohort-audit.json" \
  --project examples/nz_venture_screening/project.json \
  --rubrics examples/nz_venture_screening/rubrics.json \
  --cases "$MCDA_DATA/dd6/pilot-cases.json" --leaves team --out "$MCDA_DATA/dd6/packets"
```

- `sample` draws from the sorted case ids alone, so it cannot select by outcome. Keep the cases it
  draws; a case with no usable source is rated "not enough evidence", not swapped.
- A packet holds the case's snapshot date, the leaves to rate with their labels and rubric, the
  entry round, date, sector and investor as VEE recorded them, and source pointers that predate the
  snapshot. It holds **no** outcome, follow-up date, VEE outcome status or VEE commentary, and a
  source dated after the snapshot is withheld because it shows the company still existed later.
- `--leaves` picks the objective: `team` first (the highest mean weight over the two NZ profiles),
  then `problem_solution`, `traction`, `market`. Mandate leaves are never rated here.

## 2. Proposals (outside the application)

A proposer writes a `rating-proposals` version 1 file (schema: `RatingProposals` in
`src/mcda_studio/domain/rating_proposals.py`). Each proposal names the case and leaf, one label or
`not_enough_evidence: true`, a short claim in the proposer's words, the source URL, its date
(`YYYY`, `YYYY-Qn`, `YYYY-MM` or `YYYY-MM-DD`), a locator, an excerpt of at most 25 words and a
confidence. Only public sources dated on or before the snapshot: archived company pages (Wayback
Machine captures at or before the snapshot), dated news and press releases, grant notices. No paid
databases, sign-ins or paywalls.

## 3. The review sheet

```sh
mcda reference-cases review-sheet "$MCDA_DATA/dd6/proposals.json" \
  --reference-cases "$MCDA_DATA/vee-reference-cases.json" \
  --project examples/nz_venture_screening/project.json \
  --rubric-text "$MCDA_DATA/dd6/rubric-text.json" --out-dir "$MCDA_DATA/dd6/review"
```

This writes `review.csv` (UTF-8 with a byte-order mark, so Excel opens it correctly on a Korean or
any other locale) and `review.html`. **How to review:**

1. Open `review.csv` in Excel and `review.html` in a browser beside it.
2. For each row, open the source link, check the excerpt is there and the date is on or before the
   snapshot, and compare with the leaf's labels and rubric on the page.
3. Set `decision` to `accept` (the proposed label stands), `edit` (type the right label, exactly as
   listed, in `final_label`), `reject` (the proposal is wrong; the leaf stays unrated) or leave
   `skip` (not reviewed yet). Add a `reviewer_note` if useful. Do not change other columns.
4. Save as **CSV UTF-8 (Comma delimited)**. If Excel saved it in another encoding, pass
   `--sheet-encoding cp949` (or whatever it used) to the import.

## 4. Import, validate, coverage

```sh
mcda reference-cases import-reviews "$MCDA_DATA/dd6/review/review.csv" \
  --proposals "$MCDA_DATA/dd6/proposals.json" \
  --reference-cases "$MCDA_DATA/vee-reference-cases.json" \
  --project examples/nz_venture_screening/project.json \
  --reviewer "Your Name" --reviewed-on YYYY-MM-DD \
  --out "$MCDA_DATA/dd6/vee-reference-cases-reviewed.json"
mcda reference-cases validate "$MCDA_DATA/dd6/vee-reference-cases-reviewed.json" \
  --project examples/nz_venture_screening/project.json
mcda reference-cases outcomes "$MCDA_DATA/dd6/vee-reference-cases-reviewed.json" \
  --project examples/nz_venture_screening/project.json
```

Only `accept` and `edit` rows become ratings, with `review: "reviewed"` and a note naming you and
the date. Any problem (an edit to a label that is not listed, a source dated after the snapshot, a
proposal without a source or excerpt, a sheet whose proposals changed) refuses the whole import and
nothing is written. Importing the same sheet again changes nothing; an existing reviewed rating is
never changed without `--replace`.
