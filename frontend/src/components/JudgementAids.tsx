import type { JudgementAids as Aids, ReferenceComparison } from "../api/types";

const VERDICT: Record<string, string> = {
  agrees: "Your order agrees", disagrees: "Your order disagrees", tie: "Your ratings tie", "not comparable": "Not comparable here",
};

/** Swings anchored in the project's own alternatives, break-even weights and reference profiles,
 * as the engine words them. The browser computes nothing: every sentence and verdict comes from
 * the engine. A reference profile is shown beside the partner's order, never used as a default. */
export function JudgementAidsView({ aids }: { aids: Aids }) {
  if (!aids.available) return <section aria-label="Judgement aids"><h3>Aids for the weights</h3><p className="muted">{aids.reason}</p></section>;
  const be = aids.break_even;
  return <section aria-label="Judgement aids" className="judgement-aids">
    {be && <section aria-label="Break-even weights">
      <h3>Break-even weights</h3>
      <p className="muted small">{be.rule}</p>
      <ul aria-label="Where the leader changes">{be.criteria.map((c) => <li key={c.criterion_id}>{c.sentence}.</li>)}</ul>
      <p aria-label="Nearest change">{be.nearest_sentence}.</p>
    </section>}
    <section aria-label="Swings anchored in your ventures">
      <h3>Swings anchored in your ventures</h3>
      <p className="muted small">A swing weight is about the move from a criterion’s worst level to its best. These are the worst and best
        ventures in this project on each criterion, to picture that move.</p>
      <ul aria-label="Anchored swings">{aids.anchors.map((a) => <li key={a.criterion_id}>{a.swing}.</li>)}</ul>
    </section>
    {aids.references.map((r) => <ReferenceView key={r.reference_id} reference={r}/>)}
    <p className="muted small">{aids.label}</p>
  </section>;
}

function ReferenceView({ reference }: { reference: ReferenceComparison }) {
  return <section aria-label={`Reference profile: ${reference.title}`}>
    <h3>Beside the literature: {reference.title}</h3>
    <p className="muted small">{reference.purpose} {reference.criteria_note}</p>
    <p aria-label="Your order">Your order: {reference.partner_order_names.join(" > ")}</p>
    {reference.variants.map((v) => <div key={v.id} className="reference-variant">
      <h4>{v.name}{v.disagreements > 0 ? ` (${v.disagreements} disagreement${v.disagreements === 1 ? "" : "s"})` : ""}</h4>
      <p className="small">{v.summary}</p>
      <ul aria-label={`Statements: ${v.name}`}>{v.statements.map((s, i) => <li key={i}>
        {s.horizon ? <strong>{s.horizon}: </strong> : null}{s.text}
        <ul>{s.pairs.map((p) => <li key={`${p.above}-${p.below}`} className={p.partner === "disagrees" ? "disagrees" : undefined}>
          {p.above_name} above {p.below_name}: {VERDICT[p.partner]}</li>)}</ul>
      </li>)}</ul>
      <p className="muted small">Source: {v.citations.map((c) => <span key={c.url}>{c.authors} ({c.year}){c.title ? `, “${c.title}”` : ""}, {c.where}. <a href={c.url} target="_blank" rel="noopener noreferrer">{c.url}</a> </span>)}</p>
    </div>)}
  </section>;
}
