import { Fragment, useState } from "react";
import type { Elicitation, ElicitationCheck, ElicitationPreview, ElicitationGroupCheck, GroupElicitation, Operation, PairwiseJudgement, Project } from "../api/types";
import type { AnalysisContext } from "../state/analysisContext";
import { useElicitationCheck, useElicitationPreview } from "../state/useElicitation";
import { displayFixed } from "./results/numberDisplay";
import { displayShare } from "./results/RobustnessBands";
import { CommitText, Field } from "./ui";

type Conviction = NonNullable<Elicitation["conviction"]>;
const CONVICTIONS: { value: Conviction; label: string }[] = [
  { value: "high", label: "High: ratings could move about ±15%" },
  { value: "medium", label: "Medium: about ±30%" },
  { value: "low", label: "Low: about ±50%" },
];
const STRENGTHS = [9, 8, 7, 6, 5, 4, 3, 2, 1, -2, -3, -4, -5, -6, -7, -8, -9];
const STATUS_TEXT: Record<ElicitationGroupCheck["status"], string> = {
  consistent: "Consistent", check: "Worth a second look", incomplete: "Ratings incomplete",
};

const emptyRecord = (): Elicitation => ({ elicited_by: "", note: "", groups: [] });
const weight = (v: number | null | undefined) => (v === null || v === undefined ? "—" : displayFixed(v, 3));

/** A signed strength for the pair (a, b): s > 0 favours a s times, s < 0 favours b |s| times, 0 is not judged. */
function strengthOf(pairwise: PairwiseJudgement[], a: string, b: string): number {
  const j = pairwise.find((p) => (p.row === a && p.column === b) || (p.row === b && p.column === a));
  if (!j) return 0;
  if (j.value === 1) return 1;
  return j.row === a ? j.value : -j.value;
}

function withStrength(pairwise: PairwiseJudgement[], a: string, b: string, s: number): PairwiseJudgement[] {
  const rest = pairwise.filter((p) => !((p.row === a && p.column === b) || (p.row === b && p.column === a)));
  if (s === 0) return rest;
  return [...rest, s >= 1 ? { row: a, column: b, value: s } : { row: b, column: a, value: -s }];
}

/** Record how a weight profile's judgements were given (who, when, how sure, the order, an
 * optional AHP cross-check and a reason per weight), read the engine's checks, and apply a
 * proposed weight uncertainty. The browser computes no weight, rank or check. */
export function ElicitationPanel({ project, context, busy, apply }: {
  project: Project; context: AnalysisContext; busy: boolean; apply: (ops: Operation[]) => Promise<boolean>;
}) {
  const profile = project.weight_profiles.find((p) => p.id === context.resolvedProfileId);
  const check = useElicitationCheck(project, context);
  const [comparing, setComparing] = useState(false);
  const preview = useElicitationPreview(project, context, comparing);
  if (!profile) return <section className="panel" aria-label="Judgement"><h2>Judgement</h2><p>Add a weight profile first.</p></section>;
  const record = profile.elicitation ?? null;
  const name = (id: string) => project.criteria.find((c) => c.id === id)?.name ?? id;

  const save = (next: Elicitation | null) => apply([{ op: "set_elicitation", profile_id: profile.id, elicitation: next }]);
  const edit = (patch: (r: Elicitation) => void) => {
    const next = structuredClone(record ?? emptyRecord());
    patch(next);
    void save(next);
  };
  const editGroup = (parent: string | null, patch: (g: GroupElicitation) => void) => edit((r) => {
    let group = r.groups.find((g) => (g.parent_id ?? null) === parent);
    if (!group) { group = { parent_id: parent }; r.groups.push(group); }
    patch(group);
  });
  const groupRecord = (parent: string | null) => record?.groups.find((g) => (g.parent_id ?? null) === parent);

  return <section className="panel elicitation-panel" aria-label="Judgement">
    <h2>Judgement: {profile.name}</h2>
    <p className="muted">Record how this profile’s swing ratings were given: who gave them, when, how sure they are, the order they
      put each group in, an optional AHP cross-check and one reason per weight. The engine checks that the answers agree and proposes
      weight uncertainty for SMAA-2 from them. The swing ratings themselves stay in Weights. Switch profile with Partner weighting
      in the header.</p>

    <fieldset className="elicitation-record form-narrow" aria-label="Who and how sure">
      <legend>Who and how sure</legend>
      <Field label="Elicited by"><CommitText label="Elicited by" value={record?.elicited_by ?? ""} disabled={busy}
        onCommit={(v) => edit((r) => { r.elicited_by = v; })}/></Field>
      <Field label="Elicited on" htmlFor="elicited-on"><input id="elicited-on" type="date" aria-label="Elicited on" disabled={busy}
        value={record?.elicited_on ?? ""} onChange={(e) => edit((r) => { r.elicited_on = e.target.value || null; })}/></Field>
      <Field label="Conviction" htmlFor="elicitation-conviction" hint="How far the swing ratings could reasonably move. It becomes rating intervals by a stated rule.">
        <select id="elicitation-conviction" aria-label="Conviction" disabled={busy} value={record?.conviction ?? ""}
          onChange={(e) => edit((r) => { r.conviction = (e.target.value || null) as Conviction | null; })}>
          <option value="">Not stated</option>
          {CONVICTIONS.map((c) => <option key={c.value} value={c.value}>{c.label}</option>)}
        </select></Field>
      <Field label="Note"><CommitText label="Note" multiline value={record?.note ?? ""} disabled={busy}
        onCommit={(v) => edit((r) => { r.note = v; })}/></Field>
      {record && <p><button type="button" disabled={busy} onClick={() => void save(null)}>Clear the record</button></p>}
    </fieldset>

    {check.status === "loading" && <p role="status">Checking the judgement…</p>}
    {check.status === "error" && <><p role="alert">{check.message}</p><button type="button" onClick={check.retry}>Retry check</button></>}
    {check.status === "ready" && <>
      <p role="status" className={`elicitation-status ${check.result.status}`}>Overall: {STATUS_TEXT[check.result.status]}.</p>
      {check.result.groups.map((g) => <GroupEditor key={String(g.parent_id)} group={g} record={groupRecord(g.parent_id ?? null)}
        busy={busy} name={name} editGroup={editGroup}/>)}
      <Proposals check={check.result} busy={busy} apply={apply}/>
    </>}

    {check.status === "ready" && check.result.proposals.length > 0 && <section aria-label="Decision confidence by proposal">
      <h3>What each proposal does to the decision</h3>
      {!comparing ? <button type="button" onClick={() => setComparing(true)}>Compare decision confidence</button> : <>
        {preview.status === "loading" && <p role="status">Running SMAA-2 under each proposal…</p>}
        {preview.status === "error" && <><p role="alert">{preview.message}</p><button type="button" onClick={preview.retry}>Retry</button></>}
        {preview.status === "ready" && <PreviewTable preview={preview.result}/>}
      </>}
    </section>}
    <p className="muted small">{check.status === "ready" ? check.result.label : ""}</p>
  </section>;
}

function GroupEditor({ group, record, busy, name, editGroup }: {
  group: ElicitationGroupCheck; record: GroupElicitation | undefined; busy: boolean; name: (id: string) => string;
  editGroup: (parent: string | null, patch: (g: GroupElicitation) => void) => void;
}) {
  const parent = group.parent_id ?? null;
  const order = group.order;
  const ranked = order.length > 0 ? order : group.children.filter((c) => !c.held).map((c) => c.criterion_id);
  const move = (i: number, by: number) => editGroup(parent, (g) => {
    const next = [...order]; const [item] = next.splice(i, 1);
    if (item !== undefined) next.splice(i + by, 0, item);
    g.order = next;
  });
  const pairs = ranked.flatMap((a, i) => ranked.slice(i + 1).map((b) => [a, b] as const));
  const title = parent === null ? "Top level" : group.name;
  const rows = new Map(group.children.map((c) => [c.criterion_id, c]));
  if (!group.scored) return <fieldset className="elicitation-group" aria-label={`Judgement under ${title}`} data-status="not-scored">
    <legend>{title}</legend>
    <p className="muted small">Not varied: this branch is rated 0 in this profile, so its weights carry no value and no proposal varies them.</p>
  </fieldset>;
  return <fieldset className="elicitation-group" aria-label={`Judgement under ${title}`} data-status={group.status}>
    <legend>{title} <span className={`elicitation-badge ${group.status}`}>{STATUS_TEXT[group.status]}</span></legend>
    {order.length === 0 && group.suggested_order.length > 1 && <p>
      <button type="button" disabled={busy} onClick={() => editGroup(parent, (g) => { g.order = group.suggested_order; })}>
        Rank {title.toLowerCase() === "top level" ? "the top level" : `the criteria under ${title}`}, starting from the swing ratings</button>
    </p>}
    <div className="table-scroll" tabIndex={0}><table className="grid compact" aria-label={`Weights under ${title}`}>
      <thead><tr>
        {order.length > 0 && <th scope="col">Rank</th>}<th scope="col">Criterion</th>
        <th scope="col" className="num">Swing rating</th><th scope="col" className="num">Swing weight</th>
        <th scope="col" className="num">Rank-order weight</th><th scope="col" className="num">AHP weight</th>
        <th scope="col">Reason</th>{order.length > 0 && <th scope="col"><span className="visually-hidden">Move</span></th>}
      </tr></thead>
      <tbody>{[...ranked, ...group.children.filter((c) => !ranked.includes(c.criterion_id)).map((c) => c.criterion_id)].map((id) => {
        const row = rows.get(id);
        const i = order.indexOf(id);
        return <tr key={id} className={row?.held ? "muted" : undefined}>
          {order.length > 0 && <td>{i >= 0 ? i + 1 : row?.held ? "held" : "—"}</td>}
          <th scope="row">{name(id)}</th>
          <td className="num">{row?.rating ?? "not supplied"}</td>
          <td className="num">{weight(row?.swing)}</td>
          <td className="num">{weight(row?.roc)}</td>
          <td className="num">{weight(row?.ahp)}</td>
          <td>{row?.held ? <span className="muted small">Rated 0: held at 0</span>
            : <CommitText label={`Reason for ${name(id)}`} value={record?.reasons?.[id] ?? ""} disabled={busy}
              onCommit={(v) => editGroup(parent, (g) => {
                g.reasons = Object.fromEntries(Object.entries({ ...(g.reasons ?? {}), [id]: v }).filter(([, text]) => text.trim()));
              })}/>}</td>
          {order.length > 0 && <td>{i >= 0 && <>
            <button type="button" disabled={busy || i === 0} aria-label={`Move ${name(id)} up`} onClick={() => move(i, -1)}>↑</button>
            <button type="button" disabled={busy || i === order.length - 1} aria-label={`Move ${name(id)} down`} onClick={() => move(i, 1)}>↓</button>
          </>}</td>}
        </tr>;
      })}</tbody>
    </table></div>
    {group.findings.length > 0 && <ul className="elicitation-findings" aria-label={`Findings under ${title}`}>
      {group.findings.map((f, i) => <li key={i}>{f}</li>)}</ul>}
    {group.ahp?.complete && <p className="small">AHP consistency ratio {displayFixed(group.ahp.consistency_ratio ?? 0, 3)}
      {group.ahp.inconsistent ? " (above 0.10: revisit the answers)" : " (within 0.10)"}
      {group.largest_gap != null && <>; largest gap between swing and AHP weights {displayFixed(group.largest_gap, 3)}</>}.</p>}
    {group.missing_reasons.length > 0 && order.length > 0 && <p className="muted small">No reason yet for {group.missing_reasons.map(name).join(", ")}.</p>}
    {ranked.length >= 2 && <details className="elicitation-ahp">
      <summary>AHP cross-check (optional): {pairs.length} pairs</summary>
      <p className="muted small">Answer each pair the other way: how many times more important is the swing on the left than on the right?
        The engine compares the result with the swing weights.</p>
      {pairs.map(([a, b]) => <label key={`${a}|${b}`} className="class-row">{name(a)} vs {name(b)}{" "}
        <select aria-label={`${name(a)} compared with ${name(b)}`} disabled={busy} value={strengthOf(record?.pairwise ?? [], a, b)}
          onChange={(e) => editGroup(parent, (g) => { g.pairwise = withStrength(g.pairwise ?? [], a, b, Number(e.target.value)); })}>
          <option value={0}>Not answered</option>
          {STRENGTHS.map((s) => <option key={s} value={s}>{s === 1 ? "Equal" : s > 0 ? `${name(a)} ${String(s)}×` : `${name(b)} ${String(-s)}×`}</option>)}
        </select></label>)}
    </details>}
  </fieldset>;
}

function Proposals({ check, busy, apply }: { check: ElicitationCheck; busy: boolean; apply: (ops: Operation[]) => Promise<boolean> }) {
  return <section aria-label="Proposed weight uncertainty">
    <h3>Proposed weight uncertainty</h3>
    {check.proposals.length === 0
      ? <p className="muted">State a conviction or rank a group to get a proposal.</p>
      : <ul className="elicitation-proposals">{check.proposals.map((p) => <li key={p.mode}>
        <strong>{p.label}</strong>
        <p className="small">{p.explanation}</p>
        <button type="button" disabled={busy} onClick={() => void apply([{ op: "set_smaa_model", smaa: p.smaa }])}>
          Use as the weight uncertainty</button>
      </li>)}</ul>}
    <p className="muted small">Using a proposal replaces the weight groups in Uncertainty and keeps the stated measurement ranges. You can undo it.</p>
  </section>;
}

type PreviewMode = ElicitationPreview["previews"][number]["mode"];
const WEIGHT_CHOICES: { mode: PreviewMode; label: string }[] = [
  { mode: "conviction", label: "Conviction bands" },
  { mode: "ranking", label: "Ranking only" },
  { mode: "dirichlet", label: "True conviction (Dirichlet κ)" },
];

function PreviewTable({ preview }: { preview: ElicitationPreview }) {
  const offered = WEIGHT_CHOICES.filter((c) => preview.previews.some((p) => p.mode === c.mode));
  const [chosen, setChosen] = useState<PreviewMode | null>(null);
  const selected = chosen ?? offered[0]?.mode ?? null;
  const available = preview.previews.filter((p) => p.robustness.available);
  const unavailable = preview.previews.filter((p) => !p.robustness.available);
  const first = available[0]?.robustness;
  const description = preview.previews.find((p) => p.mode === selected);
  const dirichlet = preview.previews.find((p) => p.mode === "dirichlet");
  const choice = offered.length > 0 && <fieldset className="weight-uncertainty-choice" aria-label="Weight uncertainty">
    <legend>Weight uncertainty</legend>
    {offered.map((c) => <label key={c.mode}><input type="radio" name="weight-uncertainty" value={c.mode}
      checked={selected === c.mode} onChange={() => setChosen(c.mode)}/> {c.label}</label>)}
    {description && <p className="small" aria-label="Chosen weight uncertainty">{description.label}. {description.robustness.basis}</p>}
    {selected === "dirichlet" && dirichlet && (dirichlet.kappas ?? []).length > 0 &&
      <p className="small" aria-label="Concentration used">κ used: {(dirichlet.kappas ?? []).map((k) => `${k.name} ${k.kappa}`).join(", ")}.
        Each sibling group’s weights are drawn from a Dirichlet around this profile’s ratings, with κ matched to the conviction.</p>}
  </fieldset>;
  if (!first) return <>{choice}<p>{unavailable[0]?.robustness.reason ?? preview.previews[0]?.robustness.reason ?? "Decision confidence is unavailable."}</p></>;
  const best = first.rows[0]?.classes[0]?.name;
  return <>
    {choice}
    {unavailable.map((p) => <p key={p.mode} role="note">{p.label}: {p.robustness.reason}</p>)}
    <div className="table-scroll" tabIndex={0}><table className="grid compact" aria-label="Decision confidence by proposal">
      <thead>
        <tr><th scope="col" rowSpan={2}>Alternative</th>{available.map((p) => <th key={p.mode} scope="colgroup" colSpan={best ? 2 : 1}
          className={p.mode === selected ? "chosen" : undefined} aria-current={p.mode === selected ? "true" : undefined}>{p.label}</th>)}</tr>
        <tr>{available.map((p) => <Fragment key={p.mode}><th scope="col" className="num">First</th>{best && <th scope="col" className="num">{best}</th>}</Fragment>)}</tr>
      </thead>
      <tbody>{first.rows.map((row) => <tr key={row.alternative_id}>
        <th scope="row">{row.name}</th>
        {available.map((p) => {
          const r = p.robustness.rows.find((x) => x.alternative_id === row.alternative_id);
          return <Fragment key={p.mode}><td className="num">{r ? displayShare(r.first.share) : "—"}</td>
            {best && <td className="num">{r?.classes[0] ? displayShare(r.classes[0].share) : "—"}</td>}</Fragment>;
        })}
      </tr>)}</tbody>
    </table></div>
    <details><summary>What varied in each run</summary><ul className="small">{available.map((p) => <li key={p.mode}><strong>{p.label}.</strong> {p.robustness.basis}</li>)}</ul></details>
    <p className="muted small">{preview.label} Shares of draws under each stated uncertainty; not probabilities of success.</p>
  </>;
}
