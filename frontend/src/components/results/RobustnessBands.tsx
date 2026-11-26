import type { ScreeningRobustness } from "../../api/types";
import type { AnalysisRequest } from "../../state/useAnalysisRequest";
import { InfoDisclosure } from "../InfoDisclosure";

type Band = ScreeningRobustness["rows"][number];
export type BandsRequest = AnalysisRequest<ScreeningRobustness>;

/** Engine bands indexed for layout: lookups by id only; every range and group is Python's. */
export interface BandLayout {
  result: ScreeningRobustness;
  byId: Map<string, Band>;
  groupOf: Map<string, number>; // index into result.groups
}

export function bandLayout(request: BandsRequest | undefined): BandLayout | null {
  if (request?.status !== "ready" || !request.result.available) return null;
  const result = request.result;
  const groupOf = new Map<string, number>();
  result.groups.forEach((group, index) => { for (const id of group) groupOf.set(id, index); });
  return { result, byId: new Map(result.rows.map((row) => [row.alternative_id, row])), groupOf };
}

/** A share of draws as a whole percent; never 0 % or 100 % for a share that is neither. */
export function displayShare(share: number): string {
  if (share > 0 && share < .005) return "< 1%";
  if (share < 1 && share >= .995) return "> 99%";
  return `${(share * 100).toFixed(0)}%`;
}

function list(names: string[]): string {
  return names.length <= 1 ? names.join("") : `${names.slice(0, -1).join(", ")} and ${names[names.length - 1] ?? ""}`;
}

/** One lead sentence, chosen by template from engine fields: who is MAVT's rank 1, its rank-1
 * share, and the engine's groups in their own order. Nothing is ranked, counted or compared here. */
export function leadSentence(layout: BandLayout, names: Record<string, string>): string {
  const { result, groupOf } = layout;
  const name = (id: string) => names[id] ?? id;
  const leaders = result.rows.filter((row) => row.mavt_rank === 1);
  const parts: string[] = [];
  const leaderGroups = new Set(leaders.map((row) => groupOf.get(row.alternative_id)));
  const [leader] = leaders;
  if (leaders.length > 1) parts.push(`${list(leaders.map((row) => name(row.alternative_id)))} share first place.`);
  else if (leader) {
    const group = result.groups[groupOf.get(leader.alternative_id) ?? -1] ?? [];
    const others = group.filter((id) => id !== leader.alternative_id).map(name);
    parts.push(`${name(leader.alternative_id)} is first in ${displayShare(leader.rank1_share)} of draws${others.length ? `, but its rank overlaps with ${list(others)}` : ""}.`);
  }
  const overlapping = result.groups.filter((group, index) => group.length > 1 && !leaderGroups.has(index));
  const [first, ...more] = overlapping;
  if (!first) parts.push(leaders.length ? "Every other rank is separated under the stated uncertainty." : "Every rank is separated under the stated uncertainty.");
  else {
    parts.push(`${list(first.map(name))} overlap in rank under the stated uncertainty${more.length ? `, and ${String(more.length)} more group${more.length > 1 ? "s" : ""} overlap too (see the table)` : ""}.`);
  }
  return parts.join(" ");
}

/** How one row's Rank cell joins its group's marker. */
export interface GroupMarker { group: number; start: boolean; end: boolean; label: string | null }

/** Layout only: where each multi-member group's marker starts and ends in the table's row order.
 * Adjacent members share one marker labelled "Too close to separate"; when a group's members are
 * not all adjacent, each run's label names the other members, so a marker never implies more. */
export function groupMarkers(order: readonly string[], layout: BandLayout | null, names: Record<string, string>): Map<string, GroupMarker> {
  const markers = new Map<string, GroupMarker>();
  if (!layout) return markers;
  order.forEach((id, index) => {
    const group = layout.groupOf.get(id);
    const members = group === undefined ? [] : layout.result.groups[group] ?? [];
    if (group === undefined || members.length < 2) return;
    const start = layout.groupOf.get(order[index - 1] ?? "") !== group;
    const end = layout.groupOf.get(order[index + 1] ?? "") !== group;
    let label: string | null = null;
    if (start) {
      let run = 0;
      while (layout.groupOf.get(order[index + run] ?? "") === group) run += 1;
      label = run === members.length ? "Too close to separate"
        : `Too close to separate (with ${members.filter((member) => member !== id).map((member) => names[member] ?? member).join(", ")})`;
    }
    markers.set(id, { group, start, end, label });
  });
  return markers;
}

/** The Rank cell: the engine's rank, its range once bands arrive, and the group marker. */
export function RankCell({ id, rank, layout, marker }: { id: string; rank: number; layout: BandLayout | null; marker?: GroupMarker }) {
  const band = layout?.byId.get(id);
  if (!layout || !band) return <td className="rank-cell">{rank}</td>;
  const position = marker ? ` in-group${marker.start ? " group-start" : ""}${marker.end ? " group-end" : ""}` : "";
  return <td className={`rank-cell${position}`}>
    <span className="rank-value">{rank}</span> <span className="rank-range">({band.rank_low}–{band.rank_high})</span>
    {marker && <span className="sr-only">Group {marker.group + 1} of {layout.result.groups.length}: too close to separate</span>}
    {marker?.label && <span className="band-label" aria-hidden="true">{marker.label}</span>}
  </td>;
}

/** Above the table: the lead sentence and how the ranges were found, or why there are none. */
export function BandsLead({ request, layout, names }: { request: BandsRequest | undefined; layout: BandLayout | null; names: Record<string, string> }) {
  if (!request) return null;
  if (request.status === "loading") return <p className="summary-lead pending" aria-hidden="true"/>;
  if (request.status === "error") return <p className="summary-lead muted small" role="status">Rank ranges could not be loaded. <button type="button" className="link" onClick={request.retry}>Retry rank ranges</button></p>;
  if (!layout) return <p className="summary-lead muted small">Rank ranges unavailable. <InfoDisclosure label="Why there are no rank ranges">{request.result.reason}</InfoDisclosure></p>;
  // A no-break space keeps the ⓘ with the sentence's last word.
  return <p className="summary-lead" aria-live="polite">{leadSentence(layout, names)}{" "}
    <InfoDisclosure label="How rank ranges are found">Each rank range is the shortest run of ranks that holds at least {displayShare(layout.result.coverage)} of
      the draws; alternatives whose ranges overlap are too close to separate. {layout.result.basis} {layout.result.label}</InfoDisclosure></p>;
}

type Share = Band["first"];

/** A share of draws with the engine's interval, widened outward to whole percents for display. */
export function shareText(estimate: Share): { share: string; interval: string | null } {
  const interval = estimate.lower != null && estimate.upper != null
    ? `(${String(Math.floor(estimate.lower * 100))}–${String(Math.ceil(estimate.upper * 100))}%)` : null;
  return { share: displayShare(estimate.share), interval };
}

function ShareCell({ estimate }: { estimate: Share }) {
  const { share, interval } = shareText(estimate);
  return <td className="num">{share}{interval && <> <span className="rank-range">{interval}</span></>}</td>;
}

/** Decision confidence: the engine's shares of draws in which each alternative is first, is in the
 * top ranks and falls in each class, from the same run as the rank ranges. Display only. */
export function DecisionConfidence({ layout, names }: { layout: BandLayout | null; names: Record<string, string> }) {
  if (!layout || !layout.result.rows.length) return null;
  const { result } = layout;
  const classes = result.rows[0]?.classes ?? [];
  const how = result.confidence != null
    ? `Brackets give the exact ${displayShare(result.confidence)} interval for each share, widened to whole percents.`
    : "Exact enumeration: the shares are probabilities under the stated uncertainty.";
  return <section className="decision-confidence" aria-label="Decision confidence">
    <p className="decision-confidence-title"><strong>Decision confidence</strong>{" "}
      <InfoDisclosure label="About decision confidence">{result.confidence_label} {how} {result.basis}</InfoDisclosure></p>
    <div className="table-scroll" tabIndex={0}>
      <table className="grid" aria-label="Decision confidence">
        <thead><tr>
          <th scope="col">Alternative</th><th scope="col" className="num">First</th><th scope="col" className="num">Top {String(result.top_k ?? "")}</th>
          {classes.map((c) => <th scope="col" className="num" key={c.class_id}>{c.name}</th>)}
        </tr></thead>
        <tbody>{result.rows.map((row) => <tr key={row.alternative_id}>
          <th scope="row">{names[row.alternative_id] ?? row.name}</th>
          <ShareCell estimate={row.first}/><ShareCell estimate={row.top_ranks}/>
          {row.classes.map((c) => <ShareCell estimate={c} key={c.class_id}/>)}
        </tr>)}</tbody>
      </table>
    </div>
    <p className="muted small">Share of draws under the uncertainty stated in the project; not probabilities of success.</p>
  </section>;
}
