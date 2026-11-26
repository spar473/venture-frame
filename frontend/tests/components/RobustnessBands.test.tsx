import { act, fireEvent, render, renderHook, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../../src/api/client";
import type { Project, ScreeningRobustness, ScreeningSummary } from "../../src/api/types";
import { bandLayout, DecisionConfidence, displayShare, groupMarkers, leadSentence, shareText } from "../../src/components/results/RobustnessBands";
import { ScreeningSummaryView } from "../../src/components/results/ScreeningSummaryView";
import { BASE_SELECTION, resolveContext } from "../../src/state/analysisContext";
import { useScreeningRobustness } from "../../src/state/useScreeningRobustness";
import projectJSON from "../fixtures/screening-study-project.json";
import summaryJSON from "../fixtures/screening-summary.json";

const project = projectJSON as unknown as Project;
const summary = summaryJSON as unknown as ScreeningSummary;
const names = Object.fromEntries(project.alternatives.map((a) => [a.id, a.name]));
const order = summary.rows.map((r) => r.alternative_id);

// Engine output shape for the fixture's eight companies (Commercial/traction-led, default run).
const RANGES: [number, number][] = [[1, 1], [2, 3], [2, 4], [3, 6], [3, 6], [5, 7], [5, 7], [8, 8]];
function robustness(patch: Partial<ScreeningRobustness> = {}): ScreeningRobustness {
  return {
    project_fingerprint: "rev", configuration: summary.configuration, weight_profile_id: summary.weight_profile_id,
    scenario_id: summary.scenario_id, available: true, reason: null, issue_codes: [], coverage: 0.8, run: "combined",
    evaluation: "sampled", draws: 10000, seed: 0, parameters: "defaults",
    rows: summary.rows.map((row, i) => ({ alternative_id: row.alternative_id, name: row.name, mavt_rank: row.rank,
      rank_low: RANGES[i]?.[0] ?? row.rank, rank_high: RANGES[i]?.[1] ?? row.rank, band_share: 0.85, rank1_share: i === 0 ? 0.9247 : 0.01,
      first: { share: i === 0 ? 0.9247 : 0.01, lower: i === 0 ? 0.919 : 0.008, upper: i === 0 ? 0.9298 : 0.0122 },
      top_ranks: { share: i < 3 ? 0.9991 : 0.004, lower: i < 3 ? 0.9983 : 0.003, upper: i < 3 ? 0.9996 : 0.0053 },
      classes: [
        { class_id: "prioritise", name: "Prioritise for diligence", share: i === 0 ? 0.9744 : 0, lower: i === 0 ? 0.9711 : 0, upper: i === 0 ? 0.9774 : 0.0004 },
        { class_id: "monitor", name: "Monitor", share: i === 0 ? 0.0256 : 1, lower: i === 0 ? 0.0226 : 0.9996, upper: i === 0 ? 0.0289 : 1 },
      ] })),
    top_k: 3, confidence: 0.95, confidence_label: "Decision confidence is the share of draws; not a probability of success.",
    groups: [[order[0] ?? ""], order.slice(1, 7), [order[7] ?? ""]],
    basis: "Varied: 87 stated measurement ranges.", label: "Rank ranges are not probabilities.", ...patch,
  };
}
const ready = (result: ScreeningRobustness) => ({ status: "ready" as const, result });
const layoutOf = (result: ScreeningRobustness) => {
  const layout = bandLayout(ready(result));
  if (!layout) throw new Error("layout");
  return layout;
};

describe("the lead sentence", () => {
  it("names the leader's share and the engine's overlapping group in its own order", () => {
    expect(leadSentence(layoutOf(robustness()), names)).toBe(
      "Upstock is first in 92% of draws. Hectre, Wych, Hnry, Wellumio, Captivate Technology and Scentian Bio overlap in rank under the stated uncertainty.");
  });
  it("says when the leader itself overlaps, and when every rank is separated", () => {
    const joined = robustness({ groups: [order.slice(0, 2), order.slice(2, 3), ...order.slice(3).map((id) => [id])] });
    expect(leadSentence(layoutOf(joined), names)).toBe("Upstock is first in 92% of draws, but its rank overlaps with Hectre. Every other rank is separated under the stated uncertainty.");
    const more = robustness({ groups: [[order[0] ?? ""], order.slice(1, 3), order.slice(3, 5), order.slice(5, 7), [order[7] ?? ""]] });
    expect(leadSentence(layoutOf(more), names)).toBe("Upstock is first in 92% of draws. Hectre and Wych overlap in rank under the stated uncertainty, and 2 more groups overlap too (see the table).");
  });
  it("never shows a share as 0 % or 100 % unless it is", () => {
    expect([displayShare(0), displayShare(0.004), displayShare(0.9302), displayShare(0.996), displayShare(1)]).toEqual(["0%", "< 1%", "93%", "> 99%", "100%"]);
  });
});

describe("group markers", () => {
  it("join adjacent members under one label", () => {
    const markers = groupMarkers(order, layoutOf(robustness()), names);
    expect(markers.get(order[0] ?? "")).toBeUndefined();
    expect(markers.get(order[1] ?? "")).toEqual({ group: 1, start: true, end: false, label: "Too close to separate" });
    expect(markers.get(order[6] ?? "")).toMatchObject({ start: false, end: true, label: null });
  });
  it("name the other members when a group is split by another row", () => {
    const split = robustness({ groups: [[order[0] ?? ""], [order[1] ?? "", order[3] ?? ""], [order[2] ?? ""], ...order.slice(4).map((id) => [id])] });
    const markers = groupMarkers(order, layoutOf(split), names);
    expect(markers.get(order[1] ?? "")?.label).toBe("Too close to separate (with Hnry)");
    expect(markers.get(order[3] ?? "")?.label).toBe("Too close to separate (with Hectre)");
  });
});

describe("bands in the summary table", () => {
  const rankCells = () => within(screen.getByRole("table", { name: "Screening summary" })).getAllByRole("row").slice(1)
    .map((row) => within(row).getAllByRole("cell")[0]?.textContent);

  it("shows each rank with its range, the group in words for screen readers, and the lead", () => {
    render(<ScreeningSummaryView summary={summary} project={project} bands={ready(robustness())}/>);
    expect(screen.getByRole("columnheader", { name: "Rank (range)" })).toBeInTheDocument();
    expect(rankCells()[0]).toBe("1 (1–1)");
    expect(rankCells()[1]).toBe("2 (2–3)Group 2 of 3: too close to separateToo close to separate");
    expect(screen.getAllByText("Group 2 of 3: too close to separate")).toHaveLength(6);
    expect(screen.getByText(/^Upstock is first in 92% of draws\./)).toBeInTheDocument();
  });
  it("shows the rank alone while bands load", () => {
    render(<ScreeningSummaryView summary={summary} project={project} bands={{ status: "loading" }}/>);
    expect(screen.getByRole("columnheader", { name: "Rank" })).toBeInTheDocument();
    expect(rankCells()[1]).toBe("2");
  });
  it("gives the reason when there are no bands, and invents none", () => {
    render(<ScreeningSummaryView summary={summary} project={project}
      bands={ready(robustness({ available: false, reason: "No stated uncertainty in this project.", rows: [], groups: [] }))}/>);
    expect(rankCells()[1]).toBe("2");
    fireEvent.click(screen.getByRole("button", { name: "Why there are no rank ranges" }));
    expect(screen.getByText("No stated uncertainty in this project.")).toBeInTheDocument();
  });
});

describe("the rank-range request", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => { vi.restoreAllMocks(); vi.useRealTimers(); });
  const context = (profileId: string) => resolveContext(project, "rev", { ...BASE_SELECTION, profileId });
  const start = () => act(async () => { await vi.advanceTimersByTimeAsync(121); });

  it("waits for the summary, then accepts only its own revision and configuration", async () => {
    const own = context("commercial_traction");
    const spy = vi.spyOn(api, "screeningRobustness").mockResolvedValue(robustness({ configuration: own.configuration, weight_profile_id: "commercial_traction" }));
    const { result, rerender } = renderHook(({ on }) => useScreeningRobustness(project, own, on), { initialProps: { on: false } });
    await start(); expect(spy).not.toHaveBeenCalled();
    rerender({ on: true }); await start();
    expect(result.current.status).toBe("ready");
  });
  it("refuses a late reply for another weighting", async () => {
    const own = context("commercial_traction");
    vi.spyOn(api, "screeningRobustness").mockResolvedValue(robustness({ configuration: own.configuration, weight_profile_id: "technical_ip" }));
    const { result } = renderHook(() => useScreeningRobustness(project, own, true));
    await start();
    expect(result.current).toMatchObject({ status: "error", message: "Rank ranges do not match the selected revision and configuration." });
  });
});

describe("decision confidence", () => {
  it("shows the engine's shares with intervals widened to whole percents", () => {
    expect(shareText({ share: 0.9247, lower: 0.919, upper: 0.9298 })).toEqual({ share: "92%", interval: "(91–93%)" });
    expect(shareText({ share: 0, lower: 0, upper: 0.0004 })).toEqual({ share: "0%", interval: "(0–1%)" });
    expect(shareText({ share: 0.5, lower: null, upper: null })).toEqual({ share: "50%", interval: null });
  });
  it("renders one row per engine row, in the engine's order, with a column per class", () => {
    render(<DecisionConfidence layout={layoutOf(robustness())} names={names}/>);
    const table = screen.getByRole("table", { name: "Decision confidence" });
    const headers = within(table).getAllByRole("columnheader").map((h) => h.textContent);
    expect(headers).toEqual(["Alternative", "First", "Top 3", "Prioritise for diligence", "Monitor"]);
    const rows = within(table).getAllByRole("row").slice(1);
    expect(rows.map((r) => within(r).getByRole("rowheader").textContent)).toEqual(summary.rows.map((r) => r.name));
    expect(within(rows[0] as HTMLElement).getAllByRole("cell").map((c) => c.textContent))
      .toEqual(["92% (91–93%)", "> 99% (99–100%)", "97% (97–98%)", "3% (2–3%)"]);
    expect(screen.getByText(/not probabilities of success/)).toBeInTheDocument();
  });
  it("shows nothing without bands, and appears under the summary table once they arrive", () => {
    const { container } = render(<DecisionConfidence layout={null} names={names}/>);
    expect(container).toBeEmptyDOMElement();
    render(<ScreeningSummaryView summary={summary} project={project} bands={ready(robustness())}/>);
    expect(screen.getByRole("table", { name: "Decision confidence" })).toBeInTheDocument();
  });
});
