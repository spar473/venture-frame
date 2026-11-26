import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../../src/api/client";
import type { Elicitation, ElicitationCheck, ElicitationPreview, Operation, Project, ScreeningRobustness } from "../../src/api/types";
import { ElicitationPanel } from "../../src/components/ElicitationPanel";
import { BASE_SELECTION, resolveContext } from "../../src/state/analysisContext";
import { node, projectWith } from "../fixtures";

const RECORD: Elicitation = {
  elicited_by: "Partner A", note: "", conviction: "medium",
  groups: [{ parent_id: null, order: ["team", "market"], reasons: { team: "verifiable" } }],
};

function project(record: Elicitation | null): Project {
  return projectWith([node("team", null, 0, { name: "Team" }), node("market", null, 1, { name: "Market" }), node("mandate", null, 2, { name: "Mandate" })], {
    weight_profiles: [{ id: "p", name: "Partner A", method: "swing", judgements: { team: 80, market: 40, mandate: 0 }, ...(record ? { elicitation: record } : {}) }],
  });
}

const child = (id: string, name: string, rating: number, extra: Partial<ElicitationCheck["groups"][number]["children"][number]> = {}) =>
  ({ criterion_id: id, name, rating, swing: rating / 120, roc: null, ahp: null, held: rating === 0, reason: "", ...extra });

function checkOf(recorded: boolean): ElicitationCheck {
  return {
    project_fingerprint: "fp", profile_id: "p", profile_name: "Partner A", recorded, elicited_by: recorded ? "Partner A" : "",
    elicited_on: null, conviction: recorded ? "medium" : null, status: recorded ? "check" : "consistent",
    label: "Checks and proposals are derived.",
    groups: [{
      parent_id: null, name: "the top level", scored: true, order: recorded ? ["team", "market"] : [], suggested_order: ["team", "market"],
      children: [child("team", "Team", 80, { roc: 0.75 }), child("market", "Market", 40, { roc: 0.25 }), child("mandate", "Mandate", 0)],
      order_agrees: recorded ? true : null, order_conflicts: [], ahp: null, largest_gap: null,
      missing_reasons: recorded ? ["market"] : ["team", "market"],
      findings: recorded ? ["The order puts Market above Team, but the swing ratings give Team more (80 against 40)."] : [],
      status: recorded ? "check" : "consistent",
    }],
    proposals: recorded ? [{
      mode: "ranking", label: "Ranking only: every split that keeps the stated order", explanation: "Every split that keeps the order.",
      smaa: { dependence: "independent", cells: [], shared_draws: [], groups: [{ parent_id: null, kind: "ranking", intervals: {}, order: ["team", "market"], held: ["mandate"] }] },
    }] : [],
  };
}

function bands(first: number): ScreeningRobustness {
  const share = (s: number) => ({ share: s, lower: null, upper: null });
  return {
    project_fingerprint: "fp", configuration: { method: "mavt", parameters: {}, scenario_id: null, weight_profile_id: null },
    weight_profile_id: "p", scenario_id: null, available: true, reason: null, method: "smaa2", run: "combined", draws: 1000, seed: 0,
    evaluation: "sampled", parameters: "defaults", basis: "Varied: the weights.", label: "", confidence_label: "", top_k: 2, confidence: null,
    groups: [], rows: [
      { alternative_id: "x", name: "Xeno", mavt_rank: 1, rank_low: 1, rank_high: 1, band_share: 1, rank1_share: first, first: share(first), top_ranks: share(1),
        classes: [{ class_id: "go", name: "Go", ...share(first) }] },
    ],
  } as unknown as ScreeningRobustness;
}

const DEFAULT_PREVIEWS: ElicitationPreview["previews"] = [
  { mode: "current", label: "Stated uncertainty now", robustness: bands(0.93) },
  { mode: "ranking", label: "Ranking only", robustness: bands(0.42) },
];

function setup(record: Elicitation | null, previews: ElicitationPreview["previews"] = DEFAULT_PREVIEWS) {
  const p = project(record);
  vi.spyOn(api, "elicitationCheck").mockResolvedValue(checkOf(record !== null));
  const preview: ElicitationPreview = { project_fingerprint: "fp", profile_id: "p", label: "Decision confidence now and under each proposal.",
    previews };
  const previewSpy = vi.spyOn(api, "elicitationPreview").mockResolvedValue(preview);
  const apply = vi.fn<(ops: Operation[]) => Promise<boolean>>(() => Promise.resolve(true));
  render(<ElicitationPanel project={p} context={resolveContext(p, "fp", BASE_SELECTION)} busy={false} apply={apply}/>);
  return { apply, previewSpy };
}
const last = (apply: ReturnType<typeof setup>["apply"]) => apply.mock.calls.at(-1)?.[0];
async function settle() { await act(async () => { await vi.advanceTimersByTimeAsync(130); }); }
beforeEach(() => { vi.useFakeTimers(); });
afterEach(() => { vi.restoreAllMocks(); vi.useRealTimers(); });

describe("the Judgement panel", () => {
  it("starts a ranking from the engine's suggested order and records who and how sure", async () => {
    const { apply } = setup(null);
    await settle();
    fireEvent.click(screen.getByRole("button", { name: /Rank the top level, starting from the swing ratings/ }));
    expect(last(apply)).toEqual([{ op: "set_elicitation", profile_id: "p",
      elicitation: { elicited_by: "", note: "", groups: [{ parent_id: null, order: ["team", "market"] }] } }]);
    fireEvent.change(screen.getByLabelText("Conviction"), { target: { value: "high" } });
    expect(last(apply)).toEqual([{ op: "set_elicitation", profile_id: "p",
      elicitation: { elicited_by: "", note: "", groups: [], conviction: "high" } }]);
    expect(screen.getByText("Overall: Consistent.")).toBeInTheDocument();
  });

  it("shows the engine's weights and findings, moves a child, and records reasons and AHP answers", async () => {
    const { apply } = setup(RECORD);
    await settle();
    const table = screen.getByRole("table", { name: "Weights under Top level" });
    const team = within(table).getByRole("row", { name: /Team/ });
    expect(within(team).getAllByRole("cell").slice(0, 5).map((c) => c.textContent)).toEqual(["1", "80", "0.667", "0.750", "—"]);
    expect(within(table).getByRole("row", { name: /Mandate/ })).toHaveTextContent("Rated 0: held at 0");
    expect(screen.getByRole("list", { name: "Findings under Top level" })).toHaveTextContent("puts Market above Team");
    fireEvent.click(screen.getByRole("button", { name: "Move Market up" }));
    expect(last(apply)?.[0]).toMatchObject({ op: "set_elicitation", elicitation: { groups: [{ parent_id: null, order: ["market", "team"] }] } });
    fireEvent.change(screen.getByLabelText("Team compared with Market"), { target: { value: "-3" } });
    expect(last(apply)?.[0]).toMatchObject({ elicitation: { groups: [{ pairwise: [{ row: "market", column: "team", value: 3 }] }] } });
    const reason = screen.getByLabelText("Reason for Market");
    fireEvent.change(reason, { target: { value: "big if true" } });
    fireEvent.blur(reason);
    expect(last(apply)?.[0]).toMatchObject({ elicitation: { groups: [{ reasons: { team: "verifiable", market: "big if true" } }] } });
  });

  it("applies a proposal as the weight uncertainty and compares decision confidence on request", async () => {
    const { apply, previewSpy } = setup(RECORD);
    await settle();
    fireEvent.click(screen.getByRole("button", { name: "Use as the weight uncertainty" }));
    expect(last(apply)).toEqual([{ op: "set_smaa_model", smaa: checkOf(true).proposals[0]?.smaa }]);
    expect(previewSpy).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Compare decision confidence" }));
    await settle();
    const table = screen.getByRole("table", { name: "Decision confidence by proposal" });
    expect(within(table).getByRole("row", { name: /Xeno/ })).toHaveTextContent("93%93%42%42%");
    expect(screen.getByText(/not probabilities of success/)).toBeInTheDocument();
  });

  it("offers a weight uncertainty choice and shows the concentration the engine used for true conviction", async () => {
    const { previewSpy } = setup(RECORD, [
      { mode: "current", label: "Stated uncertainty now", robustness: bands(0.93) },
      { mode: "conviction", label: "Medium conviction: every swing rating ±30%", robustness: bands(0.8) },
      { mode: "dirichlet", label: "True conviction: Dirichlet around the ratings (medium)", robustness: bands(0.79),
        kappas: [{ parent_id: null, name: "the top level", kappa: 136 }, { parent_id: "team", name: "Team", kappa: 139 }] },
    ]);
    await settle();
    fireEvent.click(screen.getByRole("button", { name: "Compare decision confidence" }));
    await settle();
    expect(previewSpy).toHaveBeenCalledTimes(1);
    const choice = screen.getByRole("group", { name: "Weight uncertainty" });
    expect(within(choice).getByRole("radio", { name: "Conviction bands" })).toBeChecked();
    const table = screen.getByRole("table", { name: "Decision confidence by proposal" });
    expect(within(table).getByRole("row", { name: /Xeno/ })).toHaveTextContent("93%93%80%80%79%79%");
    const chosenHeader = () => within(table).getAllByRole("columnheader").find((h) => h.getAttribute("aria-current") === "true");
    expect(chosenHeader()).toHaveTextContent("Medium conviction");
    expect(screen.queryByLabelText("Concentration used")).not.toBeInTheDocument();
    fireEvent.click(within(choice).getByRole("radio", { name: "True conviction (Dirichlet κ)" }));
    expect(chosenHeader()).toHaveTextContent("True conviction");
    expect(screen.getByLabelText("Chosen weight uncertainty")).toHaveTextContent("True conviction: Dirichlet around the ratings (medium).");
    expect(screen.getByLabelText("Concentration used")).toHaveTextContent("κ used: the top level 136, Team 139.");
    expect(previewSpy).toHaveBeenCalledTimes(1); // the browser re-reads the engine's previews; it computes nothing
  });

  it("clears the record", async () => {
    const { apply } = setup(RECORD);
    await settle();
    fireEvent.click(screen.getByRole("button", { name: "Clear the record" }));
    expect(last(apply)).toEqual([{ op: "set_elicitation", profile_id: "p", elicitation: null }]);
  });
});
