import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { JudgementAids } from "../../src/api/types";
import { JudgementAidsView } from "../../src/components/JudgementAids";

const AIDS = {
  project_fingerprint: "fp", weight_profile_id: "p", available: true, reason: "",
  label: "Derived from the project and the profile.",
  anchors: [{ criterion_id: "team_completeness", name: "Team completeness", tied_worst: 1, tied_best: 1,
    worst: { alternative_id: "kea", name: "Kea", level: "One key role named", value: 0 },
    best: { alternative_id: "upstock", name: "Upstock", level: "All three full-time", value: 0.75 },
    swing: "From Kea's level (One key role named) to Upstock's (All three full-time) on Team completeness" }],
  break_even: {
    project_fingerprint: "fp", weight_profile_id: "p", scenario_id: null, leader: "upstock", leader_name: "Upstock",
    rule: "Per criterion: siblings rescaled in proportion.",
    criteria: [{ criterion_id: "traction", name: "Traction", local_weight: 0.17, share: 0.17, rises_above: null,
      falls_below: { local_weight: 0.12, share: 0.12, new_leader: "wellumio" },
      sentence: "Upstock stays first unless Traction falls below 12%" }],
    nearest: null, challengers: [], classes: [],
    nearest_sentence: "The smallest change that lets another alternative tie Upstock moves 4.8% of the weight",
  },
  references: [{
    reference_id: "vc_literature", title: "What the venture literature says", purpose: "For comparison only.",
    criteria_note: "Traction is left out.", partner_order: ["problem_solution", "team"], partner_order_names: ["Problem and solution", "Team"],
    variants: [{ id: "stated_by_vcs", name: "Stated by VCs", summary: "Team above business.", disagreements: 1,
      citations: [{ authors: "Gompers et al.", year: 2020, title: "How do venture capitalists make decisions?", where: "JFE", url: "https://www.nber.org/papers/w22587" }],
      statements: [{ horizon: null, text: "Team above Problem and solution", disagreements: 1,
        pairs: [{ above: "team", below: "problem_solution", above_name: "Team", below_name: "Problem and solution", partner: "disagrees" }] }] }],
  }],
} as unknown as JudgementAids;

describe("the judgement aids", () => {
  it("shows the engine's break-even sentence, anchored swings and the reference beside the partner's order", () => {
    render(<JudgementAidsView aids={AIDS}/>);
    expect(screen.getByRole("list", { name: "Where the leader changes" })).toHaveTextContent("Upstock stays first unless Traction falls below 12%.");
    expect(screen.getByLabelText("Nearest change")).toHaveTextContent("moves 4.8% of the weight");
    expect(screen.getByRole("list", { name: "Anchored swings" })).toHaveTextContent("From Kea's level (One key role named) to Upstock's");
    const reference = screen.getByRole("region", { name: "Reference profile: What the venture literature says" });
    expect(within(reference).getByLabelText("Your order")).toHaveTextContent("Your order: Problem and solution > Team");
    expect(within(reference).getByRole("heading", { name: "Stated by VCs (1 disagreement)" })).toBeInTheDocument();
    const pair = within(reference).getByText("Team above Problem and solution: Your order disagrees");
    expect(pair).toHaveClass("disagrees");
    expect(within(reference).getByRole("link", { name: "https://www.nber.org/papers/w22587" })).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("says why when the model cannot be evaluated", () => {
    render(<JudgementAidsView aids={{ ...AIDS, available: false, reason: "the model cannot be evaluated yet: missing values" }}/>);
    expect(screen.getByText("the model cannot be evaluated yet: missing values")).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Break-even weights" })).toBeNull();
  });
});
