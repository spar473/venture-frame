import type { ScreeningSummaryRow } from "../../api/types";
import { displayNumber, displayWeightPoints } from "./numberDisplay";

export type Fragility = NonNullable<ScreeningSummaryRow["fragility"]>;
/** Display names by id; lookup only. */
export interface ChangeNames { alternatives: Record<string, string>; classes: Record<string, string> }

export const POINTS_NOTE = "pts = percentage points of the total weight. One weight changes at a time; the others in its group keep their proportions.";

function tiedText(f: Fragility, names: ChangeNames): string {
  return `Tied now with ${f.tied_with.map((id) => names.alternatives[id] ?? id).join(", ")}`;
}

function outcome(f: Fragility, names: ChangeNames): { kind: "swap" | "move"; name: string } {
  return f.kind === "rank"
    ? { kind: "swap", name: names.alternatives[f.other_alternative_id ?? ""] ?? f.other_alternative_id ?? "" }
    : { kind: "move", name: names.classes[f.class_after_id ?? ""] ?? f.class_after_id ?? "" };
}

/** Builder view: the engine's numbers as written since the summary was introduced. */
export function closestChangeExact(f: Fragility, names: ChangeNames): string {
  if (f.status === "tied") return tiedText(f, names);
  if (f.status === "none") return "No one-way change in range";
  const sign = f.direction === "raise" ? "+" : "−";
  const result = outcome(f, names);
  return `${sign}${displayNumber(f.weight_change ?? 0)} of total weight on ${f.criterion_name ?? ""} → ${result.kind === "swap" ? "swaps with" : "moves to"} ${result.name}`;
}

/** Analyst view: the same engine change, stated as percentage points of the total weight. */
export function closestChangePlain(f: Fragility, names: ChangeNames): string {
  if (f.status === "tied") return tiedText(f, names);
  if (f.status === "none") return "No one-way change in range";
  const result = outcome(f, names);
  const verb = f.direction === "raise" ? "gains" : "loses";
  return `${result.kind === "swap" ? "Swaps with" : "Moves to"} ${result.name} if ${f.criterion_name ?? ""} ${verb} ${displayWeightPoints(f.weight_change ?? 0)} pts of weight`;
}
