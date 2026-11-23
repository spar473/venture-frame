import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import type { StarterTemplate } from "../../src/api/types";
import { startPageStarters } from "../../src/components/Dashboard";
import { closestChangeExact, closestChangePlain, type ChangeNames, type Fragility } from "../../src/components/results/closestChange";
import { displayAnalyst, displayWeightPoints } from "../../src/components/results/numberDisplay";
import { ANALYST_DECIMALS, PRODUCT_NAME, START_PAGE_STARTERS } from "../../src/product";

const names: ChangeNames = { alternatives: { hectre: "Hectre", wych: "Wych", a: "Alpha", b: "Beta" }, classes: { monitor: "Monitor", top: "Prioritise for diligence" } };
const change = (patch: Partial<Fragility>): Fragility => ({
  status: "change", kind: "rank", criterion_id: "c", criterion_name: "Traction", direction: "raise", base_local_weight: 0.2,
  local_weight: 0.3, weight_change: 0.061454, other_alternative_id: "hectre", class_after_id: null, tied_with: [], ...patch,
});

describe("closest change in plain language (Analyst view)", () => {
  it("states every rank and class move, rise and fall, as points of total weight", () => {
    expect(closestChangePlain(change({ kind: "class", class_after_id: "monitor" }), names)).toBe("Moves to Monitor if Traction gains 6.1 pts of weight");
    expect(closestChangePlain(change({ kind: "class", class_after_id: "top", direction: "lower", weight_change: 0.0010058 }), names))
      .toBe("Moves to Prioritise for diligence if Traction loses 0.1 pts of weight");
    expect(closestChangePlain(change({ other_alternative_id: "wych", weight_change: 0.41398 }), names)).toBe("Swaps with Wych if Traction gains 41.4 pts of weight");
    expect(closestChangePlain(change({ direction: "lower", weight_change: 0.016787 }), names)).toBe("Swaps with Hectre if Traction loses 1.7 pts of weight");
  });
  it("never shows a nonzero change as 0.0 points", () => {
    expect(closestChangePlain(change({ weight_change: 0.0004 }), names)).toBe("Swaps with Hectre if Traction gains < 0.1 pts of weight");
    expect(displayWeightPoints(0.0005)).toBe("0.1");
    expect(displayWeightPoints(0.00049)).toBe("< 0.1");
  });
  it("keeps tied and no-change wording in both views", () => {
    const tied = change({ status: "tied", kind: null, direction: null, weight_change: null, tied_with: ["a", "b"] });
    const none = change({ status: "none", kind: null, direction: null, weight_change: null });
    for (const text of [closestChangePlain, closestChangeExact]) {
      expect(text(tied, names)).toBe("Tied now with Alpha, Beta");
      expect(text(none, names)).toBe("No one-way change in range");
    }
  });
  it("leaves the Builder view's exact wording unchanged", () => {
    expect(closestChangeExact(change({ kind: "class", class_after_id: "monitor" }), names)).toBe("+0.0615 of total weight on Traction → moves to Monitor");
    expect(closestChangeExact(change({ direction: "lower", weight_change: 0.016787 }), names)).toBe("−0.0168 of total weight on Traction → swaps with Hectre");
  });
});

describe("Analyst-view numbers", () => {
  it("uses the configured decimals and floors at that resolution", () => {
    expect(ANALYST_DECIMALS).toBe(3);
    expect(displayAnalyst(0.641415)).toBe("0.641");
    expect(displayAnalyst(0.000938)).toBe("0.001");
    expect(displayAnalyst(0.0004)).toBe("< 0.001");
    expect(displayAnalyst(-0.0004)).toBe("−< 0.001");
    expect(displayAnalyst(0)).toBe("0.000");
    expect(displayAnalyst(1)).toBe("1.000");
  });
});

describe("product shell", () => {
  it("shows only the configured starters, in the configured order", () => {
    const starter = (id: string) => ({ id, title: id }) as unknown as StarterTemplate;
    const api = ["supplier_selection", "venture_screening", "blank", "infrastructure_procurement", "venture_screening_study"].map(starter);
    expect(startPageStarters(api).map((t) => t.id)).toEqual(START_PAGE_STARTERS);
    expect(startPageStarters([starter("blank")]).map((t) => t.id)).toEqual(["blank"]);
  });
  it("names the product in the static page title", () => {
    const html = readFileSync(resolve(__dirname, "../../index.html"), "utf8");
    expect(html).toContain(`<title>${PRODUCT_NAME}</title>`);
  });
});
