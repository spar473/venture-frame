import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { starterCard } from "./helpers";

// Structured judgement (deep dive 3): rank a group from the engine's suggestion, state a
// conviction, read the engine's checks, compare decision confidence under each proposal and
// apply one as the weight uncertainty, against the real engine.

async function openStudyInBuilder(page: Page) {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/");
  await starterCard(page, "Venture screening").click();
  await page.getByRole("button", { name: "Open worked example — Venture screening", exact: true }).click();
  await page.getByLabel("Partner weighting", { exact: true }).selectOption("technical_ip");
  await page.getByRole("button", { name: "Builder view", exact: true }).click();
  await page.getByRole("tab", { name: "Judgement", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Judgement: Technical/IP-led" })).toBeVisible();
}

test("rank, state conviction, compare decision confidence and apply a proposal", async ({ page }) => {
  await openStudyInBuilder(page);
  await expect(page.getByText("Overall: Consistent.")).toBeVisible({ timeout: 30_000 });
  await page.getByRole("button", { name: "Rank the top level, starting from the swing ratings" }).click();
  const top = page.getByRole("table", { name: "Weights under Top level" });
  await expect(top.getByRole("row", { name: /Problem and solution/ }).getByRole("cell").nth(0)).toHaveText("1");
  await expect(top.getByRole("row", { name: /Problem and solution/ }).getByRole("cell").nth(3)).toHaveText("0.521");
  await expect(top.getByRole("row", { name: /Mandate/ })).toContainText("Rated 0: held at 0");
  await page.getByLabel("Conviction").selectOption("medium");
  const proposals = page.getByRole("region", { name: "Proposed weight uncertainty" });
  await expect(proposals.getByText("Medium conviction: every swing rating ±30%")).toBeVisible({ timeout: 30_000 });
  await expect(proposals.getByText("Ranking only: every split that keeps the stated order")).toBeVisible();

  // a ranking the ratings contradict is named
  await page.getByRole("button", { name: "Move Traction up" }).click();
  await page.getByRole("button", { name: "Move Traction up" }).click();
  await page.getByRole("button", { name: "Move Traction up" }).click();
  await expect(page.getByRole("list", { name: "Findings under Top level" })).toContainText(
    "The order puts Traction above Problem and solution, but the swing ratings give Problem and solution more (100 against 50).",
    { timeout: 30_000 });
  for (let i = 0; i < 3; i++) await page.getByRole("button", { name: "Move Traction down" }).click();
  await expect(page.getByRole("list", { name: "Findings under Top level" })).toHaveCount(0, { timeout: 30_000 });

  await page.getByRole("button", { name: "Compare decision confidence" }).click();
  const table = page.getByRole("table", { name: "Decision confidence by proposal" });
  await expect(table).toBeVisible({ timeout: 60_000 });
  await expect(table.getByRole("row", { name: /Upstock/ }).getByRole("cell")).toHaveText(["93%", "97%", "82%", "83%", "40%", "48%"]);
  const { violations } = await new AxeBuilder({ page }).analyze();
  expect(violations.filter(v => v.impact === "serious" || v.impact === "critical").map(v => `${v.id}: ${v.help}`)).toEqual([]);

  await proposals.getByRole("listitem").filter({ hasText: "Ranking only" }).getByRole("button", { name: "Use as the weight uncertainty" }).click();
  await page.getByRole("tab", { name: "Uncertainty", exact: true }).click();
  await expect(page.getByLabel("Weight uncertainty under Top level")).toHaveValue("ranking");
  await expect(page.getByRole("list", { name: "Order under Top level" }).getByRole("listitem")).toHaveText(
    [/^Problem and solution/, /^Team/, /^Market/, /^Traction/]);
  await expect(page.getByLabel("Hold Mandate (screened, not scored) at the profile’s weight")).toBeChecked();
});
