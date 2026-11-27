import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { starterCard } from "./helpers";

// Judgement aids (DD11): break-even sentences, swings anchored in the project's own ventures, and the
// literature reference profile beside the partner's order, all worded by the real engine.

test("the Judgement tab shows break-even weights, anchored swings and the reference profile", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/");
  await starterCard(page, "Venture screening").click();
  await page.getByRole("button", { name: "Open worked example — Venture screening", exact: true }).click();
  await page.getByLabel("Partner weighting", { exact: true }).selectOption("technical_ip");
  await page.getByRole("button", { name: "Builder view", exact: true }).click();
  await page.getByRole("tab", { name: "Judgement", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Judgement: Technical/IP-led" })).toBeVisible();

  const leader = page.getByRole("list", { name: "Where the leader changes" });
  await expect(leader).toContainText("Upstock stays first unless Team falls below 4.0% or rises above 58%", { timeout: 30_000 });
  await expect(page.getByLabel("Nearest change")).toContainText("Wellumio draws level");
  await expect(page.getByRole("list", { name: "Anchored swings" })).toContainText("to Wellumio's (Next-stage leads named) on Team completeness");

  const reference = page.getByRole("region", { name: /^Reference profile:/ });
  await expect(reference.getByLabel("Your order")).toHaveText("Your order: Problem and solution > Team > Market");
  await expect(reference.getByRole("heading", { name: "Stated by VCs (1 disagreement)" })).toBeVisible();
  await expect(reference.getByRole("heading", { name: "Outcome-predictive (2 disagreements)" })).toBeVisible();
  await expect(reference.getByText("Team above Problem and solution: Your order disagrees").first()).toBeVisible();

  const { violations } = await new AxeBuilder({ page }).analyze();
  expect(violations.filter(v => v.impact === "serious" || v.impact === "critical").map(v => `${v.id}: ${v.help}`)).toEqual([]);
});
