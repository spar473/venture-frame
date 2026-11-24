import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import { PRODUCT_NAME, START_PAGE_STARTERS } from "../src/product";
import { starterCard } from "./helpers";

// Venture Frame's partner view (run 02): the product shell, the Analyst header, Analyst precision,
// plain closest changes, rank ranges and their groups, layout and accessibility.
//
// Screenshot assertions run on Linux only: system fonts render differently on each platform, and
// the committed baselines are the -linux variants that CI checks. Every other assertion runs on
// every platform.
const screenshots = process.platform === "linux";
const summaryTable = (page: Page) => page.getByRole("table", { name: "Screening summary", exact: true });
const VIEWPORTS = [[1440, 900], [1920, 1080], [1280, 800]] as const;
const quiet = /^(Checking|Calculating|Preparing|Finding|Running|Resolving|Loading|Deriving|Probing|Evaluating|Inspecting)\b/i;

async function settled(page: Page) {
  await page.waitForTimeout(250);
  await page.waitForLoadState("networkidle");
  await expect.poll(() => page.getByRole("status").evaluateAll((nodes, source) => {
    const pattern = new RegExp(source, "i");
    return nodes.filter(n => n.getClientRects().length && !n.closest("[hidden],[inert]")).some(n => pattern.test(n.textContent.trim()));
  }, quiet.source), { timeout: 30_000 }).toBe(false);
}

async function openStudy(page: Page) {
  await page.goto("/");
  await starterCard(page, "Venture screening").click();
  await page.getByRole("button", { name: "Open worked example — Venture screening", exact: true }).click();
  await expect(page.getByRole("button", { name: "Analyst view", exact: true })).toHaveAttribute("aria-pressed", "true");
  await page.getByLabel("Partner weighting", { exact: true }).selectOption("technical_ip");
  await expect(summaryTable(page).getByText("(2–6)")).toBeVisible({ timeout: 30_000 });
  await settled(page);
}

/** Zero serious or critical violations; the rest are printed for the run report. */
async function axe(page: Page, screen: string) {
  const { violations } = await new AxeBuilder({ page }).analyze();
  for (const v of violations) console.log(`AXE ${screen} ${v.impact ?? "none"} ${v.id} (${String(v.nodes.length)} nodes): ${v.help}`);
  expect(violations.filter(v => v.impact === "serious" || v.impact === "critical").map(v => `${v.id}: ${v.help}`), `axe ${screen}`).toEqual([]);
}

test("the start page names the product and shows only the configured starters", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/");
  await expect(page).toHaveTitle(PRODUCT_NAME);
  await expect(page.locator(".brand")).toHaveText(PRODUCT_NAME);
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Screen companies for due diligence");
  const cards = page.locator(".starter-card");
  await expect(cards).toHaveCount(START_PAGE_STARTERS.length);
  await expect(cards.locator(".starter-title")).toHaveText(["Blank", "Venture screening", "Venture screening: evidence and model workflows (invented data)"]);
  await expect(page.getByText("Runs locally · your files stay on this computer")).toBeVisible();
  await settled(page);
  if (screenshots) {
    // The app bar and starters only: the project list below depends on what earlier specs stored.
    const hero = await page.locator("section.hero").boundingBox();
    if (!hero) throw Error("missing start-page hero");
    await expect(page).toHaveScreenshot("start-page.png", { animations: "disabled", caret: "hide", clip: { x: 0, y: 0, width: 1440, height: Math.ceil(hero.y + hero.height) } });
  }
  await axe(page, "start-page");
  // Every other starter stays one step away, in New project… → Starter gallery.
  await page.getByText("More starters:").getByRole("button", { name: "New project…" }).click();
  const gallery = page.getByRole("region", { name: "Starter gallery" });
  for (const title of ["Supplier selection", "Infrastructure procurement"])
    await expect(gallery.getByRole("button", { name: `Open worked example — ${title}`, exact: true })).toBeVisible();
});

test("the Analyst header leads with Partner weighting and keeps method and scenario under Advanced", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await openStudy(page);
  const header = page.getByRole("region", { name: "Analysis context" });
  await expect(header.getByLabel("Weight profile")).toHaveCount(0);
  const advanced = header.locator("details.analysis-advanced");
  await expect(advanced).not.toHaveAttribute("open");
  await expect(header.getByLabel("Method", { exact: true })).toBeHidden();
  // Keyboard: from Partner weighting, Tab reaches Advanced; Enter opens it; Tab reaches Method.
  await header.getByLabel("Partner weighting", { exact: true }).focus();
  await page.keyboard.press("Tab");
  await expect(advanced.locator("summary")).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(advanced).toHaveAttribute("open");
  await page.keyboard.press("Tab");
  await expect(header.getByLabel("Method", { exact: true })).toBeFocused();
  await expect(header.getByLabel("Scenario", { exact: true })).toBeVisible();
  await expect(header.getByRole("button", { name: "Set up AHP" })).toBeVisible();
  // A non-default reading stays named on the closed summary.
  await header.getByLabel("Scenario", { exact: true }).selectOption("linear_market_size");
  await advanced.locator("summary").click();
  await expect(advanced.locator("summary")).toContainText("Scenario: ");
  await expect(advanced).not.toHaveAttribute("open");
});

test("Analyst numbers use three decimals with full values on hover; Builder keeps four", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await openStudy(page);
  const upstock = summaryTable(page).getByRole("row").filter({ has: page.getByRole("rowheader", { name: "Upstock" }) });
  const value = upstock.locator(".value-text").first();
  await expect(value).toHaveText("0.641");
  expect(Number(await value.getAttribute("title"))).toBeCloseTo(0.641415, 6);
  await expect(upstock).toContainText("Moves to Monitor if Problem severity and willingness to pay gains 6.1 pts of weight");
  await expect(page.getByText("pts = percentage points of the total weight.", { exact: false })).toBeVisible();
  await page.getByRole("button", { name: "Builder view", exact: true }).click();
  await page.getByRole("tab", { name: "Results", exact: true }).click();
  const builderRow = summaryTable(page).getByRole("row").filter({ has: page.getByRole("rowheader", { name: "Upstock" }) });
  await expect(builderRow).toContainText("0.6414");
  await expect(builderRow).toContainText("+0.0615 of total weight on Problem severity and willingness to pay → moves to Monitor");
  await expect(builderRow.locator(".value-text")).toHaveCount(0);
});

test("case 2 shows rank ranges, one too-close group and the lead sentence", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await openStudy(page);
  const table = summaryTable(page);
  await expect(table.getByRole("columnheader", { name: "Rank (range)" })).toBeVisible();
  const ranks = await table.locator("td.rank-cell .rank-range").allTextContents();
  expect(ranks).toEqual(["(1–1)", "(2–6)", "(2–4)", "(2–4)", "(4–7)", "(3–6)", "(5–7)", "(8–8)"]);
  await expect(table.locator("td.rank-cell.in-group")).toHaveCount(6);
  await expect(table.getByText("Group 2 of 3: too close to separate")).toHaveCount(6);
  await expect(table.locator(".band-label")).toHaveText(["Too close to separate"]);
  await expect(page.locator(".summary-lead")).toContainText(
    "Upstock is first in 93% of draws. Wellumio, Hectre, Wych, Captivate Technology, Scentian Bio and Hnry overlap in rank under the stated uncertainty.");
  if (screenshots) await expect(page).toHaveScreenshot("analyst-summary.png", { animations: "disabled", caret: "hide" });
  await axe(page, "analyst-summary");
  // Another weighting re-requests; an old reply is never shown under the new one.
  await page.getByLabel("Partner weighting", { exact: true }).selectOption("commercial_traction");
  await expect(table.locator("td.rank-cell .rank-range").nth(1)).toHaveText("(2–3)");
});

test("case 2 shows decision confidence under the summary, from the same draws", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await openStudy(page);
  const table = page.getByRole("table", { name: "Decision confidence", exact: true });
  await expect(table).toBeVisible({ timeout: 30_000 });
  await expect(table.getByRole("columnheader")).toHaveText(["Alternative", "First", "Top 3", "Prioritise for diligence", "Monitor", "Not now"]);
  await expect(table.getByRole("rowheader")).toHaveText(["Upstock", "Wellumio", "Hectre", "Wych", "Captivate Technology", "Scentian Bio", "Hnry", "Kea Aerospace"]);
  await expect(table.getByRole("row").nth(1).getByRole("cell")).toHaveText(["93% (92–94%)", "> 99% (99–100%)", "97% (97–98%)", "3% (2–3%)", "0% (0–1%)"]);
  await expect(table.getByRole("row").nth(4).getByRole("cell").nth(2)).toHaveText("36% (34–37%)"); // Wych, Prioritise
  await expect(page.getByText("Share of draws under the uncertainty stated in the project; not probabilities of success.")).toBeVisible();
  await axe(page, "decision confidence");
});

test("the company drawer and partner comparison read in the Analyst view's terms", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await openStudy(page);
  await summaryTable(page).getByText("Upstock", { exact: true }).click();
  const drawer = page.getByRole("dialog");
  await expect(drawer.getByRole("heading", { name: "All scored contributions" })).toBeVisible();
  await expect(drawer.getByRole("region", { name: "Company closest change" })).toContainText("Moves to Monitor if Problem severity and willingness to pay gains 6.1 pts of weight.");
  await settled(page);
  if (screenshots) await expect(page).toHaveScreenshot("company-drawer.png", { animations: "disabled", caret: "hide" });
  await axe(page, "company-drawer");
  await page.keyboard.press("Escape");
  await page.getByLabel("Compare partners").check();
  const comparison = page.getByRole("region", { name: "Partner comparison" });
  await expect(comparison.getByRole("table", { name: "Partner rankings and movements" })).toContainText("1 · 0.641 · Prioritise for diligence");
  await settled(page);
  await comparison.scrollIntoViewIfNeeded();
  if (screenshots) await expect(page).toHaveScreenshot("partner-comparison.png", { animations: "disabled", caret: "hide" });
  await axe(page, "partner-comparison");
});

test("a project with no stated uncertainty says so and invents no ranges", async ({ page, request }) => {
  const imported = await request.post("/api/examples/supplier_qualification/import?on_conflict=new_id");
  expect(imported.status()).toBe(201);
  const id = ((await imported.json()) as { project: { id: string } }).project.id;
  await page.addInitScript(project => sessionStorage.setItem(`mcda:editor-view:${project}`, "analyst"), id);
  await page.goto(`/#/p/${encodeURIComponent(id)}`);
  await expect(summaryTable(page)).toBeVisible();
  await settled(page);
  await expect(summaryTable(page).getByRole("columnheader", { name: "Rank", exact: true })).toBeVisible();
  await expect(summaryTable(page).locator(".rank-range")).toHaveCount(0);
  await expect(page.locator(".summary-lead")).toContainText("Rank ranges unavailable.");
  await page.getByRole("button", { name: "Why there are no rank ranges" }).click();
  await expect(page.getByText("No stated uncertainty in this project.")).toBeVisible();
});

for (const [width, height] of VIEWPORTS) {
  test(`the screening summary fits ${String(width)}×${String(height)} without a horizontal page scroll`, async ({ page }) => {
    await page.setViewportSize({ width, height });
    await openStudy(page);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow).toBeLessThanOrEqual(0);
    if (width === 1440) {
      const lead = await page.locator(".summary-lead").boundingBox();
      const third = await summaryTable(page).getByRole("row").nth(3).boundingBox();
      expect(lead && lead.y + lead.height).toBeLessThanOrEqual(height);
      expect(third && third.y + third.height).toBeLessThanOrEqual(height);
    }
  });
}
