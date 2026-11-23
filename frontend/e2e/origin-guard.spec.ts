// The explicit cross-site guard must not get in the real interface's way: the browser sends Origin
// and Sec-Fetch-Site with every save, and the server accepts them because they name itself. A page
// on another origin that slips past the preflight with a "simple" request is refused with 403.
import { expect, test } from "@playwright/test";
import { at, openTab, readJson, saveState, storedFile, VC_FILE } from "./helpers";

interface Raw {
  id: string;
  title: string;
  alternatives: { id: string; name: string }[];
}
const original = readJson(VC_FILE) as Raw;
const FIRST_ALT = at(original.alternatives, 0).name;

test("the real interface still saves, sending its own origin", async ({ page, request, baseURL }) => {
  const id = `origin-guard-${String(Date.now())}`;
  const imported = await request.post("/api/projects/import", {
    data: { ...original, id, title: "Origin guard" },
  });
  expect(imported.status()).toBe(201);
  const before = await storedFile(request, id);

  await page.goto(`/#/p/${id}`);
  await expect(saveState(page)).toHaveText("All changes saved");
  await openTab(page, "Alternatives");
  const field = page.getByLabel(`Name of alternative ${FIRST_ALT}`);
  await field.fill("Saved through the guard");
  await field.press("Enter");
  await expect(saveState(page)).toHaveText("● Unsaved changes");

  const saved = page.waitForResponse(
    (r) => r.request().method() === "PUT" && r.url().endsWith(`/api/projects/${id}`),
  );
  await page.getByRole("button", { name: "Save", exact: true }).click();
  const response = await saved;
  const headers = await response.request().allHeaders();
  expect(headers.origin).toBe(new URL(baseURL ?? "").origin);
  expect(headers["sec-fetch-site"]).toBe("same-origin");
  expect(response.status()).toBe(200);
  await expect(saveState(page)).toHaveText("All changes saved");

  const after = await storedFile(request, id);
  expect(after.equals(before)).toBe(false);
  expect(after.toString("utf-8")).toContain("Saved through the guard");
  await page.reload();
  await openTab(page, "Alternatives");
  await expect(page.getByLabel("Name of alternative Saved through the guard")).toBeVisible();
});

test("a page on another origin cannot create a project", async ({ page, request, baseURL }) => {
  const base = new URL(baseURL ?? "");
  // Same server, different origin: localhost instead of 127.0.0.1.
  const other = `http://localhost:${base.port}`;
  await page.goto(`${other}/`);
  const before = (await (await request.get("/api/projects")).json()) as { projects: unknown[] };

  const refused = page.waitForResponse((r) => r.request().method() === "POST");
  await page.evaluate(async (target) => {
    // A "simple" request: no preflight, so it reaches the server with this page's Origin.
    await fetch(`${target}/api/projects`, {
      method: "POST",
      mode: "no-cors",
      headers: { "Content-Type": "text/plain" },
      body: JSON.stringify({ title: "From another origin" }),
    }).catch(() => undefined);
  }, base.origin);
  const response = await refused;
  expect(response.status()).toBe(403);
  expect((await response.request().allHeaders()).origin).toBe(other);

  const after = (await (await request.get("/api/projects")).json()) as { projects: unknown[] };
  expect(after.projects.length).toBe(before.projects.length);
});
