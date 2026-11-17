import { afterEach, expect, it, vi } from "vitest";
import type { StarterMaterialization } from "../src/api/types";
import { chooseEditorView, consumeWorkedEntry, rememberWorkedEntry, viewIntent } from "../src/state/viewMode";

afterEach(() => { sessionStorage.clear(); vi.restoreAllMocks(); });
const receipt = (template = "venture_screening_study", mode = "worked_example") => ({
  project: { id: "study" }, receipt: { request: { template_id: template, mode }, project_fingerprint: "exact-source" },
}) as StarterMaterialization;
it("defaults ordinary projects and direct imports to Builder, scopes explicit choices by project and browser session", () => {
  expect(viewIntent("study")).toEqual({ view: "builder", preference: "builder", entry: null });
  chooseEditorView("study", "analyst");
  expect(viewIntent("study").view).toBe("analyst"); expect(viewIntent("another").view).toBe("builder");
  chooseEditorView("study", "builder"); expect(viewIntent("study").view).toBe("builder");
  sessionStorage.clear(); expect(viewIntent("study").view).toBe("builder");
});
it("uses only the real worked study receipt and retains its exact fingerprint for editor verification", () => {
  chooseEditorView("study", "builder"); rememberWorkedEntry(receipt());
  expect(viewIntent("study")).toEqual({ view: "analyst", preference: "builder", entry: {
    projectId: "study", templateId: "venture_screening_study", fingerprint: "exact-source",
  } });
  consumeWorkedEntry("study"); expect(viewIntent("study").entry).toBeNull();
  rememberWorkedEntry(receipt("venture_screening")); expect(viewIntent("study").view).toBe("builder");
  rememberWorkedEntry(receipt("venture_screening_study", "fresh")); expect(viewIntent("study").view).toBe("builder");
});
it("ignores malformed or mismatched gallery hints while retaining a valid explicit choice", () => {
  chooseEditorView("study", "analyst");
  for (const text of ["{bad JSON", JSON.stringify({ projectId: "other", fingerprint: "fp", templateId: "venture_screening_study" }), JSON.stringify({ projectId: "study", fingerprint: 12, templateId: "venture_screening_study" })]) {
    sessionStorage.setItem("mcda:worked-entry:study", text);
    expect(viewIntent("study")).toEqual({ view: "analyst", preference: "analyst", entry: null });
  }
});
it("keeps a working Builder default if session storage is unavailable", () => {
  vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw Error("storage denied"); });
  expect(viewIntent("study")).toEqual({ view: "builder", preference: "builder", entry: null });
});
