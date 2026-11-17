import type { StarterMaterialization } from "../api/types";

export type EditorView = "analyst" | "builder";
export interface WorkedEntry { projectId: string; fingerprint: string; templateId: string }
export interface ViewIntent { view: EditorView; preference: EditorView; entry: WorkedEntry | null }
const choiceKey = (id: string) => `mcda:editor-view:${id}`;
const entryKey = (id: string) => `mcda:worked-entry:${id}`;
const studyTemplate = "venture_screening_study";

/** Session preferences and a verified gallery receipt never enter Project or undo history. */
export function viewIntent(projectId: string): ViewIntent {
  try {
    const saved = sessionStorage.getItem(choiceKey(projectId));
    const preference: EditorView = saved === "analyst" ? "analyst" : "builder";
    const text = sessionStorage.getItem(entryKey(projectId));
    let value: unknown = null;
    try { value = text ? JSON.parse(text) : null; } catch { /* ignore a corrupt gallery hint, retain the explicit choice */ }
    const entry = value && typeof value === "object" && "projectId" in value && value.projectId === projectId &&
      "templateId" in value && value.templateId === studyTemplate && "fingerprint" in value && typeof value.fingerprint === "string"
      ? { projectId, fingerprint: value.fingerprint, templateId: studyTemplate } : null;
    return { view: entry ? "analyst" : preference, preference, entry };
  } catch { return { view: "builder", preference: "builder", entry: null }; }
}
export function chooseEditorView(projectId: string, view: EditorView): void {
  try { sessionStorage.setItem(choiceKey(projectId), view); } catch { /* the current view still works without storage */ }
}
export function rememberWorkedEntry(materialization: StarterMaterialization): void {
  if (materialization.receipt.request.mode !== "worked_example" || materialization.receipt.request.template_id !== studyTemplate) return;
  const entry: WorkedEntry = { projectId: materialization.project.id, fingerprint: materialization.receipt.project_fingerprint, templateId: studyTemplate };
  try { sessionStorage.setItem(entryKey(entry.projectId), JSON.stringify(entry)); } catch { /* default Builder if storage is unavailable */ }
}
export function consumeWorkedEntry(projectId: string): void {
  try { sessionStorage.removeItem(entryKey(projectId)); } catch { /* optional session hint */ }
}
