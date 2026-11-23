import { createContext, use } from "react";
import type { EditorView } from "./viewMode";

/** Which workspace view encloses a component. Presentation only: it changes wording and display
 * precision, never a request, a value or the project. Outside the editor it is the Builder view. */
export const EditorViewContext = createContext<EditorView>("builder");

export function useIsAnalystView(): boolean {
  return use(EditorViewContext) === "analyst";
}
