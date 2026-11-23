import { useIsAnalystView } from "../../state/editorViewContext";
import { displayAnalyst, displayNumber } from "./numberDisplay";

/** An engine number as text: four decimals in the Builder view; in the Analyst view the product's
 * shorter precision, with the full value in `title`. Formatting only. */
export function ValueText({ value }: { value: number }) {
  return useIsAnalystView() ? <span className="value-text" title={String(value)}>{displayAnalyst(value)}</span> : <>{displayNumber(value)}</>;
}
