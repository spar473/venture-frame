import type { Project, ScreeningSummary, ScreeningSummaryRow } from "../../api/types";
import { useIsAnalystView } from "../../state/editorViewContext";
import { CompanyName, companyRow, ratingExplanation, useCompanyExploration } from "../companyExploration";
import type { StudyEvidenceOffer } from "../CompanyExplorer";
import { Caveat, InfoDisclosure } from "../InfoDisclosure";
import { RatingHint } from "../RatingHint";
import { closestChangeExact, closestChangePlain, POINTS_NOTE, type ChangeNames } from "./closestChange";
import { ValueText } from "./ValueText";

const COLUMN_LABELS: Record<ScreeningSummary["columns"][number]["column"], string> = {
  classes: "Class", requirements: "Requirements", fragility: "Closest change", open_questions: "Open questions",
};

/** Where the open-question counts come from: the request this view's data was fetched with. */
export interface QuestionSource {
  /** True when the summary was requested with an evidence dossier. */
  dossier: boolean;
  /** The bundled study's evidence, when it can be loaded in one step. */
  offer?: StudyEvidenceOffer | null;
}

function requirementsText(row: ScreeningSummaryRow, names: Record<string, string>): string {
  const r = row.requirements;
  if (!r) return "";
  const list = (ids: string[]) => ids.map((id) => names[id] ?? id).join(", ");
  if (r.status === "fail") return `Fails: ${list(r.failed)}`;
  if (r.status === "unknown") return `Unknown: ${list(r.unknown)}`;
  return "Meets all";
}

function QuestionsHeader({ source }: { source: QuestionSource }) {
  return <>Open questions <span className="column-source">({source.dossier ? "model and evidence dossier" : "model only"})</span>
    <InfoDisclosure label="About the open-question counts">Counts follow the Open Questions tab.{" "}
      {source.dossier ? "They include outdated, disputed and unreviewed claims from the loaded evidence dossier."
        : <>Load the study's evidence to include outdated, disputed and unreviewed claims.{source.offer && <> <button type="button" className="link" onClick={source.offer.load}>Load the study's evidence</button></>}</>}
    </InfoDisclosure></>;
}

/** The engine's screening summary as a table: one row per alternative, nothing computed here. */
export function ScreeningSummaryView({ summary, project, hideOpenQuestions = false, questions = { dossier: false } }: {
  summary: ScreeningSummary; project: Project; hideOpenQuestions?: boolean; questions?: QuestionSource;
}) {
  const exploration = useCompanyExploration();
  const analyst = useIsAnalystView();
  if (summary.status === "unavailable") return <p role="status" aria-label="Screening summary unavailable">{summary.reason}</p>;
  const names: ChangeNames = {
    alternatives: Object.fromEntries(project.alternatives.map((a) => [a.id, a.name])),
    classes: Object.fromEntries((project.sorting?.classes ?? []).map((c) => [c.id, c.name])),
  };
  const requirementNames = Object.fromEntries((project.requirements ?? []).map((r) => [r.id, r.name]));
  const shown = new Set(summary.columns.filter((c) => c.available && !(hideOpenQuestions && c.column === "open_questions")).map((c) => c.column));
  const absent = summary.columns.filter((c) => !c.available && !(hideOpenQuestions && c.column === "open_questions"));
  const closestChange = analyst ? closestChangePlain : closestChangeExact;
  return <div className="screening-summary">
    <div className="table-scroll" tabIndex={0}>
      <table className="grid" aria-label="Screening summary">
        <thead><tr>
          <th scope="col">Rank</th><th scope="col">Alternative</th><th scope="col">Overall value</th>
          {shown.has("classes") && <th scope="col">Class (margin)</th>}
          {shown.has("requirements") && <th scope="col">Requirements</th>}
          <th scope="col">Largest contributions</th>
          {shown.has("fragility") && <th scope="col" title={summary.fragility_definition}>Closest change</th>}
          {shown.has("open_questions") && <th scope="col"><QuestionsHeader source={questions}/></th>}
        </tr></thead>
        <tbody>{summary.rows.map((row) => <tr key={row.alternative_id} {...companyRow(exploration, row.alternative_id)}>
          <td>{row.rank}</td>
          <th scope="row"><CompanyName id={row.alternative_id}>{row.name}</CompanyName></th>
          <td><ValueText value={row.value}/></td>
          {shown.has("classes") && <td>{row.class_name}{row.margin != null && <> (<ValueText value={row.margin}/>)</>}</td>}
          {shown.has("requirements") && <td data-requirements={row.requirements?.status}>{requirementsText(row, requirementNames)}</td>}
          <td>{row.drivers.length ? row.drivers.map((driver, index) => <span className="summary-driver" key={driver.criterion_id}><RatingHint
            text={ratingExplanation(project, row.alternative_id, driver.criterion_id, driver.contribution)}>{driver.name} <ValueText value={driver.contribution}/>{index < row.drivers.length - 1 ? ";" : ""}</RatingHint>{index < row.drivers.length - 1 ? " " : ""}</span>) : "none"}</td>
          {shown.has("fragility") && <td>{row.fragility ? closestChange(row.fragility, names) : ""}</td>}
          {shown.has("open_questions") && <td>{row.open_questions}</td>}
        </tr>)}</tbody>
      </table>
    </div>
    {analyst && shown.has("fragility") && <p className="muted small summary-note">{POINTS_NOTE}</p>}
    <Caveat id="screening" caption="Preference values under this profile; not probabilities or predictions.">{summary.label} {summary.fragility_definition}
      {!hideOpenQuestions && summary.project_open_questions ? ` ${String(summary.project_open_questions)} open question(s) concern the whole project.` : ""}</Caveat>
    {absent.length > 0 && <p className="muted small" aria-label="Summary columns not shown">Not shown: {absent.map((c) => `${COLUMN_LABELS[c.column]} (${c.reason})`).join("; ")}</p>}
  </div>;
}
