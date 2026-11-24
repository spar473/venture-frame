import { useCallback } from "react";
import { api } from "../api/client";
import type { Project } from "../api/types";
import { analysisKey, canonical, type AnalysisContext } from "./analysisContext";
import { useAnalysisRequest } from "./useAnalysisRequest";

/** Rank ranges and overlap groups for one revision, profile and scenario. Python computes every
 * range and group; a response for another context is never shown, and an older request is
 * aborted when the context changes. Requested only once the summary itself is ready. */
export function useScreeningRobustness(project: Project, context: AnalysisContext, enabled: boolean) {
  const key = analysisKey({ ...context, criterionId: null }, "screening-robustness");
  const load = useCallback(async (signal: AbortSignal) => {
    const result = await api.screeningRobustness(project, context.configuration, signal);
    if (result.project_fingerprint !== context.fingerprint || canonical(result.configuration) !== canonical(context.configuration) ||
        (result.available && (result.weight_profile_id !== context.resolvedProfileId || result.scenario_id !== context.scenarioId))) {
      throw new Error("Rank ranges do not match the selected revision and configuration.");
    }
    return result;
  }, [project, context]);
  return useAnalysisRequest(key, enabled && project.alternatives.length > 0, load);
}
