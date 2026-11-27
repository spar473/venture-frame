import { useCallback } from "react";
import { api } from "../api/client";
import type { Project } from "../api/types";
import { analysisKey, type AnalysisContext } from "./analysisContext";
import { useAnalysisRequest } from "./useAnalysisRequest";

/** The engine's checks of the viewed profile's structured judgement, and its proposals. A
 * response for another revision or profile is never shown. */
export function useElicitationCheck(project: Project, context: AnalysisContext) {
  const key = analysisKey({ ...context, criterionId: null }, "elicitation-check");
  const load = useCallback(async (signal: AbortSignal) => {
    const result = await api.elicitationCheck(project, context.configuration, signal);
    if (result.project_fingerprint !== context.fingerprint || result.profile_id !== context.resolvedProfileId) {
      throw new Error("The judgement checks do not match the selected revision and profile.");
    }
    return result;
  }, [project, context]);
  return useAnalysisRequest(key, project.weight_profiles.length > 0, load);
}

/** Decision confidence now and under each proposal: requested only when asked for. */
export function useElicitationPreview(project: Project, context: AnalysisContext, enabled: boolean) {
  const key = analysisKey({ ...context, criterionId: null }, "elicitation-preview");
  const load = useCallback(async (signal: AbortSignal) => {
    const result = await api.elicitationPreview(project, context.configuration, signal);
    if (result.project_fingerprint !== context.fingerprint || result.profile_id !== context.resolvedProfileId) {
      throw new Error("The decision-confidence preview does not match the selected revision and profile.");
    }
    return result;
  }, [project, context]);
  return useAnalysisRequest(key, enabled && project.weight_profiles.length > 0, load, 0);
}

/** Aids for stating weights (anchored swings, break-even weights, reference profiles), all
 * derived by the engine for the viewed revision and profile. */
export function useJudgementAids(project: Project, context: AnalysisContext) {
  const key = analysisKey({ ...context, criterionId: null }, "judgement-aids");
  const load = useCallback(async (signal: AbortSignal) => {
    const result = await api.judgementAids(project, context.configuration, signal);
    if (result.project_fingerprint !== context.fingerprint || result.weight_profile_id !== context.resolvedProfileId) {
      throw new Error("The judgement aids do not match the selected revision and profile.");
    }
    return result;
  }, [project, context]);
  return useAnalysisRequest(key, project.weight_profiles.length > 0, load);
}
