/**
 * Product configuration: the name, wording and display settings of Venture Frame, built on the
 * domain-neutral `mcda_studio` engine. Wording only; nothing here enters a calculation or an export.
 */

export const PRODUCT_NAME = "Venture Frame";

export const PRODUCT_TAGLINE = "Decision support for early-stage venture screening";

/** Starter ids shown as cards on the start page, in this order. Every starter stays available in
 * New project… → Starter gallery. */
export const START_PAGE_STARTERS: readonly string[] = ["blank", "venture_screening_study", "venture_screening"];

/** Decimal places for values, margins and contributions in the Analyst view only. */
export const ANALYST_DECIMALS = 3;
