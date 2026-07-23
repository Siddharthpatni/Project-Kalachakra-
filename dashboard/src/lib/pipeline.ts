/**
 * The 50-domain aggregation contract.
 *
 * This is the "proper sum": probabilities are not additive, so each domain
 * layer contributes a signed nudge to the log-odds. Contributions are gated
 * (a domain with mutual information below the noise floor contributes zero),
 * weighted by out-of-sample skill, summed onto a baseline prior, and mapped to
 * a calibrated probability. The Python meta-model (a stacked ensemble over the
 * domain outputs) mirrors this same contract server-side.
 */

export type LayerId = "L1" | "L2" | "L3" | "L4" | "L5" | "L6";

/** One of the six domain layers the fifty domains are grouped into. */
export interface DomainLayer {
  id: LayerId;
  name: string;
  /** Human-readable count of domains folded into this layer, e.g. "10 dom". */
  count: string;
  /** True if the layer's mutual information cleared the noise floor. */
  gatePassed: boolean;
  /** Signed contribution to the aggregate log-odds (already gated + weighted). */
  dz: number;
}

/** A calibrated forecast: how likely, when, and whether to trust it. */
export interface Forecast {
  /** Calibrated probability in [0, 1]. */
  p: number;
  /** Aggregate log-odds z = b + Σ dz. */
  logit: number;
  /** Confidence interval on p. */
  ci: [number, number];
  /** Baseline probability (e.g. XGBoost on the same features). */
  baselineP: number;
  /** p − baselineP. */
  deltaP: number;
  /** MI gate tally across layers. */
  gate: { passed: number; total: number };
  /** Timing window, read from the Vimshottari dasha (Domain 2). */
  window: string;
}

export const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z));
export const logit = (p: number): number => Math.log(p / (1 - p));

/**
 * Aggregate layer contributions into a calibrated forecast.
 *
 * @param layers      The six domain layers with their signed log-odds nudges.
 * @param baseLogit   Baseline prior log-odds (b).
 * @param ciHalfWidth Half-width of the confidence band, in probability units.
 * @param window      Timing window label from the dasha engine.
 */
export function aggregate(
  layers: DomainLayer[],
  baseLogit: number,
  ciHalfWidth: number,
  window: string,
): Forecast {
  const sumDz = layers.reduce((acc, l) => acc + l.dz, 0);
  const z = baseLogit + sumDz;
  const p = sigmoid(z);
  const baselineP = sigmoid(baseLogit);
  const lo = Math.max(0, p - ciHalfWidth);
  const hi = Math.min(1, p + ciHalfWidth);
  const passed = layers.filter((l) => l.gatePassed).length;
  return {
    p,
    logit: z,
    ci: [lo, hi],
    baselineP,
    deltaP: p - baselineP,
    gate: { passed, total: layers.length },
    window,
  };
}
