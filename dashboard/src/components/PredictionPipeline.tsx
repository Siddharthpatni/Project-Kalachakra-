import { aggregate, type DomainLayer } from "../lib/pipeline";

/** The six layers the fifty domains fold into, with illustrative gated,
 *  weighted contributions to the aggregate log-odds. Real values come from
 *  the Python meta-model (stacked ensemble over domain outputs). */
const LAYERS: DomainLayer[] = [
  { id: "L1", name: "Foundation & Features", count: "8 dom", gatePassed: true, dz: 0.3 },
  { id: "L2", name: "Predictive Core", count: "10 dom", gatePassed: true, dz: 0.24 },
  { id: "L3", name: "Reasoning & Causality", count: "8 dom", gatePassed: true, dz: 0.1 },
  { id: "L4", name: "Interpretation & Uncertainty", count: "6 dom", gatePassed: true, dz: -0.08 },
  { id: "L5", name: "Meta & Optimization", count: "10 dom", gatePassed: false, dz: 0.06 },
  { id: "L6", name: "Rigor & Aggregation", count: "9 dom", gatePassed: false, dz: -0.13 },
];

const BASE_LOGIT = 0.405; // baseline prior ≈ 0.60

function Gauge({ p, ci, baselineP }: { p: number; ci: [number, number]; baselineP: number }) {
  const cx = 130, cy = 124, r = 96, sw = 16;
  const pt = (v: number): [number, number] => {
    const a = Math.PI * (1 - v);
    return [cx + r * Math.cos(a), cy - r * Math.sin(a)];
  };
  const arc = (a: number, b: number): string => {
    const s = Math.PI * (1 - a), e = Math.PI * (1 - b);
    return `M ${cx + r * Math.cos(s)} ${cy - r * Math.sin(s)} A ${r} ${r} 0 ${b - a > 0.5 ? 1 : 0} 1 ${cx + r * Math.cos(e)} ${cy - r * Math.sin(e)}`;
  };
  const base = pt(baselineP);
  return (
    <svg viewBox="0 0 260 150" style={{ width: 260, maxWidth: "100%", height: "auto" }} role="img" aria-label={`Calibrated probability ${p.toFixed(2)}`}>
      <path d={arc(0, 1)} fill="none" stroke="rgba(139,130,166,0.18)" strokeWidth={sw} strokeLinecap="round" />
      <path d={arc(ci[0], ci[1])} fill="none" stroke="rgba(91,214,232,0.22)" strokeWidth={sw} />
      <path d={arc(0, p)} fill="none" stroke="#5bd6e8" strokeWidth={sw} strokeLinecap="round" />
      <circle cx={base[0]} cy={base[1]} r={4} fill="#e8b24c" />
      <text x={130} y={112} textAnchor="middle" fill="#efeafa" fontFamily="var(--sans)" fontSize={46} fontWeight={600}>{p.toFixed(2)}</text>
      <text x={130} y={134} textAnchor="middle" fill="#8b82a6" fontFamily="var(--mono)" fontSize={10.5} letterSpacing={1}>{`CALIBRATED P · CI ${ci[0].toFixed(2)}–${ci[1].toFixed(2)}`}</text>
    </svg>
  );
}

/** The pipeline console: layers emit signals, the aggregator sums them into a
 *  calibrated forecast with a timing window. */
export function PredictionPipeline() {
  const forecast = aggregate(LAYERS, BASE_LOGIT, 0.12, "2026–2028 · Jupiter period");
  const maxAbs = 0.32;

  return (
    <section style={{ padding: "72px 0", borderTop: "1px solid var(--line)" }}>
      <div className="wrap">
        <span className="eyebrow">The Proper Sum</span>
        <h2 style={{ fontFamily: "var(--sans)", color: "var(--ink)", fontSize: "clamp(24px,4vw,36px)", margin: "10px 0 28px" }}>
          Fifty domains, one calibrated probability.
        </h2>

        <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 24 }}>
          {/* layer contributions */}
          <div style={{ display: "grid", gap: 10 }}>
            {LAYERS.map((l) => {
              const w = Math.min(50, (50 * Math.abs(l.dz)) / maxAbs);
              const pos = l.dz >= 0;
              return (
                <div key={l.id} style={{ display: "grid", gridTemplateColumns: "180px 1fr 88px", alignItems: "center", gap: 12 }}>
                  <span style={{ fontSize: 13, color: "var(--text)" }}>
                    {l.name} <span className="mono" style={{ color: "var(--faint)", fontSize: 11 }}>· {l.count}</span>
                  </span>
                  <div style={{ position: "relative", height: 20 }}>
                    <div style={{ position: "absolute", left: "50%", top: -2, bottom: -2, width: 1, background: "var(--line-2)" }} />
                    <div style={{ position: "absolute", top: 3, height: 14, borderRadius: 4, left: pos ? "50%" : `${50 - w}%`, width: `${w}%`, background: pos ? "linear-gradient(90deg,rgba(91,214,232,0.4),var(--cyan))" : "linear-gradient(90deg,var(--vermilion),rgba(229,67,59,0.4))" }} />
                  </div>
                  <span className="mono" style={{ fontSize: 12, textAlign: "right", color: pos ? "var(--cyan)" : "var(--vermilion)" }}>
                    {pos ? "+" : ""}{l.dz.toFixed(2)} · {l.gatePassed ? "pass" : "gated"}
                  </span>
                </div>
              );
            })}
          </div>

          {/* forecast output */}
          <div style={{ display: "grid", gridTemplateColumns: "300px 1fr", gap: 22, alignItems: "center", border: "1px solid var(--line-2)", borderRadius: "var(--r)", padding: 24, background: "linear-gradient(180deg,rgba(18,10,40,0.66),rgba(10,7,20,0.5))" }}>
            <div style={{ display: "grid", placeItems: "center" }}>
              <Gauge p={forecast.p} ci={forecast.ci} baselineP={forecast.baselineP} />
            </div>
            <div className="mono" style={{ display: "grid", gap: 12, fontSize: 13, color: "var(--text)" }}>
              <div>logit z = b + Σ Δz = <b style={{ color: "var(--moonlight)" }}>{forecast.logit.toFixed(2)}</b></div>
              <div>P = σ(z) = <b style={{ color: "var(--cyan)" }}>{forecast.p.toFixed(2)}</b> · CI [{forecast.ci[0].toFixed(2)}–{forecast.ci[1].toFixed(2)}]</div>
              <div>window: <b style={{ color: "var(--gold)" }}>{forecast.window}</b></div>
              <div>vs baseline: +{forecast.deltaP.toFixed(2)} (p₀ = {forecast.baselineP.toFixed(2)})</div>
              <div>MI gate: {forecast.gate.passed}/{forecast.gate.total} layers passed</div>
            </div>
          </div>
        </div>
        <p style={{ fontSize: 12.5, color: "var(--muted)", marginTop: 14 }}>
          Illustrative. Weights are learned by out-of-sample skill; the gate zeroes any domain whose mutual information is below the noise floor — including astrological ones. Strong pipeline, honest effect size.
        </p>
      </div>
    </section>
  );
}
