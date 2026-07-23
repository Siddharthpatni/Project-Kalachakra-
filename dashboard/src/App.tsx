import { KalachakraWheel } from "./components/KalachakraWheel";
import { PredictionPipeline } from "./components/PredictionPipeline";
import { DASHAS } from "./data/referenceChart";

const lordColor: Record<string, string> = {
  RAHU: "#b98cff", GURU: "#ff9a3d", SHANI: "#9c8bd6", BUDHA: "#8fe38f", KETU: "#c8a0ff",
  SHUKRA: "#8fd8ff", SURYA: "#ffcf5a", CHANDRA: "#e7e2f5", MANGALA: "#e5433b",
};
const hexA = (hex: string, a: number): string => {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
};

export default function App() {
  return (
    <>
      <header style={{ position: "sticky", top: 0, zIndex: 30, backdropFilter: "blur(10px)", background: "linear-gradient(to bottom,rgba(8,6,15,0.86),rgba(8,6,15,0.25))", borderBottom: "1px solid var(--line)" }}>
        <div className="wrap" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", height: 60 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
            <span className="deva" style={{ fontSize: 22, color: "var(--ink)" }}>कालचक्र</span>
            <span className="mono" style={{ fontSize: 11, letterSpacing: "0.3em", color: "var(--muted)", textTransform: "uppercase" }}>Kalachakra</span>
          </div>
          <span className="mono" style={{ fontSize: 11, color: "var(--muted)" }}>50-Domain Prediction Pipeline</span>
        </div>
      </header>

      {/* hero */}
      <section style={{ position: "relative", minHeight: "88svh", display: "grid", placeItems: "center", overflow: "hidden" }}>
        <KalachakraWheel />
        <div style={{ position: "absolute", inset: 0, pointerEvents: "none", background: "radial-gradient(62% 62% at 50% 46%, transparent 38%, rgba(8,6,15,0.62) 100%)" }} />
        <div style={{ position: "relative", zIndex: 5, textAlign: "center", padding: "0 24px", maxWidth: 820, pointerEvents: "none" }}>
          <span className="eyebrow">Fifty-Domain Future-Prediction Pipeline</span>
          <h1 className="deva" style={{ fontWeight: 600, color: "var(--ink)", fontSize: "clamp(46px,10.5vw,116px)", lineHeight: 0.92, margin: "18px 0 0", textShadow: "0 0 44px rgba(120,70,220,0.5)" }}>
            कालचक्र
            <span style={{ display: "block", fontFamily: "var(--serif)", fontStyle: "italic", fontWeight: 500, fontSize: "clamp(17px,3vw,27px)", color: "var(--gold)", marginTop: 14, textShadow: "none" }}>
              one forecast, summed from fifty domains
            </span>
          </h1>
          <p style={{ margin: "20px auto 0", maxWidth: "60ch", color: "var(--text)", fontSize: "clamp(15px,1.9vw,19px)" }}>
            Birth data enters as one signal among many. Fifty computational domains each emit a graded signal; a gated, weighted ensemble sums them into a single calibrated probability with a timing window — trusted only when it beats the baselines.
          </p>
        </div>
      </section>

      <PredictionPipeline />

      {/* dasha timing */}
      <section style={{ padding: "72px 0", borderTop: "1px solid var(--line)" }}>
        <div className="wrap">
          <span className="eyebrow">The "When" · Domain 2</span>
          <h2 style={{ fontFamily: "var(--serif)", color: "var(--ink)", fontSize: "clamp(24px,4vw,36px)", margin: "10px 0 24px" }}>
            The timing window is read off the wheel.
          </h2>
          <div style={{ display: "flex", width: "100%", borderRadius: "var(--r)", overflow: "hidden", border: "1px solid var(--line-2)" }}>
            {DASHAS.map((d) => (
              <div key={d.en} style={{ position: "relative", flex: `${d.years} 0 0`, padding: "16px 10px 14px", borderRight: "1px solid rgba(8,6,15,0.6)", display: "flex", flexDirection: "column", gap: 3, minWidth: 0, background: `linear-gradient(180deg,${hexA(lordColor[d.en], 0.32)},${hexA(lordColor[d.en], 0.08)})`, outline: d.now ? "2px solid var(--gold)" : "none", outlineOffset: -2 }}>
                <span className="deva" style={{ fontSize: 16, color: "var(--ink)" }}>{d.dev}</span>
                <span className="mono" style={{ fontSize: 9, letterSpacing: "0.12em", color: "rgba(255,255,255,0.6)" }}>{d.en}</span>
                <span className="mono" style={{ fontSize: 10, color: "rgba(255,255,255,0.55)" }}>{d.start} · {d.years}y</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <footer style={{ borderTop: "1px solid var(--line)", padding: "40px 0 60px" }}>
        <div className="wrap mono" style={{ fontSize: 11.5, color: "var(--faint)", display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 16 }}>
          <span>कालचक्र · Kalachakra — research instrument, not an oracle. H₀ until the data says otherwise.</span>
          <span>MI-gated weighted ensemble · isotonic calibration</span>
        </div>
      </footer>
    </>
  );
}
