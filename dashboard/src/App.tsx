import { BirthChartRing } from "./components/BirthChartRing";
import { PredictionPipeline } from "./components/PredictionPipeline";
import { DASHAS, GRAHAS } from "./data/referenceChart";

function hexA(hex: string, a: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

export default function App() {
  return (
    <>
      <div className="bg"><div className="blob a" /><div className="blob b" /><div className="blob c" /></div>

      <header style={{ position: "sticky", top: 12, zIndex: 40, marginTop: 12 }}>
        <div className="wrap">
          <div className="glass" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 16px", borderRadius: 999 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 9, fontWeight: 700, color: "var(--ink)" }}>
              <span style={{ width: 22, height: 22, borderRadius: "50%", background: "conic-gradient(from 210deg, var(--accent), var(--accent-2), var(--accent-3), var(--accent))" }} />
              Kalachakra
            </div>
            <span className="mono" style={{ fontSize: 11.5, color: "var(--muted)" }}>50-Domain Prediction Engine</span>
          </div>
        </div>
      </header>

      <main>
        <section style={{ padding: "64px 0 24px" }}>
          <div className="wrap">
            <span className="eyebrow">50-Domain Future-Prediction Engine</span>
            <h1 style={{ fontSize: "clamp(44px,8vw,80px)", lineHeight: 0.98, letterSpacing: "-0.03em", color: "var(--ink)", margin: "16px 0 0", fontWeight: 700 }}>
              One forecast,<br />
              <span style={{ background: "linear-gradient(120deg,var(--accent),var(--accent-2) 55%,var(--accent-3))", WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent" }}>
                summed from fifty domains.
              </span>
            </h1>
            <p style={{ fontSize: "clamp(16px,2vw,20px)", color: "var(--text)", maxWidth: "54ch", marginTop: 18 }}>
              Birth data enters as one signal among many. Fifty computational domains each emit a graded signal; a mutual-information-gated, weighted ensemble sums them into a single calibrated probability with a timing window — trusted only when it beats the baselines.
            </p>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 24 }}>
              {["Subject · Vadodara, India", "Born · 13 Nov 2002 · 12:37 IST", "Ascendant · Capricorn", "Current period · Jupiter"].map((c) => (
                <span key={c} className="glass" style={{ fontSize: 12.5, color: "var(--text)", borderRadius: 999, padding: "7px 13px" }}>{c}</span>
              ))}
            </div>
          </div>
        </section>

        <PredictionPipeline />

        {/* chart */}
        <section style={{ padding: "56px 0" }}>
          <div className="wrap">
            <span className="eyebrow">The Input · Domain 2</span>
            <h2 style={{ fontSize: "clamp(24px,3.6vw,34px)", letterSpacing: "-0.02em", color: "var(--ink)", margin: "10px 0 24px", fontWeight: 700 }}>Your birth chart, computed.</h2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
              <div className="glass" style={{ padding: 24 }}><BirthChartRing /></div>
              <div className="glass" style={{ padding: 24, overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                  <thead><tr>{["Planet", "Longitude", "Sign", "Nakshatra", "House"].map((h) => (
                    <th key={h} className="mono" style={{ textAlign: "left", fontSize: 10, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--muted)", padding: "6px 8px", borderBottom: "1px solid var(--hair)" }}>{h}</th>
                  ))}</tr></thead>
                  <tbody>{GRAHAS.map((p) => (
                    <tr key={p.en}>
                      <td style={{ padding: "7px 8px", borderBottom: "1px solid var(--hair)", color: "var(--ink)", fontWeight: 600 }}>
                        <span style={{ display: "inline-block", width: 9, height: 9, borderRadius: 3, marginRight: 8, background: p.color }} />{p.en}{p.retro ? " ℞" : ""}
                      </td>
                      <td className="mono" style={{ padding: "7px 8px", borderBottom: "1px solid var(--hair)", color: "var(--text)" }}>{p.lon.toFixed(2)}°</td>
                      <td style={{ padding: "7px 8px", borderBottom: "1px solid var(--hair)", color: "var(--text)" }}>{p.sign}</td>
                      <td style={{ padding: "7px 8px", borderBottom: "1px solid var(--hair)", color: "var(--text)" }}>{p.nak}</td>
                      <td style={{ padding: "7px 8px", borderBottom: "1px solid var(--hair)", color: "var(--text)" }}>{p.house}</td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
            </div>
          </div>
        </section>

        {/* timing */}
        <section style={{ padding: "56px 0" }}>
          <div className="wrap">
            <span className="eyebrow">The "When" · Domain 2</span>
            <h2 style={{ fontSize: "clamp(24px,3.6vw,34px)", letterSpacing: "-0.02em", color: "var(--ink)", margin: "10px 0 24px", fontWeight: 700 }}>Timing comes from the dasha wheel.</h2>
            <div style={{ display: "flex", width: "100%", borderRadius: 14, overflow: "hidden", border: "1px solid var(--hair-2)" }}>
              {DASHAS.map((d) => (
                <div key={d.lord} style={{ position: "relative", flex: `${d.years} 0 0`, padding: "15px 10px 13px", borderRight: "1px solid var(--hair)", display: "flex", flexDirection: "column", gap: 3, minWidth: 0, background: `linear-gradient(180deg,${hexA(d.color, 0.22)},${hexA(d.color, 0.05)})`, outline: d.now ? "2px solid var(--accent)" : "none", outlineOffset: -2 }}>
                  <span style={{ fontSize: 14, color: "var(--ink)", fontWeight: 600 }}>{d.lord}</span>
                  <span className="mono" style={{ fontSize: 10, color: "var(--muted)" }}>{d.start} · {d.years}y</span>
                </div>
              ))}
            </div>
          </div>
        </section>

        <footer style={{ padding: "48px 0 64px" }}>
          <div className="wrap">
            <div className="glass" style={{ padding: "22px 24px", fontSize: 14, color: "var(--text)" }}>
              <b style={{ color: "var(--ink)" }}>Scientific disclaimer.</b> A research instrument, not an oracle. It estimates whether specific encodings of astronomical configurations carry statistically detectable predictive information, and reports every forecast with calibrated uncertainty against strong baselines. A single chart is a demonstration; hypothesis testing requires a dataset.
            </div>
          </div>
        </footer>
      </main>
    </>
  );
}
