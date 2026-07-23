import { useEffect, useRef } from "react";
import { ASCENDANT, GRAHAS, SIGN_ABBR } from "../data/referenceChart";

/**
 * A clean, static birth-chart ring rendered on canvas: concentric rings, the
 * twelve signs, the nine planets at their real sidereal longitudes (English
 * two-letter labels), and the ascendant marker. Theme-aware — colours are read
 * from CSS tokens and it redraws when the theme changes.
 */
export function BirthChartRing() {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const css = (name: string): string =>
      getComputedStyle(document.body).getPropertyValue(name).trim();

    const draw = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const size = canvas.getBoundingClientRect().width || 360;
      canvas.width = size * dpr;
      canvas.height = size * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const cx = size / 2, cy = size / 2, R = size * 0.42;
      const ringC = css("--chart-ring") || "rgba(0,0,0,0.15)";
      const tickC = css("--chart-tick") || "rgba(0,0,0,0.3)";
      const labelC = css("--chart-label") || "rgba(60,64,102,0.85)";
      const accent = css("--accent") || "#5b5bf0";
      const mono = css("--mono") || "monospace";
      const sans = css("--sans") || "sans-serif";
      const ang = (lon: number): number => Math.PI - (lon * Math.PI) / 180;

      ctx.clearRect(0, 0, size, size);
      [R, R * 0.72, R * 0.5].forEach((rr, i) => {
        ctx.beginPath();
        ctx.arc(cx, cy, rr, 0, Math.PI * 2);
        ctx.strokeStyle = ringC;
        ctx.lineWidth = i === 0 ? 1.5 : 1;
        ctx.stroke();
      });
      for (let s = 0; s < 12; s++) {
        const a0 = ang(s * 30);
        ctx.beginPath();
        ctx.moveTo(cx + R * 0.72 * Math.cos(a0), cy + R * 0.72 * Math.sin(a0));
        ctx.lineTo(cx + R * Math.cos(a0), cy + R * Math.sin(a0));
        ctx.strokeStyle = tickC;
        ctx.lineWidth = 1;
        ctx.stroke();
        const am = ang(s * 30 + 15), lr = R * 0.86;
        ctx.fillStyle = labelC;
        ctx.font = `600 11px ${mono}`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(SIGN_ABBR[s], cx + lr * Math.cos(am), cy + lr * Math.sin(am));
      }
      const aa = ang(ASCENDANT);
      ctx.beginPath();
      ctx.moveTo(cx + R * 0.5 * Math.cos(aa), cy + R * 0.5 * Math.sin(aa));
      ctx.lineTo(cx + R * Math.cos(aa), cy + R * Math.sin(aa));
      ctx.strokeStyle = accent;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = accent;
      ctx.font = `700 10px ${mono}`;
      ctx.fillText("ASC", cx + R * 1.06 * Math.cos(aa), cy + R * 1.06 * Math.sin(aa));

      const pr = R * 0.61;
      for (const p of GRAHAS) {
        const a = ang(p.lon), x = cx + pr * Math.cos(a), y = cy + pr * Math.sin(a);
        ctx.beginPath();
        ctx.arc(x, y, 4.5, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.shadowColor = p.color;
        ctx.shadowBlur = 10;
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.fillStyle = labelC;
        ctx.font = `600 11px ${sans}`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(p.short + (p.retro ? "℞" : ""), cx + R * 0.42 * Math.cos(a), cy + R * 0.42 * Math.sin(a));
      }
    };

    draw();
    window.addEventListener("resize", draw);
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    media.addEventListener("change", draw);
    const observer = new MutationObserver(draw);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

    return () => {
      window.removeEventListener("resize", draw);
      media.removeEventListener("change", draw);
      observer.disconnect();
    };
  }, []);

  return <canvas ref={ref} style={{ width: "100%", aspectRatio: "1 / 1", display: "block" }} aria-label="Birth chart ring: nine planets at their sidereal longitudes." />;
}
