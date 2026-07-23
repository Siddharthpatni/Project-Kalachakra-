import { useEffect, useRef } from "react";
import { GRAHAS, RASHIS } from "../data/referenceChart";

/**
 * The signature visual: a black hole whose accretion disk forms the Vedic
 * wheel of time. The disk is Doppler-colored (approaching edge blue/cyan,
 * receding edge warm/saffron); the nine grahas orbit at their real sidereal
 * longitudes; the whole wheel turns once per ~120s — the 120-year cycle.
 */
export function KalachakraWheel() {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let W = 0, H = 0, cx = 0, cy = 0, R = 60;
    const TILT = 0.4;

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rect = canvas.getBoundingClientRect();
      W = Math.max(320, rect.width);
      H = Math.max(360, rect.height);
      canvas.width = Math.floor(W * dpr);
      canvas.height = Math.floor(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      cx = W / 2;
      cy = H / 2;
      R = Math.max(44, Math.min(W, H) * 0.135);
    };

    const d2r = (d: number) => (d * Math.PI) / 180;
    const lonA = (l: number, s: number) => Math.PI - d2r(l) + s;

    const disk = (spin: number) => {
      const rIn = R * 1.16, rOut = R * 2.4, N = 200;
      ctx.save();
      ctx.translate(cx, cy);
      ctx.scale(1, TILT);
      for (let i = 0; i < N; i++) {
        const a0 = (i / N) * Math.PI * 2, a1 = ((i + 1.4) / N) * Math.PI * 2;
        const beam = Math.cos(a0 - Math.PI);
        const turb = 0.5 + 0.5 * Math.sin(a0 * 7 + spin * 6) * Math.sin(a0 * 3 - spin * 4);
        const b = Math.max(0.05, 0.32 + 0.68 * (beam * 0.5 + 0.5)) * (0.55 + 0.45 * turb);
        const g = ctx.createRadialGradient(0, 0, rIn, 0, 0, rOut);
        if (beam < 0) {
          g.addColorStop(0, `rgba(255,232,180,${b})`);
          g.addColorStop(0.4, `rgba(255,138,61,${b * 0.9})`);
          g.addColorStop(1, "rgba(229,67,59,0)");
        } else {
          g.addColorStop(0, `rgba(232,255,255,${b})`);
          g.addColorStop(0.4, `rgba(91,214,232,${b * 0.85})`);
          g.addColorStop(1, "rgba(120,90,220,0)");
        }
        ctx.beginPath();
        ctx.arc(0, 0, rOut, a0, a1);
        ctx.arc(0, 0, rIn, a1, a0, true);
        ctx.closePath();
        ctx.fillStyle = g;
        ctx.fill();
      }
      ctx.restore();
    };

    const photon = () => {
      ctx.save();
      ctx.translate(cx, cy);
      const g = ctx.createRadialGradient(0, 0, R * 0.9, 0, 0, R * 1.5);
      g.addColorStop(0, "rgba(0,0,0,0)");
      g.addColorStop(0.62, "rgba(255,180,90,0)");
      g.addColorStop(0.74, "rgba(255,206,120,0.55)");
      g.addColorStop(0.8, "rgba(255,240,210,0.9)");
      g.addColorStop(0.86, "rgba(255,206,120,0.35)");
      g.addColorStop(1, "rgba(255,180,90,0)");
      ctx.beginPath();
      ctx.arc(0, 0, R * 1.5, 0, Math.PI * 2);
      ctx.fillStyle = g;
      ctx.fill();
      ctx.restore();
    };

    const horizon = () => {
      ctx.save();
      ctx.translate(cx, cy);
      const g = ctx.createRadialGradient(0, 0, R * 0.2, 0, 0, R * 1.08);
      g.addColorStop(0, "#000");
      g.addColorStop(0.82, "#000");
      g.addColorStop(1, "rgba(4,3,10,0.2)");
      ctx.beginPath();
      ctx.arc(0, 0, R * 1.02, 0, Math.PI * 2);
      ctx.fillStyle = g;
      ctx.fill();
      ctx.restore();
    };

    const zodiac = (spin: number) => {
      const rMid = R * 2.82, rIn = R * 2.62, rOut = R * 2.98;
      ctx.save();
      ctx.translate(cx, cy);
      for (let s = 0; s < 12; s++) {
        const a0 = lonA(s * 30, spin), a1 = lonA((s + 1) * 30, spin);
        ctx.beginPath();
        ctx.arc(0, 0, rOut, a1, a0, true);
        ctx.arc(0, 0, rIn, a0, a1, false);
        ctx.closePath();
        ctx.fillStyle = s % 2 === 0 ? "rgba(178,160,232,0.045)" : "rgba(232,178,76,0.03)";
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(rIn * Math.cos(a0), rIn * Math.sin(a0));
        ctx.lineTo(rOut * Math.cos(a0), rOut * Math.sin(a0));
        ctx.strokeStyle = "rgba(232,178,76,0.3)";
        ctx.lineWidth = 1;
        ctx.stroke();
        const am = lonA(s * 30 + 15, spin);
        ctx.save();
        ctx.translate(rMid * Math.cos(am), rMid * Math.sin(am));
        ctx.fillStyle = "rgba(201,192,217,0.62)";
        ctx.font = "600 11px ui-monospace, Menlo, monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(RASHIS[s], 0, 0);
        ctx.restore();
      }
      ctx.restore();
    };

    const grahaRing = (spin: number) => {
      const rG = R * 2.12;
      ctx.save();
      ctx.translate(cx, cy);
      ctx.beginPath();
      ctx.arc(0, 0, rG, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(178,160,232,0.14)";
      ctx.lineWidth = 1;
      ctx.stroke();
      for (const p of GRAHAS) {
        const a = lonA(p.lon, spin), x = rG * Math.cos(a), y = rG * Math.sin(a);
        ctx.beginPath();
        ctx.arc(x, y, 4.2, 0, Math.PI * 2);
        ctx.shadowColor = p.color;
        ctx.shadowBlur = 16;
        ctx.fillStyle = p.color;
        ctx.fill();
        ctx.shadowBlur = 0;
        const lr = R * 2.28, lx = lr * Math.cos(a), ly = lr * Math.sin(a);
        ctx.fillStyle = "rgba(239,234,250,0.92)";
        ctx.font = "16px 'Devanagari Sangam MN','Noto Sans Devanagari', serif";
        ctx.textAlign = Math.cos(a) < -0.15 ? "end" : Math.cos(a) > 0.15 ? "start" : "center";
        ctx.textBaseline = "middle";
        ctx.fillText(p.dev, lx, ly);
      }
      ctx.restore();
    };

    let raf = 0;
    const t0 = performance.now();
    const frame = (now: number) => {
      const t = (now - t0) / 1000;
      const ls = reduce ? 0.6 : (t * Math.PI * 2) / 120;
      const ds = reduce ? 0 : t * 0.35;
      ctx.clearRect(0, 0, W, H);
      disk(ds);
      photon();
      horizon();
      zodiac(ls);
      grahaRing(ls);
      if (!reduce) raf = requestAnimationFrame(frame);
    };

    resize();
    const onResize = () => {
      resize();
      if (reduce) frame(performance.now());
    };
    window.addEventListener("resize", onResize);
    if (reduce) frame(performance.now());
    else raf = requestAnimationFrame(frame);

    return () => {
      window.removeEventListener("resize", onResize);
      cancelAnimationFrame(raf);
    };
  }, []);

  return <canvas ref={ref} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", display: "block" }} aria-label="Black-hole Kalachakra: the nine grahas orbit a lensed event horizon at their computed sidereal longitudes." />;
}
