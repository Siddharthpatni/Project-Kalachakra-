/**
 * Reference chart data — real output from the Python astro engine
 * (kalachakra.astro) for Delhi, 1990-03-15 10:30 IST, Lahiri sidereal.
 * Regenerate with: `python -m kalachakra.astro` style scripts.
 */

export interface Graha {
  dev: string;
  en: string;
  lon: number; // sidereal ecliptic longitude, degrees
  color: string;
  retro: boolean;
}

export interface DashaPeriod {
  dev: string;
  en: string;
  years: number;
  start: number;
  now?: boolean;
}

export const GRAHAS: Graha[] = [
  { dev: "सूर्य", en: "Sun", lon: 330.63, color: "#ffcf5a", retro: false },
  { dev: "चन्द्र", en: "Moon", lon: 192.55, color: "#e7e2f5", retro: false },
  { dev: "मङ्गल", en: "Mars", lon: 278.9, color: "#e5433b", retro: false },
  { dev: "बुध", en: "Mercury", lon: 326.94, color: "#8fe38f", retro: false },
  { dev: "गुरु", en: "Jupiter", lon: 67.64, color: "#ff9a3d", retro: false },
  { dev: "शुक्र", en: "Venus", lon: 285.16, color: "#8fd8ff", retro: false },
  { dev: "शनि", en: "Saturn", lon: 269.59, color: "#9c8bd6", retro: false },
  { dev: "राहु", en: "Rahu", lon: 290.86, color: "#b98cff", retro: true },
  { dev: "केतु", en: "Ketu", lon: 110.86, color: "#c8a0ff", retro: true },
];

export const RASHIS: string[] = [
  "Me", "Vr", "Mi", "Ka", "Si", "Kn", "Tu", "Vc", "Dh", "Mk", "Km", "Mn",
];

export const DASHAS: DashaPeriod[] = [
  { dev: "राहु", en: "RAHU", years: 10.06, start: 1990 },
  { dev: "गुरु", en: "GURU", years: 16, start: 2000 },
  { dev: "शनि", en: "SHANI", years: 19, start: 2016, now: true },
  { dev: "बुध", en: "BUDHA", years: 17, start: 2035 },
  { dev: "केतु", en: "KETU", years: 7, start: 2052 },
  { dev: "शुक्र", en: "SHUKRA", years: 20, start: 2059 },
  { dev: "सूर्य", en: "SURYA", years: 6, start: 2079 },
  { dev: "चन्द्र", en: "CHANDRA", years: 10, start: 2085 },
  { dev: "मङ्गल", en: "MANGALA", years: 7, start: 2095 },
];
