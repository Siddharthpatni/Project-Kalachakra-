/**
 * Reference chart — real output from the Python astro engine
 * (kalachakra.astro) for Vadodara, India, 2002-11-13 12:37 IST, Lahiri sidereal.
 * Personal identifiers (name, profession) are NOT stored here; they live only
 * in the gitignored data/raw/ record.
 */

export interface Graha {
  short: string;
  en: string;
  lon: number; // sidereal ecliptic longitude, degrees
  sign: string;
  nak: string;
  house: number;
  retro: boolean;
  color: string;
}

export interface DashaPeriod {
  lord: string;
  start: number;
  years: number;
  color: string;
  now?: boolean;
}

export const ASCENDANT = 288.56; // Capricorn

export const SIGN_ABBR: string[] = [
  "Ar", "Ta", "Ge", "Cn", "Le", "Vi", "Li", "Sc", "Sg", "Cp", "Aq", "Pi",
];

export const GRAHAS: Graha[] = [
  { short: "Su", en: "Sun", lon: 206.83, sign: "Libra", nak: "Vishakha", house: 10, retro: false, color: "#f59e0b" },
  { short: "Mo", en: "Moon", lon: 312.91, sign: "Aquarius", nak: "Shatabhisha", house: 2, retro: false, color: "#38bdf8" },
  { short: "Ma", en: "Mars", lon: 174.36, sign: "Virgo", nak: "Chitra", house: 9, retro: false, color: "#ef4444" },
  { short: "Me", en: "Mercury", lon: 206.3, sign: "Libra", nak: "Vishakha", house: 10, retro: false, color: "#22c55e" },
  { short: "Ju", en: "Jupiter", lon: 113.49, sign: "Cancer", nak: "Ashlesha", house: 7, retro: false, color: "#f97316" },
  { short: "Ve", en: "Venus", lon: 187.48, sign: "Libra", nak: "Swati", house: 10, retro: true, color: "#ec4899" },
  { short: "Sa", en: "Saturn", lon: 64.23, sign: "Gemini", nak: "Mrigashira", house: 6, retro: true, color: "#8b5cf6" },
  { short: "Ra", en: "Rahu", lon: 45.72, sign: "Taurus", nak: "Rohini", house: 5, retro: true, color: "#a855f7" },
  { short: "Ke", en: "Ketu", lon: 225.72, sign: "Scorpio", nak: "Anuradha", house: 11, retro: true, color: "#6366f1" },
];

export const DASHAS: DashaPeriod[] = [
  { lord: "Rahu", start: 2002, years: 9.57, color: "#a855f7" },
  { lord: "Jupiter", start: 2012, years: 16, color: "#f97316", now: true },
  { lord: "Saturn", start: 2028, years: 19, color: "#8b5cf6" },
  { lord: "Mercury", start: 2047, years: 17, color: "#22c55e" },
  { lord: "Ketu", start: 2064, years: 7, color: "#6366f1" },
  { lord: "Venus", start: 2071, years: 20, color: "#ec4899" },
  { lord: "Sun", start: 2091, years: 6, color: "#f59e0b" },
  { lord: "Moon", start: 2097, years: 10, color: "#38bdf8" },
  { lord: "Mars", start: 2107, years: 7, color: "#ef4444" },
];
