import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
// Kalachakra research console — Vite + React + TS.
export default defineConfig({
    plugins: [react()],
    server: { port: 5173 },
});
