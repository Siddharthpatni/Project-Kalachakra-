# Kalachakra Dashboard

The frontend for Project Kalachakra (Domains 46 / 47): a **50-domain
future-prediction console**. Birth data enters as one signal among many; fifty
computational domains each emit a graded signal; a mutual-information-gated,
weighted ensemble sums them into a single calibrated probability with a timing
window — trusted only when it beats the baselines.

Astrology is *one knowledge input*, never the arbiter. The MI gate can silence
any domain — including the astrological ones — to zero contribution.

## Stack

React + Vite + TypeScript. Single dark "cosmic" theme whose palette is the
Doppler physics of an accretion disk mapped onto an Indian gilt palette
(saffron/gold = receding edge, cyan = approaching edge).

## Run

```bash
cd dashboard
npm install
npm run dev      # http://localhost:5173
npm run build    # type-check + production build
```

## Layout

```
dashboard/
├── design/kalachakra.html          # full standalone reference design (open directly)
├── index.html
├── src/
│   ├── main.tsx
│   ├── App.tsx                      # hero (wheel) + pipeline + dasha timing
│   ├── components/
│   │   ├── KalachakraWheel.tsx      # canvas black-hole / zodiac / grahas
│   │   └── PredictionPipeline.tsx   # layer contributions → calibrated gauge
│   ├── lib/pipeline.ts             # the aggregation contract (shared with Python meta-model)
│   ├── data/referenceChart.ts      # real astro-engine output (Delhi 1990)
│   └── styles/tokens.css           # design tokens
```

## The aggregation contract

`src/lib/pipeline.ts` encodes the "proper sum": log-odds are additive, not
probabilities. Each domain layer contributes a signed, gated, weighted nudge
`Δz`; these sum onto a baseline prior `b`; the total passes through a calibrated
link `σ` to a probability with a confidence band:

```
P = σ( b + Σ_d gate_d · w_d · s_d )
```

The Python side (`kalachakra` meta-model, a stacked ensemble over domain
outputs) mirrors this same contract. The published design concept lives at the
project's artifact URL; `design/kalachakra.html` is the standalone copy.
