# Development Process

This project is built with a disciplined sprint loop rather than ad-hoc coding.
The methodology is adapted from **[gstack](https://github.com/garrytan/gstack)**
(Garry Tan's "AI software factory") — specifically its core idea of *process over
tools*: treat each stage of work as a distinct role with defined outputs that feed
the next stage, so nothing is skipped.

We adapt the **thinking**, not the tooling. gstack is a TypeScript/Bun system with
its own 23 skills and headless-browser QA; Kalachakra is a Python scientific
research codebase. What carries over is the loop and the role-based rigor, mapped
onto this project's own scientific-methodology (Domain 50), reproducibility
(Domain 43), and benchmarking (Domain 44) domains.

## The loop

```
Think  →  Plan  →  Build  →  Review  →  Test  →  Ship  →  Reflect
  ↑                                                          |
  └──────────────────────────────────────────────────────────┘
```

Every domain (and every non-trivial change) passes through all seven stages.

| Stage | Role lens | What it means here | Output |
|-------|-----------|--------------------|--------|
| **Think** | *CEO / skeptic* | Question the assumption before writing code. For a research platform the sharpest question is always: *does this add measurable value, or just complexity?* Challenge scope. | A stated assumption + the risk it addresses |
| **Plan** | *Architect* | Decide module boundaries, dependencies, and the public API. Lowest domains first (math → astro → features). | The files to write and what each owns |
| **Build** | *Engineer* | Implement with strict typing, Google-style docstrings, and the conventions in existing `core/`. No domain imports upward. | Working modules |
| **Review** | *Reviewer* | Re-read the diff for correctness, reuse, and dead code before testing. Prefer `/simplify` and `/code-review`. | A clean diff |
| **Test** | *QA* | Unit tests per module; property/round-trip tests for math; validate astro output against known ephemeris values. `pytest`, `ruff`, `mypy --strict`. | Green test suite |
| **Ship** | *Release* | Commit in a coherent unit with a descriptive message; push to `origin/main`. Small, frequent, reversible. | A pushed commit |
| **Reflect** | *Scientist* | Record what was learned, what surprised us, and what to revisit. Feeds the next Think. | A note in `docs/reflections/` or the commit body |

## Definition of Done (per domain)

A domain is "done" only when **all** hold:

1. Modules import cleanly and expose a documented public API via `__init__.py`.
2. Unit tests cover the core behavior and the tricky edges (wrap-around, empty
   input, constant input, degenerate cases).
3. `ruff check` and `mypy --strict` pass on the new code.
4. A smoke check demonstrates the domain doing something real end-to-end.
5. The work is committed and pushed.

## The scientist's veto (project-specific role)

Because this is a null-hypothesis investigation (H₀: Vedic features add no
predictive value), one role overrides the others: **the science must be able to
fail.** No stage may quietly optimize toward "astrology works." Concretely:

- Features are validated by mutual information **before** any model is trusted
  (Domain 25). Zero MI ⇒ the feature is noise, regardless of model accuracy.
- Every reported result carries a significance test, an effect size, and a
  confidence interval (Domains 43, 50).
- Every model is measured against strong baselines — random, majority, XGBoost
  on the same features (Domain 44). Beating them is the bar; not beating them is
  a publishable result too.

## Commit conventions

- One coherent change per commit; descriptive subject + body explaining *why*.
- Commits are co-authored by the AI pair (`Co-Authored-By:` trailer).
- Ship frequently; keep `main` importable and test-green at every push.
