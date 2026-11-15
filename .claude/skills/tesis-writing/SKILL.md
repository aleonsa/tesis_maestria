---
name: tesis-writing
description: Use when writing, editing or reviewing thesis prose in tesis_documento/ (chapters .tex, Resumen, Conclusiones, LaTeX figures, TikZ, citations, acronyms). Enforces the author's Spanish academic voice, strips AI-sounding prose, and runs a prose checker over the .tex files.
---

# Thesis writing (Spanish, LaTeX)

Read `TESIS_ESCRITURA.md` first: style rules, order of writing, scope decisions, per-block status. The author's rules there are not suggestions. The anti-AI patterns and before/after examples are in `patrones.md` (same folder): read it before drafting prose.

## Before writing a section
1. Open the matching paper in `papers/` (`coronado25-draft.pdf`, `shao03.pdf`, `coronado23.pdf`) and copy the tone, not the sentences.
2. Read the neighbor chapters so that notation (`Documentos/Notacion.tex`), acronyms (`Siglas.tex`) and `\label`s stay consistent.
3. Check scope: no STM32 detail in the theory chapters (it lives in `cap:experimental`), nothing about Repetitive Control.
4. Collect the numbers first. Every number must trace to a script or a log in the repo. If it cannot be traced, write `\pend{...}` (red, visible in the PDF), never an invented value.

## Prose
- Short sentences. One idea per paragraph. Say direct things directly.
- Banned phrases and patterns: see `patrones.md`. The checker enforces them.
- Alternate first-person plural ("medimos", "obtuvimos") with impersonal "se"; do not stack "se".
- Bullets only for real enumerations (objectives, properties of an algorithm).
- Introduce every equation and read its symbols ("donde $R$ es la resistencia..."), without over-explaining trivial terms.
- State a negative result as a result: what was measured, with what controls, what it bounds. No apologies, no spin.
- A claim that is an extrapolation says so, or it gets backed by a run, or it goes.

## LaTeX conventions
- Citations: `\citep{key}` / `\citet{key}`; the key must exist in `Bibliografia.bib`. The 7 entries flagged as unverified in `TESIS_ESCRITURA.md` go through the `bib-checker` agent first.
- Acronyms: `\gls{x}` at first use, `\acrshort{x}` afterwards; new acronyms in `Siglas.tex`, new symbols in `Notacion.tex`.
- Units with `siunitx`.
- Each `tikzpicture` starts with `\shorthandoff{<>}`. Figures live in `FigureCN/` of their chapter.
- Percent with babel-spanish: write `$<1$ \%`, never `\,\%` inside math (it breaks compilation: "Incompatible glue units").
- Pending items: `\pend{...}` (defined with `\providecommand` at the top of the chapter that uses it). `\todo{}` is not loaded.
- Cross-references with `\ref{sec:...}`, `\ref{fig:...}`, `\ref{eq:...}`; record forward references that are still dangling in `TESIS_ESCRITURA.md`.

## After writing (do all three)
1. Run the checker on what you touched and fix every hit, or justify it:
   `python3 .claude/skills/tesis-writing/scripts/check_prosa.py tesis_documento/Documentos/Chapters/CapituloN/CapN.tex`
2. Compile: `cd tesis_documento && pdflatex Tesis_JALS_Maestria && bibtex Tesis_JALS_Maestria && pdflatex Tesis_JALS_Maestria && pdflatex Tesis_JALS_Maestria`. TeX Live may need `eval "$(/usr/libexec/path_helper)"` in a fresh shell. If it fails or is missing, say "no pude compilar".
3. Update the block status in `TESIS_ESCRITURA.md`.
