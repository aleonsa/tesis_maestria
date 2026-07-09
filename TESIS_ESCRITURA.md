# Log de Escritura — Tesis

> Este archivo trackea el progreso de redacción del documento de tesis en `tesis_documento/`. Se actualiza al cierre de cada bloque para no perder hilo entre sesiones.

**Última actualización:** 2026-07-08 (cerrado Bloque F)
**Archivo LaTeX raíz:** `tesis_documento/Tesis_JALS_Maestria.tex`
**PDF generado actual:** 53 páginas (frontmatter + Cap. 2 + Cap. 3 + Cap. 4 + Cap. 5 + placeholders restantes)

---

## Convenciones de estilo (NO IGNORAR)

Estas reglas aplican a TODA la prosa que se escriba en los capítulos. Vinieron de feedback directo del autor — no son sugerencias.

- **Voz humana, no IA.** Frases cortas, sin adverbios de relleno ("además", "asimismo", "por consiguiente"). Cuando una idea es directa, decirla directa.
- **Calcar el tono de los papers del repo.** Ver especialmente:
  - `papers/coronado25-draft.pdf` (paper del asesor) → tono académico pero no rebuscado
  - `papers/shao03.pdf` (tesis de maestría) → estructura clara, párrafos cortos
  - `papers/coronado23.pdf` → introducciones de capítulo
- **Nada de listas innecesarias en prosa.** Las listas con viñetas se reservan para enumeraciones reales (objetivos específicos, propiedades de un algoritmo, etc.), no para descomponer cada idea.
- **Evitar muletillas de IA**: "es importante destacar", "cabe mencionar", "vale la pena resaltar", "en este sentido", "por otro lado".
- **Una idea por párrafo.** Si un párrafo tiene dos ideas, son dos párrafos.
- **No abusar de "se" impersonal**; alternar con primera persona del plural ("proponemos", "consideramos", "obtuvimos") cuando el contexto lo permite — coherente con la voz de Coronado et al.
- **Las ecuaciones se introducen y se leen.** No se dejan flotando: "donde $R$ es la resistencia y $L$ la inductancia". Tampoco se sobreexplican términos triviales.
- **Citar con `\citep{}` (paréntesis) o `\citet{}` (texto)**, según el estilo natbib ya configurado.

---

## Orden de escritura acordado

El usuario decidió escribir la **introducción al final**. Orden real:

1. ~~Bloque A — Infraestructura (siglas, notación, bib)~~ **HECHO**
2. **Bloque C — Cap. 2 Preliminares** (siguiente)
3. Bloque D — Cap. 3 Modelo y planteamiento
4. Bloque E — Cap. 4 Desarrollo (el corazón teórico)
5. Bloque F — Cap. 5 Validación numérica
6. Bloque B — Cap. 1 Introducción ← al final
7. (Fuera de este sprint) Cap. 6 Conclusiones + capítulo/sección de hardware

---

## Decisiones de alcance ya tomadas

- **Hardware fuera de este sprint.** Toda la parte de STM32G431 + B-G431B-ESC1 + AS5600 + 2804 va en una sección/capítulo separado posterior. No se menciona en el cuerpo teórico.
- **Control Repetitivo (RC) IGNORADO por completo en este sprint.** No aparece en preliminares, no aparece en desarrollo, no aparece en trabajo futuro. Se trata cuando se ataque la Fase 3.
- **Resultados del Cap. 5 con números P=4.** Se redacta con los datos actuales (LUT sintética, 4 pares de polos) y se marca con `\todo{}` que se regenerarán cuando P=8 corregido esté validado. No bloquea la redacción.
- **El Cap. 4 todavía no tiene título.** Propuesta: "Esquema FCS-M2PC con estimación adaptable de BEMF" o similar. A decidir cuando lo abordemos.

---

## Estado por bloque

### Bloque A — Infraestructura ✅ COMPLETO
- `Documentos/Siglas.tex`: 30+ acrónimos (BLDC, PMSM, BEMF, MPC, FCS-MPC, FCS-M2PC, ADALINE, LMS, etc.)
- `Documentos/Notacion.tex`: tabla con todos los símbolos del proyecto
- `Bibliografia.bib`: 18 entradas con metadatos reales extraídos de los PDFs del repo
- **Compila limpio** (verificado)

### Bloque C — Cap. 2 Preliminares ✅ COMPLETO
Secciones escritas:
- 2.1 El motor BLDC y el origen del rizo de par (\ref{sec:bldc-rizo})
- 2.2 Modelo eléctrico-mecánico en el marco abc (\ref{sec:modelo-abc})
- 2.3 Transformada de Clarke (\ref{sec:clarke}) — incluye justificación de αβ vs dq
- 2.4 El inversor trifásico y sus vectores admisibles (\ref{sec:vsi}) — con cuadro de los 8 vectores
- 2.5 Control predictivo basado en modelo (\ref{sec:mpc}) — MPC clásico → FCS-MPC → FCS-M2PC (intuición)
- 2.6 ADALINE y el algoritmo LMS (\ref{sec:adaline})
- 2.7 Series de Fourier truncadas como base para señales periódicas (\ref{sec:fourier})

**Referencias forward usadas (resolver al escribir los capítulos):**
- `cap:modelo` (Cap. 3, sec. 2.1)
- `cap:desarrollo` (Cap. 4, secs. 2.5 y 2.7)

**Citas usadas en este bloque:** coronado2025, coronado2023, krishnan2009pmbldc, krause2013analysis, shao2003direct, rodriguez2012predictive, kouro2009mpc, widrow1960adaline, widrow1985adaptive, haykin2014adaptive.

**Compila limpio** (verificado: bibtex resuelve, 31 páginas).

### Bloque D — Cap. 3 Modelo y planteamiento ✅ COMPLETO
Secciones escritas (chapter etiquetado `\label{cap:modelo}`):
- 3.1 Modelo del motor en αβ (\ref{sec:modelo-ab-cap3}) — forma de estado, factorización $\mathbf{e}=K_e\omega_m\mathbf{s}(\theta_e)$, par como producto interno, discretización Euler
- 3.2 Generación de la referencia de corriente (\ref{sec:iref}) — derivación tipo Le-Huy: min $\|\mathbf{i}\|^2$ s.a. restricción de par → $\mathbf{i}^* = T_\mathrm{ref}\mathbf{s}/(K_t\|\mathbf{s}\|^2)$
- 3.3 El rizo como model mismatch (\ref{sec:rizo-mismatch}) — dos vías de propagación: generación de referencia + predicción
- 3.4 Planteamiento formal con entorno `problema` (\ref{sec:problema})
- 3.5 Hipótesis (\ref{sec:hipotesis})
- 3.6 Objetivos con entorno `objetivo` (\ref{sec:objetivos}) — general + 5 específicos

**Referencias nuevas agregadas al bib:**
- `lehuy1986minimization` (paper original de la fórmula de referencia óptima)
- `chiasson2005modeling` (libro de modelado de máquinas, citado por Coronado25)

**Referencias forward pendientes:** `cap:desarrollo` (5 ocurrencias, se resuelven al escribir Cap. 4).

**Compila limpio** (34 páginas, sin warnings de bibtex).

### Bloque E — Cap. 4 Desarrollo ✅ HECHO (2026-06-06)
Título tentativo: "Control FCS-M2PC con estimación adaptable de la BEMF". `\label{cap:desarrollo}` puesto.
- 4.1 FCS-M2PC: derivación completa (sectores eq, predicción eq 6 del paper, combinaciones C^R, costo, Algoritmo) ✅
- 4.2 ADALINE+Fourier: estructura (W ∈ R^{2H×2}, dos ADALINE compartiendo regresor) ✅
- 4.3 Entrenamiento offline (pseudoinversa, ortogonalidad → coef. Fourier) ✅
- 4.4 Aprendizaje online (LMS): R = ½I, convergencia uniforme, misadjustment, costo O(H) ✅
- 4.5 Justificación ADALINE vs NN profunda (convergencia / parsimonia / interpretabilidad) ✅
- 4.6 Observador de BEMF en lazo cerrado (observador de corriente eq, s_med = ê/(Ke ωm)) ✅
- 4.7 Arquitectura integrada ✅

**Cambios de infraestructura:** agregados `algorithm` + `algpseudocode` al preámbulo del main (con `\floatname` → "Algoritmo" y Entradas/devolver en español). Compila limpio, 42 págs, sin refs/citas pendientes.

**Figuras (5 de 5 hechas en TikZ, 2026-06-16):**
- ✅ `fig:sectores-ab` (4.1) — hexágono + 6 sectores + set reducido. TikZ, geometría fiel a `eq:sectores` (vectores en bordes de sector, u_k a −90°+(k−1)60°).
- ✅ `fig:operacion-m2pc` (4.1) — escalera de corriente M2PC (3 tramos u_a/u_b/u_0) vs recta FCS, referencia punteada, llaves Ta/Tb/T0. TikZ.
- ✅ `fig:adaline` (4.2) — red de una capa: θe → regresor de Fourier (boxes cos/sin) → 2 combinadores Σ → ŝ_α, ŝ_β; pesos w_α/w_β; "sin no linealidad". TikZ.
- ✅ `fig:observador` (4.6) — diagrama de bloques: observador + L_o sobre error i−î + normalización 1/(Ke ωm) → s_med. TikZ.
- ✅ `fig:arquitectura` (4.7) — diagrama completo: cadena directa + realim. ωm (vía superior) + bus de mediciones (vía inferior) + lazo adaptable; **doble inyección de ŝ resaltada en naranja**. TikZ con `\resizebox{\linewidth}` para encajar al ancho de texto.

**Infra TikZ agregada al main:** `\usepackage{tikz}` + librerías (`positioning,arrows.meta,calc,shapes.geometric,shapes.misc,fit,backgrounds,decorations.pathreplacing`). **Truco babel:** cada `tikzpicture` lleva `\shorthandoff{<>}` porque babel-spanish hace `<`/`>` activos (atajos de guillemets) y rompen la sintaxis `->`/`>=` de TikZ.

Nota: Cap. 2 y Cap. 3 quedaron SIN figuras (revisar después; el autor lo señaló).

### Bloque F — Cap. 5 Validación numérica ✅ HECHO (2026-07-08)
`\label{cap:validacion}`. Secciones escritas (estructura final, ajustada a los datos disponibles):
- 5.1 Configuración del caso de estudio (`sec:config`) — tabla de parámetros (`tab:params-sim`), escenario (80 rad/s, escalón de carga 0.5→1.5 Nm en 150 ms, 300 ms, Ts=30µs, ρ=10 → 66 combinaciones), los 6 métodos, métricas (ventana último 15%), dos escenarios de aprendizaje (supervisión ideal / observador)
- 5.2 Comparación con supervisión ideal (`sec:comparativo`) — fig overview + steadystate + `tab:metricas-ideal`. Hallazgo destacado: **SIN da MÁS rizo que TRAP (6.43 vs 6.06%) pese a mejor RMSE de corriente** → firma de las dos vías de propagación (sec:rizo-mismatch); el rizo no es monótono en el error de seguimiento
- 5.3 Convergencia del aprendizaje en línea (`sec:convergencia-lms`) — <5% en 25–35 ms ideal, monótona (R=½I, sin modos lentos); pico en 150 ms = escalón de carga, no pérdida de convergencia
- 5.4 Aprendizaje con observador (`sec:validacion-observador`) — `tab:metricas-observador`, rizo online sube a 2.36/2.48% (misadjustment), convergencia 50 ms (SIN-init) / 120 ms (TRAP-init) con transitorio errático por baja velocidad inicial; fig tracking αβ
- 5.5 Discusión (`sec:discusion`) — hipótesis sostenida (69% ideal / 61% con observador), piso de 1.9% = límite del M2PC (ρ, Ts), no del modelo BEMF; limitaciones: baja velocidad (LMS off bajo 5 rad/s, híbrido encoder-observador), sensibilidad a R/L no caracterizada

**Números finales (corridas 2026-07-07, P=8, LUT sintética):**
| Método | Ideal [%] | Observador [%] |
|---|---|---|
| TRAP | 6.06 | 6.06 |
| SIN | 6.43 | 6.43 |
| LEARNED | 1.89 | 1.89 |
| ADALINE_OFF | 1.90 | 1.90 |
| ADALINE_ON_T | 1.88 | 2.36 |
| ADALINE_ON_S | 1.87 | 2.48 |

**Figuras (5, PNG 300dpi en `Capitulo5/FigureC5/`):** `f1_overview`, `f1_steadystate`, `f1_convergence` (de `fase1_sintetica`), `f2_convergence`, `f2_tracking` (de `fase2_sintetica`). Nota: los PNG traen títulos MATLAB embebidos (informales, en español); si se quiere pulir, regenerar con títulos limpios y re-exportar — los captions LaTeX ya llevan la descripción formal. El run fase1 no generó `tracking.png` (warning del export); el de fase2 sirve porque los 4 métodos fijos son idénticos entre fases.

**Compila limpio** (53 págs, sin refs indefinidas; 3 cuadros + 5 figuras colocados).

### Bloque B — Cap. 1 Introducción ⏳ AL FINAL
- 1.1 Motivación
- 1.2 Estado del arte
- 1.3 Contribución
- 1.4 Estructura de la tesis

---

## Recordatorios técnicos

- **Comandos LaTeX disponibles** (`Comandos.sty`): `\begin{problema}`, `\begin{objetivo}` (entornos especiales con estilo `problemstyle`); teoremas estándar (`theorem`, `definition`, `lemma`, `proposition`, etc.)
- **Acrónimos:** usar `\gls{xxx}` para la primera aparición, `\acrshort{xxx}` para las siguientes
- **Citas:** `\citep{coronado2025}` para citar entre paréntesis, `\citet{coronado2025}` para citar en el texto
- **Figuras:** cada capítulo tiene su carpeta `FigureCN/` configurada en `\graphicspath`
- **Compilación rápida** (sin bib): `cd tesis_documento && pdflatex -interaction=nonstopmode Tesis_JALS_Maestria.tex`
- **Compilación completa** (con bib): `pdflatex && bibtex && pdflatex && pdflatex`

---

## Pendientes globales (no bloquean redacción)

- [ ] Regenerar resultados con P=8 corregido (afecta tablas y figuras del Cap. 5)
- [ ] Decidir título definitivo del Cap. 4
- [ ] Llenar Portada con datos reales (asesor, fecha de grado, etc.)
- [ ] Llenar Resumen y Abstract (al final, con números finales)
- [ ] Verificar que figuras existentes en `figures/` se referencien correctamente desde los capítulos
