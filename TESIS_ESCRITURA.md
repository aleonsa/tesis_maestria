# Log de Escritura — Tesis

> Este archivo trackea el progreso de redacción del documento de tesis en `tesis_documento/`. Se actualiza al cierre de cada bloque para no perder hilo entre sesiones.

**Última actualización:** 2026-05-23 (cerrado Bloque D)
**Archivo LaTeX raíz:** `tesis_documento/Tesis_JALS_Maestria.tex`
**PDF generado actual:** 34 páginas (frontmatter + Cap. 2 + Cap. 3 + placeholders restantes)

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

### Bloque E — Cap. 4 Desarrollo ⏳ PENDIENTE (siguiente)
Recordar usar `\label{cap:desarrollo}` en el `\chapter{}`.
- 4.1 FCS-M2PC: derivación completa (sub-intervalos, predicción, costo)
- 4.2 ADALINE+Fourier: estructura
- 4.3 Entrenamiento offline (pseudoinversa)
- 4.4 Aprendizaje online (LMS): convergencia, misadjustment, costo
- 4.5 Justificación: por qué ADALINE y no NN profunda (3 razones estructurales)
- 4.6 Observador de BEMF en lazo cerrado
- 4.7 Arquitectura integrada (diagrama de bloques)

### Bloque F — Cap. 5 Validación numérica ⏳ PENDIENTE
- 5.1 Configuración (motor BLY-344S, parámetros, escenarios)
- 5.2 Estudio comparativo (TRAP / SIN / NN / ADALINE off / ADALINE on TRAP / ADALINE on SIN)
- 5.3 Convergencia del LMS online
- 5.4 Rizo residual y caracterización a baja velocidad
- 5.5 Discusión

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
