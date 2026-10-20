# Planning de Implementación Experimental — FCS-M2PC + ADALINE en B-G431B-ESC1

> **Documento vivo.** Revisar al final de cada fase y ajustar tiempos según el ritmo real.
> Fecha de inicio del planning: 2026-05-16. Última actualización: 2026-05-17. Presupuesto promedio: **2 h/día**, ~14 h/semana, ~60 h/mes.

---

## 0. Resumen ejecutivo

| Concepto | Valor |
|---|---|
| Horas comprometidas | ~2 h/día × 5 días/semana = **10–14 h/semana** |
| Duración total estimada | **7–9 meses** desde llegada de la placa |
| Fases | **6** (alineadas con `project_hw_bench_2804.md`) |
| Hito final | Comparación experimental rizado: FCS-M2PC sin ADALINE vs FCS-M2PC + ADALINE online |
| Plataforma | B-G431B-ESC1 + motor 2804 (12N14P, 7 pp) + AS5600 |
| Toolchain | STM32CubeIDE + MCSDK 6.2 (referencia) + código bare-metal CMSIS propio (final) |

**Filosofía del planning**: cada fase termina con un **entregable medible** (oscilograma, .mat, métrica) que alimenta directamente un párrafo del capítulo experimental de la tesis. **Nada que no se mida o no se documente cuenta como avanzado.**

---

## 1. Aclaración crítica sobre el rol del ADALINE (NO confundir)

Existen dos usos posibles del ADALINE en control de motores. Hay que ser explícito sobre cuál implementamos:

| Aplicación | Qué estima | Esta tesis |
|---|---|---|
| **ADALINE para identificación paramétrica** (Energies 2020) | Rs, Ld, Lq del modelo dq | ❌ NO |
| **ADALINE + Fourier para BEMF** (esta tesis) | Forma de e_α(θ), e_β(θ) — 21 coef. | ✅ SÍ |

**Implicación operativa**: el bloque que se programa en C en el STM32 es:

```
x(θ) = [1, cos(θ), sin(θ), cos(2θ), sin(2θ), ..., cos(10θ), sin(10θ)]^T   (21 elementos)
e_hat = W^T · x(θ)         (W es la matriz de pesos 21×2 para α y β)
ε = e_med - e_hat          (BEMF medida menos predicha)
W <- W + μ · x(θ) · ε^T    (regla LMS, μ ≈ 5e-3)
```

La BEMF medida se obtiene del observador (no del sensor directamente — no hay sensor de BEMF en el banco). El observador requiere modelo del motor: por eso la **Fase 2 (identificación)** es prerequisito antes de la **Fase 6 (ADALINE)**.

---

## 2. Hardware (recap rápido) — estado 2026-05-17

- **Motor**: 2804 12N14P, 7 pp, BEMF cuasi-senoidal, Rs≈2.3 Ω, Ls≈0.86 mH, λ_pm≈0.0035 Wb, KV220, Inom 0.5 A. ✅ adquirido.
- **Sensor**: **AS5600 integrado al motor** (imán diametral y montaje mecánico ya resueltos por fabricante). I²C, 12 bits, tasa interna ~7 kHz → extrapolación en software. ✅ integrado al motor.
- **Placa**: B-G431B-ESC1 (STM32G431CBU6 @170 MHz, CORDIC, FMAC, 3-shunt, ST-LINK integrado, Vbus 11–25 V). ✅ recibida.
- **Osciloscopio**: ✅ adquirido.
- **Programador**: PC local, USB directo al ST-LINK de la placa (desde 2026-10-11; antes Raspberry Pi vía SSH desde la Mac del trabajo).
- **Ts objetivo**: 33 μs (30 kHz). Coordenadas αβ.

### Pares de polos — registro definitivo (2026-10-12)

El proyecto usa **dos motores distintos** y la letra `P` significa cosas distintas en cada
código. Esta tabla manda; cualquier otro comentario que la contradiga es un error.

| | Motor de **simulación** | Motor del **banco** |
|---|---|---|
| Modelo | Anaheim BLY-344S-240V-3000 (paper de Coronado) | 2804 12N14P con AS5600 integrado |
| Polos (imanes) | 8 | 14 |
| **Pares de polos p** | **4** | **7** |
| Dónde se usa | `fcs_m2pc_v2/`, Cap. 5 de la tesis | `firmware/` |
| Cómo aparece en código | `p.P = 8` → cuenta **polos**; el código usa `P/2` | `THETA_PLL_POLE_PAIRS = 7` → cuenta **pares** |
| Evidencia | Datos del fabricante; `pmsm.p = 4` en el script de Coronado | Especificación 12N14P + medición en banco (sesión 12): a f_e = 2 Hz el rotor da una vuelta en 3.5 s → f_m = 0.286 Hz → p = 7 |

Convención para todo código nuevo: nombrar explícito `POLE_PAIRS` / `p` (pares), nunca una `P`
suelta. En la tesis, $p$ = pares de polos (`Notacion.tex`), $\theta_e = p\,\theta_m$.

Restos históricos conocidos (no afectan resultados vigentes):
- `matlab_scripts/*.m` (legado, anterior a `fcs_m2pc_v2/`): `p.P = 4` comentado como "Número de
  polos". Es el origen del error corregido el 2026-07-07; no se usa para la tesis.
- `presentation/sections/04_results.tex`: la tabla dice "Polos & 4"; físicamente el BLY-344S
  tiene 4 **pares** (8 polos). Hay que revisar con qué corrida se hizo esa presentación antes de
  corregirla.
- **Rango de operación**: < 1500 rpm mecánicas (límite por latencia del AS5600).

**Conector confirmado por UM2516 Fig. 10 y Tabla 4**:
- **J7**: 3 terminales de fase (OUT1/OUT2/OUT3) — conexión al motor.
- **J8**: 5 pads para sensor (GND, 5V, PB8/Z+/H3, PB7/B+/H2, PB6/A+/H1).
- **J5/J6**: entrada Vbus (3S-6S LiPo) — para lab usamos fuente DC.
- **U4**: micro-USB en daughterboard para ST-LINK + alimentación lógica.

---

## 3. Mapa de recursos (qué leer y cuándo)

### 3.1 Manuales obligatorios

| Documento | Páginas relevantes | Cuándo | Local |
|---|---|---|---|
| **UM2516** — B-G431B-ESC1 User Manual | 30 (entero) | Fase 0 | ✅ `papers/um2516-*.pdf` |
| **RM0440** — STM32G4 Reference Manual | Cap 7 (RCC), 9 (GPIO), 21 (ADC), 24 (OPAMP), 28 (TIM1) | Fase 0–1 | ✅ `papers/rm0440-*.pdf` |
| **RM0440** — caps avanzados | Cap 17 (CORDIC), 18 (FMAC), 11 (interconexiones) | Fase 1–3 | (mismo PDF) |
| **DS12589** — Datasheet STM32G431x6/x8/xB | Pinout, abs max, electrical char. | Consulta puntual | ✅ `papers/STM32G431X6.PDF` |
| **MB1419-G431CB-B01** — Esquemático oficial de la placa | Tabla de pinout completo, solder bridges | Fase 0 y consulta puntual | ✅ `papers/en.MB1419-*.pdf` |
| **AN5325** — CORDIC en STM32 | Entero (20 pág., teoría + ejemplo) | Fase 1 | ✅ `papers/an5325-*.pdf` |
| **AN5305** — FMAC (filtros adaptativos) | Solo si se acelera filtrado de corriente o LMS | Opcional Fase 1 / Fase 6 | ✅ `papers/an5305-*.pdf` |
| **PM0214** — Cortex-M4 Programming Manual | Solo si se optimiza con SIMD | Opcional Fase 5+ | ❌ bajar si se necesita |
| **AN4539** — HRTIM Cookbook | Solo si saltamos a HRTIM | Fase 4+ opcional | ❌ bajar si se necesita |

### 3.2 Bibliografía de control por fase

**Fase 2 (identificación)**
- Krause, Wasynczuk, Sudhoff — "Analysis of Electric Machinery and Drive Systems", caps. 4-5.
- AN1078 (Microchip) — sensorless FOC con Luenberger (teoría limpia).

**Fase 3 (FOC baseline)**
- UM1052 (ST) — teoría del MCSDK (la teoría sí, el código no).
- Bose — "Modern Power Electronics and AC Drives", cap. FOC.

**Fase 4 (FCS-MPC clásico)**
- **Kouro, Cortés, Vargas, Ammann, Rodríguez (2009)** — "MPC: A Simple and Powerful Method to Control Power Converters", IEEE Trans. Ind. Electron. 56(6):1826-1838. **Lectura obligatoria.**
- Rodríguez & Cortés (2012) — "Predictive Control of Power Converters and Electrical Drives" (Wiley), cap. 5.

**Fase 5 (FCS-M2PC)**
- **Coronado 2025** — `papers/coronado25-draft.pdf` (paper del asesor, **lectura obligatoria**).
- Tarisciotti et al. — "Modulated MPC for a three-phase active rectifier" (origen del término M²PC).

**Fase 6 (ADALINE online)**
- Widrow & Stearns — "Adaptive Signal Processing" (1985), caps. 1-4 (LMS).
- Coronado 2025 — sección de adaptación de BEMF.
- Tu propio análisis en `python_scripts/generate_lut_adaline.py` y `fcs_m2pc_v2/adaptation/lms_update.m`.

### 3.3 Código de referencia (estudiar, no copiar)

| Repo / Recurso | Uso |
|---|---|
| `brushless.zone` blog post CORDIC vs sinf | Benchmark de referencia, idiomas de bare-metal en G4 |
| `pat92fr/BrushlessServoController` (GitHub) | FOC en B-G431B-ESC1 con CORDIC, estructura de proyecto (solo código; el esquemático ya lo tenemos oficial) |
| `eirbot/eirbot-B-G431B-ESC1-guide` | Pinout, cómo lidiar con la placa |
| ST `STM32CubeG4` (CMSIS device pack) | Headers y startup |
| `odriverobotics/ODrive` (F4) | `motor.cpp`, `current_controller.cpp` (calidad profesional) |
| `simplefoc/Arduino-FOC` | Clarke/Park conceptual; `b_g431_hal.cpp`/`b_g431_mcu.cpp` tienen la config real de OPAMP/ADC para esta placa; PR #210 es el bug del trigger en el valle |
| Tu `fcs_m2pc_v2/` (MATLAB) | Referencia algorítmica directa para el port a C |

### 3.4 Sensado de corriente en B-G431B-ESC1 (añadido 2026-10-13, ver FIELD_NOTES N1.23)

| Recurso | Qué aporta |
|---|---|
| MB1419 hoja 5 "SHUNT SENSING CIRCUIT" | Red 1.5k/22k/2.2k por fase: polarización 128.6 mV, atenuación 4/7. **Leer antes de calcular cualquier ganancia** |
| `district9prawn/B-G431B-ESC1-foc-LL` | FOC en registros LL (sin HAL), PGA `IO0_BIAS` ×16, TRGO=OC4REF, CORDIC. El más cercano a nuestro estilo |
| `pat92fr/FOC2` | Calibración continua de offset con las muestras del valle (`RCR=0` + `DIR`) |
| `RicoKirstein/PrinterBLDCCooler` | Proyecto MCSDK 6.0 generado: parámetros oficiales de ST (`AMPLIFICATION_GAIN 9.14`, `TNOISE`, punto de muestreo `CCR4 = Half−1`, `R3_2_SetADCSampPointSectX`) |
| Foro ST 51574 / 38573 / 3872 | Ganancia fija 16 por la polarización; fórmula de transferencia; OPAMP3→ADC2_IN18 |
| Foro ST 16269 | UEV con RCR impar en center-aligned: underflow si RCR se escribe antes de CEN |
| DS12589 Tabla 73 | `TS_OPAMP_VOUT ≥ 200 ns`, GBW 13 MHz, R2/R1 ±15 % |
| AN5306 | Uso de OPAMP en G4 (modos PGA, calibración) |

---

### 3.5 BEMF del 2804: fabricante y literatura (búsqueda 2026-10-18)

- **Motor del banco = DFRobot FIT1034 / kit SimpleFOC** ([DFRobot](https://www.dfrobot.com/product-3005.html), [wiki](https://wiki.dfrobot.com/fit1034/)): 2804, 220 KV, **delta**, 2.3 Ω y 0.86 mH por fase (5.1 Ω / 2.8 mH devanado), 0.0035 Wb, 7 pp, 0.4 A nominal, 0.03 N·m. Otros 2804 (iPower GBM2804H‑100T, 10 Ω; GM2804, 9 Ω; T‑Motor GB2208, 14.4 Ω) son variantes de más vueltas. **Ningún fabricante publica forma de BEMF, THD ni inercia.**
- **12/14 doble capa**: k_w1 = 0.933, k_w3 = 0.5, **k_w5 = k_w7 = 0.067**, k_w11 = k_w13 = 0.933 (Libert & Soulard, ICEM 2004). En delta el 3º circula dentro del triángulo. Tang & Brown, IEEE TEC 38(3) 2023 (THD espacial de FMM 12/14 = 1.69 vs 0.98 del 12/10). Ishak, Zhu & Howe, IEEE TIA 41(2) 2005 e IEEE TEC 21(1) 2006 (BEMF en 12 ranuras, "todos los dientes bobinados" = la menos trapezoidal).
- **Mediciones en 12N14P pequeños**: solo cualitativas — Parsons, *Things in Motion* 2018 (osciloscopio a mano, ODrive D5065: "clearly sinusoidal"). SimpleFOC, pat92fr/FOC2, ODrive, VESC: nada cuantitativo para 2804.
- **Identificación de armónicos de BEMF con ADALINE/Fourier**: Wang, Zhu, Bin & Gong, IEEE TIE 67(11) 2020 (doi 10.1109/TIE.2019.2955414); Nguyen, Semail, De Belie & Kestelyn, IECON 2016; Vu, Nguyen, Semail & Wu, Energies 15(1):249, 2022; revisión Zhu, Liang & Liu, IEEE Access 9, 2021 (doi 10.1109/ACCESS.2021.3072959).
- **Dead‑time vs BEMF**: Bedetti, Calligaro & Petrella, IEEE TIA 52(5) 2016 (regresión lineal múltiple); Qiu, Wen & Zhao, IEEE TPEL 31(3) 2016 (ADALINE); Lv et al., Energies 17(7):1693, 2024. El dead‑time produce 6º en dq, el mismo orden que 5º/7º de BEMF: no se separan desde la tensión comandada. **Pfeifer et al., TechRxiv 2025**: identificación en rueda libre (inversor apagado) — es nuestra Exp. 2b.
- Inercia del rotor 2804: sin dato; estimación geométrica 4–7·10⁻⁶ kg·m². Medido en banco con llanta: 1.5·10⁻⁵ (sesión 23).

## 4. Plan fase por fase

> **Notación**: cada fase declara un **objetivo único**, **entregables**, **tiempo en horas** y un **desglose semanal a 2 h/día**.
> Al final de cada fase: actualizar `project_hw_bench_2804.md` con resultados.

---

### Fase 0 — Toolchain, blink, familiarización (~30 h, 3 semanas)

**Estado 2026-05-18**: ✅ Semana 1 cerrada (en ~1 día real, no 1 semana — pista de que la estimación de la fase era conservadora). Semana 2 en curso. Semana 3 ajustada (ya validamos la placa con firmware propio en vez del demo de ST).

**Objetivo único**: tener el ciclo edit → compile → flash → debug funcional en bare-metal CMSIS, sin depender de CubeMX.

**Por qué bare-metal y no CubeMX**: control total del timing del ISR (crítico para FCS-MPC), código portable, lectura de código de referencia más limpia. CubeMX se usa **solo para inspirar configuraciones de registros** (vista "Pinout & Configuration"), no para generar código del proyecto.

**Semana 1 (10 h)** ✅ COMPLETADA 2026-05-18
- ✅ Lectura UM2516 entero (cap. 5 layout + Tabla 4 pinout).
- ❌ **FALSO**: se dio por validado el pinout J8 contra la Tabla 13 del DS12589, pero ese
  datasheet nunca estuvo en el repo. AF4 en PB6 NO es I2C1_SCL. Corregido 2026-08-10:
  **SCL = PB8**, SDA = PB7. Ver FIELD_NOTES N1.16.
- ✅ Toolchain en la Pi (sin CubeIDE — bare-metal directo): arm-none-eabi-gcc 10.3, OpenOCD 0.11, reglas udev, SSH alias `raspi`.
- ✅ CMSIS Device Pack via submódulo `cmsis_device_g4`.
- ✅ Proyecto `firmware/blink/` con `CMakeLists.txt`, compilado a 880 B Flash, flasheado, LED PC6 parpadeando.
- Cadena Mac → rsync → Pi → cmake/arm-gcc → openocd → STM32G431 → PC6 validada end-to-end.

**Semana 2 (10 h)** — EN CURSO
- SysTick a 1 ms (reemplazar `delay()` por NOP loop).
- HSE/PLL @170 MHz vía registros directos (RCC + Flash latency).
- UART debug (USART2, PA2/PA3 según UM2516 Tabla 4) con `printf` retargeting hacia `/dev/ttyACM0` de la Pi.
- Capítulo 7 RM0440 (RCC) — lectura mientras se programa PLL.

**Semana 3 (10 h)** — ajustada
- Capítulo 9 RM0440 (GPIO + alternate functions) — lectura.
- Capítulo 28 RM0440 (TIM1) — lectura introductoria, identificar registros clave para Fase 1.
- Documentar setup en `docs/firmware/setup.md`.
- ~~Bajar firmware demo de la placa~~ — innecesario (validamos con firmware propio).

**Entregables Fase 0**:
- ✅ Repo `firmware/` con build reproducible (`cmake --build`).
- 🟡 Binario que parpadea LED (LED ✅) y emite "hello world" por UART (pendiente Semana 2).
- ⏳ Documento de setup paso a paso (en `PROGRESO_HARDWARE.md`, falta sintetizar a guía).
- ✅ Verificación de hardware (con firmware propio, no con demo de ST).

---

### Fase 1 — Bring-up periférico (PWM + ADC + encoder) (~40 h, 4 semanas)

**Objetivo único**: poder generar PWM trifásico complementario sincronizado con lectura sincronizada de las 3 corrientes de fase y posición AS5600, todo a 30 kHz, sin motor conectado.

**Why**: este es el "esqueleto temporal" del controlador. Si esto no funciona, nada más funciona.

**Semana 4 (10 h)**
- Capítulo 28 RM0440 (TIM1) — lectura profunda (4 h).
- Configurar TIM1 en center-aligned, 30 kHz, complementario con dead-time ~500 ns (3 h).
- Verificar en osciloscopio: 6 salidas PWM, complementariedad, dead-time medido (3 h).

**Semana 5 (10 h)**
- Capítulo 21 RM0440 (ADC) + Capítulo 24 (OPAMP) (4 h).
- Configurar OPAMPs internos como PGA para las 3 shunts (2 h).
- ADC1 + ADC2 en dual regular simultáneo, JEXTSEL = TIM1_TRGO, scan multi-canal (4 h).

**Semana 6 (10 h)**
- ISR de fin de conversión ADC: leer 3 corrientes + Vbus + temperatura NTC en cada ciclo PWM (4 h).
- Calibración de offsets: 1000 muestras con motor desconectado, promedio en flash retentivo o en RAM al boot (2 h).
- Verificar jitter del ISR con GPIO toggle al inicio/fin → osciloscopio (2 h).
- Documentar el timing del loop: tiempo de ISR usado, headroom disponible (2 h).

**Semana 7 (10 h)**
- AS5600 vía I²C1 (PB6/PB7), modo "fast mode plus" 1 MHz si soportado (4 h).
- Manejo asíncrono: I²C inicia lectura en ciclo N, dato disponible en ciclo N+1; entre tanto, extrapolar posición con velocidad estimada (3 h).
- Verificar resolución angular efectiva girando el motor a mano (3 h).
- **Plan B**: si I²C en CN5 no funciona, usar salida analógica del AS5600 → ADC (1–2 h extra).

**Entregables Fase 1**:
- ✅ Oscilograma de PWM complementario con dead-time.
- ✅ Log UART con corrientes calibradas (ruido en reposo < 50 mA RMS).
- ✅ Log UART con posición AS5600 a 30 kHz (extrapolada).
- ✅ Tiempo de ISR medido (target: < 15 μs de los 33 μs disponibles).

**Riesgo crítico**: si el ISR consume > 25 μs, NO hay margen para FCS-M2PC con N=2 vectores. Acción: optimizar (CORDIC, fixed-point) o subir Ts a 50 μs (40 kHz → 20 kHz).

---

### Fase 2 — Identificación de parámetros del motor (~30 h, 3 semanas)

**Objetivo único**: tener un modelo eléctrico-mecánico del motor 2804 validado y un mapa de BEMF real (LUT) para alimentar simulación y observador.

**Por qué importa**: el FCS-MPC es model-based. Sin parámetros correctos, todo lo que sigue está sobre arena.

**Semana 8 (10 h)**
- **Rs por inyección DC**: aplicar V_dc conocido en una fase, leer corriente estacionaria. Repetir 3 veces (3 h).
- **Ls por escalón**: aplicar escalón de voltaje, registrar i(t), ajustar exponencial (3 h).
- **λ_pm (BEMF) por back-driving**: girar el motor a mano (o con otro motor), registrar voltaje fase-fase en bornes abiertos vía ADC en modo libre (4 h).

**Semana 9 (10 h)**
- **Forma de BEMF**: registrar e_a(θ), e_b(θ), e_c(θ) sincronizado con AS5600 a velocidad constante (5 h).
- Procesar en Python: FFT, extraer coeficientes de Fourier, guardar como `lut_2804_real.mat` (3 h).
- Comparar con asunción senoidal pura: cuantificar % de armónicos (2 h).

**Semana 10 (10 h)**
- **J e b (inercia y fricción)**: deceleration test desde velocidad alta a parada libre, ajustar curva exponencial (4 h).
- **Cogging torque map**: rotación cuasi-estática (paso angular pequeño), registrar par estático (4 h, si hay celda de torque) o estimar indirectamente vía rizado de corriente sin carga.
- Actualizar `fcs_m2pc_v2/params/motor_params_2804.m` (2 h).

**Entregables Fase 2**:
- ✅ `motor_params_2804.m` con Rs, Ls, λ_pm, J, b identificados.
- ✅ `lut_2804_real.mat` con BEMF real (coeficientes Fourier + LUT).
- ✅ Análisis comparativo: simulación con params identificados vs medición real (figura).
- ✅ Reporte de armónicos de BEMF (relevante para narrativa: el motor 12N14P es "casi senoidal", cuantificar el "casi").

**Decisión a tomar al final**: ¿la BEMF es lo suficientemente no-senoidal como para que el ADALINE aporte vs asunción senoidal pura? Si los armónicos > 1° son < 3% del fundamental, el ADALINE va a ser "demostrativo" más que "necesario" — esto **debe** quedar escrito en la tesis.

---

### Fase 3 — FOC clásico baseline (~50 h, 5 semanas)

**Objetivo único**: motor girando a velocidad controlada con FOC clásico (Clarke/Park + PI de corriente + PI de velocidad + SVPWM). Es el baseline contra el cual se comparará FCS-M2PC.

**Estrategia**: dos caminos en paralelo:
1. **Camino A (rápido, semanas 11-12)**: usar MCSDK con Motor Profiler. Valida que el motor funciona, da números de referencia, pero código de ST.
2. **Camino B (definitivo, semanas 13-15)**: FOC propio en bare-metal CMSIS, basado en pat92fr como inspiración.

**Semana 11 (10 h) — MCSDK como sanity check**
- Configurar workbench del MCSDK con los parámetros de Fase 2 (3 h).
- Ejecutar Motor Profiler — comparar parámetros del Profiler con los identificados manualmente (3 h).
- Lanzar FOC, capturar oscilogramas de corriente y velocidad (4 h).

**Semana 12 (10 h)**
- Caracterizar baseline MCSDK: rizado de corriente, ripple de par estimado, ancho de banda de velocidad (5 h).
- Guardar logs en `data/foc_mcsdk_baseline/` (2 h).
- **Hito**: motor girando bajo FOC, métricas registradas. (3 h colchón).

**Semana 13 (10 h) — FOC propio: estructura**
- Implementar Clarke/Park en C (float primero) (3 h).
- Lectura de encoder AS5600 → ángulo eléctrico (θ_e = 7 · θ_m mod 2π) (2 h).
- Lazo abierto V/f: aplicar V_q = const, rampar θ_e, motor debe girar suavemente (3 h).
- **⚠ Punto crítico**: aquí es donde se quema el inversor si hay error de dead-time o polaridad. Empezar con Vbus = 7 V y limit de corriente en fuente a 300 mA (2 h verificación).

**Semana 14 (10 h)**
- SVPWM (cálculo de sectores, dwell times, asignación a TIM1 CCRx) (5 h).
- PI de corriente sobre i_d, i_q (sintonizar con regla rule-of-thumb: K_p = L · ω_bw, K_i = R · ω_bw) (3 h).
- Verificar respuesta a escalón de i_q_ref (2 h).

**Semana 15 (10 h)**
- PI de velocidad exterior → genera i_q_ref (4 h).
- Sintonizar lazo de velocidad (3 h).
- Comparar FOC propio vs MCSDK: mismas métricas, ¿similar performance? (3 h).

**Entregables Fase 3**:
- ✅ `data/foc_mcsdk_baseline.mat` con métricas del MCSDK.
- ✅ `firmware/foc_custom/` con FOC propio funcional.
- ✅ `data/foc_custom_baseline.mat` con métricas del FOC propio.
- ✅ Comparación FOC propio vs MCSDK documentada.

---

### Fase 4 — FCS-MPC convencional (1 vector) (~40 h, 4 semanas)

**Objetivo único**: implementar FCS-MPC clásico — evaluar los 7 vectores activos del inversor por costo, aplicar el ganador durante todo Ts. Comparar contra FOC.

**Why en el contexto de la tesis**: este NO es el método propuesto, pero es el peldaño intermedio. La tesis muestra cómo el M2PC mejora al MPC clásico, y ese MPC clásico tiene que existir como baseline experimental.

**Semana 16 (10 h)**
- Lectura Kouro 2009 (3 h, lectura cuidadosa).
- Lectura Rodríguez/Cortés cap. 5 (3 h).
- Diseño en papel del modelo discreto del motor en αβ: i_α[k+1] = f(i_α[k], v_α, e_α, Rs, Ls) (2 h).
- Tabla de los 7 vectores activos del inversor (αβ) precalculada (2 h).

**Semana 17 (10 h)**
- Port del modelo predictivo a C (forward Euler) (4 h).
- Implementar la función de costo g = ||i_ref - i_pred||² (2 h).
- Loop de evaluación: predecir, evaluar, escoger arg min, aplicar (4 h).

**Semana 18 (10 h)**
- **One-step delay compensation**: predecir i[k+1] con el vector aplicado en k, luego optimizar sobre i[k+2] (4 h).
- Verificar timing del ISR: con 7 evaluaciones, ¿se mantiene < 25 μs? (3 h).
- Si no: optimizar con CORDIC + fixed-point Q1.15 (3 h, puede arrastrar a semana 19).

**Semana 19 (10 h)**
- Tuning de pesos de costo si se incluyen términos secundarios (switching count, common-mode) (3 h).
- Caracterización completa: rizado, espectro de corriente (FFT del log), frecuencia de switching media (5 h).
- Comparativa FCS-MPC vs FOC documentada (2 h).

**Entregables Fase 4**:
- ✅ `firmware/fcs_mpc/` funcional.
- ✅ `data/fcs_mpc_baseline.mat`.
- ✅ Espectro de corriente del FCS-MPC (debería mostrar el característico espectro disperso que justifica el M2PC).
- ✅ Timing del ISR documentado.

---

### Fase 5 — FCS-M2PC (2 vectores + dwell time) (~50 h, 5 semanas)

**Objetivo único**: implementar el método del paper de Coronado — combinación de 2 vectores activos + nulo dentro de Ts, con dwell times calculados desde la función de costo. Sin ADALINE todavía (BEMF asumida senoidal o de la LUT real).

**Semana 20 (10 h)**
- Lectura profunda `papers/coronado25-draft.pdf` (3 h).
- Reproducir en papel el algoritmo: cómo se calculan t1, t2, t0 (3 h).
- Verificar contra `fcs_m2pc_v2/control/fcs_m2pc.m` (la implementación MATLAB es la referencia exacta) (4 h).

**Semana 21 (10 h)**
- Port del algoritmo M2PC a C (5 h).
- Adaptar el cálculo de dwell para SVPWM en el TIM1 (en M2PC los dwell determinan los CCRx, no van por arg min de 7 vectores) (5 h).

**Semana 22 (10 h)**
- Integración y debug. Timing crítico: ¿el ISR aguanta? (5 h).
- Si no: optimizar lo que falte, reducir Ts si es necesario (5 h).

**Semana 23 (10 h)**
- Pruebas en bajo voltaje (7 V), corriente limitada (4 h).
- Subir progresivamente a Vbus nominal (12 V), validar estabilidad (3 h).
- Caracterización: rizado, espectro, frecuencia de switching (debería ser **fija** ahora) (3 h).

**Semana 24 (10 h)**
- Barrido de velocidad (200, 500, 1000, 1500 rpm) con métricas (5 h).
- Comparativa FCS-M2PC vs FCS-MPC vs FOC — tabla final de baseline (3 h).
- Documentación experimental: figuras candidatas para la tesis (2 h).

**Entregables Fase 5**:
- ✅ `firmware/fcs_m2pc/` funcional, BEMF de la LUT real (sin adaptación).
- ✅ `data/fcs_m2pc_baseline.mat` (este es el **comparador directo** del Fase 6).
- ✅ Tabla con 3 métodos × 4 velocidades.
- ✅ Espectro de corriente del M2PC mostrando frecuencia de switching fija.

---

### Fase 6 — FCS-M2PC + ADALINE online (BEMF) (~50 h, 5 semanas)

**Objetivo único**: agregar el ADALINE+Fourier online al FCS-M2PC. Validar que **converge** sobre el motor real y que **reduce el rizado** vs Fase 5.

**Semana 25 (10 h)**
- Repaso `fcs_m2pc_v2/adaptation/lms_update.m` (1 h).
- Diseñar la estructura en C: base de Fourier x(θ) calculada vía CORDIC (3 h).
- Implementar evaluación de pesos: e_hat_α = W_α^T · x(θ), e_hat_β = W_β^T · x(θ) (3 h).
- Decidir fuente del error ε: **observador de BEMF** (no medición directa) — portar `observers/bemf_observer.m` (3 h).

**Semana 26 (10 h)**
- Implementar observador en C, validar offline contra logs (5 h).
- Implementar regla LMS: W += μ · x(θ) · ε (3 h).
- Decidir frecuencia de update: ¿a 30 kHz o submúltiplo (10 kHz, 5 kHz)? (2 h análisis).

**Semana 27 (10 h)**
- Inicialización: ¿partir con W=0, con LUT senoidal, o con la LUT real? Trade-off entre tiempo de convergencia y "demostración honesta" (3 h).
- Pruebas de convergencia: arrancar con W=0, registrar W(t), verificar que converge a la LUT real en < 1 s (5 h).
- Documentar transitorio de convergencia (2 h).

**Semana 28 (10 h)**
- Pruebas de robustez: cambiar carga, perturbar manualmente, verificar re-adaptación (5 h).
- Pruebas con condición fuera de diseño: BEMF con armónico inducido artificialmente (p.ej., ajuste mecánico que genere desbalance) → ¿el ADALINE lo captura? (5 h).

**Semana 29 (10 h)**
- Caracterización final: rizado, espectro, comparación contra Fase 5 (5 h).
- Tabla final completa: FOC / FCS-MPC / FCS-M2PC / FCS-M2PC+ADALINE (3 h).
- Documentación final, figuras y oscilogramas para la tesis (2 h).

**Entregables Fase 6**:
- ✅ `firmware/fcs_m2pc_adaline/` funcional.
- ✅ Curva de convergencia W(t) en al menos un transitorio.
- ✅ `data/fcs_m2pc_adaline.mat` con todas las métricas.
- ✅ **Comparativa final** que sustenta la contribución central de la tesis.

**Nota de optimización (no comprometida)**: si el LMS de 21 coeficientes saturara el ISR, el **FMAC** (AN5305) puede offload-ear las multiplicaciones acumuladas vía DMA. El ejemplo §1 del AN5305 implementa un FIR adaptativo con regla auto-regresiva — matemáticamente análoga al LMS — y libera 100% del tiempo de CPU dedicado a la operación. Decisión a tomar **después** de medir el timing real en semana 25.

---

## 5. Recorrido total y calendarización aproximada

| Fase | Horas | Semanas (a 10 h/sem) | Mes aproximado* |
|------|-------|----------------------|-----------------|
| 0 — Toolchain | 30 | 3 | Mes 1 |
| 1 — Bring-up | 40 | 4 | Mes 2 |
| 2 — Identificación | 30 | 3 | Mes 3 |
| 3 — FOC baseline | 50 | 5 | Mes 4–5 |
| 4 — FCS-MPC clásico | 40 | 4 | Mes 6 |
| 5 — FCS-M2PC | 50 | 5 | Mes 7–8 |
| 6 — FCS-M2PC + ADALINE | 50 | 5 | Mes 8–9 |
| **TOTAL** | **290** | **29 semanas** | **~7.5 meses** |

\* Asumiendo arranque cuando llegue la placa. Añadir ~15% de contingencia → **8.5 meses realistas**.

---

## 6. Puntos de decisión que NO se pueden saltar

Cada uno de estos requiere una decisión consciente, escrita y justificada **antes** de avanzar.

1. **Fin Fase 1**: ¿el ISR a 30 kHz es viable? Si no, ¿bajamos a 20 kHz o reducimos vectores predichos?
2. **Fin Fase 2**: ¿la BEMF del 2804 es lo suficientemente no-senoidal como para que el ADALINE aporte? Si no, **se reescribe la narrativa del capítulo experimental** hacia "demostración de portabilidad + captura de cogging" en vez de "captura de BEMF".
3. **Fin Fase 3**: ¿FOC propio en bare-metal o seguir con MCSDK como base y solo agregar M2PC arriba? El bare-metal da control total pero cuesta semanas; MCSDK simplifica pero ata a configuración de ST.
4. **Fin Fase 4**: si el M2PC no cabe en el ISR, ¿reducimos horizonte, bajamos Ts, o saltamos directamente a HRTIM?
5. **Fin Fase 5**: si el M2PC no muestra mejora clara vs FOC en este motor, el comparador final del Fase 6 cambia (vs FOC en vez de vs M2PC).

---

## 7. Riesgos identificados y mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|
| AS5600 latencia domina el control | Media | Alto | Plan B: AS5048A/AS5047P por SPI |
| Inversor se quema en Fase 3 | Media | Crítico | Vbus 7 V + límite corriente 300 mA al inicio |
| ISR no alcanza para M2PC a 30 kHz | Media | Alto | Bajar a 20 kHz o fixed-point Q1.15 |
| BEMF muy senoidal → ADALINE poco visible | Alta | Medio | Reencuadre narrativo a "convergencia + cogging" |
| Tiempos optimistas | Alta | Medio | +15% contingencia ya incluida en total |

---

## 8. Lista de compras (estado actualizado 2026-05-17)

**Adquirido y cableado**
- ✅ Osciloscopio.
- ✅ Motor 2804 (12N14P) con AS5600 integrado de fábrica.
- ✅ B-G431B-ESC1 + cableado físico completo (motor a J7, AS5600 a J8: **PB8=SCL**, PB7=SDA, 5V, GND).
- ✅ **Fuente DC regulada**: buck-boost ZK-4KX alimentada desde fuente de PC vieja. Modos CV/CC.
- ✅ Raspberry Pi como host de programación (USB → ST-LINK de la placa).

**Necesario para Fase 3+ (dyno mecánico, no bloqueante para los primeros encendidos)**
- ⚠ Coupler flexible 8 mm + segundo motor 2804 (para carga mecánica). Diferible hasta caracterización con carga.

**Plan B opcional**
- (Solo si AS5600 falla) AS5048A o AS5047P por SPI.

**Sin pendientes bloqueantes para empezar Fase 0.**

---

## 9. Plan de lectura — Fase 0, Semana 1 (mientras llega la placa)

Toda la bibliografía bloqueante ya está descargada en `papers/`. Dedicar las primeras ~10 h así:

1. **UM2516 completo** (~2 h) — pinout exacto del CN5, solder bridges, conectores accesibles.
2. **MB1419 schematic** (~1 h) — leer las redes que importan: TIM1→gates, OPAMP→ADC, I²C en CN5, USART2 en J3.
3. **RM0440 Cap 7 (RCC)** (~3 h) — necesario para configurar HSE/PLL a 170 MHz.
4. **RM0440 Cap 28 (TIM1) — lectura introductoria** (~2 h) — identificar los registros que usaremos en Fase 1.
5. **Paper Kouro 2009** (~3 h) — frame mental de FCS-MPC, lectura cuidadosa.
6. Setup STM32CubeIDE + MCSDK 6.2 + STM32CubeMonitor (~1 h) — solo instalación, sin proyecto.

Lectura opcional si sobra tiempo: post de brushless.zone sobre CORDIC (15 min) y AN5325 §1-2 (1 h).

---

## 10. Tracking — cómo usar este documento

- Al final de cada **semana**: marcar tareas completadas, anotar horas reales vs planificadas, escribir un párrafo de status en `progreso_hardware.md` (crear).
- Al final de cada **fase**: actualizar `project_hw_bench_2804.md` en memoria con métricas y aprendizajes.
- Cada **decisión** del §6: documentar con fecha y justificación.
- Las **figuras y .mat** generados se versionan en git con LFS si pesan.

---

**Próximo paso inmediato (esta semana)**: cuando confirmes que el planning te sirve, empezamos por leer UM2516 juntos y armamos el repo `firmware/` con la estructura inicial.

---

## 11. Subplan Fase 5 → 6 (escrito 2026-10-18, sesión 22)

### Contexto

Al cerrar la sesión 22 el estado real contra este plan es: Fases 0–1 cerradas; Fase 2 parcial
(R medido, λ = 3.47 mWb y forma de BEMF identificadas en tres corridas consistentes; L del
planning; J y b sin medir); Fases 3 y 4 **saltadas a propósito**; Fase 5 corriendo en banco
(`07_m2pc`, 66 candidatos a 50 kHz, `overruns = 0`) pero **sin métrica válida**: todo lo medido
en lazo cerrado antes de la sesión 22 se tomó con la Park rota (CORDIC, N1.27). Fase 6 sin
empezar. Calendario: ~5 meses efectivos para llegar aquí contra 7–8 del plan.

Lo que falta para la contribución central de la tesis es corto y está acotado: una métrica
del M²PC con BEMF sinusoidal, un dataset de BEMF vs θ, y el ADALINE encima. Este subplan
lo ordena en tres experimentos encadenados, cada uno con entregable.

### Decisiones tomadas (§6, puntos 3 y 5, y una nueva)

| Decisión | Resolución | Justificación |
|---|---|---|
| **Fase 3, FOC baseline** | **No se implementa.** | La comparación del banco es M²PC-sinusoidal vs M²PC+ADALINE: es la que sustenta la contribución (el ADALINE mejora *al predictor*, no al controlador). La comparación vs TRAP/FOC queda en simulación (Cap. de simulación, 69 %/61 %). Un FOC bare-metal son ~50 h que no aportan un número nuevo a la tesis. |
| **Fase 4, FCS-MPC 1 vector** | No se implementa. | Mismo argumento; el M²PC ya corre. |
| **Fase 2, L, J, b** | No se identifican salvo necesidad. | L pesa ~6 mV de 755 en la reconstrucción de e a 300 rpm (L·ω_e·I); J y b solo afectan al PI de velocidad, que ya cumple. Si el ADALINE a 600–900 rpm muestra sesgo atribuible a L, se mide entonces (escalón de tensión, 3 h). |
| **Punto de decisión 2** (¿ADALINE necesario o demostrativo?) | **Demostrativo**: armónicos h2…h7 ≤ 1.1 % con el CORDIC sano (sesión 22). | Queda escrito aquí y debe ir al capítulo experimental: el 12N14P es casi sinusoidal; la aportación del ADALINE en este banco es demostrar convergencia, portabilidad y captura de lo que *sí* hay (dead-time en los cruces por cero, asimetrías), no una reducción grande de rizo. |

### Experimento 1 — Seguimiento de corriente puro (M²PC con BEMF sinusoidal) — ~1 sesión

**Objetivo**: la cifra de desempeño del M²PC que sirve de baseline al ADALINE: error de
seguimiento de i_q e i_d y rizo de corriente, **medidos muestra a muestra en la ISR**.

**Por qué no sirve el logger actual**: promedia i_αβ en 1 ms (esconde el rizo de 50 kHz) y
manda θ_e al cierre de la ventana (desfasa i_dq). La métrica calculada de ese log está
sesgada hacia abajo y desplazada. README_scope.md ya lo advierte.

**Firmware (`07_m2pc`), telemetría sincronizada**:
- En `control_step`, tras la Park: ε_q = i_q − i_q*, ε_d = i_d − i_d* (= i_d). Acumular en la
  ventana de 1 ms: Σε_q, Σε_q², Σε_d², Σi_q, Σi_d, n (int64 en Q8²: 50 muestras × (40 LSB·256)² cabe de sobra).
- Publicar a 1 kHz en la línea `L` (o una línea `E` nueva, con **secuencia** para detectar
  pérdidas): media y RMS de ε_q y ε_d, media de i_q e i_d, i_q* vigente. Todo en Q8 LSB.
- Costo esperado: < 0.3 µs (seis MAC en int64). Comprobar `paso_max` antes/después.
- `scope_m2pc.py`: leer la línea nueva y mostrar ε_q/ε_d *reales* en lugar de los aproximados.

**Protocolo de banco** (sin carga, i_q* constante = runaway hasta 1390 rpm, sesión 20):
1. `SPEED_LOOP = 1` como contención, pero con **escalones de i_q\*** inyectados *sumados* a la
   salida del PI (feedforward): ±100 mA durante 200 ms cada 2 s. El PI corrige la velocidad
   lentamente (ancho de banda ~3 Hz); dentro de los 200 ms el lazo de corriente responde solo.
   Métrica: tiempo de subida de i_q (esperado: 2–3 periodos de 20 µs) y RMS de ε_q en régimen.
2. Régimen a 300 rpm, 60 s: RMS(ε_q), RMS(ε_d), rizo pico-pico de i_q. Repetir a 600 rpm.
3. Opcional, si hay carga mecánica disponible (freno por fricción en la llanta): i_q* = 200 mA
   constante con `SPEED_LOOP = 0`, 10 s, límite de velocidad por software (trip si ω > 800 rpm).

**Entregable**: tabla "M²PC sinusoidal: RMS(ε_q), RMS(ε_d), rizo, t_subida a 300 y 600 rpm"
+ figura de un escalón (i_q*, i_q muestra a muestra si cabe en el UART; si no, media y
envolvente por ms). Reemplaza al "160 de 200 mA" de la sesión 20, que no vale.

### Experimento 2 — Trayectoria senoidal de velocidad, ambos sentidos — ~1 sesión

**Objetivo doble**: (a) demostrar seguimiento de trayectoria y paso por cero velocidad con
inversión de giro; (b) **generar el dataset del ADALINE**: BEMF vs θ_e barriendo velocidad y
signo, que es mucho más informativo que 300 rpm constantes (separa lo que escala con ω —la
BEMF— de lo que no —dead-time, offsets).

**Firmware (`07_m2pc`)**: referencia ω* = A·sin(2π·f·t) generada en el lazo de 1 kHz; A y f
por `#define`. Primera corrida A = 300 rpm, f = 0.1 Hz; luego A = 600, f = 0.2 Hz. Checks
previos (no deberían fallar, pero se verifican): ω̂ con signo en el PLL, sector por θ_e (no
depende del signo), saturación simétrica del PI, `I_TRIP` intacto. El runaway no aplica: la
referencia es acotada.

**Dataset**: el log existente (`L`: i_αβ, u_αβ, θ_e, ω̂ a 1 kHz) ya sirve. Limitación:
el armónico 10 a 600 rpm está a 700 Hz, al borde del logger de 1 kHz; **no subir de 600 rpm
para el dataset** salvo que se añada un log decimado desde la ISR. A 300 rpm (h10 = 350 Hz)
no hay problema.

**Entregable**: figura ω*(t) vs ω̂(t) con la inversión; RMS del error de velocidad; log
`datos_banco/m2pc_s2x_seno_±600.log` etiquetado como dataset.

### Experimento 2b — BEMF en circuito abierto (medida directa, sin modelo) — HECHO (sesión 24)

**Resultado**: THD 0.42/0.67/0.64 % por fase, h3 ≤ 0.6 %, h5 ≤ 0.5 %, resto < 0.1 %; λ = 3.49 mWb;
alineación del encoder +2.85°. Hizo falta una resistencia externa de 10 kΩ de +12 V a OUT1 para
fijar el modo común (la red de la placa es para 6‑step). Detalle en PROGRESO sesión 24.

**Por qué** (sesión 23, Exp. 3a): reconstruir e = u − R·i − L·di/dt desde la tensión comandada
deja los armónicos del 2804 en "≤ 2 %" pero no los resuelve: el dead‑time y R se confunden
con ellos, y el L·di/dt de medias de 1 ms mete ±120 mV de ruido. La literatura dice lo mismo
(§3.5). La placa tiene la red de sensado de BEMF del 6‑step: fase ─10 kΩ─┬─2.2 kΩ─GND (ratio
0.18) en **PA4 (BEMF1), PC4 (BEMF2), PB11 (BEMF3)**, diodos a `GPIO_BEMF` (PB5) que debe
quedar en alta impedancia/alto para no clampar (esquemático MB1419 hoja 4, UM2516 Tabla 4).

**App `08_bemf_oc`**: puente apagado (`MOE = 0`, los seis MOSFET off), ADC regular sobre los
tres canales de BEMF disparado por TIM2 a 10 kHz, PLL del encoder como en 06, log de
(t, θ_e, ω̂, v_a, v_b, v_c) por UART. Con las tres fases en abierto los tres divisores forman
un neutro virtual: e_x − ē = v_x − mean(v). La llanta se gira a mano (o con una cuerda): un
barrido de ~600 → 0 rpm en decaimiento libre da todas las velocidades de una vez y, de regalo,
**J y b** por el decaimiento (si se quiere confirmar el 1.5·10⁻⁵).

**Análisis**: e_abc(θ_e) por bins y por mínimos cuadrados con la base de Fourier, normalizado
por ω̂ muestra a muestra; armónicos con intervalo de confianza; comparar con la forma de la
Exp. 3a (misma figura). Resolución esperada: 4.4 mV por LSB en la fase, ~340 LSB a 600 rpm →
armónicos al 0.3 %. Verificar el ratio del divisor con el DMM antes (tolerancias).

**Entregable**: **la BEMF real del 2804 con incertidumbre** — un dato que no existe publicado
(§3.5) — y la forma de referencia contra la que se juzga lo que aprende el ADALINE en línea.

### Experimento 3 — ADALINE: offline primero, en línea después — 2–3 sesiones

**3a. Offline (Python, sin tocar firmware)** — **HECHO en la sesión 23**: `python_scripts/bemf_reconstruct.py`
(mínimos cuadrados + LMS; λ = 3.457 mWb, fase +1.5°, THD ≤ 2 %, μ = 2·10⁻³ converge en ~2 s;
coeficientes en `python_scripts/bemf_2804_fourier.json`). Lo planeado era:
- Entrada: logs de 300 rpm (sesión 22) y del Experimento 2.
- e_αβ reconstruida como ya hace `analyze_m2pc_log.py` (e = u − R·i − L·Δi/Δt, DC quitado).
- Forma objetivo s = e/(λ·ω̂) (normalizar por velocidad, como en `fcs_m2pc_v2`: así el mismo W
  vale a cualquier rpm y signo).
- Base x(θ) = [cos kθ, sin kθ], k = 1…10 (20 coef. por eje; el término constante se omite:
  el DC de e no es BEMF). LMS de `lms_update.m` traducido literal.
- Validar: W converge a los coeficientes que ya da la FFT de `analyze_m2pc_log.py`; curva
  W(t); μ y tiempo de convergencia; comparar W de 300 vs 600 rpm vs ±.
- Decidir aquí **μ**, **número de armónicos** (probablemente 5–7 bastan; 10 es el techo), y
  la **cadencia** del update.

**3b. En línea (C)**:
**Estado s25**: `lib/adaline.{c,h}` integrado en 07 y compilado OFF/ON, con pruebas
C↔Python en host. Pesos Q27 (no Q31), centro eléctrico con factor de 7 polos,
doble buffer para publicación e inicio sinusoidal. Banco pendiente; protocolo en
`firmware/apps/07_m2pc/README_adaline.md`. La versión actual no corrige la atenuación
sinc ni separa DC/dead-time: aprende un modelo efectivo, no BEMF pura. La cadencia
real del LMS debe medirse (UART polling); no asumir 1000 updates/s.
- **Observador**: el mismo algebraico, pero **no a 50 kHz**: L/Ts = 43 Ω → un LSB de ruido de
  corriente son 1.3 V de "BEMF" (lo advierte `bemf_observer.m`). Se calcula sobre los
  promedios de la ventana de 1 ms que el logger ya acumula (L/Ts_eff = 0.86 Ω → 26 mV/LSB), con
  θ_e *promediado* en la ventana (no al cierre). LMS a 1 kHz en el lazo lento, fuera de la ISR.
- **Evaluación** e_hat = W·x(θ_e) sí va en la ISR, a 50 kHz, sustituyendo la línea
  `λ·ω·[−sin, cos]` de `control_step`. x(θ) con el CORDIC (ahora fiable) por recurrencia de
   ángulo múltiple (implementada s25, reutilizando sin/cos de Park). El baseline s24
   cuesta 15.8 µs: quedan ~4.2 µs, no 7; **medir antes de declarar que cabe**.
- Inicialización: W = sinusoidal (λ, k = 1) para una "demostración honesta" de que converge a
  lo mismo partiendo de la hipótesis del modelo; y una corrida desde W = 0 para la curva de
  convergencia.
- Opción `ADALINE = 0/1` por `#define` para la comparativa en condiciones idénticas.

**3c. Comparativa final** (la contribución): Experimento 1 repetido con `ADALINE = 1`. Tabla
RMS(ε_q), RMS(ε_d), rizo, a 300 y 600 rpm, sinusoidal vs ADALINE, + curva W(t), + forma
aprendida vs FFT. Con armónicos ≤ 1 % la mejora será pequeña: **se reporta tal cual**, con la
discusión de que lo que captura es el residuo no sinusoidal (dead-time en los cruces).

### Secuencia y criterio de avance

```
Exp. 1 (telemetría ISR + escalones) ──► cifra baseline M²PC      [HECHO s23]
        │
        ▼
Exp. 2 (seno ±600 rpm) ──► figura trayectoria + dataset          [HECHO s23]
        │
        ▼
Exp. 3a (ADALINE offline) ──► μ, nº armónicos, W de referencia   [HECHO s23]
        │
        ▼
Exp. 2b (BEMF circuito abierto) ──► forma real con incertidumbre   [HECHO s24: THD 0.4–0.7 %]
        │
        ▼
Exp. 3b (ADALINE en línea) ──► [s25: réplica Q27 + tests C/Python y port C compilados] ──► [s26 BANCO: presupuesto ✓ a 300 rpm con H=3; el seguimiento es INSENSIBLE a ≤20 % de error de BEMF (rizo de conmutación domina): no hay mejora medible posible; el LMS identifica λ en ~90 s pero su punto fijo es la tensión efectiva (deriva lenta). 3c = tabla con controles a 300 rpm]
        │
        ▼
Exp. 3c (comparativa) ──► tabla final de la tesis
```

Regla heredada de la sesión 22: **tras cualquier cambio en `lib/` o en flags, correr 06** y
mirar el autotest del CORDIC y que i_dq sea DC.

### Riesgos de este tramo

| Riesgo | Mitigación |
|---|---|
| El UART a 921600 no da para telemetría por muestra | No se intenta: estadísticos por ms (media, RMS) y, para la figura del escalón, un buffer de 2000 muestras en RAM volcado después. |
| La inversión de giro excita algo no previsto (signo en sector/BEMF) | Primera corrida a A = 100 rpm, f = 0.05 Hz, mirando el scope; `I_TRIP` activo. |
| El observador en línea es ruidoso | Promedios de 1 ms (decidido arriba); si no basta, LMS a 100 Hz sobre 10 ms. |
| El ADALINE no mejora nada medible | Es un resultado válido con el punto de decisión 2 ya escrito. Se complementa con la prueba de robustez del plan original (Fase 6, semana 28): inducir un armónico y ver que lo captura. |
| x(θ) con 10 armónicos no cabe en la ISR | Recurrencia cos(kθ) = 2cosθ·cos((k−1)θ) − cos((k−2)θ): 2 MAC por armónico, sin CORDIC extra. |

---
