# Progreso — Implementación experimental FCS-M2PC + ADALINE en B-G431B-ESC1

> Bitácora cronológica del avance. Cada entrada documenta qué se hizo, qué se aprendió y qué quedó pendiente.
> Documentos guía: [PLANNING_HARDWARE.md](./PLANNING_HARDWARE.md), `papers/`, memoria del proyecto.

---

## Sesión 22 — 2026-10-18 — Setup final (llanta + base), scope en vivo, y el CORDIC que devolvía el coseno dos veces

**Hito**: el setup mecánico definitivo queda validado (λ y BEMF sin cambio, rizo de velocidad a la mitad), el scope en vivo funciona con el motor, y una prueba de regresión de un minuto destapó que **el CORDIC entregaba `sin ≡ cos` desde la sesión 20**: el lazo cerrado de las sesiones 20–21 corrió con la Park y el modelo de BEMF rotos. Corregido en `lib/cordic`, con autotest permanente en 06. FIELD_NOTES N1.27.

### Resultados de banco (en orden cronológico)

1. **Llanta + base impresas, 07 a 300 rpm, 220 s** (CORDIC aún roto, sin saberlo): λ = 3.465 mWb (−1.0 %), e_q = 755 mV, h2/h3/h5/h7 = 0.4/0.8/1.1/0.9 %. ω = 297.1 ± 4.4 rpm (sin llanta: ± 11.8). Los parámetros eléctricos no cambian; la inercia baja el rizo de velocidad a la mitad y, de paso, los armónicos aparentes (parte de lo que medíamos era el rizo de velocidad "manchando" la forma). Régimen en < 2 s; el PI no necesita retoque.
2. **Regresión de 06 con `tim1_trgo2`** (deuda de la sesión 21): lag y ω idénticos a la sesión 19 (105.0 ± 1.9°, 85.7 rpm) pero **i_dq pulsante a 2·f_e**, 169 ± 160 / −186 ± 171 mA, con |i| colapsando a ~30 mA cada medio ciclo.
3. **Motor y medida descartados con 02** (barrido 0/170/510): pico 16/17/17 y 38/39/41 raw, RMS 1/8/23 — idéntico a la sesión 18. De paso, la cadena ADC por `tim1_trgo2` queda validada en 02 y 06.
4. **Firma cuantificada**: DC + 2·f_e con ceros = Park con el mismo factor en ambos productos. Con I = 343 mA (RMS 8 raw × √2) predice DC (171, −171) y |i|máx 485; medido (169, −186) y ~500. Publicados `s`, `c`, θ_e en el stream de 06: **`s == c == cos θ_e`** en todas las muestras.
5. **Causa 1 — el `CSR` del CORDIC seguía en su reset (`0x50`, `NRES = 0`)**: la escritura justo tras habilitar el reloj se perdía. En la sesión 19 (lib en `-Og`, sin prefetch) los wait-states tapaban el hueco; con `-O2` + `PRFTEN` (sesión 20) dejó de funcionar sin tocar `cordic.c`. Fix: `__DSB()` + relectura del `CSR` hasta que coincida.
6. **Causa 2 — dos `ldr` consecutivos a `RDATA`**: con `NRES = 1` ya efectivo, el 3.2 % de los pasos seguía con `s == c` (la segunda lectura llega antes de que el periférico cargue el segundo resultado; `__DSB()` entre lecturas no lo arregla). Fix: `RESSIZE = 1`, sin y cos en **una sola lectura** (como el MCSDK). Y una trampa del manual por el camino: con `RESSIZE = 1` hay que poner `NRES = 0` — `NRES` cuenta lecturas de 32 bits, no resultados; con `NRES = 1` cada lectura devolvía el ángulo *anterior* y `RRDY` se quedaba en 1. `CSR` final `0x00200060`.
7. **06 con el fix**: i_d = **323 ± 16**, i_q = **0.7 ± 7.0 mA**, s² + c² = 1.00000 ± 0.00002 (2257 muestras). Mejor que la sesión 19 (305 ± 25 / 4 ± 16), que ya llevaba el 3 % de lecturas repetidas.
8. **07 con el fix, 240 s a 300 rpm**: i_q* medio **43 → 7.7 Q8** (el controlador dejó de compensar su propio modelo), rizo de velocidad dominante **2·f_e → f_e** (34.7 Hz), i_d = +8 / i_q = +20 mA, λ = **3.476 mWb** (−0.7 %), h2/h3/h5/h7 = 1.1/0.4/1.0/0.5 %, `overruns = 0`, `paso_max` 13.32 µs. Gráfica: `firmware/apps/07_m2pc/doc/bemf_s22_cordic_fix.png`.
9. **Scope en vivo** (`scope_m2pc.py`) probado con el motor girando: lectura exclusiva del puerto, log automático, los ocho paneles en tiempo real. Dos tropiezos de operación anotados en `README_scope.md`: hay que abrirlo con la placa ya en 07 (si corre otra app a 115200 el log es basura), y no sobreescribe un log existente.

### Consecuencia para la tesis

Las cifras de *lazo cerrado* de las sesiones 20–21 (seguimiento i_q 160/200, i_q* de −20 a +70, rizo 2·f_e) **no valen como datos**: se tomaron con Park y BEMF rotas. Las identificaciones de λ y la forma de la BEMF sí valen (offline, en αβ, con θ del log). La prueba de seguimiento de corriente (`SPEED_LOOP = 0`, i_q* = 200 mA) hay que repetirla.

### Código

`lib/cordic.{c,h}` (relectura del CSR, `RESSIZE = 1`, `NRES = 0`, una lectura; API Q31 intacta), `apps/06_openloop_sync` (autotest del CORDIC al arrancar: CSR + cinco ángulos; `the`, `s`, `c`, `csr` en el stream). Carpeta nueva **`datos_banco/`** para los logs del banco propio (`datos_axel/` son las grabaciones de otra persona con otro motor); `scope_m2pc.py`, `analyze_m2pc_log.py` y `README_scope.md` apuntan ahí; `.gitignore` excluye `datos_banco/*.log` explícitamente. Logs de hoy: `m2pc_live_20261018_111051_678526.log` (llanta, CORDIC roto), `m2pc_s22_cordic_fix.log`.

### Deuda, conciliada

| Item | Estado |
|---|---|
| Apps 02–06 con `tim1_trgo2` | **Cerrado**: 02 y 06 corridas en banco, señal idéntica. |
| Sesgo +0.5 LSB en Σi | No se persigue (< resolución). Hoy 274 Q8 ≈ 1.1 LSB con el CORDIC sano; vigilar. |
| Glitch fase C | No reproducido en ~10 min de scope en vivo. Pasa a "limitaciones" salvo que reaparezca. |
| Race `adc_get_vbus_raw` | Diferido hasta que Vbus entre en el modelo (`VBUS_MV` fijo hoy). |
| M²PC en CCM-SRAM | Opcional; `overruns = 0` con retardo entero. |
| 7 citas del `.bib` de memoria | Tarea de escritorio pendiente. |

### Regla nueva

**Cada vez que cambie `lib/` (o los flags de compilación), correr 06** y mirar el autotest del CORDIC y que i_dq sea DC. Es un minuto y es la única prueba que ve la cadena completa sin un lazo externo que tape errores.

### Para las sesiones siguientes

1. Repetir el seguimiento de corriente puro (`SPEED_LOOP = 0`, i_q* = 200 mA, con carga o con límite de velocidad) → cifra que reemplaza al "160 de 200".
2. Telemetría sincronizada en la ISR (errores muestra a muestra, promediados, con secuencia) para el scope y para la métrica de seguimiento.
3. Identificación a 600–900 rpm; ADALINE offline y luego en línea; rizo de par vs baseline.

---

## Sesión 21 — 2026-10-16/17 — Offset de corriente en línea: el valle del PWM como cero de corriente

**Hito**: el offset de los tres canales de corriente se mide en continuo, 50 000 veces por segundo, en el valle del PWM (high-side ON, shunts sin corriente). La media de ia+ib+ic —que es cero físicamente— pasa de derivar ±2 LSB en 10 min a quedarse en **+0.48 ± 0.38 LSB**. FIELD_NOTES N1.26.

### Resultados de banco (en orden cronológico)

1. **Paso 0 — offsets en Q8 + Clarke completo** (`lib/adc`, `07_m2pc`). El `uint16_t` truncado escondía hasta 1 LSB por fase; con fracción, el DC de β al arrancar bajó de −1.35 a −0.7 LSB. Clarke completo (i_α = (2ia−ib−ic)/3) elimina el offset común de α. Diagnóstico nuevo a 1 Hz: media de ia+ib+ic.
2. **La deriva no es térmica-monótona.** Cuatro calibraciones de arranque en dos días se mueven ±1 LSB sin relación limpia con la temperatura del motor (c: +1.0 ayer al calentarse, −0.7 hoy). Dentro de una corrida, la suma de corrientes se pasea 0 → +1.5 → 0 → −2 LSB. El OPAMP es interno al die.
3. **Balance de tensión descartado** como estimador: su DC (−0.7 LSB en β) no sigue al offset; lo domina el dead-time.
4. **Disparo del valle** (`pwm_valley_trigger_enable`, `adc_valley_offset_enable`): TIM1 CH6 interno con `CCR6 = 1`, `MMS2 = 1100` (OC4REF ∪ OC6REF → TRGO2), ADC inyectado en `tim1_trgo2`. ISR a 100 kHz separada por `CR1.DIR`; JEOS se limpia al entrar (el del valle llega durante el control del pico). Validez: los tres CCR activos ≥ 500 ticks (1.6 % descartado a 300 rpm). Bloques de 2¹⁶ muestras + EWMA 1/4 (τ ≈ 5 s). `paso_max` 13.07 µs, `overruns = 0`.
5. **Validación cruzada** (modo solo reporte, 3 min): Σ(valle − cal) sigue a −media(ia+ib+ic) en todo el tramo (+1.46/−1.09 … +2.44/−1.96 LSB) mientras el offset se mueve 1 LSB/min.
6. **Offset en línea activo** (5 min a 300 rpm, `OFFSET_ONLINE = 1`): media(ia+ib+ic) = +0.48 ± 0.38 LSB sin tendencia; DC de e_αβ constante (−0.5/−0.8 LSB, ±0.03) → lo que queda es dead-time/modelo. λ = 3.43 mWb, h2 0.8 %, h3 2.5 %. Gráfica: `firmware/apps/07_m2pc/doc/bemf_s21_valle_on.png`.

### Código

`lib/adc.{c,h}` (offsets Q8, rama del valle en la ISR, `tim1_trgo2`), `lib/pwm.{c,h}` (`MMS2`, `pwm_valley_trigger_enable`), `apps/07_m2pc` (`CLARKE_FULL`, `OFFSET_ONLINE`, línea `[off]`), `python_scripts/analyze_m2pc_log.py` (`--win`: DC de e_αβ por ventanas). Logs en `datos_banco/m2pc_s21_*.log` (local, no versionados).

### Pendiente al cerrar sesión 21

- Las apps 02–06 ahora disparan el ADC por `tim1_trgo2` (misma señal, otro mux): no re-probadas en banco.
- Sesgo fijo de +0.5 LSB en la suma (0.15 LSB por fase): por debajo de la resolución, no se persigue.
- Sigue la lista de la sesión 20: identificación a 600–900 rpm, ADALINE offline, PI y rizo de par vs. baseline, escritura.

### Sesión 21b — 2026-10-18 — Scopes en vivo (sin tocar firmware)

Antes de subir a 600–900 rpm se quiso *ver* los dos lazos, no solo sus medias. Se evaluaron
PlotJuggler (no instalado), Serial Studio (parte Pro) y un visor propio; se hizo el visor
propio con lo ya disponible (Matplotlib + pyserial): `python_scripts/scope_m2pc.py`,
documentado en `python_scripts/README_scope.md`, pruebas en `test_scope_m2pc.py` (5, sin
hardware).

- Ocho paneles con eje de tiempo común: ω y ω*, error de velocidad, i_d, i_q con i_q*,
  errores de corriente, i_αβ, u_αβ comandada, salida del PI. Pausa (solo la vista; la
  adquisición y el log siguen), ventanas 0.1/1/3/10 s, zoom con la barra de Matplotlib.
- Es el **único lector** del puerto (`exclusive=True`): no correr `cat /dev/ttyACM0` a la vez.
  Guarda todo lo recibido en `datos_banco/m2pc_live_<fecha>.log` (ignorado por git).
- `--replay <log>` reproduce un log existente a ritmo real sin abrir el puerto; `--rate 2`
  acelera. Validado con `m2pc_300rpm_2026-10-14.log` y `m2pc_s21_valle_on.log`.
- ω* sale del banner `w* = N rpm`; si se conecta con el motor ya girando se asume 300 rpm
  (marcado en pantalla; `--speed-ref` lo cambia).

**Limitaciones del stream actual, escritas en el propio panel**: i_dq y sus errores son
*aproximados* (el logger promedia i_αβ en 1 ms, manda θ_e al cierre de la ventana y lee i_q*
al imprimir → desfase aparente, más a más rpm); no se ve el rizado de 50 kHz; u es comando,
no medida. Para el error de seguimiento como métrica de tesis hace falta telemetría
sincronizada en la ISR (errores calculados muestra a muestra y promediados) con secuencia —
es el siguiente paso del visor, antes de la identificación a 600–900 rpm.

En banco no se llegó a ver en vivo: la placa quedó en `reset halt` de la sesión 21 y el
puerto abrió sin muestras. Pendiente: encender fuente, `reset run` (o ciclo de alimentación)
y abrir el visor. **Cerrar el visor no detiene el motor.**

---

## Sesión 20 — 2026-10-14 — FCS-M²PC en lazo cerrado: primer giro, lazo de velocidad y BEMF identificada

**Hito**: el controlador de corriente de la tesis (Coronado 2025, Alg. 3) corre en el banco a 50 kHz con `overruns = 0`, el lazo de velocidad lo mantiene a 300 rpm, y del log a 1 kHz sale la BEMF real del motor: **λ = 3.46 mWb** (−1 % del planning), forma casi sinusoidal (armónicos < 2.5 %), e_d ≈ 0.

### Resultados de banco (en orden cronológico)

1. **Primer cierre de lazo** (iq* = 200 mA, sin lazo de velocidad): el motor gira, cero disparos. Pero `overruns ≈ 50 000/s`: el paso tarda 13 µs y el modelo de medio periodo exigía < 10 µs.
2. **Presupuesto medido por tramos**: sensado 3.0 µs, M²PC 9.5 µs — 24 ciclos por candidato (dos saltos tomados por iteración + 4 WS de flash), no los 10 estimados. Desenrollar empeora (958 instrucciones no caben en caché: 10.7 µs). Prefetch de flash activado (`clock.c`): 9.8 → 9.5 µs.
3. **Cambio a retardo entero (Cortés 2012)**: los CCR deben latchearse en el pico. La UEV se mueve al pico arrancando con `RCR = 0` antes del `UG` y escribiendo `RCR = 1` tras `CEN` (RM0440 §29.3.5, "written after launching"); el detector `UIF` de `m2pc_step` lo verificó: `overruns` 50 000/s → **0**. El modelo predice i(k+1) con el comando en curso y i(k+2) con cada candidato. Con eso el seguimiento mejoró: iq 95 → 160 mA (ref 200), id 90 → 40.
4. **Sin carga, control de corriente puro = runaway** hasta el límite de tensión (1390 rpm): física, no bug. Y con iq* = 0 el motor arranca igual (~100 rpm): resolución de 30 mA/LSB (un sesgo de 0.7 LSB es invisible para el costo).
5. **Lazo de velocidad** (PI de `pi_speed_controller.m`, 1 kHz, anti-windup; Kp 2 mA/rpm, Ki 20 mA/(rpm·s), |iq| ≤ 300 mA): **300 rpm ± 5 %** con iq* entre −20 y +70 mA. Ripple de velocidad a la frecuencia eléctrica (~35 Hz), que el lazo de 3 Hz no persigue.
6. **Logger a 1 kHz** (921600 baud): medias de ventana de i_αβ y u_αβ, θ_e y ω̂. `python_scripts/analyze_m2pc_log.py` reconstruye e = u − R·i − L·di/dt y la proyecta en dq:

```
λ identificado = 3.456 mWb (nominal 3.5, −1.2 %)   e_d = +3.5 mV ≈ 0   e_q = 754 mV
armónicos de e_α/(λω) [% del fundamental]: h2=2.5 h3=1.9 h4=0.4 h5=1.1 h6=0.1 h7=1.2
```

   Gráfica: `firmware/apps/07_m2pc/doc/bemf_300rpm_2026-10-14.png`. e_d ≈ 0 confirma por tercera vía (independiente de 05 y 06) que el offset y el signo del encoder son correctos.

7. **Deriva del offset de corriente**: u_β tenía −118 mV de media con i_β medida ≈ 0 → el lazo regula a cero la corriente *medida* mientras circula un DC real de ~40 mA (1.3 LSB). El offset calibrado al arranque deriva con la temperatura del OPAMP. Para el análisis se quita el DC de e (una BEMF no tiene DC). **Deuda**: recalibración de offset en línea (pat92fr usa las muestras del valle, donde la corriente por el shunt es cero).

### Consecuencia para la narrativa de la tesis

PLANNING §Fase 2 fijó la regla: "si los armónicos > 1° son < 3 % del fundamental, el ADALINE va a ser demostrativo más que necesario — esto debe quedar escrito". Con h2…h7 < 2.5 % **estamos en ese caso**. Matiz: a 300 rpm el dead-time (≈ 0.3 V contra 0.76 V de BEMF) deforma los cruces por cero (visible en la figura ~150° y ~200°); hay que repetir la identificación a 600–900 rpm antes de dar el número por definitivo.

### Código (todo en el commit de la sesión)

`lib/m2pc.{c,h}` (retardo entero, detector UIF), `lib/motor_params.h`, `lib/pwm.c` (UEV en el pico), `lib/clock.c` (prefetch), `apps/07_m2pc` (PI + logger), `python_scripts/analyze_m2pc_log.py`. Lib y app en `-O2`.

---

### Notas del port (escritas antes de probar; siguen vigentes)

**Hito original**: el controlador existe en firmware: `lib/m2pc.{c,h}`, `lib/motor_params.h`, `apps/07_m2pc`.

### Fuentes contrastadas

- Paper (§III.B, Alg. 3, Tabla II): sector por posición del rotor (eq. 4), dos vectores adyacentes + nulo, Ts en ρ divisiones, CR(3,ρ) = (ρ+1)(ρ+2)/2 candidatos, modelo Euler (6), costo (3), modulación SVM.
- MATLAB: idéntico con ρ = 10 (66 candidatos); el sector sale del ángulo de la BEMF (= θ_e + 90° con forma sinusoidal), que coincide con la eq. 4 del paper.

### Decisiones del port (todas documentadas en `m2pc.h`)

| Tema | Simulación | Firmware | Por qué |
|---|---|---|---|
| Sector | por BEMF estimada; "arranque" con 8 vectores si ω < umbral | por θ_e del encoder, siempre | tenemos encoder: el sector existe a velocidad cero; se elimina la rama de arranque |
| Retardo | aplicación instantánea del promedio | **un periodo** (Cortés 2012): i(k+1) con el comando en curso, i(k+2) con el candidato | Se intentó medio periodo (CCR en el valle, ū = ½u(k−1) + ½u(k)): exigía paso < 10 µs y los 66 candidatos cuestan 9.5 µs desde flash. Con la UEV en el pico el presupuesto es 20 µs |
| BEMF | forma `[sin; −cos]` (convención de la planta MATLAB) | `λ·ω_e·[−sin θ_e; cos θ_e]` (flujo en d, BEMF en +q) | física del banco; coherente con el fasor medido en 06 |
| Unidades | SI, double | corriente Q8 LSB, tensión mV, θ Q32 | cero conversiones en el camino caliente |
| Candidatos | 66 evaluaciones con 2 multiplicaciones vectoriales cada una | incrementos por división **precalculados por sector** (independientes de la corriente); rejilla por restas: 10 instrucciones/candidato | ~4.5 µs estimados |
| Vbus | 240 V constante | 12 V constante (`VBUS_MV`) | TODO: medir (race N1.15) |

### Parámetros (todos en `motor_params.h`, con la derivación de cada constante Q)

R = 2.7 Ω (medido), L = 0.86 mH y λ = 3.5 mWb (**planning, sin identificar — Fase 2**), K = 32.93 raw/A, p = 7, Vbus = 12 V. λ = 3.5 mWb cuadra con el banco: 0.22 V de BEMF a 86 rpm, consistente con el adelanto de 15° de la tensión en 06.

### App `07_m2pc` (arranque seguro)

i_q* = 0 durante 1 s (regula corriente cero: el motor debe quedarse quieto), rampa a 200 mA en 1 s, luego constante. Disparo por sobrecorriente a 1.2 A (MOE = 0). Monitor a 100 Hz: i_dq, referencia, ω̂, sector, (n, m), costo; a 1 Hz: overruns y tiempo del paso.

### Código generado (`-O2`)

`m2pc_step`: 168 instrucciones, una llamada (`pwm_set_duties`), sin `sdiv`; bucle interno de 10 instrucciones con `smull/smlal` para el costo en 64 bits. `control_step`: 184 instrucciones.

### Para las sesiones siguientes

1. **Recalibración de offset de corriente en línea** (deriva de ~1.3 LSB medida). Opción barata: muestras del valle (V7, corriente cero por los shunts) con un segundo trigger, como pat92fr.
2. **Identificación a 600–900 rpm** (dead-time proporcionalmente menor) y log largo → dataset del ADALINE.
3. **ADALINE offline** en Python con la base de Fourier de `fcs_m2pc_v2/adaptation`, entrenado con e(θ_e) del log; después en línea sustituyendo la línea de la BEMF en `control_step`.
4. Afinar el PI de velocidad (ripple ±5 %) y medir rizo de par/corriente vs. el baseline sinusoidal — la comparación central de la tesis.
5. Opcional: paso del M²PC en CCM-SRAM para recuperar el modelo de medio periodo.

---

## Sesión 19 — 2026-10-14 — K calibrado con multímetro: −32.93 ± 0.47 raw/A

**Hito**: la cadena de corriente queda cerrada con un anclaje externo. Teoría del esquemático (34.0), empírico de junio corregido (33.9) y DMM (32.9) convergen al 3 %. De paso, dos bugs del modo de calibración que lo tenían inservible desde mayo.

### Resultado

Barrido DC de 8 puntos (duty_A 50 → 80 %, B/C al 50 %), DMM en serie con la fase A (entrada de corriente DC), promedio de 1024 muestras por punto:

| duty_A | 50 | 52 | 55 | 60 | 65 | 70 | 75 | 80 |
|---|---|---|---|---|---|---|---|---|
| DMM [A] | 0.001 | 0.030 | 0.074 | 0.147 | 0.221 | 0.301 | 0.375 | 0.456 |
| cal_avg [raw] | +0.25 | −1.16 | −2.23 | −4.36 | −7.13 | −9.77 | −12.33 | −14.79 |

Regresión: **K = −32.93 ± 0.47 raw/A**, intercepto +0.17 raw, R² = 0.99879, residuos < 0.35 raw.
Adoptado `ADC_FACTOR_RAW_TO_MA_Q12 = 124369` (30.36 mA/raw).

- **Signo negativo = física, no error.** Con i_A entrando al motor, en el instante de muestreo (V0) la inductancia mantiene la corriente hacia el motor: sube desde GND por el shunt A → V_shunt < 0. SimpleFOC lo codifica como ganancia −64/7. Clarke debe negar la lectura. Anotado en `adc.h` y N1.23.
- La corriente a 80 % fue 0.456 A y no los ~0.65 A teóricos: el DMM en serie añade ~2 Ω de burden. No afecta a K (el DMM mide lo que pasa por el shunt).
- Intercepto ≈ 0 → el offset calibrado al arranque (2545) es exacto.

### Dos bugs del modo de calibración (`CAL_TARGET_PHASE`), corregidos

1. **El openloop sobrescribía los duties DC.** `openloop_start()` corría antes del bloque de calibración y su paso en la ISR reescribía CCR1/2/3 cada 20 µs: los "puntos DC" eran la senoide de 10 Hz, de media cero → `cal_avg ≈ 0` en todos los puntos. Así que el modo nunca había funcionado (la calibración de junio fue con el barrido de amplitudes, no con este). Fix: `OPENLOOP_ENABLE` se deriva de `CAL_TARGET_PHASE`.
   Lo cazó un stream de diagnóstico (`DIAG_DC_STREAM`) que imprime `CCR1` y el encoder a 1 kHz: CCR1 barría 680…1019 y el rotor giraba cuando "debía" estar en DC.
2. **Promedio truncado a entero.** Con 12 raw a fondo de escala, 1 raw de truncamiento era 8 % de error en K (primera pasada: K = −38.7 ± 3). Ahora el promedio se imprime con 2 decimales; de ±3 a ±0.47.

### App `06_openloop_sync` — la cadena de sensado completa en la ISR, validada

Nuevo en `lib/`: `cordic.{c,h}` (sin/cos por hardware, 24 iteraciones, polling sin flag: leer
RDATA bloquea el bus hasta que el resultado está listo) y `as5600_nb_tick()` (el bombeo del
encoder de 04, promovido). El paso de control, llamado por hook desde la ISR de JEOS:

1. PLL: extrapolar, bombear el encoder, corregir si hay muestra.
2. θ_e = p·θ̂ − p·OFF: **una resta** de constante (el ×7, la latencia y el mod 2π ya vienen del PLL).
3. Open-loop: avanza θ_ol y escribe duties.
4. Clarke con el signo del shunt (`offset − raw`): 3 restas, 1 producto Q15.
5. Park con CORDIC: 4 productos Q31 (umull+mla en `-Og`).
6. Monitor: retardo θ_ol − θ_e y medias de i_d, i_q por ventana de 10 ms.

**Prueba definitiva**: en open-loop a velocidad constante, i_d e i_q deben ser DC; si el offset
o el signo están mal, oscilan a la frecuencia eléctrica. Resultado (10 Hz_e, amp 10 %, 25 s):

```
om  = +85.7 rpm        lag = θ_ol − θ_e = 105.1° ± 1.9°   (constante)
i_d = 305 ± 25 mA      i_q = 4 ± 16 mA                     (DC, sin carga no hay par)
paso_max = 4.74 µs     corr/s = 2000   i2c_err = 0
```

El 105° es físico: el generador usa `v_a = sin θ_ol`, así que el vector de tensión está en
θ_ol − 90°; la tensión adelanta al rotor **15°**. Cuadra con V = R·I + E: I sobre d (0.82 V) y
la BEMF sobre q (~0.39 V a 86 rpm) inclinan V ~25° respecto a d. Con 4.74 µs quedan ~15 µs
para M2PC + ADALINE (y la app está en `-Og`).

### Dos errores de signo que se cancelaban (destapados por 06)

La primera corrida de 06 dio `lag` barriendo a 20 Hz (θ_e y θ_ol girando en sentidos
opuestos), el rotor a **−86 rpm**… y sin embargo i_d/i_q casi DC. Solo es posible si el marco
de Park gira *con* la corriente real: el marco era correcto y lo que giraba al revés era el
campo. Causa: `openloop.c` ponía la fase B en θ+120° (B **adelanta** a A) = secuencia a-c-b,
negativa en la convención estándar de Clarke (B atrasa). Corregido a secuencia positiva
(b = −120° ≡ +171 de 256, c = −240° ≡ +85). Desde la sesión 12 el motor giraba "al revés" y
nadie podía notarlo sin un marco de referencia medido.

Y la prueba de sentido de `05` tenía dos fallos que se cancelaron: asumía "θ_ol crece ⇒ campo
positivo" (falso con la secuencia negativa) y el desenvuelto de dos lecturas forzaba el signo
positivo al reconstruir vueltas. Acertó +1 por casualidad. Reescrita: **integración de deltas**
leyendo el encoder cada 10 ms (≈12 cuentas por lectura, inequívoco) y un bug más al pasar
(`uint16_t` envuelve a 65536, no a 4096: ahora `& 0xFFF`). Corrida final: +6860 cuentas de
7025 esperados, ENCODER_SIGN = +1, alineación en 811 ≡ 2558 (mod 585, +8).

Constantes definitivas del control: `PHASE_A_OFFSET_RAW = 2558` (cualquier posición congruente
mod 585 es equivalente), `ENCODER_SIGN = +1`, secuencia positiva en el generador.

### Pendiente observado

- Glitch esporádico en la **fase C**: en un stream de 12 s, `ic` dio una muestra de **2** (debía ser ~2531). Una muestra corrupta, no ruido. Sospecha: lectura de `JDR2` en una ventana rara del dual injected, o el OPAMP3 por el canal interno 18. No bloquea (una muestra en 12 000), pero al cerrar lazo conviene un filtro de mediana o cazarlo con scope.
- `-Og` en toda la build: para el controlador conviene `-O2` por app (medir antes/después con el DWT).

---

## Sesión 18 — 2026-10-13 — La paradoja de corriente, explicada antes de volver al banco

**Hito**: la "paradoja de medición" de la sesión 12 queda **cerrada**: explicada en sus tres capas con esquemático, manual y literatura externa, y reproducida en banco. Dos de las capas ya estaban corregidas en el código desde el 2026-06-06 sin que la bitácora lo registrara. Bonus: la referencia Kelvin del PGA eliminó la anomalía de la fase C y reconcilió el K empírico con el teórico.

### Resultados de banco (Vbus 12 V, motor en J7, barrido 0 → 170 → 510, 30 s c/u)

| amp | PGA `00011` (original) — PICO / RMS raw a/b/c | PGA `01011` (Kelvin) — PICO / RMS raw a/b/c |
|---|---|---|
| 0 | 5/5/5 · **1/1/1** | 5/5/5 · **1/1/1** |
| 170 | 18/18/17 · **9/9/9** | 17/16/17 · **8/8/8** |
| 510 | 46/44/**53** · **28/26/27** | **38/38/40** · **23/23/22** |

- Offsets calibrados: 2540/2530/2524 (`00011`), 2545/2532/2529 (`01011`) — contra **2553** predicho por la red 22k/2.2k del esquemático. Vbus = 1444 raw = 12.0 V (sin el race de N1.15 esta vez); cae a 1380 con carga.
- **Escala lineal**: ×3 de amplitud → ×3 de RMS. Paradoja cerrada. amp=0 da RMS 1 = ruido de ±1 LSB: era lo que la sesión 12 veía en *todos* los tramos (trigger en el valle).
- **Fase C**: con `00011` el pico va 17 % alto (53 vs 46/44; en junio 30 %) con RMS balanceado. Con `01011` queda **38/38/40**: era rebote de tierra, que la referencia Kelvin a Vshunt− cancela.
- **Ganancia**: con Kelvin las lecturas bajan ~18 % en las tres fases (28→23 a amp=510). El `00011` sumaba la caída IR del plano, un error proporcional a la corriente que inflaba el K empírico de junio: **41.3 × 23/28 = 33.9 ≈ 34.0 teórico**. El 21 % de "tolerancia" era rebote de tierra.
- **Adoptado**: `OPAMP_PGA_KELVIN=1` (PGA_GAIN=01011) y `ADC_FACTOR_RAW_TO_MA_Q12 = 120422` (K = 34.0 raw/A, 29.4 mA/raw). Pendiente revalidar con DMM en modo Kelvin.

### Corrección de la bitácora

Las sesiones 13-17 dicen que la paradoja "sigue abierta" y que el barrido amp=0/170/510 "nunca se corrió". **Es falso.** El commit `c9bcaa6` ("avances 6jun", 2026-06-06) hizo tres cosas que solo quedaron en comentarios de `adc.c`/`adc.h`/`pwm.c`:

1. TRGO de `MMS=010` (UEV) a `MMS=111` (OC4REF) con `CCR4 = ARR−1`.
2. Sample time de 6.5 a 47.5 ciclos.
3. Barrido amp={170, 510} con multímetro true-RMS en serie con una fase: **(8 raw_rms ↔ 200 mA), (24 raw_rms ↔ 580 mA)** → escalan linealmente → medición sana → K empírico = 41.3 raw/A (`ADC_FACTOR_RAW_TO_MA_Q12 = 99123`).

Ese trabajo nunca tuvo entrada en esta bitácora. Lección anotada en N1.23 §4.

### Las tres capas (detalle completo en FIELD_NOTES N1.23)

| Capa | Causa | Evidencia | Estado |
|---|---|---|---|
| Señal plana (pico 5-6, RMS 1, no escala) | **Hipótesis A**: con `MMS=010` + `RCR=1` escrito antes de CEN, la UEV cae en el **underflow = valle**, donde conducen los high-side y los shunts low-side ven cero | RM0440 §29.3.5 texto literal; SimpleFOC tuvo el bug idéntico (PR #210); MCSDK y district9prawn usan OC4REF como nosotros | Corregido en código (jun); **por reproducir en banco** |
| K = 59.6 raw/A "no cuadra" (empírico 41.3) | La red de la placa **atenúa el shunt ×4/7** (1.5k serie, 22k a 3.3 V, 2.2k a GND) | Esquemático MB1419 hoja 5; ST/SimpleFOC/district9prawn usan ganancia total 9.14 = 16·4/7; K teórico corregido = **34 LSB/A**, con ±15 % de la red interna del PGA (DS12589) | Cerrado; empírico manda |
| Offset DC "misterioso" de 128 mV (Task #9) | La misma red polariza el nodo a **3.3·(1.5k∥2.2k)/(22k+1.5k∥2.2k) = 128.6 mV** → 2.057 V = 2553 raw con ×16 | Medido 128 mV (sesión 11); district9prawn mide 2552 | Cerrado |

Hipótesis B (sample time) descartada como causa de la señal plana, pero válida como corrección: DS12589 exige `TS_OPAMP_VOUT ≥ 200 ns` con salida interna; 6.5 ciclos @ 42.5 MHz = 153 ns la violaba. Hipótesis C (ruido) descartada: pico/RMS = 5 sobre 50 000 muestras es cuantización sobre una constante, no ruido físico.

### Hallazgos adicionales

- **Tabla de pines corregida** (N1.2): PA5 no es "Curr_fdbk3_OPAmp+", es la entrada inversora del OPAMP2 (VINM0 = Vshunt_2−). OPAMP3 usa PB0/PB2. El `[VERIFY]` de `adc.c` queda resuelto: VP_SEL=0 → PA1/PA7/PB0 es correcto.
- **Modo PGA**: usamos `PGA_GAIN=00011` (R1 a VSSA). ST/SimpleFOC/district9prawn usan `01011` (R1 al pin VINM0 = Vshunt−, referencia Kelvin). Misma ganancia; mejor rechazo de la caída IR del plano. Cambio de un bit, pendiente de probar.
- **Signo**: la convención del shunt (corriente saliendo del motor = lectura sube) es opuesta a la de control; fijarlo al integrar Clarke.
- **Ventana a 50 kHz**: MCSDK exige ≈ 1.75 µs de low-side ON antes del muestreo → duty máximo ≈ 82 % con tres shunts; relevante al cerrar lazo.
- `pat92fr/FOC2` calibra el offset continuamente con las muestras del valle (`RCR=0`, separa por `DIR`). Idea para la integración.
- Comentarios obsoletos corregidos en `adc.c` (decía standalone / 6.5 ciclos / JEXTSEL=01001) y en el README de la app (decía "regular simultaneous").

### Alineación θ_e ↔ fase A y sentido de giro — HECHO (app `05_phase_align`)

Refactor previo: `pwm/adc/openloop` promovidos a `lib/` (la app nueva los necesitaba;
criterio "2+ apps" del README). `adc.c` quedó desacoplado de openloop: la ISR de JEOS
llama al paso de control por hook (`adc_set_control_hook`).

Resultados (Vbus 12 V, límite 500 mA, dos corridas):

```
[align] raw alineado = 2558   →  PHASE_A_OFFSET_RAW = 2558
[align] raw alineado = 3733   →  (3733 − 2558) mod 585 = 5 cuentas
[align] PASO 2: delta = +5435 cuentas (θ del generador crece) → ENCODER_SIGN = +1
```

- El vector estático fase A (60 %/45 %/45 % ≈ 0.44 A) alineó el rotor en ~4 s. Con p = 7
  hay **7 posiciones de equilibrio por vuelta mecánica** (separadas 4096/7 ≈ 585 cuentas);
  ambas corridas cayeron en posiciones distintas pero congruentes módulo 585 — como debe
  ser, porque 7·585·2²⁰ = 2³² ≡ 0: cualquiera de las 7 es un θ_e = 0 válido. Dispersión
  de 5 cuentas (0.4° mec) = cuantización + ruido de lectura.
- Sentido: +1 (el raw crece con θ_e creciente). No hace falta invertir nada.
- Observación: a 2 Hz el rotor siguió el campo a ~1.55 Hz de media (delta 5435 de 7025
  esperados) — slip de arranque open-loop al 10 % de amplitud; esperado, no afecta al signo.
- Constantes finales del control:
  `θ_e [Q32] = ENCODER_SIGN · 7 · (uint32_t)(raw − PHASE_A_OFFSET_RAW) · 2²⁰`
  (equivale a `theta_pll` con offset: θ̂ − OFF·2²⁰, luego ×7).

### Para las sesiones siguientes (con fuente)

1. ~~Revalidar K con DMM~~ → sesión 19: −32.93 ± 0.47 raw/A.
2. ~~PLL + Clarke + Park en la ISR con el motor en open-loop~~ → sesión 19, app 06: i_dq DC,
   lag constante, 4.74 µs.
3. **Portar el FCS-M2PC** (y luego el ADALINE) de `fcs_m2pc_v2/` a C en punto fijo sobre el
   esqueleto de 06, y medir el presupuesto completo con el DWT. Es el riesgo técnico mayor.
4. Fase 2 (identificación): Rs, Ls, λ_pm y forma de la BEMF con el encoder ya alineado.
5. Al cerrar lazo: ventana de muestreo (duty ≤ 82 % a 50 kHz con tres shunts), el race de
   `adc_get_vbus_raw` (N1.15) y el glitch esporádico de la fase C.

---

## Sesión 17 — 2026-10-12 — Observador PLL tipo II validado en banco

**Hito**: θ̂ continuo a 50 kHz con error máximo de 0.27° mecánicos contra el encoder, ISR completa en 2.11 µs (10 % del presupuesto), cero errores I²C en 180 s de prueba con el rotor girado a mano. El sensing de posición queda cerrado.

### El problema que resuelve

La ISR de control corre a 50 kHz; el bus entrega muestra nueva cada ~468 µs (2 139 lecturas/s medidos en la sesión 16). En ~23 de cada 24 ciclos **no hay dato fresco**: sostener la última lectura mete θ a saltos al lazo y derivarla amplifica la cuantización a ~24 rad/s de ruido (N1.18).

### Implementado

- **`lib/theta_pll.{c,h}`** — observador PLL tipo II (≡ filtro α-β ≡ Kalman estacionario del modelo velocidad-constante), 100 % punto fijo Q32:
  - Cada tick (50 kHz): extrapolación `θ̂ += ω̂`.
  - Cada muestra nueva: corrección `e = wrap(θ_med − θ̂)`; `θ̂ += Kp·e`; `ω̂ += Ki·e`. El wrap se hace con resta unsigned + cast signed, sin ramas.
  - Ganancias: ωn = 2π·100 rad/s, ζ = 1 → Kp = 0.588, Ki = 3.695e-3 (constantes Q30 plegadas en compilación).
  - θ eléctrico = 7·θ̂_mec: el overflow del uint32 implementa el mod 2π eléctrico.
  - **Compensación de latencia proporcional a ω̂** (`θ̂_e + P·ω̂·30 ticks`, 600 µs): mejora sobre el "adelanto constante en cuentas" del plan de la sesión 13, que solo era exacto a una velocidad.
- **`apps/04_theta_pll`** — TIM2 a 50 kHz (prio 1, como la ISR de control final) bombea el I2C no bloqueante (prio 3) con el patrón start/take. Sonda de scope en PB6. Deadline del bus en ticks con abort desde ISR y reset solo desde foreground. Stream a 100 Hz para el plotter + estado a 1 Hz.
- **`python_scripts/plot_theta.py`** — plotter serie en tiempo real (pyserial + matplotlib), parsea tanto el formato de `03_as5600` como el stream `S t=... th=... om=...` de esta app; guarda CSV.

### Validación en banco (rotor a mano, movimiento arbitrario, 180 s)

```
corr/s=2000   i2c_err=0   isr_max=2.11 us
error θ̂ − raw:  mean +0.001°   max |e| = 0.27° mec
ω̂ limpio en reposo y en movimiento; 1 solo boot en toda la corrida
```

Gráfica: `firmware/apps/04_theta_pll/doc/validacion_pll_2026-10-12.png` (θ̂ y raw desenvuelto indistinguibles; el offset de 720° del análisis es artefacto de la referencia de desenvuelto, no error del lazo).

### Decisiones

- **PLL tipo II y no algo "más avanzado"**: la entrada es una rampa (θ a velocidad constante) y el tipo II es el observador mínimo con error estacionario cero ante rampa; el Kalman estacionario del mismo modelo es matemáticamente idéntico; EKF/SMO/Luenberger con modelo del motor importarían sensibilidad paramétrica sin beneficio (tenemos encoder). Tipo III queda como upgrade si el banco muestra error por aceleración. Discusión completa en FIELD_NOTES N1.22.
- **Pares de polos, cerrado de forma definitiva**: banco = 2804 12N14P = **7 pares**; simulación = Anaheim BLY-344S = **4 pares** (`p.P = 8` en MATLAB cuenta polos). El comentario de `openloop.h` que atribuía 4 pares al 2804 se corrigió. Tabla de referencia en PLANNING_HARDWARE §2.

### Para las sesiones siguientes

**Sin fuente (se puede avanzar ya):**

- **Presupuesto completo de ISR**: portar Clarke + M²PC + ADALINE a C en punto fijo y medir con DWT dentro de una ISR de 50 kHz con datos sintéticos. Hoy extrapolación+pump+corrección = 2.11 µs de 20 µs. Es el riesgo técnico más grande que queda: si no cabe, hay que rediseñar (CORDIC, Ts mayor) antes de seguir.

**Con fuente** (Vbus 12 V, límite de corriente 300–500 mA, motor en J7), en este orden:

1. **Cerrar la paradoja de medición de corriente (sesión 12, sigue ABIERTA).** El barrido amp = 0/170/510 ya está programado en `02_pwm_adc` (`DIAG_AMP_SWEEP_ENABLE=1`, 30 s por amplitud) y nunca se corrió. Si los picos escalan ~0/18/53 raw → medición sana; si son iguales → medición rota (hipótesis A: TRGO en el valle del PWM; B: OPAMP sin tiempo de asentamiento con SMP = 6.5 ciclos; C: ruido). **Bloqueante**: el FCS-MPC predice corrientes; sin medición confiable no hay control.
2. **Calibración de ganancia raw → A** con multímetro en serie (`CAL_TARGET_PHASE` 0/1/2, ya armado). Validar K teórico = 59.6 raw/A; si difiere > 10 %, calibrar fase por fase.
3. **Offset θ_e ↔ fase A y sentido de giro**: open-loop con θ_e fijo y corriente baja alinea el rotor al eje de la fase A; el raw del encoder en ese punto es el offset. Repetir en varias posiciones (7 alineaciones por vuelta mecánica) y promediar. Verificar además que θ_e del encoder **crece** con la secuencia de fases positiva; si decrece, invertir signo.
4. **PLL en movimiento real**: integrar `theta_pll` en la ISR JEOS de `02_pwm_adc`, girar en open-loop y comparar θ_e(PLL, con offset) contra θ_e(open-loop): deben avanzar juntos con un desfase constante (el ángulo de carga). Valida offset, sentido y PLL a velocidad sostenida, y mide la ISR real con ADC + open-loop + PLL.
5. **Fase 2 — identificación** (PLANNING §4): Rs por inyección DC, Ls por escalón, λ_pm y forma de la BEMF girando el motor (dato clave para la tesis: ¿cuán no-senoidal es la BEMF del 12N14P? decide si el ADALINE es "necesario" o "demostrativo").

---

## Sesión 16 — 2026-10-11 — Migración a PC local + validación en banco de la errata I²C

**Hito**: el workaround de silicio de la sesión 15 queda **validado en banco**: primera lectura IRQ limpia, secuencia nominal sin `TXIS`, 2139 lecturas/s con cero errores. El toolchain migra de la Pi a la PC local. El caos del día fue un cable micro-USB muriendo, no firmware.

### Migración Pi → PC local

El setup Mac→rsync→Pi existía porque la PC del trabajo bloqueaba USB. Con la PC personal se trabaja directo. Lo que hizo falta:

- `openocd` 0.12, `arm-none-eabi-gcc` 13.2.1, `cmake` — **ya instalados**.
- Regla udev `49-stlinkv2.rules` (0483:3748/374b, modo 0666) — **ya existía**; usuario en grupo `dialout`.
- Clone de `STM32CubeG4` en `tesis_maestria/vendor/STM32CubeG4` (depth 1, submódulo `cmsis_device_g4`) — era lo único que faltaba; el path lo espera `firmware/CMakeLists.txt` como `../vendor/STM32CubeG4`.
- Build local OK: los tres targets compilan; solo los stubs `nosys` esperados.
- Flash con `reset_config srst_only connect_assert_srst`: sin él el primer intento dio "examination failed / Unable to reset target" (posiblemente agravado por el cable; el connect-under-reset queda como default del banco).
- El binario local difiere en tamaño del documentado (18 040 B vs 15 644 B de `as5600.elf`): es GCC 13.2.1 vs 15.2.1, no un cambio de código.

### La saga del cable USB (3 síntomas, 1 causa)

Síntomas aparentemente independientes durante la sesión:

1. Error `-71` (EPROTO) al enumerar, en tres puertos distintos; un puerto (`usb 1-1`) sí funcionaba.
2. **Boot-loop del target**: 159 banners `[boot]` en 30 s, tramas UART cortadas a mitad de línea. Sin watchdog ni `SystemReset` en el firmware: resets externos.
3. Disconnects USB espontáneos, cada vez más frecuentes.

Causa única: **VBUS marginal por cable micro-USB degradado**. Si VBUS cae: el ST-LINK pierde energía (USB disconnect), el regulador 3.3V del target colapsa (brownout → boot-loop), y la señal de datos se degrada (EPROTO). El bisect fue: mismo banco, solo cambio de cable → un solo boot, secuencia completa, USB estable.

Lección completa en FIELD_NOTES N1.21.

### Validación en banco del workaround de la errata (pendientes de sesión 15, CERRADOS)

```
[scan]   ACK en 0x36   <-- AS5600
[as5600] STATUS=0x20  MD=1 ML=0 MH=0
[shot] 0: OK  raw=4089
[shot] IRQ: TXIS=0 TC=1 RXNE=1 STOPF=1 | entradas=4
[perf] 2139 lecturas/s  (teorico ~2200 a 100kHz)
[perf] no-arrancadas=0
[perf] errores=0  abortadas=0  ultimo raw=4089
```

- ✅ `[shot] 0: OK` — primera lectura IRQ aislada, camino de 4 eventos nominal, **sin `TXIS`**.
- ✅ Throughput: **2139 lecturas/s** = 97 % del teórico a 100 kHz, `errores=0`, `abortadas=0`. El bus limita frente al refresco interno de ~7 kHz del AS5600, como se predijo.
- ✅ Giro manual del rotor: `raw` recorre 0..4095 monótono con un solo salto por vuelta (verificado por el usuario en stream en vivo).
- ✅ Bonus: 9 306 NACKs consecutivos sin sensor conectado dejaron el bus impecable — la recuperación de error de la sesión 14 también quedó ejercida.

### Para la sesión siguiente

El plan de la sesión 13 sigue vigente, ya sin el paso 1:

1. **Observador PLL tipo II** para θ/ω en punto fijo (θ como acumulador Q32, coherente con `OPENLOOP_DELTA_Q32`). La ISR corre a 50 kHz y el bus entrega muestra nueva a ~2.1 kHz: la mayoría de los ciclos no hay dato fresco y hay que extrapolar. El PLL filtra el ruido de cuantificación y da la extrapolación gratis.
2. **App `04_theta_pll`** a 50 kHz por timer, sin PWM/ADC/motor, validada girando el rotor a mano.
3. **Compensación de latencia**: ~450 µs de transacción + ~150 µs de refresco ≈ 19° eléctricos a 80 rad/s mecánicos con P=7 — adelanto constante en cuentas.
4. Luego: offset θ_e ↔ fase A (necesita Vbus y motor) y presupuesto completo de ISR.

---

## Sesión 15 — 2026-08-24 — Workaround oficial para bloqueo I²C por IRQ

**Hito**: identificada causa de silicio y corregida la secuencia polling/IRQ. Compila; pendiente validar en banco.

### Causa raíz

ES0431 Rev. 9 §2.11.5, `I2C2-190208`, documenta **“Transmission stalled after first byte”**. Su workaround exige precargar el primer dato en `TXDR` antes de generar `START`. La secuencia anterior hacía lo contrario: `START`, esperar `TXIS`, escribir registro. `vendor/STM32CubeG4/Release_Notes.html` confirma que ST incorporó exactamente esta precarga al HAL.

### Cambios

- `i2c1_read_regs()` e `i2c1_read_regs_start()` cargan el registro en `TXDR` antes de `CR2.START`.
- Camino IRQ normal queda en 4 eventos para dos bytes: `TC`, `RXNE`, `RXNE`, `STOPF`. Un `TXIS` inesperado aborta; nunca reenvía el registro.
- Fuentes IRQ se habilitan por transferencia y se deshabilitan antes de publicar estado terminal.
- Abort desde ISR es asíncrono: pide `STOP`, drena `RXDR` y espera `STOPF`; no contiene polling ni reset de `PE`.
- `closing` y secciones críticas cortas eliminan carreras con ISR de control de mayor prioridad.
- Flags de bus se capturan antes de limpiarlos; `seq/errors` y `DONE/ERR` se publican juntos al final.
- Timeout de `nb_pump()` conserva su timestamp entre llamadas y fuerza recuperación desde foreground si `STOPF` no llega.
- Prueba de arranque reducida a una primera lectura IRQ aislada antes del throughput.

### Verificación

- Build externo completo: `blink.elf`, `pwm_adc.elf`, `as5600.elf`; sin warnings C. Solo stubs `nosys` esperados.
- `objdump` confirma store a `TXDR` antes de habilitar IRQ y antes del store de `CR2.START`.
- `as5600.elf`: 15 644 bytes totales (`text=13 524`, `data=92`, `bss=2 028`).

### Pendiente

1. Flashear `03_as5600`.
2. Confirmar `[shot] 0: OK` y secuencia normal sin `TXIS`.
3. Registrar `[perf] N lecturas/s`, `errores=0`, `abortadas=0`.

---

## Sesión 14 — 2026-08-23 — El camino por IRQ, depurado: un fallo que se multiplicaba

**Hito**: encontrados y corregidos los tres defectos del driver no bloqueante de la sesión 13. Añadida recuperación de bus. Diagrama de conexiones del banco documentado. Compila limpio; pendiente de flashear.

### Lo que devolvió el banco

```
[perf] 0 lecturas/s      errores=5195  abortadas=0
[loop] error #1..#100: BUSY (bus ocupado al arrancar)
```

Y, en una corrida, el bus clavado: `SCL(PB8)=HIGH  SDA(PB7)=LOW`, 112 direcciones en TIMEOUT.

Todo eso es **una** causa reproduciéndose, no un desastre con muchas. Detalle completo en
FIELD_NOTES [N1.19](./FIELD_NOTES.md).

### Los tres defectos

| # | Defecto | Consecuencia observada |
|---|---|---|
| 1 | `nb_finish()` no dejaba el periférico usable tras un error: `BUSY=1`, `CR2` a medias, un byte vivo en `TXDR` | Los cien `error: BUSY` (un fallo + 99 negativas a arrancar) **y** las trazas con `TXIS=0 bytes=1/2`, que eran el byte del viaje anterior saliendo solo |
| 2 | Las ramas `RXNE`/`STOPF` del handler no comprobaban `state == BUSY` | Un `RXNE` tardío escribía en el buffer de una transferencia cerrada. Es lo que hacía aparecer `RXNE=1` junto a `bytes=0/2` |
| 3 | `phase/addr7/reg/n/idx/buf` sin `volatile` en una estructura cuyos otros campos sí lo eran | Latente: el compilador puede mover los stores del contexto respecto al `START` |

Arreglo del 1, apoyado en RM0440 Rev 9 §39.4.6: STOP con guardia corta (~300 µs) y luego
ciclo `PE=0` → releer → `PE=1`. Eso borra `CR2`, borra `ISR` con `TXE=1` (vacía `TXDR`) y
baja `BUSY`, **sin** tocar `TIMINGR` ni `CR1` — los `xxIE` sobreviven. Más
`NVIC_ClearPendingIRQ` de ambos vectores.

### Añadido: recuperación de bus en `i2c1_init()`

`SCL=HIGH` con `SDA=LOW` y el maestro en reposo es un **esclavo colgado a mitad de byte**,
no un cable suelto. Lo provoca reiniciar el maestro (flash, `reset` de openocd) en medio de
una transacción: el STM32 arranca de cero, el AS5600 no. Con el defecto 1 fallando 5 200
veces por segundo, la probabilidad de que un flash cayera a mitad de transacción era ~1.

Cura estándar: hasta 9 pulsos de SCL bit-bang con SDA liberada + STOP manual. Va antes de
poner los pines en AF4 —el periférico no puede arrancar sobre un bus que no ve libre, y en
AF4 el `BSRR` ya no llega a la línea— y solo actúa si SDA se lee en bajo.

### Otros cambios

- `CR2` en una sola escritura con `START` incluido (lo que hace el HAL de ST), en vez de dos.
- `nb.phase = NB_PH_RX` **antes** del write: el primer `RXNE` puede entrar en la instrucción siguiente.
- `nb_pump()` ya no descarta el status del arranque: contador `no-arrancadas` aparte del de `errores`. «No arrancó» y «arrancó y falló» tienen causas distintas.
- Corregido el número de eventos por lectura: son **5** IRQ (TXIS, TC, RXNE×2, STOPF), no 4. La entrada de la sesión 13 dice 4.

### Documentación

- **FIELD_NOTES N1.19** — la nota sobre por qué un error de periférico tiene dos mitades (informar y limpiar), y sobre el sesgo de hipótesis que costó la sesión.
- **Diagrama de conexiones del banco** en la sección del I²C1: J8 pad por pad, la red de 1.8 kΩ + 10 kΩ, y la tabla de cuatro filas que traduce los niveles de reposo de SCL/SDA a un diagnóstico único.
- Anotadas las dos líneas obsoletas de la Sesión 2 que todavía decían «PB8 sin conectar, reservado para instrumentación».

### Pendiente al cerrar

1. **Flashear y leer el resultado.** Si el camino por IRQ ya completa lecturas, el número que interesa es `[perf] N lecturas/s` (teórico ~2200 a 100 kHz contra ~7 kHz de refresco del AS5600).
2. Si sigue fallando, la traza ordenada de IRQ por transferencia ya está instrumentada y ahora sale **sin la cascada** encima, así que muestra la primera causa real.
3. Alternativa más simple que sigue sobre la mesa: polling desde el lazo principal (código ya validado) + extrapolación de θ en la ISR. La lectura bloqueante bloquea el lazo, **no** la ISR, que la preempta sin problema.

---

## Sesión 13 — 2026-08-18 — I²C no bloqueante por IRQ (el encoder ya cabe en el lazo)

**Hito**: la lectura del AS5600 deja de bloquear. Máquina de estados por interrupción, 4 eventos por lectura, <1 % de CPU. Con esto el sensado de posición ya es compatible con la ISR de 50 kHz. Sin flashear todavía.

### Verificado en el manual ANTES de escribir código

- **RM0440 Rev 9 §39.4.7 / §39.4.9**: el maestro **estira SCL** en los tres puntos donde la máquina de estados tiene que actuar (TXE, RXNE, TC con AUTOEND=0). Esto es lo que autoriza correr el I²C en **prioridad NVIC 3** por debajo del control (prio 1): que el control preempte solo alarga la transacción, no pierde ni corrompe un byte. Sin esta garantía todo el diseño de prioridades estaría al revés.
- **Los capítulos del RM0440 se renumeran entre revisiones.** En la Rev 9 (la de `papers/`) el I²C es **§39** y el GPIO es **§9**; los comentarios decían §40 y §11, que apuntan a otra cosa. Corregido, y desde ahora las citas llevan la revisión explícita.
- **Vectores confirmados en el binario**, no asumidos: offset `0x00BC` = `I2C1_EV_IRQHandler`, `0x00C0` = `I2C1_ER_IRQHandler`.

### Implementado

- **`lib/i2c.{c,h}` y `lib/as5600.{c,h}`** — promovidos desde `apps/03_as5600/src/`. El lazo de control los necesita y duplicarlos garantizaba divergencia. `swi2c` se queda en la app: es red de seguridad de bring-up, no código de producción.
- **Camino no bloqueante** en `i2c.c`: `i2c1_nb_init()`, `i2c1_read_regs_start()`, `i2c1_nb_take()`, `i2c1_nb_abort()`, contadores `seq`/`errors` y foto del ISR en fallo de bus. Estados IDLE/BUSY/DONE/ERR, cierre en un único punto (`nb_finish`, solo desde BUSY) para que un STOPF tardío no cuente dos veces.
- **Sin timeout propio, a propósito**: el driver no tiene reloj. El plazo lo pone el consumidor, que sabe a qué ritmo corre.
- **`as5600_raw_angle_start()` / `_take()`** — el desempaquetado de 12 bits queda en un solo sitio para los dos caminos.
- **`i2c1_reset_peripheral()` corregido**: la secuencia de ST (§39.4.6) es `PE=0` → **releer PE** → `PE=1`, no un retardo. El bucle de 100 iteraciones `volatile` que había funcionaba por accidente.
- **Colisión de pines resuelta**: `adc.c` usaba **PB8** como sonda de scope de la ISR, y PB8 es I2C1_SCL. Un `MODER=01` push-pull ahí habría hundido el bus al fusionar encoder y control, en silencio. Sonda movida a **PB6**, libre justo porque su AF4 está vacía pero como GPIO puro va perfecto.
- **`03_as5600/main.c`**: modo de banco `USE_NONBLOCKING` que ejerce el patrón start/poll/take, más `THROUGHPUT_TEST` que mide lecturas/s reales durante 1 s.
- **`FIELD_NOTES.md` N1.18** — nota pedagógica completa, incluido el mapa de pines actualizado.

### Estado

Compila limpio en el Mac (`arm-none-eabi-gcc` 15.2.1, `-Wall -Wextra`, sin warnings). **No flasheado.**

### Para la sesión siguiente

1. **Flashear `03_as5600` y leer `[perf] N lecturas/s`.** Es la medición que decide si hace falta Fast Mode: teórico ~2200 a 100 kHz contra ~7 kHz de refresco interno del AS5600. Si el bus limita y queremos subir a 400 kHz, hay que mirar con scope los tiempos de subida sobre la red de J8 (1.8 kΩ serie + 10 kΩ pull-up, pensada para Hall push-pull) antes de dar por bueno el cambio.
2. **Observador PLL tipo II** para θ/ω en punto fijo, θ como acumulador Q32 de una vuelta eléctrica (coherente con `OPENLOOP_DELTA_Q32`). Filtra el ruido de cuantificación (~24 rad/s si se deriva θ a pelo) y da la extrapolación gratis.
3. **App `04_theta_pll`** a 50 kHz por timer, sin PWM/ADC/motor, validada girando el rotor a mano: θ̂ continuo entre muestras y ω̂ limpio.
4. **Compensación de latencia** como adelanto constante en cuentas: ~450 µs de transacción + ~150 µs de refresco son ~19° eléctricos a 80 rad/s mecánicos con P=7. Sin corregir, el ADALINE aprendería una BEMF rotada — indistinguible de un offset de fase A mal puesto.
5. **Offset θ_e ↔ fase A** (necesita Vbus y motor) y **presupuesto completo de ISR** (Clarke + M²PC + LMS juntos) siguen fuera de este bloque.

---

## Sesión 10 — 2026-05-22/23 — ISR JEOS funcionando (Semana 6 abre)

**Hito**: lazo de control a 50 kHz montado. ISR JEOS entra cada 20 μs, lee i_a/i_b/i_c en globals, toca PB8 para instrumentación (la sonda se movió a **PB6** en la sesión 13: PB8 es I2C1_SCL). Cierre del primer paso del plan de Semana 6.

### Implementado

- `apps/02_pwm_adc/src/adc.h` — declara globals `g_ia_raw / g_ib_raw / g_ic_raw / g_isr_count` (volatile) y la API `adc_isr_init()`.
- `apps/02_pwm_adc/src/adc.c` — `adc_isr_init()` habilita JEOSIE en ADC2, configura PB8 como output (hoy PB6, ver sesión 13), registra el handler con prioridad 1 en NVIC. `ADC1_2_IRQHandler()` togglea la sonda, lee los 3 JDR, incrementa counter, limpia flag.
- `apps/02_pwm_adc/src/main.c` — llama `adc_isr_init()` entre `adc_init()` y `pwm_enable()`. El while reporta `isr_count` y Δ por segundo.
- `FIELD_NOTES.md` N1.14 — nota pedagógica: panorama, analogía (médico de guardia con busca), detalle de JEOS, NVIC, latencia/jitter, reglas de oro para handlers, decisión "JEOSIE en ADC2 no ADC1" justificada.

### Validación

- VCP estable: `Δ=50350` ISRs entre prints separados por 1.007 s → **50000 Hz exactos**.
- `ADC2.IER=0x40` (bit 6 JEOSIE) post-init ✓.
- Lecturas raw consistentes con sesión 9: `i_a/i_b/i_c ≈ 317-319`, `Vbus ≈ 552`.
- Multímetro DC en pad Z+/H3 del J8: **44 mV** → consistente con pulso de ~270 ns cada 20 μs (duty ~1.35%, coherente con un handler de ~45 ciclos del Cortex-M4 a 170 MHz). **Evidencia indirecta, no scope explícito**.
- Calibración de offset implementada (state machine en handler, N=4096, división por shift right 12). Offsets resultantes: `i_a=317, i_b=317, i_c=319` (diferencia max 2 raw → OPAMPs bien trimmed de fábrica).
- Lecturas compensadas (`cal`) estables en ±1 raw → dentro del ruido de cuantificación del ADC.
- **Sanity check con motor conectado**: consumo Vbus estable en **80 mA** (baseline antes y después del motor conectado, idéntico). Motor quieto, sin vibración, lecturas `cal` siguen en ±1.
- **Baseline de consumo Vbus a anotar** (post-fix del bug AF de sesión 8): **80 mA** con todo configurado (TIM1 corriendo + ADC + ISR + motor conectado, duty 50% balanceado). Cualquier desviación significativa en sesiones futuras es señal de problema.

### Drama del scope: bug que no era

Pasamos buena parte de la sesión persiguiendo "PB8 no se ve en scope". Razón real: pulso de 270 ns es demasiado angosto para los settings que probamos primero (5+ μs/div). El multímetro DC en 44 mV fue la pista que cerró el caso. Lección persistida en N1.14 sección "Sospechas pendientes" — si en algún momento futuro el control no converge, volver a validar visualmente con scope a 1-2 μs/div + auto trigger.

### Limpieza de código

Quitado el bloque temporal `[diag PB8]` del `main.c` (dump de GPIOB.MODER/OTYPER/AFR/PUPDR/ODR + blink lento). El código vuelve a su forma "limpia": init → enable → while loop con prints.

### Para sesión 11 — Semana 6 sigue

Plan original mantenido:

1. **Calibración de offset DC** — 1000 muestras motor off, promediar, guardar `i_offset_a/b/c`. Esto convierte el ~318 raw en "i = 0 sin corriente". Lo más simple: hacer la calibración dentro del while al boot, antes de habilitar el control.
2. **Calibración de ganancia raw → A** — inyectar corriente DC conocida con fuente bench → leer raw → escala.
3. **Medir tiempo de ISR con GPIO toggle + scope**. Pendiente del scope explícito de PB8.
4. **Decisión OPAMP topology** (PGA + offset SW vs standalone con bias del PCB) — la calibración da los datos para decidir.
5. **Dead-time empírico** (pendiente de sesión 8).

### Sesión 11 — 2026-05-23 — Ganancia teórica + PGA x16

**Hito**: cadena `raw → mA` lista con K teórico. Listo para excitación open-loop / test de motor.

**Decisiones**:

- **R_shunt confirmado**: 0.003 Ω (R54, R55, R56 según esquemático MB1419 página "SHUNT RESISTOR", 3 W). Los JP1/2/3 son jumpers de bypass, dejados abiertos = shunts activos.
- **PGA cambiado x2 → x16** en `adc.c` (PGGAIN=00011). Razón: con x2, sensibilidad era 7.45 raw/A — el sweep de 0-1 A daba apenas 7 raw, dominado por ruido. Con x16, 59.6 raw/A → 60 raw/A en 1 A, SNR razonable.
- **No subimos a x32**: el offset DC del front-end (318 raw con x2, ≈ 256 mV en VINP_equiv) escala con la ganancia. x32 lo llevaría a ~5000 raw → SATURA el ADC. x16 lo deja en ~2540 raw, queda headroom +26 A.
- **Constante K hardcoded en adc.h**: `ADC_FACTOR_RAW_TO_MA_Q12 = 68735` (= round(16.78 × 4096)). Inline `adc_raw_to_ma(int16_t) → int32_t` hace `(raw × FACTOR) >> 12`. 100% entero, ~3 ciclos por conversión.
- **Sweep de calibración física DEJADO en main.c, activable cambiando `#define CAL_TARGET_PHASE`** entre -1 (off, normal), 0/1/2 (calibrar A/B/C). Default = -1.

**Pendientes que registro explícitamente**:

1. **Validar K teórico vs medido** — el cálculo tiene ~5-8% de error teórico (tolerancias de R_shunt, G_PGA, V_ref). Para tesis riguroso: hacer al menos un sweep físico con multímetro en serie en una fase y validar. Si difiere < 10%, confiar en teórico para las 3 fases. Si difiere más, calibrar cada una por separado.
2. **Origen del offset DC alto (~256 mV con PGA x2)** — no esperado para input flotante / shunt sin corriente. Hipótesis: pin VINP del OPAMP conectado a un bias network del PCB no documentado en UM2516. Sin investigar todavía. Si el control no converge bien, esto puede ser un factor.
3. **Limitación bipolar conocida** — el offset alto restringe rango negativo. Corriente max negativa ≈ -42 A (no problema para nosotros), positiva ≈ +26 A (tampoco). Pero la asimetría puede afectar zero-crossing de corrientes AC. Para FCS-M2PC eventualmente: investigar bias correcto del PCB, o switch a standalone mode con bias a Vref/2.

### Para sesión 12 — qué sigue

Con `raw → mA` listo, las opciones son:

1. **Excitación open-loop básica** — generar 3 sinusoides desfasadas 120° en el handler, baja amplitud (5-10% duty), frecuencia eléctrica baja (e.g. 2 Hz). Confirmar que el motor gira. **Es el primer test que mueve el motor.**
2. **Validación de K con sweep físico** (multímetro en serie) — ya quedó armado el `CAL_TARGET_PHASE`.
3. **Medir tiempo de ISR con scope** (con settings agresivas, 1-2 μs/día).
4. **Dead-time empírico**.

El usuario quiere "conectar el motor" — opción 1 es el camino natural.

### Sesión 12 — 2026-05-23 — Motor gira por primera vez + paradoja de medición

**Hito**: el motor giró por primera vez 🎉. Datos confirman P=7 (no P=8 como decía CLAUDE.md). Pero quedó abierta una paradoja: las mediciones de corriente no parecen reflejar la corriente real del motor — siguiente sesión es debug.

#### Implementado

- **Módulo `openloop.{h,c}`**: excitación trifásica sinusoidal en open-loop. LUT Q15 de 256 entradas (generada offline con Python para no depender de libm). Theta Q32 con overflow natural = vuelta eléctrica. Desfase 120° por offset entero en índice (85, 170). Aritmética 100% entera.
- **Hook en handler**: `openloop_step()` llamado desde `ADC1_2_IRQHandler` después de leer JDR. Coste extra ~80-100 ciclos cuando running, ~3 ciclos cuando no.
- **Stats pico+RMS**: acumuladores `max(|i|)` y `Σi²` en uint64_t dentro del handler. Snapshot a globals cada N=50000 muestras (1 segundo). RMS calculado en main con `isqrt32` (entera, sin float). Coste extra del handler ~25 ciclos.
- **`adc_isqrt32`**: sqrt entera digit-by-digit base 4, ~20 ciclos M4.

#### Validaciones y descubrimientos

1. **Motor gira** a f_e=2 Hz: 3.5 s/vuelta → confirma f_m = 0.286 Hz → **P=7 (14 polos, 7 pares)**, NO P=8 como decía CLAUDE.md. A f_e=10 Hz: 0.65 s/vuelta → f_m = 1.54 Hz (predicho 1.43 Hz, 7% error del cronómetro).
2. **CLAUDE.md actualizado** con sección "Motores en el proyecto" — distinción Anaheim (paper, P=8, trap) vs 2804 (banco, P=14, sin). Memoria del banco también actualizada.
3. **R_fase medida = 2.7 Ω** (R_línea_línea = 5.4 Ω, Y-connection). Confirmó interpretación del datasheet x-teamrc.
4. **AS5600** confirmado por el usuario (datasheet x-teamrc menciona AS5048A pero el motor real tiene AS5600).
5. **Bug latente en `adc_get_vbus_raw`**: race condition entre conversiones regular e inyectada del mismo ADC. Sesión 11 mostraba Vbus_raw=552 (no físico, 4.6V); sesión 12 muestra Vbus_raw=1443 (= 11.6V real con divider 0.0963). El handler más lento "movió" el timing accidentalmente. Documentado en FIELD_NOTES N1.15 + Task #13.

#### Paradoja abierta — mediciones de corriente NO siguen al motor

Con `amp=170` (10% PWM) y motor girando:
- A 2 Hz: pico_raw=5-6, RMS_raw=1, consumo Vbus=127 mA.
- A 10 Hz: pico_raw=5-6, RMS_raw=1, consumo Vbus=123 mA.

**Mismas magnitudes a 2 y 10 Hz** — debería haber cambiado si midiéramos corriente real del motor. Si fuera sinusoide pura, Pico/RMS = √2; medimos Pico/RMS ≈ 5. **Lo que estamos midiendo NO refleja la corriente del motor.**

Hipótesis a evaluar en sesión 13 (en orden de probabilidad):
- **(A)** TRGO dispara en valle del PWM (high-side ON, shunts ven 0) en lugar del pico. RM0440 §29.4.5 no es 100% explícito sobre dónde cae el UEV con RCR=1 en center-aligned.
- **(B)** OPAMP con PGA x16 + sample time 6.5 ciclos (153 ns) no se estabiliza (tau OPAMP ≈ 200 ns con BW 0.8 MHz). Solución: subir SMP a 47.5 ciclos.
- **(C)** Ruido térmico/drift del front-end dominando sobre señal real (que es pequeña porque la corriente del motor no es tan grande).

#### Test definitivo propuesto (no ejecutado, para sesión 13)

Variar `OPENLOOP_AMP_TICKS` entre {0, 170, 510} y medir Pico/RMS de cada caso:
- amp=0: si pico>0, hay un baseline de ruido/drift.
- amp=170 (10%): pico esperado ~18 raw si medición OK.
- amp=510 (30%): pico esperado ~53 raw si medición OK.

Si las stats son **idénticas** entre los 3 casos → medición rota.
Si escalan proporcionalmente → medición OK pero corriente real es baja (problema distinto).

#### Estado del firmware al cierre

- `apps/02_pwm_adc/src/main.c`: OPENLOOP_ENABLE=1, AMP=170, DELTA=10 Hz_e.
- `apps/02_pwm_adc/src/openloop.{c,h}`: LUT + step + start/stop.
- `apps/02_pwm_adc/src/adc.{c,h}`: ISR + cal offset + stats + isqrt32.
- Sweep de ganancia armado pero desactivado (`CAL_TARGET_PHASE=-1`).

#### Tasks abiertas al cierre de sesión 12

- Task #8: validar K teórico con multímetro físico (no urgente).
- Task #9: investigar origen del offset DC alto (256 mV con PGA x2 = anómalo).
- Task #13: refactor adc_get_vbus_raw (race condition).
- Nuevas implícitas para sesión 13:
  - Test amp=0/170/510 para discriminar medición rota vs corriente real baja.
  - Si medición rota: probar SMP=47.5 ciclos primero (cambio menor); si persiste, cambiar MMS para forzar TRGO al pico.

### Para sesión 13 — punto de partida concreto

**Primera acción al sentarse**: implementar el test de 3 amps (0, 170, 510) con prints separados por amp, idealmente en un mismo flash con cambio automático cada 30 s. Eso descarta o confirma la hipótesis de medición rota en ~2 minutos.

Si medición rota:
1. Subir SMP del ADC de 6.5 a 47.5 ciclos (cambio de 1 línea en `adc.c`, paso 7 del adc_init). Re-test.
2. Si sigue mal: cambiar MMS a OC4REF con CCR4=ARR-1 para forzar TRGO explícitamente en el pico. Re-test.
3. Si sigue mal: instrumentar con scope. Mirar OUT1 vs PB8 (toggle del ISR) para inferir cuándo se hace el sampling.

Si medición OK pero corriente baja:
1. Calcular potencia disipada — debe cuadrar con consumo Vbus.
2. Considerar que la corriente real del motor outrunner pequeño con poca fricción es genuinamente baja.

---

## Sesión 1 — 2026-05-16

**Hito**: Definición del planning y descarga de bibliografía bloqueante.

**Acciones**:
- Creado `PLANNING_HARDWARE.md` con 6 fases, ~290 h totales, 2 h/día → ~8 meses.
- Aclarado rol del ADALINE: estima **forma de BEMF** (21 coef. Fourier), NO parámetros eléctricos.
- Descargados a `papers/`:
  - UM2516 (manual de la placa).
  - RM0440 (Reference Manual STM32G4).
  - DS12589 / `STM32G431X6.PDF` (datasheet del chip).
  - `MB1419-G431CB-B01_schematic.pdf` (esquemático oficial).
  - AN5325 (CORDIC en STM32).
  - AN5305 (FMAC — opcional, opción de optimización para Fase 6).
  - Kouro 2009 — paper canónico de FCS-MPC.

**Pendiente al cerrar**: confirmar pinout de J8, asignación de fases, llegada de la placa.

---

## Sesión 2 — 2026-05-17

**Hito**: Placa recibida, hardware completamente cableado, toolchain de Pi confirmado.

### Verificaciones de documentación
- **J8 (sensor) — pinout confirmado por UM2516 Tabla 4 + DS12589 Tabla 13 (Alternate functions)**:

  | Pad J8 | Pin STM32 | Función AF4 |
  |---|---|---|
  | A+/H1 | PB6 | ~~I2C1_SCL~~ **libre** — AF4 aquí no es I2C1_SCL (2026-08-10) |
  | B+/H2 | PB7 | **I2C1_SDA** |
  | Z+/H3 | PB8 | **I2C1_SCL** (corregido 2026-08-10; en su día se anotó como «libre, reservado instrumentación») |
  | 5V | — | alimentación |
  | GND | — | tierra |

- **J7 (motor)**: 3 terminales OUT1/OUT2/OUT3 (mismos pads accesibles desde top y bottom de la placa).

### Cableado físico (decisiones)

- **Fases del motor → J7**:
  - Blanco → OUT1 (fase A lógica)
  - Rojo → OUT2 (fase B lógica)
  - Negro → OUT3 (fase C lógica)
- **AS5600 (integrado al motor) → J8**:
  - VCC → 5V (confirmado por usuario: módulo acepta 5V)
  - GND → GND
  - SDA → B+/H2 (PB7)
  - SCL → **Z+/H3 (PB8)** — corregido 2026-08-10, ver FIELD_NOTES N1.16
  - ~~PB8 (Z+/H3) sin conectar — reservado para instrumentación con osciloscopio.~~ **Obsoleto desde 2026-08-10**: PB8 ES el SCL. El pin de sonda pasó a PB6 (`A+/H1`), que queda libre porque su AF4 está sin asignar.
- **J5/J6 (Vbus)**: NO conectado todavía.

### Hardware adicional confirmado

- **Fuente DC**: buck-boost ZK-4KX alimentada desde fuente de PC vieja, con modos CV/CC. Apta para Vbus 5–15 V.
- **Osciloscopio**: disponible.

### Workflow de desarrollo (decisión)

- Mac corporativa con JumpCloud → bloquea conexión USB de dispositivos.
- **Solución**: Raspberry Pi 4B 8GB + Ubuntu 22.04 LTS aarch64, conectada al ST-LINK por USB. Mac edita por Zed Remote SSH.
- Bonus: la Pi queda en el banco junto a osciloscopio + fuente + motor + placa. La Mac queda en el escritorio.

### Toolchain en la Pi — verificado

| Herramienta | Versión |
|---|---|
| arm-none-eabi-gcc | 10.3.1 (2021.07-4) |
| OpenOCD | 0.11.0 |
| make | instalado |
| cmake | instalado |

- Scripts de OpenOCD disponibles: `interface/stlink.cfg` (auto-detect, lo usaremos), `target/stm32g4x.cfg`.
- Grupos del usuario: `dialout` y `plugdev` ✓.
- Reglas udev para ST-LINK: **faltan**, se crean en sesión actual.

### ST-LINK detectado por la Pi

```
$ lsusb | grep -i stmicro
Bus 001 Device 004: ID 0483:374b STMicroelectronics ST-LINK/V2.1

$ ls /dev/ttyACM*
/dev/ttyACM0
```

VID:PID `0483:374b` = ST-LINK V2.1 ✓. VCP (puerto serial virtual sobre USB) en `/dev/ttyACM0` — útil para `printf` debug más adelante.

### Reglas udev creadas

Archivo `/etc/udev/rules.d/49-stlinkv2-1.rules`:

```
SUBSYSTEM=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="374b", MODE="660", GROUP="plugdev", TAG+="uaccess"
```

Verificado: `/dev/bus/usb/001/00X` con grupo `plugdev` y modo `crw-rw----`. Acceso al ST-LINK sin sudo confirmado.

### OpenOCD ↔ STM32G431 — cadena completa funcionando

Comando: `openocd -f interface/stlink.cfg -f target/stm32g4x.cfg`

Resultado clave:
- `STLINK V2J34M25 (API v2) VID:PID 0483:374B` — firmware del ST-LINK moderno.
- `Target voltage: 3.269681` — alimentación lógica del G431 dentro de spec (3.3V nominal).
- `stm32g4x.cpu: hardware has 6 breakpoints, 4 watchpoints` — SWD funcional, Cortex-M4 responde a queries de debug.
- `Listening on port 3333 for gdb connections` — servidor GDB activo.

**Conclusión sesión 2**: hardware y toolchain validados. Listos para flashear código.

### Pendiente al cerrar sesión 2

- Bajar CMSIS Device Pack para STM32G4 (headers de registros, startup, linker script).
- Crear estructura `firmware/blink/` con CMakeLists, main.c, archivo de OpenOCD para flash.
- Compilar y flashear blink en PC6 (LED user).

---

## Sesión 3 — 2026-05-18

**Hito**: Primer firmware bare-metal compilado, flasheado y funcionando. **Fase 0 completada.**

### Vendor pack instalado

- Clone `STM32CubeG4` en `~/projects/stm32g4/vendor/STM32CubeG4/` con `--depth=1`.
- Submódulo `Drivers/CMSIS/Device/ST/STM32G4xx` inicializado vía `git submodule update --init` (solo ese, no los demás).
- Archivos clave disponibles: `stm32g431xx.h`, `startup_stm32g431xx.s`, `core_cm4.h`, linker scripts del NUCLEO-G431KB (compatibles por igual layout Flash 128KB / RAM 32KB).

### Workflow Mac→Pi establecido

- **SSH alias `raspi`** configurado en `~/.ssh/config` apuntando a `192.168.68.52`.
- **Passwordless SSH** vía `ssh-copy-id` + `id_ed25519`.
- **Script `firmware/sync_to_pi.sh`** (rsync con `--delete`, excluye `build/`). Corregido bug de array vacío bajo `set -u` en bash 3.2 macOS (cambiado a variable string).
- **Source of truth**: Mac (`~/Documents/tesis_maestria/firmware/`). Build/flash: Pi.

### Proyecto `firmware/blink/` creado

Estructura:

```
firmware/
├── .gitignore
├── sync_to_pi.sh
└── blink/
    ├── CMakeLists.txt    # cross-compile Cortex-M4 + FPU hard FPv4-SP
    ├── README.md
    └── src/
        └── main.c        # toggle PC6 con HSI 16 MHz por defecto
```

Características técnicas:
- **Bare-metal CMSIS sin HAL**: solo headers de registros + Core M4.
- **`SystemInit()` vacío** (HSI default suficiente).
- **`delay()` por NOP loop** (~0.5s a 16 MHz, no preciso — pendiente migrar a SysTick).
- **Flags compilación**: `-Og -g3 -ffunction-sections -fdata-sections --gc-sections --specs=nano.specs --specs=nosys.specs`.

### Build y flash exitosos

Footprint:

```
text       data        bss        dec        hex    filename
 880          0       1568       2448        990    blink.elf
```

- **880 bytes Flash** (0.7% de los 128 KB disponibles) — código + constantes.
- **1568 bytes RAM** (4.8% de 32 KB) — mayormente stack/heap reservados por el linker script (no uso runtime real).

Flash con OpenOCD:
- `device idcode = 0x20036468 (STM32G43/G44xx)` confirma chip.
- `Programming Finished` + `Verified OK` + `Resetting Target`.

### LED parpadeando — Fase 0 cerrada

Cadena end-to-end funcional:

```
Mac (Zed) → rsync → Pi → cmake/arm-gcc → openocd → SWD → STM32G431 → LED PC6
```

### Pendiente al cerrar sesión 3 (Semana 2 del planning)

1. **Migrar `delay()` a SysTick** — base de tiempo precisa por hardware.
2. **Configurar PLL @170 MHz** — pasar de HSI 16 MHz a la velocidad máxima del G431. Indispensable para FCS-MPC.
3. **UART debug por USART2** (PA2/PA3) — `printf` desde firmware visible en `/dev/ttyACM0` de la Pi.

Con estos tres, terminamos infraestructura básica y entramos a Fase 1 (PWM + ADC).

---

## Sesión 4 — 2026-05-18/19

**Hitos**:
- Metodología de aprendizaje formalizada en memoria persistente.
- Migración de `delay()` NOP a **SysTick** (código en Mac, falta build/flash).
- Refactor del firmware a **estructura multi-app** (`apps/01_blink/` + top-level `CMakeLists`).
- Vendor `STM32CubeG4` clonado también en Mac, espejo de la Pi.
- Setup de **`.clangd`** para que Zed/clangd en Mac resuelva los headers de CMSIS.

### Metodología: explicación línea por línea

Guardado en `memory/feedback_deep_understanding.md`. Resumen:
- Cada cambio de código se explica línea por línea, cross-ref con UM2516/RM0440/DS12589/AN5325.
- Sin código defensivo para casos imposibles, sin abstracciones prematuras.
- Macros CMSIS (`*_Msk`, `*_Pos`) se inspeccionan en su header, no se usan como caja negra.
- Verificación: hardware (LED, scope, UART log) — no solo build.

### SysTick implementado

Cambios en `main.c`:
- Variable `static volatile uint32_t g_ticks` incrementada por `SysTick_Handler`.
- `SysTick_Handler` (sin `static`) sobreescribe el weak del startup ST.
- `systick_init(ticks_per_irq)` configura `LOAD = ticks - 1`, resetea `VAL`, habilita 3 bits del `CTRL`: CLKSOURCE (AHB), TICKINT (IRQ), ENABLE.
- `delay_ms(ms)` espera por diferencia unsigned de `g_ticks` (robusto a wrap-around).
- `main()` llama `systick_init(16000)` → 1 ms tick a HSI 16 MHz.

Se diseccionó la referencia `SysTick_Config()` de CMSIS para identificar qué líneas omitir:
- ❌ Defensivo `if ((ticks-1) > MASK)` — input hardcoded, no puede fallar.
- ❌ `NVIC_SetPriority` — sin otras IRQs aún; cuando entre ADC se agrega.
- ❌ `return error` — función `void`.

**Pendiente**: build + flash + verificación con osciloscopio de período = 1.000 s en PC6.

### Refactor a estructura multi-app

Estructura nueva:

```
firmware/
├── CMakeLists.txt          # config compartida (toolchain, vendor, flags)
├── README.md
├── .gitignore
├── .clangd                 # solo para Mac/clangd
├── sync_to_pi.sh
└── apps/
    └── 01_blink/
        ├── CMakeLists.txt  # solo declara el target (5 líneas)
        ├── README.md
        └── src/main.c
```

Cambios:
- Top-level `CMakeLists.txt` movido a `firmware/` con cross-compile setup, paths del vendor, flags y `include_directories`. Usa `add_subdirectory(apps/01_blink)`.
- App-level `CMakeLists.txt` reducido a 3 líneas funcionales (declarar target, link map, post-build .bin/.hex).
- Convención de nombres: `apps/NN_<nombre>/` con prefijo numérico para marcar progresión pedagógica.
- `lib/` reservada para cuando exista código compartido entre 2+ apps (no antes).

### Vendor clonado en Mac

Mismo espejo que en la Pi para que clangd resuelva headers localmente:

```
tesis_maestria/
├── firmware/   ← edición con Zed local
├── vendor/STM32CubeG4/   ← clone con submódulo cmsis_device_g4 inicializado
└── ...
```

- `.gitignore` raíz agrega `vendor/` (no se versiona — código de terceros).
- Simetría con Pi: `~/projects/stm32g4/{firmware,vendor}` ↔ `tesis_maestria/{firmware,vendor}`.
- Bug intermedio: primer clone quedó en `firmware/vendor/` por error de `cd`. Corregido con `mv`.

### `.clangd` config

Archivo `firmware/.clangd` con `CompileFlags.Add`:
- `-DSTM32G431xx` — selecciona variante del chip dentro de `stm32g4xx.h`.
- `-I` absolutos a `vendor/STM32CubeG4/Drivers/CMSIS/{Device/ST/STM32G4xx,Core}/Include`.

Decisiones:
- Paths absolutos en lugar de relativos para evitar ambigüedad sobre el "directorio raíz" según la versión de clangd.
- Sin `--target=arm-none-eabi` ni `-mcpu` — Apple clangd 21 parsea bien sin ellos; solo se requiere si aparecen errores con macros tipo `__ARM_ARCH_7EM__`.
- **No usa `compile_commands.json`** — alternativa más manual pero suficiente para nuestro caso (sin necesidad de arm-gcc en Mac).

### Pendiente al cerrar sesión 4

1. **Confirmar que squiggles desaparecen** tras `killall clangd` + reabrir `main.c` en Zed.
2. **Sincronizar y compilar SysTick** en Pi (`./sync_to_pi.sh` + `cmake -B build && cmake --build build`).
3. **Flashear y verificar visualmente** que LED parpadea a 1 Hz.
4. **Medir período con osciloscopio** en PC6 para validar precisión del SysTick vs el viejo NOP-delay.

Después: **UART por USART2** (PA2/PA3) con `printf` retargeting a `/dev/ttyACM0` de la Pi.

---

## Sesión 5 — 2026-05-19

**Hito**: **PLL @170 MHz operativo desde HSE 8 MHz**. SysTick recalibrado y validado por prueba cruzada visual. Cierre parcial de Semana 2 del planning.

### Apertura: validaciones previas

- **Squiggles de clangd**: desaparecieron tras `killall clangd` + reabrir en Zed. `.clangd` con paths absolutos confirmado como solución estable.
- **SysTick en Pi**: build + flash OK desde sesión anterior, LED parpadeando a 1 Hz visualmente correcto.
- **Medición con osciloscopio**: pospuesta. El LED de status (PC6) es físicamente muy pequeño, difícil de puntear. PC6 no sale por ningún conector (confirmado en UM2516 Tabla 4, fila 29: PC6 → STATUS, sin pad accesible). Decisión: la validación visual a ~1 Hz es suficiente prueba de que SysTick funciona (interrupt + LOAD + delay_ms son condiciones necesarias para parpadeo periódico). Precisión absoluta del clock se medirá cuando esté UART operativa (printf de `g_ticks` vs timestamps de `/dev/ttyACM0`).

### Verificación del esquemático MB1419 — habilitar HSE

Antes de configurar el PLL desde HSE, verificación del circuito del cristal en el esquemático oficial:

- **Y2 = cristal de 8 MHz** (NO 24 MHz como se asumió en un momento).
- **R27 = resistor poblado de 220 Ω** (NO solder bridge abierto). Configuración Pierce estándar: damping resistor en serie con OSC-OUT (PF1).
  - Limita corriente de excitación del oscilador hacia el cristal.
  - Atenúa armónicos espurios.
  - Reduce EMI.
- Conclusión: **HSE conectado y operativo sin necesidad de soldar nada**.

### Configuración del PLL — clock.c

Nuevo archivo `apps/01_blink/src/clock.c` con función única `clock_init_170mhz_hse()`. Filosofía:
- 8 pasos en orden estricto (no se puede reordenar — restricciones físicas + del silicio).
- Cada paso con comentario "qué + por qué" y cross-ref a RM0440.
- Sin código defensivo: si HSE no engancha, el `while ((RCC->CR & RCC_CR_HSERDY) == 0U) { }` atrapa la ejecución para siempre → señal clara de fallo de hardware.

**Cadena de clocks elegida**:

| Parámetro | Valor | Justificación |
|---|---|---|
| HSE | 8 MHz | Cristal Y2 |
| M | 2 → encoding `PLLM = 1` | f_PLL_IN = 4 MHz ∈ [2.66, 16] |
| N | 85 | f_VCO = 340 MHz ∈ [96, 344] |
| R | 2 → encoding `PLLR = 0b00` | SYSCLK = 170 MHz |
| HPRE | /1 | HCLK = 170 MHz |
| PPRE1, PPRE2 | /1 | PCLK1 = PCLK2 = 170 MHz |

**Orden de operaciones implementado**:

1. `RCC.APB1ENR1.PWREN = 1` — clock gating del periférico PWR.
2. `PWR.CR5.R1MODE = 0` — Range 1 Boost (obligatorio para > 150 MHz).
3. `FLASH.ACR.LATENCY = 4 WS` — RM0440 Table 9. **Antes** de subir el reloj.
4. `RCC.CR.HSEON = 1`, esperar `HSERDY`.
5. `RCC.PLLCFGR = ...` — escribir todo el registro en un solo write atómico (con PLL apagado).
6. `RCC.CR.PLLON = 1`, esperar `PLLRDY`.
7. `RCC.CFGR.SW = PLL`, esperar `SWS == PLL`.
8. Prescalers HPRE/PPRE1/PPRE2 = /1 (explícitos por autodocumentación).

### Cambios a main.c

- `#include "clock.h"`.
- Llamada a `clock_init_170mhz_hse()` como **primera** instrucción de `main()`.
- `systick_init(170000U)` (antes `16000U`) → mantiene tick de 1 ms con HCLK=170 MHz.
- `SystemInit()` se mantiene vacía (decisión consciente: clock setup en main para mayor control + debugging paso a paso).

### Cambios a CMakeLists.txt

- Agregada `src/clock.c` al `add_executable(blink.elf ...)`.

### Validación cruzada SysTick ↔ PLL

Resultado: **LED parpadea a 1 Hz visual exactamente igual que antes.**

Esto es prueba simultánea de que:
- PLL multiplicó correctamente (340 MHz VCO, 170 MHz SYSCLK).
- `LOAD = 170000` con HCLK efectivo de 170 MHz da 1 ms tick.
- Si cualquier divisor (M, N, R) estuviera mal, el período cambiaría perceptiblemente.

### Continuación sesión 5 — UART USART2 + cierre Fase 0

**Corrección importante**: PA2/PA3 NO son USART en esta placa. Están ocupados por **OP1_OUT** (PA2) y **Curr_fdbk1_OPAmp-** (PA3) — frontend analógico del current sensing del shunt 1. ST routeó USART2 a **PB3/PB4** (UM2516 Tabla 4 filas 41–42). Lección de método: la Tabla 4 es la fuente autoritativa; toda asunción "PA2/PA3 son USART2" basada en defaults genéricos del G4 hay que validarla contra la tabla específica de la placa.

### Verificación del cableado VCP

- **Esquemático MB1419 hoja ST-LINK**: existen las redes `USART2_TX_ST_LINK` y `USART2_RX_ST_LINK`, conectadas a PA2/PA3 del STM32F103 (ST-LINK) vía **R23/R24 = 0 Ω** (resistores poblados de fábrica, actúan como solder bridges normalmente cerrados — desoldables si se quiere usar PB3/PB4 del G431 para otra cosa).
- **Esquemático MB1419 hoja MCU principal**: los labels exactos de las redes sobre PB3/PB4 no se renderizan legibles en el PDF (etiquetas vectorizadas en zona vacía superior del símbolo). Decisión metodológica: aceptar la evidencia indirecta (UM2516 Tabla 4 oficial + redes existentes del lado F103 + convención ST para VCP en placas Discovery/Nucleo) y validar empíricamente con el firmware.
- **Validación empírica**: se ejecutó. Resultado abajo.

### Implementación

Dos archivos nuevos en `apps/01_blink/src/`:

- **`uart.h`**: declaraciones públicas (`uart2_init`, `uart2_putc`, `uart2_puts`).
- **`uart.c`**: 7 pasos de init en polling-mode + retargeting `_write` para `printf`.

**Cadena de configuración USART2**:

1. `RCC.AHB2ENR.GPIOBEN = 1` — clock al puerto B.
2. `RCC.APB1ENR1.USART2EN = 1` — clock al USART2.
3. `GPIOB.MODER` PB3/PB4 = `0b10` (Alternate Function).
4. `GPIOB.AFR[0]` PB3/PB4 = `7` (AF7 = USART2 en G4, DS12589 Table 13).
5. `USART2.CR1 = CR2 = CR3 = 0` (estado conocido, UE=0 → registros modificables).
6. `USART2.BRR = (170e6 + baud/2) / baud` — para 115200 → 1476, error −0.02%.
7. `CR1 = TE | RE`, luego `CR1 |= UE` (UE último, obligatorio).

**Retargeting `printf`**:

```c
int _write(int fd, char *buf, int len) {
    (void)fd;
    for (int i = 0; i < len; i++) uart2_putc(buf[i]);
    return len;
}
```

`--specs=nosys.specs` (en CMakeLists top-level) provee `_write` como weak stub; nuestra definición no-weak lo sobrescribe. `setvbuf(stdout, NULL, _IONBF, 0)` desactiva el buffering interno de newlib-nano para que cada `printf` salga inmediatamente.

### Resultado del flash — VCP confirmado

`stty -F /dev/ttyACM0 115200 cs8 -cstopb -parenb raw && cat /dev/ttyACM0`:

```
[boot] STM32G431 @170MHz, USART2 OK
tick 28  uptime=14050 ms
tick 29  uptime=14552 ms
tick 30  uptime=15054 ms
tick 31  uptime=15556 ms
```

### Validación cuádruple desde una sola observación

El delta entre ticks = **502 ms** (no 500 ms exactos) confirma simultáneamente:

| Sistema | Cómo se valida | Resultado |
|---|---|---|
| PLL @170 MHz | Si f_CK ≠ 170 MHz, baud sale mal → texto sería basura | ✅ Texto legible |
| BRR cálculo | Mismo razonamiento | ✅ Error −0.02% real |
| SysTick + delay_ms | Los 500 ms del `delay_ms` se respetan | ✅ |
| UART Tx polling | Los 2 ms extra = costo de `printf` bloqueando en TXE | ✅ |

**Predicción que dio la cifra exacta**: 24 chars × 10 bits × (1/115200 s) ≈ **2.08 ms**. Coincide al milisegundo con el delta observado (502 − 500 = 2 ms). Esto demuestra que el modelo mental de UART polling es correcto y que el clock está bien calibrado.

### Estado: Fase 0 cerrada ✅

| Hito Fase 0 | Estado |
|---|---|
| Toolchain Mac→Pi→ST-LINK→G431 | ✅ Sesión 2 |
| Vendor CMSIS Device Pack | ✅ Sesión 3 |
| Blink LED PC6 | ✅ Sesión 3 |
| Refactor multi-app + .clangd | ✅ Sesión 4 |
| SysTick 1 ms | ✅ Sesión 4–5 |
| PLL HSE → 170 MHz | ✅ Sesión 5 |
| UART2 + `printf` por VCP | ✅ Sesión 5 |

**~30 h presupuestadas para Fase 0 → realizadas en ~4 sesiones / mucho menos tiempo del estimado**. La estimación era conservadora, lo que nos da colchón para Fase 1.

---

## Sesión 6 — 2026-05-19/20

**Hito**: **Estudio TIM1 completado + skeleton `apps/02_pwm_adc/` + refactor `lib/`**. Apertura formal de Fase 1.

### Cuaderno pedagógico `FIELD_NOTES.md` creado

Nuevo archivo en raíz del proyecto: `FIELD_NOTES.md`. Es el **libro del razonamiento** (no bitácora de avance — eso vive aquí). Cada nota sigue: panorama → analogía → detalle técnico con cross-ref al manual → por qué importa.

Backfill de **Fase 0** (6 notas): bare-metal vs HAL, toolchain Mac→Pi→ST-LINK, CMSIS y vendor packs, árbol de relojes + PLL 170 MHz, SysTick, UART + VCP.

Estudio completo de **TIM1** (5 notas, N1.3–N1.7):
- N1.3 — Counter modes (edge vs center-aligned). Decisión: **CMS = 01 (center-aligned mode 1)**, ARR = 2833 para 30.0025 kHz exactos (error +82 ppm, despreciable).
- N1.4 — Cadena contador → pin: PWM mode 1 (OCxM=0110), polaridad active-high (CCxP=0), habilitación (CCxE + MOE), preload con shadow registers (OCxPE + ARPE + EGR.UG).
- N1.5 — Complementarios + dead-time. Encoding no-lineal del DTG[7:0]: 4 sub-rangos con resolución decreciente. Para 500 ns @ CKD=00: **DTG = 0x55**. La trampa de OISx/OISxN: estado de las salidas cuando MOE=0.
- N1.6 — TRGO al ADC. Decisión: **MMS = 010 (update event)** + RCR = 1 → 1 muestra por periodo PWM. EXTSEL = 9 en ADC para escuchar TIM1_TRGO. Truco avanzado de OC4REF como TRGO programable (no se usa en Fase 1).
- N1.7 — Break inputs (BKIN/BKIN2). Decisión: **no implementar en Fase 1**. La placa no expone BKIN externo; protección por software en la ISR del ADC. Documentado para futuras iteraciones (OPAMP → COMP interno → BKIN).

Convención: cuando alguien dé una explicación pedagógica nueva, va a `FIELD_NOTES.md` con la misma estructura. Workflow guardado en memoria como `feedback-field-notes-workflow`.

### Pinout TIM1 confirmado (UM2516 Tabla 4)

| Canal | High-side | Complementario (low-side) |
|---|---|---|
| TIM1_CH1 | **PA8** | **PC13** (compartido con TAMP/RTC en otros chips; aquí dedicado al gate driver) |
| TIM1_CH2 | **PA9** | **PA12** |
| TIM1_CH3 | **PA10** | **PB15** |

Todos en **AF6** (DS12589 Tabla 13).

### Correcciones de numeración de capítulos en RM0440

El plan original tenía mal los caps. Corregido:
- TIM1 = **Cap 29** (no Cap 28). Cap 28 es HRTIM (otro timer, no aplica).
- ADC = Cap 21 ✓
- OPAMP = **Cap 25** (no Cap 24).

### Refactor a `firmware/lib/`

Como ahora hay **2 apps** que comparten clock + UART, se cumplió la condición para crear `lib/`:

```
firmware/
├── CMakeLists.txt           ← + add_subdirectory(lib) y apps/02_pwm_adc
├── .clangd                  ← + path a firmware/lib/
├── lib/                     ← NUEVO
│   ├── CMakeLists.txt       (add_library stm32g4_lib STATIC)
│   ├── clock.{c,h}          (movidos desde apps/01_blink/src/)
│   └── uart.{c,h}           (movidos desde apps/01_blink/src/)
└── apps/
    ├── 01_blink/            ← target_link_libraries(blink.elf PRIVATE stm32g4_lib)
    │   ├── CMakeLists.txt
    │   ├── README.md
    │   └── src/main.c       (sin cambios; #include "clock.h" sigue funcionando)
    └── 02_pwm_adc/          ← NUEVO
        ├── CMakeLists.txt
        ├── README.md
        └── src/
            ├── main.c       (init clock+UART, llama stubs, heartbeat 1Hz)
            ├── pwm.h        (API: pwm_init, pwm_set_duties, pwm_enable, pwm_disable + constantes PWM_ARR, PWM_DEAD_DTG)
            └── pwm.c        (stubs vacíos + plan de implementación documentado en comentario)
```

`stm32g4_lib` es STATIC library con `target_include_directories ... PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}` — los consumers heredan automáticamente el include path.

### Build limpio en la Pi

```
[ 30%] Linking C static library libstm32g4_lib.a
[ 30%] Built target stm32g4_lib
[ 60%] Linking C executable blink.elf
   text       data        bss        dec        hex
   8632        100       1588      10320       2850   blink.elf       (idéntico antes y después del refactor)
[100%] Linking C executable pwm_adc.elf
   text       data        bss        dec        hex
   8680        100       1588      10368       2880   pwm_adc.elf     (+48 bytes flash vs blink; stubs + 1 printf extra)
```

Diferencia de 48 bytes Flash = 4 funciones stub (`pwm_init`, `pwm_set_duties`, `pwm_enable`, `pwm_disable`) + un `printf("[pwm_init] STUB ...")`. Razonable.

### Skeleton arrancando — validación quíntuple por VCP

```
[boot] STM32G431 @170MHz, USART2 OK (app: 02_pwm_adc)
[pwm_init] STUB — TIM1 todavía no configurado
alive tick=0 uptime=9 ms
alive tick=1 uptime=1011 ms
alive tick=2 uptime=2013 ms
```

Validaciones simultáneas en una sola lectura:
1. **PLL @ 170 MHz** OK (texto coherente → baud correcto → SYSCLK correcto).
2. **BRR del USART2** OK.
3. **SysTick 1 ms** OK (delta entre prints = 1002 ms; los 2 ms extra son `printf alive` de 30 chars × 87 μs).
4. **Retargeting de `printf`** funciona desde el nuevo target.
5. **Refactor a `lib/`** OK — clock.c y uart.c linkean correctamente desde un .elf distinto al blink.

Predicción teórica del primer `uptime`:
- 55 chars del `[boot]...` + 51 chars del `[pwm_init]...` = 106 chars
- A 87 μs/char (115200 baud) = **9.22 ms**
- Observado: **9 ms**. Coincide al ms.

### Pendiente al cerrar sesión 6

1. **Implementar `pwm_init()` registro por registro**, con cross-ref al manual y verificación incremental con osciloscopio:
   - Clock gating (RCC.AHB2ENR + RCC.APB2ENR).
   - GPIOs en AF6 (PA8, PA9, PA10, PA12, PB15, PC13).
   - TIM1 base (CR1.CMS, CR1.ARPE, ARR, RCR).
   - Cada canal x ∈ {1,2,3}: CCMRx, CCER, CCRx.
   - BDTR: DTG = 0x55, MOE = 0 todavía.
   - CR2: MMS = 010 (TRGO), OISx/OISxN = 0.
   - EGR.UG para cargar shadows.
2. **Validar PWM con osciloscopio** sobre J7 (puntas a OUT1 vs OUT1N): complementariedad + dead-time medido + frecuencia 30 kHz.
3. **Solo entonces** continuar a Semana 5 (OPAMPs + ADC).

---

## Revisión de f_PWM al cierre de Sesión 6 — 50 kHz en lugar de 30 kHz

**Decisión revisada el 2026-05-20**, antes de implementar registros del TIM1. Análisis cuantitativo de 6 factores (rizado de corriente, constante eléctrica, presupuesto computacional, AS5600, switching losses, dead-time fraccionario) → 50 kHz balancea mejor que 30 kHz. **Detalle completo en FIELD_NOTES.md N1.8.**

Resumen del veredicto:
- A 30 kHz, rizado de corriente = 92% relativo (borderline insuficiente para una L=0.86 mH).
- A 50 kHz, rizado baja a 56% y `ARR = 1700` cae exacto (vs 2833.33 → +82 ppm a 30 kHz).
- Presupuesto computacional: 3400 ciclos de CPU disponibles por ISR @ 50 kHz — cabe según literatura para FCS-M2PC (~1500-2500 ciclos típicos), pero hay que vigilar en Semana 6.
- Fallbacks si Semana 6 muestra que no cabe: bajar a 30 kHz, o cambiar AS5600 → AS5048A/AS5047P (SPI, ~6× ancho de banda).

Cambios concretos:
- `apps/02_pwm_adc/src/pwm.h`: `PWM_ARR` = 1700 (era 2833). Comentario explica el cálculo y referencia a N1.8.
- `FIELD_NOTES.md` N1.3 actualizada con el nuevo cálculo de ARR.
- `FIELD_NOTES.md` N1.8 nueva, dedicada al análisis cuantitativo de los 6 factores.

---

## Sesión 9 — 2026-05-20/21 — Bring-up ADC + OPAMP funcionando ✅ Cierre Semana 5

**Hito**: cadena completa **shunt → OPAMP → ADC → memoria** funcionando con motor desconectado. 3 corrientes leen valores consistentes (~318 raw c/u), trigger TIM1_TRGO disparando, dual injected simultaneous operativo.

### Lo que funciona

- `apps/02_pwm_adc/src/adc.{c,h}` implementado con todas las funciones (opamp_init, adc_init, get_currents_raw, get_vbus_raw).
- OPAMP1/2/3 en PGA mode interno gain x2, OPAMPINTEN=1 (salida directa a canales ADC internos 13/16/18).
- ADC1+ADC2 en dual injected simultaneous (DUAL=00101).
- Trigger inyectado: TIM1_TRGO en rising edge → conversiones a 50 kHz exactos.
- Vbus en canal regular ADC1_IN1, lectura ~553 raw = ~12 V correcto.
- Sample time 6.5 ciclos, resolución 12 bits.

### 3 bugs encontrados y resueltos durante el bring-up

Todos documentados en FIELD_NOTES.md N1.13:

1. **SYSCFG clock no habilitado** → writes a OPAMP_CSR se ignoraban silenciosamente (devolvían 0 al leer). Fix: `RCC->APB2ENR |= RCC_APB2ENR_SYSCFGEN`.

2. **JEXTSEL tiene tabla distinta a EXTSEL** (RM0440 Tabla 167 vs 162). Para TIM1_TRGO inyectado, JEXTSEL = 0x00, NO 0x09. Fix: cambiar valor en JSQR.

3. **Standalone mode sin topología clara del PCB** → OPAMPs saturaban a rail. Switch a PGA mode interno gain x2 para tener feedback definido internamente, no depende del board layout.

### Limitaciones aceptadas para esta iteración

- **PGA gain x2 sin bias a Vrefint/2** → solo medimos corrientes **positivas**. Para AC bipolar de FCS-M2PC, eventualmente: volver a standalone con bias correcto del PCB, o mantener PGA + offset DC restado en software.
- **Ganancia raw → amperios** no calibrada (pendiente Semana 6).
- **Offset DC ~318 raw** no compensado (pendiente Semana 6).

### Dead-time empíricamente NO verificado

Quedó pendiente de sesión 8: medir el dead-time programado (500 ns, DTG=0x55) con scope en el flanco de OUT2. Validar empíricamente queda como TODO para Semana 6 también.

### Para sesión 10 — Semana 6

Plan:
1. ISR JEOS (end of injected sequence) → callback a 50 kHz donde vivirá el FCS-M2PC.
2. Calibración de offset: 1000 muestras motor off → promediar → guardar `i_offset_a/b/c`.
3. Calibración de ganancia: inyectar corriente conocida (DC con fuente bench externa) → medir raw → calcular escala raw → A.
4. Medir tiempo de ISR con GPIO toggle + scope. Target: < 15 μs de los 20 μs disponibles a 50 kHz.
5. **Decisión final OPAMP topology**: PGA + offset SW vs standalone con bias del PCB.
6. (Si tiempo) verificar dead-time empíricamente.

---

## Sesión 8 — 2026-05-20 — Bug AF resuelto ✅ las 3 fases conmutan

**Hito**: el bug que dejó la sesión 7 abierta fue resuelto en ~30 minutos al inicio de la sesión 8. Las 3 fases del puente trifásico generan PWM 50 kHz, duty 50%, simétricas. **Cierre completo del bring-up del PWM.**

### Root cause

DS12589 Tabla 13 (Alternate Function table) es **por-pin**, no por periférico. Cada pin tiene su propio mapeo de qué función está en cada AF0–AF15. ST distribuye el TIM1 en distintos AFs según el pin:

- En GPIOA: TIM1_CH1/2/3/CH2N en AF6 (PA8, PA9, PA10, PA12).
- En GPIOB: TIM1_CH3N en **AF4** (PB15).
- En GPIOC: TIM1_CH1N en **AF4** (PC13).

Mi código en `pwm_init()` asumía AF6 universal. Resultado: PC13 quedó routeado a TIM8_CH4N (función no usada → output indefinido), y PB15 quedó en una AF sin función específica → ambos low-sides nunca recibían PWM → bootstrap caps de fase A y C nunca se cargaban → high-side tampoco conmutaba → output flotante en ~8V.

### Fix

Dos líneas modificadas en `apps/02_pwm_adc/src/pwm.c`:

```c
gpio_set_af(GPIOB, 15U, 4U);   // PB15 CH3N → AF4 (era 6)
gpio_set_af(GPIOC, 13U, 4U);   // PC13 CH1N → AF4 (era 6)
```

### Validación

- Build limpio, 0 warnings.
- Dump por VCP confirma `PB15 AFR=4` y `PC13 AFR=4`.
- Scope muestra las 3 OUTs (OUT1/OUT2/OUT3 en J7) con PWM idéntica: 50 kHz, duty 50%, amplitud 0–12V.
- Consumo de la fuente: ~30–50 mA estable.

### Lección persistida

`FIELD_NOTES.md` N1.9 — "La trampa del Alternate Function". Incluye:
- Tabla maestra de AFs para los 6 pines TIM1 de la placa.
- Meta-lección: los dumps de validación esconden bugs si el "expected" viene del mismo modelo mental erróneo del código.
- Recomendación: para pines futuros (I²C1 PB6/PB7 del AS5600), verificar AF directamente del datasheet pin por pin.

### Tareas cerradas

- Task #3 — Implementar TIM1 50 kHz center-aligned + 6 PWMs ✅
- Task #4 — TIM1 TRGO en update-event ✅ (parte del mismo pwm_init, validado por dump CR2=0x20).

### Próximo paso — Semana 5 del planning

Avanzar a **OPAMP1/2/3 + ADC1/ADC2 dual simultaneous** (Task #5). Lectura previa requerida:
- RM0440 Cap 21 (ADC) — JEXTSEL=TIM1_TRGO para sincronizar con pico/valle del PWM.
- RM0440 Cap 25 (OPAMP) — configuración como PGA con feedback externo.
- UM2516 — ruta de shunts → OPAMP → ADC en MB1419.

Crear nota N1.10 en FIELD_NOTES (estructura ADC con sus secciones panorama → analogía → detalle → por qué importa) antes de tocar código.

---

## Sesión 7 — 2026-05-20 — `pwm_init()` implementado, bug abierto

**Hito**: pwm_init() implementado y verificado por dump exhaustivo, pero **solo fase B (OUT2) conmuta**; OUT1 y OUT3 quedan flotantes en ~8V.

### Verificación del firmware (positivo)

Dump completo por VCP confirma:
- 12 registros TIM1 con valores esperados (CR1=0xB1 con bit DIR variable, CR2=0x20, ARR=1700, RCR=1, PSC=0, CCMR1=0x6868, CCMR2=0x68, CCER=0x555, BDTR=0x8C55 post-enable, CCRx=850).
- 6 pines GPIO con MODER=AF y AFR=6: PA8, PA9, PA10, PA12, PB15, PC13.
- Clock gating: RCC->AHB2ENR=0x07 (GPIOA/B/C), RCC->APB2ENR=0x800 (TIM1).
- CNT cambia entre prints → counter corriendo a 50 kHz.

### Comportamiento del puente (negativo)

- **OUT2** (CH2 = PA9 + PA12, ambos en GPIOA): PWM cuadrada limpia, 50 kHz, duty como configurado. ✅
- **OUT1** (CH1 = PA8 + PC13): output flotante en ~8 V constante. ❌
- **OUT3** (CH3 = PA10 + PB15): output flotante en ~8 V constante. ❌

Patrón: la única fase que funciona tiene **high-side Y low-side ambos en GPIOA**. Las que fallan tienen el low-side en otro puerto (GPIOB o GPIOC).

### Vbus operacional

- Fuente buck-boost ZK-4KX configurada CV=12V CC=500mA.
- Vbus medido en placa: 12 V estables.
- Consumo: ~30 mA estable (placa + L6387 quiescente). No entra en CC.

### Hipótesis priorizadas para sesión 8

1. **DBP backup domain** (PC13): falta habilitar acceso al backup domain antes de configurar PC13.
   ```c
   RCC->APB1ENR1 |= RCC_APB1ENR1_PWREN;
   PWR->CR1 |= PWR_CR1_DBP;
   ```
2. **RTC/TAMP**: leer `RTC->TAMPCR` y `RTC->CR` para descartar posesión de PC13.
3. **SYSCFG**: leer registros de SYSCFG para detectar routing especial de PB15 (también puede ser USB_DP en AF12).
4. **Reset state**: dumpear MODER/AFR ANTES de pwm_init() para ver estado de partida.
5. **MCSDK reference**: descargar X-CUBE-MCSDK ejemplo para B-G431B-ESC1, comparar línea por línea.

### Estado del firmware al cierre

- `apps/02_pwm_adc/src/pwm.c`: duties extremos para diagnóstico (CCR1=0, CCR2=850, CCR3=1690).
- `apps/02_pwm_adc/src/main.c`: dump diagnóstico GPIO + post-enable verification.
- ⚠ El test de duties extremos NO se llegó a medir — usuario pausó para retomar con cabeza fresca.

### Para arrancar sesión 8

1. Revisar firmware actual (pwm.c + main.c) antes de modificar.
2. Aplicar H1 (DBP) como primera prueba: agregar PWR_CR1_DBP en pwm.c antes de configurar PC13.
3. Si no funciona, ir a H2/H3 (dumps de RTC/SYSCFG).
4. Como último recurso, H5 (comparar con MCSDK).

### Documentación pedagógica pendiente

Cuando resolvamos el bug, escribir nota N1.9 en FIELD_NOTES.md: "Trampas de pines especiales en STM32G4 — PC13 y backup domain". Cualquier solución que encontremos vale la pena documentar.

---

## Próxima sesión — debug fase A/C del puente (continuación)

### Punto de partida

`apps/02_pwm_adc/src/pwm.c` tiene el plan documentado como comentario (9 pasos). Cada paso corresponde a una sección del `FIELD_NOTES.md` N1.3–N1.6, con la frecuencia revisada en N1.8.

### Estrategia de verificación incremental

Después de cada paso del plan, validar antes de seguir:

| Paso | Verificación |
|---|---|
| 1. Clock gating | Leer registros RCC.AHB2ENR, RCC.APB2ENR con GDB → bits seteados |
| 2. GPIOs en AF6 | GPIOA.MODER, GPIOA.AFRH, GPIOB.MODER, GPIOC.MODER con GDB |
| 3. TIM1 base | TIM1.CR1, TIM1.ARR, TIM1.RCR con GDB. Aún sin output. |
| 4. Canales con MOE=0 | TIM1.CCMR1/2, TIM1.CCER. Habilitar CEN=1 → contador corre, pero salidas todavía silenciadas por MOE=0. |
| 5. Dead-time + MOE=1 | Punteo de scope en PA8 (CH1): aparece PWM al 50% a 30 kHz. |
| 6. Complementario | Punteo simultáneo en PA8 y PC13: forma complementaria, dead-time visible. |
| 7. TRGO | Setear OC4 con CCR4=ARR-100 + MMS=0111 temporalmente como debug, ver TRGO en CH4 → confirma timing. (Luego revertir a MMS=010.) |

### Decisión de seguridad

`pwm_init()` deja MOE = 0. La primera vez que llamemos `pwm_enable()` (que pone MOE=1) las 6 salidas se activan instantáneamente. **Antes de esa llamada, hay que tener listo**:

- Duty cycles inicializados a 50% (medio del rango, voltaje promedio cero entre fases).
- Mecanismo para apagar (botón, watchdog, IRQ del UART).
- Motor desconectado en la primera prueba (verificar PWM con scope, no con motor cargado).

### Comandos rápidos de retoma

```bash
# Mac → Pi
cd ~/Documents/tesis_maestria/firmware
./sync_to_pi.sh

# Pi (vía SSH alias `raspi`)
ssh raspi
cd ~/projects/stm32g4/firmware
cmake -B build && cmake --build build

# Flashear (OJO: el .elf vive en build/, no en el directorio fuente)
openocd -f interface/stlink.cfg -f target/stm32g4x.cfg \
        -c "program build/apps/02_pwm_adc/pwm_adc.elf verify reset exit"

# Lectura VCP (terminal aparte)
stty -F /dev/ttyACM0 115200 cs8 -cstopb -parenb raw
cat /dev/ttyACM0
```

**Nota**: la versión anterior de los comandos en este archivo omitía el prefijo `build/` y solo funcionaba si se corría openocd desde dentro de `build/`. Esta es la versión explícita.

### Estado del firmware al cierre

```
firmware/
├── CMakeLists.txt            # toolchain + vendor + add_subdirectory(lib, apps/01_blink, apps/02_pwm_adc)
├── .clangd                   # paths absolutos para Zed/clangd, incluye lib/
├── .gitignore
├── sync_to_pi.sh
├── README.md
├── lib/
│   ├── CMakeLists.txt        # add_library(stm32g4_lib STATIC ...)
│   ├── clock.{c,h}           # PLL HSE 8 MHz → 170 MHz
│   └── uart.{c,h}            # USART2 polling + retargeting _write
└── apps/
    ├── 01_blink/             # validador de Fase 0 (LED + UART), todavía operativo
    │   ├── CMakeLists.txt
    │   ├── README.md
    │   └── src/main.c
    └── 02_pwm_adc/           # ← Fase 1: TIM1 PWM + ADC sync
        ├── CMakeLists.txt
        ├── README.md
        └── src/
            ├── main.c        # arranca pero pwm_init() es stub
            ├── pwm.h         # API + constantes PWM_ARR=2833, PWM_DEAD_DTG=0x55
            └── pwm.c         # STUBS + plan documentado en comentario
```

---
