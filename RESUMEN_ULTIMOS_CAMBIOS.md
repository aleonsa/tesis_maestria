# Resumen de los últimos cambios (al 2026-10-18, sesión 22)

Cubre el trabajo posterior al commit `dee0967` (2026-08-23): redacción del Cap. 1 de la
tesis (2026-08-17), la sesión 15 de firmware (2026-08-24), la sesión 16 (2026-10-11),
donde el I²C por IRQ quedó validado en banco y el toolchain migró a la PC local, y la
sesión 17 (2026-10-12), donde el observador PLL tipo II quedó validado en banco.

---

## 1. Tesis — Cap. 1 Introducción escrito (Bloque B cerrado)

Archivo: `tesis_documento/Documentos/Chapters/Capitulo1/Cap1.tex` (reemplaza los `\lipsum`).

- **1.1 Motivación**: ubicuidad del BLDC → rizo de par → la BEMF trapezoidal ideal no se cumple
  → los controladores modernos cargan una hipótesis de forma de onda en su modelo interno
  → caracterizar offline resuelve un motor, no el problema → ¿puede el controlador aprender
  la forma de onda mientras opera?
- **1.2 Estado del arte**, cuatro subsecciones:
  - Reducción de rizo (diseño de máquina vs. control; referencia óptima de Le-Huy; rizo de
    conmutación; DTC; control repetitivo e ILC).
  - FCS-MPC: sus dos debilidades (rizo intra-periodo, sensibilidad al modelo). El M2PC se
    atribuye a Tarisciotti et al. 2015; Coronado 2025 lo adapta al BLDC.
  - Estimación de BEMF y aprendizaje: observadores orientados a posición, NN profundas y sus
    tres inconvenientes, ADALINE; zhao2025 como lo más cercano (compensador aditivo, no toca
    el predictor).
  - La brecha que ocupa la tesis.
- **1.3 Contribuciones**: 5 párrafos alineados 1:1 con los objetivos específicos. Números:
  69 % / 61 % de reducción de rizo vs. TRAP, 40 parámetros vs. ~25k.
- **1.4 Estructura de la tesis**: un párrafo por capítulo con `\ref`.

Otros cambios:
- `Cap6.tex` ahora lleva `\label{cap:conclusiones}` (lo referencia 1.4).
- `graphicspath` de Cap. 1 corregido.
- PDF regenerado: **56 páginas**, 0 referencias/citas indefinidas, 0 overfull boxes.
- `TESIS_ESCRITURA.md` actualizado.

**⚠ Pendiente de verificar:** 7 entradas nuevas en `Bibliografia.bib` con metadatos escritos de
memoria (sin PDF en `papers/`): `jahns1996pulsating`, `carlson1992commutation`, `liu2005dtc`,
`qian2004ilc`, `rodriguez2013stateofart`, `cortes2012delay`, `tarisciotti2015m2pc`.

---

## 2. Firmware — Sesión 15: errata de silicio en el I²C por IRQ

Archivos: `firmware/lib/i2c.c`, `firmware/lib/i2c.h`, `firmware/apps/03_as5600/src/main.c`.
Documentación: `FIELD_NOTES.md` N1.20, `PROGRESO_HARDWARE.md` sesión 15.

**Causa raíz.** ES0431 Rev. 9 §2.11.5 (`I2C2-190208`, "Transmission stalled after first byte"):
el workaround exige precargar el primer byte en `TXDR` **antes** de generar `START`. El driver
hacía lo contrario (`START` → esperar `TXIS` → escribir registro). Las release notes de
STM32CubeG4 confirman que el HAL ya aplica esa precarga.

**Cambios en el driver:**
- `i2c1_read_regs()` e `i2c1_read_regs_start()` precargan el registro en `TXDR` antes de `CR2.START`.
- Camino IRQ nominal: 4 eventos para 2 bytes (`TC`, `RXNE`, `RXNE`, `STOPF`). Un `TXIS`
  inesperado aborta, nunca reenvía el registro.
- Fuentes de IRQ habilitadas por transferencia y deshabilitadas antes de publicar estado terminal.
- `i2c1_nb_abort()` es asíncrono: pide STOP y retorna; `STOPF` autoriza el cierre. Sin polling
  ni reset de `PE` dentro de la ISR.
- Flag `closing` + secciones críticas cortas contra la ISR de control (prio 1 > I²C prio 3).
- Flags de bus capturados antes de limpiarlos; `seq/errors` y `DONE/ERR` se publican juntos al final.

**Cambios en la app `03_as5600`:**
- Timeout de `nb_pump()` conserva su timestamp entre llamadas; si `STOPF` no llega, recuperación
  con `i2c1_reset_peripheral()` desde foreground.
- Prueba de arranque reducida a una sola lectura IRQ aislada, que imprime la traza de eventos.
- `nb_report_failure()` compara contra el contador previo en lugar de contra cero.

---

## 3. Firmware — Sesión 16: validación en banco + migración a PC local (2026-10-11)

Documentación: `PROGRESO_HARDWARE.md` sesión 16, `FIELD_NOTES.md` N1.21, `firmware/README.md`
actualizado con el workflow local.

**Migración Pi → PC local.** Ya no se usa `sync_to_pi.sh` ni SSH: la PC propia conecta el
B-G431B-ESC1 directo por USB. Herramientas ya estaban (openocd 0.12, arm-none-eabi-gcc 13.2.1,
cmake, udev, grupo dialout); lo único que faltaba era el clone de `STM32CubeG4`, ahora en
`tesis_maestria/vendor/STM32CubeG4`. Flash local usa
`reset_config srst_only connect_assert_srst` (default del banco desde hoy).

**El bug del día era el cable.** Error -71 al enumerar + boot-loop del target (159 banners en
30 s, tramas UART cortadas a mitad de línea) + disconnects espontáneos = un solo culpable:
VBUS marginal por cable micro-USB degradado. Bisect con cable nuevo: un solo boot, USB estable.

**Pendientes de la sesión 15 — CERRADOS en banco:**

```
[scan]   ACK en 0x36   <-- AS5600
[as5600] STATUS=0x20  MD=1 ML=0 MH=0
[shot] 0: OK  raw=4089     | IRQ: TXIS=0 TC=1 RXNE=1 STOPF=1 | entradas=4
[perf] 2139 lecturas/s (teorico ~2200)  no-arrancadas=0  errores=0  abortadas=0
```

Giro manual del rotor: `raw` monótono 0..4095 con un solo salto por vuelta.

---

## 4. Firmware — Sesión 17: observador PLL tipo II validado (2026-10-12)

Archivos nuevos: `firmware/lib/theta_pll.{c,h}`, `firmware/apps/04_theta_pll/`,
`python_scripts/plot_theta.py`. Documentación: `PROGRESO_HARDWARE.md` sesión 17,
`FIELD_NOTES.md` N1.22.

**Problema**: la ISR corre a 50 kHz pero el bus entrega muestra nueva cada ~468 µs → en
~23 de 24 ciclos no hay dato fresco; ZOH mete θ a saltos y derivarla da ~24 rad/s de ruido.

**Solución**: observador PLL tipo II en Q32 puro — extrapolación `θ̂ += ω̂` cada tick,
corrección `θ̂ += Kp·e`, `ω̂ += Ki·e` por muestra (ωn = 2π·100, ζ = 1). θ_e = 7·θ̂_mec con el
overflow como mod 2π eléctrico. Compensación de latencia proporcional a ω̂ (30 ticks =
600 µs), no constante en cuentas. Validación con TIM2 a 50 kHz sin motor.

**Resultados en banco** (rotor a mano, 180 s): error máx θ̂−raw = **0.27° mec**, ISR completa
**2.11 µs** de 20 µs, corr/s = 2000, i2c_err = 0. Gráfica:
`firmware/apps/04_theta_pll/doc/validacion_pll_2026-10-12.png`.

---

**Pares de polos — cerrado**: banco (2804 12N14P) = 7 pares; simulación (Anaheim
BLY-344S) = 4 pares, donde `p.P = 8` cuenta polos. Corregido el comentario erróneo de
`openloop.h`; tabla definitiva en `PLANNING_HARDWARE.md` §2.

---

## 5. Qué sigue

Detalle completo en `PROGRESO_HARDWARE.md` sesión 17.

**Sin fuente:**
1. Presupuesto completo de ISR: portar Clarke + M²PC + ADALINE a C y medir con DWT (riesgo
   técnico mayor que queda; hoy el PLL usa 2.11 µs de 20 µs).
2. Verificar las 7 citas nuevas del bib; Cap. 6 y capítulo de hardware.

**Con fuente** (Vbus 12 V, límite 300–500 mA, motor en J7):
1. ~~Paradoja de corriente~~ **CERRADA (sesiones 18-19, 2026-10-13/14).** Tres capas: TRGO
   en el valle del PWM (corregido en jun con OC4REF), K ingenuo que ignoraba la atenuación
   4/7 de la red de la placa, y el offset de 128 mV que es la polarización por diseño.
   Reproducido en banco (RMS 1 → 8 → 23 raw para amp 0/170/510, lineal). Referencia Kelvin
   del PGA (`01011`) adoptada: eliminó la anomalía de fase C. **K calibrado con DMM:
   −32.93 ± 0.47 raw/A** (teórico 34.0, −3 %). Signo negativo = física del shunt en V0.
   FIELD_NOTES N1.23.
2. ~~Offset θ_e ↔ fase A + sentido~~ **HECHO (sesión 18, app `05_phase_align`)**:
   `PHASE_A_OFFSET_RAW = 2558` (mod 585, las 7 posiciones son equivalentes),
   `ENCODER_SIGN = +1`. `pwm/adc/openloop` promovidos a `lib/`; la ISR llama al control
   por hook (`adc_set_control_hook`).
2. Calibración de ganancia raw→A con multímetro (`CAL_TARGET_PHASE`).
3. ~~PLL + Clarke + Park en la ISR~~ **HECHO (sesión 19, app `06_openloop_sync`)**: i_dq DC
   (305 ± 25 / 4 ± 16 mA), lag constante 105 ± 2°, paso completo 4.74 µs. Destapó que el
   open-loop giraba en secuencia negativa desde mayo (corregido) y una prueba de sentido en
   05 que acertaba por casualidad (reescrita). `cordic` y `as5600_nb_tick` en `lib/`.
   FIELD_NOTES N1.24.
4. ~~Portar el FCS-M2PC~~ **HECHO Y EN LAZO CERRADO (sesión 20, app `07_m2pc`)**: 66
   candidatos a 50 kHz con retardo entero (UEV movida al pico), `overruns = 0`, paso 13 µs.
   PI de velocidad a 300 rpm ± 5 %. Logger a 1 kHz → **λ = 3.46 mWb identificado, BEMF casi
   sinusoidal (armónicos < 2.5 %)** → el ADALINE será "demostrativo" (regla del PLANNING).
5. ~~Recalibración de offset de corriente en línea~~ **HECHO (sesión 21, 2026-10-16/17)**:
   la mitad de la "deriva" era truncamiento (`uint16_t`; ahora Q8 + Clarke completo); la otra
   mitad no es térmica-monótona (±1 LSB entre arranques, ±2 LSB dentro de una corrida). El
   ADC dispara también en el **valle del PWM** (TIM1 CH6 interno + `MMS2`, `tim1_trgo2`), donde
   los shunts ven corriente cero: offset medido en continuo, filtrado en bloques de 2¹⁶.
   Validado contra el invariante ia+ib+ic = 0: de ±2 LSB de deriva a **+0.48 ± 0.38 LSB**.
   `paso_max` sin cambio (13.1 µs). FIELD_NOTES N1.26.
6. **Scopes en vivo (sesión 21b, 2026-10-18)**: `python_scripts/scope_m2pc.py` grafica los dos
   lazos desde el stream existente (ocho paneles, pausa, ventanas, replay de logs, graba
   todo). i_dq y errores son aproximados por el desfase del logger; el siguiente paso es
   telemetría sincronizada en la ISR (errores muestra a muestra + secuencia). Ver
   `python_scripts/README_scope.md`. Aún no probado con el motor girando.
7. **Sesión 22 (2026-10-18) — setup final y bug del CORDIC.** Llanta + base montadas: λ y
   BEMF sin cambio (3.476 mWb, armónicos ≤ 1.1 %), rizo de velocidad a la mitad por la
   inercia. Scope en vivo validado con el motor. La regresión de 06 destapó que **el CORDIC
   devolvía `sin ≡ cos` desde la sesión 20** (la escritura del CSR tras habilitar el reloj se
   perdía con `-O2` + prefetch; además dos lecturas seguidas a RDATA se adelantaban el 3 % de
   las veces). Fix en `lib/cordic`: relectura del CSR, resultados de 16 bits en una sola
   lectura (`RESSIZE=1`, `NRES=0`). 06: i_dq 323 ± 16 / 0.7 ± 7 mA. 07: i_q* medio 43 → 7.7,
   rizo 2·f_e → f_e. **Las cifras de lazo cerrado de las sesiones 20–21 no valen como datos**
   (la identificación de λ sí). Logs del banco propio ahora en `datos_banco/`. FIELD_NOTES
   N1.27. Regla: cada cambio en `lib/` → correr 06.
8. **Sesión 23 (2026-10-18) — Exp. 1 y 2 del subplan (PLANNING §11).** Telemetría dq en la
   ISR (línea `E`, Park muestra a muestra, RMS de error) + escalones de i_q* con PI congelado +
   captura a 50 kHz: M²PC sinusoidal responde en < 200 µs, RMS(ε_q) ≈ 37–40 mA sobre un rizo
   de conmutación de 225–334 mA pk‑pk (suelo físico V·Ts/L a 12 V), error de fase del modelo
   ≈ 2°. J ≈ 1.5e‑5 kg·m². Trayectoria senoidal ±100/300/600 rpm con inversión: RMS ≈ 10 rpm.
   Dos bugs más: FPU nunca habilitado (primer `float` del proyecto) y el **sector del M²PC
   elegido por la BEMF no puede frenar ni invertir** → ahora por la tensión deadbeat
   (`lib/m2pc.c`), lo que además eliminó el sesgo de i_q. Dataset del ADALINE grabado.
9. **Sesión 24 (2026-10-18) — Exp. 2b**: app `08_bemf_oc`, BEMF en circuito abierto con la red de BEMF
   de la placa (PA4/PC4/PB11) + 10 kΩ externa de +12 V a OUT1 (modo común 3.7 V, verificado): **THD
   0.42/0.67/0.64 %**, h3 ≤ 0.6 %, λ = 3.49 mWb, alineación del encoder **+2.85°** → `PHASE_A_OFFSET_RAW`
   2558 → 2563. La BEMF "vista por el controlador" (Exp. 3a) sobreestimaba h2–h6 2–5×: dead-time
   disfrazado de armónicos. A/B de la regla de sector a 600 rpm: la deadbeat metía ruido en el sector
   (pk-pk 667 mA); adoptada la de régimen `u_ss = e + R·i*` (pk-pk 377, RMS ε_q 44, invierte: ±300 rpm
   RMS 6.5 rpm). Reporte visual `docs/banco_2804.html` (BEMF animada con los coeficientes medidos).
10. **Exp. 3b — sesión 25 (PC)**: réplica Python (pesos Q27), tests C↔Python con UBSan, port `lib/adaline`
    integrado en 07 (OFF por defecto). **Sesión 26 (banco, dos tandas)**: eval plano (365 ciclos; con H=3,
    235) y presupuesto cumplido a 300 rpm; a 600 rpm falta margen (CCM-SRAM pendiente). Hallazgos:
    (a) el logger L a 1 kHz degradaba el baseline 15 % (55 → 46.7 mA con L decimado; afecta cifras s23/s24);
    (b) **el seguimiento del M²PC es insensible a errores de BEMF ≤20 %/10°** (ε_q 44 mA con modelo erróneo
    sin aprender): el piso es el rizo de conmutación, no la forma del modelo → ninguna forma aprendida
    puede mostrar mejora en este banco; (c) el LMS recupera λ en ~90 s en lazo cerrado (mecanismo ✓) pero
    converge a la tensión efectiva (secuencia negativa + h2 creciente) y a largo plazo degrada; μ=2e-3 era
    inestable (TRIP). Exp. 3c = tabla OFF′/ON-μ0/ON/E a 300 rpm con el argumento cuantitativo. Overrides
    `EXP_*` en CMake; `bench_run.sh`/`bench_summary.py`. Logs `datos_banco/m2pc_s26_*`. Banco en OFF sano.
10. Deuda conciliada en PROGRESO sesión 22: apps 02–06 con `tim1_trgo2` cerrado; glitch de
   fase C no reproducido; race de `adc_get_vbus_raw` y CCM-SRAM diferidos; 7 citas del `.bib`
   pendientes.
