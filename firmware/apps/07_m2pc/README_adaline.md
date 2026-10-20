# ADALINE experimental — Exp. 3b

**Port compilado y probado en PC; aún no validado en el motor.** No se ha
flasheado durante esta sesión. La opción predeterminada conserva la BEMF sinusoidal.

Desde la raíz del repositorio, builds separados para no confundir binarios:

```bash
cmake -S firmware -B /tmp/opencode/tesis-adaline-off -DM2PC_ADALINE=OFF
cmake --build /tmp/opencode/tesis-adaline-off --parallel 4
cmake -S firmware -B /tmp/opencode/tesis-adaline-on -DM2PC_ADALINE=ON
cmake --build /tmp/opencode/tesis-adaline-on --parallel 4
python3 -m unittest discover -s python_scripts -p test_adaline_replica.py
```

## Implementación

- `lib/adaline.{c,h}`: 7 armónicos, pesos Q27; base Q31; μ=131/65536.
- Inicialización sinusoidal, aprendizaje después de 3 s; puerta |λω|≥150 mV.
- LMS en foreground sobre medias de ventanas consecutivas de 1 ms. Al perder
  una ventana se renueva la corriente previa y **no** se actualiza ese paso.
- La ISR reutiliza sin/cos de Park y lee una instantánea completa de pesos
  mediante doble buffer; foreground nunca accede al CORDIC.
- ε normalizado limitado a ±4; cada peso a ±2 y norma L1 por eje ≤4.
  Son protecciones numéricas, no una garantía de estabilidad del lazo.
- `W t a1_alpha a1_beta b1_alpha b1_beta ... b7_beta`: 28 enteros Q27, 1 Hz.
- `[adaline] updates=... skipped=... gated=... update_wall_max_cycles=...`:
  contadores acumulados; diferencia entre reportes para medir updates/s.
  `gated` incluye velocidad baja y rechazo por norma. El tiempo DWT del
  foreground incluye las interrupciones del control.

El UART usa polling: no se asume que el foreground logre 1000 updates/s.
No retocar μ ni declarar tiempos de convergencia hasta medir esa cadencia.

### Validación de banco (sesión 26, 2026-10-20) — RESULTADO

Protocolo ejecutado: 06 ✓ → 07 OFF (ε_q 45 mA, overruns=0) → 07 ON. Tres hallazgos:

1. **Presupuesto ISR**: el eval con llamadas costaba 766 ciclos (flash) / 1109 (.data); la versión plana
   desenrollada cuesta **365 ciclos** y cumple: overruns=4, paso_max 17.85 µs. Colateral: corr/s 1960→1785.
2. **Cadencia del LMS**: el loop foreground solo sostiene ~730 ventanas/s con L a 1 kHz (printf). Con L
   decimado a 10 Hz: **updates ≈ 230/s** (la ISR de control ocupa ~80 % de CPU). μ efectivo ~4× menor.
3. **Estabilidad (lo importante)**: μ=2e-3 → inestable en lazo (ε_q 73→138 mA en 25 s, TRIP por 1.2 A).
   μ=1e-4 → estable pero **peor que el baseline** (ε_q ~102 vs 45 mA): W aprende la distorsión del inversor
   (h2 α 0→17.6 % en 100 s), no la BEMF (THD real 0.4-0.7 %), y meterla al predictor daña el sector.
   **Conclusión: con BEMF casi sinusoidal, el modelo sinusoidal es ya el mejor predictor.**

**Segunda tanda (controles)**, 300 rpm constante, H=3, L decimado en ambos builds: OFF′ 46.7/43.1 mA;
ON μ=0 48.0/41.0; ON μ=1e-4 44.8/39.3; **modelo erróneo (λ·0.8, +10°) sin aprender 44.1/39.7**; ídem con
aprendizaje recupera λ en ~90 s pero deriva (h2 8 % a 180 s, ε_q 46→55). Conclusión: el seguimiento es
insensible a la forma de la BEMF a esta escala (G·Δe ≪ rizo de conmutación); el online no puede mostrar
mejora y, sin compensar dead-time/asimetría, deriva. PROGRESO sesión 26 §4.

### Overrides de experimento (CMake, sin editar fuentes)

`-DEXP_ADALINE_MU_Q16=`, `EXP_ADALINE_H_HARM`, `EXP_ADALINE_EPK_MIN_MV`, `EXP_ADALINE_SECTOR_MODEL` (1: la
sinusoide decide el sector), `EXP_ADALINE_INIT_AMP` / `EXP_ADALINE_INIT_PHI_DEG` (modelo inicial erróneo),
`EXP_TRAJ_ENABLE`, `EXP_W_REF_RPM`, `EXP_LOG_L_DECIM` (**usar el mismo en OFF y ON**). Flujo:
`python_scripts/bench_run.sh <elf> <s> <log>` flashea, captura y resume (`bench_summary.py`).

## Validación de banco pendiente (histórico, pre-s26)

1. Retirar la resistencia de polarización de `08_bemf_oc`. Montaje de control
   habitual, fuente 12 V limitada a 500 mA y apagado accesible.
2. Correr `06_openloop_sync`: autotest CORDIC e i_dq DC (regla de regresión).
3. Correr `07` con ADALINE **OFF**; capturar baseline del mismo binario/configuración.
4. Correr `07` con ADALINE **ON**: comprobar `overruns=0`, `paso_max<20 µs`,
   ausencia de trip y crecimiento de `updates`. Si no cumple, detener la prueba;
   revisar coste de evaluación y bloqueo del UART antes de comparar desempeño.
5. Tras validar tiempos y convergencia, repetir Exp. 1 a 300/600 rpm con
   idénticos ajustes de trayectoria/escalones en ambos builds. La configuración
   actual sigue siendo trayectoria ±300 rpm; no es aún esa comparativa.

## Alcance de los resultados offline

La prueba host compila el mismo C con UBSan y contrasta pesos con Python.
La identidad bit a bit usa las mismas entradas trigonométricas cuantizadas;
**no emula exactamente el CORDIC del STM32** ni prueba tiempos del target.

El observador aprende tensión efectiva (motor + inversor + errores del modelo).
No separa dead-time/DC, ni corrige la atenuación sinc de las medias de 1 ms.
Los logs anteriores al offset 2563 tienen otra referencia angular. No usar
sus armónicos como BEMF física ni como evidencia de reducción de rizo.
La Exp. 3c debe comparar RMS ε_q/ε_d y rizo medidos en la ISR, no el residuo offline.
