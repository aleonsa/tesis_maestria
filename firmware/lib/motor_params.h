#ifndef MOTOR_PARAMS_H
#define MOTOR_PARAMS_H

#include <stdint.h>

/*
 * motor_params.h — parámetros del banco (motor 2804 + B-G431B-ESC1) y las
 * constantes de punto fijo que se derivan de ellos.
 *
 * Un solo sitio para todos los números físicos. Las constantes derivadas
 * están escritas como expresiones que el compilador pliega: la derivación
 * queda a la vista y no hay números mágicos.
 *
 * UNIDADES INTERNAS DEL CONTROL
 *   corriente  : LSB del ADC (int32). 1 LSB = 1/K_ADC A ≈ 30.4 mA.
 *   tensión    : mV (int32).
 *   ángulo     : Q32 (2^32 = una vuelta eléctrica).
 *   velocidad  : ω̂ del PLL, Q32 vueltas MECÁNICAS por tick de 20 µs.
 *   tiempo     : ticks de la ISR (Ts = 20 µs).
 *
 * Elegir LSB y mV como unidades evita toda conversión en el camino caliente:
 * las medidas entran como vienen del ADC y los duties salen en ticks.
 */

/* ── Motor 2804 ──────────────────────────────────────────────────────── */
#define MOTOR_POLE_PAIRS     7U
#define MOTOR_R_OHM          2.7      /* fase, medido (sesión 12: 5.4 Ω línea-línea en Y) */
#define MOTOR_L_H            0.86e-3  /* fase, PLANNING §2 — pendiente identificación Fase 2 */
#define MOTOR_LAMBDA_WB      0.0035   /* flujo del imán, PLANNING §2 — pendiente Fase 2.
                                       * Coherente con el banco: a 86 rpm da 0.22 V de BEMF,
                                       * y el fasor de 06_openloop_sync (tensión 15° adelante
                                       * del rotor) cuadra con ese valor. */

/* ── Inversor y muestreo ─────────────────────────────────────────────── */
#define VBUS_MV              12000    /* nominal; TODO leer del ADC (race N1.15 pendiente) */
#define TS_S                 20e-6    /* 50 kHz */
#define FS_HZ                50000.0

/* ── Cadena de medida ────────────────────────────────────────────────── */
#define ADC_K_RAW_PER_A      32.93    /* sesión 19, DMM: −32.93 ± 0.47; el signo ya lo
                                       * absorbe Clarke (offset − raw) */

/* ── Constantes derivadas, punto fijo ────────────────────────────────── */

/* Ganancia del modelo de predicción: Δi [LSB] = G · v [mV] · (t/Ts)
 *   G = (Ts / L) · K / 1000  [LSB/mV por Ts]  ≈ 7.66e-4  → Q20 ≈ 803 */
#define M2PC_G_Q20 \
    ((int32_t)((TS_S / MOTOR_L_H) * ADC_K_RAW_PER_A / 1000.0 * 1048576.0 + 0.5))

/* Caída resistiva: v_R [mV] = R · i = R · 1000 / K · i[LSB]  ≈ 82 mV/LSB → Q8 ≈ 20990 */
#define M2PC_RK_Q8 \
    ((int32_t)(MOTOR_R_OHM * 1000.0 / ADC_K_RAW_PER_A * 256.0 + 0.5))

/* BEMF de pico [mV] por unidad de ω̂ (Q32 rev_mec/tick):
 *   e = λ · ω_e = λ · p · (ω̂ · 2π · f_s / 2^32)   → mV/unidad ≈ 1.79e-3 → Q18 ≈ 470
 *   A 86 rpm (ω̂ ≈ 123 000) → 220 mV. Máx ω̂ ~1e6 (700 rpm) × 470 < 2^31: cabe en int32. */
#define KE_MV_Q18 \
    ((int32_t)(MOTOR_LAMBDA_WB * MOTOR_POLE_PAIRS * 6.283185307 * FS_HZ * 1000.0 \
               / 4294967296.0 * 262144.0 + 0.5))

#endif /* MOTOR_PARAMS_H */
