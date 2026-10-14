/*
 * openloop.c — Implementación de excitación sinusoidal trifásica.
 *
 * Diseño:
 *   - theta Q32: overflow natural = vuelta eléctrica completa.
 *   - delta Q32: incremento por ISR. delta = f_e / f_s × 2^32.
 *   - LUT precomputada offline (Python: round(sin(2π i/256) × 32767)).
 *   - Desfase 120° / 240° por offset entero en el índice (85, 170).
 *   - Producto amplitude × LUT[idx] en int32 → >>15 → delta_duty Q0.
 *   - CCRx = ARR/2 + delta_duty. Clampeo implícito en uint16 (no clampeamos
 *     en software: la amplitud está limitada para que CCR siempre quede
 *     dentro de [0, ARR]).
 *
 * Aritmética 100% entera. ~80-100 ciclos en el handler.
 */

#include "openloop.h"
#include "pwm.h"
#include "stm32g431xx.h"

/* --------------------------------------------------------------------------
 * LUT Q15 — 256 entradas de sin(2π i / 256). Generada con Python:
 *   round(sin(2π × i / 256) × 32767) para i en 0..255.
 * Máximo +32767, mínimo -32767. Simetría: sin_lut[i+128] = -sin_lut[i].
 * -------------------------------------------------------------------------- */
static const int16_t sin_lut[256] = {
         0,    804,   1608,   2410,   3212,   4011,   4808,   5602,
      6393,   7179,   7962,   8739,   9512,  10278,  11039,  11793,
     12539,  13279,  14010,  14732,  15446,  16151,  16846,  17530,
     18204,  18868,  19519,  20159,  20787,  21403,  22005,  22594,
     23170,  23731,  24279,  24811,  25329,  25832,  26319,  26790,
     27245,  27683,  28105,  28510,  28898,  29268,  29621,  29956,
     30273,  30571,  30852,  31113,  31356,  31580,  31785,  31971,
     32137,  32285,  32412,  32521,  32609,  32678,  32728,  32757,
     32767,  32757,  32728,  32678,  32609,  32521,  32412,  32285,
     32137,  31971,  31785,  31580,  31356,  31113,  30852,  30571,
     30273,  29956,  29621,  29268,  28898,  28510,  28105,  27683,
     27245,  26790,  26319,  25832,  25329,  24811,  24279,  23731,
     23170,  22594,  22005,  21403,  20787,  20159,  19519,  18868,
     18204,  17530,  16846,  16151,  15446,  14732,  14010,  13279,
     12539,  11793,  11039,  10278,   9512,   8739,   7962,   7179,
      6393,   5602,   4808,   4011,   3212,   2410,   1608,    804,
         0,   -804,  -1608,  -2410,  -3212,  -4011,  -4808,  -5602,
     -6393,  -7179,  -7962,  -8739,  -9512, -10278, -11039, -11793,
    -12539, -13279, -14010, -14732, -15446, -16151, -16846, -17530,
    -18204, -18868, -19519, -20159, -20787, -21403, -22005, -22594,
    -23170, -23731, -24279, -24811, -25329, -25832, -26319, -26790,
    -27245, -27683, -28105, -28510, -28898, -29268, -29621, -29956,
    -30273, -30571, -30852, -31113, -31356, -31580, -31785, -31971,
    -32137, -32285, -32412, -32521, -32609, -32678, -32728, -32757,
    -32767, -32757, -32728, -32678, -32609, -32521, -32412, -32285,
    -32137, -31971, -31785, -31580, -31356, -31113, -30852, -30571,
    -30273, -29956, -29621, -29268, -28898, -28510, -28105, -27683,
    -27245, -26790, -26319, -25832, -25329, -24811, -24279, -23731,
    -23170, -22594, -22005, -21403, -20787, -20159, -19519, -18868,
    -18204, -17530, -16846, -16151, -15446, -14732, -14010, -13279,
    -12539, -11793, -11039, -10278,  -9512,  -8739,  -7962,  -7179,
     -6393,  -5602,  -4808,  -4011,  -3212,  -2410,  -1608,   -804
};

/* --------------------------------------------------------------------------
 * Estado de la excitación. Volatile porque la ISR escribe theta y main puede
 * leerlo / cambiar amplitude o delta.
 * -------------------------------------------------------------------------- */
volatile uint32_t g_theta_q32       = 0U;
static volatile uint32_t s_delta_q32 = 0U;
static volatile uint16_t s_amplitude = 0U;
static volatile uint8_t  s_running   = 0U;

void openloop_init(void) {
    g_theta_q32 = 0U;
    s_delta_q32 = 0U;
    s_amplitude = 0U;
    s_running   = 0U;
}

void openloop_set_amplitude(uint16_t amp_ticks) {
    s_amplitude = amp_ticks;
}

void openloop_set_freq_q32(uint32_t delta) {
    s_delta_q32 = delta;
}

void openloop_start(void) {
    g_theta_q32 = 0U;   /* arranca desde 0° eléctricos */
    s_running   = 1U;
}

void openloop_stop(void) {
    s_running = 0U;
    /* Restaurar duty 50% balanceado (sin par neto sobre el motor). */
    pwm_set_duties(PWM_ARR / 2U, PWM_ARR / 2U, PWM_ARR / 2U);
}

/* --------------------------------------------------------------------------
 * openloop_step — llamado desde la ISR del ADC, ~80-100 ciclos.
 *
 * Si no estamos corriendo, retorna en ~3 ciclos (early return).
 * -------------------------------------------------------------------------- */
void openloop_step(void) {
    if (s_running == 0U) {
        return;
    }

    g_theta_q32 += s_delta_q32;

    /* Top 8 bits → índice de LUT [0..255]. */
    uint8_t idx_a = (uint8_t)(g_theta_q32 >> 24);
    uint8_t idx_b = (uint8_t)(idx_a + 85U);    /* +120° = 256/3 ≈ 85.33 */
    uint8_t idx_c = (uint8_t)(idx_a + 170U);   /* +240° = 2 × 85 */

    int16_t s_a = sin_lut[idx_a];
    int16_t s_b = sin_lut[idx_b];
    int16_t s_c = sin_lut[idx_c];

    /* delta_duty = amplitude × sin_q15 / 2^15. amplitude ≤ PWM_ARR/2 evita
     * overflow del CCR (PWM_ARR/2 ± amplitude ∈ [0, PWM_ARR]). */
    int32_t d_a = ((int32_t)s_amplitude * s_a) >> 15;
    int32_t d_b = ((int32_t)s_amplitude * s_b) >> 15;
    int32_t d_c = ((int32_t)s_amplitude * s_c) >> 15;

    pwm_set_duties(
        (uint16_t)((int32_t)(PWM_ARR / 2U) + d_a),
        (uint16_t)((int32_t)(PWM_ARR / 2U) + d_b),
        (uint16_t)((int32_t)(PWM_ARR / 2U) + d_c)
    );
}
