/*
 * theta_pll.c — implementación del observador. Ver theta_pll.h para el diseño.
 *
 * Todo entero, sin float en runtime: las ganancias son constantes Q30
 * plegadas por el compilador. Los productos caben en int64 de sobra
 * (|e| ≤ 2^31, ganancias < 2^30 → producto < 2^61).
 */

#include "theta_pll.h"

/* Ganancias Q30 (ver derivación en el header):
 *   Kp = 2ζ·ωn·Tm = 0.5881
 *   Ki = ωn²·Tm·Ts = 3.6952e-3
 * Expresadas como constantes flotantes plegadas en compilación: el Cortex-M4
 * jamás ejecuta un float por ellas. */
#define PLL_KP_Q30   ((int32_t)(0.58810615 * 1073741824.0))   /* 631 509 293 aprox */
#define PLL_KI_Q30   ((int32_t)(3.69518e-3 * 1073741824.0))   /* 3 967 849 aprox   */

/* Estados. Volátiles: los escribe la ISR y los lee el foreground. Cada uno es
 * de 32 bits → lectura atómica en Cortex-M; no hay secciones críticas porque
 * el foreground solo los observa para imprimir. */
static volatile uint32_t s_theta_q32;   /* vueltas mecánicas, Q32            */
static volatile int32_t  s_omega_q32;   /* vueltas mecánicas por tick, Q32   */

void theta_pll_init(uint32_t theta0_q32) {
    s_theta_q32 = theta0_q32;
    s_omega_q32 = 0;
}

void theta_pll_extrapolate(void) {
    s_theta_q32 += (uint32_t)s_omega_q32;
}

void theta_pll_update(uint16_t raw12) {
    /* Muestra a Q32: 12 bits significativos a la cabeza. */
    uint32_t meas_q32 = (uint32_t)raw12 << 20;

    /* Error envuelto en ±π sin ramas: la resta unsigned desborda y el cast a
     * int32 la reinterpreta en [-2^31, 2^31). Correcto incluso cruzando el
     * wrap 4095→0: es toda la razón de ser de la convención Q32. */
    int32_t e = (int32_t)(meas_q32 - s_theta_q32);

    s_theta_q32 += (uint32_t)(int32_t)(((int64_t)e * PLL_KP_Q30) >> 30);
    s_omega_q32 += (int32_t)(((int64_t)e * PLL_KI_Q30) >> 30);
}

uint32_t theta_pll_theta_mech_q32(void) {
    return s_theta_q32;
}

uint32_t theta_pll_theta_elec_q32(void) {
    return s_theta_q32 * (uint32_t)THETA_PLL_POLE_PAIRS;
}

uint32_t theta_pll_theta_elec_comp_q32(void) {
    /* Adelanto P·ω̂·N_LAT. En el peor régimen del banco (~50 rev/s mec) ω̂ por
     * tick es ~4.3e6 (Q32) → producto ≈ 9e8, cómodo en 32 bits. El adelanto
     * se aplica en el dominio eléctrico: P multiplica al adelanto mecánico. */
    int32_t adv_mec = s_omega_q32 * (int32_t)THETA_PLL_LAT_TICKS;
    return s_theta_q32 * (uint32_t)THETA_PLL_POLE_PAIRS
         + (uint32_t)adv_mec * (uint32_t)THETA_PLL_POLE_PAIRS;
}

int32_t theta_pll_omega_tick_q32(void) {
    return s_omega_q32;
}
