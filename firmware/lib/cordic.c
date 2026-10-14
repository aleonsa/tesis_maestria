#include "cordic.h"

void cordic_init(void) {
    RCC->AHB1ENR |= RCC_AHB1ENR_CORDICEN;
    (void)RCC->AHB1ENR;

    /* FUNC = 0 (coseno; con NRES=1 entrega cos y luego sin).
     * PRECISION = 6 → 24 iteraciones: error residual < 2^-18 (RM0440 Tabla
     * 103), de sobra para Q15 de corriente. SCALE = 0. NARGS = 0 (solo el
     * ángulo; módulo implícito = 1). ARGSIZE = RESSIZE = 0 (32 bits). */
    CORDIC->CSR = (0U << CORDIC_CSR_FUNC_Pos)
                | (6U << CORDIC_CSR_PRECISION_Pos)
                | CORDIC_CSR_NRES;
}
