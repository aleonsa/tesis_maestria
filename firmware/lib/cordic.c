#include "cordic.h"

void cordic_init(void) {
    RCC->AHB1ENR |= RCC_AHB1ENR_CORDICEN;
    (void)RCC->AHB1ENR;

    /* FUNC = 0 (coseno: entrega cos y sin). PRECISION = 6 → 24 iteraciones:
     * error residual < 2^-18 (RM0440 Tabla 103). SCALE = 0. NARGS = 0 (solo
     * el ángulo; módulo implícito = 1). ARGSIZE = 0 (ángulo de 32 bits).
     * RESSIZE = 1: los dos resultados en 16 bits, empaquetados en UNA
     * palabra. NRES = 0 obligatorio con RESSIZE = 1: NRES cuenta lecturas
     * de 32 bits, no resultados; con NRES = 1 el periférico espera dos
     * lecturas por cálculo y, haciendo una, cada lectura devuelve el
     * resultado del cálculo ANTERIOR (RRDY se queda en 1). Medido en 06. */
    const uint32_t csr = (0U << CORDIC_CSR_FUNC_Pos)
                       | (6U << CORDIC_CSR_PRECISION_Pos)
                       | CORDIC_CSR_RESSIZE;

    /* 2026-10-18 (sesión 22): la escritura a CSR justo tras habilitar el
     * reloj SE PERDÍA (CSR seguía en su reset 0x50: NRES=0) con la lib en
     * -O2 y el prefetch de flash activo; con -Og y sin prefetch los ciclos
     * de fetch entre instrucciones tapaban el hueco. Con NRES=0 la segunda
     * lectura de RDATA devuelve otra vez el coseno: sin ≡ cos, y Park sale
     * pulsante a 2·f_e (06 lo destapó; 07 corrió así en las sesiones 20-21).
     * RM0440 §7.2.17: tras habilitar el reloj de un periférico hay que
     * esperar antes de acceder a sus registros. Barrera + relectura hasta
     * que el valor quede, para que nunca vuelva a fallar en silencio. */
    __DSB();
    do {
        CORDIC->CSR = csr;
        __DSB();
    } while ((CORDIC->CSR & ~CORDIC_CSR_RRDY_Msk) != csr);
}
