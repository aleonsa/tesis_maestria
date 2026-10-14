#ifndef CORDIC_H
#define CORDIC_H

#include <stdint.h>
#include "stm32g431xx.h"

/*
 * cordic.h — sin/cos por hardware (periférico CORDIC del STM32G4).
 *
 * Una rotación de Park necesita sin(θ) y cos(θ) cada 20 µs. Las opciones:
 *   - libm sinf/cosf:    ~150-300 ciclos cada una, float.
 *   - LUT de 256 Q15:    ~10 ciclos, pero 1.4° de resolución (openloop la usa
 *                        para generar PWM, donde basta).
 *   - CORDIC:            ~24 ciclos los dos juntos, 32 bits de entrada, error
 *                        < 2^-18, cero RAM. AN5325.
 *
 * Convención de ángulo: el CORDIC toma Q31 en [-1, 1) = [-π, π). Nuestro θ
 * es Q32 sin signo en [0, 2^32) = [0, 2π). Reinterpretar los mismos 32 bits
 * como signed mapea [π, 2π) → [-π, 0): es la misma vuelta. Cero costo.
 *
 * Modo: polling. Leer RDATA antes de que el resultado esté listo detiene el
 * bus AHB hasta que lo esté (RM0440 §17.3.5), así que no hay flag que mirar:
 * escribir el argumento y leer dos veces.
 */

/* Habilita el reloj y configura: coseno, 2 resultados (cos, sin), 1
 * argumento (módulo = 1), 24 iteraciones, 32 bits. */
void cordic_init(void);

/* sin y cos de θ (Q32, una vuelta = 2^32), en Q31 [-1, 1). */
static inline void cordic_sincos_q31(uint32_t theta_q32, int32_t *sin_q31,
                                     int32_t *cos_q31) {
    CORDIC->WDATA = theta_q32;
    *cos_q31 = (int32_t)CORDIC->RDATA;   /* primer resultado: cos */
    *sin_q31 = (int32_t)CORDIC->RDATA;   /* segundo: sin           */
}

#endif /* CORDIC_H */
