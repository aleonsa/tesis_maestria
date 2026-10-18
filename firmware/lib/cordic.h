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
 * bus AHB hasta que lo esté (RM0440 §17.3.5), así que no hay flag que mirar.
 *
 * Resultados de 16 bits (RESSIZE = 1), los dos en UNA lectura de RDATA:
 * [15:0] = cos, [31:16] = sin, ambos q1.15. Sesión 22 (2026-10-18): con
 * resultados de 32 bits y dos lecturas consecutivas, la segunda devolvía
 * otra vez el coseno en ~3 % de los pasos (la lectura llegaba antes de que
 * el periférico cargara el segundo resultado): sin ≡ cos en esos pasos y
 * ruido en i_dq. Una sola lectura no tiene ese race (es lo que hace el
 * MCSDK de ST). Precisión q1.15 = 3·10⁻⁵, irrelevante para corrientes de
 * decenas de LSB y para λω·sin θ. La API sigue entregando Q31 (los 16 bits
 * bajos en cero) para no tocar a los usuarios.
 */

/* Habilita el reloj y configura: coseno, cos y sin de 16 bits en una lectura,
 * 1 argumento de 32 bits (módulo = 1), 24 iteraciones. Verifica por
 * relectura que la configuración quedó escrita. */
void cordic_init(void);

/* sin y cos de θ (Q32, una vuelta = 2^32), en Q31 [-1, 1). */
static inline void cordic_sincos_q31(uint32_t theta_q32, int32_t *sin_q31,
                                     int32_t *cos_q31) {
    CORDIC->WDATA = theta_q32;
    const uint32_t r = CORDIC->RDATA;        /* [31:16] sin, [15:0] cos (q1.15) */
    *cos_q31 = (int32_t)(r << 16);
    *sin_q31 = (int32_t)(r & 0xFFFF0000U);
}

#endif /* CORDIC_H */
