/*
 * theta_pll.h — Observador PLL tipo II para θ/ω del rotor, en punto fijo.
 *
 * POR QUÉ EXISTE
 *
 * La ISR de control corre a 50 kHz (Ts = 20 µs), pero el AS5600 por I2C1
 * entrega una muestra nueva cada ~468 µs (2 139 lecturas/s medidos en banco
 * el 2026-10-11). Es decir, en ~23 de cada 24 ciclos de control NO hay dato
 * fresco. Sostener la última lectura ("zero-order hold") metería θ a saltos
 * al lazo, y derivarla para obtener ω amplifica el ruido de cuantización a
 * niveles inútiles (~24 rad/s, FIELD_NOTES N1.18).
 *
 * La solución es un observador de tracking de dos estados (PLL tipo II,
 * equivalente al filtro α-β / Kalman estacionario del modelo de velocidad
 * constante):
 *
 *   - Cada tick de la ISR:  extrapolación pura   θ̂ += ω̂
 *   - Cada muestra nueva:   corrección            e  = wrap(θ_med − θ̂)
 *                                               θ̂ += Kp·e
 *                                               ω̂ += Ki·e
 *
 * La entrada nominal es una RAMPA (θ crece lineal a velocidad constante) y el
 * tipo II es el observador mínimo con error estacionario cero ante rampa. La
 * discusión contra alternativas (tipo III, EKF, SMO) está en PROGRESO_HARDWARE
 * sesión 16.
 *
 * CONVENCIONES (coherentes con openloop.h)
 *
 *   - θ̂ en Q32: 2^32 = 1 vuelta MECÁNICA. El overflow del uint32 implementa
 *     el mod 2π gratis; la resta unsigned + reinterpretación signed da el
 *     error envuelto en ±π (±2^31) sin una sola rama.
 *   - ω̂ en Q32 signado: vueltas mecánicas por tick de la ISR (20 µs).
 *   - θ eléctrico = P · θ̂_mecánico, con P = 7 pares de polos del 2804: la
 *     multiplicación uint32 desborda y eso ES el mod 2π eléctrico.
 *
 * GANANCIAS (dominio discreto, corrección al ritmo de llegada de muestras)
 *
 *   Ancho de banda ωn = 2π·100 rad/s, amortiguamiento ζ = 1 (crítico):
 *     Tm ≈ 468 µs (periodo nominal entre muestras del bus)
 *     Kp = 2ζ·ωn·Tm        = 0.5881      (adimensional, sobre θ̂)
 *     Ki = ωn²·Tm·Ts       = 3.6952e-3   (sobre ω̂ por tick de Ts = 20 µs)
 *
 *   ωn·Tm = 0.29 ≪ 1: la discretización de Euler es inocua a este ancho de
 *   banda. 100 Hz sigue holgado cualquier aceleración que la mecánica del
 *   2804 puede producir entre muestras y filtra la cuantización de 12 bits.
 *
 * COMPENSACIÓN DE LATENCIA
 *
 *   La muestra describe al rotor ~600 µs en el pasado (~450 µs de transacción
 *   I2C + ~150 µs de refresco interno del AS5600). En vez de un adelanto
 *   constante en cuentas —que solo es correcto a UNA velocidad— se adelanta
 *   proporcional a ω̂: θ_salida = θ̂_e + P·ω̂·N_LAT ticks. Sin este adelanto el
 *   ADALINE aprendería una BEMF rotada ~19° eléctricos a 80 rad/s mecánicos,
 *   indistinguible de un offset de fase A mal puesto.
 *
 * USO
 *
 *   uint16_t raw0;  as5600_raw_angle(&raw0);       // lectura bloqueante al boot
 *   theta_pll_init((uint32_t)raw0 << 20);
 *   // en la ISR de control (50 kHz):
 *   theta_pll_extrapolate();
 *   if (hay muestra nueva) theta_pll_update(raw);
 *   uint32_t th_e = theta_pll_theta_elec_comp_q32();
 */

#ifndef THETA_PLL_H
#define THETA_PLL_H

#include <stdint.h>

/* Pares de polos (NO polos) del BLDC 2804 del banco: 12N14P → 14 imanes →
 * 7 pares. Especificación del fabricante y medición en banco (sesión 12) de
 * acuerdo. ⚠ No confundir con el Anaheim BLY-344S de la simulación MATLAB
 * (4 pares; allí `p.P = 8` cuenta POLOS). Ver PLANNING_HARDWARE §2. */
#define THETA_PLL_POLE_PAIRS   7U

/* Frecuencia de la ISR que llama a theta_pll_extrapolate(). */
#define THETA_PLL_FS_HZ        50000U

/* Adelanto por latencia en ticks de ISR: 600 µs / 20 µs = 30. */
#define THETA_PLL_LAT_TICKS    30U

/* Inicializa θ̂ = theta0_q32, ω̂ = 0. Llamar con una lectura real del encoder. */
void theta_pll_init(uint32_t theta0_q32);

/* Extrapolación: θ̂ += ω̂. Una llamada por tick de la ISR, siempre. */
void theta_pll_extrapolate(void);

/* Corrección con una muestra nueva del AS5600 (RAW ANGLE, 0..4095). */
void theta_pll_update(uint16_t raw12);

/* θ̂ mecánico, Q32 (2^32 = 1 vuelta mecánica). */
uint32_t theta_pll_theta_mech_q32(void);

/* θ̂ eléctrico, Q32: P·θ̂_mec, el overflow hace el mod 2π eléctrico. */
uint32_t theta_pll_theta_elec_q32(void);

/* θ̂ eléctrico con adelanto por latencia: el que debe consumir el control. */
uint32_t theta_pll_theta_elec_comp_q32(void);

/* ω̂ en Q32 signado: vueltas mecánicas por tick de ISR (20 µs). */
int32_t theta_pll_omega_tick_q32(void);

#endif /* THETA_PLL_H */
