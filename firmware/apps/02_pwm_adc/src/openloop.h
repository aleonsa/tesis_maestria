/*
 * openloop.h — Excitación trifásica sinusoidal open-loop (sin lazo cerrado).
 *
 * Genera 3 sinusoides desfasadas 120° y las aplica a CCR1/2/3 del TIM1.
 * Pensado como bring-up para mover el motor por primera vez, validar el
 * orden de fases, y verificar que las corrientes medidas siguen las
 * consignas.
 *
 * Theta es Q32 (un overflow = una vuelta eléctrica). La acumulación
 * theta += delta cada ISR no acumula error de redondeo: el overflow del
 * uint32_t implementa naturalmente el módulo 2π.
 *
 * LUT de 256 valores Q15. El índice viene de los 8 bits altos de theta.
 * Multiplicaciones int32 × int16 >> 15 son ~3 ciclos en M4.
 *
 * Aritmética 100% entera. Float solo está en este header como comentario
 * para documentar las constantes.
 *
 * Uso:
 *   openloop_init();
 *   openloop_set_amplitude(170U);        // 170 ticks = 10% de ARR=1700
 *   openloop_set_freq_q32(171799U);      // 2 Hz eléctrico a 50 kHz ISR
 *   openloop_start();
 *   ...
 *   openloop_step();                     // llamado desde el handler ADC
 */

#ifndef OPENLOOP_H
#define OPENLOOP_H

#include <stdint.h>

/* Constante de conversión Hz_eléctrico → delta_q32:
 *   delta_q32 = round(f_e / f_s × 2^32)
 *
 * Para f_s = 50 kHz (fijo, ISR del ADC):
 *   1 Hz eléctrico → 85899 (≈ 0.5e5 / 5e4 × 4.3e9)
 *   2 Hz eléctrico → 171799
 *   5 Hz eléctrico → 429497
 *   10 Hz eléctrico → 858993
 *
 * Para BLDC 2804 con P=8 (4 pole pairs):
 *   f_mecánico = f_eléctrico / 4
 *   2 Hz eléctrico → 0.5 Hz mecánico = 30 RPM mecánico = una vuelta cada 2 s
 */
#define OPENLOOP_DELTA_1HZ_ELEC   85899U
#define OPENLOOP_DELTA_2HZ_ELEC   171799U

void openloop_init(void);

/* Setters runtime — escriben a globals volátiles. */
void openloop_set_amplitude(uint16_t amp_ticks);
void openloop_set_freq_q32(uint32_t delta);

/* Control de habilitación. stop() restaura duty 50% balanceado. */
void openloop_start(void);
void openloop_stop(void);

/* Llamado desde la ISR del ADC. Cuando running=0, retorna inmediato. */
void openloop_step(void);

/* Para debugging desde main: estado actual del ángulo eléctrico. */
extern volatile uint32_t g_theta_q32;

#endif /* OPENLOOP_H */
