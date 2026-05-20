/*
 * pwm.h — API pública del módulo PWM (TIM1) para FCS-M2PC.
 *
 * Configuración fija de la app:
 *   - 3 fases complementarias (6 salidas físicas)
 *   - center-aligned mode 1 (CMS = 01)
 *   - f_PWM = 30.0025 kHz (ARR = 2833 @ HCLK 170 MHz)
 *   - dead-time = 500 ns (DTG = 0x55 @ CKD = 00)
 *   - TRGO = update event (MMS = 010), RCR = 1 → 1 trigger por periodo PWM
 *
 * Referencias: ver FIELD_NOTES.md N1.3–N1.6.
 */

#ifndef PWM_H
#define PWM_H

#include <stdint.h>

/* ARR para 30 kHz con HCLK = 170 MHz en center-aligned (2*ARR ciclos por periodo). */
#define PWM_ARR        (2833U)

/* Encoding del DTG para 500 ns con CKD = 00 (t_DTS = 5.88 ns).
 * 500 / 5.88 ≈ 85 = 0x55. Bit 7 = 0 → primer rango lineal (DT = DTG * t_DTS). */
#define PWM_DEAD_DTG   (0x55U)

/*
 * pwm_init — configura GPIOs (AF6), TIM1, dead-time y TRGO.
 *
 * Después de esta llamada, las 6 salidas están armadas pero **inhabilitadas**
 * (MOE = 0). Llamar a pwm_enable() para arrancar la conmutación.
 *
 * Pre-condiciones:
 *   - clock_init_170mhz_hse() ya ejecutado.
 *
 * STUB por ahora — implementación viene en el siguiente paso.
 */
void pwm_init(void);

/*
 * pwm_set_duties — escribe los 3 duty cycles (uno por fase).
 *
 * Cada duty es un valor entero en [0, PWM_ARR]. El nuevo valor se aplica
 * en el siguiente update event (preload buffer).
 *
 * STUB por ahora.
 */
void pwm_set_duties(uint16_t duty_a, uint16_t duty_b, uint16_t duty_c);

/*
 * pwm_enable — habilita las 6 salidas (MOE = 1, CEN = 1).
 *
 * STUB por ahora.
 */
void pwm_enable(void);

/*
 * pwm_disable — apaga las salidas y detiene el contador.
 *
 * Equivalente a un "soft break": MOE = 0, CEN = 0. Las salidas van a su
 * estado OISx/OISxN configurado (todas inactivas por defecto).
 *
 * STUB por ahora.
 */
void pwm_disable(void);

#endif /* PWM_H */
