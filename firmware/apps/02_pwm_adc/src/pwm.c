/*
 * pwm.c — STUBS. Implementación real en sesiones siguientes.
 *
 * Plan de implementación (FIELD_NOTES.md N1.3–N1.6):
 *
 *   1. Habilitar clocks (RCC):
 *        - GPIOA, GPIOB, GPIOC (para los 6 pines TIM1)
 *        - TIM1 (APB2)
 *
 *   2. Configurar los 6 pines en AF6:
 *        - PA8  (CH1),  PA9  (CH2),  PA10 (CH3)
 *        - PC13 (CH1N), PA12 (CH2N), PB15 (CH3N)
 *
 *   3. TIM1 base:
 *        - CR1.CMS    = 01     (center-aligned mode 1)
 *        - CR1.ARPE   = 1      (preload de ARR)
 *        - ARR        = 2833   (30 kHz @ 170 MHz)
 *        - RCR        = 1      (1 update event por periodo PWM)
 *
 *   4. Por canal x ∈ {1,2,3}:
 *        - CCMRx.OCxM  = 110   (PWM mode 1)
 *        - CCMRx.OCxPE = 1     (preload de CCRx)
 *        - CCER.CCxE   = 1     (enable salida principal)
 *        - CCER.CCxNE  = 1     (enable complementario)
 *        - CCER.CCxP   = 0     (active high, default)
 *        - CCER.CCxNP  = 0     (active high, default)
 *        - CCRx        = ARR/2 (duty 50% como valor inicial seguro)
 *
 *   5. Dead-time + master enable (BDTR):
 *        - BDTR.DTG    = 0x55  (500 ns)
 *        - BDTR.MOE    = 0     (todavía no — pwm_enable() lo activa)
 *
 *   6. TRGO al ADC:
 *        - CR2.MMS     = 010   (update event como TRGO)
 *
 *   7. Output Idle State (seguridad):
 *        - CR2.OISx    = 0     (high-side inactivo si MOE=0)
 *        - CR2.OISxN   = 0     (low-side inactivo si MOE=0)
 *
 *   8. Cargar shadow registers:
 *        - EGR.UG      = 1     (force update event manual)
 *
 *   9. Arrancar contador (en pwm_enable()):
 *        - BDTR.MOE    = 1
 *        - CR1.CEN     = 1
 */

#include "pwm.h"

void pwm_init(void) {
    /* TODO: implementar según el plan documentado arriba. */
}

void pwm_set_duties(uint16_t duty_a, uint16_t duty_b, uint16_t duty_c) {
    /* TODO: TIM1->CCR1 = duty_a; CCR2 = duty_b; CCR3 = duty_c; */
    (void)duty_a;
    (void)duty_b;
    (void)duty_c;
}

void pwm_enable(void) {
    /* TODO: TIM1->BDTR |= MOE; TIM1->CR1 |= CEN; */
}

void pwm_disable(void) {
    /* TODO: TIM1->CR1 &= ~CEN; TIM1->BDTR &= ~MOE; */
}
