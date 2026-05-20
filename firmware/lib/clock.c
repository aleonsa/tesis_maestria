/*
 * clock.c — Configuración del PLL del STM32G431 a 170 MHz.
 *
 * Fuente:   HSE 8 MHz (cristal Y2 de la B-G431B-ESC1, R27 = 220 Ω damping)
 * Cadena:   f_PLL_IN = 8/M = 8/2 = 4 MHz     (∈ [2.66, 16])
 *           f_VCO    = N × f_PLL_IN = 85 × 4 = 340 MHz  (∈ [96, 344])
 *           SYSCLK   = f_VCO / R = 340 / 2 = 170 MHz
 *
 * Cross-ref: RM0440 §6 (PWR), §7 (RCC), §3.3.3 (Flash latency).
 */

#include "stm32g431xx.h"
#include "clock.h"

void clock_init_170mhz_hse(void) {
    /*
     * Paso 1 — Habilitar el clock del periférico PWR.
     * El registro PWR_CR5 está apagado por defecto al reset (clock gating).
     * Sin esto, escrituras a PWR no tienen efecto.
     * Bit 28 de RCC_APB1ENR1 = PWREN.
     */
    RCC->APB1ENR1 |= RCC_APB1ENR1_PWREN;

    /*
     * Paso 2 — Entrar a Range 1 Boost (modo > 150 MHz).
     * R1MODE = 0 ⇒ Boost. R1MODE = 1 ⇒ Normal (default tras reset).
     * Sin Boost, el regulador interno no soporta SYSCLK > 150 MHz.
     * RM0440 §6.1.5.
     */
    PWR->CR5 &= ~PWR_CR5_R1MODE;

    /*
     * Paso 3 — Subir Flash wait states ANTES de subir el reloj.
     * A 170 MHz, la Flash necesita 4 WS para servir instrucciones sin corrupción.
     * RM0440 Table 9 (Number of wait states vs CPU clock frequency).
     * Si subimos SYSCLK con WS insuficientes, el CPU lee basura → cuelgue.
     */
    FLASH->ACR = (FLASH->ACR & ~FLASH_ACR_LATENCY) | FLASH_ACR_LATENCY_4WS;

    /*
     * Paso 4 — Encender HSE y esperar a que se estabilice.
     * El cristal arranca con oscilación incipiente; HSERDY se pone en 1
     * cuando la amplitud y la fase son confiables (~ms con cristal de 8 MHz).
     * Si HSERDY nunca aparece, este loop atrapa la ejecución para siempre
     * (señal clara de que el cristal o R27 están mal).
     */
    RCC->CR |= RCC_CR_HSEON;
    while ((RCC->CR & RCC_CR_HSERDY) == 0U) { }

    /*
     * Paso 5 — Configurar el PLL (con PLLON = 0 todavía).
     * PLLCFGR es read/write solo cuando el PLL está apagado (RM0440 §7.4.4).
     *
     * Encoding (verificado en stm32g431xx.h):
     *   PLLSRC = 0b11   → HSE como fuente            (RCC_PLLCFGR_PLLSRC_HSE)
     *   PLLM   = M - 1  → M=2 → PLLM = 1
     *   PLLN   = N      → N=85 directo (encoding sin offset)
     *   PLLR   = 0b00   → R=2 (encoding: 00=2, 01=4, 10=6, 11=8) ⇒ no escribir nada
     *   PLLREN = 1      → habilita la salida R hacia SYSCLK
     *
     * Escribimos PLLCFGR completo en una operación atómica para evitar
     * estados intermedios indefinidos.
     */
    RCC->PLLCFGR = RCC_PLLCFGR_PLLSRC_HSE
                 | (1U  << RCC_PLLCFGR_PLLM_Pos)   /* M = 2 → PLLM = 1 */
                 | (85U << RCC_PLLCFGR_PLLN_Pos)   /* N = 85 */
                 /* PLLR = 0 ⇒ R = 2, default tras reset, no se escribe */
                 | RCC_PLLCFGR_PLLREN;

    /*
     * Paso 6 — Encender el PLL y esperar enganche.
     * PLLRDY se pone en 1 cuando el lazo está enganchado en fase con la referencia.
     */
    RCC->CR |= RCC_CR_PLLON;
    while ((RCC->CR & RCC_CR_PLLRDY) == 0U) { }

    /*
     * Paso 7 — Conmutar SYSCLK al PLL.
     * SW[1:0]  = 0b11 → fuente solicitada = PLL.
     * SWS[1:0] = 0b11 → fuente efectiva confirmada por hardware.
     * El hardware tarda ~ciclos en hacer el cambio; esperamos a SWS para tener certeza.
     */
    RCC->CFGR = (RCC->CFGR & ~RCC_CFGR_SW) | RCC_CFGR_SW_PLL;
    while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL) { }

    /*
     * Paso 8 — Prescalers AHB y APB explícitos a /1.
     * Tras reset todos están en /1 ya, pero lo dejamos escrito para que el código
     * sea autodescriptivo: HCLK = PCLK1 = PCLK2 = SYSCLK = 170 MHz.
     * Los máximos del G4 (170/170/170 MHz) toleran esto sin problema.
     */
    RCC->CFGR |= RCC_CFGR_HPRE_DIV1
              |  RCC_CFGR_PPRE1_DIV1
              |  RCC_CFGR_PPRE2_DIV1;
}
