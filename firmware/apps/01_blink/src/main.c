/*
 * blink.c — bare-metal CMSIS sin HAL.
 * Toggle del LED user (PC6) + heartbeat por UART2 a 1 Hz.
 *
 * Reloj:        HSE 8 MHz → PLL → SYSCLK = 170 MHz (ver clock.c).
 * Periféricos:  GPIOC bit 6 (LED), SysTick (timing), USART2 (debug VCP).
 * Salida UART:  PB3 (TX) / PB4 (RX) → ST-LINK VCP → /dev/ttyACM0 en la Pi.
 *
 * Validación:   LED parpadea 1 Hz visual + el host ve "tick N" cada segundo
 *               por `cat /dev/ttyACM0` a 115200 8N1.
 */

#include <stdio.h>
#include "stm32g431xx.h"
#include "clock.h"
#include "uart.h"

/* Contador de ticks de SysTick — 1 tick = 1 ms. Wrap-around a ~49.7 días. */
static volatile uint32_t g_ticks = 0;

/*
 * ISR del SysTick. Nombre exacto exigido por la vector table del startup.
 * No es static: el linker debe verlo para sobrescribir el weak Default_Handler.
 */
void SysTick_Handler(void) {
    g_ticks++;
}

void SystemInit(void) {
    /*
     * Vacío a propósito: el startup la llama antes de main(), pero preferimos
     * hacer el setup del reloj desde main() para tener control explícito del
     * orden de ejecución y poder depurar paso a paso con GDB.
     */
}

/*
 * Configura SysTick para generar una IRQ cada `ticks_per_irq` ciclos del AHB.
 * Para 1 ms a HCLK = 170 MHz: ticks_per_irq = 170000.
 * (A 16 MHz default sería 16000 — útil de recordar si se vuelve sin PLL).
 */
static void systick_init(uint32_t ticks_per_irq) {
    SysTick->LOAD = ticks_per_irq - 1U;
    SysTick->VAL  = 0U;
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk
                  | SysTick_CTRL_TICKINT_Msk
                  | SysTick_CTRL_ENABLE_Msk;
}

/*
 * Espera `ms` milisegundos. La resta unsigned hace el cálculo de delta
 * correcto incluso ante wrap-around de g_ticks.
 */
static void delay_ms(uint32_t ms) {
    uint32_t start = g_ticks;
    while ((g_ticks - start) < ms) { }
}

int main(void) {
    /*
     * PRIMER paso: subir SYSCLK a 170 MHz vía PLL desde HSE.
     * Cualquier setup posterior que dependa del reloj (SysTick, UART, TIM, ADC)
     * debe ejecutarse DESPUÉS de esta llamada, con el reloj final ya estable.
     */
    clock_init_170mhz_hse();

    /* Habilitar clock al bus AHB2 del GPIOC (donde vive PC6, el LED user). */
    RCC->AHB2ENR |= RCC_AHB2ENR_GPIOCEN;

    /* PC6 en modo "general purpose output" (MODER = 0b01). */
    GPIOC->MODER &= ~GPIO_MODER_MODE6_Msk;
    GPIOC->MODER |=  (0b01U << GPIO_MODER_MODE6_Pos);

    /* SysTick a 1 ms con HCLK = 170 MHz → 170000 ciclos por interrupción. */
    systick_init(170000U);

    /* USART2 sobre PB3/PB4 → VCP del ST-LINK → /dev/ttyACM0 en la Pi. */
    uart2_init(115200U);

    /*
     * Deshabilita buffering en stdout para que cada printf salga inmediatamente.
     * Sin esto, newlib puede acumular bytes en un buffer hasta que se llene
     * o aparezca '\n' — incómodo cuando se depura paso a paso.
     */
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("\r\n[boot] STM32G431 @170MHz, USART2 OK\r\n");

    uint32_t tick = 0;
    while (1) {
        GPIOC->ODR ^= GPIO_ODR_OD6;
        printf("tick %lu  uptime=%lu ms\r\n",
               (unsigned long)tick++, (unsigned long)g_ticks);
        delay_ms(500U);
    }
}
