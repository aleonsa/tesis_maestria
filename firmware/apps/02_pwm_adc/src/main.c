/*
 * main.c — Bring-up Fase 1: TIM1 PWM + ADC sincronizado.
 *
 * Estado actual: skeleton. Inicializa clock + UART y reporta por VCP.
 * pwm_init() es un stub; implementación real viene en el siguiente paso.
 *
 * Reloj:       HSE 8 MHz → PLL → SYSCLK = 170 MHz (compartido vía lib/clock.c).
 * Periféricos: USART2 sobre PB3/PB4 → /dev/ttyACM0 en la Pi.
 */

#include <stdio.h>
#include "stm32g431xx.h"
#include "clock.h"
#include "uart.h"
#include "pwm.h"

static volatile uint32_t g_ticks = 0;

void SysTick_Handler(void) {
    g_ticks++;
}

void SystemInit(void) {
    /* Vacío a propósito: clock setup en main() para control explícito. */
}

static void systick_init(uint32_t ticks_per_irq) {
    SysTick->LOAD = ticks_per_irq - 1U;
    SysTick->VAL  = 0U;
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk
                  | SysTick_CTRL_TICKINT_Msk
                  | SysTick_CTRL_ENABLE_Msk;
}

static void delay_ms(uint32_t ms) {
    uint32_t start = g_ticks;
    while ((g_ticks - start) < ms) { }
}

int main(void) {
    clock_init_170mhz_hse();
    systick_init(170000U);
    uart2_init(115200U);
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("\r\n[boot] STM32G431 @170MHz, USART2 OK (app: 02_pwm_adc)\r\n");

    pwm_init();
    printf("[pwm_init] STUB — TIM1 todavía no configurado\r\n");

    uint32_t tick = 0;
    while (1) {
        printf("alive tick=%lu uptime=%lu ms\r\n",
               (unsigned long)tick++, (unsigned long)g_ticks);
        delay_ms(1000U);
    }
}
