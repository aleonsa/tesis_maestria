/*
 * main.c — bring-up del encoder AS5600 vía I2C1 (PB6/PB7).
 *
 * Reloj:        HSE 8 MHz → PLL → SYSCLK = 170 MHz (ver clock.c).
 *               I2C1 kernel clock = HSI16 (lo enciende i2c1_init()).
 * Periféricos:  I2C1 master 100 kHz, USART2 (debug VCP), SysTick (timing).
 *
 * Validación:   girar el rotor a mano → ver por `cat /dev/ttyACM0` que el
 *               ángulo recorre 0..4095 (y 0..360°) de forma monótona y
 *               continua, sin saltos, y STATUS reporta MD=1 (imán detectado).
 */

#include <stdio.h>
#include "stm32g431xx.h"
#include "clock.h"
#include "uart.h"
#include "i2c.h"
#include "as5600.h"

static volatile uint32_t g_ticks = 0;

void SysTick_Handler(void) {
    g_ticks++;
}

void SystemInit(void) {
    /* Vacío a propósito: el setup del reloj se hace desde main() para tener
     * control explícito del orden (mismo criterio que 01_blink/02_pwm_adc). */
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
    systick_init(170000U);     /* 1 ms @ 170 MHz */
    uart2_init(115200U);
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("\r\n[boot] STM32G431 @170MHz (app: 03_as5600)\r\n");

    i2c1_init();
    printf("[i2c] I2C1 100kHz en PB6(SCL)/PB7(SDA), kernel=HSI16\r\n");

    /* Diagnóstico del imán antes del loop. Si esto se cuelga (sin más prints),
     * el AS5600 no hizo ACK → revisar pull-ups / 5V tolerance / cableado J8
     * con el osciloscopio en PB6. */
    uint8_t st = as5600_status();
    printf("[as5600] STATUS=0x%02X  MD=%d ML=%d MH=%d\r\n",
           (unsigned)st,
           (st & AS5600_STATUS_MD) ? 1 : 0,
           (st & AS5600_STATUS_ML) ? 1 : 0,
           (st & AS5600_STATUS_MH) ? 1 : 0);

    while (1) {
        uint16_t raw = as5600_raw_angle();

        /* Grados ×10 en entero para no arrastrar float fuera de necesidad:
         *   deg10 = raw · 3600 / 4096.  raw·3600 ≤ 4095·3600 < 2^24, cabe en u32. */
        uint32_t deg10 = ((uint32_t)raw * 3600U) >> 12;

        printf("raw=%4u  ang=%lu.%lu deg\r\n",
               (unsigned)raw,
               (unsigned long)(deg10 / 10U),
               (unsigned long)(deg10 % 10U));

        delay_ms(100U);   /* ~10 Hz: suficiente para leer a ojo girando a mano */
    }
}
