/*
 * main.c — bring-up del encoder AS5600 vía I2C1 (PB8 = SCL, PB7 = SDA).
 *
 * Reloj:        HSE 8 MHz → PLL → SYSCLK = 170 MHz (ver clock.c).
 *               I2C1 kernel clock = HSI16 (lo enciende i2c1_init()).
 * Periféricos:  I2C1 master 100 kHz, USART2 (debug VCP), SysTick (timing).
 *
 * Validación:   girar el rotor a mano → ver por `cat /dev/ttyACM0` que el
 *               ángulo recorre 0..4095 de forma monótona y continua, con un
 *               solo salto por vuelta mecánica (el wrap 4095→0), y que STATUS
 *               reporta MD=1 (imán detectado).
 *               ✅ VALIDADO 2026-08-10.
 *
 * Alimentación: basta el USB del ST-LINK. El pad de 5 V de J8 sale del riel
 *               lógico, no de Vbus — comprobado leyendo el encoder con la
 *               fuente de banco apagada. Vbus solo hace falta para mover el
 *               motor.
 *
 * Cableado J8:  pad 2 (B+/H2) = PB7 = SDA
 *               pad 3 (Z+/H3) = PB8 = SCL
 *               pad 1 (A+/H1) = PB6 = libre  ← NO usar para I2C, ver i2c.c
 */

#include <stdio.h>
#include "stm32g431xx.h"
#include "clock.h"
#include "uart.h"
#include "i2c.h"
#include "as5600.h"
#include "swi2c.h"

/*
 * Camino de lectura. 0 = periférico I2C1 (normal). 1 = bit-banging.
 *
 * El bit-bang se conserva porque durante el bring-up fue lo que permitió
 * probar que el AS5600 estaba vivo mientras el periférico fallaba: al manejar
 * los pines como GPIO saca al I2C1 del circuito y desacopla "¿sirve el bus?"
 * de "¿sirve el sensor?". Vale su espacio en flash como red de seguridad.
 */
#define USE_SOFTWARE_I2C   0

/* Volcado de registros + estado de líneas al arrancar. Barato y ha pagado. */
#define DIAG_ON_BOOT       1

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

#if (USE_SOFTWARE_I2C == 1)
    swi2c_init();
    printf("[i2c] bit-banging ~20kHz en PB8(SCL)/PB7(SDA)\r\n");
#else
    i2c1_init();
    printf("[i2c] I2C1 100kHz en PB8(SCL)/PB7(SDA), kernel=HSI16\r\n");

#if (DIAG_ON_BOOT == 1)
    /* Nivel real de los cables antes de hablar. Un bus I2C en reposo debe
     * estar en alto por los pull-ups; cualquier línea en bajo es un fallo
     * eléctrico, no de protocolo, y conviene saberlo antes de culpar al
     * esclavo. */
    uint8_t scl, sda, busy;
    i2c1_lines(&scl, &sda, &busy);
    printf("[diag] lineas en reposo: SCL(PB8)=%s  SDA(PB7)=%s  BUSY=%u\r\n",
           scl ? "HIGH" : "LOW", sda ? "HIGH" : "LOW", (unsigned)busy);
    if (!scl || !sda) {
        printf("[diag] linea clavada en BAJO: corto, esclavo reteniendo el\r\n"
               "       bus, o falta de pull-ups.\r\n");
        i2c1_dump_regs();
    }

    /* Confirma que el bus completa transacciones y que el 0x36 responde.
     * Un scan sin timeouts prueba que el periférico funciona aunque no
     * hubiera ningún esclavo. */
    i2c1_scan();
#endif
#endif

    /* Diagnóstico del imán antes del loop. MD=0 significa que el chip
     * contesta pero no ve el imán: problema mecánico, no de bus. */
    uint8_t st = 0U;
#if (USE_SOFTWARE_I2C == 1)
    int ok_st = (swi2c_read_regs(AS5600_ADDR7, 0x0BU, &st, 1U) == 0);
#else
    int ok_st = (as5600_status(&st) == I2C_OK);
#endif
    if (ok_st) {
        printf("[as5600] STATUS=0x%02X  MD=%d ML=%d MH=%d\r\n",
               (unsigned)st,
               (st & AS5600_STATUS_MD) ? 1 : 0,
               (st & AS5600_STATUS_ML) ? 1 : 0,
               (st & AS5600_STATUS_MH) ? 1 : 0);
        if ((st & AS5600_STATUS_MD) == 0U) {
            printf("[as5600] MD=0: contesta pero NO detecta iman.\r\n"
                   "         Revisar distancia y alineacion del iman.\r\n");
        }
    } else {
        printf("[as5600] no se pudo leer STATUS\r\n");
    }

    printf("\r\n[loop] gira el rotor a mano: raw debe recorrer 0..4095 de\r\n"
           "       forma monotona, con un solo salto por vuelta mecanica.\r\n");

    uint32_t err_count = 0U;

    while (1) {
        uint16_t raw = 0U;
        int ok;

#if (USE_SOFTWARE_I2C == 1)
        uint8_t b[2];
        ok = (swi2c_read_regs(AS5600_ADDR7, 0x0CU, b, 2U) == 0);
        if (ok) raw = (uint16_t)(((uint16_t)(b[0] & 0x0FU) << 8) | b[1]);
#else
        i2c_status_t r = as5600_raw_angle(&raw);
        ok = (r == I2C_OK);
#endif

        if (ok) {
            if (err_count != 0U) {
                printf("[loop] recuperado tras %lu errores\r\n",
                       (unsigned long)err_count);
                err_count = 0U;
            }

            /* Grados ×10 en entero para no arrastrar float fuera de necesidad:
             *   deg10 = raw · 3600 / 4096.  raw·3600 ≤ 4095·3600 < 2^24,
             * así que cabe holgado en 32 bits. 4096 = 2^12 → shift, no
             * división. Coherente con la decisión de punto fijo para todo el
             * path de control. */
            uint32_t deg10 = ((uint32_t)raw * 3600U) >> 12;

            printf("raw=%4u  ang=%lu.%lu deg\r\n",
                   (unsigned)raw,
                   (unsigned long)(deg10 / 10U),
                   (unsigned long)(deg10 % 10U));
        } else {
            err_count++;
            /* Primeros 3 errores, luego uno de cada 50: sin rate-limit un bus
             * caído ahoga el VCP y no se ve nada más. */
            if ((err_count <= 3U) || ((err_count % 50U) == 0U)) {
#if (USE_SOFTWARE_I2C == 1)
                printf("[loop] error de lectura #%lu\r\n",
                       (unsigned long)err_count);
#else
                printf("[loop] error #%lu: %s\r\n",
                       (unsigned long)err_count, i2c_status_str(r));
#endif
            }
        }

        delay_ms(100U);   /* ~10 Hz: suficiente para leer a ojo girando a mano */
    }
}
