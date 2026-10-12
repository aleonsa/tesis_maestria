/*
 * main.c — app 04_theta_pll: observador PLL tipo II sobre el AS5600.
 *
 * Banco de validación del observador de posición (sesión 17). Sin PWM, sin
 * ADC, sin motor: solo TIM2 a 50 kHz simulando el latido del lazo de control
 * y el AS5600 por I2C1 no bloqueante, exactamente el patrón start/take que
 * usará la ISR JEOS en la integración final.
 *
 * Reloj:        HSE 8 MHz → PLL → SYSCLK = 170 MHz. TIM2 en APB1 (÷1) = 170 MHz.
 * ISR:          TIM2 update a 50 kHz, prioridad NVIC 1 (la misma que tendrá
 *               la ISR de control; el I2C queda en prio 3 por debajo, como en
 *               el sistema final — ver la cabecera de i2c.h).
 *
 * Qué valida (girando el rotor a mano):
 *   1. θ̂ continuo entre muestras: la línea de stream `S ...` debe verse como
 *      una rampa suave en el plotter, no escalones.
 *   2. ω̂ limpio: sin los picos de ~24 rad/s de la derivada cruda.
 *   3. Presupuesto: la ISR completa (extrapolación + pump del I2C + corrección
 *      cuando hay muestra) debe quedar muy por debajo de los 20 µs del tick;
 *      se mide con el contador de ciclos DWT y se reporta el máximo.
 *
 * Sonda de scope: PB6 = HIGH durante la ISR (convención del proyecto desde la
 * sesión 13: PB6 es GPIO libre, su AF4 está vacía — no confundir con I2C).
 *
 * Stream para python_scripts/plot_theta.py (100 Hz):
 *   S t=<ms> raw=<0..4095> th=<deg desenvueltos> om=<rpm mec>
 */

#include <stdio.h>
#include "stm32g431xx.h"
#include "clock.h"
#include "uart.h"
#include "i2c.h"
#include "as5600.h"
#include "theta_pll.h"

/* --------------------------------------------------------------------------
 * TIM2 a 50 kHz: ARR = 170 MHz / 50 kHz = 3400. Mismo latido que la ISR JEOS
 * del sistema final, sin necesidad de PWM ni ADC en el banco.
 * -------------------------------------------------------------------------- */
#define TIM2_ARR   (3400U - 1U)

/* Deadline del bus dentro de la ISR, en ticks: una lectura legítima tarda
 * ~24 ticks (468 µs). 100 ticks = 2 ms es factor ~4. La recuperación con
 * reset del periférico es SOLO foreground (ver i2c.h); la ISR pide el STOP
 * asíncrono y levanta la bandera. */
#define NB_ABORT_TICKS   100U
#define NB_RESET_TICKS   200U

static volatile uint32_t g_ticks = 0;

void SysTick_Handler(void) {
    g_ticks++;
}

void SystemInit(void) {
    /* Vacío a propósito: el setup del reloj se hace desde main(). */
}

static void systick_init(uint32_t ticks_per_irq) {
    SysTick->LOAD = ticks_per_irq - 1U;
    SysTick->VAL  = 0U;
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk
                  | SysTick_CTRL_TICKINT_Msk
                  | SysTick_CTRL_ENABLE_Msk;
}

/* --------------------------------------------------------------------------
 * Estado compartido ISR → foreground (todo de 32 bits: atómico en M4).
 * -------------------------------------------------------------------------- */
static volatile uint32_t g_isr_cycles_max = 0;   /* peor duración de la ISR   */
static volatile uint32_t g_corr_count     = 0;   /* correcciones aplicadas    */
static volatile uint32_t g_i2c_err_count  = 0;   /* cierres fallidos          */
static volatile uint32_t g_need_i2c_reset = 0;   /* bandera: reset foreground */
static volatile uint16_t g_last_raw       = 0;   /* última muestra cruda      */

/*
 * ISR del latido de control a 50 kHz. Esqueleto deliberadamente idéntico al
 * que llevará la integración: extrapolar siempre, bombear el bus un paso,
 * corregir solo cuando hay muestra. El hueco entre start y take es donde la
 * ISR de control real hará Clarke + M2PC + ADALINE.
 */
void TIM2_IRQHandler(void) {
    GPIOB->BSRR = (1U << 6);                 /* sonda: entra la ISR          */
    uint32_t cyc0 = DWT->CYCCNT;
    TIM2->SR = ~TIM_SR_UIF;

    theta_pll_extrapolate();

    static uint32_t busy_ticks = 0;
    switch (i2c1_nb_state()) {
        case I2C_NB_IDLE:
            if (as5600_raw_angle_start() == I2C_OK) {
                busy_ticks = 0;
            }
            break;

        case I2C_NB_BUSY:
            busy_ticks++;
            if (busy_ticks == NB_ABORT_TICKS) {
                i2c1_nb_abort();             /* asíncrono, legal desde ISR   */
            } else if (busy_ticks == NB_RESET_TICKS) {
                g_need_i2c_reset = 1U;       /* el STOP no llegó: foreground */
            }
            break;

        default: {                           /* DONE o ERR: recoger y liberar */
            uint16_t raw = 0U;
            if (as5600_raw_angle_take(&raw) == I2C_OK) {
                theta_pll_update(raw);
                g_last_raw = raw;
                g_corr_count++;
            } else {
                g_i2c_err_count++;
            }
            break;
        }
    }

    uint32_t cyc = DWT->CYCCNT - cyc0;
    if (cyc > g_isr_cycles_max) g_isr_cycles_max = cyc;

    GPIOB->BSRR = (1U << (6U + 16U));        /* sonda: sale la ISR           */
}

static void tim2_init_50khz(void) {
    RCC->APB1ENR1 |= RCC_APB1ENR1_TIM2EN;
    TIM2->PSC  = 0U;
    TIM2->ARR  = TIM2_ARR;
    TIM2->DIER = TIM_DIER_UIE;
    NVIC_SetPriority(TIM2_IRQn, 1);
    NVIC_EnableIRQ(TIM2_IRQn);
    TIM2->CR1 = TIM_CR1_CEN;
}

int main(void) {
    clock_init_170mhz_hse();
    systick_init(170000U);                   /* 1 ms @ 170 MHz */
    uart2_init(115200U);
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("\r\n[boot] STM32G431 @170MHz (app: 04_theta_pll)\r\n");

    i2c1_init();
    printf("[i2c] I2C1 100kHz en PB8(SCL)/PB7(SDA), kernel=HSI16\r\n");

    /* Mismo diagnóstico de arranque que 03_as5600: líneas en reposo, scan y
     * STATUS del imán. Si esto falla no hay observador que observar. */
    uint8_t scl, sda, busy;
    i2c1_lines(&scl, &sda, &busy);
    printf("[diag] lineas en reposo: SCL(PB8)=%s  SDA(PB7)=%s  BUSY=%u\r\n",
           scl ? "HIGH" : "LOW", sda ? "HIGH" : "LOW", (unsigned)busy);
    i2c1_scan();

    uint8_t st = 0U;
    if (as5600_status(&st) == I2C_OK) {
        printf("[as5600] STATUS=0x%02X  MD=%d ML=%d MH=%d\r\n",
               (unsigned)st,
               (st & AS5600_STATUS_MD) ? 1 : 0,
               (st & AS5600_STATUS_ML) ? 1 : 0,
               (st & AS5600_STATUS_MH) ? 1 : 0);
    } else {
        printf("[as5600] no se pudo leer STATUS — sin sensor no hay PLL\r\n");
    }

    i2c1_nb_init();
    printf("[i2c] camino por IRQ armado (I2C1_EV/ER, prioridad NVIC 3)\r\n");

    /* Condición inicial del observador: lectura bloqueante. Vale aquí: todavía
     * no corre la ISR de 50 kHz. */
    uint16_t raw0 = 0U;
    if (as5600_raw_angle(&raw0) == I2C_OK) {
        printf("[pll] theta0 = raw %u\r\n", (unsigned)raw0);
    } else {
        printf("[pll] AVISO: lectura inicial fallo, theta0 = 0\r\n");
    }
    theta_pll_init((uint32_t)raw0 << 20);

    printf("[pll] tipo II: wn=2pi*100 rad/s, zeta=1, correccion cada ~468us\r\n");
    printf("[pll] adelanto por latencia: %u ticks (%lu us)\r\n",
           (unsigned)THETA_PLL_LAT_TICKS,
           (unsigned long)THETA_PLL_LAT_TICKS * 20UL);

    /* Contador de ciclos DWT para el presupuesto de la ISR. */
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0U;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

    /* PB6: sonda de scope de la ISR (convención del proyecto). */
    RCC->AHB2ENR |= RCC_AHB2ENR_GPIOBEN;
    GPIOB->MODER &= ~GPIO_MODER_MODE6_Msk;
    GPIOB->MODER |=  (1U << GPIO_MODER_MODE6_Pos);

    tim2_init_50khz();
    printf("[tim2] latido a 50 kHz armado (ARR=%lu, prio 1)\r\n",
           (unsigned long)TIM2_ARR);

    printf("\r\n[loop] gira el rotor a mano: th debe ser una rampa suave,\r\n"
           "       om estable. Stream 100 Hz para plot_theta.py.\r\n");

    /* Desenvuelto de θ̂ para el stream: acumulador en fracción Q32 como
     * int64 — cada diff es la distancia signed entre lecturas consecutivas,
     * que a 100 Hz de stream y velocidades de mano es ≪ media vuelta.
     * El acumulador arranca del valor UNSIGNED (sin cast a int32): con cast,
     * un θ0 > media vuelta arrancaba en −2.5° en vez de +357.5° (mismo ángulo
     * mod 2π, pero confuso en el plot). */
    uint32_t th_prev = theta_pll_theta_mech_q32();
    int64_t  th_acc_frac = (int64_t)th_prev;

    uint32_t t_stream = 0U, t_stat = 0U;
    uint32_t corr_prev = 0U;

    while (1) {
        /* Stream a 100 Hz para el plotter. */
        if ((g_ticks - t_stream) >= 10U) {
            t_stream = g_ticks;

            uint32_t th = theta_pll_theta_mech_q32();
            th_acc_frac += (int32_t)(th - th_prev);
            th_prev = th;

            /* grados ×100 = acc · 36000 / 2^32 (int64: cabe de sobra) */
            int64_t deg100 = (th_acc_frac * 36000LL) >> 32;
            int32_t om = theta_pll_omega_tick_q32();
            /* rpm ×10: omega [rev/tick Q32] · 50 000 tick/s · 60 s/min · 10 */
            int64_t rpm10 = ((int64_t)om * 30000000LL) >> 32;

            printf("S t=%lu raw=%u th=%ld.%02ld om=%s%ld.%ld\r\n",
                   (unsigned long)g_ticks,
                   (unsigned)g_last_raw,
                   (long)(deg100 / 100),
                   (long)(deg100 % 100 < 0 ? -(deg100 % 100) : deg100 % 100),
                   (rpm10 < 0) ? "-" : "",
                   (long)(rpm10 < 0 ? -rpm10 : rpm10) / 10L,
                   (long)(rpm10 < 0 ? -rpm10 : rpm10) % 10L);
        }

        /* Estado del observador a 1 Hz. */
        if ((g_ticks - t_stat) >= 1000U) {
            t_stat = g_ticks;
            uint32_t corr = g_corr_count;
            printf("[pll] corr/s=%lu  i2c_err=%lu  isr_max=%lu.%02lu us%s\r\n",
                   (unsigned long)(corr - corr_prev),
                   (unsigned long)g_i2c_err_count,
                   (unsigned long)(g_isr_cycles_max / 170U),
                   (unsigned long)((g_isr_cycles_max % 170U) * 100U / 170U),
                   (g_need_i2c_reset != 0U) ? "  RESET PENDIENTE" : "");
            corr_prev = corr;
        }

        /* Recuperación de bus: solo foreground, nunca desde la ISR (i2c.h). */
        if (g_need_i2c_reset != 0U) {
            printf("[pll] bus colgado: reset de periferico desde foreground\r\n");
            i2c1_reset_peripheral();
            i2c1_nb_abort();
            g_need_i2c_reset = 0U;
        }
    }
}
