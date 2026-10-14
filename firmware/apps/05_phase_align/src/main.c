/*
 * main.c — app 05_phase_align: alineación del eje eléctrico con la fase A
 * y verificación del sentido de giro del encoder.
 *
 * POR QUÉ
 *
 * El control (Clarke/Park, M2PC, ADALINE) trabaja con θ_e, posición
 * ELÉCTRICA. El AS5600 entrega posición MECÁNICA (0..4095 por vuelta). La
 * traducción θ_e = signo · p · (θ_m − θ_0) necesita dos constantes que el
 * hardware no entrega: el offset θ_0 (qué raw del encoder corresponde al eje
 * de la fase A) y el signo (si el encoder cuenta en el mismo sentido en que
 * avanza θ_e para la secuencia de fases positiva). Ambas se obtienen aquí.
 *
 * FÍSICA DE LA ALINEACIÓN
 *
 * Con duty fijo (no sinusoidal): A al 60 %, B y C al 45 %, el puente aplica
 * un vector estático sobre el eje de la fase A. Corriente DC ≈ 0.44 A por A
 * ((0.60−0.45)·12 V / 2.7 Ω), por debajo del Inom (0.5 A), por unos segundos
 * — suficiente para que el rotor gire y se asiente con su eje d sobre ese
 * vector. El rotor tiene 7 posiciones de equilibrio por vuelta mecánica
 * (p = 7); cualquiera de ellas es un "θ_e = 0" válido, y la fórmula de abajo
 * lo absorbe.
 *
 * PRUEBA DE SENTIDO
 *
 * Luego se corre la excitación sinusoidal open-loop a 2 Hz eléctricos (lento,
 * el rotor sigue). θ del generador siempre crece; si el raw del encoder
 * también crece (neto), el signo es +1; si decrece, −1.
 *
 * SALIDA (las dos constantes que va a usar el control):
 *
 *   PHASE_A_OFFSET_RAW = <raw del encoder con el rotor alineado al eje A>
 *   ENCODER_SIGN       = +1 o -1
 *
 *   θ_e [Q32, 1 vuelta eléctrica = 2^32] =
 *       ENCODER_SIGN · p · (uint32_t)(raw − PHASE_A_OFFSET_RAW) · 2^20
 *
 *   (raw − OFF) es uint32: el wrap 4095→0 y el mod 2π eléctrico los hace el
 *    overflow, como en theta_pll.c.)
 *
 * BANCO: Vbus 12 V, límite 500 mA, motor en J7, AS5600 en J8. El rotor
 * girará a propósito y quedará sujeto al eje A; no es un fallo.
 */

#include <stdio.h>
#include "stm32g431xx.h"
#include "clock.h"
#include "uart.h"
#include "i2c.h"
#include "as5600.h"
#include "pwm.h"
#include "adc.h"
#include "openloop.h"

/* Duty de alineación: A al 60 %, B/C al 45 % (ticks de ARR=1700).
 * Corriente DC ≈ 0.44 A < Inom 0.5 A. No subir sin recalcular. */
#define ALIGN_DUTY_A    (uint16_t)((PWM_ARR * 60U) / 100U)   /* 1020 */
#define ALIGN_DUTY_BC   (uint16_t)((PWM_ARR * 45U) / 100U)   /*  765 */

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

static void delay_ms(uint32_t ms) {
    uint32_t start = g_ticks;
    while ((g_ticks - start) < ms) { }
}

/* Lectura del encoder por el camino bloqueante (aquí no corre ningún lazo
 * que lo prohíba). Promedia n lecturas para limpiar cuantización. */
static uint16_t read_encoder_avg(uint8_t n) {
    uint32_t acc = 0U;
    for (uint8_t k = 0U; k < n; k++) {
        uint16_t raw = 0U;
        if (as5600_raw_angle(&raw) != I2C_OK) {
            printf("[align] AVISO: lectura encoder %u fallo\r\n", (unsigned)k);
        }
        acc += raw;
        delay_ms(20U);
    }
    return (uint16_t)(acc / n);
}

int main(void) {
    clock_init_170mhz_hse();
    systick_init(170000U);                   /* 1 ms @ 170 MHz */
    uart2_init(115200U);
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("\r\n[boot] STM32G431 @170MHz (app: 05_phase_align)\r\n");

    /* Encoder: misma batería de diagnóstico que 03/04. Si el bus no está
     * sano, la alineación no tiene sentido. */
    i2c1_init();
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
    }

    /* Potencia + corriente. */
    pwm_init();
    opamp_init();
    adc_init();
    adc_isr_init();

    printf("\r\n[align] encendiendo PWM (duty 50%% balanceado)...\r\n");
    pwm_enable();

    printf("[calib] offsets de corriente (PWM corriendo, sin corriente)...\r\n");
    adc_calibrate_offsets(4096U);
    printf("[calib] offsets: i_a=%u  i_b=%u  i_c=%u (raw)\r\n",
           g_offset_a, g_offset_b, g_offset_c);

    /* ---------------------------------------------------------------
     * PASO 1 — alineación al eje de la fase A.
     * --------------------------------------------------------------- */
    printf("\r\n[align] PASO 1: vector estatico fase A (60%%/45%%/45%%) "
           "durante 4 s...\r\n");
    printf("[align]   >>> el rotor va a GIRAR y quedarse sujeto <<<"
           " (es el proposito)\r\n");

    pwm_set_duties(ALIGN_DUTY_A, ALIGN_DUTY_BC, ALIGN_DUTY_BC);
    delay_ms(4000U);                          /* asentamiento mecánico */

    uint16_t raw_align = read_encoder_avg(8U);
    printf("[align] raw alineado = %u  (promedio de 8 lecturas)\r\n",
           (unsigned)raw_align);

    /* Liberar el rotor: vuelta a neutro. */
    pwm_set_duties(PWM_ARR / 2U, PWM_ARR / 2U, PWM_ARR / 2U);
    printf("[align] rotor liberado (duty 50%% neutro)\r\n");

    /* ---------------------------------------------------------------
     * PASO 2 — sentido de giro. Excitación sinusoidal 2 Hz eléctricos,
     * 6 s. θ del generador crece siempre; el signo del encoder sale del
     * signo neto del raw.
     * --------------------------------------------------------------- */
    printf("\r\n[align] PASO 2: giro open-loop 2 Hz_e durante 6 s "
           "(el rotor gira lento)...\r\n");

    openloop_init();
    adc_set_control_hook(openloop_step);      /* paso de control en la ISR */
    openloop_set_amplitude(170U);             /* 10 % PWM: ~0.7 A pico */
    openloop_set_freq_q32(OPENLOOP_DELTA_2HZ_ELEC);

    uint16_t raw0 = read_encoder_avg(4U);
    openloop_start();
    delay_ms(6000U);
    openloop_stop();                          /* restaura duty 50% */
    uint16_t raw1 = read_encoder_avg(4U);

    /* Delta neto desenvuelto en ±media vuelta: a 2 Hz por 6 s son 12 ciclos
     * eléctricos = 12/7 vueltas mec ≈ 7000 cuentas, así que el camino corto
     * es el correcto sin ambigüedad. */
    int32_t d = (int32_t)(raw1 - raw0);
    if (d > 2048)  d -= 4096;
    if (d < -2048) d += 4096;
    /* Pero el desplazamiento real es ~7000: reconstruir con vueltas enteras
     * eligiendo el representante congruente de mayor magnitud (el movimiento
     * real domina sobre el camino corto). */
    while (d > 0 && d < 3000)  d += 4096;
    while (d < 0 && d > -3000) d -= 4096;

    int encoder_sign = (d > 0) ? +1 : -1;

    printf("[align] raw0=%u  raw1=%u  delta=%+ld cuentas\r\n",
           (unsigned)raw0, (unsigned)raw1, (long)d);
    printf("[align] θ del generador CRECIÓ; el encoder %s → ENCODER_SIGN = %+d\r\n",
           (encoder_sign > 0) ? "también crece" : "decrece", encoder_sign);

    /* ---------------------------------------------------------------
     * Resumen: las dos constantes del control.
     * --------------------------------------------------------------- */
    printf("\r\n=== CONSTANTES DE CONTROL ===\r\n");
    printf("PHASE_A_OFFSET_RAW = %u\r\n", (unsigned)raw_align);
    printf("ENCODER_SIGN       = %+d\r\n", encoder_sign);
    printf("\r\nθ_e [Q32, 2^32 = 1 vuelta eléctrica] =\r\n");
    printf("    ENCODER_SIGN · 7 · (uint32_t)(raw − PHASE_A_OFFSET_RAW) · 2^20\r\n");
    printf("(verificacion) θ_e con el rotor alineado ≈ 0; una vuelta\r\n");
    printf("mecanica del raw = 7 vueltas electricas.\r\n");
    printf("==============================\r\n");

    printf("\r\n[done] Apagar Vbus antes de desconectar el motor.\r\n");

    while (1) {
        /* Heartbeat mínimo: deja la plataforma viva para un re-flash. */
        delay_ms(1000U);
    }
}
