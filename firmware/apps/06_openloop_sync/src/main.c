/*
 * main.c — app 06_openloop_sync: la cadena de sensado completa dentro de la
 * ISR de control, con el motor girando en open-loop.
 *
 * QUÉ VALIDA
 *
 * Todo lo que el FCS-M2PC va a consumir, junto y a 50 kHz:
 *   - θ_e desde el encoder: PLL (04) + offset de fase A y signo (05).
 *   - i_α, i_β desde los shunts: Clarke con el signo del shunt (N1.23).
 *   - i_d, i_q: Park con el CORDIC.
 *
 * La prueba es dura y barata: en open-loop a velocidad constante, si θ_e es
 * correcta, i_d e i_q son CONSTANTES (el vector de corriente gira solidario
 * con el rotor). Si el offset o el signo están mal, oscilan a la frecuencia
 * eléctrica. Y el retardo θ_ol − θ_e debe ser un ángulo de carga constante.
 *
 * PRESUPUESTO: se mide con DWT cuánto cuesta este paso completo dentro de
 * la ISR. Lo que sobre de 20 µs es lo que le queda al M2PC + ADALINE.
 *
 * BANCO: Vbus 12 V, límite 500 mA, motor en J7, AS5600 en J8. El motor gira
 * a 10 Hz eléctricos (86 rpm).
 *
 * Stream (100 Hz): S t=<ms> lag=<deg_e> id=<mA> iq=<mA> om=<rpm> raw=<enc>
 */

#include <stdio.h>
#include "stm32g431xx.h"
#include "clock.h"
#include "uart.h"
#include "i2c.h"
#include "as5600.h"
#include "theta_pll.h"
#include "pwm.h"
#include "adc.h"
#include "openloop.h"
#include "cordic.h"

/* --------------------------------------------------------------------------
 * Constantes del banco (05_phase_align, 2026-10-13).
 * -------------------------------------------------------------------------- */
#define PHASE_A_OFFSET_RAW   2558U
#define ENCODER_SIGN         (+1)

/* θ_e = SIGN · p · (θ̂ − OFF) = SIGN · (p·θ̂ − p·OFF). El PLL ya entrega p·θ̂
 * con latencia compensada; aquí solo se resta p·OFF, precalculado. La
 * multiplicación desborda a propósito: es el mod 2π eléctrico. */
#define THETA_E_OFFSET_Q32   (THETA_PLL_POLE_PAIRS * (PHASE_A_OFFSET_RAW << 20))

/* Open-loop: 10 Hz eléctricos = 86 rpm mec, 10 % de amplitud (~0.5 A pico). */
#define OL_DELTA_Q32         (OPENLOOP_DELTA_1HZ_ELEC * 10U)
#define OL_AMP_TICKS         170U

/* 1/√3 en Q15 para Clarke (amplitud-invariante). */
#define INV_SQRT3_Q15        18919

/* Ventana del monitor: 500 ticks = 10 ms = 100 Hz. */
#define MON_WINDOW           500U

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
 * Publicado por la ISR para el monitor (32 bits cada uno → atómico).
 * Las medias se publican cada MON_WINDOW ticks; el resto es instantáneo.
 * -------------------------------------------------------------------------- */
static volatile int32_t  g_pub_id_sum, g_pub_iq_sum;   /* Σ raw en la ventana */
static volatile uint32_t g_pub_lag_q32;                /* θ_ol − θ_e           */
static volatile uint16_t g_pub_raw;                    /* última muestra enc.  */
static volatile uint32_t g_pub_theta_e;                /* θ_e del último paso  */
static volatile int32_t  g_pub_sin, g_pub_cos;         /* CORDIC del último paso */
static volatile uint32_t g_pub_corr;                   /* correcciones del PLL */
static volatile uint32_t g_pub_i2c_err;
static volatile uint32_t g_pub_cyc_max;                /* peor paso, ciclos    */
static volatile uint8_t  g_need_i2c_reset;

/* --------------------------------------------------------------------------
 * Paso de control: llamado por la ISR de JEOS a 50 kHz, tras leer las
 * corrientes. Todo entero; sin ramas en el camino caliente salvo el switch
 * del bombeo del encoder.
 * -------------------------------------------------------------------------- */
static void control_step(void) {
    uint32_t cyc0 = DWT->CYCCNT;

    /* 1. Posición: extrapolar siempre, corregir cuando hay muestra. */
    theta_pll_extrapolate();
    uint16_t raw;
    switch (as5600_nb_tick(&raw)) {
        case AS5600_NB_SAMPLE:
            theta_pll_update(raw);
            g_pub_raw = raw;
            g_pub_corr++;
            break;
        case AS5600_NB_ERROR:  g_pub_i2c_err++;     break;
        case AS5600_NB_STUCK:  g_need_i2c_reset = 1U; break;
        default:                                       break;
    }

    uint32_t theta_e = theta_pll_theta_elec_comp_q32() - THETA_E_OFFSET_Q32;
#if (ENCODER_SIGN < 0)
    theta_e = 0U - theta_e;
#endif

    /* 2. Excitación: avanza θ_ol y escribe los duties. */
    openloop_step();

    /* 3. Clarke. El shunt lee −i_fase (N1.23): offset − raw ya trae el signo. */
    int32_t ia = (int32_t)g_offset_a - (int32_t)g_ia_raw;
    int32_t ib = (int32_t)g_offset_b - (int32_t)g_ib_raw;
    int32_t ic = (int32_t)g_offset_c - (int32_t)g_ic_raw;
    int32_t i_alpha = ia;
    int32_t i_beta  = ((ib - ic) * INV_SQRT3_Q15) >> 15;

    /* 4. Park con CORDIC. Productos Q0×Q31 → int64 (smull) → >>31. */
    int32_t s, c;
    cordic_sincos_q31(theta_e, &s, &c);
    int32_t i_d = (int32_t)(((int64_t)i_alpha * c + (int64_t)i_beta * s) >> 31);
    int32_t i_q = (int32_t)(((int64_t)i_beta * c - (int64_t)i_alpha * s) >> 31);

    /* 5. Monitor: retardo instantáneo y medias de ventana. */
    static int32_t  id_acc, iq_acc;
    static uint32_t n_acc;
    id_acc += i_d;
    iq_acc += i_q;
    if (++n_acc >= MON_WINDOW) {
        g_pub_id_sum  = id_acc;
        g_pub_iq_sum  = iq_acc;
        g_pub_lag_q32 = g_theta_q32 - theta_e;
        g_pub_theta_e = theta_e;
        g_pub_sin = s; g_pub_cos = c;
        id_acc = 0; iq_acc = 0; n_acc = 0U;
    }

    uint32_t cyc = DWT->CYCCNT - cyc0;
    if (cyc > g_pub_cyc_max) g_pub_cyc_max = cyc;
}

int main(void) {
    clock_init_170mhz_hse();
    systick_init(170000U);
    uart2_init(115200U);
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("\r\n[boot] STM32G431 @170MHz (app: 06_openloop_sync)\r\n");

    /* Encoder. */
    i2c1_init();
    uint8_t scl, sda, busy;
    i2c1_lines(&scl, &sda, &busy);
    printf("[diag] lineas en reposo: SCL(PB8)=%s  SDA(PB7)=%s  BUSY=%u\r\n",
           scl ? "HIGH" : "LOW", sda ? "HIGH" : "LOW", (unsigned)busy);
    i2c1_scan();
    uint8_t st = 0U;
    if (as5600_status(&st) == I2C_OK) {
        printf("[as5600] STATUS=0x%02X  MD=%d\r\n", (unsigned)st,
               (st & AS5600_STATUS_MD) ? 1 : 0);
    }
    i2c1_nb_init();

    uint16_t raw0 = 0U;
    (void)as5600_raw_angle(&raw0);
    theta_pll_init((uint32_t)raw0 << 20);
    printf("[pll] theta0 = raw %u; offset fase A = raw %u; signo %+d\r\n",
           (unsigned)raw0, (unsigned)PHASE_A_OFFSET_RAW, ENCODER_SIGN);

    cordic_init();
    /* Autotest del CORDIC (sesión 22): CSR debe quedar en 0x00200060 y los
     * cinco ángulos dar (cos, sin) exactos en Q15. Si una lectura devuelve
     * el ángulo ANTERIOR, NRES/RESSIZE están mal (ver cordic.c). */
    printf("[cordic] CSR = %08lX (esperado 00200060)\r\n", (unsigned long)CORDIC->CSR);
    {
        static const uint32_t test_th[] = { 0U, 0x20000000U, 0x40000000U, 0x80000000U, 0xC0000000U };
        for (unsigned k = 0; k < 5; k++) {
            CORDIC->WDATA = test_th[k];
            uint32_t r = CORDIC->RDATA;
            int32_t ss, cc; cordic_sincos_q31(test_th[k], &ss, &cc);
            printf("[cordic] th=%08lX rdata=%08lX  api: s=%ld c=%ld (Q15)\r\n",
                   (unsigned long)test_th[k], (unsigned long)r, (long)(ss >> 16), (long)(cc >> 16));
        }
    }

    /* Potencia + corriente. */
    pwm_init();
    opamp_init();
    adc_init();
    adc_isr_init();
    pwm_enable();
    adc_calibrate_offsets(4096U);
    printf("[calib] offsets: i_a=%u  i_b=%u  i_c=%u\r\n",
           g_offset_a, g_offset_b, g_offset_c);

    /* DWT para el presupuesto del paso. */
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0U;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

    /* Arranque: el hook entra en la ISR desde este instante. */
    openloop_init();
    openloop_set_amplitude(OL_AMP_TICKS);
    openloop_set_freq_q32(OL_DELTA_Q32);
    adc_set_control_hook(control_step);
    openloop_start();
    printf("[sync] open-loop 10 Hz_e, amp 10%%. Si θ_e es correcta, id/iq son DC.\r\n\r\n");

    uint32_t t_stream = 0U, t_stat = 0U, corr_prev = 0U;

    while (1) {
        if ((g_ticks - t_stream) >= 10U) {
            t_stream = g_ticks;

            /* lag en grados eléctricos ×10, signed en ±180°. */
            int64_t lag10 = ((int64_t)(int32_t)g_pub_lag_q32 * 3600LL) >> 32;
            int32_t id_ma = adc_raw_to_ma((int16_t)(g_pub_id_sum / (int32_t)MON_WINDOW));
            int32_t iq_ma = adc_raw_to_ma((int16_t)(g_pub_iq_sum / (int32_t)MON_WINDOW));
            int64_t rpm10 = ((int64_t)theta_pll_omega_tick_q32() * 30000000LL) >> 32;

            printf("S t=%lu lag=%s%ld.%ld id=%ld iq=%ld om=%s%ld.%ld raw=%u the=%lu s=%ld c=%ld csr=%08lX\r\n",
                   (unsigned long)g_ticks,
                   (lag10 < 0) ? "-" : "", (long)((lag10 < 0 ? -lag10 : lag10) / 10),
                   (long)((lag10 < 0 ? -lag10 : lag10) % 10),
                   (long)id_ma, (long)iq_ma,
                   (rpm10 < 0) ? "-" : "", (long)((rpm10 < 0 ? -rpm10 : rpm10) / 10),
                   (long)((rpm10 < 0 ? -rpm10 : rpm10) % 10),
                   (unsigned)g_pub_raw,
                   (unsigned long)(g_pub_theta_e >> 16),
                   (long)(g_pub_sin >> 16), (long)(g_pub_cos >> 16),
                   (unsigned long)CORDIC->CSR);
        }

        if ((g_ticks - t_stat) >= 1000U) {
            t_stat = g_ticks;
            uint32_t corr = g_pub_corr;
            printf("[sync] corr/s=%lu  i2c_err=%lu  paso_max=%lu.%02lu us\r\n",
                   (unsigned long)(corr - corr_prev),
                   (unsigned long)g_pub_i2c_err,
                   (unsigned long)(g_pub_cyc_max / 170U),
                   (unsigned long)((g_pub_cyc_max % 170U) * 100U / 170U));
            corr_prev = corr;
        }

        if (g_need_i2c_reset != 0U) {
            i2c1_reset_peripheral();
            i2c1_nb_abort();
            g_need_i2c_reset = 0U;
            printf("[sync] bus I2C colgado: reset desde foreground\r\n");
        }
    }
}
