/*
 * main.c — app 07_m2pc: FCS-M²PC de corriente en lazo cerrado sobre el banco.
 *
 * Primer cierre de lazo del proyecto. El paso de control (ISR de JEOS, 50 kHz):
 *
 *   1. θ_e y ω̂ del encoder (PLL + offset de fase A), sin/cos por CORDIC.
 *   2. i_αβ de los shunts (Clarke con el signo del shunt).
 *   3. Disparo por sobrecorriente: |i_α| o |i_β| > I_TRIP → MOE = 0 y fin.
 *   4. BEMF del modelo: e = λ·ω_e·[−sin θ_e; cos θ_e] (sinusoidal; el ADALINE
 *      sustituirá esta línea por la forma aprendida).
 *   5. Referencia: i_d* = 0, i_q* por rampa → i_αβ* = Park⁻¹.
 *   6. m2pc_step(): predice, elige la combinación de vectores, escribe duties.
 *   7. Monitor: i_dq medidos, costo, sector, tiempos elegidos, presupuesto.
 *
 * SECUENCIA DE ARRANQUE (segura para un primer encendido):
 *   t < 1 s     i_q* = 0.   El controlador regula corriente cero: el motor
 *               debe quedarse quieto y i_dq ≈ 0. Si algo tiene el signo mal,
 *               se ve aquí con corriente mínima.
 *   1 s → 2 s   rampa lineal hasta IQ_REF_MA.
 *   después     i_q* = IQ_REF_MA constante: el motor acelera hasta que el
 *               par iguala la fricción (sin lazo de velocidad todavía).
 *
 * BANCO: Vbus 12 V, límite 500 mA, motor en J7, AS5600 en J8.
 *
 * Stream (100 Hz): S t=<ms> id=<mA> iq=<mA> ref=<mA> om=<rpm> sec=<0-5>
 *                   n=<0-ρ> m=<0-ρ> cost=<LSB²>
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
#include "cordic.h"
#include "motor_params.h"
#include "m2pc.h"

/* ── Constantes del banco (05_phase_align) ───────────────────────────── */
#define PHASE_A_OFFSET_RAW   2558U
#define ENCODER_SIGN         (+1)
#define THETA_E_OFFSET_Q32   (THETA_PLL_POLE_PAIRS * (PHASE_A_OFFSET_RAW << 20))

/* ── Referencia y protección ─────────────────────────────────────────── */
#define IQ_REF_MA            200      /* ~0.4·Inom: par suficiente para girar sin carga */
#define I_TRIP_MA            1200     /* 2.4·Inom; la fuente limita a 500 mA en DC */
#define RAMP_START_MS        1000U
#define RAMP_END_MS          2000U

/* mA → Q8 LSB: mA/1000 · K · 256 */
#define MA_TO_Q8(ma)         ((int32_t)((ma) * ADC_K_RAW_PER_A * 256.0 / 1000.0 + 0.5))
#define IQ_REF_Q8            MA_TO_Q8(IQ_REF_MA)
#define I_TRIP_Q8            MA_TO_Q8(I_TRIP_MA)

#define INV_SQRT3_Q15        18919
#define MON_WINDOW           500U     /* 10 ms */

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

/* ── Compartido ISR ↔ foreground (32 bits: atómico) ─────────────────── */
static volatile int32_t  g_iq_ref_q8;             /* la escribe main (rampa) */
static volatile int32_t  g_pub_id_sum, g_pub_iq_sum;
static volatile uint32_t g_pub_corr, g_pub_i2c_err, g_pub_cyc_max;
static volatile uint32_t g_pub_cyc_sense, g_pub_cyc_m2pc;   /* máximos por tramo */
static volatile uint8_t  g_need_i2c_reset;
static volatile uint8_t  g_tripped;
static volatile int32_t  g_trip_ia, g_trip_ib;

/* ── Paso de control ─────────────────────────────────────────────────── */
static void control_step(void) {
    const uint32_t cyc0 = DWT->CYCCNT;

    /* 1. Posición y velocidad. */
    theta_pll_extrapolate();
    uint16_t raw;
    switch (as5600_nb_tick(&raw)) {
        case AS5600_NB_SAMPLE: theta_pll_update(raw); g_pub_corr++; break;
        case AS5600_NB_ERROR:  g_pub_i2c_err++;                     break;
        case AS5600_NB_STUCK:  g_need_i2c_reset = 1U;               break;
        default:                                                     break;
    }
    uint32_t theta_e = theta_pll_theta_elec_comp_q32() - THETA_E_OFFSET_Q32;
#if (ENCODER_SIGN < 0)
    theta_e = 0U - theta_e;
#endif
    int32_t s, c;
    cordic_sincos_q31(theta_e, &s, &c);

    /* 2. Corrientes en Q8 LSB (el shunt lee −i: offset − raw). */
    const int32_t ia = ((int32_t)g_offset_a - (int32_t)g_ia_raw) << 8;
    const int32_t ib = ((int32_t)g_offset_b - (int32_t)g_ib_raw) << 8;
    const int32_t ic = ((int32_t)g_offset_c - (int32_t)g_ic_raw) << 8;
    const int32_t i_alpha = ia;
    const int32_t i_beta  = (int32_t)(((int64_t)(ib - ic) * INV_SQRT3_Q15) >> 15);

    /* 3. Protección. Tras disparar, el paso no vuelve a tocar el PWM. */
    if (g_tripped != 0U) return;
    if ((i_alpha > I_TRIP_Q8) || (i_alpha < -I_TRIP_Q8) ||
        (i_beta  > I_TRIP_Q8) || (i_beta  < -I_TRIP_Q8)) {
        pwm_disable();
        g_trip_ia = i_alpha;
        g_trip_ib = i_beta;
        g_tripped = 1U;
        return;
    }

    /* 4. BEMF del modelo (mV): e = λ·ω_e·[−sin; cos]. */
    const int32_t e_pk    = (theta_pll_omega_tick_q32() * KE_MV_Q18) >> 18;
    const int32_t e_alpha = (int32_t)(-((int64_t)e_pk * s) >> 31);
    const int32_t e_beta  = (int32_t)( ((int64_t)e_pk * c) >> 31);

    /* 5. Referencia: i_d* = 0 → i_αβ* = i_q*·[−sin; cos]. */
    const int32_t iq_ref    = g_iq_ref_q8;
    const int32_t ref_alpha = (int32_t)(-((int64_t)iq_ref * s) >> 31);
    const int32_t ref_beta  = (int32_t)( ((int64_t)iq_ref * c) >> 31);

    /* 6. Controlador. */
    const m2pc_in_t in = {
        .i_alpha = i_alpha,     .i_beta = i_beta,
        .ref_alpha = ref_alpha, .ref_beta = ref_beta,
        .e_alpha = e_alpha,     .e_beta = e_beta,
        .theta_e_q32 = theta_e,
    };
    const uint32_t cyc1 = DWT->CYCCNT;
    m2pc_step(&in);
    const uint32_t cyc2 = DWT->CYCCNT;
    if ((cyc1 - cyc0) > g_pub_cyc_sense) g_pub_cyc_sense = cyc1 - cyc0;
    if ((cyc2 - cyc1) > g_pub_cyc_m2pc)  g_pub_cyc_m2pc  = cyc2 - cyc1;

    /* 7. Monitor: i_dq medidos (Park), medias por ventana. */
    static int32_t  id_acc, iq_acc;
    static uint32_t n_acc;
    id_acc += (int32_t)(((int64_t)i_alpha * c + (int64_t)i_beta * s) >> 31);
    iq_acc += (int32_t)(((int64_t)i_beta * c - (int64_t)i_alpha * s) >> 31);
    if (++n_acc >= MON_WINDOW) {
        g_pub_id_sum = id_acc;
        g_pub_iq_sum = iq_acc;
        id_acc = 0; iq_acc = 0; n_acc = 0U;
    }

    const uint32_t cyc = DWT->CYCCNT - cyc0;
    if (cyc > g_pub_cyc_max) g_pub_cyc_max = cyc;
}

/* Q8 LSB → mA para imprimir: q8 · 1000 / (K · 256) */
static int32_t q8_to_ma(int32_t q8) {
    return (int32_t)(((int64_t)q8 * 1000) / (int64_t)(ADC_K_RAW_PER_A * 256.0));
}

int main(void) {
    clock_init_170mhz_hse();
    systick_init(170000U);
    uart2_init(115200U);
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("\r\n[boot] STM32G431 @170MHz (app: 07_m2pc)\r\n");

    i2c1_init();
    i2c1_scan();
    i2c1_nb_init();
    uint16_t raw0 = 0U;
    (void)as5600_raw_angle(&raw0);
    theta_pll_init((uint32_t)raw0 << 20);

    cordic_init();
    m2pc_init();

    pwm_init();
    opamp_init();
    adc_init();
    adc_isr_init();
    pwm_enable();
    adc_calibrate_offsets(4096U);
    printf("[calib] offsets: i_a=%u  i_b=%u  i_c=%u\r\n",
           g_offset_a, g_offset_b, g_offset_c);

    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0U;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

    printf("[m2pc] rho=%u (%u candidatos)  G_Q20=%ld  RK_Q8=%ld  KE_Q18=%ld\r\n",
           (unsigned)M2PC_RHO, (unsigned)((M2PC_RHO + 1U) * (M2PC_RHO + 2U) / 2U),
           (long)M2PC_G_Q20, (long)M2PC_RK_Q8, (long)KE_MV_Q18);
    printf("[m2pc] iq* = 0 (1 s) -> rampa -> %d mA; trip a %d mA\r\n",
           IQ_REF_MA, I_TRIP_MA);

    g_iq_ref_q8 = 0;
    adc_set_control_hook(control_step);
    const uint32_t t0 = g_ticks;

    uint32_t t_stream = 0U, t_stat = 0U, corr_prev = 0U;

    while (1) {
        /* Rampa de referencia. */
        const uint32_t t = g_ticks - t0;
        if (t < RAMP_START_MS) {
            g_iq_ref_q8 = 0;
        } else if (t < RAMP_END_MS) {
            g_iq_ref_q8 = (int32_t)((int64_t)IQ_REF_Q8 * (int32_t)(t - RAMP_START_MS)
                                    / (int32_t)(RAMP_END_MS - RAMP_START_MS));
        } else {
            g_iq_ref_q8 = IQ_REF_Q8;
        }

        if (g_tripped != 0U) {
            printf("\r\n[TRIP] sobrecorriente: i_alpha=%ld mA  i_beta=%ld mA. PWM apagado.\r\n",
                   (long)q8_to_ma(g_trip_ia), (long)q8_to_ma(g_trip_ib));
            while (1) { }
        }

        if ((g_ticks - t_stream) >= 10U) {
            t_stream = g_ticks;
            const m2pc_mon_t *mon = m2pc_mon();
            const int64_t rpm10 = ((int64_t)theta_pll_omega_tick_q32() * 30000000LL) >> 32;
            printf("S t=%lu id=%ld iq=%ld ref=%ld om=%s%ld.%ld sec=%u n=%u m=%u cost=%ld\r\n",
                   (unsigned long)g_ticks,
                   (long)q8_to_ma(g_pub_id_sum / (int32_t)MON_WINDOW),
                   (long)q8_to_ma(g_pub_iq_sum / (int32_t)MON_WINDOW),
                   (long)q8_to_ma(g_iq_ref_q8),
                   (rpm10 < 0) ? "-" : "", (long)((rpm10 < 0 ? -rpm10 : rpm10) / 10),
                   (long)((rpm10 < 0 ? -rpm10 : rpm10) % 10),
                   (unsigned)mon->sector, (unsigned)mon->n, (unsigned)mon->m,
                   (long)mon->cost_lsb2);
        }

        if ((g_ticks - t_stat) >= 1000U) {
            t_stat = g_ticks;
            const uint32_t corr = g_pub_corr;
            printf("[m2pc] corr/s=%lu  i2c_err=%lu  overruns=%lu  paso_max=%lu.%02lu us "
                   "(sensado %lu.%02lu, m2pc %lu.%02lu)\r\n",
                   (unsigned long)(corr - corr_prev), (unsigned long)g_pub_i2c_err,
                   (unsigned long)m2pc_overruns(),
                   (unsigned long)(g_pub_cyc_max / 170U),
                   (unsigned long)((g_pub_cyc_max % 170U) * 100U / 170U),
                   (unsigned long)(g_pub_cyc_sense / 170U),
                   (unsigned long)((g_pub_cyc_sense % 170U) * 100U / 170U),
                   (unsigned long)(g_pub_cyc_m2pc / 170U),
                   (unsigned long)((g_pub_cyc_m2pc % 170U) * 100U / 170U));
            corr_prev = corr;
        }

        if (g_need_i2c_reset != 0U) {
            i2c1_reset_peripheral();
            i2c1_nb_abort();
            g_need_i2c_reset = 0U;
        }
    }
}
