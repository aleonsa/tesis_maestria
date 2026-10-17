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
 * Logger (1 kHz, 921600 baud): L t th ia ib ua ub w ref  — ver LOG_WINDOW.
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
#define I_TRIP_MA            1200     /* 2.4·Inom; la fuente limita a 500 mA en DC */
#define RAMP_START_MS        1000U    /* 1 s regulando cero antes de pedir nada */
#define RAMP_END_MS          2000U

/* mA → Q8 LSB: mA/1000 · K · 256 */
#define MA_TO_Q8(ma)         ((int32_t)((ma) * ADC_K_RAW_PER_A * 256.0 / 1000.0 + 0.5))
#define I_TRIP_Q8            MA_TO_Q8(I_TRIP_MA)

/* ── Lazo de velocidad (PI a 1 kHz en foreground, pi_speed_controller.m) ──
 * SPEED_LOOP = 1: ω* = W_REF_RPM; el PI da i_q*. Sin carga, el control de
 * corriente puro es un runaway hasta el límite de tensión (sesión 20); con el
 * lazo de velocidad el seguimiento de corriente se juzga en un punto con
 * margen. SPEED_LOOP = 0: i_q* = IQ_REF_MA tras la rampa (prueba de corriente).
 *
 * Ganancias en mA por rpm. Del arranque de la sesión 20: ~150 mA aceleraron
 * ~1500 rpm/s → la planta vale ~10 (rpm/s)/mA. Con Kp = 2 mA/rpm el lazo
 * queda en ~20 s⁻¹ (~3 Hz), Ki pone el cero en 10 s⁻¹. Conservador. */
#define SPEED_LOOP           1
#define W_REF_RPM            300
#define IQ_REF_MA            200      /* solo con SPEED_LOOP = 0 */
#define IQ_MAX_MA            300      /* saturación del PI (0.6·Inom) */
#define PI_KP_MA_PER_RPM     2.0
#define PI_KI_MA_PER_RPM_S   20.0
/* ω̂ llega en rpm×10; el PI corre cada 1 ms. */
#define PI_KP_Q8   ((int32_t)(PI_KP_MA_PER_RPM / 10.0 * ADC_K_RAW_PER_A * 256.0 / 1000.0 * 256.0 + 0.5))
#define PI_KI_Q16  ((int32_t)(PI_KI_MA_PER_RPM_S / 10.0 / 1000.0 * ADC_K_RAW_PER_A * 256.0 / 1000.0 * 65536.0 + 0.5))
#define IQ_MAX_Q8            MA_TO_Q8(IQ_MAX_MA)

#define INV_SQRT3_Q15        18919
#define ONE_THIRD_Q15        10923

/* CLARKE_FULL = 1: i_α = (2ia − ib − ic)/3, inmune a un offset común a las tres
 * fases (y a cualquier suma ≠ 0). CLARKE_FULL = 0: i_α = ia, que da por hecho
 * ia + ib + ic = 0 (comportamiento hasta la sesión 20). */
#define CLARKE_FULL          1

/* ── Logger: ventanas de 50 muestras (1 kHz) por UART a 921600 ──────────
 * Línea: L t=<ms> th=<θ_e en 1/65536 de vuelta> ia=<Q8> ib=<Q8> ua=<mV>
 *        ub=<mV> w=<rpm×10>
 * i y u son MEDIAS de la ventana (el rizado de conmutación se promedia); θ_e
 * y ω̂ son los valores al cierre de la ventana (el PLL ya es suave). Con esto
 * se reconstruye e = u − R·i − L·di/dt offline, para identificar y para
 * entrenar el ADALINE con la misma metodología que en simulación. */
#define LOG_WINDOW           50U
#define UART_BAUD            921600U

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
static volatile int32_t  g_iq_ref_q8;             /* la escribe main (PI o rampa) */
static volatile uint32_t g_pub_corr, g_pub_i2c_err, g_pub_cyc_max;
static volatile uint32_t g_pub_cyc_sense, g_pub_cyc_m2pc;   /* máximos por tramo */
static volatile uint8_t  g_need_i2c_reset;
static volatile uint8_t  g_tripped;
static volatile int32_t  g_pub_zsum_q8;           /* media de ia+ib+ic en 1 s (Q8 LSB) */
static volatile int32_t  g_trip_ia, g_trip_ib;

/* Ventana del logger. La ISR escribe los campos y al final incrementa seq;
 * main copia y verifica que seq no cambió durante la copia. */
typedef struct {
    int32_t  ia_sum, ib_sum;        /* Q8 LSB × LOG_WINDOW */
    int32_t  ua_sum, ub_sum;        /* mV × LOG_WINDOW     */
    uint32_t theta_e;               /* al cierre           */
    int32_t  omega;                 /* ω̂ Q32/tick al cierre */
    uint32_t seq;
} log_win_t;
static volatile log_win_t g_log;

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

    /* 2. Corrientes en Q8 LSB (el shunt lee −i: offset − raw). Offset con
     * fracción: el entero truncado dejaba hasta 1 LSB (30 mA) de sesgo. */
    const int32_t ia = g_offset_a_q8 - ((int32_t)g_ia_raw << 8);
    const int32_t ib = g_offset_b_q8 - ((int32_t)g_ib_raw << 8);
    const int32_t ic = g_offset_c_q8 - ((int32_t)g_ic_raw << 8);
#if (CLARKE_FULL == 1)
    const int32_t i_alpha = (int32_t)(((int64_t)(2 * ia - ib - ic) * ONE_THIRD_Q15) >> 15);
#else
    const int32_t i_alpha = ia;
#endif
    const int32_t i_beta  = (int32_t)(((int64_t)(ib - ic) * INV_SQRT3_Q15) >> 15);

    /* Suma de las tres corrientes: la real es 0 (estrella sin neutro), así que
     * su media es la suma de los errores de offset (modo común). */
    static int32_t  zs_acc;
    static uint32_t zs_n;
    zs_acc += ia + ib + ic;
    if (++zs_n >= 50000U) {
        g_pub_zsum_q8 = zs_acc / 50000;
        zs_acc = 0;
        zs_n = 0U;
    }

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

    /* 7. Logger: sumas de la ventana; θ_e y ω̂ al cierre. 4 sumas por tick. */
    static int32_t  ia_acc, ib_acc, ua_acc, ub_acc;
    static uint32_t n_acc;
    const m2pc_mon_t *mon = m2pc_mon();
    ia_acc += i_alpha;
    ib_acc += i_beta;
    ua_acc += mon->u_alpha;
    ub_acc += mon->u_beta;
    if (++n_acc >= LOG_WINDOW) {
        g_log.ia_sum  = ia_acc;
        g_log.ib_sum  = ib_acc;
        g_log.ua_sum  = ua_acc;
        g_log.ub_sum  = ub_acc;
        g_log.theta_e = theta_e;
        g_log.omega   = theta_pll_omega_tick_q32();
        g_log.seq++;
        ia_acc = 0; ib_acc = 0; ua_acc = 0; ub_acc = 0; n_acc = 0U;
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
    uart2_init(UART_BAUD);
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
    printf("[calib] offsets Q8 (LSB x256): a=%ld b=%ld c=%ld  -> %ld.%03ld  %ld.%03ld  %ld.%03ld LSB\r\n",
           (long)g_offset_a_q8, (long)g_offset_b_q8, (long)g_offset_c_q8,
           (long)(g_offset_a_q8 >> 8), (long)((g_offset_a_q8 & 255) * 1000 / 256),
           (long)(g_offset_b_q8 >> 8), (long)((g_offset_b_q8 & 255) * 1000 / 256),
           (long)(g_offset_c_q8 >> 8), (long)((g_offset_c_q8 & 255) * 1000 / 256));
    printf("[calib] clarke=%s\r\n", CLARKE_FULL ? "completo" : "i_alpha=ia");

    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0U;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

    printf("[m2pc] rho=%u (%u candidatos)  G_Q20=%ld  RK_Q8=%ld  KE_Q18=%ld\r\n",
           (unsigned)M2PC_RHO, (unsigned)((M2PC_RHO + 1U) * (M2PC_RHO + 2U) / 2U),
           (long)M2PC_G_Q20, (long)M2PC_RK_Q8, (long)KE_MV_Q18);
#if (SPEED_LOOP == 1)
    printf("[m2pc] lazo de velocidad: w* = %d rpm, Kp=%ld/256 Ki=%ld/65536 (Q8 por rpm10), "
           "iq max %d mA; trip a %d mA\r\n",
           W_REF_RPM, (long)PI_KP_Q8, (long)PI_KI_Q16, IQ_MAX_MA, I_TRIP_MA);
#else
    printf("[m2pc] iq* = 0 (1 s) -> rampa -> %d mA; trip a %d mA\r\n",
           IQ_REF_MA, I_TRIP_MA);
#endif
    printf("[log] L t th ia ib ua ub w ref  (1 kHz; th: 65536 = vuelta e; ia/ib: Q8 LSB; "
           "ua/ub: mV; w: rpm x10)\r\n");

    g_iq_ref_q8 = 0;
    adc_set_control_hook(control_step);
    const uint32_t t0 = g_ticks;

    uint32_t t_ctrl = 0U, t_stat = 0U, corr_prev = 0U, log_seq = 0U;
    int32_t  pi_int_q8 = 0;

    while (1) {
        const uint32_t t = g_ticks - t0;

        /* ── Lazo externo a 1 kHz ─────────────────────────────────────── */
        if ((g_ticks - t_ctrl) >= 1U) {
            t_ctrl = g_ticks;
#if (SPEED_LOOP == 1)
            if (t < RAMP_START_MS) {
                g_iq_ref_q8 = 0;                     /* 1 s regulando cero */
            } else {
                /* PI con anti-windup por "deshacer la integración"
                 * (pi_speed_controller.m). Error en rpm×10. */
                const int32_t rpm10 = (int32_t)(((int64_t)theta_pll_omega_tick_q32() * 30000000LL) >> 32);
                const int32_t err   = W_REF_RPM * 10 - rpm10;
                const int32_t dint  = (err * PI_KI_Q16) >> 16;
                pi_int_q8 += dint;
                int32_t iq = ((err * PI_KP_Q8) >> 8) + pi_int_q8;
                if (iq > IQ_MAX_Q8)       { iq = IQ_MAX_Q8;  pi_int_q8 -= dint; }
                else if (iq < -IQ_MAX_Q8) { iq = -IQ_MAX_Q8; pi_int_q8 -= dint; }
                g_iq_ref_q8 = iq;
            }
#else
            if (t < RAMP_START_MS) {
                g_iq_ref_q8 = 0;
            } else if (t < RAMP_END_MS) {
                g_iq_ref_q8 = (int32_t)((int64_t)MA_TO_Q8(IQ_REF_MA) * (int32_t)(t - RAMP_START_MS)
                                        / (int32_t)(RAMP_END_MS - RAMP_START_MS));
            } else {
                g_iq_ref_q8 = MA_TO_Q8(IQ_REF_MA);
            }
#endif
        }

        if (g_tripped != 0U) {
            printf("\r\n[TRIP] sobrecorriente: i_alpha=%ld mA  i_beta=%ld mA. PWM apagado.\r\n",
                   (long)q8_to_ma(g_trip_ia), (long)q8_to_ma(g_trip_ib));
            while (1) { }
        }

        /* ── Logger: una línea por ventana cerrada ────────────────────── */
        if (g_log.seq != log_seq) {
            log_win_t w;
            do {
                log_seq = g_log.seq;
                w.ia_sum = g_log.ia_sum; w.ib_sum = g_log.ib_sum;
                w.ua_sum = g_log.ua_sum; w.ub_sum = g_log.ub_sum;
                w.theta_e = g_log.theta_e; w.omega = g_log.omega;
            } while (g_log.seq != log_seq);        /* copia consistente */
            const int32_t rpm10 = (int32_t)(((int64_t)w.omega * 30000000LL) >> 32);
            printf("L %lu %u %ld %ld %ld %ld %ld %ld\r\n",
                   (unsigned long)t,
                   (unsigned)(w.theta_e >> 16),
                   (long)(w.ia_sum / (int32_t)LOG_WINDOW),
                   (long)(w.ib_sum / (int32_t)LOG_WINDOW),
                   (long)(w.ua_sum / (int32_t)LOG_WINDOW),
                   (long)(w.ub_sum / (int32_t)LOG_WINDOW),
                   (long)rpm10,
                   (long)q8_to_ma(g_iq_ref_q8));
        }

        /* ── Estado a 1 Hz ────────────────────────────────────────────── */
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
            printf("[off] a=%ld b=%ld c=%ld (Q8)  media(ia+ib+ic)=%ld (Q8 LSB)\r\n",
                   (long)g_offset_a_q8, (long)g_offset_b_q8, (long)g_offset_c_q8,
                   (long)g_pub_zsum_q8);
        }

        if (g_need_i2c_reset != 0U) {
            i2c1_reset_peripheral();
            i2c1_nb_abort();
            g_need_i2c_reset = 0U;
        }
    }
}
