/*
 * main.c — Bring-up Fase 1: TIM1 PWM + ADC sincronizado.
 *
 * Estado: PWM 3-fásico complementario funcionando a 50 kHz, duty 50%.
 * Cierre de Semana 4 del planning (TIM1).
 *
 * Reloj:       HSE 8 MHz → PLL → SYSCLK = 170 MHz (compartido vía lib/clock.c).
 * Periféricos: USART2 sobre PB3/PB4 → /dev/ttyACM0 en la Pi.
 *               TIM1: 6 PWMs complementarias, dead-time 500 ns, TRGO update event.
 *
 * Para próximas semanas: agregar ADC + OPAMPs + AS5600.
 */

#include <stdio.h>
#include "stm32g431xx.h"
#include "clock.h"
#include "uart.h"
#include "pwm.h"
#include "adc.h"
#include "openloop.h"

/* --------------------------------------------------------------------------
 * Bring-up del motor (Semana 6 sigue):
 *   OPENLOOP_ENABLE = 0 → duty 50% balanceado, motor quieto.
 *   OPENLOOP_ENABLE = 1 → excitación sinusoidal trifásica activa.
 *
 * Flujo de bring-up:
 *   Flash #1: OPENLOOP_ENABLE = 1, MOTOR DESCONECTADO. Scope en
 *             OUT1/OUT2/OUT3 → 3 sinusoides desfasadas 120°, período 0.5 s
 *             (= 2 Hz eléctrico). Valida el firmware sin riesgo.
 *   Flash #2: motor conectado. Si gira despacio → éxito. Si vibra sin
 *             girar → swap fases B/C en el cable o cambia los offsets
 *             de idx_b/idx_c en openloop.c.
 * -------------------------------------------------------------------------- */
#define OPENLOOP_ENABLE        1
#define OPENLOOP_AMP_TICKS     170U     /* 10% de ARR=1700 → ~0.7 A pico */
#define OPENLOOP_DELTA_Q32     (OPENLOOP_DELTA_1HZ_ELEC * 10U)   /* 10 Hz eléctrico → 1.43 Hz mec con P=7 = 86 RPM */

/* --------------------------------------------------------------------------
 * Calibración de ganancia raw → A. Recompilar cambiando CAL_TARGET_PHASE
 * entre runs para calibrar A, B y C por separado.
 *
 *   CAL_TARGET_PHASE = 0 → calibra fase A (varía CCR1, multímetro en fase A)
 *   CAL_TARGET_PHASE = 1 → calibra fase B (varía CCR2, multímetro en fase B)
 *   CAL_TARGET_PHASE = 2 → calibra fase C (varía CCR3, multímetro en fase C)
 *   CAL_TARGET_PHASE = -1 → sin calibración (operación normal)
 * -------------------------------------------------------------------------- */
#define CAL_TARGET_PHASE   -1   /* -1 = operación normal (sin sweep) */

/* --------------------------------------------------------------------------
 * Sesión 13 — diagnóstico de medición de corriente (Task #14).
 *
 * Barre 3 amplitudes con f_e fija y reporta Pico/RMS. La intención es
 * discriminar entre:
 *   (A) Medición rota — TRGO en valle, OPAMP no estabiliza, etc. Síntoma:
 *       stats idénticas para los 3 amps.
 *   (B) Corriente real baja — stats escalan ~lineal con amp.
 *
 * Predicción si la cadena raw→mA está sana (sinusoide pura, K=59.6 raw/A):
 *   amp=  0 → pico_raw ≈ 0,  rms_raw ≈ 0   (solo ruido baseline)
 *   amp=170 → pico_raw ≈ 18, rms_raw ≈ 13  (170/PWM_ARR × K_phase_inv …)
 *   amp=510 → pico_raw ≈ 53, rms_raw ≈ 37
 *
 * El primer frame tras cada switch va marcado [transient]: cubre el
 * transitorio eléctrico (τ_e≈L/R≈320 μs) + cualquier reacción mecánica
 * antes de que entre el régimen permanente.
 *
 * Desactivar (volver al print de stats simple) cambiando a 0.
 * -------------------------------------------------------------------------- */
#define DIAG_AMP_SWEEP_ENABLE   1
#define DIAG_SECS_PER_AMP       30U

#if (DIAG_AMP_SWEEP_ENABLE == 1)
static const uint16_t DIAG_AMPS[] = { 0U, 170U, 510U };
#define DIAG_N_AMPS  (sizeof(DIAG_AMPS) / sizeof(DIAG_AMPS[0]))
#endif

#if (CAL_TARGET_PHASE >= 0)
/* Sweep: duties (en %, después convertidos a ticks de ARR=1700).
 * El punto i=0 (50%) es referencia: corriente debe ser ~0 A. */
static const uint16_t CAL_DUTIES_PCT_X10[] = { 500U, 520U, 550U, 600U, 650U, 700U };
#define CAL_N_POINTS  (sizeof(CAL_DUTIES_PCT_X10) / sizeof(CAL_DUTIES_PCT_X10[0]))

/* Tiempo por punto del sweep, en ms. */
#define CAL_TIME_PER_POINT_MS  15000U

/* Muestras a promediar por punto (para reducir ruido del raw). */
#define CAL_AVG_SAMPLES  1024U
#endif

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

#if (CAL_TARGET_PHASE >= 0)
/* Solo se usa dentro del bloque de calibración de ganancia. Wrapped para
 * evitar warning -Wunused-function cuando CAL_TARGET_PHASE = -1. */
static void delay_ms(uint32_t ms) {
    uint32_t start = g_ticks;
    while ((g_ticks - start) < ms) { }
}
#endif

/*
 * Imprime los registros clave del TIM1 con sus valores esperados al lado.
 * Sirve como checksum de configuración tras pwm_init() y pwm_enable().
 * Si alguna línea reporta un valor distinto al esperado → bug en pwm.c.
 */
static void pwm_dump_regs(const char *label) {
    printf("\r\n=== TIM1 registers (%s) ===\r\n", label);
    printf("CR1   = 0x%08lX\r\n", (unsigned long)TIM1->CR1);
    printf("CR2   = 0x%08lX  (MMS=111 = TRGO OC4REF → al pico del PWM)\r\n",
           (unsigned long)TIM1->CR2);
    printf("ARR   = %lu  (50 kHz @ 170 MHz)\r\n", (unsigned long)TIM1->ARR);
    printf("RCR   = %lu  (1 UEV per PWM period)\r\n", (unsigned long)TIM1->RCR);
    printf("CCMR1 = 0x%08lX  CCMR2 = 0x%08lX\r\n",
           (unsigned long)TIM1->CCMR1, (unsigned long)TIM1->CCMR2);
    printf("CCER  = 0x%08lX  (CCxE+CCxNE for x=1,2,3)\r\n",
           (unsigned long)TIM1->CCER);
    printf("CCR1=%lu  CCR2=%lu  CCR3=%lu  CCR4=%lu  (CCR4 debe ser ARR-1=%u)\r\n",
           (unsigned long)TIM1->CCR1, (unsigned long)TIM1->CCR2,
           (unsigned long)TIM1->CCR3, (unsigned long)TIM1->CCR4,
           (unsigned)(PWM_ARR - 1U));
    printf("BDTR  = 0x%08lX  (DTG=0x55 500ns, OSSI/OSSR=1, MOE=bit15)\r\n",
           (unsigned long)TIM1->BDTR);
    printf("===========================\r\n");
}

int main(void) {
    clock_init_170mhz_hse();
    systick_init(170000U);
    uart2_init(115200U);
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("\r\n[boot] STM32G431 @170MHz, USART2 OK (app: 02_pwm_adc)\r\n");

    pwm_init();
    printf("[pwm_init] OK\r\n");
    pwm_dump_regs("post-init, MOE=0");

    opamp_init();
    printf("[opamp_init] OK\r\n");

    adc_init();
    printf("[adc_init] OK\r\n");

    adc_isr_init();
    printf("[adc_isr_init] OK — JEOSIE on ADC2, NVIC ADC1_2 enabled (prio 1)\r\n");
    adc_dump_regs();

    /*
     * ⚠ Salidas activas. Precondiciones de seguridad:
     *   - Motor DESCONECTADO de J7.
     *   - Vbus (J5/J6) a 12V con CC ≥ 500 mA.
     */
    pwm_enable();
    printf("\r\n[pwm_enable] MOE=1, CEN=1 — 3 fases PWM @ 50 kHz, duty 50%%\r\n");
    pwm_dump_regs("post-enable, MOE=1");

    /* Calibración de offset DC. ⚠ Motor DEBE estar desconectado: lo que
     * promediamos aquí se asume que corresponde a corriente = 0 A. */
    printf("\r\n[calib] promediando 4096 muestras (~82 ms)...\r\n");
    adc_calibrate_offsets(4096U);
    printf("[calib] offsets: i_a=%4u  i_b=%4u  i_c=%4u  (raw)\r\n",
           g_offset_a, g_offset_b, g_offset_c);

    openloop_init();
#if (OPENLOOP_ENABLE == 1)
    openloop_set_amplitude(OPENLOOP_AMP_TICKS);
    openloop_set_freq_q32(OPENLOOP_DELTA_Q32);
    openloop_start();
    /* Aritmética entera (printf sin float):
     *   amp_pct_x10 = amp * 1000 / ARR (e.g. 170*1000/1700 = 100 = 10.0%)
     *   f_e_mHz     = delta * f_s * 1000 / 2^32 (con uint64_t intermedio) */
    uint32_t amp_pct_x10 = (uint32_t)OPENLOOP_AMP_TICKS * 1000U / PWM_ARR;
    uint32_t f_e_mhz = (uint32_t)(((uint64_t)OPENLOOP_DELTA_Q32 * 50000UL * 1000UL) >> 32);
    printf("\r\n[openloop] START amp=%u ticks (%lu.%lu%% PWM) delta=%lu (~%lu mHz_elec)\r\n",
           (unsigned)OPENLOOP_AMP_TICKS,
           (unsigned long)(amp_pct_x10 / 10U),
           (unsigned long)(amp_pct_x10 % 10U),
           (unsigned long)OPENLOOP_DELTA_Q32,
           (unsigned long)f_e_mhz);
#else
    printf("\r\n[openloop] DISABLED (duty 50%% balanceado).\r\n");
#endif

#if (CAL_TARGET_PHASE >= 0)
    /* ---- SWEEP DE CALIBRACIÓN DE GANANCIA ----
     * Procedimiento físico:
     *   1. Apagar Vbus.
     *   2. Insertar multímetro DC amperios EN SERIE con la fase elegida
     *      (cortar el cable entre J7 OUT_x y el motor, multímetro en medio).
     *   3. Volver a encender Vbus a 12 V (CC 1.5 A).
     *   4. Esperar el sweep (10 s grace + 6×15 s = ~100 s).
     *   5. Anotar la lectura del multímetro en cada punto.
     */
    const char target_letter = (CAL_TARGET_PHASE == 0) ? 'A'
                             : (CAL_TARGET_PHASE == 1) ? 'B' : 'C';

    printf("\r\n[cal_gain] === CALIBRACION DE GANANCIA fase %c ===\r\n", target_letter);
    printf("[cal_gain] Verifica: multimetro DC A en serie con fase %c. CC=1.5A.\r\n", target_letter);
    printf("[cal_gain] Sweep arranca en 10 s ...\r\n");
    for (int i = 10; i > 0; i--) {
        printf("[cal_gain]   %d ...\r\n", i);
        delay_ms(1000U);
    }

    for (uint32_t p = 0; p < CAL_N_POINTS; p++) {
        /* Convertir duty% (x10) a ticks de ARR. duty_ticks = ARR * duty%/100.
         * Para evitar overflow: (PWM_ARR * pct_x10) / 1000U.
         * Ej.: 50.0% → (1700 * 500) / 1000 = 850. */
        uint16_t target_duty = (uint16_t)((PWM_ARR * CAL_DUTIES_PCT_X10[p]) / 1000U);
        uint16_t other_duty  = (uint16_t)((PWM_ARR * 500U) / 1000U);  /* 50.0% */

        uint16_t duty_a = (CAL_TARGET_PHASE == 0) ? target_duty : other_duty;
        uint16_t duty_b = (CAL_TARGET_PHASE == 1) ? target_duty : other_duty;
        uint16_t duty_c = (CAL_TARGET_PHASE == 2) ? target_duty : other_duty;

        pwm_set_duties(duty_a, duty_b, duty_c);

        printf("\r\n[cal_gain] punto %lu/%lu: duty_%c=%u.%u%% (ticks=%u) — 15s\r\n",
               (unsigned long)(p + 1), (unsigned long)CAL_N_POINTS,
               target_letter,
               CAL_DUTIES_PCT_X10[p] / 10U, CAL_DUTIES_PCT_X10[p] % 10U,
               (unsigned)target_duty);

        /* Esperar 2 s a que la corriente se estabilice antes de promediar. */
        delay_ms(2000U);

        /* Promediar CAL_AVG_SAMPLES lecturas del raw de la fase objetivo.
         * Acumulamos sobre g_isr_count para no contar dos veces la misma muestra. */
        uint32_t sum = 0U;
        uint32_t prev_count = g_isr_count;
        uint32_t taken = 0U;
        while (taken < CAL_AVG_SAMPLES) {
            uint32_t now = g_isr_count;
            if (now != prev_count) {
                uint16_t v = (CAL_TARGET_PHASE == 0) ? g_ia_raw
                           : (CAL_TARGET_PHASE == 1) ? g_ib_raw : g_ic_raw;
                sum += v;
                taken++;
                prev_count = now;
            }
        }
        uint16_t raw_avg = (uint16_t)(sum / CAL_AVG_SAMPLES);
        uint16_t offset  = (CAL_TARGET_PHASE == 0) ? g_offset_a
                         : (CAL_TARGET_PHASE == 1) ? g_offset_b : g_offset_c;
        int16_t cal_avg  = (int16_t)raw_avg - (int16_t)offset;

        printf("[cal_gain] punto %lu RESULTADO: raw_avg=%u  cal_avg=%+d  ← ANOTA mA del multimetro\r\n",
               (unsigned long)(p + 1), raw_avg, cal_avg);

        /* Mantener el punto otros 13 s para que tomes la lectura tranquilo. */
        delay_ms(13000U);
    }

    /* Volver a duty 50% balanceado (corriente = 0). */
    pwm_set_duties(PWM_ARR / 2U, PWM_ARR / 2U, PWM_ARR / 2U);
    printf("\r\n[cal_gain] === SWEEP COMPLETO. Duty 50%% restablecido. ===\r\n");
    printf("[cal_gain] Pegame los 6 (raw_avg, mA_medidos) para calcular K.\r\n\r\n");
#endif

    uint32_t tick = 0;

#if (DIAG_AMP_SWEEP_ENABLE == 1)
    /* Override del amp inicial (OPENLOOP_AMP_TICKS de arriba) para arrancar
     * el sweep desde DIAG_AMPS[0]=0. Esto puede generar un "switch" inicial
     * silencioso vs lo que se imprimió arriba: el banner aquí lo aclara. */
    uint32_t amp_idx       = 0U;
    uint32_t frame_in_amp  = 0U;
    openloop_set_amplitude(DIAG_AMPS[0]);
    printf("\r\n[diag] === SWEEP arranca: amp=%u por %lus (predicción "
           "pico_raw≈0 si OK) ===\r\n",
           (unsigned)DIAG_AMPS[0], (unsigned long)DIAG_SECS_PER_AMP);
#endif

    while (1) {
        /* Espera frame de stats listo (1 por segundo). Mientras tanto el
         * main no consume CPU — la ISR sigue corriendo en background. */
        while (g_stats_ready == 0U) { __asm__ volatile("nop"); }

        /* Snapshot atómico de las stats. uint64_t no es atómico en M4 → IRQ off. */
        __disable_irq();
        uint16_t max_a_raw = g_stats_max_a_raw;
        uint16_t max_b_raw = g_stats_max_b_raw;
        uint16_t max_c_raw = g_stats_max_c_raw;
        uint64_t sumsq_a   = g_stats_sumsq_a;
        uint64_t sumsq_b   = g_stats_sumsq_b;
        uint64_t sumsq_c   = g_stats_sumsq_c;
        g_stats_ready = 0U;
        __enable_irq();

        /* RMS = sqrt(Σ i² / N). Σ i² / N cabe en uint32_t (max ~4.2M con
         * raw_cal max ~2048). isqrt32 es entero, sin float. */
        uint32_t rms_a_raw = adc_isqrt32((uint32_t)(sumsq_a / ADC_STATS_WINDOW));
        uint32_t rms_b_raw = adc_isqrt32((uint32_t)(sumsq_b / ADC_STATS_WINDOW));
        uint32_t rms_c_raw = adc_isqrt32((uint32_t)(sumsq_c / ADC_STATS_WINDOW));

        /* Conversión a mA. max_*_raw es uint16 < 32767 → cast seguro a int16. */
        int32_t max_a_ma = adc_raw_to_ma((int16_t)max_a_raw);
        int32_t max_b_ma = adc_raw_to_ma((int16_t)max_b_raw);
        int32_t max_c_ma = adc_raw_to_ma((int16_t)max_c_raw);
        int32_t rms_a_ma = adc_raw_to_ma((int16_t)rms_a_raw);
        int32_t rms_b_ma = adc_raw_to_ma((int16_t)rms_b_raw);
        int32_t rms_c_ma = adc_raw_to_ma((int16_t)rms_c_raw);

        /* Lectura instantánea de Vbus (regular, por software trigger). */
        uint16_t vbus_raw = adc_get_vbus_raw();

#if (DIAG_AMP_SWEEP_ENABLE == 1)
        /* Frame 1/30 del amp actual cubre el transitorio L/R + reacción
         * mecánica al cambio de amplitud. Frames 2..30 son régimen permanente. */
        const char *tag = (frame_in_amp == 0U) ? " [transient]" : "";
        printf("[s=%lu amp=%u f=%lu/%lu]%s PICO raw: %4u/%4u/%4u  mA: %5ld/%5ld/%5ld\r\n",
               (unsigned long)tick,
               (unsigned)DIAG_AMPS[amp_idx],
               (unsigned long)(frame_in_amp + 1U),
               (unsigned long)DIAG_SECS_PER_AMP,
               tag,
               max_a_raw, max_b_raw, max_c_raw,
               (long)max_a_ma, (long)max_b_ma, (long)max_c_ma);
        printf("                       RMS  raw: %4lu/%4lu/%4lu  mA: %5ld/%5ld/%5ld  | Vbus=%4u\r\n",
               (unsigned long)rms_a_raw, (unsigned long)rms_b_raw, (unsigned long)rms_c_raw,
               (long)rms_a_ma, (long)rms_b_ma, (long)rms_c_ma,
               vbus_raw);

        frame_in_amp++;
        if (frame_in_amp >= DIAG_SECS_PER_AMP) {
            frame_in_amp = 0U;
            amp_idx = (amp_idx + 1U) % DIAG_N_AMPS;
            openloop_set_amplitude(DIAG_AMPS[amp_idx]);
            printf("\r\n[diag] === SWITCH amp=%u (%lus, siguiente frame "
                   "marcado [transient]) ===\r\n",
                   (unsigned)DIAG_AMPS[amp_idx],
                   (unsigned long)DIAG_SECS_PER_AMP);
        }
#else
        printf("[s=%lu] PICO  raw: %4u/%4u/%4u  mA: %5ld/%5ld/%5ld\r\n",
               (unsigned long)tick,
               max_a_raw, max_b_raw, max_c_raw,
               (long)max_a_ma, (long)max_b_ma, (long)max_c_ma);
        printf("        RMS   raw: %4lu/%4lu/%4lu  mA: %5ld/%5ld/%5ld  | Vbus=%4u\r\n",
               (unsigned long)rms_a_raw, (unsigned long)rms_b_raw, (unsigned long)rms_c_raw,
               (long)rms_a_ma, (long)rms_b_ma, (long)rms_c_ma,
               vbus_raw);
#endif
        tick++;
    }
}
