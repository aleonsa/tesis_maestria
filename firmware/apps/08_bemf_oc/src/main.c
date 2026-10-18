/*
 * main.c — app 08_bemf_oc: BEMF del motor en circuito abierto, medida directa.
 * PLANNING §11, Exp. 2b (sesión 24).
 *
 * POR QUÉ
 *
 * Reconstruir e = u − R·i − L·di/dt desde la tensión comandada deja los
 * armónicos del 2804 en "≤ 2 %" pero no los resuelve: el dead-time y R se
 * confunden con ellos (sesión 23, Exp. 3a). La placa tiene la red de sensado
 * de BEMF del 6-step (esquemático MB1419 hoja 4): por fase, OUTx ─10 kΩ─┬─
 * 2.2 kΩ─(D)─GPIO_BEMF, con el nodo en PA4/PC4/PB11 (BEMF1/2/3). Con el
 * puente apagado y el rotor girado a mano se lee e_abc(θ_e) sin modelo.
 *
 * QUÉ HACE
 *
 *   - Los seis pines de compuerta (PA8/PA9/PA10, PC13/PA12/PB15) como GPIO en
 *     BAJO: los seis MOSFET apagados por construcción. TIM1 no se inicializa.
 *   - adc_init() calibra y habilita los ADC (la secuencia inyectada queda
 *     armada esperando un TIM1 que nunca dispara). Aquí se añade la secuencia
 *     REGULAR del ADC2: IN17 (PA4), IN5 (PC4), IN14 (PB11), disparo por
 *     software, 92.5 ciclos de muestreo (fuente ≈ 1.8 kΩ).
 *   - TIM2 a 50 kHz (prio 1): PLL + bombeo del encoder cada tick, como en 04 y
 *     06 (las ganancias del PLL son por tick a 50 kHz). Cada 5 ticks (10 kHz)
 *     convierte los tres canales (polling, ~7 µs) y cada 4 conversiones
 *     (2.5 kHz) encola (θ_e, ω̂, va, vb, vc) para el UART.
 *   - GPIO_BEMF (PB5): al arrancar, con el motor quieto, prueba Hi-Z / ALTO /
 *     BAJO y imprime el nivel DC de los tres canales en cada modo (qué
 *     polarización deja margen a los semiciclos negativos); luego fija el
 *     modo de BEMF_GPIO_MODE.
 *
 * Stream (2.5 kHz, 921600 baud): B <seq> <θ_e/65536> <rpm×10> <va> <vb> <vc>
 * (raw del ADC, 12 bits; la fase es ≈ 5.55 × raw × 0.806 mV con el divisor
 * 10k/2.2k — verificar el ratio con el DMM). Estado a 1 Hz: [oc] ...
 *
 * Con las tres fases en abierto los tres divisores forman un neutro virtual:
 * e_x − ē = v_x − mean(v). La forma, los armónicos y λ salen offline
 * (python_scripts/bemf_oc_analyze.py). Girar la llanta a mano desde ~600 rpm
 * y dejar decaer: todas las velocidades en una pasada, y J·b del decaimiento.
 */

#include <stdio.h>
#include "stm32g431xx.h"
#include "clock.h"
#include "uart.h"
#include "i2c.h"
#include "as5600.h"
#include "theta_pll.h"
#include "adc.h"

#define PHASE_A_OFFSET_RAW   2563U   /* 2558 + 4.6: +2.85° e medidos en 08 (sesión 24) */
#define ENCODER_SIGN         (+1)
#define THETA_E_OFFSET_Q32   (THETA_PLL_POLE_PAIRS * (PHASE_A_OFFSET_RAW << 20))

#define UART_BAUD            921600U
#define TIM2_ARR             (3400U - 1U)      /* 170 MHz / 3400 = 50 kHz */
#define ADC_EVERY_TICKS      5U                /* 10 kHz de conversión    */
#define TX_EVERY_CONV        4U                /* 2.5 kHz al UART         */

/* 0 = Hi-Z (entrada), 1 = salida ALTO, 2 = salida BAJO. Medido (sesión 24,
 * motor quieto): Hi-Z deja el nodo flotando (deriva hasta saturar); ALTO lo
 * cuelga de 3.3 V con los terminales pegados a 0 V por los diodos de cuerpo;
 * BAJO da el divisor 10k/2.2k a GND (ratio ≈ 0.18) pero con el neutro en
 * ~0 V: los semiciclos negativos se recortan. La red es para 6-step, donde el
 * inversor fija el modo común. Solución: modo BAJO + UNA resistencia externa
 * (10 kΩ de +12 V a OUT1, o 2.2 kΩ del pad 5 V de J8 a OUT1) que sube el modo
 * común de los tres terminales (unidos por los devanados) a ~3.3 V → nodo a
 * ~0.6 V, margen para ±0.45 V de BEMF en el nodo (≈ 1200 rpm). */
#define BEMF_GPIO_MODE       2

#define ADC_CH_BEMF1         17U   /* PA4  */
#define ADC_CH_BEMF2         5U    /* PC4  */
#define ADC_CH_BEMF3         14U   /* PB11 */
#define SMP_92_5             5U    /* SMPx = 101 → 92.5 ciclos */

/* Cola ISR → main. Potencia de 2. */
#define Q_N                  1024U
typedef struct {
    uint32_t seq;
    uint32_t theta_e;
    int32_t  omega;
    uint16_t v[3];
} sample_t;
static volatile sample_t g_q[Q_N];
static volatile uint32_t g_q_head, g_q_tail;    /* head: ISR escribe; tail: main lee */
static volatile uint32_t g_q_dropped;

static volatile uint32_t g_ticks;
static volatile uint32_t g_pub_corr, g_pub_i2c_err, g_pub_cyc_max, g_pub_conv;
static volatile uint8_t  g_need_i2c_reset;
static volatile uint8_t  g_run;                 /* 0 hasta terminar [bias]  */

void SysTick_Handler(void) { g_ticks++; }
void SystemInit(void) { }

static void systick_init(uint32_t ticks_per_irq) {
    SysTick->LOAD = ticks_per_irq - 1U;
    SysTick->VAL  = 0U;
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk | SysTick_CTRL_TICKINT_Msk | SysTick_CTRL_ENABLE_Msk;
}

/* ── Seis compuertas en bajo ─────────────────────────────────────────── */
static void gates_off(void) {
    RCC->AHB2ENR |= RCC_AHB2ENR_GPIOAEN | RCC_AHB2ENR_GPIOBEN | RCC_AHB2ENR_GPIOCEN;
    (void)RCC->AHB2ENR;
    /* PA8, PA9, PA10, PA12 */
    for (uint32_t p = 8U; p <= 12U; p++) {
        if (p == 11U) continue;
        GPIOA->BSRR  = (1U << (p + 16U));
        GPIOA->MODER = (GPIOA->MODER & ~(0x3U << (2U * p))) | (0x1U << (2U * p));
    }
    GPIOB->BSRR  = (1U << (15U + 16U));
    GPIOB->MODER = (GPIOB->MODER & ~(0x3U << 30)) | (0x1U << 30);
    GPIOC->BSRR  = (1U << (13U + 16U));
    GPIOC->MODER = (GPIOC->MODER & ~(0x3U << 26)) | (0x1U << 26);
}

/* ── GPIO_BEMF (PB5) ─────────────────────────────────────────────────── */
static void bemf_gpio_mode(unsigned mode) {
    GPIOB->MODER &= ~(0x3U << 10);                /* entrada */
    GPIOB->PUPDR &= ~(0x3U << 10);
    if (mode == 1U) { GPIOB->BSRR = (1U << 5);          GPIOB->MODER |= (0x1U << 10); }
    if (mode == 2U) { GPIOB->BSRR = (1U << (5U + 16U)); GPIOB->MODER |= (0x1U << 10); }
}

/* ── ADC2 regular: tres canales de BEMF ──────────────────────────────── */
static void bemf_adc_init(void) {
    /* PA4, PC4, PB11 en analógico (MODER = 11). */
    GPIOA->MODER |= (0x3U << 8);
    GPIOC->MODER |= (0x3U << 8);
    GPIOB->MODER |= (0x3U << 22);
    /* Tiempo de muestreo: SMPR1 canales 0-9 (ch5 → bits 15), SMPR2 10-18
     * (ch14 → bits 12, ch17 → bits 21). */
    ADC2->SMPR1 = (ADC2->SMPR1 & ~(0x7U << 15)) | (SMP_92_5 << 15);
    ADC2->SMPR2 = (ADC2->SMPR2 & ~((0x7U << 12) | (0x7U << 21)))
                | (SMP_92_5 << 12) | (SMP_92_5 << 21);
    /* Secuencia regular: L = 2 (3 conversiones). Disparo por software
     * (EXTEN = 00), una pasada (CONT = 0). */
    ADC2->SQR1 = (2U << ADC_SQR1_L_Pos)
               | (ADC_CH_BEMF1 << ADC_SQR1_SQ1_Pos)
               | (ADC_CH_BEMF2 << ADC_SQR1_SQ2_Pos)
               | (ADC_CH_BEMF3 << ADC_SQR1_SQ3_Pos);
    ADC2->CFGR &= ~(ADC_CFGR_EXTEN_Msk | ADC_CFGR_CONT | ADC_CFGR_DMAEN);
    ADC2->ISR = ADC_ISR_EOC | ADC_ISR_EOS | ADC_ISR_OVR;
}

/* Conversión de los tres canales por polling (~3 × (92.5 + 12.5) / 42.5 MHz ≈ 7.4 µs). */
static void bemf_adc_read(uint16_t v[3]) {
    ADC2->ISR = ADC_ISR_EOC | ADC_ISR_EOS | ADC_ISR_OVR;
    ADC2->CR |= ADC_CR_ADSTART;
    for (unsigned k = 0U; k < 3U; k++) {
        while ((ADC2->ISR & ADC_ISR_EOC) == 0U) { }
        v[k] = (uint16_t)(ADC2->DR & 0xFFFFU);   /* leer DR limpia EOC */
    }
    while ((ADC2->ISR & ADC_ISR_EOS) == 0U) { }
    ADC2->ISR = ADC_ISR_EOS;
}

/* ── ISR a 50 kHz ────────────────────────────────────────────────────── */
void TIM2_IRQHandler(void) {
    const uint32_t cyc0 = DWT->CYCCNT;
    TIM2->SR = ~TIM_SR_UIF;

    theta_pll_extrapolate();
    uint16_t raw;
    switch (as5600_nb_tick(&raw)) {
        case AS5600_NB_SAMPLE: theta_pll_update(raw); g_pub_corr++; break;
        case AS5600_NB_ERROR:  g_pub_i2c_err++;                     break;
        case AS5600_NB_STUCK:  g_need_i2c_reset = 1U;               break;
        default:                                                     break;
    }

    static uint32_t tick, conv;
    if (g_run && (++tick >= ADC_EVERY_TICKS)) {
        tick = 0U;
        uint16_t v[3];
        bemf_adc_read(v);
        g_pub_conv++;
        if (++conv >= TX_EVERY_CONV) {
            conv = 0U;
            uint32_t theta_e = theta_pll_theta_elec_comp_q32() - THETA_E_OFFSET_Q32;
#if (ENCODER_SIGN < 0)
            theta_e = 0U - theta_e;
#endif
            const uint32_t next = (g_q_head + 1U) & (Q_N - 1U);
            if (next == g_q_tail) {
                g_q_dropped++;
            } else {
                volatile sample_t *s = &g_q[g_q_head];
                s->seq = g_pub_conv; s->theta_e = theta_e;
                s->omega = theta_pll_omega_tick_q32();
                s->v[0] = v[0]; s->v[1] = v[1]; s->v[2] = v[2];
                g_q_head = next;
            }
        }
    }

    const uint32_t cyc = DWT->CYCCNT - cyc0;
    if (cyc > g_pub_cyc_max) g_pub_cyc_max = cyc;
}

static void tim2_init_50khz(void) {
    RCC->APB1ENR1 |= RCC_APB1ENR1_TIM2EN;
    (void)RCC->APB1ENR1;
    TIM2->PSC  = 0U;
    TIM2->ARR  = TIM2_ARR;
    TIM2->DIER = TIM_DIER_UIE;
    NVIC_SetPriority(TIM2_IRQn, 1);
    NVIC_EnableIRQ(TIM2_IRQn);
    TIM2->CR1 = TIM_CR1_CEN;
}

/* Media de n conversiones (foreground, con la ISR aún sin convertir). */
static void bias_probe(const char *label) {
    uint32_t acc[3] = {0U, 0U, 0U};
    uint16_t v[3];
    for (unsigned n = 0U; n < 256U; n++) {
        bemf_adc_read(v);
        acc[0] += v[0]; acc[1] += v[1]; acc[2] += v[2];
    }
    printf("[bias] GPIO_BEMF %-5s: va=%4lu vb=%4lu vc=%4lu raw  (%lu/%lu/%lu mV en el nodo)\r\n",
           label, (unsigned long)(acc[0] >> 8), (unsigned long)(acc[1] >> 8), (unsigned long)(acc[2] >> 8),
           (unsigned long)((acc[0] >> 8) * 3300U / 4096U), (unsigned long)((acc[1] >> 8) * 3300U / 4096U),
           (unsigned long)((acc[2] >> 8) * 3300U / 4096U));
}

static void delay_ms(uint32_t ms) {
    const uint32_t t0 = g_ticks;
    while ((g_ticks - t0) < ms) { }
}

int main(void) {
    clock_init_170mhz_hse();
    systick_init(170000U);
    uart2_init(UART_BAUD);
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("\r\n[boot] STM32G431 @170MHz (app: 08_bemf_oc)\r\n");
    gates_off();
    printf("[gates] PA8/PA9/PA10/PC13/PA12/PB15 en BAJO: puente apagado, TIM1 sin inicializar\r\n");

    i2c1_init();
    i2c1_scan();
    i2c1_nb_init();
    uint16_t raw0 = 0U;
    (void)as5600_raw_angle(&raw0);
    theta_pll_init((uint32_t)raw0 << 20);

    adc_init();            /* reloj, calibración, ADEN; inyectado armado sin disparo */
    bemf_adc_init();

    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0U;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

    /* Polarización del nodo con el motor quieto, en los tres modos. */
    printf("[bias] motor QUIETO: nivel DC de BEMF1/2/3 (PA4/PC4/PB11) segun GPIO_BEMF (PB5)\r\n");
    bemf_gpio_mode(0U); delay_ms(300U); bias_probe("Hi-Z");
    bemf_gpio_mode(1U); delay_ms(300U); bias_probe("ALTO");
    bemf_gpio_mode(2U); delay_ms(300U); bias_probe("BAJO");
    bemf_gpio_mode(BEMF_GPIO_MODE); delay_ms(300U);
    printf("[bias] modo elegido: %s\r\n", (BEMF_GPIO_MODE == 0) ? "Hi-Z" : (BEMF_GPIO_MODE == 1) ? "ALTO" : "BAJO");

    tim2_init_50khz();
    printf("[log] B seq th w va vb vc  (2.5 kHz; th: 65536 = vuelta e; w: rpm x10; v: raw ADC 12 bit)\r\n");
    printf("[oc] gira la llanta a mano y dejala decaer\r\n");
    g_run = 1U;

    uint32_t t_stat = 0U, corr_prev = 0U, conv_prev = 0U;
    while (1) {
        while (g_q_tail != g_q_head) {
            volatile sample_t *s = &g_q[g_q_tail];
            const int32_t rpm10 = (int32_t)(((int64_t)s->omega * 30000000LL) >> 32);
            printf("B %lu %u %ld %u %u %u\r\n",
                   (unsigned long)s->seq, (unsigned)(s->theta_e >> 16), (long)rpm10,
                   (unsigned)s->v[0], (unsigned)s->v[1], (unsigned)s->v[2]);
            g_q_tail = (g_q_tail + 1U) & (Q_N - 1U);
        }

        if ((g_ticks - t_stat) >= 1000U) {
            t_stat = g_ticks;
            printf("[oc] corr/s=%lu i2c_err=%lu conv/s=%lu drop=%lu isr_max=%lu.%02lu us\r\n",
                   (unsigned long)(g_pub_corr - corr_prev), (unsigned long)g_pub_i2c_err,
                   (unsigned long)(g_pub_conv - conv_prev), (unsigned long)g_q_dropped,
                   (unsigned long)(g_pub_cyc_max / 170U), (unsigned long)((g_pub_cyc_max % 170U) * 100U / 170U));
            corr_prev = g_pub_corr; conv_prev = g_pub_conv;
        }

        if (g_need_i2c_reset != 0U) {
            i2c1_reset_peripheral();
            i2c1_nb_abort();
            g_need_i2c_reset = 0U;
        }
    }
}
