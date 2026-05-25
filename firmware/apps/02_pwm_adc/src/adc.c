/*
 * adc.c — Implementación OPAMPs + ADC1/ADC2 dual injected simultaneous.
 *
 * Referencias: FIELD_NOTES N1.10–N1.12, RM0440 §21 (ADC) y §25 (OPAMP).
 *
 * Decisiones técnicas (FIELD_NOTES N1.11/N1.12):
 *   - OPAMPs: standalone mode (gain externa, R's del PCB), OPAINTOEN=1.
 *   - Auto-calibración OPAMP de offset (CALON loop, ~40 ms por OPAMP).
 *   - ADC: dual mode = injected simultaneous (DUAL=00101).
 *   - JEXTSEL=01001 (TIM1_TRGO), JEXTEN=01 (rising edge).
 *   - Sample time = 6.5 ciclos (SMP=001) → fast con OPAMP como fuente.
 *   - 12 bits de resolución.
 *   - ADC clock: HCLK/4 = 42.5 MHz (síncrono CKMODE=11).
 *
 * ⚠ TODOs marcados con [VERIFY] requieren confirmación al medir/calibrar.
 */

#include "adc.h"
#include "openloop.h"
#include "stm32g431xx.h"

/* --------------------------------------------------------------------------
 * Helpers
 * -------------------------------------------------------------------------- */

static void delay_cycles(volatile uint32_t n) {
    while (n--) { __asm__ volatile("nop"); }
}

/* Configurar pin GPIO como analog input (MODER=0b11). Para Vbus (PA0) y Temp (PB14). */
static void gpio_set_analog(GPIO_TypeDef *port, uint32_t pin) {
    port->MODER |= (0x3U << (pin * 2U));   /* analog */
    /* PUPDR queda en 00 por reset (no pull). Para entradas analógicas no debe haber pull. */
}


/* --------------------------------------------------------------------------
 * OPAMP auto-calibration helper.
 *
 * Procedimiento RM0440 §25.3.7:
 *   1. OPAEN = 1, CALON = 1.
 *   2. CALSEL=01 (P pair), incrementar TRIMOFFSETP de 0 hasta que CALOUT
 *      flip de 1 a 0. Guardar ese valor.
 *   3. CALSEL=11 (N pair), incrementar TRIMOFFSETN similar.
 *   4. CALON = 0, USERTRIM = 1.
 *
 * NOTA: Por simplicidad inicial, este código usa los valores de "factory
 * trim" (USERTRIM=0). La calibración por software se puede agregar después
 * si el offset DC es significativo en pruebas.
 * -------------------------------------------------------------------------- */
static void opamp_quick_enable(OPAMP_TypeDef *opamp, uint32_t vp_sel) {
    /* PGA MODE INTERNAL GAIN — feedback resistors están en el silicio.
     * Ventajas vs standalone:
     *   - Gain definida internamente, no depende del PCB.
     *   - Saturación menos probable porque la red de feedback siempre está cerrada.
     *
     * PGGAIN encoding (RM0440 §25.4.1):
     *   00000=x2, 00001=x4, 00010=x8, 00011=x16, 00100=x32, 00101=x64.
     *
     * Sesión 11 (2026-05-23): cambio de x2 → x16. Razón: con R_shunt=3 mΩ
     * y V_ref=3.3 V, x2 da solo 7.4 raw/A — sensibilidad insuficiente para
     * el rango operativo del banco (0.1-5 A). PGA x16 da 59.6 raw/A.
     * x32 saturaría el ADC porque el offset DC del front-end es alto.
     *
     * Config:
     *   - VPSEL: configurable, selecciona qué pin físico amplifica.
     *   - VMSEL = 10: PGA mode con feedback interno.
     *   - PGGAIN = 00011: gain x16.
     *   - OPAMPINTEN = 1: output al ADC sin pasar por pin físico. */
    opamp->CSR = (vp_sel << OPAMP_CSR_VPSEL_Pos)
               | (0x2U << OPAMP_CSR_VMSEL_Pos)   /* PGA mode */
               | (0x3U << OPAMP_CSR_PGGAIN_Pos)  /* gain x16 (PGGAIN=00011) */
               | OPAMP_CSR_OPAMPINTEN
               | OPAMP_CSR_OPAMPxEN;

    delay_cycles(1000U);
}


/* --------------------------------------------------------------------------
 * opamp_init — configura los 3 OPAMPs en standalone mode.
 *
 * Las conexiones de pines GPIO no son necesarias explícitamente para
 * VINP/VINM (cuando OPAEN=1, el silicio conecta automáticamente las
 * entradas según VP_SEL/VM_SEL).
 *
 * Para VOUT con OPAINTOEN=1, la salida va directo al ADC SIN pasar por
 * el pin físico, así que tampoco hay que configurar GPIOs de salida.
 *
 * Asunciones [VERIFY con esquemático MB1419 + medición]:
 *   - OPAMP1: VP_SEL=00 (VINP0 = PA1), VM_SEL=00 (VINM0 = PA3).
 *   - OPAMP2: VP_SEL=10 (VINP2 = PA7), VM_SEL=01 (VINM1 = PC5).
 *   - OPAMP3: VP_SEL=01 (VINP1 = PB0), VM_SEL=00 (VINM0 = PB2).
 * -------------------------------------------------------------------------- */
void opamp_init(void) {
    /* Los OPAMPs viven en el bus APB2 a través de SYSCFG. SIN este clock,
     * las escrituras a OPAMPx_CSR se ignoran silenciosamente (lo descubrí
     * tras ver OPAMPx_CSR = 0 después de escribirles en una iteración
     * anterior). */
    RCC->APB2ENR |= RCC_APB2ENR_SYSCFGEN;
    (void)RCC->APB2ENR;

    /* PGA mode gain x2 — VP_SEL=0 (VINP0) para todos los OPAMPs:
     *   - OPAMP1 VINP0 = PA1 (Curr_fdbk1_OPAmp+ per UM2516)
     *   - OPAMP2 VINP0 = PA7 (Curr_fdbk2_OPAmp+ per UM2516)
     *   - OPAMP3 VINP0 = PB0 (no documentado en UM2516; podría ser
     *     que la phase 3 use otro pin/topología) [VERIFY]
     */
    opamp_quick_enable(OPAMP1, 0U);
    opamp_quick_enable(OPAMP2, 0U);
    opamp_quick_enable(OPAMP3, 0U);
}


/* --------------------------------------------------------------------------
 * adc_init — configura ADC1 + ADC2.
 *
 * Pasos (orden estricto, RM0440 §21):
 *   1. Habilitar clock al bus ADC12 (AHB2ENR.ADC12EN).
 *   2. Configurar pines analógicos PA0 (Vbus) y PB14 (Temp).
 *   3. Configurar ADC clock prescaler en CCR (común a ambos ADCs).
 *   4. Salir de Deep Power Down y habilitar regulador interno.
 *   5. Calibrar (single-ended).
 *   6. Configurar dual mode = injected simultaneous.
 *   7. Configurar sample times.
 *   8. Configurar secuencias inyectadas (i_a en ADC1, i_b + i_c en ADC2).
 *   9. Configurar secuencia regular (Vbus + Temp en ADC1).
 *   10. Configurar trigger inyectado = TIM1_TRGO.
 *   11. Habilitar ADCs (ADEN, esperar ADRDY).
 *   12. Arrancar conversiones (ADSTART/JADSTART — esperan trigger).
 * -------------------------------------------------------------------------- */
void adc_init(void) {
    /* ---- Paso 1: clock ---- */
    RCC->AHB2ENR |= RCC_AHB2ENR_ADC12EN;
    (void)RCC->AHB2ENR;

    /* También GPIOA (ya habilitado por pwm_init) + GPIOB para PB14. */
    RCC->AHB2ENR |= RCC_AHB2ENR_GPIOAEN | RCC_AHB2ENR_GPIOBEN;

    /* ---- Paso 2: pines analógicos ---- */
    gpio_set_analog(GPIOA, 0U);   /* PA0 = Vbus_sense (ADC1_IN1) */
    gpio_set_analog(GPIOB, 14U);  /* PB14 = Temp feedback (ADC1_IN5) */

    /* ---- Paso 3: ADC clock prescaler ----
     * CCR es el common control register de ADC1+ADC2.
     * CKMODE = 11 → HCLK/4 síncrono. HCLK=170 MHz → ADC clock = 42.5 MHz. */
    ADC12_COMMON->CCR &= ~ADC_CCR_CKMODE_Msk;
    ADC12_COMMON->CCR |=  (0x3U << ADC_CCR_CKMODE_Pos);

    /* ---- Paso 4: salir de Deep Power Down + regulador ----
     * Para ambos ADCs. */
    ADC1->CR &= ~ADC_CR_DEEPPWD;
    ADC1->CR |=  ADC_CR_ADVREGEN;
    ADC2->CR &= ~ADC_CR_DEEPPWD;
    ADC2->CR |=  ADC_CR_ADVREGEN;
    delay_cycles(2000U);  /* estabilización del regulador interno (~20 μs) */

    /* ---- Paso 5: calibración ----
     * Modo single-ended (ADCALDIF=0). */
    ADC1->CR &= ~ADC_CR_ADCALDIF;
    ADC1->CR |=  ADC_CR_ADCAL;
    while (ADC1->CR & ADC_CR_ADCAL) { }

    ADC2->CR &= ~ADC_CR_ADCALDIF;
    ADC2->CR |=  ADC_CR_ADCAL;
    while (ADC2->CR & ADC_CR_ADCAL) { }

    /* ---- Paso 6: dual mode = injected simultaneous ----
     * DUAL = 00101 en CCR. */
    ADC12_COMMON->CCR &= ~ADC_CCR_DUAL_Msk;
    ADC12_COMMON->CCR |=  (0x5U << ADC_CCR_DUAL_Pos);

    /* ---- Paso 7: sample times ----
     * SMP=001 (6.5 ciclos) para todos los canales que usamos.
     * SMPR1 cubre canales 0-9, SMPR2 cubre 10-18.
     *
     * Canales relevantes en ADC1: 1 (Vbus), 5 (Temp), 13 (OPAMP1).
     * Canales relevantes en ADC2: 16 (OPAMP2), 18 (OPAMP3). */
    const uint32_t SMP_FAST = 0x1U;  /* 6.5 ciclos */

    /* ADC1->SMPR1: canales 1 y 5 (bits 3*1=3 y 3*5=15) */
    ADC1->SMPR1 |= (SMP_FAST << (3U * 1U)) | (SMP_FAST << (3U * 5U));
    /* ADC1->SMPR2: canal 13 (bits 3*(13-10)=9) */
    ADC1->SMPR2 |= (SMP_FAST << (3U * (13U - 10U)));

    /* ADC2->SMPR2: canales 16 y 18 (bits 3*6=18 y 3*8=24) */
    ADC2->SMPR2 |= (SMP_FAST << (3U * (16U - 10U)))
                | (SMP_FAST << (3U * (18U - 10U)));

    /* ---- Paso 8: secuencias inyectadas ----
     * JSQR: bits 0-1 = JL[1:0] (sequence length - 1), bits 2-6 = JEXTSEL,
     * bit 7-8 = JEXTEN, bits 9-13 = JSQ1, etc.
     *
     * ADC1 inyectado: 1 canal (i_a en OPAMP1 = ch 13). JL = 0.
     * ADC2 inyectado: 2 canales (i_b en ch 16, i_c en ch 18). JL = 1. */

    /* ADC1.JSQR: JL=0 (1 canal), JEXTSEL=01001 (TIM1_TRGO), JEXTEN=01 (rising),
     *            JSQ1 = canal 13 (OPAMP1). */
    ADC1->JSQR = (0U << ADC_JSQR_JL_Pos)
               | (0x0U << ADC_JSQR_JEXTSEL_Pos)  /* RM0440 Tabla 167: 00000 = TIM1_TRGO para JEXTSEL (NO 01001 que es para EXTSEL regular) */
               | (0x1U << ADC_JSQR_JEXTEN_Pos)
               | (ADC_CHAN_OPAMP1 << ADC_JSQR_JSQ1_Pos);

    /* ADC2.JSQR: JL=1 (2 canales), JEXTSEL/EXTEN don't care en slave (master
     *            controla el trigger), JSQ1=ch16, JSQ2=ch18.
     *
     * Aunque RM0440 dice "don't care", por buena práctica programamos el
     * mismo trigger en ambos. */
    ADC2->JSQR = (1U << ADC_JSQR_JL_Pos)
               | (0x0U << ADC_JSQR_JEXTSEL_Pos)  /* RM0440 Tabla 167: 00000 = TIM1_TRGO para JEXTSEL (NO 01001 que es para EXTSEL regular) */
               | (0x1U << ADC_JSQR_JEXTEN_Pos)
               | (ADC_CHAN_OPAMP2 << ADC_JSQR_JSQ1_Pos)
               | (ADC_CHAN_OPAMP3 << ADC_JSQR_JSQ2_Pos);

    /* ---- Paso 9: secuencia regular en ADC1 ----
     * Vbus y Temp. Trigger por software (no TIM1) — los leeremos en el
     * loop principal cuando lo necesitemos, sin sincronía con PWM.
     *
     * SQR1: L=1 (2 canales -1), SQ1=ch1 (Vbus), SQ2=ch5 (Temp). */
    ADC1->SQR1 = (1U << ADC_SQR1_L_Pos)
               | (ADC_CHAN_VBUS << ADC_SQR1_SQ1_Pos)
               | (ADC_CHAN_TEMP << ADC_SQR1_SQ2_Pos);

    /* CFGR: EXTEN=00 (software trigger only for regular).
     * Resolución por default = 12-bit (RES=00). */
    ADC1->CFGR &= ~(ADC_CFGR_EXTEN_Msk | ADC_CFGR_RES_Msk);

    /* ---- Paso 11: habilitar ADCs ---- */
    ADC1->ISR = ADC_ISR_ADRDY;  /* limpiar flag */
    ADC1->CR |= ADC_CR_ADEN;
    while ((ADC1->ISR & ADC_ISR_ADRDY) == 0U) { }
    ADC1->ISR = ADC_ISR_ADRDY;

    ADC2->ISR = ADC_ISR_ADRDY;
    ADC2->CR |= ADC_CR_ADEN;
    while ((ADC2->ISR & ADC_ISR_ADRDY) == 0U) { }
    ADC2->ISR = ADC_ISR_ADRDY;

    /* ---- Paso 12: arrancar conversiones inyectadas (esperan trigger) ---- */
    ADC1->CR |= ADC_CR_JADSTART;
    /* En dual injected simultaneous, JADSTART del ADC2 se setea automático,
     * pero por defensa lo seteamos también. */
    ADC2->CR |= ADC_CR_JADSTART;
}


/* --------------------------------------------------------------------------
 * Lecturas raw para diagnóstico (12 bits, 0-4095).
 * -------------------------------------------------------------------------- */

uint16_t adc_get_vbus_raw(void) {
    /* Disparo software de la secuencia regular y espera EOS. */
    ADC1->ISR = ADC_ISR_EOS | ADC_ISR_EOC;
    ADC1->CR |= ADC_CR_ADSTART;
    while ((ADC1->ISR & ADC_ISR_EOC) == 0U) { }
    uint16_t vbus = (uint16_t)(ADC1->DR & 0xFFFFU);
    /* Continúa la secuencia con el siguiente canal (Temp); lo descartamos aquí. */
    while ((ADC1->ISR & ADC_ISR_EOS) == 0U) {
        if (ADC1->ISR & ADC_ISR_EOC) (void)ADC1->DR;
    }
    ADC1->ISR = ADC_ISR_EOS;
    return vbus;
}

uint16_t adc_get_temp_raw(void) {
    /* Para diagnóstico simple: hacer una secuencia y devolver el segundo valor.
     * Versión más eficiente vendría con DMA. */
    ADC1->ISR = ADC_ISR_EOS | ADC_ISR_EOC;
    ADC1->CR |= ADC_CR_ADSTART;
    while ((ADC1->ISR & ADC_ISR_EOC) == 0U) { }
    (void)ADC1->DR;  /* descarta Vbus */
    while ((ADC1->ISR & ADC_ISR_EOC) == 0U) { }
    uint16_t temp = (uint16_t)(ADC1->DR & 0xFFFFU);
    ADC1->ISR = ADC_ISR_EOS;
    return temp;
}

void adc_get_currents_raw(uint16_t *ia, uint16_t *ib, uint16_t *ic) {
    /* Los JDRs se actualizan automáticamente con cada trigger inyectado (TIM1_TRGO).
     * Solo leemos los valores actuales — pueden ser de cualquier ciclo PWM reciente. */
    if (ia) *ia = (uint16_t)(ADC1->JDR1 & 0xFFFFU);
    if (ib) *ib = (uint16_t)(ADC2->JDR1 & 0xFFFFU);
    if (ic) *ic = (uint16_t)(ADC2->JDR2 & 0xFFFFU);
}


/* --------------------------------------------------------------------------
 * Dump de registros — análogo al pwm_dump_regs() de main.c.
 * -------------------------------------------------------------------------- */
#include <stdio.h>

void adc_dump_regs(void) {
    printf("\r\n=== OPAMP / ADC registers ===\r\n");
    printf("OPAMP1.CSR = 0x%08lX  (OPAEN=bit0)\r\n", (unsigned long)OPAMP1->CSR);
    printf("OPAMP2.CSR = 0x%08lX\r\n", (unsigned long)OPAMP2->CSR);
    printf("OPAMP3.CSR = 0x%08lX\r\n", (unsigned long)OPAMP3->CSR);

    printf("ADC12_COMMON.CCR = 0x%08lX  (CKMODE=bits 17:16, DUAL=bits 4:0)\r\n",
           (unsigned long)ADC12_COMMON->CCR);

    printf("ADC1.CR    = 0x%08lX  (ADEN=bit0, JADSTART=bit3)\r\n",
           (unsigned long)ADC1->CR);
    printf("ADC1.JSQR  = 0x%08lX  (JL+JEXTSEL+JEXTEN+JSQ1)\r\n",
           (unsigned long)ADC1->JSQR);
    printf("ADC1.SQR1  = 0x%08lX  (L+SQ1+SQ2 for Vbus+Temp)\r\n",
           (unsigned long)ADC1->SQR1);
    printf("ADC1.SMPR2 = 0x%08lX  (canal 13 = OPAMP1)\r\n",
           (unsigned long)ADC1->SMPR2);

    printf("ADC2.CR    = 0x%08lX\r\n", (unsigned long)ADC2->CR);
    printf("ADC2.IER   = 0x%08lX  (JEOSIE=bit6 esperado tras adc_isr_init)\r\n",
           (unsigned long)ADC2->IER);
    printf("ADC2.JSQR  = 0x%08lX  (JL+canales 16,18)\r\n",
           (unsigned long)ADC2->JSQR);
    printf("ADC2.SMPR2 = 0x%08lX\r\n", (unsigned long)ADC2->SMPR2);
    printf("=============================\r\n");
}


/* --------------------------------------------------------------------------
 * ISR JEOS — esqueleto del lazo de control de 50 kHz (FIELD_NOTES N1.14).
 *
 * Disparo: ADC2 levanta JEOS cuando termina i_c (último canal de su secuencia
 * inyectada de 2 canales). ADC1 ya terminó antes (1 solo canal, i_a) y su
 * valor espera en JDR1.
 *
 * Línea NVIC: ADC1_2_IRQn (= 18). Handler compartido entre ADC1 y ADC2;
 * solo ADC2 dispara IRQ (JEOSIE en ADC1 queda en 0).
 *
 * Pin de instrumentación: PB8 — toggle al entrar y salir para medir
 * frecuencia y duración con scope. PB8 corresponde a Z+/H3 del J8,
 * reservado en sesión 2 para esto. No interfiere con I²C1 (PB6/PB7,
 * AS5600) ni con TIM1 PWM (PA8/9/10/12, PB15, PC13).
 * -------------------------------------------------------------------------- */

volatile uint16_t g_ia_raw    = 0U;
volatile uint16_t g_ib_raw    = 0U;
volatile uint16_t g_ic_raw    = 0U;
volatile uint32_t g_isr_count = 0U;

/* --------------------------------------------------------------------------
 * Estadísticas pico/RMS — acumuladores internos + buffer "result" expuesto.
 * -------------------------------------------------------------------------- */
static volatile uint16_t stats_max_a_acc  = 0U;
static volatile uint16_t stats_max_b_acc  = 0U;
static volatile uint16_t stats_max_c_acc  = 0U;
static volatile uint64_t stats_sumsq_a_acc = 0U;
static volatile uint64_t stats_sumsq_b_acc = 0U;
static volatile uint64_t stats_sumsq_c_acc = 0U;
static volatile uint32_t stats_count       = 0U;

volatile uint16_t g_stats_max_a_raw  = 0U;
volatile uint16_t g_stats_max_b_raw  = 0U;
volatile uint16_t g_stats_max_c_raw  = 0U;
volatile uint64_t g_stats_sumsq_a    = 0U;
volatile uint64_t g_stats_sumsq_b    = 0U;
volatile uint64_t g_stats_sumsq_c    = 0U;
volatile uint8_t  g_stats_ready      = 0U;

/* --------------------------------------------------------------------------
 * sqrt entera de 32 bits — algoritmo "digit-by-digit" en base 4.
 * ~20 ciclos en Cortex-M4. Resultado exacto: floor(sqrt(x)).
 * -------------------------------------------------------------------------- */
uint32_t adc_isqrt32(uint32_t x) {
    uint32_t r = 0U;
    uint32_t b = 0x40000000U;
    while (b > x) { b >>= 2; }
    while (b != 0U) {
        uint32_t t = r + b;
        r >>= 1;
        if (x >= t) {
            x -= t;
            r += b;
        }
        b >>= 2;
    }
    return r;
}

/* --------------------------------------------------------------------------
 * Calibración de offset DC — state machine dentro del handler.
 *
 * Estados (cal_state):
 *   0 = idle (handler solo lee JDR, sin acumular).
 *   1 = running (handler acumula y decrementa contador).
 *   2 = done (handler vuelve a estado idle).
 *
 * El handler nunca calcula el promedio (división); eso lo hace el main al
 * detectar cal_state==2. Esto mantiene el handler en ~30 ciclos incluso
 * durante la calibración.
 *
 * Aritmética: cada muestra es 12 bits (max 4095). N=32768 muestras × 4095
 * = ~134M, cabe en uint32_t (max 4.29G). Para N=4096 (típico, ~82 ms a
 * 50 kHz), suma máxima = 16.8M.
 * -------------------------------------------------------------------------- */

uint16_t g_offset_a = 0U;
uint16_t g_offset_b = 0U;
uint16_t g_offset_c = 0U;

static volatile uint8_t  cal_state    = 0U;  /* 0=idle, 1=running, 2=done */
static volatile uint16_t cal_remain   = 0U;  /* muestras restantes; 0 → terminó */
static volatile uint32_t cal_acc_a    = 0U;
static volatile uint32_t cal_acc_b    = 0U;
static volatile uint32_t cal_acc_c    = 0U;

void adc_calibrate_offsets(uint16_t n_samples_pow2) {
    /* Calcular shift a partir de N: 4096 = 2^12 → shift = 12. */
    uint8_t shift = 0U;
    uint16_t n = n_samples_pow2;
    while (n > 1U) { n >>= 1U; shift++; }

    /* Reset acumuladores y arranque atómico de la calibración. La ISR
     * empieza a acumular en el próximo disparo. */
    cal_acc_a  = 0U;
    cal_acc_b  = 0U;
    cal_acc_c  = 0U;
    cal_remain = n_samples_pow2;
    cal_state  = 1U;

    /* Espera bloqueante hasta que la ISR termine. ~N * 20 μs. */
    while (cal_state != 2U) { __asm__ volatile("nop"); }

    /* División por shift right (potencia de 2). 100% entero. */
    g_offset_a = (uint16_t)(cal_acc_a >> shift);
    g_offset_b = (uint16_t)(cal_acc_b >> shift);
    g_offset_c = (uint16_t)(cal_acc_c >> shift);

    cal_state = 0U;  /* idle */
}

void adc_isr_init(void) {
    /* GPIOB clock probablemente ya habilitado por adc_init (PB14 = Temp),
     * pero lo aseguramos por si el orden cambia más adelante. */
    RCC->AHB2ENR |= RCC_AHB2ENR_GPIOBEN;
    (void)RCC->AHB2ENR;

    /* PB8 = general purpose output, push-pull, sin pull, speed default (low).
     * Bits del MODER para pin 8: 17:16. */
    GPIOB->MODER &= ~(0x3U << (8U * 2U));
    GPIOB->MODER |=  (0x1U << (8U * 2U));   /* MODER=01 = output */

    /* JEOSIE en ADC2 (no ADC1). Sin esto, JEOS solo levanta el flag pero
     * el NVIC nunca dispara. */
    ADC2->IER |= ADC_IER_JEOSIE;

    /* NVIC: prioridad 1 (alta pero no máxima — reservar 0 para faults).
     * Habilitar la línea ADC1_2_IRQn = 18. */
    NVIC_SetPriority(ADC1_2_IRQn, 1U);
    NVIC_EnableIRQ(ADC1_2_IRQn);
}

void ADC1_2_IRQHandler(void) {
    /* PB8 HIGH — borde de subida = entrada al handler. */
    GPIOB->BSRR = (1U << 8U);

    /* Snapshot atómico de los 3 JDRs. JDR* es de 16 bits útiles (resultado
     * de 12 bits alineado a la derecha). */
    uint16_t ia = (uint16_t)(ADC1->JDR1 & 0xFFFFU);
    uint16_t ib = (uint16_t)(ADC2->JDR1 & 0xFFFFU);
    uint16_t ic = (uint16_t)(ADC2->JDR2 & 0xFFFFU);

    g_ia_raw = ia;
    g_ib_raw = ib;
    g_ic_raw = ic;

    g_isr_count++;

    /* Branch de calibración. En estado idle, la comparación con 1 falla
     * y el handler ejecuta sin penalty extra (~1 ciclo). */
    if (cal_state == 1U) {
        cal_acc_a += ia;
        cal_acc_b += ib;
        cal_acc_c += ic;
        if (--cal_remain == 0U) {
            cal_state = 2U;  /* main hace la división y limpia */
        }
    }

    /* Stats pico + RMS. Solo si la calibración de offset terminó. */
    if (g_offset_a != 0U) {
        int32_t ia_c = (int32_t)ia - (int32_t)g_offset_a;
        int32_t ib_c = (int32_t)ib - (int32_t)g_offset_b;
        int32_t ic_c = (int32_t)ic - (int32_t)g_offset_c;

        uint16_t abs_a = (uint16_t)((ia_c < 0) ? -ia_c : ia_c);
        uint16_t abs_b = (uint16_t)((ib_c < 0) ? -ib_c : ib_c);
        uint16_t abs_c = (uint16_t)((ic_c < 0) ? -ic_c : ic_c);

        if (abs_a > stats_max_a_acc) stats_max_a_acc = abs_a;
        if (abs_b > stats_max_b_acc) stats_max_b_acc = abs_b;
        if (abs_c > stats_max_c_acc) stats_max_c_acc = abs_c;

        stats_sumsq_a_acc += (uint32_t)(ia_c * ia_c);
        stats_sumsq_b_acc += (uint32_t)(ib_c * ib_c);
        stats_sumsq_c_acc += (uint32_t)(ic_c * ic_c);

        if (++stats_count >= ADC_STATS_WINDOW) {
            /* Snapshot a "result" globals si el main ya consumió el anterior. */
            if (g_stats_ready == 0U) {
                g_stats_max_a_raw = stats_max_a_acc;
                g_stats_max_b_raw = stats_max_b_acc;
                g_stats_max_c_raw = stats_max_c_acc;
                g_stats_sumsq_a   = stats_sumsq_a_acc;
                g_stats_sumsq_b   = stats_sumsq_b_acc;
                g_stats_sumsq_c   = stats_sumsq_c_acc;
                g_stats_ready     = 1U;
            }
            /* Si el main aún no consumió, descartamos este frame
             * (el siguiente sobrescribe los acumuladores). */
            stats_max_a_acc   = 0U;
            stats_max_b_acc   = 0U;
            stats_max_c_acc   = 0U;
            stats_sumsq_a_acc = 0U;
            stats_sumsq_b_acc = 0U;
            stats_sumsq_c_acc = 0U;
            stats_count       = 0U;
        }
    }

    /* Excitación open-loop: avanza theta, calcula sinusoides 120°
     * desfasadas, actualiza CCR1/2/3. Si openloop no está corriendo
     * (s_running == 0), retorna en ~3 ciclos. */
    openloop_step();

    /* Limpiar el flag JEOS de ADC2 (write-1-to-clear). Sin esto, al return
     * del handler el flag sigue activo → NVIC reentra inmediatamente. */
    ADC2->ISR = ADC_ISR_JEOS;

    /* PB8 LOW — borde de bajada = salida del handler. */
    GPIOB->BSRR = (1U << (8U + 16U));
}
