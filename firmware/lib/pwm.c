/*
 * pwm.c — Implementación TIM1 PWM 3-fásico complementario @ 50 kHz.
 *
 * Estructura de la inicialización (FIELD_NOTES.md N1.3–N1.6, N1.8):
 *
 *   1. Clock gating (RCC):
 *        - AHB2ENR: GPIOA, GPIOB, GPIOC (los 6 pines de TIM1)
 *        - APB2ENR: TIM1
 *
 *   2. GPIOs en AF6:
 *        - PA8  (CH1),  PA9  (CH2),  PA10 (CH3)
 *        - PC13 (CH1N), PA12 (CH2N), PB15 (CH3N)
 *        - MODER = 0b10 (AF), AFR = 6, OSPEEDR = 0b11 (very high speed),
 *          OTYPER = 0 (push-pull), PUPDR = 0 (sin pull).
 *
 *   3. TIM1 base:
 *        - CR1.CMS    = 01     (center-aligned mode 1)
 *        - CR1.ARPE   = 1      (preload de ARR)
 *        - ARR        = 1700   (50 kHz @ 170 MHz, exacto; ver FIELD_NOTES N1.8)
 *        - RCR        = 1      (1 update event por periodo PWM completo)
 *
 *   4. Por canal x ∈ {1,2,3}:
 *        - CCMRx.OCxM  = 0110  (PWM mode 1)
 *        - CCMRx.OCxPE = 1     (preload de CCRx — anti-glitch)
 *        - CCER.CCxE   = 1     (enable salida principal)
 *        - CCER.CCxNE  = 1     (enable complementario)
 *        - CCER.CCxP   = 0     (active high; gate driver L6387 es active-high)
 *        - CCER.CCxNP  = 0     (active high)
 *        - CCRx        = ARR/2 (duty 50% como valor inicial seguro)
 *
 *   5. Dead-time + master enable (BDTR):
 *        - BDTR.DTG    = 0x55  (500 ns @ CKD = 00)
 *        - BDTR.OSSI   = 1     (timer controla salidas cuando MOE=0)
 *        - BDTR.OSSR   = 1     (timer controla salidas no habilitadas)
 *        - BDTR.MOE    = 0     (apagado — pwm_enable() lo activa)
 *
 *   6. TRGO al ADC (CR2):
 *        - CR2.MMS     = 010   (update event como TRGO)
 *        - CR2.OISx    = 0     (high-side inactivo si MOE=0)
 *        - CR2.OISxN   = 0     (low-side inactivo si MOE=0)
 *
 *   7. Cargar shadow registers:
 *        - EGR.UG      = 1     (force update event manual)
 *
 *   8. Habilitar/deshabilitar (pwm_enable / pwm_disable):
 *        - BDTR.MOE    = 1/0   (master output enable)
 *        - CR1.CEN     = 1/0   (counter enable)
 */

#include "pwm.h"
#include "stm32g431xx.h"


/* --------------------------------------------------------------------------
 * Helpers para construir máscaras de registros multi-bit en orden compacto.
 *
 * MODER es de 32 bits con 2 bits por pin: [pin0 lsb, pin0 msb, pin1 lsb, ...]
 * AFR  es array [AFRL, AFRH], 4 bits por pin: AFRL pins 0–7, AFRH pins 8–15.
 *
 * Se usan funciones inline para que el compilador resuelva todo a constantes
 * en tiempo de compilación (verificable en el .map / godbolt).
 * -------------------------------------------------------------------------- */

static inline void gpio_set_mode_af(GPIO_TypeDef *port, uint32_t pin) {
    /* MODER: bits 2*pin..2*pin+1. 0b10 = Alternate Function. */
    port->MODER &= ~(0x3U << (pin * 2U));
    port->MODER |=  (0x2U << (pin * 2U));
}

static inline void gpio_set_af(GPIO_TypeDef *port, uint32_t pin, uint32_t af) {
    /* AFR[0] = AFRL (pins 0..7), AFR[1] = AFRH (pins 8..15).
     * Cada pin ocupa 4 bits dentro de su half-word. */
    uint32_t idx = pin >> 3U;          /* 0 si pin<8, 1 si pin>=8 */
    uint32_t shift = (pin & 0x7U) * 4U;
    port->AFR[idx] &= ~(0xFU << shift);
    port->AFR[idx] |=  ((af & 0xFU) << shift);
}

static inline void gpio_set_speed_vhigh(GPIO_TypeDef *port, uint32_t pin) {
    /* OSPEEDR: bits 2*pin..2*pin+1. 0b11 = Very High Speed.
     * Importante para PWM: rise/fall time < 5 ns, asegura que el gate driver
     * vea flancos limpios y no introduzca delay extra. */
    port->OSPEEDR &= ~(0x3U << (pin * 2U));
    port->OSPEEDR |=  (0x3U << (pin * 2U));
}


/* --------------------------------------------------------------------------
 * pwm_init — configuración completa del TIM1 + GPIOs. Deja MOE=0, CEN=0.
 * Después de esta llamada las 6 salidas están armadas pero inhabilitadas.
 * -------------------------------------------------------------------------- */

void pwm_init(void) {
    /* ------------------ Paso 1: Clock gating ------------------ */

    /* GPIOA en AHB2 (RM0440 §7.4.17): pines PA8 (CH1), PA9 (CH2),
     * PA10 (CH3), PA12 (CH2N). */
    RCC->AHB2ENR |= RCC_AHB2ENR_GPIOAEN;

    /* GPIOB en AHB2: pin PB15 (CH3N). */
    RCC->AHB2ENR |= RCC_AHB2ENR_GPIOBEN;

    /* GPIOC en AHB2: pin PC13 (CH1N). */
    RCC->AHB2ENR |= RCC_AHB2ENR_GPIOCEN;

    /* TIM1 en APB2 (RM0440 §7.4.21). */
    RCC->APB2ENR |= RCC_APB2ENR_TIM1EN;

    /* Pequeña pausa para que el reloj del periférico se estabilice antes
     * de tocar sus registros. La errata de los STM32G4 menciona que
     * inmediatamente después de habilitar el clock el acceso al periférico
     * puede leer valores incorrectos. Patrón estándar: leer-back el bit
     * (lectura completa fuerza la sincronización del bus). */
    (void)RCC->AHB2ENR;
    (void)RCC->APB2ENR;


    /* ------------------ Paso 2: GPIOs en AF (varía por pin) ------------------
     *
     * ⚠ El AF para TIM1 NO es uniforme — DS12589 Tabla 13:
     *   - PA8/PA9/PA10 (CH1/CH2/CH3) → AF6
     *   - PA12 (CH2N)                → AF6
     *   - PC13 (CH1N)                → AF4   ← TIM1_CH1N (AF6 sería TIM8_CH4N)
     *   - PB15 (CH3N)                → AF4   ← TIM1_CH3N
     *
     * Si se asume AF6 universal, OUT1 y OUT3 quedan rotos porque PC13/PB15
     * salen con otra función (TIM8_CH4N en PC13, nada útil en PB15). Por eso
     * solo OUT2 conmuta — es la única fase cuyos pines (PA9+PA12) tienen
     * TIM1_CHx en AF6.
     */

    /* GPIOA: PA8/9/10/12 → AF6 (TIM1_CH1/CH2/CH3/CH2N). */
    gpio_set_mode_af   (GPIOA, 8U);
    gpio_set_af        (GPIOA, 8U, 6U);
    gpio_set_speed_vhigh(GPIOA, 8U);

    gpio_set_mode_af   (GPIOA, 9U);
    gpio_set_af        (GPIOA, 9U, 6U);
    gpio_set_speed_vhigh(GPIOA, 9U);

    gpio_set_mode_af   (GPIOA, 10U);
    gpio_set_af        (GPIOA, 10U, 6U);
    gpio_set_speed_vhigh(GPIOA, 10U);

    gpio_set_mode_af   (GPIOA, 12U);
    gpio_set_af        (GPIOA, 12U, 6U);
    gpio_set_speed_vhigh(GPIOA, 12U);

    /* GPIOB: PB15 → AF4 (TIM1_CH3N). NO AF6. */
    gpio_set_mode_af   (GPIOB, 15U);
    gpio_set_af        (GPIOB, 15U, 4U);
    gpio_set_speed_vhigh(GPIOB, 15U);

    /* GPIOC: PC13 → AF4 (TIM1_CH1N). NO AF6 (AF6 sería TIM8_CH4N). */
    gpio_set_mode_af   (GPIOC, 13U);
    gpio_set_af        (GPIOC, 13U, 4U);
    gpio_set_speed_vhigh(GPIOC, 13U);


    /* ------------------ Paso 3: TIM1 base ------------------ */

    /* Asegurar contador detenido antes de tocar la configuración
     * (si pwm_init se llamara dos veces, evita escribir mientras corre). */
    TIM1->CR1 &= ~TIM_CR1_CEN;

    /* CR1: center-aligned mode 1 + ARR preload enable.
     * - CMS = 01 (bits 6:5): center-aligned mode 1. Update event ocurre en
     *   overflow Y underflow (FIELD_NOTES N1.3).
     * - ARPE = 1 (bit 7): cambios en ARR se aplican en el siguiente update
     *   event, no inmediatamente. Anti-glitch.
     * - CKD[1:0] = 00 (bits 9:8, default): t_DTS = t_CK_INT = 1/170 MHz.
     *   Importante para que DTG=0x55 corresponda a 500 ns (FIELD_NOTES N1.5). */
    TIM1->CR1 = TIM_CR1_ARPE | (0x1U << TIM_CR1_CMS_Pos);

    /* PSC = 0 (default): no prescaler. tim_ker_ck = 170 MHz directo. */
    TIM1->PSC = 0U;

    /* ARR = 1700 → f_PWM = 170 MHz / (2 * 1700) = 50.000 kHz exactos. */
    TIM1->ARR = PWM_ARR;

    /* RCR = 1 → update event cada 2 over/underflows = 1 vez por periodo PWM.
     * (FIELD_NOTES N1.6: si RCR=0, habría 2 trigger events por periodo, lo
     * que dispararía el ADC al doble de la frecuencia deseada.) */
    TIM1->RCR = 1U;


    /* ------------------ Paso 4: Canales 1, 2, 3 ------------------ */

    /* CCMR1: configura los CHANNELS 1 y 2 (output mode).
     *
     * Channel 1 (low half):
     *   - CC1S[1:0]   bits 1:0  = 00  (output mode)
     *   - OC1FE       bit 2     = 0   (fast enable, no aplica en PWM clásico)
     *   - OC1PE       bit 3     = 1   (preload CCR1 → anti-glitch)
     *   - OC1M[2:0]   bits 6:4  = 110 (PWM mode 1)
     *   - OC1M[3]     bit 16    = 0   (PWM mode 1 cabe en 3 bits)
     *
     * Channel 2 (high half): símil, bits 8–15 y 24.
     *
     * Por simplicidad: escribimos directo con bitfields nombrados. */
    TIM1->CCMR1 = (0x6U << TIM_CCMR1_OC1M_Pos)  /* PWM mode 1 */
                | TIM_CCMR1_OC1PE                /* preload CCR1 */
                | (0x6U << TIM_CCMR1_OC2M_Pos)  /* PWM mode 1 */
                | TIM_CCMR1_OC2PE;               /* preload CCR2 */

    /* CCMR2: channel 3 (bits 0–7) y channel 4 (bits 8–15).
     *
     * Channel 4 SE USA — pero no como salida física, sino como SOURCE del
     * TRGO. OC4M=110 (PWM mode 1) + CCR4=ARR-1 + MMS=0111 (en CR2) hace que
     * TRGO suba al pico del contador (cerca de ARR), justo donde el low-side
     * está conduciendo y los shunts ven la corriente real. Es la fix de la
     * paradoja de sesión 12/13: con MMS=010 (UEV en center-aligned) el TRGO
     * caía en el VALLE, donde el low-side está OFF y los shunts miden cero.
     * OC4PE no es estrictamente necesario porque nunca cambiamos CCR4, pero
     * lo dejamos por simetría con OC1/2/3. */
    TIM1->CCMR2 = (0x6U << TIM_CCMR2_OC3M_Pos)  /* PWM mode 1 */
                | TIM_CCMR2_OC3PE                /* preload CCR3 */
                | (0x6U << TIM_CCMR2_OC4M_Pos)  /* PWM mode 1 (solo para REF interno) */
                | TIM_CCMR2_OC4PE;               /* preload CCR4 */

    /* CCER: enable salidas + polaridad.
     *
     * Bits por canal:
     *   - CCxE  : main output enable
     *   - CCxP  : main output polarity (0 = active high)
     *   - CCxNE : complementary output enable
     *   - CCxNP : complementary output polarity (0 = active high)
     *
     * Para active high (gate driver L6387) → CCxP = CCxNP = 0 (default por reset).
     * Solo necesitamos setear los enables. */
    TIM1->CCER = TIM_CCER_CC1E | TIM_CCER_CC1NE
               | TIM_CCER_CC2E | TIM_CCER_CC2NE
               | TIM_CCER_CC3E | TIM_CCER_CC3NE;

    /* Duty inicial 50% para las 3 fases. Estado eléctricamente neutro:
     * los 3 fases conmutan al mismo duty → voltaje promedio entre fases = 0V →
     * sin corriente neta a través del motor cuando MOE se prenda. */
    TIM1->CCR1 = PWM_ARR / 2U;
    TIM1->CCR2 = PWM_ARR / 2U;
    TIM1->CCR3 = PWM_ARR / 2U;

    /* CCR4 = ARR-1 → OC4REF tiene su flanco de subida en counter=ARR-2
     * (durante el down-count, justo después del pico). Con JEXTEN=01 (rising
     * edge) en el ADC, esto dispara el TRGO en el pico, dentro del intervalo
     * en que el low-side está ON (ventana ≈ 3.5 μs para amp=510 con DTG=500ns
     * → margen amplio). El conversion time del ADC (47.5+12.5 = 60 ciclos
     * @ 42.5 MHz ≈ 1.4 μs) cae enteramente dentro de esa ventana. */
    TIM1->CCR4 = PWM_ARR - 1U;


    /* ------------------ Paso 5: Dead-time + safety en BDTR ------------------ */

    /* BDTR config:
     *   - DTG[7:0]  bits 7:0    = 0x55 (500 ns, FIELD_NOTES N1.5)
     *   - LOCK[1:0] bits 9:8    = 00   (no lock)
     *   - OSSI      bit 10      = 1    (timer controla salidas cuando MOE=0)
     *   - OSSR      bit 11      = 1    (timer controla salidas no habilitadas)
     *   - BKE       bit 12      = 0    (sin break por ahora, ver N1.7)
     *   - BKP       bit 13      = 0
     *   - AOE       bit 14      = 0    (no re-enable automático tras break)
     *   - MOE       bit 15      = 0    (OUTPUT MASTER DISABLE — pwm_enable lo prende)
     *
     * OSSI = 1: cuando MOE=0, las salidas físicas siguen siendo controladas por
     * el timer y van al estado OISx/OISxN (que dejamos en 0 → todos OFF).
     * Lo opuesto (OSSI=0) liberaría las salidas al GPIO controller → posible
     * Hi-Z según el modo del GPIO, lo cual sería ambiguo para el gate driver. */
    TIM1->BDTR = (PWM_DEAD_DTG << TIM_BDTR_DTG_Pos)
               | TIM_BDTR_OSSI
               | TIM_BDTR_OSSR;
    /* MOE explícitamente NO seteado aquí — se enciende en pwm_enable(). */


    /* ------------------ Paso 6: TRGO + Output Idle States (CR2) ------------------ */

    /* CR2 config:
     *   - MMS[2:0]  bits 6:4   = 111  (OC4REF como TRGO → al ADC)
     *   - OISx, OISxN bits 8–13 = 0   (todas las salidas LOW cuando MOE=0)
     *   - TI1S, CCDS, CCUS, CCPC bits = 0 (defaults; no aplican aquí)
     *
     * Escribimos el valor completo del registro (no &= ni |=) porque queremos
     * estado conocido en TODOS los bits, no parchear el reset value.
     *
     * Sesión 13 — cambio MMS=010 (UEV) → MMS=111 (OC4REF). En center-aligned
     * mode 1 + RCR=1 la UEV caía en el underflow (counter=0 = valle del PWM
     * = high-side ON = shunts en cero), por eso las stats de corriente no
     * escalaban con la amplitud. OC4REF + CCR4=ARR-1 fuerza el trigger al
     * pico (counter≈ARR = low-side ON = shunts con corriente real). */
    TIM1->CR2 = (0x7U << TIM_CR2_MMS_Pos);


    /* ------------------ Paso 7: Cargar shadow registers ------------------ */

    /* EGR.UG = 1 fuerza un update event manual. Esto causa:
     *   - PSC se copia a su shadow (no aplica, PSC=0).
     *   - ARR se copia a su shadow → comparador del contador usa 1700.
     *   - CCRx se copian a sus shadows → comparadores usan ARR/2.
     *   - RCR se reload.
     *
     * Sin esta línea, los comparadores tendrían valores no inicializados en
     * el primer ciclo y los flancos del PWM serían erráticos. */
    TIM1->EGR = TIM_EGR_UG;
}


/* --------------------------------------------------------------------------
 * pwm_set_duties — escribe los 3 duty cycles (preload, aplicado en next UEV).
 *
 * Cada duty es un valor entero en [0, PWM_ARR]. Clampea en el caller
 * si es necesario. */
void pwm_set_duties(uint16_t duty_a, uint16_t duty_b, uint16_t duty_c) {
    TIM1->CCR1 = duty_a;
    TIM1->CCR2 = duty_b;
    TIM1->CCR3 = duty_c;
}


/* --------------------------------------------------------------------------
 * pwm_enable — habilita las 6 salidas (MOE) y arranca el contador (CEN).
 *
 * Orden importante: MOE antes que CEN. Si invirtiéramos:
 *   - CEN=1 con MOE=0 → contador corre pero salidas en estado OISx (0).
 *   - Cuando luego MOE=1 → salidas saltan a sus valores reales a mitad del
 *     ciclo PWM. Glitch en el primer flanco.
 *
 * Con MOE primero y CEN después, ambos coinciden al inicio del ciclo. */
void pwm_enable(void) {
    TIM1->BDTR |= TIM_BDTR_MOE;
    TIM1->CR1  |= TIM_CR1_CEN;
}


/* --------------------------------------------------------------------------
 * pwm_disable — detiene contador y apaga salidas.
 *
 * Orden inverso al enable: primero CEN=0 (contador se queda donde estaba),
 * después MOE=0 (salidas a OISx/OISxN = 0). */
void pwm_disable(void) {
    TIM1->CR1  &= ~TIM_CR1_CEN;
    TIM1->BDTR &= ~TIM_BDTR_MOE;
}
