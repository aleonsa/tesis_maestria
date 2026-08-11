/*
 * i2c.c — I2C1 master, polling mode, sobre PB8 (SCL) / PB7 (SDA).
 *
 * Cross-ref: RM0440 §40 (I2C), §7.4.27 (RCC_CCIPR / I2C1SEL), §11 (GPIO).
 *
 * Decisiones de diseño:
 *   - Kernel clock = HSI16 (16 MHz), NO PCLK1 (170 MHz). Desacopla el I2C
 *     del PLL y permite usar el TIMINGR canónico de ST tabulado para 16 MHz.
 *   - Standard mode 100 kHz: robusto para bring-up; el AS5600 soporta hasta
 *     FM+ 1 MHz, subimos después si conviene.
 *   - Polling, sin IRQ/DMA: igual que el primer uart2. Una cosa a la vez.
 *     (Ver la nota de latencia al final: para el lazo de control esto NO
 *     alcanza y habrá que migrarlo.)
 *   - Todas las esperas llevan timeout y vigilan NACKF. Un esclavo mudo
 *     produce un código de error, no un cuelgue.
 *
 * ⚠ PINOUT — SCL va en PB8, no en PB6 (sesión 2026-08-10):
 *   DS12589 Tabla 13, fila PB6: la casilla de AF4 está VACÍA. No es que ahí
 *   viva otra función — es que no vive ninguna. Un AF sin asignar deja el
 *   driver de salida sin fuente de señal y el pin emite 0; con open-drain eso
 *   hunde la línea permanentemente, el bus nunca arranca y todo probe da
 *   TIMEOUT.
 *
 *     PB6  AF4 = (vacío)      ← NO usar para I2C
 *     PB7  AF4 = I2C1_SDA
 *     PB8  AF4 = I2C1_SCL
 *
 *   Se descubrió empíricamente (barrido de AF0..AF15 y comparación PB6/PB8
 *   sobre la misma red de J8) porque el DS12589 no estaba en el repo; ya está
 *   en papers/. Es reincidencia del patrón de N1.9: el alternate function es
 *   propio de cada PIN, no del periférico.
 *
 *   Mapeo vigente en J8:  pad 2 (B+/H2) = PB7 = SDA
 *                         pad 3 (Z+/H3) = PB8 = SCL
 *                         pad 1 (A+/H1) = PB6 = libre
 */

#include <stdio.h>
#include "stm32g431xx.h"
#include "i2c.h"

/*
 * TIMINGR para I2CCLK = 16 MHz, SCL = 100 kHz (Standard mode).
 * Valor canónico de ST (RM0440 tabla de ejemplos de timing / CubeMX).
 *
 * Desglose (t_I2CCLK = 1/16 MHz = 62.5 ns):
 *   PRESC  = 0x3 → t_PRESC = (3+1)·62.5 ns = 250 ns
 *   SCLL   = 0x13 = 19 → t_LOW  = (19+1)·250 ns = 5.00 µs   (≥ 4.7 µs SM)  ✓
 *   SCLH   = 0x0F = 15 → t_HIGH = (15+1)·250 ns = 4.00 µs   (≥ 4.0 µs SM)  ✓
 *   SDADEL = 0x2  → 2·250 ns = 500 ns
 *   SCLDEL = 0x4  → (4+1)·250 ns = 1250 ns
 *   t_SCL ≈ t_LOW + t_HIGH = 9.0 µs (+ rise/fall) → ~100 kHz reales.
 */
#define I2C1_TIMINGR_100KHZ_16MHZ   0x30420F13U

/*
 * Guardia de las esperas. Cada iteración es una lectura de ISR (acceso a bus
 * de periférico) + test + decremento + salto: del orden de 10 ciclos a
 * 170 MHz. 200 000 iteraciones ≈ 12 ms.
 *
 * El orden de magnitud es lo que importa: una transacción legítima de 2 bytes
 * a 100 kHz son ~450 µs, así que 12 ms deja factor ~25 de margen y aun así el
 * fallo se siente instantáneo. Es un tope, no una medida.
 */
#define I2C_WAIT_LOOPS   200000U

#define I2C_SCL_PIN   8U
#define I2C_SDA_PIN   7U

const char *i2c_status_str(i2c_status_t st) {
    switch (st) {
        case I2C_OK:                return "OK";
        case I2C_ERR_BUSY:          return "BUSY (bus ocupado al arrancar)";
        case I2C_ERR_NACK_ADDR:     return "NACK de direccion (nadie contesta)";
        case I2C_ERR_NACK_DATA:     return "NACK de dato";
        case I2C_ERR_TIMEOUT_TXIS:  return "TIMEOUT esperando TXIS";
        case I2C_ERR_TIMEOUT_TC:    return "TIMEOUT esperando TC";
        case I2C_ERR_TIMEOUT_RXNE:  return "TIMEOUT esperando RXNE";
        case I2C_ERR_TIMEOUT_STOPF: return "TIMEOUT esperando STOPF";
        default:                    return "??";
    }
}

/* Configura un pin de GPIOB como alternate function open-drain.
 * Genérico en el número de pin: AFR[0] cubre 0-7, AFR[1] cubre 8-15. */
static void gpio_af_od(uint32_t pin, uint32_t af) {
    GPIOB->MODER   &= ~(0x3U << (pin * 2U));
    GPIOB->MODER   |=  (0x2U << (pin * 2U));      /* AF */
    GPIOB->OTYPER  |=  (1U << pin);                /* open-drain */
    GPIOB->OSPEEDR |=  (0x3U << (pin * 2U));       /* alta velocidad */
    GPIOB->PUPDR   &= ~(0x3U << (pin * 2U));       /* sin pull interno */
    GPIOB->AFR[pin >> 3U] &= ~(0xFU << ((pin & 7U) * 4U));
    GPIOB->AFR[pin >> 3U] |=  (af   << ((pin & 7U) * 4U));
}

void i2c1_init(void) {
    /* ---- Paso 1: kernel clock del I2C = HSI16 ----
     * El PLL corre desde HSE; HSI16 es un oscilador aparte. Lo encendemos
     * explícitamente y esperamos a que esté listo antes de enrutarlo al I2C.
     * RCC_CCIPR.I2C1SEL: 00=PCLK1, 01=SYSCLK, 10=HSI16. Queremos 0b10, que
     * es exactamente el bit RCC_CCIPR_I2C1SEL_1. */
    RCC->CR |= RCC_CR_HSION;
    while ((RCC->CR & RCC_CR_HSIRDY) == 0U) { }

    RCC->CCIPR &= ~RCC_CCIPR_I2C1SEL;
    RCC->CCIPR |=  RCC_CCIPR_I2C1SEL_1;   /* 0b10 = HSI16 */

    /* ---- Paso 2: clocks de periférico ----
     * GPIOB en AHB2, I2C1 en APB1. Sin estos, los registros se ignoran. */
    RCC->AHB2ENR  |= RCC_AHB2ENR_GPIOBEN;
    RCC->APB1ENR1 |= RCC_APB1ENR1_I2C1EN;

    /* ---- Paso 3: GPIOs en AF4, open-drain, sin pull interno ----
     *
     * Open-drain es OBLIGATORIO en I2C: la línea solo se tira a 0; el '1' lo
     * provee el pull-up. Con push-pull, dos dispositivos hablando a la vez
     * sería un cortocircuito.
     *
     * PUPDR = 00 (sin pull interno) a propósito. El módulo AS5600 está
     * alimentado a 5 V y trae sus propios pull-ups. Activar el pull-up interno
     * del STM32 (que va a VDD = 3.3 V) crearía, con la línea en reposo a 5 V,
     * un camino de corriente desde el bus hacia el riel de 3.3 V a través del
     * pin FT. Los pines son 5V-tolerant, pero la práctica correcta con bus de
     * 5 V es NO usar el pull interno.
     *
     * La placa además aporta lo suyo: el esquemático MB1419 hoja 5 muestra que
     * cada línea de J8 lleva 1.8 kΩ EN SERIE (R74/R75/R77), 10 kΩ de pull-up
     * (R71/R72/R73) y un clamp Schottky. Esa red está pensada para sensores
     * Hall push-pull, no para open-drain; medida a 100 kHz pasa sin problema,
     * pero es la primera sospechosa si algún día subimos a Fast Mode Plus. */
    gpio_af_od(I2C_SCL_PIN, 4U);
    gpio_af_od(I2C_SDA_PIN, 4U);

    /* ---- Paso 4: TIMINGR con PE = 0, luego PE = 1 ----
     * TIMINGR solo se puede escribir con el periférico deshabilitado.
     * Patrón "enable último", igual que UART/TIM1. */
    I2C1->CR1 &= ~I2C_CR1_PE;
    I2C1->TIMINGR = I2C1_TIMINGR_100KHZ_16MHZ;
    I2C1->CR1 |= I2C_CR1_PE;
}

void i2c1_reset_peripheral(void) {
    I2C1->CR1 &= ~I2C_CR1_PE;
    /* RM0440 §40.4.1: PE debe quedar en 0 al menos 3 ciclos de APB para que el
     * reset interno se complete. A 170 MHz son ~18 ns; el bucle corto es más
     * honesto que confiar en el timing que elija el compilador. */
    for (volatile uint32_t d = 0U; d < 100U; d++) { }
    I2C1->CR1 |= I2C_CR1_PE;
}

/*
 * Espera a que suba `flag`, abortando si aparece NACKF o si se agota la
 * guardia. Es el único lugar donde este archivo hace polling.
 */
static i2c_status_t i2c_wait(uint32_t flag,
                             i2c_status_t on_timeout,
                             i2c_status_t on_nack) {
    uint32_t guard = I2C_WAIT_LOOPS;
    for (;;) {
        uint32_t isr = I2C1->ISR;
        if ((isr & flag) != 0U)            return I2C_OK;
        if ((isr & I2C_ISR_NACKF) != 0U)   return on_nack;
        if (--guard == 0U)                 return on_timeout;
    }
}

/*
 * Deja el periférico en estado usable después de un fallo.
 *
 * Con AUTOEND = 0 el maestro RETIENE el bus tras un NACK (SCL en hold): si no
 * soltamos un STOP a mano, la siguiente transacción arranca sobre un bus que
 * nunca se liberó. Con AUTOEND = 1 el hardware ya lo mandó y pedirlo de nuevo
 * es inocuo. La guardia cubre el caso de bus físicamente clavado.
 */
static void i2c_abort(void) {
    if ((I2C1->ISR & I2C_ISR_STOPF) == 0U) {
        I2C1->CR2 |= I2C_CR2_STOP;
        uint32_t guard = I2C_WAIT_LOOPS;
        while (((I2C1->ISR & I2C_ISR_STOPF) == 0U) && (--guard != 0U)) { }
    }
    I2C1->ICR = I2C_ICR_STOPCF | I2C_ICR_NACKCF
              | I2C_ICR_BERRCF | I2C_ICR_ARLOCF;
}

i2c_status_t i2c1_read_regs(uint8_t addr7, uint8_t reg, uint8_t *buf, uint8_t n) {
    i2c_status_t st;

    /* El bus debe estar libre antes de arrancar. BUSY pegado en 1 significa
     * que alguien retiene SCL o SDA — diagnóstico distinto a "no contesta". */
    {
        uint32_t guard = I2C_WAIT_LOOPS;
        while (((I2C1->ISR & I2C_ISR_BUSY) != 0U) && (--guard != 0U)) { }
        if (guard == 0U) return I2C_ERR_BUSY;
    }

    I2C1->ICR = I2C_ICR_NACKCF | I2C_ICR_STOPCF;

    /* ---- Fase 1: escribir el puntero de registro (1 byte, sin STOP) ----
     *
     * CR2 en una sola escritura (estado conocido en todos los campos):
     *   SADD[7:1] = dirección 7-bit << 1
     *   NBYTES    = 1        (mandamos 1 byte: el número de registro)
     *   RD_WRN    = 0        (write)
     *   AUTOEND   = 0        (NO STOP: haremos repeated-START)
     * Luego START dispara el arranque + envío de dirección. */
    I2C1->CR2 = ((uint32_t)addr7 << (I2C_CR2_SADD_Pos + 1U))
              | (1U << I2C_CR2_NBYTES_Pos);
    I2C1->CR2 |= I2C_CR2_START;

    /* TXIS = 1 cuando TXDR está libre para el dato (tras el ACK de dirección).
     * Si el esclavo no hace ACK, sube NACKF en su lugar. */
    st = i2c_wait(I2C_ISR_TXIS, I2C_ERR_TIMEOUT_TXIS, I2C_ERR_NACK_ADDR);
    if (st != I2C_OK) { i2c_abort(); return st; }
    I2C1->TXDR = reg;

    /* TC = 1 cuando NBYTES se envió y AUTOEND=0: el bus queda en hold
     * esperando repeated-START o STOP. */
    st = i2c_wait(I2C_ISR_TC, I2C_ERR_TIMEOUT_TC, I2C_ERR_NACK_DATA);
    if (st != I2C_OK) { i2c_abort(); return st; }

    /* ---- Fase 2: repeated-START + lectura (AUTOEND manda el STOP) ----
     * El AS5600 auto-incrementa el puntero, así que n bytes consecutivos
     * salen de registros consecutivos (0x0C, 0x0D, ...) en un solo viaje. */
    I2C1->CR2 = ((uint32_t)addr7 << (I2C_CR2_SADD_Pos + 1U))
              | ((uint32_t)n << I2C_CR2_NBYTES_Pos)
              | I2C_CR2_RD_WRN
              | I2C_CR2_AUTOEND;
    I2C1->CR2 |= I2C_CR2_START;

    for (uint8_t i = 0U; i < n; i++) {
        st = i2c_wait(I2C_ISR_RXNE, I2C_ERR_TIMEOUT_RXNE, I2C_ERR_NACK_ADDR);
        if (st != I2C_OK) { i2c_abort(); return st; }
        buf[i] = (uint8_t)I2C1->RXDR;
    }

    st = i2c_wait(I2C_ISR_STOPF, I2C_ERR_TIMEOUT_STOPF, I2C_ERR_NACK_ADDR);
    if (st != I2C_OK) { i2c_abort(); return st; }
    I2C1->ICR = I2C_ICR_STOPCF;

    return I2C_OK;
}

int i2c1_probe(uint8_t addr7) {
    /* Escritura de CERO bytes: la transacción más corta que existe. Con
     * NBYTES=0 y AUTOEND=1 el hardware manda START + dirección + STOP sin
     * tocar TXDR. Lo único que interesa es si hubo ACK.
     *
     * La distinción NACK vs TIMEOUT es la herramienta de diagnóstico más útil
     * de este archivo: NACK significa que el reloj corrió hasta el noveno
     * pulso (bus sano, nadie en esa dirección); TIMEOUT significa que la
     * transacción ni arrancó (bus roto o SCL mal mapeada). Esa distinción fue
     * la que localizó el pin equivocado el 2026-08-10. */
    I2C1->ICR = I2C_ICR_NACKCF | I2C_ICR_STOPCF;

    I2C1->CR2 = ((uint32_t)addr7 << (I2C_CR2_SADD_Pos + 1U))
              | (0U << I2C_CR2_NBYTES_Pos)
              | I2C_CR2_AUTOEND;
    I2C1->CR2 |= I2C_CR2_START;

    uint32_t guard = I2C_WAIT_LOOPS;
    while (((I2C1->ISR & I2C_ISR_STOPF) == 0U) && (--guard != 0U)) { }
    if (guard == 0U) { i2c_abort(); return -1; }

    int present = ((I2C1->ISR & I2C_ISR_NACKF) == 0U) ? 1 : 0;
    I2C1->ICR = I2C_ICR_STOPCF | I2C_ICR_NACKCF;
    return present;
}

int i2c1_scan(void) {
    int found = 0;
    int timeouts = 0;

    printf("[scan] barriendo 0x08..0x77 ...\r\n");

    /* 0x00-0x07 y 0x78-0x7F están reservados por el estándar I2C. */
    for (uint8_t a = 0x08U; a <= 0x77U; a++) {
        int r = i2c1_probe(a);
        if (r == 1) {
            printf("[scan]   ACK en 0x%02X%s\r\n", (unsigned)a,
                   (a == 0x36U) ? "   <-- AS5600" : "");
            found++;
        } else if (r < 0) {
            timeouts++;
        }
    }

    if (timeouts > 0) {
        printf("[scan] %d direcciones dieron TIMEOUT (bus no completa el STOP)\r\n",
               timeouts);
    }
    printf("[scan] dispositivos encontrados: %d\r\n", found);

    return (timeouts >= 0x70) ? -1 : found;
}

void i2c1_lines(uint8_t *scl_high, uint8_t *sda_high, uint8_t *busy) {
    /* El Schmitt trigger de entrada sigue conectado aunque el pin esté en
     * modo alternate function, así que IDR refleja el nivel real del cable.
     * Distingue "bus en reposo sano, el esclavo no habla" de "línea clavada". */
    uint32_t idr = GPIOB->IDR;
    if (scl_high != 0) *scl_high = ((idr & (1U << I2C_SCL_PIN)) != 0U) ? 1U : 0U;
    if (sda_high != 0) *sda_high = ((idr & (1U << I2C_SDA_PIN)) != 0U) ? 1U : 0U;
    if (busy     != 0) *busy     = ((I2C1->ISR & I2C_ISR_BUSY) != 0U) ? 1U : 0U;
}

void i2c1_dump_regs(void) {
    /* Metodología de N1.9: dejar de inferir y leer lo que el silicio tiene.
     * TIMINGR es el renglón decisivo: si vuelve 0 pese a habérsele escrito,
     * el periférico no acepta escrituras (sin clock gating o en reset). */
    printf("[dump] RCC->APB1ENR1 = 0x%08lX   (I2C1EN=%u)\r\n",
           (unsigned long)RCC->APB1ENR1,
           (RCC->APB1ENR1 & RCC_APB1ENR1_I2C1EN) ? 1U : 0U);
    printf("[dump] RCC->APB1RSTR1= 0x%08lX   (I2C1RST=%u)\r\n",
           (unsigned long)RCC->APB1RSTR1,
           (RCC->APB1RSTR1 & RCC_APB1RSTR1_I2C1RST) ? 1U : 0U);
    printf("[dump] RCC->CCIPR    = 0x%08lX   (I2C1SEL=%lu, 2=HSI16)\r\n",
           (unsigned long)RCC->CCIPR,
           (unsigned long)((RCC->CCIPR & RCC_CCIPR_I2C1SEL) >> RCC_CCIPR_I2C1SEL_Pos));
    printf("[dump] I2C1->CR1     = 0x%08lX   (PE=%u)\r\n",
           (unsigned long)I2C1->CR1,
           (I2C1->CR1 & I2C_CR1_PE) ? 1U : 0U);
    printf("[dump] I2C1->TIMINGR = 0x%08lX   (esperado 0x%08lX)%s\r\n",
           (unsigned long)I2C1->TIMINGR,
           (unsigned long)I2C1_TIMINGR_100KHZ_16MHZ,
           (I2C1->TIMINGR == I2C1_TIMINGR_100KHZ_16MHZ) ? "" : "   <== NO COINCIDE");
    printf("[dump] I2C1->ISR     = 0x%08lX   (BUSY=%u)\r\n",
           (unsigned long)I2C1->ISR,
           (I2C1->ISR & I2C_ISR_BUSY) ? 1U : 0U);
    printf("[dump] GPIOB MODER PB%u=%lu PB%u=%lu (AF=2)   AFR PB%u=AF%lu PB%u=AF%lu\r\n",
           (unsigned)I2C_SCL_PIN,
           (unsigned long)((GPIOB->MODER >> (I2C_SCL_PIN * 2U)) & 0x3U),
           (unsigned)I2C_SDA_PIN,
           (unsigned long)((GPIOB->MODER >> (I2C_SDA_PIN * 2U)) & 0x3U),
           (unsigned)I2C_SCL_PIN,
           (unsigned long)((GPIOB->AFR[I2C_SCL_PIN >> 3U] >> ((I2C_SCL_PIN & 7U) * 4U)) & 0xFU),
           (unsigned)I2C_SDA_PIN,
           (unsigned long)((GPIOB->AFR[I2C_SDA_PIN >> 3U] >> ((I2C_SDA_PIN & 7U) * 4U)) & 0xFU));
}

/*
 * ─────────────────────────────────────────────────────────────────────────
 * LÍMITE CONOCIDO — este driver no cabe dentro de la ISR de control.
 *
 * Una lectura de ángulo cuesta ~45 pulsos de SCL (START + dirección + ACK,
 * registro + ACK, repeated-START + dirección + ACK, 2 bytes + ACK). A 100 kHz
 * cada pulso son 10 µs, así que la transacción completa son ~450 µs.
 *
 * El periodo de la ISR a 50 kHz son 20 µs. La lectura tarda 22× MÁS que el
 * ciclo de control entero. Subir a Fast Mode Plus (1 MHz) la deja en ~45 µs,
 * todavía más del doble del presupuesto.
 *
 * La salida combina tres cosas: DMA o IRQ para que la transferencia avance en
 * segundo plano, leer a la tasa real del sensor (~7 kHz, su refresco interno)
 * en vez de a la de la ISR, y extrapolar θ en los ciclos intermedios con la
 * velocidad estimada. No es optimización: es rediseño obligatorio antes de
 * meter el control en la ISR.
 * ─────────────────────────────────────────────────────────────────────────
 */
