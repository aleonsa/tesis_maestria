/*
 * i2c.c — I2C1 master sobre PB8 (SCL) / PB7 (SDA). Polling + camino por IRQ.
 *
 * Cross-ref: RM0440 Rev 9 — §39 (I2C), §7.4.27 (RCC_CCIPR / I2C1SEL), §9 (GPIO).
 * Los números de capítulo cambian entre revisiones del RM0440; los de aquí son
 * los de la Rev 9, que es la que está en papers/. Antes decían §40/§11, que son
 * los de una revisión distinta y no llevan a ninguna parte en la nuestra.
 *
 * Decisiones de diseño:
 *   - Kernel clock = HSI16 (16 MHz), NO PCLK1 (170 MHz). Desacopla el I2C
 *     del PLL y permite usar el TIMINGR canónico de ST tabulado para 16 MHz.
 *   - Standard mode 100 kHz: robusto para bring-up; el AS5600 soporta hasta
 *     FM+ 1 MHz, subimos después si conviene.
 *   - Dos caminos: polling para bring-up y arranque, IRQ para el lazo de
 *     control. El polling se quedó porque es la herramienta de diagnóstico
 *     (i2c1_scan, i2c1_probe) y porque antes de que exista una ISR no hay nada
 *     que proteger. Ver la sección de lectura no bloqueante al final.
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
        case I2C_ERR_ARG:           return "ARG invalido (n fuera de rango)";
        case I2C_ERR_INCOMPLETE:    return "INCOMPLETA (STOP con bytes de menos)";
        case I2C_ERR_BUS:           return "ERROR DE BUS (BERR/ARLO/OVR)";
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
    /* RM0440 Rev 9 §39.4.6 pide exactamente esto: «PE must be kept low during at
     * least three APB clock cycles [...] 1. Write PE = 0  2. Check PE = 0
     * 3. Write PE = 1». El paso 2 es una RELECTURA, no un retardo: la lectura
     * del APB no retorna hasta que el bus completó el ciclo, así que garantiza
     * el tiempo mínimo sin depender de cuántos NOP deje el compilador.
     *
     * La versión anterior giraba un bucle de 100 iteraciones `volatile`.
     * Funcionaba, pero por accidente: medía tiempo del lado equivocado. */
    I2C1->CR1 &= ~I2C_CR1_PE;
    while ((I2C1->CR1 & I2C_CR1_PE) != 0U) { }
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
 * ═════════════════════════════════════════════════════════════════════════
 * LECTURA NO BLOQUEANTE POR IRQ
 *
 * El problema, en números. Una lectura de ángulo cuesta ~45 pulsos de SCL:
 * START + dirección + ACK (9), registro + ACK (9), repeated-START + dirección
 * + ACK (9), dos bytes + ACK (18). A 100 kHz cada pulso son 10 µs, así que la
 * transacción completa son ~450 µs. El periodo de la ISR de control a 50 kHz
 * son 20 µs: la lectura bloqueante tarda 22× MÁS que el ciclo de control
 * entero. Subir a Fast Mode Plus (1 MHz) la deja en ~45 µs, todavía más del
 * doble del presupuesto. No hay margen que rascar afinando; hay que sacar la
 * espera del camino.
 *
 * Esta es la primera de las tres piezas de la solución: la transferencia
 * avanza sola. Las otras dos —leer a la tasa real del sensor (~7 kHz de
 * refresco interno del AS5600) en vez de a la de la ISR, y extrapolar θ en los
 * ciclos intermedios con la ω estimada— viven en la capa de arriba.
 *
 * Coste para la CPU: 5 IRQ por lectura de 2 bytes, cada una mueve un byte o
 * reprograma CR2. Del orden de 1 µs cada una contra 450 µs de bus, menos del 1% de
 * ocupación. Por qué IRQ y no DMA: el DMA baja de 5 eventos a 2, pero paga
 * DMAMUX, dos canales y coherencia de buffer para ahorrar ~2 µs de cada 450.
 * Se mide primero; si el número molesta, se escala con datos.
 * ═════════════════════════════════════════════════════════════════════════
 */

/*
 * Fase dentro de la transferencia. Sirve para UNA cosa: decir de qué fue el
 * NACK. NACKF mientras se escribe el puntero de registro significa "nadie en
 * esa dirección"; después de haberlo escrito significa "contestó y luego
 * rechazó el dato". Esa distinción es la que hace útil al camino de polling y
 * no se pierde aquí.
 */
typedef enum {
    NB_PH_ADDR_W = 0,   /* START mandado, esperando TXIS              */
    NB_PH_REG,          /* puntero de registro escrito, esperando TC  */
    NB_PH_RX,           /* repeated-START hecho, recibiendo           */
} nb_phase_t;

/* Bits de la traza de eventos. */
#define NB_EV_TXIS    (1U << 0)
#define NB_EV_TC      (1U << 1)
#define NB_EV_RXNE    (1U << 2)
#define NB_EV_STOPF   (1U << 3)
#define NB_EV_NACKF   (1U << 4)
#define NB_EV_ERR     (1U << 5)

static const char *nb_phase_str(nb_phase_t ph) {
    switch (ph) {
        case NB_PH_ADDR_W: return "ADDR_W (esperando TXIS)";
        case NB_PH_REG:    return "REG (esperando TC)";
        case NB_PH_RX:     return "RX (recibiendo)";
        default:           return "??";
    }
}

/* Sin inicializador: los estáticos arrancan en cero, que es exactamente
 * IDLE / I2C_OK / NB_PH_ADDR_W / contadores a 0. */
static struct {
    volatile i2c_nb_state_t state;
    volatile i2c_status_t   status;
    volatile uint32_t       seq;
    volatile uint32_t       errors;
    volatile uint32_t       err_isr;
    volatile nb_phase_t     err_phase;
    volatile i2c_status_t   err_status;
    /* Traza de la última transferencia. No es adorno: sin ella, "STOP sin
     * datos" tiene varias causas indistinguibles desde fuera. */
    volatile uint16_t       ev;         /* bitmask NB_EV_*                   */
    volatile uint16_t       irqs;       /* entradas al handler de eventos    */
    volatile uint32_t       cr2_rx;     /* CR2 tal como quedó para la lectura*/
    volatile uint32_t       isr_tc;     /* ISR en el momento del TC          */
    volatile uint8_t        err_idx;
    nb_phase_t              phase;
    uint8_t                 addr7;
    uint8_t                 reg;
    uint8_t                 n;
    uint8_t                 idx;
    uint8_t                 buf[I2C_NB_MAXBYTES];
} nb;

void i2c1_nb_init(void) {
    /* Fuentes de evento que la máquina de estados necesita. Dejarlas armadas en
     * reposo no genera IRQ espurias: TXIS solo sube tras el ACK de dirección de
     * una escritura, y RXNE/TC/STOPF solo dentro de una transferencia. El
     * periférico en reposo tiene TXE=1, que NO es TXIS. */
    I2C1->CR1 |= I2C_CR1_TXIE | I2C_CR1_RXIE | I2C_CR1_TCIE
               | I2C_CR1_STOPIE | I2C_CR1_NACKIE | I2C_CR1_ERRIE;

    /* Prioridad 3 contra la 1 de la ISR de control: el control SIEMPRE preempta
     * (y 0 queda reservada para faults). Es seguro porque el maestro estira SCL
     * en los tres puntos donde esta máquina tiene que actuar —RM0440 Rev 9
     * §39.4.7 para TXE/RXNE, §39.4.9 para TC—: llegar tarde ALARGA la
     * transacción, no pierde ni corrompe un byte.
     *
     * Son dos vectores distintos, no uno: I2C1_EV (eventos, nº 31) e I2C1_ER
     * (errores de bus, nº 32). */
    NVIC_SetPriority(I2C1_EV_IRQn, 3U);
    NVIC_SetPriority(I2C1_ER_IRQn, 3U);
    NVIC_EnableIRQ(I2C1_EV_IRQn);
    NVIC_EnableIRQ(I2C1_ER_IRQn);
}

/*
 * Cierre único de la transferencia. Solo actúa desde BUSY, así que el STOPF
 * que llega DESPUÉS de un abort no vuelve a contar el error ni pisa el estado.
 */
static void nb_finish(void) {
    if (nb.state != I2C_NB_BUSY) return;

    /* Barrera antes de publicar el estado: el consumidor corre en una ISR de
     * prioridad más alta y en cuanto vea DONE va a leer nb.buf. */
    __DMB();

    if (nb.status == I2C_OK) {
        nb.seq++;
        nb.state = I2C_NB_DONE;
    } else {
        /* La foto del fallo se toma AQUÍ, no en cada sitio que pone un status:
         * un único punto de cierre es también un único punto de captura, y así
         * no hay forma de añadir un camino de error que se olvide de dejar
         * rastro. err_isr puede venir ya escrito por el handler de errores; el
         * de aquí es el estado en el momento del cierre, que es el que importa
         * para saber dónde nos quedamos. */
        nb.err_isr    = I2C1->ISR;
        nb.err_phase  = nb.phase;
        nb.err_idx    = nb.idx;
        nb.err_status = nb.status;
        nb.errors++;
        nb.state = I2C_NB_ERR;
    }
}

i2c_status_t i2c1_read_regs_start(uint8_t addr7, uint8_t reg, uint8_t n) {
    if ((n == 0U) || (n > I2C_NB_MAXBYTES)) return I2C_ERR_ARG;
    if (nb.state == I2C_NB_BUSY)            return I2C_ERR_BUSY;

    /* Una sola mirada al bus, sin esperar: si está ocupado, el llamador
     * reintenta en su próximo ciclo. Girar aquí sería reintroducir justo la
     * espera que vinimos a quitar. */
    if ((I2C1->ISR & I2C_ISR_BUSY) != 0U)   return I2C_ERR_BUSY;

    nb.addr7  = addr7;
    nb.reg    = reg;
    nb.n      = n;
    nb.idx    = 0U;
    nb.phase  = NB_PH_ADDR_W;
    nb.status = I2C_OK;
    nb.ev     = 0U;
    nb.irqs   = 0U;
    nb.cr2_rx = 0U;
    nb.isr_tc = 0U;
    /* Todo el contexto escrito ANTES de mover el periférico: llamada desde
     * código de fondo, la IRQ puede dispararse en la instrucción siguiente al
     * START y encontrarse la estructura a medio llenar. */
    nb.state  = I2C_NB_BUSY;

    I2C1->ICR = I2C_ICR_NACKCF | I2C_ICR_STOPCF
              | I2C_ICR_BERRCF | I2C_ICR_ARLOCF | I2C_ICR_OVRCF;

    /* Fase 1: puntero de registro, 1 byte, AUTOEND=0 para retener el bus de
     * cara al repeated-START. Misma secuencia que el camino de polling. */
    I2C1->CR2 = ((uint32_t)addr7 << (I2C_CR2_SADD_Pos + 1U))
              | (1U << I2C_CR2_NBYTES_Pos);
    I2C1->CR2 |= I2C_CR2_START;

    return I2C_OK;
}

void I2C1_EV_IRQHandler(void) {
    uint32_t isr = I2C1->ISR;
    nb.irqs++;

    /* NACK. El hardware manda el STOP por su cuenta incluso con AUTOEND=0
     * (§39.4.9: «When a NACK is received, the TXIS flag is not set and a STOP
     * condition is automatically sent»), así que aquí solo anotamos la causa y
     * dejamos que STOPF cierre. Un único punto de cierre, sin caminos gemelos. */
    if ((isr & I2C_ISR_NACKF) != 0U) {
        nb.ev |= NB_EV_NACKF;
        I2C1->ICR = I2C_ICR_NACKCF;
        nb.status = (nb.phase == NB_PH_REG) ? I2C_ERR_NACK_DATA
                                            : I2C_ERR_NACK_ADDR;
    }

    if ((isr & I2C_ISR_TXIS) != 0U) {
        nb.ev |= NB_EV_TXIS;
        I2C1->TXDR = nb.reg;          /* escribir TXDR es lo que baja TXIS */
        nb.phase   = NB_PH_REG;
    }

    if ((isr & I2C_ISR_TC) != 0U) {
        /* Fase 2: repeated-START + lectura. AUTOEND=1 → el STOP lo pone el
         * hardware tras el último byte. Setear START es lo que baja TC. */
        I2C1->CR2 = ((uint32_t)nb.addr7 << (I2C_CR2_SADD_Pos + 1U))
                  | ((uint32_t)nb.n << I2C_CR2_NBYTES_Pos)
                  | I2C_CR2_RD_WRN
                  | I2C_CR2_AUTOEND;
        I2C1->CR2 |= I2C_CR2_START;
        nb.phase   = NB_PH_RX;

        nb.ev    |= NB_EV_TC;
        nb.isr_tc = isr;
        nb.cr2_rx = I2C1->CR2;        /* lo que de verdad quedó programado */
    }

    /* RXDR se lee SIEMPRE que RXNE esté alto, aunque el buffer ya esté lleno:
     * leerlo es lo único que baja el flag, y un RXNE que no se atiende con
     * RXIE armado es una tormenta de interrupciones. El descarte se decide
     * después, sobre el dato ya sacado. */
    if ((isr & I2C_ISR_RXNE) != 0U) {
        nb.ev |= NB_EV_RXNE;
        uint8_t d = (uint8_t)I2C1->RXDR;
        if (nb.idx < nb.n) nb.buf[nb.idx++] = d;
    }

    /* STOPF al final a propósito: con AUTOEND el último byte y el STOP pueden
     * estar pendientes en la misma entrada al handler, y drenar RXDR antes
     * evita perderlo. */
    if ((isr & I2C_ISR_STOPF) != 0U) {
        nb.ev |= NB_EV_STOPF;
        I2C1->ICR = I2C_ICR_STOPCF;
        if ((nb.status == I2C_OK) && (nb.idx < nb.n)) {
            nb.status = I2C_ERR_INCOMPLETE;
        }
        nb_finish();
    }
}

void I2C1_ER_IRQHandler(void) {
    /* BERR (START/STOP fuera de sitio), ARLO (arbitraje perdido) y OVR. La
     * trama está perdida. */
    nb.status  = I2C_ERR_BUS;
    nb.ev     |= NB_EV_ERR;
    I2C1->ICR  = I2C_ICR_BERRCF | I2C_ICR_ARLOCF | I2C_ICR_OVRCF;

    /* A diferencia del NACK, aquí el hardware NO manda STOP solo: lo pedimos y
     * cerramos sin esperarlo. Si el bus está clavado el STOP no saldrá, el
     * STOPF tardío se encontrará el estado ya fuera de BUSY y nb_finish() lo
     * ignorará. */
    I2C1->CR2 |= I2C_CR2_STOP;
    nb_finish();
}

void i2c1_nb_abort(void) {
    if (nb.state != I2C_NB_BUSY) return;

    /* TIMEOUT_STOPF es literal: la transacción no llegó a su STOP dentro del
     * plazo que fijó el consumidor. */
    if (nb.status == I2C_OK) nb.status = I2C_ERR_TIMEOUT_STOPF;

    I2C1->CR2 |= I2C_CR2_STOP;
    nb_finish();
}

i2c_nb_state_t i2c1_nb_state(void) { return nb.state; }
uint32_t       i2c1_nb_seq(void)     { return nb.seq; }
uint32_t       i2c1_nb_errors(void)  { return nb.errors; }
uint32_t       i2c1_nb_err_isr(void) { return nb.err_isr; }
const char    *i2c1_nb_err_phase(void) { return nb_phase_str(nb.err_phase); }
i2c_status_t   i2c1_nb_err_status(void) { return nb.err_status; }

void i2c1_nb_trace(i2c_nb_trace_t *t) {
    t->ev     = nb.ev;
    t->irqs   = nb.irqs;
    t->cr2_rx = nb.cr2_rx;
    t->isr_tc = nb.isr_tc;
    t->idx    = nb.err_idx;
    t->n      = nb.n;
}

i2c_status_t i2c1_nb_take(uint8_t *buf, uint8_t n) {
    switch (nb.state) {
        case I2C_NB_BUSY:
            return I2C_ERR_BUSY;

        case I2C_NB_ERR: {
            i2c_status_t st = nb.status;
            nb.state = I2C_NB_IDLE;
            return st;
        }

        case I2C_NB_DONE:
            break;

        default:                       /* IDLE: no hay nada que recoger */
            return I2C_ERR_INCOMPLETE;
    }

    uint8_t k = (n < nb.n) ? n : nb.n;
    for (uint8_t i = 0U; i < k; i++) buf[i] = nb.buf[i];

    nb.state = I2C_NB_IDLE;
    return I2C_OK;
}

