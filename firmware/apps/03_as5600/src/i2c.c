/*
 * i2c.c — I2C1 master, polling mode, sobre PB6 (SCL) / PB7 (SDA).
 *
 * Cross-ref: RM0440 §40 (I2C), §7.4.27 (RCC_CCIPR / I2C1SEL),
 *            DS12589 Table 13 (AF mapping: PB6/PB7 = I2C1 en AF4).
 *
 * Decisiones de diseño (ver plan sesión 15):
 *   - Kernel clock = HSI16 (16 MHz), NO PCLK1 (170 MHz). Desacopla el I2C
 *     del PLL y permite usar el TIMINGR canónico de ST tabulado para 16 MHz.
 *   - Standard mode 100 kHz: robusto para bring-up; el AS5600 soporta hasta
 *     FM+ 1 MHz, subimos después si conviene.
 *   - Polling, sin IRQ/DMA: igual que el primer uart2. Una cosa a la vez.
 */

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
 *
 * Se valida con osciloscopio midiendo la frecuencia de SCL en PB6.
 */
#define I2C1_TIMINGR_100KHZ_16MHZ   0x30420F13U

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

    /* ---- Paso 3: GPIO PB6/PB7 en AF4, open-drain, pull-up ----
     *
     * MODER = 10 (AF). OTYPER = 1 (open-drain: OBLIGATORIO en I2C — la línea
     * solo se tira a 0; el '1' lo provee el pull-up). OSPEEDR alto para
     * flancos limpios. AFR = 4 (I2C1 en G4).
     *
     * PUPDR = 00 (sin pull interno) a propósito. El módulo AS5600 está
     * alimentado a 5 V y trae sus propios pull-ups (externos, a 5 V). Activar
     * el pull-up interno del STM32 (que va a VDD = 3.3 V) crearía, con la
     * línea en reposo a 5 V, un camino de corriente desde el bus hacia el
     * riel de 3.3 V a través del pin FT. Los pines son 5V-tolerant, pero la
     * práctica correcta con bus de 5 V es NO usar el pull interno y confiar
     * en los externos del módulo. Si el scope mostrara que SCL/SDA no suben,
     * el módulo no tendría pull-ups → poner 4.7k externos (no reactivar el
     * interno: seguiría siendo a 3.3 V sobre un bus de 5 V). */
    GPIOB->MODER &= ~(GPIO_MODER_MODE6 | GPIO_MODER_MODE7);
    GPIOB->MODER |=  (0b10U << GPIO_MODER_MODE6_Pos)
                 |   (0b10U << GPIO_MODER_MODE7_Pos);

    GPIOB->OTYPER |= GPIO_OTYPER_OT6 | GPIO_OTYPER_OT7;

    GPIOB->OSPEEDR |= (0b11U << GPIO_OSPEEDR_OSPEED6_Pos)
                   |  (0b11U << GPIO_OSPEEDR_OSPEED7_Pos);

    GPIOB->PUPDR &= ~(GPIO_PUPDR_PUPD6 | GPIO_PUPDR_PUPD7);

    GPIOB->AFR[0] &= ~((0xFU << (6 * 4)) | (0xFU << (7 * 4)));
    GPIOB->AFR[0] |=  ((0x4U << (6 * 4)) | (0x4U << (7 * 4)));

    /* ---- Paso 4: TIMINGR con PE = 0, luego PE = 1 ----
     * TIMINGR solo se puede escribir con el periférico deshabilitado.
     * Patrón "enable último", igual que UART/TIM1. */
    I2C1->CR1 &= ~I2C_CR1_PE;
    I2C1->TIMINGR = I2C1_TIMINGR_100KHZ_16MHZ;
    I2C1->CR1 |= I2C_CR1_PE;
}

void i2c1_read_regs(uint8_t addr7, uint8_t reg, uint8_t *buf, uint8_t n) {
    /* ---- Fase 1: escribir el puntero de registro (1 byte, sin STOP) ----
     *
     * CR2 en una sola escritura (estado conocido en todos los campos):
     *   SADD[7:1] = dirección 7-bit << 1   (bits 9:0, addr en 7:1)
     *   NBYTES    = 1                        (mandamos 1 byte: el registro)
     *   RD_WRN    = 0                        (write)
     *   AUTOEND   = 0                        (NO STOP: haremos repeated-START)
     * Luego START dispara la condición de arranque + envío de dirección. */
    I2C1->CR2 = ((uint32_t)addr7 << (I2C_CR2_SADD_Pos + 1U))
              | (1U << I2C_CR2_NBYTES_Pos);
    I2C1->CR2 |= I2C_CR2_START;

    /* TXIS = 1 cuando el TXDR está libre para el byte de datos (tras el ACK
     * de dirección). Si el esclavo no hace ACK, TXIS nunca sube → trap. */
    while ((I2C1->ISR & I2C_ISR_TXIS) == 0U) { }
    I2C1->TXDR = reg;

    /* TC = 1 (Transfer Complete) cuando NBYTES se envió y AUTOEND=0: el bus
     * queda en hold esperando repeated-START o STOP. */
    while ((I2C1->ISR & I2C_ISR_TC) == 0U) { }

    /* ---- Fase 2: repeated-START + lectura de n bytes (AUTOEND envía STOP) ----
     *
     * El AS5600 auto-incrementa el puntero, así que n bytes consecutivos
     * salen de registros consecutivos (0x0C, 0x0D, ...). */
    I2C1->CR2 = ((uint32_t)addr7 << (I2C_CR2_SADD_Pos + 1U))
              | ((uint32_t)n << I2C_CR2_NBYTES_Pos)
              | I2C_CR2_RD_WRN
              | I2C_CR2_AUTOEND;
    I2C1->CR2 |= I2C_CR2_START;

    for (uint8_t i = 0U; i < n; i++) {
        /* RXNE = 1 cuando hay un byte recibido en RXDR. */
        while ((I2C1->ISR & I2C_ISR_RXNE) == 0U) { }
        buf[i] = (uint8_t)I2C1->RXDR;
    }

    /* AUTOEND generó la condición STOP. Esperamos STOPF y lo limpiamos vía
     * ICR para dejar el periférico listo para la próxima transacción. */
    while ((I2C1->ISR & I2C_ISR_STOPF) == 0U) { }
    I2C1->ICR = I2C_ICR_STOPCF;
}
