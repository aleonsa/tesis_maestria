/*
 * uart.c — USART2 polling-mode sobre PB3/PB4 (VCP del ST-LINK).
 *
 * Frame:      8N1 (8 bits, sin paridad, 1 stop bit) — el default tras reset.
 * Baud:       parametrizable (115200 típico para terminales).
 * Clock src:  PCLK1 = 170 MHz (default RCC_CCIPR.USART2SEL = 0b00).
 *
 * Cross-ref: RM0440 §38 (USART), §9 (GPIO alternate functions).
 *            DS12589 Table 13 (Alternate function mapping) — confirma AF7 = USART2.
 *
 * Por qué polling y no IRQ/DMA: para el primer hello-world basta. Cuando el
 * loop de control esté corriendo no podemos perder tiempo en TXE-spin; ahí
 * migramos a DMA circular. Una cosa a la vez.
 */

#include "stm32g431xx.h"
#include "uart.h"

void uart2_init(uint32_t baud) {
    /*
     * Paso 1 — Habilitar clock al GPIOB (donde están PB3 y PB4).
     * Bit 1 de RCC_AHB2ENR.
     */
    RCC->AHB2ENR |= RCC_AHB2ENR_GPIOBEN;

    /*
     * Paso 2 — Habilitar clock al USART2 en el bus APB1.
     * Bit 17 de RCC_APB1ENR1.
     * Sin esto, los registros del USART2 son inaccesibles (lecturas/escrituras silenciosas).
     */
    RCC->APB1ENR1 |= RCC_APB1ENR1_USART2EN;

    /*
     * Paso 3 — Configurar PB3 y PB4 en modo Alternate Function.
     *
     * MODER tiene 2 bits por pin:
     *   00 = input,  01 = output,  10 = AF,  11 = analog
     *
     * Limpiamos los bits actuales y escribimos 0b10 en cada par.
     * PB3 → MODER[7:6],  PB4 → MODER[9:8].
     */
    GPIOB->MODER &= ~(GPIO_MODER_MODE3 | GPIO_MODER_MODE4);
    GPIOB->MODER |=  (0b10U << GPIO_MODER_MODE3_Pos)
                 |   (0b10U << GPIO_MODER_MODE4_Pos);

    /*
     * Paso 4 — Seleccionar AF7 (= USART2) para PB3 y PB4.
     *
     * Cada pin tiene 4 bits en el registro AFR para elegir entre 16 AFs.
     * AFR[0] cubre pines 0–7  (= "AFRL" en RM0440)
     * AFR[1] cubre pines 8–15 (= "AFRH")
     *
     * PB3 → AFR[0] bits [15:12],  PB4 → AFR[0] bits [19:16].
     * El valor 7 (0b0111) selecciona AF7 en G4 — confirmado por DS12589 Table 13.
     */
    GPIOB->AFR[0] &= ~((0xFU << (3 * 4)) | (0xFU << (4 * 4)));
    GPIOB->AFR[0] |=  ((0x7U << (3 * 4)) | (0x7U << (4 * 4)));

    /*
     * Paso 5 — Configurar USART2 con UE = 0 todavía.
     *
     * Limpiamos CR1 completo para empezar de un estado conocido:
     *   M0 = M1 = 0  → 8 data bits
     *   PCE = 0      → sin paridad
     *   OVER8 = 0    → oversampling /16 (default)
     *   UE = 0       → USART aún apagado
     *
     * Limpiamos también CR2 y CR3 (1 stop bit y sin features extra).
     */
    USART2->CR1 = 0;
    USART2->CR2 = 0;
    USART2->CR3 = 0;

    /*
     * Paso 6 — Programar baud rate.
     *
     * Con OVER8 = 0:  USARTDIV = f_CK / baud, y BRR se escribe directo.
     * Con OVER8 = 1:  encoding distinto (raras veces se usa).
     *
     * f_CK = PCLK1 = 170 MHz.
     * Para 115200: USARTDIV = 170e6 / 115200 ≈ 1475.69 → redondeo a 1476.
     * Error resultante: -0.02% (muy debajo del ±2.5% que el receptor tolera).
     *
     * El cálculo lo hacemos en runtime para permitir cualquier baud.
     */
    USART2->BRR = (170000000U + baud / 2U) / baud;

    /*
     * Paso 7 — Habilitar Tx, Rx y el periférico (UE último).
     *
     * Orden importante: UE debe ser el ÚLTIMO bit que se enciende. Mientras
     * UE = 0, los demás bits son tranquilamente modificables; cuando UE = 1,
     * algunos bits pasan a read-only y cambiarlos requiere bajar UE de nuevo.
     */
    USART2->CR1 = USART_CR1_TE | USART_CR1_RE;
    USART2->CR1 |= USART_CR1_UE;
}

void uart2_putc(char c) {
    /*
     * Polling: esperar a que el TDR esté libre (TXE = 1) antes de escribir.
     * El hardware copia TDR → shift register internamente, y baja TXE mientras
     * está ocupado. Cuando TXE vuelve a 1, podemos colocar el siguiente byte.
     */
    while ((USART2->ISR & USART_ISR_TXE) == 0U) { }
    USART2->TDR = (uint8_t)c;
}

void uart2_puts(const char *s) {
    while (*s != '\0') {
        uart2_putc(*s++);
    }
}

/*
 * Retargeting de newlib: cuando printf() necesita "escribir" bytes, llama
 * internamente a _write(fd, buf, len). Por default, libnosys.a (que linkeamos
 * con --specs=nosys.specs) provee un stub vacío. Nuestra definición es no-weak
 * y el linker la prefiere.
 *
 * Signature canónica: int _write(int fd, char *buf, int len).
 * Ignoramos `fd` (mandamos todo a UART, sin distinguir stdout/stderr).
 */
int _write(int fd, char *buf, int len) {
    (void)fd;
    for (int i = 0; i < len; i++) {
        uart2_putc(buf[i]);
    }
    return len;
}
