/*
 * swi2c.c — I2C por software sobre PB8 (SCL) / PB7 (SDA).
 *
 * Cross-ref: RM0440 §11 (GPIO), especificación I2C-bus (NXP UM10204) para la
 *            forma de START/STOP/ACK.
 *
 * Idea central del open-drain: nunca se escribe un '1' en la línea. "Soltar"
 * es dejar de hundir y esperar a que el pull-up la levante. Por eso cada
 * release va seguido de una lectura de confirmación — y esa lectura es, de
 * paso, el detector de clock stretching.
 */

#include "stm32g431xx.h"
#include "swi2c.h"

#define SCL_PIN   8U
#define SDA_PIN   7U

/*
 * Medio periodo de SCL. Cada iteración del bucle volatile son ~6-8 ciclos a
 * 170 MHz, así que 500 iteraciones ≈ 20-25 µs → SCL de ~20 kHz.
 *
 * Deliberadamente MÁS LENTO que los 100 kHz del driver de hardware: con la red
 * de 1.8 kΩ + 10 kΩ de J8 los flancos de subida son perezosos, y a esta
 * velocidad el bus tiene tiempo de sobra de asentarse. Si funciona lento y
 * falla rápido, el problema es de tiempos y lo sabremos.
 */
#define HALF_PERIOD_LOOPS   500U

/* Guardia para las esperas de nivel alto (clock stretching o línea muerta). */
#define STRETCH_GUARD       200000U

static void half_period(void) {
    for (volatile uint32_t i = 0U; i < HALF_PERIOD_LOOPS; i++) { }
}

/* BSRR: escribir en los bits bajos SETEA, en los altos (pin+16) LIMPIA.
 * Es atómico, no hace read-modify-write sobre ODR. */
static inline void release(uint32_t pin)   { GPIOB->BSRR = (1U << pin); }
static inline void drive_low(uint32_t pin) { GPIOB->BSRR = (1U << (pin + 16U)); }
static inline uint32_t read_pin(uint32_t pin) {
    return (GPIOB->IDR >> pin) & 1U;
}

void swi2c_init(void) {
    RCC->AHB2ENR |= RCC_AHB2ENR_GPIOBEN;

    /* Soltar ANTES de configurar como salida, para no generar un pulso bajo
     * espurio en el bus al cambiar de modo. */
    release(SCL_PIN);
    release(SDA_PIN);

    /* MODER = 01 (output), OTYPER = 1 (open-drain), sin pull interno
     * (los pull-ups los aportan la placa y el módulo, ver i2c.c). */
    GPIOB->MODER &= ~((0x3U << (SCL_PIN * 2U)) | (0x3U << (SDA_PIN * 2U)));
    GPIOB->MODER |=  ((0x1U << (SCL_PIN * 2U)) | (0x1U << (SDA_PIN * 2U)));

    GPIOB->OTYPER |= (1U << SCL_PIN) | (1U << SDA_PIN);

    GPIOB->PUPDR &= ~((0x3U << (SCL_PIN * 2U)) | (0x3U << (SDA_PIN * 2U)));

    half_period();
}

void swi2c_pin_test(uint8_t *low_ok, uint8_t *high_ok) {
    drive_low(SCL_PIN);
    half_period();
    if (low_ok != 0) *low_ok = (read_pin(SCL_PIN) == 0U) ? 1U : 0U;

    release(SCL_PIN);
    half_period();
    if (high_ok != 0) *high_ok = (read_pin(SCL_PIN) == 1U) ? 1U : 0U;
}

/* Suelta SCL y espera a que suba de verdad. 0 = alguien la retiene. */
static int scl_release_wait(void) {
    release(SCL_PIN);
    for (uint32_t g = 0U; g < STRETCH_GUARD; g++) {
        if (read_pin(SCL_PIN) != 0U) {
            half_period();
            return 1;
        }
    }
    return 0;
}

/* Un pulso de reloj completo: alto (con espera), bajo. */
static int clock_pulse(void) {
    if (!scl_release_wait()) return 0;
    drive_low(SCL_PIN);
    half_period();
    return 1;
}

static int send_start(void) {
    release(SDA_PIN);
    if (!scl_release_wait()) return 0;
    drive_low(SDA_PIN);       /* SDA cae con SCL alto = condición de START */
    half_period();
    drive_low(SCL_PIN);
    half_period();
    return 1;
}

static int send_stop(void) {
    drive_low(SDA_PIN);
    if (!scl_release_wait()) return 0;
    release(SDA_PIN);         /* SDA sube con SCL alto = condición de STOP */
    half_period();
    return 1;
}

/* Escribe un byte, devuelve 1 si el esclavo hizo ACK, 0 si NACK, -1 si el bus
 * se quedó retenido. */
static int write_byte(uint8_t b) {
    for (uint8_t i = 0U; i < 8U; i++) {
        if ((b & 0x80U) != 0U) release(SDA_PIN);
        else                   drive_low(SDA_PIN);
        b = (uint8_t)(b << 1);

        half_period();
        if (!clock_pulse()) return -1;
    }

    /* Noveno pulso: el esclavo hunde SDA para dar ACK. Hay que soltar SDA
     * para dejarlo hablar. */
    release(SDA_PIN);
    half_period();
    if (!scl_release_wait()) return -1;
    uint32_t ack = (read_pin(SDA_PIN) == 0U) ? 1U : 0U;
    drive_low(SCL_PIN);
    half_period();

    return (int)ack;
}

/* Lee un byte. `ack_after` = 1 para pedir más bytes, 0 para el último. */
static int read_byte(uint8_t *out, uint8_t ack_after) {
    uint8_t v = 0U;

    release(SDA_PIN);   /* el esclavo maneja SDA durante la lectura */
    for (uint8_t i = 0U; i < 8U; i++) {
        if (!scl_release_wait()) return -1;
        v = (uint8_t)((v << 1) | (uint8_t)read_pin(SDA_PIN));
        drive_low(SCL_PIN);
        half_period();
    }

    /* Nuestro ACK/NACK hacia el esclavo. */
    if (ack_after != 0U) drive_low(SDA_PIN);
    else                 release(SDA_PIN);
    half_period();
    if (!clock_pulse()) return -1;
    release(SDA_PIN);

    *out = v;
    return 0;
}

int swi2c_probe(uint8_t addr7) {
    if (!send_start()) return -1;

    int ack = write_byte((uint8_t)(addr7 << 1));   /* bit0 = 0 → escritura */
    (void)send_stop();

    if (ack < 0) return -1;
    return ack;
}

int swi2c_read_regs(uint8_t addr7, uint8_t reg, uint8_t *buf, uint8_t n) {
    int ack;

    /* --- Fase 1: escribir el puntero de registro, sin STOP --- */
    if (!send_start()) return -1;

    ack = write_byte((uint8_t)(addr7 << 1));
    if (ack < 0)  { (void)send_stop(); return -1; }
    if (ack == 0) { (void)send_stop(); return -2; }   /* NACK de dirección */

    ack = write_byte(reg);
    if (ack < 0)  { (void)send_stop(); return -1; }
    if (ack == 0) { (void)send_stop(); return -3; }   /* NACK del registro */

    /* --- Fase 2: repeated-START y lectura --- */
    if (!send_start()) return -1;

    ack = write_byte((uint8_t)((addr7 << 1) | 1U));   /* bit0 = 1 → lectura */
    if (ack < 0)  { (void)send_stop(); return -1; }
    if (ack == 0) { (void)send_stop(); return -4; }   /* NACK en el re-START */

    for (uint8_t i = 0U; i < n; i++) {
        /* ACK en todos menos el último: así el esclavo sabe cuándo parar. */
        if (read_byte(&buf[i], (uint8_t)((i + 1U) < n)) != 0) {
            (void)send_stop();
            return -1;
        }
    }

    (void)send_stop();
    return 0;
}
