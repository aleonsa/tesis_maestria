#ifndef I2C_H
#define I2C_H

#include <stdint.h>

/*
 * i2c.h — I2C1 master en PB8 (SCL) / PB7 (SDA), polling mode.
 *
 * Bring-up del encoder AS5600 en J8 de la B-G431B-ESC1.
 * Kernel clock = HSI16 (16 MHz), Standard mode 100 kHz.
 *
 * ⚠ SCL en PB8, NO en PB6: AF4 en PB6 no es I2C1_SCL en el STM32G431. Ver la
 *   cabecera de i2c.c y FIELD_NOTES N1.16 para la evidencia.
 *
 * Cuando el driver se estabilice se promueve a firmware/lib/ (mismo camino
 * que clock.c y uart.c, que nacieron dentro de una app). Todavía no: falta
 * resolver la migración a DMA/IRQ que exige el presupuesto de la ISR.
 *
 * DISEÑO: ninguna espera bloquea indefinidamente. Todas llevan timeout y
 * vigilan NACKF, así que el driver REPORTA en vez de colgarse. La primera
 * versión usaba `while(flag){}` desnudos y un esclavo mudo mataba el firmware
 * sin dejar rastro de en qué paso murió.
 */

/* Resultado de una operación de bus. I2C_OK == 0. */
typedef enum {
    I2C_OK = 0,
    I2C_ERR_BUSY,            /* el bus estaba ocupado antes de arrancar     */
    I2C_ERR_NACK_ADDR,       /* nadie hizo ACK a la dirección               */
    I2C_ERR_NACK_DATA,       /* ACK a la dirección, NACK a un byte de datos */
    I2C_ERR_TIMEOUT_TXIS,    /* TXIS  nunca subió                           */
    I2C_ERR_TIMEOUT_TC,      /* TC    nunca subió                           */
    I2C_ERR_TIMEOUT_RXNE,    /* RXNE  nunca subió                           */
    I2C_ERR_TIMEOUT_STOPF,   /* STOPF nunca subió                           */
} i2c_status_t;

/* Nombre legible del status, para printf. */
const char *i2c_status_str(i2c_status_t st);

/*
 * Configura I2C1 + GPIOs PB8/PB7 + selecciona HSI16 como kernel clock.
 * Llamar DESPUÉS de clock_init_170mhz_hse().
 */
void i2c1_init(void);

/*
 * Lee `n` bytes consecutivos del esclavo `addr7` (7-bit, p.ej. 0x36 para el
 * AS5600) empezando en el registro `reg`. Patrón write-pointer +
 * repeated-START + read, aprovechando el auto-increment del esclavo.
 *
 * Bloquea (polling) pero con timeout. Devuelve I2C_OK o el error del paso
 * que falló.
 */
i2c_status_t i2c1_read_regs(uint8_t addr7, uint8_t reg, uint8_t *buf, uint8_t n);

/*
 * "¿Hay alguien en esta dirección?" con una escritura de 0 bytes: START +
 * dirección + STOP, la transacción más corta posible.
 *
 *   1 = ACK, hay dispositivo
 *   0 = NACK limpio → el reloj corrió hasta el noveno pulso: BUS SANO, nadie
 *       en esa dirección
 *  -1 = timeout → la transacción ni arrancó: bus clavado o SCL mal mapeada
 *
 * La distinción entre 0 y -1 es la herramienta de diagnóstico más útil del
 * módulo; fue la que localizó el pin equivocado el 2026-08-10.
 */
int i2c1_probe(uint8_t addr7);

/*
 * Barre las direcciones 7-bit válidas (0x08..0x77) e imprime las que
 * contestan. Devuelve cuántas encontró, o -1 si el bus está muerto (todos los
 * probes dieron timeout).
 */
int i2c1_scan(void);

/*
 * Estado eléctrico de las líneas SIN mover el bus: niveles lógicos de PB8/PB7
 * leídos del IDR del GPIO (el Schmitt trigger sigue conectado aunque el pin
 * esté en alternate function) y el bit BUSY del periférico.
 *
 * Separa dos escenarios que se confunden a simple vista:
 *   - líneas en alto  → bus en reposo sano, el esclavo simplemente no habla
 *   - alguna en bajo  → línea clavada (corto, o esclavo reteniendo el bus)
 *
 * Los tres punteros aceptan NULL.
 */
void i2c1_lines(uint8_t *scl_high, uint8_t *sda_high, uint8_t *busy);

/*
 * Reset por software del periférico (PE=0 → PE=1). RM0440 §40.4.1: bajar PE
 * ejecuta un reset interno, la única forma de despegar BUSY cuando una
 * transacción quedó a medias.
 */
void i2c1_reset_peripheral(void);

/*
 * Vuelca clock gating, reset, fuente de kernel clock, CR1/TIMINGR/ISR y la
 * configuración de PB8/PB7 en el GPIO. Marca TIMINGR si no coincide con lo
 * escrito — un TIMINGR en 0 delata un periférico que no acepta escrituras.
 */
void i2c1_dump_regs(void);

#endif
