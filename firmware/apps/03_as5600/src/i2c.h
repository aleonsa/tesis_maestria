#ifndef I2C_H
#define I2C_H

#include <stdint.h>

/*
 * i2c.h — I2C1 master en PB6 (SCL) / PB7 (SDA), polling mode.
 *
 * Pensado para el bring-up del encoder AS5600 en J8 de la B-G431B-ESC1.
 * Kernel clock = HSI16 (16 MHz), Standard mode 100 kHz.
 *
 * Cuando funcione, este archivo se promueve a firmware/lib/ (mismo camino
 * que clock.c y uart.c, que nacieron dentro de una app).
 */

/*
 * Configura I2C1 + GPIOs PB6/PB7 + selecciona HSI16 como kernel clock.
 * Llamar DESPUÉS de clock_init_170mhz_hse() (necesita HSI encendido aparte
 * del PLL; lo enciende este init si hace falta).
 */
void i2c1_init(void);

/*
 * Lee `n` bytes consecutivos del esclavo `addr7` (dirección 7-bit, p.ej.
 * 0x36 para el AS5600) empezando en el registro `reg`. Usa el patrón
 * write-pointer + repeated-START + read con auto-increment del esclavo.
 *
 * Bloquea (polling). Si el esclavo no hace ACK, el hardware nunca levanta
 * TXIS/RXNE y la ejecución se queda atrapada aquí — síntoma diagnóstico de
 * "no hay esclavo / sin pull-ups / dirección mal" (ver FIELD_NOTES).
 */
void i2c1_read_regs(uint8_t addr7, uint8_t reg, uint8_t *buf, uint8_t n);

#endif
