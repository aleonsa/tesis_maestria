#ifndef AS5600_H
#define AS5600_H

#include <stdint.h>
#include "i2c.h"

/*
 * as5600.h — encoder magnético absoluto AS5600 (12-bit, I2C).
 *
 * Integrado de fábrica al motor 2804 del banco. Dirección I2C 7-bit = 0x36.
 * Resolución: 4096 cuentas por vuelta mecánica (0–4095).
 *
 * Leemos RAW ANGLE (0x0C), NO ANGLE (0x0E): el segundo pasa por un filtro
 * interno + escalado ZPOS/MPOS que añade latencia. Queremos el dato crudo y
 * filtramos/extrapolamos nosotros en el lazo de control.
 *
 * Asume i2c1_init() ya ejecutado.
 */

#define AS5600_ADDR7   0x36U

/* Bits de STATUS (0x0B): MD = magnet detected, ML = too weak, MH = too strong. */
#define AS5600_STATUS_MH   (1U << 3)   /* AGC mínimo: imán demasiado cerca  */
#define AS5600_STATUS_ML   (1U << 4)   /* AGC máximo: imán demasiado lejos  */
#define AS5600_STATUS_MD   (1U << 5)   /* imán detectado                    */

/*
 * Lee RAW ANGLE (0x0C/0x0D) → 0..4095 en *out.
 * Devuelve I2C_OK, o el error de bus que impidió la lectura. *out solo es
 * válido con I2C_OK.
 */
i2c_status_t as5600_raw_angle(uint16_t *out);

/*
 * Lee STATUS (0x0B) → combinación de bits MD/ML/MH de arriba en *out.
 * Devuelve I2C_OK o el error de bus.
 */
i2c_status_t as5600_status(uint8_t *out);

/*
 * Variante no bloqueante de as5600_raw_angle(), para usar dentro del lazo de
 * control. Requiere i2c1_nb_init() hecho.
 *
 *   as5600_raw_angle_start()  arranca la lectura y vuelve enseguida
 *   ... el lazo sigue trabajando mientras el bus avanza por IRQ ...
 *   as5600_raw_angle_take()   cuando i2c1_nb_state() == I2C_NB_DONE
 *
 * La razón de existir está en i2c.h: la lectura bloqueante son ~450 µs contra
 * los 20 µs de periodo de la ISR.
 */
i2c_status_t as5600_raw_angle_start(void);
i2c_status_t as5600_raw_angle_take(uint16_t *out);

#endif
