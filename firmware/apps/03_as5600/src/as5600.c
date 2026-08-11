/*
 * as5600.c — lecturas del encoder AS5600 vía I2C1.
 *
 * Datasheet AS5600 (ams):
 *   - Dirección esclavo 7-bit: 0x36.
 *   - RAW ANGLE: 0x0C (high, bits 11:8) / 0x0D (low, bits 7:0). 12 bits útiles.
 *   - STATUS:    0x0B (bits MD/ML/MH de detección del imán).
 *   - Auto-increment del puntero en lectura múltiple → 2 bytes desde 0x0C
 *     devuelven 0x0C luego 0x0D.
 */

#include "as5600.h"
#include "i2c.h"

#define AS5600_REG_STATUS      0x0BU
#define AS5600_REG_RAWANGLE_H  0x0CU

i2c_status_t as5600_raw_angle(uint16_t *out) {
    uint8_t b[2];
    i2c_status_t st = i2c1_read_regs(AS5600_ADDR7, AS5600_REG_RAWANGLE_H, b, 2U);
    if (st != I2C_OK) return st;

    /* b[0] = registro 0x0C (solo bits 3:0 válidos = ángulo[11:8]).
     * b[1] = registro 0x0D (ángulo[7:0]). */
    *out = (uint16_t)(((uint16_t)(b[0] & 0x0FU) << 8) | b[1]);
    return I2C_OK;
}

i2c_status_t as5600_status(uint8_t *out) {
    uint8_t s;
    i2c_status_t st = i2c1_read_regs(AS5600_ADDR7, AS5600_REG_STATUS, &s, 1U);
    if (st != I2C_OK) return st;

    *out = s & (AS5600_STATUS_MD | AS5600_STATUS_ML | AS5600_STATUS_MH);
    return I2C_OK;
}
