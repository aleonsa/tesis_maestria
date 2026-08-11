#ifndef SWI2C_H
#define SWI2C_H

#include <stdint.h>

/*
 * swi2c.h — I2C por software (bit-banging) sobre PB8 (SCL) / PB7 (SDA).
 *
 * Existe por una razón de diagnóstico, no de producción: el periférico I2C1
 * quedó sujetando SCL en bajo y hay que separar dos preguntas que el hardware
 * mezcla en un solo síntoma:
 *
 *   1. ¿El PIN y la LÍNEA funcionan?  (¿puedo hundir y soltar SCL a mano?)
 *   2. ¿El AS5600 está vivo?          (¿contesta a una transacción legítima?)
 *
 * Manejando los pines como GPIO normales, el periférico queda fuera del
 * circuito y las respuestas dejan de estar acopladas.
 *
 * Es LENTO (decenas de kHz) y bloqueante. No sirve para el lazo de control.
 * Si el bus resulta sano por aquí, el trabajo siguiente es arreglar la
 * configuración del periférico, no adoptar esto.
 */

/* Configura PB6/PB7 como salidas open-drain liberadas. Deja el I2C1 fuera. */
void swi2c_init(void);

/*
 * Prueba de vida del pin: hunde SCL, lo lee, lo suelta, lo vuelve a leer.
 * Escribe en *low_ok / *high_ok si cada nivel se alcanzó.
 * Es la comprobación de que el driver de salida y el pull-up hacen su trabajo.
 */
void swi2c_pin_test(uint8_t *low_ok, uint8_t *high_ok);

/*
 * START + dirección + STOP, todo a mano.
 *   1 = ACK (hay dispositivo)
 *   0 = NACK (bus vivo, nadie ahí)
 *  -1 = SCL retenida por alguien más (clock stretching infinito)
 */
int swi2c_probe(uint8_t addr7);

/*
 * Lee `n` bytes desde `reg` del esclavo `addr7`, con el mismo patrón
 * write-pointer + repeated-START + read del driver de hardware.
 * Devuelve 0 en éxito, negativo en error.
 */
int swi2c_read_regs(uint8_t addr7, uint8_t reg, uint8_t *buf, uint8_t n);

#endif
