#ifndef CLOCK_H
#define CLOCK_H

/*
 * Configura el reloj del sistema:
 *   HSE 8 MHz → PLL (×85, /2, /2) → SYSCLK = 170 MHz
 *   HCLK = PCLK1 = PCLK2 = 170 MHz (todos los prescalers a /1)
 *
 * Bloquea indefinidamente si HSE no engancha (cristal o R27 dañados).
 * Esa es la conducta deseada: ningún código posterior debe correr con clock incorrecto.
 */
void clock_init_170mhz_hse(void);

#endif
