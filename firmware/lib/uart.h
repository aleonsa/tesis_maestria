#ifndef UART_H
#define UART_H

#include <stdint.h>

/*
 * Configura USART2 en PB3 (TX) y PB4 (RX) — pines conectados de fábrica al
 * VCP del ST-LINK V2.1 de la B-G431B-ESC1 (ver UM2516 Tabla 4 + esquemático
 * MB1419 redes USART2_*_ST_LINK).
 *
 * 8N1, polling mode, sin DMA, sin interrupciones.
 * Asume PCLK1 = 170 MHz (resultado de clock_init_170mhz_hse()).
 * Llamar DESPUÉS del setup del PLL.
 */
void uart2_init(uint32_t baud);

/* Envía un byte por TDR (bloquea hasta que TXE = 1). */
void uart2_putc(char c);

/* Envía una C-string terminada en '\0'. */
void uart2_puts(const char *s);

#endif
