# 01_blink

**Hito de Fase 0**: validar la cadena Mac → Pi → toolchain → OpenOCD → STM32G431 → LED.

Toggle del LED user (PC6) a **1 Hz exacto**, usando SysTick como base de tiempo (no NOP loop).

## Qué demuestra

- Build bare-metal CMSIS sin HAL.
- Vector table del startup ST funciona (en particular, override del `SysTick_Handler` weak).
- SysTick configurado: AHB clock, IRQ enabled, contador corriendo.
- Reloj del sistema: HSI 16 MHz por defecto (sin PLL aún).

## Verificación

1. LED rojo de la placa parpadea visualmente.
2. Con osciloscopio en PC6: cuadrado simétrico, período = 1.000 s ± precisión del HSI (~1%).

## Footprint esperado

~900 B Flash, ~1.6 KB RAM (mayormente stack/heap reservados por linker, no uso real).
