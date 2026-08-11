# 02_pwm_adc — Bring-up TIM1 PWM 3-fásico + ADC sincronizado

App de **Fase 1** del bring-up del banco. Objetivo: generar 6 PWMs complementarias a 50 kHz (3 high-side + 3 low-side con dead-time), sincronizadas con muestreo del ADC en el centro del PWM. **Sin motor conectado todavía.**

## Periféricos involucrados

| Periférico | Pines | Función |
|---|---|---|
| TIM1 | PA8/CH1, PA9/CH2, PA10/CH3, PC13/CH1N, PA12/CH2N, PB15/CH3N | PWM 50 kHz center-aligned con dead-time ~500 ns |
| TIM1 TRGO | (señal interna, no es un pin) | Dispara ADC al pico/valle del contador |
| ADC1, ADC2 | (en semana 5) | Dual regular simultaneous, JEXTSEL = TIM1_TRGO |
| OPAMP1/2/3 | (en semana 5) | PGA para corrientes de shunt |
| I²C1 | **PB8/SCL**, PB7/SDA | AS5600 (validado 2026-08-10; PB6 NO es SCL) |

## Plan por semana

- **Semana 4** — TIM1 50 kHz, 6 PWMs, dead-time. Verificar con osciloscopio: complementariedad + dead-time medido.
- **Semana 5** — OPAMPs + ADC dual simultaneous. Log de valores crudos por UART.
- **Semana 6** — ISR EOC + calibración de offsets + medición de tiempo de ISR.
- **Semana 7** — AS5600 vía I²C1 + extrapolación de posición.

## Conceptos (ver FIELD_NOTES.md)

- [N1.1 — Panorama de Fase 1](../../FIELD_NOTES.md#n11--panorama-de-fase-1-qué-construimos-y-por-qué)
- [N1.3 — Counter modes](../../FIELD_NOTES.md#n13--timers-y-counter-modes-edge-aligned-vs-center-aligned)
- [N1.4 — Del contador al pin](../../FIELD_NOTES.md#n14--del-contador-al-pin-cómo-el-silicio-genera-pwm)
- [N1.5 — Complementarios + dead-time](../../FIELD_NOTES.md#n15--pines-complementarios-y-dead-time-cómo-el-tim1-evita-el-shoot-through)
- [N1.6 — TRGO al ADC](../../FIELD_NOTES.md#n16--trgo-el-cordón-umbilical-entre-el-tim1-y-el-adc)
- [N1.7 — Break inputs (no en Fase 1)](../../FIELD_NOTES.md#n17--break-inputs-la-protección-de-hardware-contra-el-desastre)

## Estado actual

Skeleton creado: build pasa, `main()` arranca, configura clock/UART, llama stubs de `pwm_init()` (TODOs por ahora). Próximo paso: implementar `pwm_init()` con los registros de TIM1.
