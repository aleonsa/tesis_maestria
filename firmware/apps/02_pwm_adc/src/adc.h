/*
 * adc.h — API para OPAMPs + ADC1/ADC2 dual injected simultaneous.
 *
 * Cadena (FIELD_NOTES N1.10-N1.12):
 *   shunt → OPAMPx → ADCx canal interno → JDRx → ISR (Semana 6)
 *
 * Configuración fija:
 *   - OPAMP1/2/3 en standalone mode con feedback externo (R's del PCB)
 *   - Auto-calibración de offset al arranque (~120 ms total)
 *   - ADC1, ADC2 en dual injected simultaneous
 *   - Trigger inyectado: TIM1_TRGO en rising edge
 *   - Sample time: 6.5 ciclos del ADC clock
 *   - 12 bits de resolución
 *
 * Canales:
 *   - Inyectados:
 *       ADC1.JDR1 = i_a (OPAMP1 → ADC1 channel 13)
 *       ADC2.JDR1 = i_b (OPAMP2 → ADC2 channel 16)
 *       ADC2.JDR2 = i_c (OPAMP3 → ADC2 channel 18)
 *   - Regulares (en ADC1):
 *       SQR1 pos 1 = Vbus    (PA0  → ADC1 channel 1)
 *       SQR1 pos 2 = Temp    (PB14 → ADC1 channel 5)
 */

#ifndef ADC_H
#define ADC_H

#include <stdint.h>

/* Canales ADC (verificar al implementar/medir):
 *   OPAMP1 → ADC1 channel 13 (cuando OPAINTOEN=1)
 *   OPAMP2 → ADC2 channel 16
 *   OPAMP3 → ADC2 channel 18
 */
#define ADC_CHAN_OPAMP1   13U
#define ADC_CHAN_OPAMP2   16U
#define ADC_CHAN_OPAMP3   18U
#define ADC_CHAN_VBUS     1U   /* PA0 = ADC1_IN1 (Vbus_sense) */
#define ADC_CHAN_TEMP     5U   /* PB14 = ADC1_IN5 (Temp feedback) */

/*
 * opamp_init — configura los 3 OPAMPs en standalone mode + auto-calibración.
 *
 * Standalone mode: VINP y VINM son ambos pines externos. La ganancia depende
 * de las R's del PCB (MB1419). VOUT se routea internamente al ADC vía
 * OPAINTOEN=1.
 *
 * Toma ~120 ms (auto-calibración de offset secuencial para los 3 OPAMPs).
 */
void opamp_init(void);

/*
 * adc_init — configura ADC1 + ADC2 en dual injected simultaneous.
 *
 * Después de esta llamada, los ADCs están armados pero NO arrancando
 * conversiones — esperan trigger de TIM1_TRGO. La primera conversión
 * ocurrirá cuando pwm_enable() arranque el TIM1 y emita su primer TRGO.
 *
 * Pre-condiciones:
 *   - clock_init_170mhz_hse() ya ejecutado.
 *   - opamp_init() ya ejecutado (los OPAMPs deben estar activos antes
 *     que el ADC los muestree).
 */
void adc_init(void);

/* Lecturas crudas (raw 12-bit values, 0-4095). Para diagnóstico desde main(). */
uint16_t adc_get_vbus_raw(void);
uint16_t adc_get_temp_raw(void);
void adc_get_currents_raw(uint16_t *ia, uint16_t *ib, uint16_t *ic);

/* Dump de registros para validación sin GDB. */
void adc_dump_regs(void);

/*
 * ISR JEOS — ver FIELD_NOTES N1.14.
 *
 * Globals actualizadas a 50 kHz por ADC1_2_IRQHandler. Lectura desde main
 * es atómica para uint16 / uint32 en Cortex-M4. Volatile obliga al
 * compilador a leer/escribir de memoria en cada acceso (sin esto, el
 * compilador puede cachear valores en registros y nunca ver updates).
 */
extern volatile uint16_t g_ia_raw;
extern volatile uint16_t g_ib_raw;
extern volatile uint16_t g_ic_raw;
extern volatile uint32_t g_isr_count;

/*
 * adc_isr_init — habilita JEOSIE en ADC2 + GPIO PB6 (scope) + NVIC ADC1_2.
 *
 * Pre-condición: adc_init() ya ejecutado. Esto solo arma la interrupción,
 * sin reconfigurar el ADC. La ISR empezará a correr cuando pwm_enable()
 * dispare el primer TIM1_TRGO.
 */
void adc_isr_init(void);

/*
 * Offsets DC de las 3 fases (raw 12-bit). Se llenan tras adc_calibrate_offsets().
 * Lectura compensada: int16_t i_signed = (int16_t)g_ia_raw - (int16_t)g_offset_a;
 *
 * No son volatile porque solo se escriben una vez (en main, tras la calibración),
 * y se leen sin race condition después. La ISR no las modifica.
 */
extern uint16_t g_offset_a;
extern uint16_t g_offset_b;
extern uint16_t g_offset_c;

/*
 * adc_calibrate_offsets — promedia las N muestras siguientes de la ISR y
 * guarda el resultado en g_offset_a/b/c.
 *
 * Bloqueante. Tarda aproximadamente N * 20 μs (N=4096 → ~82 ms).
 * Pre-condición: pwm_enable() ya ejecutado (la ISR debe estar corriendo) y
 * MOTOR DESCONECTADO (sino lo que mediremos NO es offset, es i+offset).
 *
 * Aritmética entera: N debe ser potencia de 2 entre 256 y 32768 para que la
 * suma de muestras (12 bits c/u) quepa en uint32_t. División por shift right.
 *
 * Implementación: state machine dentro del handler. La ISR acumula durante
 * exactamente N ciclos, calcula el promedio, y libera el flag. Sin race
 * porque la atomicidad está garantizada por la ISR misma.
 */
void adc_calibrate_offsets(uint16_t n_samples_pow2);

/* --------------------------------------------------------------------------
 * Estadísticas de corriente: pico instantáneo y RMS sobre ventana fija.
 *
 * El handler acumula max(|i_cal|) y Σ(i_cal²) en uint64_t. Cada
 * ADC_STATS_WINDOW muestras (50000 = 1 s a 50 kHz), hace snapshot a los
 * globals "_result" y limpia los acumuladores. El flag g_stats_ready indica
 * "datos nuevos para consumir" al main.
 *
 * Main: __disable_irq() → leer result → poner ready=0 → __enable_irq().
 * Calcula RMS con isqrt32 (entero, sin float).
 *
 * Coste en el handler: ~25 ciclos extra por muestra. Snapshot (1 vez/s)
 * son ~30 ciclos más, despreciable.
 * -------------------------------------------------------------------------- */
#define ADC_STATS_WINDOW       50000U   /* 1 segundo a 50 kHz */

extern volatile uint16_t g_stats_max_a_raw;    /* pico |raw_cal| en último frame */
extern volatile uint16_t g_stats_max_b_raw;
extern volatile uint16_t g_stats_max_c_raw;
extern volatile uint64_t g_stats_sumsq_a;      /* Σ(raw_cal²) en último frame */
extern volatile uint64_t g_stats_sumsq_b;
extern volatile uint64_t g_stats_sumsq_c;
extern volatile uint8_t  g_stats_ready;        /* 1 = frame disponible, main pone 0 al consumir */

/* sqrt entera de 32-bit. ~20 ciclos en M4. Para calcular RMS = sqrt(Σi² / N). */
uint32_t adc_isqrt32(uint32_t x);

/* --------------------------------------------------------------------------
 * Conversión raw → miliamperios (fixed-point, sin floats).
 *
 *   I [mA] = ((int32_t)raw_cal × ADC_FACTOR_RAW_TO_MA_Q12) >> 12
 *
 * Coste: 1 multiplicación 32-bit + 1 shift ≈ 3 ciclos M4.
 *
 * --- K TEÓRICO ---
 *   El cálculo original (0.003 × 16 × 4096 / 3.3 = 59.6 raw/A) ignoraba la
 *   red de la placa: 1.5 kΩ serie + 22 kΩ a 3.3 V + 2.2 kΩ a GND en el nodo
 *   VINP (MB1419 hoja 5) atenúa el shunt ×4/7 además de polarizarlo a
 *   128.6 mV. Corregido (2026-10-13, FIELD_NOTES N1.23):
 *     K = 0.003 × (4/7) × 16 × 4096 / 3.3 = 34.0 raw/A  (~29 mA/raw)
 *   Es la ganancia total de 9.14 V/V que usan ST MCSDK y SimpleFOC. La red
 *   interna del PGA es 150k/10k ±15 % (DS12589 Tabla 73), así que ±15 % de
 *   incertidumbre es esperable sobre ese 34.
 *
 * --- K EMPÍRICO 2026-06-06 (con PGA_GAIN=00011, retirado) ---
 *   Sweep amp={170,510} con DMM true-RMS en serie: (8 raw, 200 mA) y
 *   (24 raw, 580 mA) → K ≈ 41.3 raw/A (Q12 = 99123). Ese 21 % sobre el
 *   teórico NO era tolerancia: con 00011 la R1 del PGA va a VSSA y la
 *   lectura incluía el rebote de tierra (caída IR del plano, proporcional a
 *   la corriente). Al pasar a 01011 (sesión 18) las lecturas bajaron 28→23
 *   raw a amp=510: 41.3 × 23/28 = 33.9 ≈ el teórico. La fase C, que daba
 *   picos 17-30 % altos con 00011, quedó balanceada con 01011.
 *
 * --- K EN USO (2026-10-13, PGA_GAIN=01011 Kelvin) ---
 *   K = 34.0 raw/A (teórico del esquemático, coherente con el empírico
 *   corregido). 1/K = 29.4 mA/raw → Q12 = round(29.4 × 4096) = 120422.
 *   ⚠ Pendiente revalidar con DMM en modo Kelvin (CAL_TARGET_PHASE en
 *   main.c). Calibración global para las 3 fases; con Kelvin el RMS está
 *   balanceado a ±1 raw, no hace falta factor por fase.
 * -------------------------------------------------------------------------- */
#define ADC_FACTOR_RAW_TO_MA_Q12   120422

static inline int32_t adc_raw_to_ma(int16_t raw_cal) {
    return ((int32_t)raw_cal * ADC_FACTOR_RAW_TO_MA_Q12) >> 12;
}

#endif /* ADC_H */
