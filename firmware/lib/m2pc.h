#ifndef M2PC_H
#define M2PC_H

#include <stdint.h>

/*
 * m2pc.h — FCS-M²PC: control predictivo de corriente con conjunto finito y
 * modulación de dos vectores + nulo (Coronado-Andrade et al. 2025, Alg. 3;
 * `fcs_m2pc_v2/control/fcs_m2pc.m`). Punto fijo, pensado para la ISR de 50 kHz.
 *
 * EL ALGORITMO EN TRES LÍNEAS
 *
 *   1. Se elige el sector: los dos vectores activos adyacentes a la tensión
 *      deadbeat u* = e + R·i* + (L/Ts)(i* − i), más el vector nulo. (El paper
 *      usa la dirección de la BEMF, θ_e + 90°: vale solo con par positivo y
 *      e dominante; ver m2pc.c, sesión 23.)
 *   2. Se divide Ts en ρ partes y se prueban todas las combinaciones
 *      (n·Ts/ρ con Va, m·Ts/ρ con Vb, el resto nulo), n + m ≤ ρ: son
 *      (ρ+1)(ρ+2)/2 candidatos (66 con ρ = 10). Para cada uno se predice
 *      i(k+1) con el modelo RL + BEMF y se evalúa G = ‖i_ref − i_pred‖².
 *   3. La combinación de menor costo se modula con SVM (7 segmentos,
 *      simétrica) y se aplica el periodo siguiente.
 *
 * MODELO DE PREDICCIÓN (Euler, marco αβ, amplitud-invariante)
 *
 *   i(k+1) = i(k) + (Ts/L) · ( ū(k) − e(k) − R·i(k) )
 *
 * RETARDO DE UN PERIODO (Cortés 2012)
 *
 *   Se muestrea en el pico del PWM; el comando que se calcula entra en los
 *   CCR en la UEV, que pwm_enable() coloca en el pico siguiente. Así el
 *   comando elegido en el paso k se aplica durante [k+1, k+2]. El paso
 *   predice primero i(k+1) con el comando ya en curso (conocido) y después
 *   i(k+2) con cada candidato; el costo se evalúa sobre i(k+2). Presupuesto:
 *   el periodo completo (20 µs). `m2pc_overruns()` cuenta las veces que el
 *   paso se pasó del siguiente pico (debe ser 0).
 *
 *   Se intentó antes un modelo de medio periodo (CCR latcheados en el valle,
 *   ū = ½u(k−1) + ½u(k)): menos retardo, pero exige paso < 10 µs y los 66
 *   candidatos cuestan 9.5 µs desde flash. Queda como opción futura con el
 *   paso en CCM-SRAM.
 *
 * CÁLCULO EFICIENTE
 *
 *   El término (Ts/L)·u de cada candidato es independiente de la corriente,
 *   así que el incremento por división de cada vector activo se precalcula
 *   por sector en init. La rejilla se recorre restando esos incrementos al
 *   residuo: por candidato quedan 2 restas, 2 productos y una comparación.
 *   Sin divisiones ni trigonometría dentro del paso.
 *
 * DIFERENCIAS CON LA SIMULACIÓN MATLAB (deliberadas, ver PROGRESO sesión 20)
 *
 *   - Sin la fase de "arranque" con búsqueda sobre 8 vectores: existía porque
 *     la simulación derivaba el sector de la BEMF estimada, indefinida a
 *     velocidad cero. Aquí θ_e viene del encoder y el sector siempre existe.
 *   - BEMF con la física del banco: e = λ·ω_e·[−sin θ_e; cos θ_e] (flujo sobre
 *     d, BEMF sobre +q). La "forma" sinusoidal de MATLAB lleva el signo opuesto
 *     por convención interna de su planta.
 *   - Retardo de un periodo compensado (arriba) en vez de aplicación instantánea.
 *
 * UNIDADES: ver motor_params.h. Aquí las corrientes entran en Q8 LSB (LSB
 * × 256): la medida se desplaza gratis y la referencia gana resolución (un
 * LSB son 30 mA, demasiado grueso como paso de referencia). Tensión en mV,
 * θ en Q32.
 */

/* Subdivisiones de Ts. 10 = tesis (66 candidatos). Debe dividir a PWM_ARR
 * para que los tiempos sean exactos en ticks (1700/10 = 170). */
#define M2PC_RHO   10U

typedef struct {
    int32_t  i_alpha,   i_beta;     /* corriente medida        [Q8 LSB] */
    int32_t  ref_alpha, ref_beta;   /* corriente de referencia [Q8 LSB] */
    int32_t  e_alpha,   e_beta;     /* BEMF del modelo         [mV]     */
    uint32_t theta_e_q32;           /* posición eléctrica (sector)      */
} m2pc_in_t;

/* Para el monitor: qué eligió el último paso. */
typedef struct {
    uint8_t  sector;                /* 0..5 → vectores V(s+1), V(s+2) */
    uint8_t  n, m;                  /* divisiones de Va y Vb           */
    int32_t  u_alpha, u_beta;       /* comando medio aplicado [mV]     */
    int32_t  cost_lsb2;             /* ‖residuo‖² del elegido [LSB²]   */
} m2pc_mon_t;

/* Precalcula los vectores del inversor y los incrementos por sector. */
void m2pc_init(void);

/* Un paso: predice, elige y ESCRIBE los duties en TIM1. Llamar una vez por
 * tick de la ISR de control, tras tener corriente, referencia, BEMF y θ_e. */
void m2pc_step(const m2pc_in_t *in);

const m2pc_mon_t *m2pc_mon(void);

/* Pasos que terminaron después del valle (los duties se aplicaron un
 * periodo tarde y el modelo de medio retardo no se cumplió). Debe ser 0. */
uint32_t m2pc_overruns(void);

#endif /* M2PC_H */
