/*
 * m2pc.c — implementación. Ver m2pc.h para el algoritmo y motor_params.h
 * para las unidades y constantes.
 *
 * Convención interna del paso: las corrientes se manejan en Q8 LSB (×256)
 * para que los incrementos por división (≈0.3 LSB) no se pierdan al
 * redondear. Los costos van en int64 (dos productos de Q8 pueden pasar de
 * 2^31): dos SMULL por candidato, que en Cortex-M4 cuestan lo mismo que un
 * MUL de 32 bits.
 */

#include "m2pc.h"
#include "motor_params.h"
#include "pwm.h"
#include "stm32g431xx.h"

/* ── Tablas por sector ──────────────────────────────────────────────────
 * Sector s (BEMF en [60s°, 60s°+60°)) usa Va = V(s+1), Vb = V(s+2), con
 * V1..V6 a 0°, 60°, …, 300°. Se guardan en mV y como incremento de corriente
 * por división de Ts en Q8 LSB, ya con el ½ del modelo de medio retardo. */
static int32_t s_v_a[6], s_v_b[6];      /* vectores V1..V6: α, β [mV] */
static int32_t s_da_a[6], s_da_b[6];    /* ½·(Ts/L)·Va/ρ por sector [Q8 LSB] */
static int32_t s_db_a[6], s_db_b[6];    /* ½·(Ts/L)·Vb/ρ por sector [Q8 LSB] */

/* SVM de 7 segmentos: por sector y fase, qué tiempos activos suma el
 * high-side. bit1 = Ta (Va), bit0 = Tb (Vb). A todos se suma T0/2.
 *   s0: V1(100) V2(110)   s1: V2(110) V3(010)   s2: V3(010) V4(011)
 *   s3: V4(011) V5(001)   s4: V5(001) V6(101)   s5: V6(101) V1(100)
 * Con el contador center-aligned esto produce V0 en el pico (donde se
 * muestrea), luego Va, Vb, V7 en el valle, Vb, Va y V0 otra vez. */
static const uint8_t s_svm[6][3] = {
    {3, 1, 0}, {2, 3, 0}, {0, 3, 1}, {0, 2, 3}, {1, 0, 3}, {3, 0, 2},
};

static const uint8_t s_next[6] = {1, 2, 3, 4, 5, 0};

/* Comando medio del paso anterior (mitad del intervalo actual). */
static int32_t s_uprev_a, s_uprev_b;

static m2pc_mon_t s_mon;
static uint32_t   s_overruns;

/* Δi [Q8 LSB] producido por v [mV] durante una división de Ts, con el ½ del
 * medio retardo: v · G · (1/(2ρ)). G está en Q20 → >>12 deja Q8. */
static int32_t div_increment(int32_t v_mv) {
    return ((v_mv * M2PC_G_Q20) >> 12) / (2 * (int32_t)M2PC_RHO);
}

void m2pc_init(void) {
    const int32_t v23 = (2 * VBUS_MV) / 3;
    const int32_t v13 = VBUS_MV / 3;
    const int32_t vs3 = (int32_t)(((int64_t)VBUS_MV * 37837) >> 16);   /* Vbus/√3 */

    /* V1..V6 en αβ (Tabla I del paper, amplitud-invariante). */
    s_v_a[0] =  v23; s_v_b[0] =    0;
    s_v_a[1] =  v13; s_v_b[1] =  vs3;
    s_v_a[2] = -v13; s_v_b[2] =  vs3;
    s_v_a[3] = -v23; s_v_b[3] =    0;
    s_v_a[4] = -v13; s_v_b[4] = -vs3;
    s_v_a[5] =  v13; s_v_b[5] = -vs3;

    for (uint32_t s = 0U; s < 6U; s++) {
        uint32_t t = s_next[s];
        s_da_a[s] = div_increment(s_v_a[s]);
        s_da_b[s] = div_increment(s_v_b[s]);
        s_db_a[s] = div_increment(s_v_a[t]);
        s_db_b[s] = div_increment(s_v_b[t]);
    }

    s_uprev_a = 0;
    s_uprev_b = 0;
    s_overruns = 0U;
}

void m2pc_step(const m2pc_in_t *in) {
    /* 1. Sector por la dirección de la BEMF: θ_e + 90°, en sextos de vuelta.
     *    (θ + 2^30)·6 >> 32 es floor((θ + 90°)/60°) sin división. */
    const uint32_t s = (uint32_t)(((uint64_t)(in->theta_e_q32 + 0x40000000U) * 6U) >> 32);

    /* 2. Predicción con vector nulo durante toda la mitad nueva, en Q8 LSB:
     *    a = i + G·(½·u_prev − e − R·i). Es el punto de partida del que los
     *    candidatos se alejan sumando n·da + m·db.
     *    R·i: (i_q8 >> 4) · RK_Q8 >> 12 = i · R en mV, sin desbordar int32. */
    const int32_t v0_a = (s_uprev_a >> 1) - in->e_alpha - (((in->i_alpha >> 4) * M2PC_RK_Q8) >> 12);
    const int32_t v0_b = (s_uprev_b >> 1) - in->e_beta  - (((in->i_beta  >> 4) * M2PC_RK_Q8) >> 12);
    const int32_t a_a  = in->i_alpha + ((v0_a * M2PC_G_Q20) >> 12);
    const int32_t a_b  = in->i_beta  + ((v0_b * M2PC_G_Q20) >> 12);

    /* 3. Rejilla sobre (n, m), n + m ≤ ρ. Residuo r = ref − pred; se recorre
     *    restando los incrementos precalculados del sector. */
    const int32_t da_a = s_da_a[s], da_b = s_da_b[s];
    const int32_t db_a = s_db_a[s], db_b = s_db_b[s];

    int32_t rn_a = in->ref_alpha - a_a;
    int32_t rn_b = in->ref_beta  - a_b;
    int64_t best = INT64_MAX;
    uint32_t bn = 0U, bm = 0U;

    for (uint32_t n = 0U; n <= M2PC_RHO; n++) {
        int32_t rm_a = rn_a, rm_b = rn_b;
        for (uint32_t m = 0U; m <= M2PC_RHO - n; m++) {
            const int64_t cost = (int64_t)rm_a * rm_a + (int64_t)rm_b * rm_b;
            if (cost < best) { best = cost; bn = n; bm = m; }
            rm_a -= db_a; rm_b -= db_b;
        }
        rn_a -= da_a; rn_b -= da_b;
    }

    /* 4. Modulación SVM: tiempos en ticks de TIM1 y reparto por fase. */
    const uint32_t ta  = bn * (PWM_ARR / M2PC_RHO);
    const uint32_t tb  = bm * (PWM_ARR / M2PC_RHO);
    const uint32_t t0h = (PWM_ARR - ta - tb) >> 1;
    const uint8_t *w = s_svm[s];
    pwm_set_duties((uint16_t)(t0h + ((w[0] >> 1) & 1U) * ta + (w[0] & 1U) * tb),
                   (uint16_t)(t0h + ((w[1] >> 1) & 1U) * ta + (w[1] & 1U) * tb),
                   (uint16_t)(t0h + ((w[2] >> 1) & 1U) * ta + (w[2] & 1U) * tb));

    /* 5. Comando medio aplicado: entra en el modelo del paso siguiente. */
    const uint32_t t = s_next[s];
    s_uprev_a = ((int32_t)bn * s_v_a[s] + (int32_t)bm * s_v_a[t]) / (int32_t)M2PC_RHO;
    s_uprev_b = ((int32_t)bn * s_v_b[s] + (int32_t)bm * s_v_b[t]) / (int32_t)M2PC_RHO;

    /* 6. ¿Llegamos antes del valle? DIR=1 es contando hacia abajo. */
    if ((TIM1->CR1 & TIM_CR1_DIR) == 0U) s_overruns++;

    s_mon.sector  = (uint8_t)s;
    s_mon.n       = (uint8_t)bn;
    s_mon.m       = (uint8_t)bm;
    s_mon.u_alpha = s_uprev_a;
    s_mon.u_beta  = s_uprev_b;
    s_mon.cost_lsb2 = (int32_t)(best >> 16);          /* Q8² → LSB² */
}

const m2pc_mon_t *m2pc_mon(void) {
    return &s_mon;
}

uint32_t m2pc_overruns(void) {
    return s_overruns;
}
