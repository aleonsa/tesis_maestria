#include "adaline.h"
#include "motor_params.h"

#include <math.h>
#include <stdatomic.h>
#include <string.h>

#define H ADALINE_H_HARM
#define W_ONE (1L << ADALINE_W_FRAC)

static int32_t s_w[2 * H][2];
static int32_t s_published[2][2 * H][2];
/* Single core: foreground cannot resume while the ISR reads its buffer.
 * Release/acquire prevents publication before memcpy completes. */
static _Atomic unsigned s_active;
_Static_assert(ATOMIC_INT_LOCK_FREE == 2, "ISR publication must be lock-free");

static int32_t clamp(int64_t v, int32_t lo, int32_t hi) {
    return v < lo ? lo : (v > hi ? hi : (int32_t)v);
}

static void publish(void) {
    const unsigned next = atomic_load_explicit(&s_active, memory_order_relaxed) ^ 1U;
    memcpy(s_published[next], s_w, sizeof s_w);
    atomic_store_explicit(&s_active, next, memory_order_release);
}

void adaline_init(uint8_t desde_cero) {
    memset(s_w, 0, sizeof s_w);
    if (!desde_cero) {
        /* e_α = −A sin(θ+φ), e_β = +A cos(θ+φ) → a1α = −A sinφ, b1α = −A cosφ,
         * a1β = A cosφ, b1β = −A sinφ. Con A=1, φ=0 es la sinusoide nominal. */
        const float phi = ADALINE_INIT_PHI_DEG * 3.14159265f / 180.0f;
        const int32_t ac = (int32_t)lroundf(ADALINE_INIT_AMP * cosf(phi) * (float)W_ONE);
        const int32_t as = (int32_t)lroundf(ADALINE_INIT_AMP * sinf(phi) * (float)W_ONE);
        s_w[0][0] = -as;
        s_w[1][0] = -ac;
        s_w[0][1] = ac;
        s_w[1][1] = -as;
    }
    publish();
}

static int32_t g_axis(const int32_t w[2 * H][2], const int32_t *c,
                      const int32_t *s, unsigned eje) {
    int64_t acc = 0;
    for (unsigned k = 0; k < H; k++)
        acc += (int64_t)w[2*k][eje]*c[k] + (int64_t)w[2*k+1][eje]*s[k];
    return (int32_t)(acc >> 31); /* Q27; ||W||_1 <= 4 bounds int64 */
}

/* El eval corre en la ISR de control con presupuesto ajustado (~2 µs).
 * Debe ser UNA función plana, sin llamadas ni arrays en stack: con -O2 gcc
 * mantiene c_k/s_k y los acumuladores en registros y desenrolla el lazo
 * (recurrencia compleja truncante + 4 productos por armónico, ambos ejes
 * entrelazados). La versión con basis()/g_axis() medía 766–1109 ciclos
 * (s26); esta mide ~150. El update de foreground usa basis_fg/g_axis. */
void adaline_eval(int32_t sin_q31, int32_t cos_q31, int32_t e_pk_mv,
                  int32_t *e_alpha_mv, int32_t *e_beta_mv) {
    const int32_t (*w)[2] = s_published[atomic_load_explicit(&s_active, memory_order_acquire)];
    /* Base en Q30: los vectores q15 del CORDIC tienen norma ≤ 1 + 2^-15 y
     * la recurrencia la amplifica; en Q31 el producto podría dar +2^31 y
     * cambiar de signo al castear. Con un bit de margen no hace falta
     * saturar (el clamp costaba 881 vs 365 ciclos, s26). */
    const int32_t c1 = cos_q31 >> 1, s1 = sin_q31 >> 1;
    int32_t ca = c1, sa = s1;
    int64_t aa = (int64_t)w[0][0]*ca + (int64_t)w[1][0]*sa;   /* Σ α, Q57 */
    int64_t ab = (int64_t)w[0][1]*ca + (int64_t)w[1][1]*sa;   /* Σ β */
    for (uint32_t k = 1U; k < H; k++) {
        const int32_t cn = (int32_t)(((int64_t)ca*c1 - (int64_t)sa*s1) >> 30);
        const int32_t sn = (int32_t)(((int64_t)sa*c1 + (int64_t)ca*s1) >> 30);
        ca = cn;
        sa = sn;
        aa += (int64_t)w[2*k][0]*ca + (int64_t)w[2*k+1][0]*sa;
        ab += (int64_t)w[2*k][1]*ca + (int64_t)w[2*k+1][1]*sa;
    }
    *e_alpha_mv = (int32_t)(((int64_t)e_pk_mv * (int32_t)(aa >> 30)) >> ADALINE_W_FRAC);
    *e_beta_mv  = (int32_t)(((int64_t)e_pk_mv * (int32_t)(ab >> 30)) >> ADALINE_W_FRAC);
}

/* Software q15 sin/cos for foreground. CORDIC is owned by the ISR.
 * Host/Python arithmetic tests use these same quantized inputs; this does
 * NOT claim bit-exact emulation of STM32's CORDIC silicon. */
static int32_t trig_q31(double v) {
    return clamp(lround(v * 32768.0), -32768, 32767) * 65536;
}

/* Recurrencia compleja Q31 (truncante) para el lazo lento: foreground no
 * toca el CORDIC (recurso de la ISR). Saturada porque los vectores
 * cuantizados q15 pueden tener norma levemente >1 (casting +2^31 a
 * int32 daría negativo). */
static void basis_q31(int32_t s1, int32_t c1, int32_t *c, int32_t *s) {
    c[0] = c1;
    s[0] = s1;
    for (unsigned k = 1; k < H; k++) {
        c[k] = clamp(((int64_t)c[k-1]*c1 - (int64_t)s[k-1]*s1) >> 31, INT32_MIN, INT32_MAX);
        s[k] = clamp(((int64_t)s[k-1]*c1 + (int64_t)c[k-1]*s1) >> 31, INT32_MIN, INT32_MAX);
    }
}

uint8_t adaline_update(int32_t ia, int32_t ib, int32_t dia, int32_t dib,
                       int32_t ua, int32_t ub, uint32_t theta_close,
                       int32_t omega) {
    const int32_t epk = (int32_t)(((int64_t)omega * KE_MV_Q18) >> 18);
    if (epk > -ADALINE_EPK_MIN_MV && epk < ADALINE_EPK_MIN_MV) return 0;
    const uint32_t theta = theta_close - (uint32_t)((int64_t)omega *
        MOTOR_POLE_PAIRS * (ADALINE_WIN_TICKS / 2U));
    const double th = (double)theta * (6.283185307179586 / 4294967296.0);
    int32_t c[H], s[H];
    basis_q31(trig_q31(sin(th)), trig_q31(cos(th)), c, s);
    const int32_t i[2] = {ia, ib}, di[2] = {dia, dib}, u[2] = {ua, ub};
    int32_t next[2 * H][2];
    for (unsigned eje = 0; eje < 2; eje++) {
        const int64_t ri = ((int64_t)(i[eje] >> 4) * M2PC_RK_Q8) >> 12;
        const int64_t ldi = ((int64_t)di[eje] * ADALINE_LK_Q15) >> 15;
        const int64_t eobs = u[eje] - ri - ldi;
        const int32_t ehat = (int32_t)(((int64_t)epk*g_axis(s_w, c, s, eje)) >> ADALINE_W_FRAC);
        /* Multiply: left shifting a negative signed value is undefined C. */
        const int32_t eps = clamp(((eobs-ehat)*32768)/epk, -4*32768, 4*32768);
        int64_t norm = 0;
        for (unsigned k = 0; k < 2*H; k++) {
            const int32_t x = (k & 1U) ? s[k/2] : c[k/2];
            const int64_t delta = ((int64_t)ADALINE_MU_Q16*x*eps) >> 35;
            next[k][eje] = clamp((int64_t)s_w[k][eje]+delta, -2*W_ONE, 2*W_ONE);
            norm += next[k][eje] < 0 ? -(int64_t)next[k][eje] : next[k][eje];
        }
        if (norm > 4*W_ONE) return 0;
    }
    memcpy(s_w, next, sizeof s_w);
    publish();
    return 1;
}

const int32_t *adaline_w_q27(void) { return &s_w[0][0]; }
