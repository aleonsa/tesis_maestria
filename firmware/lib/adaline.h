#ifndef ADALINE_H
#define ADALINE_H

#include <stdint.h>

/* Fourier/LMS, H=7, mu=131/65536 a 1 kHz. W y forma en Q27
 * (margen alrededor de ±1); base en Q31, corriente Q8 LSB, tensión mV.
 * Entrenamiento SOLO foreground; evaluación SOLO ISR. No usa periféricos.
 * Python: python_scripts/adaline_replica.py; tests: test_adaline_replica.py.
 * Es un modelo efectivo del inversor+motor, NO una medida de BEMF pura.
 * Sin corrección sinc de las medias de 1 ms: limitación experimental.
 */
#ifndef ADALINE_H_HARM
#define ADALINE_H_HARM       7
#endif
#define ADALINE_W_FRAC       27
#ifndef ADALINE_MU_Q16
#define ADALINE_MU_Q16       7    /* s26: 131 (2e-3) fue inestable en lazo
 * cerrado: ε_q 73→138 mA en 25 s hasta TRIP; la forma aprendida alimenta el
 * sector y el ruido del observador cierra un espiral. 7 ≈ 1.1e-4 rompe la
 * ganancia; convergencia ~20× más lenta pero estable en prueba. */
#endif
#define ADALINE_LK_Q15       3343
#ifndef ADALINE_EPK_MIN_MV
#define ADALINE_EPK_MIN_MV   150
#endif
#define ADALINE_WIN_TICKS    50U
/* Modelo inicial deliberadamente erróneo (experimento E, s26): amplitud
 * relativa y fase [°] del fundamental con que arranca W. 1.0/0 = nominal. */
#ifndef ADALINE_INIT_AMP
#define ADALINE_INIT_AMP     1.0f
#endif
#ifndef ADALINE_INIT_PHI_DEG
#define ADALINE_INIT_PHI_DEG 0.0f
#endif

/* Antes de habilitar el hook de control. 0=modelo sinusoidal; 1=cero.
 * En banco usar modelo como inicio; cero es una prueba separada. */
void adaline_init(uint8_t desde_cero);

/* Reutiliza sin/cos de Park: NO accede al CORDIC desde foreground.
 * Lee una instantánea completa de W publicada al final del update. */
void adaline_eval(int32_t sin_q31, int32_t cos_q31, int32_t e_pk_mv,
                  int32_t *e_alpha_mv, int32_t *e_beta_mv);

/* Medias de ventanas CONSECUTIVAS de 50 ticks. Si se pierde una ventana,
 * descartar el update y renovar i_previa (no fingir di=0).
 * omega es MECÁNICA Q32 vueltas/tick; theta es ELÉCTRICA al cierre.
 * Retorna 1 si publicó W, 0 por puerta de velocidad/límite de pesos.
 * LMS acota eps a ±4 y ||W_eje||_1 a 4 para limitar errores/overflow.
 */
uint8_t adaline_update(int32_t i_a_mean_q8, int32_t i_b_mean_q8,
                       int32_t di_a_q8, int32_t di_b_q8,
                       int32_t u_a_mean_mv, int32_t u_b_mean_mv,
                       uint32_t theta_close_q32, int32_t omega_q32);

/* Solo foreground, layout [a1_alpha,a1_beta,b1_alpha,b1_beta,...]. */
const int32_t *adaline_w_q27(void);

#endif
