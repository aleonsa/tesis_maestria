/*
 * main.c — bring-up del encoder AS5600 vía I2C1 (PB8 = SCL, PB7 = SDA).
 *
 * Reloj:        HSE 8 MHz → PLL → SYSCLK = 170 MHz (ver clock.c).
 *               I2C1 kernel clock = HSI16 (lo enciende i2c1_init()).
 * Periféricos:  I2C1 master 100 kHz, USART2 (debug VCP), SysTick (timing).
 *
 * Validación:   girar el rotor a mano → ver por `cat /dev/ttyACM0` que el
 *               ángulo recorre 0..4095 de forma monótona y continua, con un
 *               solo salto por vuelta mecánica (el wrap 4095→0), y que STATUS
 *               reporta MD=1 (imán detectado).
 *               ✅ VALIDADO 2026-08-10.
 *
 * Alimentación: basta el USB del ST-LINK. El pad de 5 V de J8 sale del riel
 *               lógico, no de Vbus — comprobado leyendo el encoder con la
 *               fuente de banco apagada. Vbus solo hace falta para mover el
 *               motor.
 *
 * Cableado J8:  pad 2 (B+/H2) = PB7 = SDA
 *               pad 3 (Z+/H3) = PB8 = SCL
 *               pad 1 (A+/H1) = PB6 = libre  ← NO usar para I2C, ver i2c.c
 */

#include <stdio.h>
#include "stm32g431xx.h"
#include "clock.h"
#include "uart.h"
#include "i2c.h"
#include "as5600.h"
#include "swi2c.h"

/*
 * Camino de lectura. 0 = periférico I2C1 (normal). 1 = bit-banging.
 *
 * El bit-bang se conserva porque durante el bring-up fue lo que permitió
 * probar que el AS5600 estaba vivo mientras el periférico fallaba: al manejar
 * los pines como GPIO saca al I2C1 del circuito y desacopla "¿sirve el bus?"
 * de "¿sirve el sensor?". Vale su espacio en flash como red de seguridad.
 */
#define USE_SOFTWARE_I2C   0

/* Volcado de registros + estado de líneas al arrancar. Barato y ha pagado. */
#define DIAG_ON_BOOT       1

/*
 * Camino por IRQ (i2c1_read_regs_start + i2c1_nb_take) en vez del bloqueante.
 * Es el que va a usar el lazo de control; aquí se ejerce el mismo patrón
 * start/poll/take a mano para validarlo antes de montarle el PLL encima.
 * Solo aplica con USE_SOFTWARE_I2C == 0.
 */
#define USE_NONBLOCKING    1

/*
 * Prueba de rendimiento al arrancar: lecturas seguidas durante 1 s para medir
 * cuántas caben de verdad. La cuenta teórica a 100 kHz da ~2.2 kHz (~45 pulsos
 * de SCL por lectura), contra los ~7 kHz de refresco interno del AS5600. Si el
 * número medido se queda corto, la conclusión no es "optimizar el driver" sino
 * subir el bus a Fast Mode — y entonces hay que mirar con scope la red de J8
 * (1.8 kΩ serie + 10 kΩ pull-up), que está pensada para Hall push-pull.
 */
#define THROUGHPUT_TEST    1

/*
 * Prueba de una sola transferencia, con el bus en reposo antes y después.
 * Separa dos preguntas que el test de rendimiento mezcla: "¿funciona UNA
 * lectura por IRQ?" y "¿aguanta que la martilleemos sin pausa?". Sin
 * separarlas, un fallo no dice cuál de las dos cosas está mal.
 */
#define NB_SINGLE_SHOT     1

static volatile uint32_t g_ticks = 0;

void SysTick_Handler(void) {
    g_ticks++;
}

void SystemInit(void) {
    /* Vacío a propósito: el setup del reloj se hace desde main() para tener
     * control explícito del orden (mismo criterio que 01_blink/02_pwm_adc). */
}

static void systick_init(uint32_t ticks_per_irq) {
    SysTick->LOAD = ticks_per_irq - 1U;
    SysTick->VAL  = 0U;
    SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk
                  | SysTick_CTRL_TICKINT_Msk
                  | SysTick_CTRL_ENABLE_Msk;
}

static void delay_ms(uint32_t ms) {
    uint32_t start = g_ticks;
    while ((g_ticks - start) < ms) { }
}

#if ((USE_SOFTWARE_I2C == 0) && (USE_NONBLOCKING == 1))

/*
 * Vuelca el estado del último fallo con los flags del ISR desglosados. El
 * status dice QUÉ pasó, la fase dice DÓNDE, y el ISR dice qué tenía el
 * periférico en la mano en ese momento. Los tres juntos nombran el bug; el
 * status solo, no.
 */
static void nb_report_failure(const char *tag, i2c_status_t returned) {
    uint32_t isr = i2c1_nb_err_isr();

    /* Lo PRIMERO, el cable. Un bus con una línea clavada en bajo invalida todo
     * lo que venga debajo, y sin este renglón se pierde una sesión leyendo
     * flags de un periférico que no tiene con quién hablar. */
    uint8_t scl, sda, busy;
    i2c1_lines(&scl, &sda, &busy);
    printf("[%s] devolvio: %s\r\n", tag, i2c_status_str(returned));
    printf("[%s]   lineas: SCL(PB8)=%s SDA(PB7)=%s BUSY=%u%s\r\n",
           tag, scl ? "HIGH" : "LOW", sda ? "HIGH" : "LOW", (unsigned)busy,
           (!scl || !sda) ? "   <== LINEA CLAVADA, el bus no es usable" : "");

    /* Si no hubo cierre, todo lo de abajo es de un fallo anterior. Decirlo, en
     * vez de presentar datos viejos como si fueran de este intento. */
    i2c_nb_trace_t tr;
    i2c1_nb_trace(&tr);
    if (tr.closes == 0U) {
        printf("[%s]   la transferencia no llego a cerrar: no hay traza de\r\n"
               "[%s]   este intento (probablemente ni arranco).\r\n", tag, tag);
        return;
    }

    printf("[%s] ultimo cierre fallido: %s\r\n",
           tag, i2c_status_str(i2c1_nb_err_status()));
    printf("[%s]   fase=%s  ISR=0x%08lX\r\n",
           tag, i2c1_nb_err_phase(), (unsigned long)isr);
    printf("[%s]   BUSY=%u TC=%u TCR=%u STOPF=%u NACKF=%u "
           "RXNE=%u TXIS=%u TXE=%u | BERR=%u ARLO=%u OVR=%u\r\n",
           tag,
           (isr & I2C_ISR_BUSY)  ? 1U : 0U,
           (isr & I2C_ISR_TC)    ? 1U : 0U,
           (isr & I2C_ISR_TCR)   ? 1U : 0U,
           (isr & I2C_ISR_STOPF) ? 1U : 0U,
           (isr & I2C_ISR_NACKF) ? 1U : 0U,
           (isr & I2C_ISR_RXNE)  ? 1U : 0U,
           (isr & I2C_ISR_TXIS)  ? 1U : 0U,
           (isr & I2C_ISR_TXE)   ? 1U : 0U,
           (isr & I2C_ISR_BERR)  ? 1U : 0U,
           (isr & I2C_ISR_ARLO)  ? 1U : 0U,
           (isr & I2C_ISR_OVR)   ? 1U : 0U);

    i2c_nb_trace_t t = tr;
    printf("[%s]   eventos: TXIS=%u TC=%u RXNE=%u STOPF=%u NACKF=%u ERR=%u "
           "| irqs=%u  bytes=%u/%u\r\n",
           tag,
           (t.ev & (1U << 0)) ? 1U : 0U,
           (t.ev & (1U << 1)) ? 1U : 0U,
           (t.ev & (1U << 2)) ? 1U : 0U,
           (t.ev & (1U << 3)) ? 1U : 0U,
           (t.ev & (1U << 4)) ? 1U : 0U,
           (t.ev & (1U << 5)) ? 1U : 0U,
           (unsigned)t.irqs, (unsigned)t.idx, (unsigned)t.n);
    printf("[%s]   CR2 de lectura=0x%08lX  (SADD=0x%02lX NBYTES=%lu RD_WRN=%lu "
           "AUTOEND=%lu START=%lu STOP=%lu)  ISR@TC=0x%08lX\r\n",
           tag,
           (unsigned long)t.cr2_rx,
           (unsigned long)((t.cr2_rx >> 1) & 0x7FU),
           (unsigned long)((t.cr2_rx >> 16) & 0xFFU),
           (unsigned long)((t.cr2_rx & I2C_CR2_RD_WRN)  ? 1U : 0U),
           (unsigned long)((t.cr2_rx & I2C_CR2_AUTOEND) ? 1U : 0U),
           (unsigned long)((t.cr2_rx & I2C_CR2_START)   ? 1U : 0U),
           (unsigned long)((t.cr2_rx & I2C_CR2_STOP)    ? 1U : 0U),
           (unsigned long)t.isr_tc);

    /* La secuencia, en orden. Una fila por interrupción. */
    uint32_t lisr[I2C_NB_LOG_N];
    uint32_t lcr2[I2C_NB_LOG_N];
    uint8_t  k = i2c1_nb_log(lisr, lcr2, (uint8_t)I2C_NB_LOG_N);

    printf("[%s]   secuencia de %u IRQ (ISR al entrar / CR2):\r\n", tag, (unsigned)k);
    for (uint8_t i = 0U; i < k; i++) {
        printf("[%s]     #%u ISR=0x%08lX [%s%s%s%s%s%s%s] CR2=0x%08lX "
               "NBYTES=%lu ST=%lu SP=%lu\r\n",
               tag, (unsigned)i, (unsigned long)lisr[i],
               (lisr[i] & I2C_ISR_TXIS)  ? "TXIS "  : "",
               (lisr[i] & I2C_ISR_TC)    ? "TC "    : "",
               (lisr[i] & I2C_ISR_TCR)   ? "TCR "   : "",
               (lisr[i] & I2C_ISR_RXNE)  ? "RXNE "  : "",
               (lisr[i] & I2C_ISR_STOPF) ? "STOPF " : "",
               (lisr[i] & I2C_ISR_NACKF) ? "NACKF " : "",
               (lisr[i] & I2C_ISR_BUSY)  ? "BUSY"   : "",
               (unsigned long)lcr2[i],
               (unsigned long)((lcr2[i] >> 16) & 0xFFU),
               (unsigned long)((lcr2[i] & I2C_CR2_START) ? 1U : 0U),
               (unsigned long)((lcr2[i] & I2C_CR2_STOP)  ? 1U : 0U));
    }
}

/*
 * Una transferencia, esperando a que termine con plazo generoso. Devuelve el
 * status y deja el driver en IDLE.
 */
static i2c_status_t nb_one_shot(uint16_t *raw) {
    i2c_status_t st = i2c1_read_regs_start(AS5600_ADDR7, 0x0CU, 2U);
    if (st != I2C_OK) {
        /* No es lo mismo que fallar la transferencia: aquí el bus ni se movió.
         * Se distingue por el contador de cierres del reporte. */
        return st;
    }

    uint32_t t0 = g_ticks;
    while (i2c1_nb_state() == I2C_NB_BUSY) {
        if ((g_ticks - t0) >= 50U) { i2c1_nb_abort(); break; }
    }

    return as5600_raw_angle_take(raw);
}

/*
 * Plazo máximo que se le da a una transferencia antes de darla por perdida.
 * Una legítima son ~450 µs; 10 ms es factor ~20 y sigue siendo imperceptible.
 * El driver no lleva reloj a propósito: el timeout es del consumidor, que es
 * quien sabe a qué ritmo corre.
 */
#define NB_DEADLINE_MS   10U

/*
 * Bombea el camino por IRQ durante `ms` y devuelve las lecturas completadas.
 * El bucle no espera a nada: mira el estado, y arranca o recoge. Es el mismo
 * esqueleto que irá dentro de la ISR de control, con la diferencia de que allá
 * cada pasada la dispara el timer en vez de un while.
 */
static uint32_t nb_pump(uint32_t ms, uint16_t *last_raw, uint32_t *aborts) {
    uint32_t t0        = g_ticks;
    uint32_t seq0      = i2c1_nb_seq();
    uint32_t busy_from = g_ticks;

    while ((g_ticks - t0) < ms) {
        switch (i2c1_nb_state()) {
            case I2C_NB_IDLE:
                busy_from = g_ticks;
                (void)as5600_raw_angle_start();
                break;

            case I2C_NB_BUSY:
                if ((g_ticks - busy_from) >= NB_DEADLINE_MS) {
                    i2c1_nb_abort();
                    if (aborts != 0) (*aborts)++;
                }
                break;

            default:   /* DONE o ERR: recoger y dejar el driver libre */
                (void)as5600_raw_angle_take(last_raw);
                break;
        }
    }

    return i2c1_nb_seq() - seq0;
}

#endif

int main(void) {
    clock_init_170mhz_hse();
    systick_init(170000U);     /* 1 ms @ 170 MHz */
    uart2_init(115200U);
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("\r\n[boot] STM32G431 @170MHz (app: 03_as5600)\r\n");

#if (USE_SOFTWARE_I2C == 1)
    swi2c_init();
    printf("[i2c] bit-banging ~20kHz en PB8(SCL)/PB7(SDA)\r\n");
#else
    i2c1_init();
    printf("[i2c] I2C1 100kHz en PB8(SCL)/PB7(SDA), kernel=HSI16\r\n");

#if (DIAG_ON_BOOT == 1)
    /* Nivel real de los cables antes de hablar. Un bus I2C en reposo debe
     * estar en alto por los pull-ups; cualquier línea en bajo es un fallo
     * eléctrico, no de protocolo, y conviene saberlo antes de culpar al
     * esclavo. */
    uint8_t scl, sda, busy;
    i2c1_lines(&scl, &sda, &busy);
    printf("[diag] lineas en reposo: SCL(PB8)=%s  SDA(PB7)=%s  BUSY=%u\r\n",
           scl ? "HIGH" : "LOW", sda ? "HIGH" : "LOW", (unsigned)busy);
    if (!scl || !sda) {
        printf("[diag] linea clavada en BAJO: corto, esclavo reteniendo el\r\n"
               "       bus, o falta de pull-ups.\r\n");
        i2c1_dump_regs();
    }

    /* Confirma que el bus completa transacciones y que el 0x36 responde.
     * Un scan sin timeouts prueba que el periférico funciona aunque no
     * hubiera ningún esclavo. */
    i2c1_scan();
#endif
#endif

    /* Diagnóstico del imán antes del loop. MD=0 significa que el chip
     * contesta pero no ve el imán: problema mecánico, no de bus. */
    uint8_t st = 0U;
#if (USE_SOFTWARE_I2C == 1)
    int ok_st = (swi2c_read_regs(AS5600_ADDR7, 0x0BU, &st, 1U) == 0);
#else
    int ok_st = (as5600_status(&st) == I2C_OK);
#endif
    if (ok_st) {
        printf("[as5600] STATUS=0x%02X  MD=%d ML=%d MH=%d\r\n",
               (unsigned)st,
               (st & AS5600_STATUS_MD) ? 1 : 0,
               (st & AS5600_STATUS_ML) ? 1 : 0,
               (st & AS5600_STATUS_MH) ? 1 : 0);
        if ((st & AS5600_STATUS_MD) == 0U) {
            printf("[as5600] MD=0: contesta pero NO detecta iman.\r\n"
                   "         Revisar distancia y alineacion del iman.\r\n");
        }
    } else {
        printf("[as5600] no se pudo leer STATUS\r\n");
    }

#if ((USE_SOFTWARE_I2C == 0) && (USE_NONBLOCKING == 1))
    i2c1_nb_init();
    printf("[i2c] camino por IRQ armado (I2C1_EV/ER, prioridad NVIC 3)\r\n");

#if (NB_SINGLE_SHOT == 1)
    printf("\r\n[shot] 5 lecturas aisladas, 200 ms entre una y otra:\r\n");
    for (int k = 0; k < 5; k++) {
        uint16_t raw_s = 0U;
        i2c_status_t st = nb_one_shot(&raw_s);

        if (st == I2C_OK) {
            printf("[shot] %d: OK  raw=%u\r\n", k, (unsigned)raw_s);
        } else {
            nb_report_failure("shot", st);
        }
        delay_ms(200U);
    }
#endif

#if (THROUGHPUT_TEST == 1)
    {
        uint16_t raw_tp  = 0U;
        uint32_t aborts  = 0U;
        uint32_t err0    = i2c1_nb_errors();
        printf("\r\n[perf] martilleando 1 s sin pausa...\r\n");
        uint32_t reads   = nb_pump(1000U, &raw_tp, &aborts);

        printf("[perf] %lu lecturas/s  (teorico ~2200 a 100kHz)\r\n",
               (unsigned long)reads);
        printf("[perf] errores=%lu  abortadas=%lu  ultimo raw=%u\r\n",
               (unsigned long)(i2c1_nb_errors() - err0),
               (unsigned long)aborts, (unsigned)raw_tp);
        if (i2c1_nb_errors() != err0) {
            nb_report_failure("perf", i2c1_nb_err_status());
        } else if (reads == 0U) {
            /* Cero lecturas y cero cierres: ninguna transferencia arrancó. */
            nb_report_failure("perf", I2C_ERR_BUSY);
        }
        if (reads > 0U) {
            printf("[perf] refresco interno del AS5600 ~7 kHz: %s\r\n",
                   (reads >= 7000U) ? "el bus NO es el cuello de botella"
                                    : "el bus limita, el sensor tiene mas que dar");
        }
    }
#endif
#endif

    printf("\r\n[loop] gira el rotor a mano: raw debe recorrer 0..4095 de\r\n"
           "       forma monotona, con un solo salto por vuelta mecanica.\r\n");

    uint32_t err_count = 0U;

    while (1) {
        uint16_t raw = 0U;
        int ok;

#if (USE_SOFTWARE_I2C == 1)
        uint8_t b[2];
        ok = (swi2c_read_regs(AS5600_ADDR7, 0x0CU, b, 2U) == 0);
        if (ok) raw = (uint16_t)(((uint16_t)(b[0] & 0x0FU) << 8) | b[1]);
#elif (USE_NONBLOCKING == 1)
        /* Arranca, deja correr, recoge. Entre el start y el take el bus avanza
         * por IRQ; aquí no hay nada que hacer salvo esperar, pero en la ISR de
         * control ese hueco es donde vive el lazo entero. */
        uint32_t aborts = 0U;
        ok = (nb_pump(5U, &raw, &aborts) > 0U);
        /* El status real del último cierre, no uno inventado a partir del
         * estado. La versión anterior reportaba INCOMPLETE cuando el driver
         * estaba en IDLE porque nada había arrancado, que es justo lo que no
         * hay que hacer con un mensaje de error. */
        i2c_status_t r = ok ? I2C_OK : i2c1_nb_err_status();
#else
        i2c_status_t r = as5600_raw_angle(&raw);
        ok = (r == I2C_OK);
#endif

        if (ok) {
            if (err_count != 0U) {
                printf("[loop] recuperado tras %lu errores\r\n",
                       (unsigned long)err_count);
                err_count = 0U;
            }

            /* Grados ×10 en entero para no arrastrar float fuera de necesidad:
             *   deg10 = raw · 3600 / 4096.  raw·3600 ≤ 4095·3600 < 2^24,
             * así que cabe holgado en 32 bits. 4096 = 2^12 → shift, no
             * división. Coherente con la decisión de punto fijo para todo el
             * path de control. */
            uint32_t deg10 = ((uint32_t)raw * 3600U) >> 12;

            printf("raw=%4u  ang=%lu.%lu deg\r\n",
                   (unsigned)raw,
                   (unsigned long)(deg10 / 10U),
                   (unsigned long)(deg10 % 10U));
        } else {
            err_count++;
            /* Primeros 3 errores, luego uno de cada 50: sin rate-limit un bus
             * caído ahoga el VCP y no se ve nada más. */
            if ((err_count <= 3U) || ((err_count % 50U) == 0U)) {
#if (USE_SOFTWARE_I2C == 1)
                printf("[loop] error de lectura #%lu\r\n",
                       (unsigned long)err_count);
#else
                printf("[loop] error #%lu: %s\r\n",
                       (unsigned long)err_count, i2c_status_str(r));
#endif
            }
        }

        delay_ms(100U);   /* ~10 Hz: suficiente para leer a ojo girando a mano */
    }
}
