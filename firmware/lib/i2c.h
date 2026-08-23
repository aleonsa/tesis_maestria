#ifndef I2C_H
#define I2C_H

#include <stdint.h>

/*
 * i2c.h — I2C1 master en PB8 (SCL) / PB7 (SDA).
 *
 * Encoder AS5600 en J8 de la B-G431B-ESC1.
 * Kernel clock = HSI16 (16 MHz), Standard mode 100 kHz.
 *
 * ⚠ SCL en PB8, NO en PB6: AF4 en PB6 no es I2C1_SCL en el STM32G431. Ver la
 *   cabecera de i2c.c y FIELD_NOTES N1.16 para la evidencia.
 *
 * DOS CAMINOS DE LECTURA, y la elección no es de estilo:
 *
 *   1. POLLING — i2c1_read_regs(). Bloquea ~450 µs por lectura de ángulo.
 *      Sirve para bring-up, diagnóstico y arranque, donde no hay lazo corriendo.
 *
 *   2. IRQ — i2c1_read_regs_start() + i2c1_nb_take(). La transferencia avanza
 *      en segundo plano; el llamador gasta ~4 µs repartidos en 4 IRQ. Es el
 *      único camino viable dentro del lazo de control: el periodo de la ISR a
 *      50 kHz son 20 µs y la lectura bloqueante tarda 22× MÁS que el ciclo
 *      entero. No es optimización, es requisito.
 *
 * DISEÑO (común a los dos): ninguna espera bloquea indefinidamente. Las de
 * polling llevan timeout y vigilan NACKF; las de IRQ no esperan nada. El driver
 * REPORTA en vez de colgarse. La primera versión usaba `while(flag){}` desnudos
 * y un esclavo mudo mataba el firmware sin dejar rastro de en qué paso murió.
 */

/* Resultado de una operación de bus. I2C_OK == 0. */
typedef enum {
    I2C_OK = 0,
    I2C_ERR_BUSY,            /* el bus estaba ocupado antes de arrancar     */
    I2C_ERR_NACK_ADDR,       /* nadie hizo ACK a la dirección               */
    I2C_ERR_NACK_DATA,       /* ACK a la dirección, NACK a un byte de datos */
    I2C_ERR_TIMEOUT_TXIS,    /* TXIS  nunca subió                           */
    I2C_ERR_TIMEOUT_TC,      /* TC    nunca subió                           */
    I2C_ERR_TIMEOUT_RXNE,    /* RXNE  nunca subió                           */
    I2C_ERR_TIMEOUT_STOPF,   /* STOPF nunca subió                           */
    I2C_ERR_ARG,             /* n = 0 o n > I2C_NB_MAXBYTES (bug del llamador)*/
    I2C_ERR_INCOMPLETE,      /* llegó el STOP con menos bytes de los pedidos */
    I2C_ERR_BUS,             /* BERR / ARLO / OVR — ver i2c1_nb_err_isr()    */
} i2c_status_t;

/* Nombre legible del status, para printf. */
const char *i2c_status_str(i2c_status_t st);

/*
 * Configura I2C1 + GPIOs PB8/PB7 + selecciona HSI16 como kernel clock.
 * Llamar DESPUÉS de clock_init_170mhz_hse().
 */
void i2c1_init(void);

/*
 * Lee `n` bytes consecutivos del esclavo `addr7` (7-bit, p.ej. 0x36 para el
 * AS5600) empezando en el registro `reg`. Patrón write-pointer +
 * repeated-START + read, aprovechando el auto-increment del esclavo.
 *
 * Bloquea (polling) pero con timeout. Devuelve I2C_OK o el error del paso
 * que falló.
 */
i2c_status_t i2c1_read_regs(uint8_t addr7, uint8_t reg, uint8_t *buf, uint8_t n);

/*
 * "¿Hay alguien en esta dirección?" con una escritura de 0 bytes: START +
 * dirección + STOP, la transacción más corta posible.
 *
 *   1 = ACK, hay dispositivo
 *   0 = NACK limpio → el reloj corrió hasta el noveno pulso: BUS SANO, nadie
 *       en esa dirección
 *  -1 = timeout → la transacción ni arrancó: bus clavado o SCL mal mapeada
 *
 * La distinción entre 0 y -1 es la herramienta de diagnóstico más útil del
 * módulo; fue la que localizó el pin equivocado el 2026-08-10.
 */
int i2c1_probe(uint8_t addr7);

/*
 * Barre las direcciones 7-bit válidas (0x08..0x77) e imprime las que
 * contestan. Devuelve cuántas encontró, o -1 si el bus está muerto (todos los
 * probes dieron timeout).
 */
int i2c1_scan(void);

/*
 * Estado eléctrico de las líneas SIN mover el bus: niveles lógicos de PB8/PB7
 * leídos del IDR del GPIO (el Schmitt trigger sigue conectado aunque el pin
 * esté en alternate function) y el bit BUSY del periférico.
 *
 * Separa dos escenarios que se confunden a simple vista:
 *   - líneas en alto  → bus en reposo sano, el esclavo simplemente no habla
 *   - alguna en bajo  → línea clavada (corto, o esclavo reteniendo el bus)
 *
 * Los tres punteros aceptan NULL.
 */
void i2c1_lines(uint8_t *scl_high, uint8_t *sda_high, uint8_t *busy);

/*
 * Reset por software del periférico (PE=0 → PE=1). RM0440 Rev 9 §39.4.6: bajar
 * PE suelta SCL/SDA y resetea las máquinas de estado internas sin tocar los
 * registros de configuración — la única forma de despegar BUSY cuando una
 * transacción quedó a medias.
 */
void i2c1_reset_peripheral(void);

/*
 * Vuelca clock gating, reset, fuente de kernel clock, CR1/TIMINGR/ISR y la
 * configuración de PB8/PB7 en el GPIO. Marca TIMINGR si no coincide con lo
 * escrito — un TIMINGR en 0 delata un periférico que no acepta escrituras.
 */
void i2c1_dump_regs(void);

/*
 * ═════════════════════════════════════════════════════════════════════════
 * LECTURA NO BLOQUEANTE POR IRQ
 *
 * El llamador arranca la transferencia y sigue con lo suyo; una máquina de
 * estados en I2C1_EV_IRQHandler la lleva hasta el final y deja los bytes en un
 * buffer interno. Cinco eventos por lectura de 2 bytes: TXIS (escribe el
 * puntero de registro), TC (repeated-START), RXNE una vez por byte, STOPF
 * (cierra).
 *
 * POR QUÉ ES SEGURO CORRERLO POR DEBAJO DEL CONTROL (prio 3 contra prio 1):
 * en los tres puntos donde esta máquina tiene que actuar, el maestro ESTIRA
 * SCL y el bus se queda esperándonos —
 *   RM0440 Rev 9 §39.4.7: «If TXE = 1 [...] the SCL line is stretched low until
 *   I2C_TXDR is written» y «If RXNE = 1 [...] stretched low until I2C_RXDR is
 *   read»;  §39.4.9, AUTOEND=0: «the TC flag is set and the SCL line is
 *   stretched low, to perform software actions».
 * Que la ISR de control preempte solo ALARGA la transacción. No se pierde ni se
 * corrompe un byte. Sin esa garantía habría que poner el I2C por encima del
 * lazo, lo que sería inaceptable.
 *
 * DUEÑO ÚNICO: un solo arrancador y un solo consumidor. No hay cola.
 * SIN TIMEOUT PROPIO: esta capa no tiene reloj. Si el bus se cuelga, el estado
 * se queda en BUSY para siempre; el consumidor (que sí tiene el tick de su ISR)
 * es quien decide cuándo llamar a i2c1_nb_abort().
 * ═════════════════════════════════════════════════════════════════════════
 */

/* Tope del buffer interno. 2 bytes es el ángulo; 4 deja aire sin gastar RAM. */
#define I2C_NB_MAXBYTES   4U

typedef enum {
    I2C_NB_IDLE = 0,   /* sin transferencia, listo para arrancar             */
    I2C_NB_BUSY,       /* en curso; la IRQ la está llevando                  */
    I2C_NB_DONE,       /* bytes listos: recogerlos con i2c1_nb_take()        */
    I2C_NB_ERR,        /* abortada; i2c1_nb_take() devuelve la causa         */
} i2c_nb_state_t;

/*
 * Arma las fuentes de evento del periférico y habilita I2C1_EV/I2C1_ER en el
 * NVIC con prioridad 3. Llamar DESPUÉS de i2c1_init(), y solo si se va a usar
 * el camino por IRQ: sin esta llamada el periférico no genera interrupciones y
 * el camino de polling sigue intacto.
 */
void i2c1_nb_init(void);

/*
 * Arranca la lectura de `n` bytes desde `reg` del esclavo `addr7`. NO bloquea:
 * vuelve en cuanto el START está pedido.
 *
 * I2C_OK          = arrancó
 * I2C_ERR_BUSY    = ya había una transferencia en curso, o el bus está ocupado
 * I2C_ERR_ARG     = n fuera de [1, I2C_NB_MAXBYTES]
 */
i2c_status_t i2c1_read_regs_start(uint8_t addr7, uint8_t reg, uint8_t n);

/* Estado de la transferencia. Barato: una lectura de RAM. */
i2c_nb_state_t i2c1_nb_state(void);

/*
 * Recoge el resultado y devuelve el driver a IDLE.
 *
 * I2C_OK  = copió min(n, bytes pedidos) en buf
 * otro    = la causa del fallo (o I2C_ERR_BUSY si aún está en curso)
 *
 * Llamar solo con estado DONE o ERR; con IDLE no hay nada que recoger.
 */
i2c_status_t i2c1_nb_take(uint8_t *buf, uint8_t n);

/*
 * Pide STOP y marca la transferencia como fallida, sin esperar. Es la salida
 * del consumidor cuando su propio timeout vence con el estado en BUSY. Si tras
 * esto el bus sigue clavado, el siguiente paso es i2c1_reset_peripheral().
 */
void i2c1_nb_abort(void);

/*
 * Contador monótono de lecturas completadas OK. Es la señal de "hay muestra
 * nueva" sin flag que limpiar: el consumidor guarda el último valor visto y
 * compara. Un contador no se pierde si el consumidor llega tarde.
 */
uint32_t i2c1_nb_seq(void);

/* Contador monótono de transferencias abortadas. */
uint32_t i2c1_nb_errors(void);

/*
 * Foto del I2C_ISR en el instante del último fallo, cualquiera que sea. Sin
 * esto un I2C_ERR_BUS no dice cuál de los tres flags fue, y un NACK no dice en
 * qué punto del bus estábamos.
 */
uint32_t i2c1_nb_err_isr(void);

/*
 * Nombre de la fase en la que murió la última transferencia fallida. Junto con
 * el status y el err_isr, es lo que convierte "falla" en un diagnóstico.
 */
const char *i2c1_nb_err_phase(void);

/* Status del último fallo. Sobrevive al i2c1_nb_take() que lo consumió. */
i2c_status_t i2c1_nb_err_status(void);

#endif
