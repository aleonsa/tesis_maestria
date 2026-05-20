# Field Notes

> Cuaderno pedagógico del banco experimental. Cada nota explica un concepto: panorama → analogía → detalle → por qué importa.
>
> **No es bitácora de avance.** El "qué hicimos" vive en [`PROGRESO_HARDWARE.md`](./PROGRESO_HARDWARE.md). Aquí vive el "por qué" y el "cómo se entiende".
>
> Cada entrada referencia el manual fuente (RM0440 §X, UM2516 Tabla N, DS12589) para profundizar.

---

## Índice

**Parte I — Fase 0: Infraestructura**

- [N0.1 — Bare-metal: programar el chip sin intermediarios](#n01--bare-metal-programar-el-chip-sin-intermediarios)
- [N0.2 — Cadena de herramientas: del editor al silicio](#n02--cadena-de-herramientas-del-editor-al-silicio)
- [N0.3 — CMSIS y vendor packs: el "header del chip"](#n03--cmsis-y-vendor-packs-el-header-del-chip)
- [N0.4 — El árbol de relojes y el PLL @ 170 MHz](#n04--el-árbol-de-relojes-y-el-pll--170-mhz)
- [N0.5 — SysTick: el latido del sistema](#n05--systick-el-latido-del-sistema)
- [N0.6 — UART y el VCP del ST-LINK: el "println" del embebido](#n06--uart-y-el-vcp-del-st-link-el-println-del-embebido)

**Parte II — Fase 1: Del silicio al motor**

- [N1.1 — Panorama de Fase 1: qué construimos y por qué](#n11--panorama-de-fase-1-qué-construimos-y-por-qué)
- [N1.2 — Pines TIM1 en B-G431B-ESC1: leyendo la Tabla 4](#n12--pines-tim1-en-b-g431b-esc1-leyendo-la-tabla-4)
- [N1.3 — Timers y counter modes: edge-aligned vs center-aligned](#n13--timers-y-counter-modes-edge-aligned-vs-center-aligned)
- [N1.4 — Del contador al pin: cómo el silicio genera PWM](#n14--del-contador-al-pin-cómo-el-silicio-genera-pwm)
- [N1.5 — Pines complementarios y dead-time: cómo el TIM1 evita el shoot-through](#n15--pines-complementarios-y-dead-time-cómo-el-tim1-evita-el-shoot-through)
- [N1.6 — TRGO: el cordón umbilical entre el TIM1 y el ADC](#n16--trgo-el-cordón-umbilical-entre-el-tim1-y-el-adc)
- [N1.7 — Break inputs: la protección de hardware contra el desastre](#n17--break-inputs-la-protección-de-hardware-contra-el-desastre)

---

# Parte I — Fase 0: Infraestructura

## N0.1 — Bare-metal: programar el chip sin intermediarios

### Panorama

Hay dos maneras de programar un STM32:

1. **Con HAL** (Hardware Abstraction Layer de ST). Llamas funciones tipo `HAL_GPIO_WritePin(...)` y la librería se encarga de tocar los registros por dentro. Es lo que usa la mayoría de tutoriales y el código autogenerado por STM32CubeIDE.

2. **Bare-metal**. Tú mismo escribes a los registros del chip. No hay librería intermedia. `*(volatile uint32_t *)0x48000814 = 0x40;` y listo, el pin PC6 cambió.

Nuestro proyecto usa **bare-metal CMSIS**, que es el punto medio: usa los headers oficiales de ARM/ST con los nombres simbólicos de los registros (`GPIOC->ODR`), pero no la HAL. Es como escribir ensamblador con nombres bonitos.

### Por qué bare-metal

Para una tesis sobre control predictivo, donde el algoritmo corre en una interrupción a 30 kHz (33 microsegundos por ciclo), HAL es problemático:

- **Indirección impredecible**. `HAL_ADC_Start_IT()` puede tener 5 capas de funciones antes de llegar al registro. Predecir el tiempo exacto es difícil.
- **Magia oculta**. La HAL hace cosas "por seguridad" que no necesitas (check de NULL, check de estado, etc.). Cada check es ciclos perdidos.
- **No se ve el chip**. Cuando algo no funciona, no sabes si el problema es tu lógica, la HAL, o el silicio.

Bare-metal te obliga a entender el periférico. Y cuando entiendes el periférico, **el debugging es trivial**: solo hay tres cosas en juego — tu código, los registros del chip, y la datasheet.

### Analogía

HAL es como manejar un carro automático con asistente de carril y frenado automático. Bare-metal es como manejar un manual: tienes que saber qué hace cada pedal, pero también tienes control total y entiendes exactamente qué está pasando.

Para un controlador de motor con timing crítico, queremos el manual.

### Por qué importa

Toda la cadena que vamos a escribir (PWM + ADC + ISR + FCS-MPC) tiene presupuesto temporal estricto: la ISR debe terminar en menos de 25 μs. Sin entender cuántos ciclos toma cada acceso a registro, no podemos garantizar ese presupuesto.

---

## N0.2 — Cadena de herramientas: del editor al silicio

### Panorama

Para que el código C que escribes en tu laptop termine ejecutándose en un chip ARM Cortex-M4 de 5×5 mm, pasa por varias herramientas en cadena:

```
   código C
       │
       ▼
  ┌─────────────────────┐
  │ arm-none-eabi-gcc   │  ← compilador cruzado: genera código ARM
  │ (cross-compiler)    │     desde una máquina x86/ARM macOS
  └─────────────────────┘
       │
       ▼
  archivo .elf (binario ARM con metadata)
       │
       ▼
  ┌─────────────────────┐
  │ openocd             │  ← traductor: habla con el ST-LINK por USB
  │                     │     y con el chip por protocolo SWD
  └─────────────────────┘
       │
       ▼
  ┌─────────────────────┐
  │ ST-LINK V2.1        │  ← hardware: USB en un lado,
  │ (sobre la placa)    │     SWD (Serial Wire Debug) en el otro
  └─────────────────────┘
       │
       ▼
  ┌─────────────────────┐
  │ STM32G431 (target)  │  ← el chip que ejecuta el código
  └─────────────────────┘
```

### Por qué Mac → Raspberry Pi → ST-LINK

La Mac corporativa tiene JumpCloud, que bloquea dispositivos USB de propósito específico (como el ST-LINK). Solución: una **Raspberry Pi 4B** corre el toolchain y se conecta físicamente al ST-LINK. La Mac edita por SSH (Zed Remote SSH) y un script `rsync` sincroniza el código.

```
Mac (edición)
    │
    │ ssh + rsync (passwordless con ed25519)
    ▼
Raspberry Pi (compilación, flash)
    │
    │ USB
    ▼
ST-LINK V2.1 (integrado a la placa B-G431B-ESC1)
    │
    │ SWD (4 hilos: SWDIO, SWCLK, GND, ref voltage)
    ▼
STM32G431 (target)
```

**Bonus inesperado**: la Pi quedó en el banco junto al osciloscopio, fuente y motor. La Mac queda en el escritorio. Workflow más limpio que tener todo en un solo sitio.

### Las herramientas, una por una

| Herramienta | Para qué | Dónde corre |
|---|---|---|
| **Zed (editor)** | Escribir código C | Mac (con SSH remoto a la Pi) |
| **rsync** | Sincronizar `firmware/` Mac → Pi | Mac |
| **cmake** | Generar Makefiles desde `CMakeLists.txt` | Pi |
| **arm-none-eabi-gcc** | Compilar C → binario ARM (.elf) | Pi |
| **openocd** | Flashear el .elf al chip | Pi |
| **ST-LINK** | Puente USB → SWD físico | Hardware (en la placa) |

### El protocolo SWD (Serial Wire Debug)

SWD es el protocolo por el cual el ST-LINK habla con el procesador del chip. Es un protocolo de 2 hilos (más GND y referencia de voltaje):

- **SWDIO** (Serial Wire Data Input/Output): hilo de datos bidireccional
- **SWCLK** (Serial Wire Clock): reloj generado por el ST-LINK

Por SWD el debugger puede:
- Leer y escribir cualquier dirección de memoria del chip (incluida la Flash).
- Pausar y arrancar el procesador.
- Leer registros del core (R0–R15, PC, etc.).
- Set/clear breakpoints en hardware.

Todo esto **sin necesidad de un bootloader corriendo en el chip**. Por eso un chip "ladrillo" se puede recuperar siempre con SWD: el silicio responde por sí mismo, no necesita que tu código colabore.

### Por qué importa

Cuando algo falle (y va a fallar), saber dónde puede estar el problema en esta cadena de 5 capas es clave. Si el flash no funciona, ¿es el `rsync`? ¿el `cmake`? ¿el ST-LINK? Conocer la cadena te dice dónde mirar primero.

---

## N0.3 — CMSIS y vendor packs: el "header del chip"

### Panorama

Un STM32G431 tiene cientos de periféricos, miles de registros. Cada registro vive en una dirección de memoria específica. Escribir esas direcciones a mano sería:

```c
*(volatile uint32_t *)0x48000814 = 0x40;   // ¿qué es esto?
```

Ilegible. Por eso ARM definió **CMSIS** (Cortex Microcontroller Software Interface Standard): una capa muy delgada de headers C estándar donde cada fabricante (ST, NXP, etc.) publica las direcciones y bits de su chip con nombres legibles.

Con CMSIS, lo anterior se escribe:

```c
GPIOC->ODR = GPIO_ODR_OD6;
```

Y eso resuelve a exactamente el mismo `0x48000814 = 0x40` después de la compilación. Cero costo en runtime, máxima legibilidad.

### Tres capas de CMSIS

```
┌──────────────────────────────────────────────────────┐
│  CMSIS-Core (ARM)                                    │
│  - Headers del Cortex-M4 (core_cm4.h, etc.)         │
│  - Mismo para CUALQUIER chip Cortex-M4               │
│  - Acceso a SysTick, NVIC, FPU                       │
└──────────────────────────────────────────────────────┘
            │
            │ usado por:
            ▼
┌──────────────────────────────────────────────────────┐
│  CMSIS-Device (vendor: ST en este caso)              │
│  - stm32g4xx.h (genérico para la familia G4)         │
│  - stm32g431xx.h (específico para nuestro chip)      │
│  - Direcciones y bits de cada periférico             │
└──────────────────────────────────────────────────────┘
            │
            │ usado por:
            ▼
┌──────────────────────────────────────────────────────┐
│  Tu código                                           │
│  GPIOC->ODR ^= GPIO_ODR_OD6;                         │
└──────────────────────────────────────────────────────┘
```

### Qué hay en el vendor pack de ST

El "vendor pack" de ST se llama **STM32CubeG4** (en GitHub: `STMicroelectronics/STM32CubeG4`). Lo descargamos con `git clone --depth=1`. Tres archivos clave para bare-metal:

1. **`stm32g431xx.h`** — define `GPIOC`, `RCC`, `TIM1`, `ADC1`, etc. como punteros a structs que mapean los registros.
2. **`startup_stm32g431xx.s`** — código ensamblador que ejecuta primero cuando arranca el chip. Setea el stack pointer, llama a `SystemInit()`, copia datos de Flash a RAM, llama a `main()`.
3. **Linker script** (`.ld`) — dice al linker dónde está la Flash (0x08000000), la RAM (0x20000000), y dónde colocar cada sección del binario.

Sin estos tres, no puedes compilar para este chip específico.

### Macros típicas: `_Msk` y `_Pos`

Vas a ver mucho:

```c
GPIOC->MODER &= ~GPIO_MODER_MODE6_Msk;
GPIOC->MODER |=  (0b01U << GPIO_MODER_MODE6_Pos);
```

Cada bit-field de un registro tiene dos macros:

- **`_Pos`** — posición del bit menos significativo del field. Ejemplo: `GPIO_MODER_MODE6_Pos = 12` (el field MODE6 ocupa los bits 12-13 del MODER).
- **`_Msk`** — máscara con 1's en todos los bits del field. Ejemplo: `GPIO_MODER_MODE6_Msk = 0b11 << 12 = 0x3000`.

Para escribir un valor a un field sin tocar los otros:

```c
REG &= ~FIELD_Msk;            // 1. limpia los bits del field
REG |=  (value << FIELD_Pos); // 2. escribe el nuevo valor
```

Este patrón aparece literalmente cien veces en tu código embebido. Internalízalo.

### Por qué importa

Sin CMSIS estarías escribiendo `0x48000814` y comparando con tablas de la datasheet a mano. Con CMSIS, leer un manual y traducirlo a código es directo: el manual dice "MODE6 bits 12:13 del MODER", y tú escribes `GPIO_MODER_MODE6_Pos`. Conexión 1:1.

---

## N0.4 — El árbol de relojes y el PLL @ 170 MHz

### Panorama

Todo lo que pasa en un chip síncrono está gobernado por un reloj. El STM32G431 puede correr hasta **170 MHz** (su máximo), pero al arrancar lo hace a **16 MHz** desde un oscilador interno barato y poco preciso (HSI).

Para subirlo a 170 MHz hay que:

1. Encender un oscilador externo de cristal de 8 MHz (HSE) — más preciso que el HSI interno.
2. Pasar esos 8 MHz por un **PLL** (Phase-Locked Loop): un multiplicador de frecuencia en hardware.
3. Configurar el árbol de prescalers para distribuir esos 170 MHz a los buses del chip.
4. Cambiar la fuente del SYSCLK del HSI al PLL.

### Qué es el árbol de relojes

No hay un solo reloj. Hay un **árbol**, con la raíz en algún oscilador y ramas hacia los periféricos:

```
                                                       ┌─→ AHB (bus rápido)  ─→ GPIO, DMA, etc.
HSE (8 MHz cristal)  ─┐                                │
                       ├─→ PLL ─→ SYSCLK (170 MHz) ───┼─→ APB1 (bus lento)  ─→ TIM2, USART2, etc.
HSI (16 MHz interno) ─┘                                │
                                                       └─→ APB2 (otro bus)   ─→ TIM1, ADC, etc.
```

El PLL multiplica la frecuencia por enteros. En nuestro caso:

```
HSE 8 MHz → ÷M=2 → 4 MHz → ×N=85 → VCO 340 MHz → ÷R=2 → SYSCLK 170 MHz
```

### Por qué importa la elección de M, N, R

Cada divisor del PLL tiene rangos de frecuencia válidos:

| Etapa | Rango válido | En nuestro caso |
|---|---|---|
| Entrada del PLL (después de M) | 2.66 – 16 MHz | 4 MHz ✓ |
| VCO interno (después de N) | 96 – 344 MHz | 340 MHz ✓ |
| Salida (después de R) | hasta 170 MHz | 170 MHz ✓ |

Si la entrada del PLL está fuera de rango, el PLL no engancha y el chip se queda esperando.

### El orden importa (mucho)

Hay un orden obligatorio para subir el reloj. **No es opcional**, viene de física del silicio:

1. **Habilitar el regulador de voltaje en "Boost mode"** (Range 1 Boost). Si vas a correr a más de 150 MHz, los transistores del chip necesitan más voltaje. Si no lo subes, falla.
2. **Subir los wait states de la Flash a 4**. Cuando el chip lee instrucciones de Flash a 170 MHz, la Flash misma no es tan rápida — necesita 4 ciclos de wait para cada lectura. Si no lo configuras, el chip lee basura.
3. **Encender HSE y esperar `HSERDY`**. El cristal tarda ~2 ms en estabilizar.
4. **Configurar el PLL** (todo el registro PLLCFGR de una vez, con el PLL apagado).
5. **Encender el PLL y esperar `PLLRDY`**.
6. **Cambiar SYSCLK al PLL** y esperar a que el switch se confirme.

Cada uno de esos pasos tiene un bit "RDY" que confirma cuando el hardware está listo. Saltarse esperarlos = chip que arranca con configuración inválida y se queda zombie.

### Por qué importa

Toda la cadena de tiempo (SysTick, baud rate UART, frecuencia PWM, frecuencia ADC, timing del FCS-MPC) depende del SYSCLK. Si el PLL no enganchó correctamente, **nada de lo que escribas después funcionará a la frecuencia que crees**.

Y peor: el bug es silencioso. El LED parpadea, la UART manda caracteres — pero a la frecuencia equivocada. Por eso la "validación cuádruple" del cierre de Fase 0 fue tan importante: confirmamos PLL + BRR + SysTick + UART **con una sola observación coherente** (el delta de 502 ms entre prints).

---

## N0.5 — SysTick: el latido del sistema

### Panorama

Un programa típico necesita esperar tiempos definidos:

- "Espera 500 ms entre parpadeos del LED."
- "Espera 1 segundo y luego retransmite el mensaje."
- "Timeout de 100 ms para esta lectura I²C."

Hacer esto con bucles vacíos (`for (i=0; i<1000000; i++);`) es terrible: depende de la frecuencia del CPU, del compilador, del optimizador. Si subes el reloj, los delays cambian.

La solución correcta es un **timer dedicado al timing del sistema**. ARM definió uno como parte del estándar Cortex-M: **SysTick**.

### Qué es SysTick

SysTick vive **dentro del core Cortex-M4**, no como un periférico de ST. Cualquier chip Cortex-M lo tiene, con los mismos registros y comportamiento. Es básicamente:

```
┌─────────────────────────────────────┐
│  SysTick (en el core ARM)           │
│                                     │
│  - Contador de 24 bits              │
│  - Decrementa desde LOAD a 0        │
│  - Cuando llega a 0, lanza una IRQ  │
│  - Se recarga automáticamente       │
└─────────────────────────────────────┘
```

Si configuras `LOAD = 170000` con clock a 170 MHz, el contador tarda 170000 ciclos = 1 ms en llegar a 0. Cada milisegundo, dispara una interrupción.

En la interrupción, incrementas una variable global:

```c
static volatile uint32_t g_ticks = 0;

void SysTick_Handler(void) {
    g_ticks++;
}
```

Y entonces `g_ticks` es un "reloj de pared" en milisegundos. Para esperar 500 ms:

```c
uint32_t start = g_ticks;
while ((g_ticks - start) < 500) { }
```

### Por qué la resta funciona con wrap-around

`g_ticks` es `uint32_t`. Llega a su máximo (~4.29 mil millones, o 49.7 días) y vuelve a 0. La resta unsigned está bien definida en C **incluso si la diferencia cruza el wrap**:

```
g_ticks = 0xFFFFFFFE   (casi al máximo)
start   = 0xFFFFFFFC

g_ticks - start = 2    (correcto)

g_ticks = 0x00000001   (después de wrap)
start   = 0xFFFFFFFC

g_ticks - start = 5    (correcto: cruzó el wrap)
```

Es decir, el código `(g_ticks - start) < ms` siempre da el delta correcto, sin importar dónde esté `g_ticks` en su ciclo de 49.7 días. **Solo si guardaras `g_ticks > start + ms` tendrías bug** (porque `start + ms` puede overflowear). Resta primero, compara después.

### Por qué el handler no es `static`

```c
void SysTick_Handler(void) { ... }  // NO es static
```

El **startup file** (`startup_stm32g431xx.s`) define una tabla de vectores de interrupción. Para cada interrupción posible (SysTick, EXTI, ADC, etc.) hay una entrada que apunta a una función. El handler de SysTick está declarado como `weak` (débil) en el startup, apuntando a un loop infinito (`Default_Handler`).

Cuando tú escribes `void SysTick_Handler(void) { ... }` en tu código sin `static`, el linker prefiere tu definición (no-weak) sobre la del startup (weak). El nombre tiene que ser exacto — `SysTick_Handler`, ni `systick_handler` ni `SysTick_handler`. Es como un override de C++ pero sin sintaxis explícita.

Si pones `static`, el linker no lo ve fuera del archivo, no lo encuentra al resolver la tabla de vectores, y se queda con el `Default_Handler`. Resultado: tu handler nunca se ejecuta, `g_ticks` se queda en 0, `delay_ms()` bloquea para siempre.

### Por qué importa

Toda la base de tiempo del programa principal (loop a 1 Hz del blink, eventualmente timeouts en lecturas, watchdogs lógicos) cuelga del SysTick. Es la primera piedra de toda la lógica temporal del firmware. Sin él, no hay determinismo.

---

## N0.6 — UART y el VCP del ST-LINK: el "println" del embebido

### Panorama

En desarrollo "normal" tienes `console.log()`, `print()`, `printf()` — el output va a una ventana de terminal en tu computadora. En embebido no hay terminal. El chip no tiene pantalla. ¿Cómo debugueas?

Tres opciones:

1. **Hacer parpadear LEDs en patrones** — primitivo y limitado.
2. **Mandar texto por un pin** — el equivalente a `println()`, pero el chip habla por un cable.
3. **Pausar con un debugger (GDB)** — útil, pero rompe el tiempo real.

La opción 2 es UART (Universal Asynchronous Receiver-Transmitter). Y es la que usamos.

### Qué es UART

UART es un protocolo serial muy simple:

```
TX (transmit del chip)  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                            └─→ a otro chip o conversor

RX (receive del chip)   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                            ←─ otro chip le habla
```

Cada byte se manda como una secuencia de bits a una **frecuencia acordada** (baud rate). A 115200 baudios, cada bit dura 1/115200 ≈ 8.68 microsegundos.

Estructura de un byte UART:

```
idle: línea alta (1)
      ┌─ start bit (siempre 0)
      │  ┌─ 8 bits de datos (LSB primero)
      │  │  ┌─ stop bit (siempre 1)
      │  │  │
___╲__|__||||||||__|___
     0  bits_datos  1
```

10 bits transmitidos por cada byte de datos. A 115200 baudios = ~11.5 KB/s útiles.

### El VCP del ST-LINK

Aquí está la magia de la placa B-G431B-ESC1: el chip ST-LINK que sirve para flashear y debuggear **también** funciona como un **conversor USB ↔ UART**. Esto se llama **VCP (Virtual COM Port)**.

```
G431 (target)              ST-LINK (debugger)             Raspberry Pi
   PB3 (TX) ────────────→ PA3 del F103                       │
                                │                            │
                                ▼                            │
                          firmware del ST-LINK               │
                                │                            │
                                ▼                            │
                          USB CDC ─────────────→ /dev/ttyACM0
                                                             │
                                                  cat /dev/ttyACM0
                                                     muestra "tick N..."
```

Ventaja: **un solo cable USB del ST-LINK a la Pi cumple 3 funciones**: alimentación lógica del chip + SWD para flash/debug + UART para `printf`. Cero hardware adicional.

### Configurar UART = 7 pasos en orden

```c
// 1. Habilitar reloj al puerto GPIO donde están los pines TX/RX
RCC->AHB2ENR |= RCC_AHB2ENR_GPIOBEN;

// 2. Habilitar reloj al periférico USART2
RCC->APB1ENR1 |= RCC_APB1ENR1_USART2EN;

// 3. Configurar PB3 y PB4 como "Alternate Function"
GPIOB->MODER ... MODE3 = 0b10, MODE4 = 0b10;

// 4. Decirle al GPIO qué AF específica (AF7 = USART2 en STM32G4)
GPIOB->AFR[0] ... AFSEL3 = 7, AFSEL4 = 7;

// 5. Limpiar CR1/CR2/CR3 (estado conocido)
USART2->CR1 = 0;
USART2->CR2 = 0;
USART2->CR3 = 0;

// 6. Configurar baud rate
USART2->BRR = (clock + baud/2) / baud;  // 1476 para 115200 @ 170 MHz

// 7. Habilitar TX, RX, y por último UE (UART Enable)
USART2->CR1 = USART_CR1_TE | USART_CR1_RE;
USART2->CR1 |= USART_CR1_UE;
```

El último paso (UE al final) no es opcional. Mientras UE=0 puedes modificar registros de configuración. Una vez UE=1, muchos quedan read-only. Si pones UE primero, los demás writes se ignoran y la UART manda basura.

### El cálculo del BRR

El registro `BRR` controla el baud rate:

$$\text{BRR} = \frac{f_{clk}}{\text{baud}}$$

A 170 MHz y 115200 baudios:

$$\text{BRR} = \frac{170\,000\,000}{115\,200} = 1475.69$$

No es entero. Hay que redondear. La fórmula `(clock + baud/2) / baud` hace round-to-nearest (sumar la mitad del divisor antes de dividir es el truco clásico para redondear con división entera):

$$\text{BRR} = \frac{170\,000\,000 + 57\,600}{115\,200} = 1476$$

Baud rate real: 170e6 / 1476 = 115176 baud. Error = (115176 − 115200) / 115200 = **−0.02%**.

UART tolera hasta ~2% de error sin perder bytes. -0.02% es perfecto.

### Retargeting de `printf`

Para que `printf()` salga por UART (en vez de buscar una pantalla que no existe), redefines la función `_write()` que newlib usa internamente:

```c
int _write(int fd, char *buf, int len) {
    (void)fd;  // ignoramos el file descriptor
    for (int i = 0; i < len; i++) uart2_putc(buf[i]);
    return len;
}
```

Cuando llamas `printf("tick %d\n", i)`, newlib formatea el string a un buffer y luego llama `_write(fd=1, buf, len)`. Tú interceptas esa llamada y mandas cada byte por UART. Funciona transparentemente.

### La validación cuádruple

Cuando vimos en `cat /dev/ttyACM0`:

```
tick 28  uptime=14050 ms
tick 29  uptime=14552 ms     ← delta = 502 ms
tick 30  uptime=15054 ms
```

Esa observación de **502 ms** entre prints (no 500) confirma simultáneamente **4 cosas**:

| Confirma | Cómo |
|---|---|
| PLL @ 170 MHz | Si el clock estuviera mal, BRR daría baud incorrecta → texto sería basura. Pero leemos texto coherente. ✓ |
| BRR calculado bien | Mismo razonamiento. ✓ |
| SysTick 1 ms | El `delay_ms(500)` espera exactamente 500 ms si SysTick está correcto. ✓ |
| UART polling | Los 2 ms extra del delta son el tiempo de transmitir 24 caracteres a 115200 baud (24 × 87 μs ≈ 2.08 ms). ✓ |

**Predicción teórica (2.08 ms) coincide con observación (2 ms) al ms.** Esa es la prueba más limpia de que toda la cadena de Fase 0 está calibrada correctamente.

### Por qué importa

`printf` no es lujo, es la única ventana al chip durante runtime. En Fase 1 lo usaremos para reportar valores del ADC, tiempos de ISR, estado del PWM. Sin él, debuggear es ciego.

---

# Parte II — Fase 1: Del silicio al motor

## N1.1 — Panorama de Fase 1: qué construimos y por qué

### Panorama

El algoritmo FCS-M2PC + ADALINE ya está implementado en MATLAB (`fcs_m2pc_v2/`). En la simulación tienes acceso a corrientes, posiciones, BEMF "perfectas". En el banco real, no — esas señales tienen que venir del hardware, sincronizadas con precisión de microsegundos.

**Fase 1 construye toda la infraestructura entre el algoritmo y el motor**, sin tocar el algoritmo todavía:

```
┌─────────────────────────────────────────────────────────────────┐
│  Algoritmo FCS-M2PC + ADALINE (todavía no portado)              │
│                                                                  │
│  recibe:  i_α, i_β  (corrientes medidas)                        │
│           θ_e       (posición eléctrica)                        │
│  produce: 6 señales PWM con duty cycles calculados              │
└─────────────────────────────────────────────────────────────────┘
       ▲                                          │
       │                                          ▼
       │                                ┌─────────────────────┐
       │                                │  TIM1 (timer 1)     │
       │                                │  - PWM 30 kHz       │
       │                                │  - 6 salidas        │
       │                                │  - dead-time        │
       │                                └─────────┬───────────┘
       │                                          │
       │                                          ▼
       │                                ┌─────────────────────┐
       │                                │  Gate drivers L6387 │
       │                                │  + MOSFETs          │
       │                                └─────────┬───────────┘
       │                                          │
       │                                          ▼
       │                                ┌─────────────────────┐
       │                                │     Motor 2804      │
       │                                └─────────┬───────────┘
       │                                          │
       │                          ┌───────────────┴─────────┐
       │                          ▼                         ▼
       │              ┌─────────────────────┐   ┌─────────────────────┐
       │              │  3 shunts + OPAMP   │   │     AS5600 (I²C)    │
       │              │  (current sensing)  │   │   (position sensor) │
       │              └─────────┬───────────┘   └─────────┬───────────┘
       │                        │                         │
       │                        ▼                         ▼
       │              ┌─────────────────────┐   ┌─────────────────────┐
       │              │  ADC1 + ADC2 dual   │   │       I²C1          │
       │              └─────────┬───────────┘   └─────────┬───────────┘
       │                        │                         │
       └────────────────────────┴─────────────────────────┘
```

Todo eso es Fase 1. Cuatro piezas:

| Pieza | Periférico del chip | Pines | Frecuencia |
|---|---|---|---|
| **PWM 3-fásico complementario** | TIM1 | PA8, PA9, PA10 + PC13, PA12, PB15 | 30 kHz |
| **Lectura de corriente sincronizada** | ADC1, ADC2, OPAMP1/2/3 | Curr_fdbk1/2/3 | 30 kHz (1 muestra/PWM) |
| **Lectura de posición** | I²C1 | PB6, PB7 | ~6.6 kHz (limitado por AS5600) |
| **Sincronización entre todo** | TIM1 TRGO → ADC JEXTSEL | (interno) | 30 kHz |

### El "metrónomo"

Lo más importante de Fase 1 no es ninguno de los periféricos individuales — es la **sincronización**.

```
tiempo →

PWM:    ↑              ↑              ↑              ↑
        │              │              │              │   ← center-aligned
   ↗↘   │   ↗↘   ↗↘   │   ↗↘   ↗↘   │   ↗↘   ↗↘   │   ↗↘
ADC:    ●              ●              ●              ●
        ▲              ▲              ▲              ▲   ← muestrea en pico de PWM
        │              │              │              │
ISR:    │  ┌─┐         │  ┌─┐         │  ┌─┐         │  ┌─┐
        │  │ │         │  │ │         │  │ │         │  │ │   ← FCS-MPC corre
        │  │ │         │  │ │         │  │ │         │  │ │     entre muestras
        │__│ │_________│__│ │_________│__│ │_________│__│ │__
        ◄───────────►
            33 μs (período de PWM)
```

El timer (TIM1) es el director de orquesta. Cada vez que llega a su pico, dispara automáticamente al ADC ("muestrea ahora"). Cuando el ADC termina, dispara la ISR ("calcula el siguiente PWM"). La ISR escribe los nuevos duty cycles. El TIM1 los aplica en el siguiente ciclo.

**Toda esta cadena tiene que completarse en menos de 33 μs**, idealmente menos de 25 μs para tener margen. Esa es la restricción dura de Fase 1.

### Plan por semana (del PROGRESO_HARDWARE.md)

| Semana | Objetivo | Entregable |
|---|---|---|
| 4 | TIM1 generando PWM 3-fásico centrado a 30 kHz con dead-time ~500 ns | Oscilograma de las 6 salidas |
| 5 | ADC1+ADC2 leyendo 3 corrientes + Vbus, sincronizados con TIM1 | Log por UART de los valores |
| 6 | ISR completa con calibración de offsets y medición de tiempo | Tiempo de ISR confirmado < 25 μs |
| 7 | AS5600 vía I²C1 con extrapolación de θ entre lecturas | Posición legible girando el motor a mano |

Final de Fase 1: **decisión crítica**. Si la ISR no entra en presupuesto, hay que ajustar Ts a 50 μs (20 kHz), usar el coprocesador CORDIC para acelerar el cálculo de sin/cos, o reducir el horizonte de predicción.

### Por qué importa

Sin Fase 1, no hay banco. El algoritmo en MATLAB es bonito pero teórico. Es esta fase donde la tesis "se hace real". Y como dijimos: si la sincronización no es perfecta, FCS-MPC no funciona — no es que dé peor resultado, simplemente colapsa porque su modelo predictivo asume timing exacto.

---

## N1.2 — Pines TIM1 en B-G431B-ESC1: leyendo la Tabla 4

### Panorama

El chip STM32G431 ofrece **múltiples opciones** para los pines de cada periférico. Por ejemplo, TIM1_CH1 puede salir físicamente por PA8 (default), PE9 (alternativo), o C5 (en chips más grandes). Esto se llama **Alternate Functions (AF)**.

Cada placa concreta (Nucleo, Discovery, B-G431B-ESC1) tiene su **routeado físico**: el fabricante eligió pines específicos para sus funciones y los soldó a sus drivers, conectores, etc. **Esa elección es irreversible en hardware**.

Para programar correctamente, necesitas saber qué pines del chip están routeados a qué función en TU placa. Esto vive en la **Tabla 4 del User Manual de la placa (UM2516)**.

### La trampa que casi cometemos

Por convención general del STM32G4, **PA2/PA3** son la opción más común para USART2. Si lees el datasheet del chip sin contexto, asumirías eso.

Pero en la B-G431B-ESC1, ST routeó:

| Pin | En el chip "normal" | En la B-G431B-ESC1 |
|---|---|---|
| PA2 | USART2_TX (por default) | **OP1_OUT** (frontend analógico del shunt 1) |
| PA3 | USART2_RX (por default) | **Curr_fdbk1_OPAmp-** (entrada inversora del OPAMP) |
| PB3 | TIM2_CH2 / SPI3_SCK | **USART2_TX** (routeado al VCP del ST-LINK) |
| PB4 | TIM3_CH1 / SPI1_MISO | **USART2_RX** |

Es decir, los pines "obvios" están ocupados por el current sensing del motor. ST tuvo que mover USART2 a pines secundarios.

**Lección de método**: nunca asumas que un pin "default del chip" es lo mismo que "default de la placa". La Tabla 4 del UM2516 es la fuente autoritativa. Esto nos pasó al final de la sesión 5 y nos costó debug.

### Pinout de TIM1 en B-G431B-ESC1 (confirmado en sesión 6)

| Canal | High-side | Complementario (low-side) | Notas |
|---|---|---|---|
| TIM1_CH1 | **PA8** | **PC13** | CH1N en PC13 es raro — ese pin está compartido con TAMP/RTC en otros chips. ST lo dedicó aquí al gate driver. |
| TIM1_CH2 | **PA9** | **PA12** | |
| TIM1_CH3 | **PA10** | **PB15** | |

Estos 6 pines van directamente al chip gate driver **L6387**, que a su vez maneja los 6 MOSFETs de potencia (STL180N6F7) del puente trifásico.

### Otros pines relevantes para Fase 1

De la Tabla 4 del UM2516:

| Pin | Función en la placa | Notas para Fase 1 |
|---|---|---|
| PA0 | Vbus_sense | ADC: voltaje de bus DC dividido |
| PA1 | Curr_fdbk1_OPAmp+ | Entrada al OPAMP1 del shunt 1 |
| PA3 | Curr_fdbk1_OPAmp- | Entrada inversora |
| PA5 | Curr_fdbk3_OPAmp+ | Shunt 3 |
| PA7 | Curr_fdbk2_OPAmp+ | Shunt 2 |
| PB14 | Temperature feedback | ADC: NTC en MOSFET |
| PB13 | N.C. | No conectado |
| PB6 | I²C1_SCL | AS5600 SCL (J8) |
| PB7 | I²C1_SDA | AS5600 SDA (J8) |
| PB8 | I/O libre (J8 pad Z+/H3) | Reservado para instrumentación con scope |
| PC6 | LED STATUS | LED user del blink |

### Por qué importa

Antes de tocar **un solo registro** del TIM1, hay que tener esta tabla en la cabeza. Si configuras `GPIOA->AFR[1]` para PA8 = AF6 cuando AF para TIM1 es **AF6** ✓ (sí, coincide), no hay problema. Pero si fuera AF2, tu PWM aparecería en otro pin y nunca llegaría al gate driver — y debugar eso a ciegas es horrible.

Confirmación de AF para TIM1 en STM32G431 (DS12589 Tabla 13): **AF6** para CH1/CH2/CH3 en PA8/PA9/PA10, **AF6** también para los CHxN en sus pines. Bonus: todos los pines TIM1 usan la misma AF, lo cual simplifica el código.

---

## N1.3 — Timers y counter modes: edge-aligned vs center-aligned

### Panorama

Un timer en un microcontrolador es, en su forma más simple, **un contador binario que incrementa una vez por ciclo del reloj** y reinicia cuando llega a un valor límite.

```
        ┌─────────────────────────────────┐
        │       TIMER (silicio)           │
        │                                 │
   clk ─→ ╔═══════════╗                   │
        │ ║  CONTADOR ║ ─→ valor actual   │
        │ ╚═══════════╝                   │
        │                                 │
        └─────────────────────────────────┘
```

A 170 MHz, ese contador incrementa 170 millones de veces por segundo. El periférico TIM1 tiene un contador de **16 bits** (0 a 65535), pero gracias a la enorme frecuencia, puede generar pulsos PWM de 30 kHz cómodamente.

### Las piezas alrededor del contador

1. **ARR (Auto-Reload Register)**: el valor límite. "Cuando llegues aquí, reinicia."
   → Controla la **frecuencia** del PWM.

2. **CCR1, CCR2, CCR3 (Capture/Compare Registers)**: valores umbral.
   → Cada uno se compara con el contador y cuando coinciden, conmuta una salida.
   → Controlan el **duty cycle** de cada fase.

3. **Salidas físicas**: el periférico ya está routeado a pines específicos (en nuestra placa: PA8, PA9, PA10).

### Edge-aligned: el modo "obvio"

El contador sube monotónicamente, llega a ARR, y salta de golpe a 0:

```
contador
ARR ────┌────┌────┌────┌
       ╱│   ╱│   ╱│   ╱│
      ╱ │  ╱ │  ╱ │  ╱ │
     ╱  │ ╱  │ ╱  │ ╱  │
    ╱   │╱   │╱   │╱   │
  0 ────┘    ┘    ┘    ┘
```

PWM resultante (asumiendo `PA8 = high cuando contador < CCR1`):

```
PA8:    ┌───┐ ┌───┐ ┌───┐ ┌───┐
        │   │ │   │ │   │ │   │
   _____┘   └─┘   └─┘   └─┘   └__
        ▲ ▲
        │ └─ flanco de bajada (cuando contador == CCR1)
        └─── flanco de subida (cuando contador == 0)
```

Es simple, pero asimétrico: el flanco de subida siempre cae en el inicio del ciclo. Si tienes 3 fases con duty cycles distintos, los 3 flancos de subida coinciden temporalmente. Eso produce **picos grandes de corriente común** en el bus DC, peor EMI, y dificulta el muestreo síncrono.

### Center-aligned: el contador como triángulo

El contador sube de 0 a ARR, **luego baja** de ARR a 0, luego vuelve a subir:

```
contador
ARR        ╱╲      ╱╲      ╱╲
          ╱  ╲    ╱  ╲    ╱  ╲
         ╱    ╲  ╱    ╲  ╱    ╲
        ╱      ╲╱      ╲╱      ╲
  0 ───╯
       ◄──────►
        un periodo completo = 2 × ARR ciclos
```

PWM resultante:

```
PA8:    ┌────┐    ┌────┐    ┌────┐
   _____│    │____│    │____│    │___
        ▲  ▲ ▲  ▲ ▲  ▲ ▲  ▲ ▲  ▲
        │  │ │  │ │  │
        │  │ │  └ flanco de bajada (en down-count, contador == CCR1)
        │  │ └─ pico del triángulo (contador == ARR)
        │  └─ flanco de subida (en up-count, contador == CCR1)
        └─ inicio del ciclo (contador == 0)
```

El pulso PWM queda **centrado** sobre el pico del triángulo. Es simétrico.

### Por qué center-aligned es obligatorio en motores BLDC/PMSM

**1. Muestreo "limpio" de la corriente.**

La corriente real en el motor tiene un rizado triangular sobre el valor promedio:

```
corriente
    ▲             pico de PWM
    │                ╲
    │              ╱──╲           ← el rizado es triangular,
    │            ╱      ╲             su valor PROMEDIO está en el medio del pulso
    │          ╱   ✱     ╲
    │        ╱            ╲
    │      ╱               ╲
    │ ───╱                  ╲───
    └─────────────────────────────→ tiempo
                ✱
                │
                └ momento óptimo de muestreo: pico o valle del contador
```

Si muestreas en el pico/valle, obtienes el valor **promedio** de la corriente en ese ciclo. Si muestreas en cualquier otro momento, obtienes corriente + rizado aleatorio.

Con center-aligned, el timer puede generar automáticamente una señal interna ("TRGO") que dispara al ADC justo en el pico. Con edge-aligned, esto no se puede sincronizar fácilmente.

**2. Menor ripple acústico y EMI.**

Los flancos de las 3 fases no caen simultáneamente, sino distribuidos simétricamente. Resultado: menos derivadas instantáneas grandes, menos zumbido audible, menos interferencia electromagnética.

**3. Modelo predictivo más limpio.**

En FCS-MPC, tu predicción de corriente para el siguiente instante asume que aplicas un voltaje promedio durante todo el periodo PWM. Eso solo es exactamente cierto si el muestreo y el centro del PWM coinciden. Con center-aligned, esto pasa por construcción.

### Los 3 sub-modos center-aligned

En el registro `CR1` del TIM1, los bits 6:5 (`CMS[1:0]`) configuran el sub-modo:

| CMS | Modo | Cuándo se setea la flag de output compare (CCxIF) |
|---|---|---|
| 00 | Edge-aligned | (otro modo, sube y reinicia) |
| 01 | **Center-aligned mode 1** | Solo en down-count |
| 10 | Center-aligned mode 2 | Solo en up-count |
| 11 | Center-aligned mode 3 | En ambos |

**Para FCS-MPC**: CMS = 01 (Mode 1). Razón:
- Una sola interrupción de output compare por ciclo PWM.
- El "update event" (overflow + underflow) sigue ocurriendo, así que TRGO puede dispararse 2 veces por periodo si quisiéramos doble muestreo (para Fase 1 una basta).

### Cálculo de ARR para 30 kHz

Con HCLK = 170 MHz y prescaler PSC = 0:

$$f_{PWM} = \frac{f_{clk}}{2 \cdot ARR} \quad\Rightarrow\quad ARR = \frac{170 \times 10^6}{2 \cdot 30 \times 10^3} = 2833.\overline{3}$$

No es entero. Opciones:

| ARR | f_PWM real | Error |
|---|---|---|
| **2833** | **30 002.5 Hz** | **+82 ppm** |
| 2834 | 29 991.9 Hz | −270 ppm |

Vamos con **ARR = 2833**. 82 ppm de error es invisible — el AS5600 tiene órdenes de magnitud más error en posición, el cristal del PCB tiene ±20 ppm de tolerancia.

### Por qué importa

Toda Fase 1 (PWM + ADC sincronizado + ISR FCS-MPC) cuelga de esta decisión. Si configuráramos edge-aligned, tendríamos que improvisar el muestreo del ADC y romperíamos el supuesto del modelo predictivo. Center-aligned no es "una optimización" — es estructural.

Y CMS = 01 vs Mode 3 no es trivial: si en el futuro quieres muestreo dual (una vez en pico, otra en valle, para promediar y reducir ruido), tendrías que ir a Mode 3 + dos triggers TRGO. Para Fase 1, Mode 1 con una muestra por periodo es suficiente.

---

## N1.4 — Del contador al pin: cómo el silicio genera PWM

### Panorama

Ya sabemos que el contador del TIM1 sube y baja como triángulo entre 0 y ARR. Ya sabemos que vamos a configurarlo a 30 kHz. La pregunta ahora:

> ¿Qué hace que el pin físico **PA8** conmute cuando el contador cruza el valor de **CCR1**?

La respuesta: hay una cadena de **4 bloques lógicos en silicio**, en serie, entre el contador y el pin. Cada bloque se controla con uno o dos bits de un registro. Visto de arriba:

```
                                                                             ┌─────────┐
contador (CNT) ──┐                                                           │ pin PA8 │
                 │                                                           └─────────┘
                 ▼                                                                ▲
            ┌─────────┐         ┌──────────┐         ┌────────────┐               │
            │COMPARADOR│ ───→  │POLARIDAD │ ───→   │HABILITACIÓN │ ──────────────┘
            └─────────┘  OCREF │ (CCxP)   │  tim_  │ (CCxE, MOE) │
                 ▲             └──────────┘  ocx   └────────────┘
                 │
            CCRx (referencia)
```

Cada nombre con flechas (`OCREF`, `tim_ocx`, etc.) es una **señal lógica interna al chip**. No la puedes ver con un osciloscopio. Solo la última (`pin PA8`) es física. Pero entender los nombres es importante porque la datasheet usa esos términos.

### Bloque 1: el comparador

El primer bloque toma `CNT` (valor actual del contador) y `CCRx` (valor que tú configuras) y los compara. El resultado es una señal lógica de 1 bit llamada **OCxREF** (Output Compare x REFerence).

La regla de comparación depende de un campo de configuración: **OCxM** (Output Compare x Mode), 4 bits en el registro `CCMRx`.

Para PWM tenemos dos opciones:

| OCxM | Modo | Regla (en up-counting) |
|---|---|---|
| `0110` | **PWM mode 1** | OCxREF = 1 mientras CNT < CCRx |
| `0111` | PWM mode 2 | OCxREF = 1 mientras CNT > CCRx |

Es decir, **PWM mode 1 y PWM mode 2 son el inverso uno del otro**. Veamos PWM mode 1 con un ejemplo de ARR=8, CCR1=4, modo center-aligned:

```
CNT:    0 1 2 3 4 5 6 7 8 7 6 5 4 3 2 1 0
        ↗     ↗   ↘     ↘
        ╱╲   ╱╲  ╱╲    ╱╲
       ╱  ╲ ╱  ╲╱  ╲  ╱  ╲

OCREF:  ┌───┐               ┌───┐
   _____│   │_______________│   │_____   ← high cuando CNT < 4

CCR1=4: ──────────────────────────────
```

El comparador es **puramente combinacional** — su salida cambia exactamente cuando CNT cruza CCRx, sin retraso. Esto es lo que hace que el PWM tenga resolución de 1 ciclo del reloj.

**Casos extremos**:
- Si `CCRx = 0`: OCREF nunca es alto. Duty cycle = 0%.
- Si `CCRx > ARR`: OCREF siempre es alto. Duty cycle = 100%.
- Si `CCRx = ARR/2`: duty cycle = 50%.

### Bloque 2: polaridad (CCxP)

OCREF es la "intención lógica" del PWM. Pero el pin físico que conecta al gate driver puede necesitar la señal **invertida** (por ejemplo, si el gate driver es activo-bajo).

El bit **CCxP** (CC x Polarity) del registro `CCER` controla esto:

| CCxP | Efecto sobre tim_ocx (la señal que va al pin) |
|---|---|
| 0 | tim_ocx = OCREF (sin inversión, "active high") |
| 1 | tim_ocx = NOT OCREF (invertido, "active low") |

En nuestro caso, los gate drivers L6387 son **activos-altos**: aplicas un 1 lógico al pin y el MOSFET correspondiente conduce. Por lo tanto: **CCxP = 0**.

### Bloque 3: habilitación de la salida (CCxE + MOE)

Aquí está la trampa típica que hace que muchos firmwares "no generen PWM" aunque todo lo demás esté bien.

Para que la señal `tim_ocx` salga físicamente por el pin, **dos bits tienen que estar a 1 simultáneamente**:

| Bit | Registro | Significado |
|---|---|---|
| **CCxE** | CCER | "Enable" del canal x (output del comparador, single-ended) |
| **MOE** | BDTR | **Main Output Enable** — el "master switch" de TODO el TIM1 |

**MOE = 0 silencia todas las salidas del TIM1, sin importar lo demás.** Es como un breaker general. Esto existe por una razón importante de seguridad: si detectas un fallo (corriente excesiva, sobrevoltaje), un solo bit a 0 en MOE corta los 6 PWMs instantáneamente. Sin software intermedio, hardware-level fault response.

Para una operación normal, después de configurar todo lo demás:

```c
TIM1->BDTR |= TIM_BDTR_MOE;  // ESTA es la línea que "enciende" el PWM
```

Si la olvidas, el código corre, las CCRx se cargan, los pines están en AF, pero el oscilograma está plano. Bug clásico.

### Bloque 4: el pin GPIO

Por último, el pin físico (PA8) tiene que estar configurado en **Alternate Function** (no en GPIO normal), apuntando al **AF correcto** para TIM1.

Del DS12589 Tabla 13: para PA8/PA9/PA10 (los TIM1_CHx) la AF es **AF6**. Para los CHxN (PC13, PA12, PB15) también es AF6. Bonus: todos los pines TIM1 usan la misma AF6.

```c
// Para PA8 (CH1):
GPIOA->MODER &= ~GPIO_MODER_MODE8_Msk;
GPIOA->MODER |=  (0b10U << GPIO_MODER_MODE8_Pos);  // AF mode

GPIOA->AFR[1] &= ~GPIO_AFRH_AFSEL8_Msk;
GPIOA->AFR[1] |=  (6U << GPIO_AFRH_AFSEL8_Pos);    // AF6 = TIM1_CH1
```

(`AFR[1]` cubre los pines 8-15; `AFR[0]` cubre 0-7.)

### El "preload" — por qué hay shadow registers

Hay una sutileza adicional importante. Si modificas `CCR1` directamente mientras el timer está corriendo, podrías escribirlo justo cuando el contador está pasando por ese valor. Resultado: **glitch en el pin** (un pulso de ancho aleatorio).

Solución del hardware: **shadow registers**. Hay 2 copias de `CCRx`:

```
tu escritura →  CCRx (preload, visible)
                       │
                       │ (transfer SOLO en update event)
                       ▼
                CCRx (shadow, lo que el comparador usa)
```

Para activar este buffering:

| Bit | Registro | Significado |
|---|---|---|
| **OCxPE** | CCMRx | Output Compare x Preload Enable |
| **ARPE** | CR1 | Auto-Reload Preload Enable (para ARR también) |

Con OCxPE = 1, tu `CCRx = nuevo_valor` se aplica al comparador **solo en el siguiente update event** (cuando CNT llega a 0 o a ARR). Resultado: las transiciones nunca producen glitches.

Para FCS-MPC esto es **esencial**: cada ISR calcula los nuevos duty cycles y los escribe en CCR1/CCR2/CCR3 sin preocuparse de timing — el hardware los aplica atomicamente al inicio del siguiente periodo.

**Importante**: para que los shadow registers se inicialicen correctamente antes de arrancar el timer, hay que forzar un update event manualmente con el bit UG (Update Generation) del registro EGR:

```c
TIM1->EGR = TIM_EGR_UG;  // dispara update event manual
```

Sin esto, el primer ciclo podría usar valores no inicializados.

### Resumen: la receta de PWM en TIM1

Para cada canal x ∈ {1, 2, 3}, hacer:

```c
// 1. Pin físico en AF6
GPIOx->MODER ... = 0b10  // AF mode
GPIOx->AFR[i] ... = 6    // AF6

// 2. CCMRx: OCxM = 0110 (PWM mode 1), OCxPE = 1 (preload)
TIM1->CCMRx |= (0b110 << OCxM_Pos) | OCxPE;

// 3. CCER: CCxP = 0 (active high), CCxE = 1 (enable)
TIM1->CCER |= CCxE;  // CCxP queda en 0 por reset

// 4. CCRx: duty cycle inicial (mitad de ARR = 50%)
TIM1->CCRx = ARR / 2;
```

Y a nivel global del timer:

```c
// 5. CR1: CMS = 01 (center-aligned mode 1), ARPE = 1
TIM1->CR1 |= (0b01 << CMS_Pos) | ARPE;

// 6. ARR: para 30 kHz @ 170 MHz
TIM1->ARR = 2833;

// 7. EGR: forzar update event para cargar shadow registers
TIM1->EGR = UG;

// 8. BDTR: MOE = 1 (master enable) ← ESTE ES EL "BIG RED BUTTON"
TIM1->BDTR |= MOE;

// 9. CR1: CEN = 1 (counter enable) — al final
TIM1->CR1 |= CEN;
```

Falta el **complementario CHxN** (el low-side) con su dead-time — eso es el bloque C, próxima nota.

### Por qué importa

PWM en silicio no es "una salida que conmuta". Son **4 bloques en serie**, cada uno con su configuración. El comparador genera la intención lógica; la polaridad la traduce al sentido eléctrico correcto; la habilitación la deja salir; el GPIO la dirige al pin físico.

Cuando algo no funcione (y va a pasar — quizá el oscilograma esté plano), tendrás que diagnosticar **en qué bloque se rompió la cadena**:

- ¿OCxM bien configurado? Si no, el comparador genera basura.
- ¿CCxP bien configurado? Si no, el motor gira al revés o el gate driver se queda apagado.
- ¿CCxE = 1? ¿MOE = 1? Si alguno está en 0, no sale nada.
- ¿GPIO en AF correcta? Si no, la señal interna existe pero no llega al pin.

Tener este modelo mental claro hace que el debug sea sistemático en lugar de adivinanza.

---

## N1.5 — Pines complementarios y dead-time: cómo el TIM1 evita el shoot-through

### Panorama

En un puente trifásico, cada fase del motor tiene **dos transistores en serie** entre Vbus y GND:

```
        Vbus (12 V)
          │
          ●
          │
       ┌──┴──┐
       │ HIGH│  ← high-side MOSFET (gate manejado por TIM1_CH1, pin PA8)
       │     │
       └──┬──┘
          │
          ●──────→ a la fase A del motor
          │
       ┌──┴──┐
       │ LOW │  ← low-side MOSFET (gate manejado por TIM1_CH1N, pin PC13)
       │     │
       └──┬──┘
          │
          ●
          │
         GND
```

**Reglas críticas de operación**:

1. **Para conectar la fase a Vbus**: HIGH conduce, LOW está apagado.
2. **Para conectar la fase a GND**: HIGH apagado, LOW conduce.
3. **Nunca, jamás, ambos al mismo tiempo conduciendo.** Sería un cortocircuito directo entre Vbus y GND a través de los dos MOSFETs. Esto se llama **shoot-through** o **cross-conduction**.

Shoot-through destruye los MOSFETs en microsegundos. No hay fusible que lo detenga porque ocurre dentro del propio puente, antes de cualquier protección externa.

### El problema: los MOSFETs no conmutan instantáneamente

Si el firmware da la orden "apaga HIGH" y "prende LOW" exactamente al mismo tiempo, en el papel funciona. **En la realidad, no.** Los MOSFETs tienen:

- **Tiempo de turn-off (`t_off`)**: tarda ~50-200 ns en dejar de conducir después de quitar la señal del gate. Mientras tanto, está en zona lineal — conduciendo parcialmente.
- **Tiempo de turn-on (`t_on`)**: similar al apagarse, ~50-150 ns.
- **Propagation delay del gate driver L6387**: agrega otros ~50-100 ns.

Resultado: si en t=0 le dices "HIGH off, LOW on", durante ~100-300 ns **ambos están parcialmente conduciendo**. Pico de corriente bestial por el puente.

### La solución: dead-time

La idea es **introducir una pequeña pausa** en la que ambos están apagados, antes de prender el otro:

```
HIGH (PA8):  ▁▁▁████████████▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
                            ◄──►
                       dead-time (~500 ns)

LOW (PC13):  ████▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁████████████
                                ◄──►
                            dead-time (~500 ns)
```

Durante los ~500 ns de dead-time, **ambos MOSFETs están apagados** (los gates a 0). La corriente del motor sigue fluyendo a través de los diodos del cuerpo de los MOSFETs (free-wheeling), pero ya no hay riesgo de shoot-through.

### Cómo el TIM1 hace esto en hardware

Aquí viene la parte elegante del periférico. Tú **no** generas las dos señales (HIGH y LOW) manualmente. Solo configuras **una** señal lógica (OCREF, lo que vimos en N1.4), y el hardware deriva automáticamente las dos señales físicas:

```
                              ┌──────────────────────┐
                              │  DEAD-TIME GENERATOR │
       OCREF (de N1.4) ────→  │   (hardware, 10-bit) │ ────→ tim_ocx  (high-side, a PA8)
                              │                      │ ────→ tim_ocxn (low-side, a PC13)
                              │   delay = DTG × tDTS │
                              └──────────────────────┘
```

El bloque dead-time:
- Toma el flanco de subida de OCREF → lo retrasa por `DTG × tDTS` antes de aplicarlo a tim_ocx.
- Toma el flanco de bajada de OCREF → lo retrasa por `DTG × tDTS` antes de aplicarlo a tim_ocxn.

Resultado (asumiendo CCxP = 0, CCxNP = 0):
- `tim_ocx` = OCREF con su rising edge retrasado (high-side).
- `tim_ocxn` = NOT OCREF con SU rising edge retrasado (low-side).

Las dos salidas son **casi complementarias**, salvo por el pequeño hueco del dead-time donde ambas valen 0.

### Habilitación del complementario: CCxNE y CCxNP

Los bits del registro `CCER` para el canal complementario son simétricos a los del principal:

| Bit | Análogo de | Significado |
|---|---|---|
| **CCxNE** | CCxE | Habilita la salida complementaria al pin |
| **CCxNP** | CCxP | Polaridad del complementario (0 = active high, 1 = active low) |

**Para que el dead-time se active automáticamente**, tienen que estar a 1 **tanto CCxE como CCxNE simultáneamente**. Si solo CCxE = 1 y CCxNE = 0, el complementario simplemente no sale por el pin (queda apagado), pero `tim_ocx` mantiene su forma original sin dead-time inserted — porque no hay riesgo de shoot-through sin complementario.

### El encoding no-lineal del DTG[7:0]

Y aquí es donde el TIM1 se pone "raro". El campo DTG en el registro BDTR es **8 bits**, pero permite expresar dead-times de **0 hasta 126 μs**. Eso son 18 bits de rango si fuera lineal. ¿Cómo cabe en 8 bits?

Respuesta: **encoding no-lineal** con 4 rangos. Mira el bit-pattern de DTG[7:5]:

| DTG[7:5] | Fórmula del dead-time | Resolución | Rango |
|---|---|---|---|
| `0xx` | DT = DTG[7:0] × t_DTS | 1 × t_DTS | 0 a 127 × t_DTS |
| `10x` | DT = (64 + DTG[5:0]) × 2 × t_DTS | 2 × t_DTS | 128 a 254 × t_DTS |
| `110` | DT = (32 + DTG[4:0]) × 8 × t_DTS | 8 × t_DTS | 256 a 504 × t_DTS |
| `111` | DT = (32 + DTG[4:0]) × 16 × t_DTS | 16 × t_DTS | 512 a 1008 × t_DTS |

Es como un **punto flotante de muy baja resolución**: el bit alto codifica el "exponente" (la resolución temporal) y los bits bajos el "mantissa" (el múltiplo). El rango cubre desde ~6 ns hasta ~126 μs.

Y `t_DTS` (Dead-Time Sampling clock) es un sub-reloj derivado del reloj principal del timer, controlado por **CKD[1:0]** en CR1:

| CKD | t_DTS |
|---|---|
| 00 | t_CK_INT (1 / 170 MHz = 5.88 ns) |
| 01 | 2 × t_CK_INT = 11.76 ns |
| 10 | 4 × t_CK_INT = 23.53 ns |
| 11 | reservado |

### Cálculo para nuestro caso: ~500 ns con CKD=0

Con CKD=00, t_DTS = 5.88 ns. Para 500 ns:

$$DTG = \frac{500\,\text{ns}}{5.88\,\text{ns}} = 85.0$$

85 en binario = `01010101` = 0x55. **El bit 7 es 0**, entonces caemos en el primer rango (lineal). Verificación:

$$DT_{\text{real}} = 85 \times 5.88\,\text{ns} = 500.0\,\text{ns}$$

Exacto. Configuración final:

```c
TIM1->BDTR |= (0x55U << TIM_BDTR_DTG_Pos);  // dead-time = 500 ns
```

(CKD = 00 es el reset value, no hay que tocarlo).

### Sobre dead-time asimétrico (DTAE)

El TIM1 puede tener dead-time **diferente en flanco de subida vs flanco de bajada** (set DTAE = 1 en DTR2; flanco descendente lo controla DTGF en DTR2 en lugar de DTG en BDTR). Esto sirve si los MOSFETs son asimétricos en sus tiempos de switching. Para Fase 1 lo dejamos simétrico (DTAE = 0, default).

### La "trampa" del OISx — qué pasa cuando MOE = 0

Aquí está una sutileza importante. Cuando `MOE = 0` (porque tú lo apagas, o porque un break input se disparó), las salidas no quedan en alta impedancia: van a un **estado fijo configurable** llamado **OISx / OISxN** (Output Idle State).

Bits del registro CR2:
- `OISx` (Output Idle State para tim_ocx): el valor que tendrá el high-side cuando MOE=0.
- `OISxN`: idem para el low-side.

**Para nuestro caso (gate drivers active-high)**: `OISx = 0, OISxN = 0`. Ambos transistores apagados → motor "free-wheeling" → seguro.

**Si pusieras OISxN = 1**: cuando MOE=0, el low-side conduce. Los 3 low-sides conducirían → corto-circuito a GND de las 3 fases → freno regenerativo brutal. Útil en algunas aplicaciones (parada de emergencia con freno), pero **no es lo que queremos por default**.

Lección: nunca dejes los OISx en valores no inicializados. El reset value es 0 (seguro), pero documenta el porqué.

### Receta completa: los 6 PWM complementarios

Agregando al código de N1.4:

```c
// Para cada canal x ∈ {1, 2, 3}:

// (lo anterior de N1.4: CCMRx, CCER.CCxE, CCRx, GPIOx en AF6, etc.)

// + NUEVO: habilitar el complementario y su polaridad
TIM1->CCER |= TIM_CCER_CC1NE;  // CH1N enable (PC13 active high)
TIM1->CCER |= TIM_CCER_CC2NE;  // CH2N enable (PA12)
TIM1->CCER |= TIM_CCER_CC3NE;  // CH3N enable (PB15)
// CC1NP/CC2NP/CC3NP quedan en 0 por reset (active high) — perfecto para L6387

// + NUEVO: los pines de CHxN también en AF6
// PC13 (CH1N), PA12 (CH2N), PB15 (CH3N) → MODER=10, AFR=AF6

// + NUEVO: dead-time en BDTR (incluye también MOE al final)
TIM1->BDTR = (0x55U << TIM_BDTR_DTG_Pos)  // 500 ns dead-time
           | TIM_BDTR_MOE;                  // Master Output Enable

// CR2: idle states explícitos a 0 (aunque es el reset value, dejarlo explícito)
TIM1->CR2 &= ~(TIM_CR2_OIS1 | TIM_CR2_OIS1N
              | TIM_CR2_OIS2 | TIM_CR2_OIS2N
              | TIM_CR2_OIS3 | TIM_CR2_OIS3N);
```

### Por qué importa

El dead-time es **invisible en el modelo de simulación** que ya tienes en MATLAB. Tu `bldc_plant_step.m` no tiene dead-time — los voltajes que aplicas a las fases son ideales. Pero en el banco real:

1. Cada ciclo PWM pierde ~500 ns de los 33 μs de periodo a dead-time. Si CCRx codifica un duty cycle de 50%, el voltaje real promedio aplicado al motor es **ligeramente menor** que 50% — porque durante el dead-time la fase queda libre.

2. La diferencia entre el voltaje ordenado y el real (**dead-time error**) **es periódica con el ángulo eléctrico** del motor. Genera armónicos no triviales en la corriente.

3. **Esto es exactamente lo que el ADALINE va a aprender y compensar.** El ADALINE no solo absorbe el shape de la BEMF — también captura los efectos del dead-time porque ambos aparecen como armónicos periódicos en la corriente.

Es decir, el dead-time no es solo una restricción de seguridad: **es una fuente de perturbación que la tesis va a atacar**. Esto refuerza el reencuadre conceptual del capítulo experimental (memoria del proyecto): aunque el motor 12N14P tiene BEMF cuasi-senoidal y debilita el contraste "TRAP vs ADALINE", el efecto de dead-time + cogging torque sigue siendo terreno fértil para el ADALINE.

---

## N1.6 — TRGO: el cordón umbilical entre el TIM1 y el ADC

### Panorama

Llegamos al punto **más importante** del estudio del TIM1: cómo le dice al ADC "muestrea AHORA, exactamente en el pico de la onda triangular del contador".

Sin esta sincronización, los demás 5 bloques (counter modes, comparator, polaridad, habilitación, dead-time) no servirían — porque el ADC estaría leyendo la corriente en momentos aleatorios del rizado, no en el promedio.

La pieza que falta se llama **TRGO** (Trigger Output): una señal interna del chip que sale del TIM1 y puede llegar a otros periféricos.

```
        TIM1                                ADC1
   ┌──────────┐                       ┌──────────┐
   │          │                       │          │
   │  CNT     │       TRGO            │  EXTSEL  │
   │  ↗╲   ───┼──────────────────────→│          │
   │  ╱  ╲    │       (señal interna  │          │
   │  ╱    ╲  │        del chip,      │  Start   │
   │  ↘╱    ╲ │        no es un pin   │  conv.   │
   │          │        físico)        │          │
   └──────────┘                       └──────────┘
```

TRGO no es un pin físico. **Es un cable de silicio interno** que une periféricos. No lo puedes ver con un osciloscopio. Pero está ahí, conectando el TIM1 con el ADC, con otros timers, con el DAC, con el HRTIM, etc.

### Cómo el TIM1 decide qué evento mandar por TRGO

El TIM1 tiene varios eventos internos que podría mandar al exterior: cuando el contador se reinicia, cuando alcanza el pico, cuando uno de los CCRx hace match, etc. **Tú eliges cuál.**

El campo **MMS[2:0]** del registro `CR2` (Master Mode Selection) selecciona qué evento del TIM1 se mapea a TRGO. RM0440 §29.6.2 enumera las opciones:

| MMS | Evento que sale por TRGO |
|---|---|
| `000` | Reset (cuando UG=1 en EGR) |
| `001` | Enable (cuando se enciende el timer) |
| **`010`** | **Update event** ← lo que vamos a usar |
| `011` | Compare pulse (cuando se setea CCxIF) |
| `100` | OC1REF (la señal del canal 1) |
| `101` | OC2REF |
| `110` | OC3REF |
| `111` | OC4REF |

(En realidad MMS es de 4 bits — MMS[3] vive en otro bit del CR2 para opciones extendidas, pero para PWM clásico nos basta con MMS[2:0].)

### Por qué MMS = 0010 (Update event)

En modo center-aligned, el **update event** ocurre cuando el contador llega a:

- Su pico (CNT = ARR, overflow), **y/o**
- Su valle (CNT = 0, underflow).

```
contador
ARR        ╱╲      ╱╲      ╱╲
          ╱  ╲    ╱  ╲    ╱  ╲
         ╱    ╲  ╱    ╲  ╱    ╲
        ╱      ╲╱      ╲╱      ╲
  0 ───╯       │       │       │
              ▼       ▼       ▼
       update event en overflow y underflow
       (2 per PWM period si RCR=0)
```

**¿Por qué esto es el momento ideal de muestreo?**

En center-aligned, los flancos de subida y bajada del PWM están **simétricos respecto al pico/valle del contador**. Eso significa:

```
PWM (PA8):       ┌────┐         ┌────┐
              ___│    │_________│    │___
                 ◄────►         ◄────►

contador      ╱╲      ╱╲      ╱╲
             ╱  ╲    ╱  ╲    ╱  ╲       ← pico y valle: centros del pulso PWM Y del hueco
            ╱    ╲  ╱    ╲  ╱    ╲
       ────╱      ╲╱      ╲╱      ╲

corriente   ╱╲       ╱╲      ╱╲           ← rizado triangular
i_ripple   ╱  ╲     ╱  ╲    ╱  ╲           pico/valle de la corriente
          ╱    ╲   ╱    ╲  ╱    ╲          coinciden con pico/valle del contador
         ╱      ╲ ╱      ╲╱      ╲
         ✱       ✱       ✱       ✱       ← muestrear en ✱ = valor PROMEDIO

         └ ADC dispara aquí (cuando TRGO se levanta)
```

Por la simetría matemática, **muestrear exactamente en el pico o valle de la onda triangular del contador da el valor promedio de la corriente durante ese intervalo**. Cualquier otro momento daría la corriente con un offset por el rizado.

Esto es **la razón estructural por la que usamos center-aligned + TRGO en update event**. No es un detalle de implementación — es el fundamento de la medición limpia.

### El otro lado: cómo el ADC escucha

El ADC tiene su propio campo de configuración para escoger qué trigger usar:

- **EXTSEL[4:0]** (para conversiones regulares) en el registro `ADC_CFGR`.
- **JEXTSEL[4:0]** (para conversiones inyectadas) en el registro `ADC_JSQR`.

Cada chip tiene una tabla en su Reference Manual que mapea valores de EXTSEL/JEXTSEL a qué trigger interno escoger. Para el STM32G4 (RM0440 Tabla 162, ADC1/ADC2):

| EXTSEL (decimal) | Trigger fuente |
|---|---|
| 0 | TIM1_CC1 |
| 1 | TIM1_CC2 |
| 2 | TIM1_CC3 |
| 3 | TIM1_CC4 |
| **9** | **TIM1_TRGO** ← lo que queremos |
| 10 | TIM1_TRGO2 |
| ... | ... |

Y para activar el trigger (no solo seleccionarlo), está **EXTEN[1:0]**:

| EXTEN | Comportamiento |
|---|---|
| 00 | Trigger deshabilitado (modo software, llamas `ADSTART` manualmente) |
| 01 | Trigger en rising edge |
| 10 | Trigger en falling edge |
| 11 | Trigger en ambos |

Para nuestro caso: `EXTSEL = 9, EXTEN = 01`.

### El "Repetition Counter": cuántas muestras por periodo

Hay una decisión importante: ¿queremos **1 muestra por periodo PWM** o **2** (una en el pico, otra en el valle)?

Con MMS=0010 directo y RCR=0 (default), el update event ocurre en **AMBOS** picos y valles → 2 triggers por periodo. La ISR del ADC correría a 60 kHz en lugar de 30 kHz.

El **Repetition Counter (RCR)** divide esa frecuencia. RCR es un registro de 8 bits en el TIM1, y funciona así:

| RCR | Update event cada N over/underflows |
|---|---|
| 0 | cada 1 (= 2 per PWM period en center-aligned) |
| 1 | cada 2 (= 1 per PWM period) |
| 2 | cada 3 (= 1.5 per PWM period — raro) |
| ... | ... |

**Para Fase 1**: RCR = 1. Una muestra por periodo PWM, en el valle (o en el pico — el comportamiento exacto depende del modo center-aligned 1/2/3; lo confirmaremos empíricamente).

### Configuración total en código

```c
// TIM1->CR2: configurar MMS = 0010 (update event como TRGO)
TIM1->CR2 &= ~TIM_CR2_MMS_Msk;
TIM1->CR2 |=  (0b010U << TIM_CR2_MMS_Pos);

// TIM1->RCR: una muestra por periodo PWM
TIM1->RCR = 1U;

// Más adelante, en la config del ADC:
ADC1->CFGR &= ~(ADC_CFGR_EXTSEL_Msk | ADC_CFGR_EXTEN_Msk);
ADC1->CFGR |= (9U << ADC_CFGR_EXTSEL_Pos)    // 9 = TIM1_TRGO
            | (0b01U << ADC_CFGR_EXTEN_Pos); // 01 = rising edge
```

### El truco avanzado: OC4REF como TRGO programable

Una capacidad interesante del TIM1: el canal 4 tiene la misma estructura que CH1/CH2/CH3, pero su salida `OC4REF` **puede mandarse por TRGO sin tener un pin físico** (PA11 también es CAN_RX en nuestra placa, así que CH4 no se usa como PWM).

Esto te permite **disparar al ADC en un momento arbitrario del ciclo PWM**, no necesariamente en el pico/valle:

```c
TIM1->CCR4 = 2500;          // dispara el ADC cuando el contador llega a 2500
TIM1->CCMR2 |= (0b110 << TIM_CCMR2_OC4M_Pos);  // OC4REF = PWM mode 1
TIM1->CR2 |= (0b111U << TIM_CR2_MMS_Pos);  // MMS = 0111 = OC4REF como TRGO
```

Por qué es útil:
- Permite saltar la zona de **switching noise** justo después de cada flanco PWM (~1 μs de transitorio EMI).
- Permite muestrear ligeramente **antes** del pico, dejando margen para que el ADC complete antes de la próxima conmutación.
- Permite **calibrar empíricamente** la mejor ventana de muestreo.

**Para Fase 1 no lo usamos** (mantenemos MMS=0010, update event), pero es bueno saber que existe — quizás en Fase 4 o 5, cuando optimicemos, valga la pena explorar.

### Por qué importa

Esta es **la pieza que une el control con la medición**. Sin TRGO bien configurado:

- Si muestrearas desde el `main()` cuando te diera la gana → corriente con rizado aleatorio → modelo predictivo no concuerda con la realidad → FCS-MPC oscila y eventualmente falla.

- Si el ADC corriera a una frecuencia distinta de la PWM → aliasing, batidos, comportamiento errático que parecería "ruido" pero es totalmente determinista.

Con TRGO + EXTSEL bien configurados:

- El ADC arranca exactamente en el pico del contador (con jitter < 1 ciclo de reloj = ~6 ns).
- La corriente medida es el promedio real del intervalo.
- La ISR del ADC se ejecuta a cadencia conocida y predecible (30 kHz exactos).

**Esa precisión de timing es la condición sine qua non del control predictivo en hardware.**

---

## N1.7 — Break inputs: la protección de hardware contra el desastre

### Panorama

Imagínate este escenario: el motor está girando, FCS-MPC controla con duty cycles razonables. De pronto, **algo falla**:

- El cable de un encoder se desconecta y el firmware pierde la posición → calcula corrientes erróneas → ordena duty cycles enormes → corriente de 50 A por una fase.
- Un transistor del puente entra en fallo (semi-cortocircuitado) → shoot-through parcial → corriente de cortocircuito.
- El rotor se bloquea físicamente (algo cae sobre el eje) → toda la corriente del controlador se concentra en una fase parada.

En cualquiera de estos casos, en menos de 1 milisegundo los MOSFETs llegan a su temperatura crítica y se destruyen. Las protecciones por software son demasiado lentas para evitar esto: si la ISR corre a 30 kHz, hay un retraso mínimo de 33 μs entre el evento y la respuesta, y eso es **antes** de que el código ejecute el chequeo.

La solución: **protección hardware con respuesta en nanosegundos**. Esto es lo que hacen los **break inputs**.

### Qué es un break input

Un break input es **una señal que, cuando se activa, fuerza al TIM1 a poner MOE = 0 en hardware**, sin pasar por software. Las 6 salidas PWM se apagan en ~50 ns (1-2 ciclos de APB clock).

```
┌──────────────────────┐
│   Fuente de fallo    │  ← OPAMP del shunt detecta i > 5A
│ (comparador interno  │     COMP interno: salida high si OPAMP > V_ref
│  o pin externo)      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Break circuitry     │  ← integrado en el TIM1
│  (silicio del TIM1)  │
└──────────┬───────────┘
           │
           ▼
       MOE = 0     ← apaga TODO el PWM (las 6 salidas), ~50 ns total
           │
           ▼
       Salidas van a estado OISx/OISxN (configurable)
```

### BKIN vs BKIN2: dos canales con prioridades distintas

El TIM1 tiene **dos break inputs separados**:

| | BKIN (Break) | BKIN2 (Break2) |
|---|---|---|
| Fuentes | Pin externo, comparadores internos, **system errors** (clock failure, ECC) | Pin externo, comparadores internos |
| Estado al que fuerza las salidas | **Configurable** vía OISx/OISxN (puede ser "activo" o "inactivo") | Siempre **inactivo** |
| Uso típico | Fallo crítico de sistema (mantener freno aplicado, por ejemplo) | Fallo de hardware (apagar todo) |

Para nuestro caso (un fallo = todos los transistores apagados = motor en free-wheel), **BKIN2 sería el más apropiado**. Pero ambos funcionan; la diferencia es flexibilidad.

### Fuentes posibles de break

RM0440 Figura 343 muestra que hay múltiples fuentes ORed antes del input del TIM1:

1. **Pin externo BKIN/BKIN2** (con polaridad y filtro digital configurable). Útil si el board tiene un pin de "fault" que viene de algún comparador externo. **En la B-G431B-ESC1: NO HAY pin BKIN externo expuesto** (UM2516 Tabla 4 no lo lista).

2. **Comparadores internos COMP1, COMP2, COMP3, COMP4**. Estos son comparadores analógicos integrados en el chip. Pueden conectarse internamente a las salidas de los OPAMPs (que amplifican los shunts) y comparar contra un V_ref. Si |i| > threshold, COMP saca high → activa el break.

3. **System break sources**: errores del sistema como clock security system (CSS) detectando fallo del HSE, parity/ECC errors en SRAM, output del MPU. Útil para apagar el PWM si el chip mismo está corrupto.

4. **Software break** (bit `BG` o `B2G` del EGR): permite disparar break desde código. Útil para auto-test del circuito de protección.

### La cadena ideal para nuestro caso: OPAMP → COMP → BKIN

```
                                                COMP1 internal
shunt 1   ─→   OPAMP1   ─→   ADC (current measurement)
              (output also routed to:)
                ↓
              COMP1 + ─→  +───┐
                              │  COMP1 output (1-bit) ─→ TIM1 BKIN
              COMP1 - ─→  V_ref
                              │
                              │  Si OPAMP_out > V_ref, COMP_out = 1 → break!
                              │
```

Es decir, **la corriente de cada fase pasa por el OPAMP (necesario para el ADC), y el mismo OPAMP también alimenta un comparador interno** que dispara el break si excede el umbral.

Esto es **lo elegante del STM32G4 para motor control**: los OPAMPs, comparators y TIM1 están diseñados para interconectarse sin tocar pines externos. Toda la cadena de protección vive dentro del chip.

### Bits relevantes del BDTR (registro de break)

```
TIMx_BDTR (32 bits):
[31:24]: ... |       MOE       |  AOE  |  BKP  |  BKE  | ...
[23:16]: ...  BK2P  |  BK2E   |  BKF[3:0]  |  BK2F[3:0]  ...
[15:8]:  ...
[7:0]:   ...                  DTG[7:0]  (ya visto en N1.5)
```

| Bit | Significado |
|---|---|
| **BKE** | Break enable (1 = activado) |
| **BKP** | Break polarity (0 = active low, 1 = active high) |
| **BK2E, BK2P** | Idem para BKIN2 |
| **BKF[3:0]** | Digital filter para BKIN — número de ciclos de muestreo coincidentes para validar el evento (rechaza glitches) |
| **AOE** | **Automatic Output Enable** — comportamiento al irse el break |
| **MOE** | Master Output Enable (ya visto en N1.4 / N1.5) |

### El bit AOE: ¿el PWM se reactiva solo después de un break?

Cuando un break se dispara, MOE → 0 automáticamente. Cuando el break **se va** (la señal vuelve a inactiva), hay dos comportamientos posibles:

| AOE | Al desactivarse el break |
|---|---|
| 0 | MOE queda en 0. **El firmware debe reactivar manualmente** (set MOE = 1). |
| 1 | MOE vuelve a 1 automáticamente en el próximo update event. |

**Recomendación general**: AOE = 0. Razón:
- Si AOE = 1 y el break sigue parpadeando (toggle rápido por una corriente cerca del threshold), el PWM se reactivaría cada vez que el break baja, generando una avalancha de events de over-current.
- AOE = 0 fuerza al firmware a reconocer el evento, hacer un diagnóstico, y decidir cuándo reactivar. Más seguro.

### El detalle de timing: write/read del MOE

Hay una sutileza importante que RM0440 menciona explícitamente:

> "If MOE is set to 1 whereas it was low, a delay must be inserted (dummy instruction) before reading it correctly. This is because the write acts on the asynchronous signal whereas the read reflects the synchronous signal."

Es decir: si haces `TIM1->BDTR |= MOE` para reactivar, no puedes leer `BDTR.MOE` inmediatamente para verificar — hay un retraso de unos ciclos por la sincronización entre el dominio asíncrono (donde vive el bit que tú escribes) y el dominio síncrono del timer. Hay que esperar unos `NOP` o leer otro registro intermedio.

Esto solo importa si tu lógica de "reactivar después de un fault" lee el bit para confirmar.

### Para Fase 1: ¿implementar break inputs?

**Decisión: NO en la primera iteración.** Razones:

1. La B-G431B-ESC1 **no expone un pin BKIN externo**. Habría que usar comparadores internos (COMP1/2/3 → BKIN via las internal signals `tim_brk_cmpx`).
2. Configurar los comparadores internos correctamente requiere otro capítulo de estudio (RM0440 Cap 27 — COMP). No queremos meternos en eso hasta tener PWM + ADC funcionando.
3. **La protección software es suficiente en la ISR**: la ISR del ADC ya lee las 3 corrientes a 30 kHz. Si añade `if (abs(i) > 5A) TIM1->BDTR &= ~MOE`, la respuesta tarda como máximo 33 μs (un periodo PWM) — suficiente para evitar daño térmico (los MOSFETs aguantan picos de 40A durante ms).

**Decisión: SÍ documentar para futuras iteraciones.** En Fase 5 o cuando bumpeemos potencia (si lo hacemos con un motor más grande), implementar el camino OPAMP→COMP→BKIN será obligatorio.

### Por qué importa (aun si no lo implementamos ahora)

Tres razones para entenderlo aunque no lo usemos en Fase 1:

1. **Saber que existe te da una salida de emergencia**. Si en pruebas algo va mal y no quieres freír un MOSFET, agregar `TIM1->EGR = TIM_EGR_BG;` (software break) desde GDB es una "parada total" instantánea.

2. **OISx (de N1.5) cobra sentido**. Esos bits no son solo para "qué pasa cuando MOE=0 por software" — son críticos cuando un break automático los activa. Si OISxN = 1, un break automático *prende* el low-side de los 3 puentes. Ya cubrimos esto en N1.5 pero ahora entiendes el por qué real.

3. **El bit MOE en BDTR es el "big red button" de toda la cadena**. Es lo último que se configura (después de ARR, CCRx, CCxE, CCxNE, DTG, etc.) y lo primero que se apaga ante un fault. Tener este modelo mental hace que el flujo de inicialización tenga sentido lógico, no solo histórico.

---

