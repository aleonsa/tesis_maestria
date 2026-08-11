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
- [N1.8 — Por qué 50 kHz: el balance de 6 restricciones](#n18--por-qué-50-khz-el-balance-de-6-restricciones)
- [N1.9 — La trampa del Alternate Function: AF no es uniforme por periférico](#n19--la-trampa-del-alternate-function-af-no-es-uniforme-por-periférico)
- [N1.10 — Cómo se mide la corriente del motor: shunt → OPAMP → ADC](#n110--cómo-se-mide-la-corriente-del-motor-shunt--opamp--adc)
- [N1.11 — OPAMPs internos del STM32G4: modos, PGA, calibración](#n111--opamps-internos-del-stm32g4-modos-pga-calibración)
- [N1.12 — El ADC del STM32G4: cómo se convierte voltaje en número](#n112--el-adc-del-stm32g4-cómo-se-convierte-voltaje-en-número)
- [N1.13 — Bring-up del ADC: tres trampas que encontramos](#n113--bring-up-del-adc-tres-trampas-que-encontramos)
- [N1.14 — ISR JEOS: el latido del lazo de control](#n114--isr-jeos-el-latido-del-lazo-de-control)
- [N1.15 — Race condition latente: regular vs injected en el mismo ADC](#n115--race-condition-latente-regular-vs-injected-en-el-mismo-adc)
- [N1.16 — Bring-up del AS5600 por I²C1: el sentido de posición](#n116--bring-up-del-as5600-por-i²c1-el-sentido-de-posición)
- [N1.17 — Bisección: cómo se depura un síntoma que mezcla cuatro cosas](#n117--bisección-cómo-se-depura-un-síntoma-que-mezcla-cuatro-cosas)

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
| **Lectura de posición** | I²C1 | **PB8 (SCL), PB7 (SDA)** | ~6.6 kHz (limitado por AS5600) |
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
| 7 | AS5600 vía I²C1 con extrapolación de θ entre lecturas | ✅ Posición legible girando el motor a mano (2026-08-10). Extrapolación pendiente |

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
| PB6 | ~~I²C1_SCL~~ **libre** | pad A+/H1 de J8. ⚠ AF4 aquí NO es I²C1_SCL — ver [N1.16](#n116--bring-up-del-as5600-por-i²c1-el-sentido-de-posición) |
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

### Cálculo de ARR para 50 kHz

Con HCLK = 170 MHz y prescaler PSC = 0:

$$f_{PWM} = \frac{f_{clk}}{2 \cdot ARR} \quad\Rightarrow\quad ARR = \frac{170 \times 10^6}{2 \cdot 50 \times 10^3} = 1700$$

**Exacto, sin redondeo.** Una de las razones para elegir 50 kHz versus 30 kHz, que daba ARR=2833.3 con +82 ppm de error.

| f_PWM | ARR | Error de cuantización |
|---|---|---|
| 30 kHz | 2833 | +82 ppm |
| **50 kHz** | **1700** | **0 ppm** |
| 100 kHz | 850 | 0 ppm (pero inviable computacionalmente) |

La decisión **50 kHz vs 30 kHz** depende de tradeoffs entre rizado de corriente, presupuesto computacional, AS5600, etc. → ver nota dedicada [N1.8](#n18--por-qué-50-khz-el-balance-de-6-restricciones).

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

## N1.8 — Por qué 50 kHz: el balance de 6 restricciones

### Panorama

`f_PWM` no es un parámetro libre. Es una decisión que pelea entre **6 restricciones simultáneas**, algunas que empujan hacia arriba y otras hacia abajo. La pregunta no es "¿cuál es la frecuencia ideal?" sino "¿cuál es el mejor compromiso?".

La decisión original del proyecto era **30 kHz**, documentada en la memoria con el argumento del "rizado de corriente alto con L baja". Pero el análisis cuantitativo de los 6 factores (sesión 7, 2026-05-20) reveló que ese argumento era subóptimo: a 30 kHz el rizado **sigue siendo del 92% relativo**, y subir a 50 kHz lo baja a 56% sin sacrificar nada crítico. Decisión revisada: **50 kHz**.

### Los 6 factores

#### 1. Rizado de corriente (empuja hacia ARRIBA)

Cuando un PWM aplica voltaje de bus durante T_on y 0 durante T_off, la corriente en la inductancia oscila con amplitud peak-to-peak:

$$\Delta i_{pp} \approx \frac{V_{bus} \cdot T_s}{L}$$

Con `L = 0.86 mH` (motor 2804 nominal) y `Vbus = 12 V`:

| f_PWM | T_s | Δi_pp | Δi / i_nominal (0.5 A) |
|---|---|---|---|
| 10 kHz | 100 μs | 1.40 A | 280% (desastre) |
| 20 kHz | 50 μs | 0.70 A | 140% (muy malo) |
| 30 kHz | 33 μs | 0.46 A | **92%** (borderline) |
| **50 kHz** | **20 μs** | **0.28 A** | **56%** (bueno) |
| 70 kHz | 14 μs | 0.20 A | 40% (muy bueno) |
| 100 kHz | 10 μs | 0.14 A | 28% (excelente) |

El motor 2804 tiene **L bajísima** por ser pequeño y de 14 polos. El motor de Coronado tiene L=2.05 mH (paper) → su rizado a 30 kHz es solo 39%, manejable. Para nosotros, 30 kHz es **estructuralmente insuficiente** porque el rizado domina la señal.

#### 2. Constante eléctrica del motor (empuja hacia ARRIBA, valor mínimo)

$$\tau_e = \frac{L}{R} = \frac{0.86 \times 10^{-3}}{2.3} = 374\,\mu s$$

Regla de diseño de control: `T_s < τ_e / 10 = 37 μs`. Eso es el límite mínimo de muestreo para que el modelo discretizado del sistema mantenga validez.

| f_PWM | Ts/τ_e | Modelo discreto válido |
|---|---|---|
| 20 kHz | 13% | No |
| 30 kHz | 9% | Justo en el borde |
| **50 kHz** | **5%** | Sí, con holgura |
| 100 kHz | 3% | Sí |

A 20 kHz el modelo predictivo del FCS-MPC pierde precisión. **Otra razón para no bajar de 30 kHz**, y para preferir 50 kHz.

#### 3. Presupuesto computacional del MCU (empuja hacia ABAJO, valor máximo)

Cada periodo PWM tenemos `T_s = 1/f_PWM` segundos para ejecutar la ISR completa: lectura de ADC, lectura de AS5600, transformaciones Clarke, predicción FCS-M2PC, escritura de PWM, ADALINE-LMS update.

| f_PWM | T_s | Ciclos del CPU @ 170 MHz | Margen |
|---|---|---|---|
| 30 kHz | 33 μs | 5667 ciclos | Muy cómodo |
| **50 kHz** | **20 μs** | **3400 ciclos** | **Apretado pero factible** |
| 70 kHz | 14 μs | 2380 ciclos | Crítico |
| 100 kHz | 10 μs | 1700 ciclos | Inviable con FCS-MPC clásico |

Para FCS-M2PC con horizonte 1 y 7 candidatos en STM32G4, el típico ronda **1500-2500 ciclos** (medido en literatura). A 50 kHz tenemos 3400 ciclos = ~700-1900 de holgura. **Suficiente, pero hay que vigilarlo en Semana 6** cuando midamos la ISR real.

#### 4. AS5600 (empuja hacia ABAJO)

El AS5600 actualiza internamente a **~7 kHz** (sample cada ~150 μs). Esto crea un problema: el algoritmo de control quiere posición fresca en cada periodo PWM, pero el sensor no la entrega tan rápido.

| f_PWM | Ciclos PWM entre updates frescos del AS5600 |
|---|---|
| 30 kHz | 4.3 |
| **50 kHz** | **7.1** |
| 70 kHz | 10.0 |
| 100 kHz | 14.3 |

Cuantos más ciclos sin update real, peor la extrapolación de θ entre lecturas. **Este factor empuja a quedarse bajo**, pero hasta 50 kHz es manejable (extrapolación lineal con ~7 ciclos sin update, ~1.05 ms de inferencia entre puntos, asumiendo ω constante en esa ventana).

Por encima de 70 kHz habría que cambiar a un encoder con ancho de banda mayor (AS5048A/AS5047P por SPI, ~6× más rápido).

#### 5. Pérdidas por conmutación (no restrictivo en nuestro rango)

Cada conmutación de MOSFET disipa una energía discreta `E_sw`. La potencia disipada por conmutaciones es `P_sw = E_sw × f_PWM × N_switches`. Crece linealmente con f_PWM.

Los MOSFETs STL180N6F7 en el L6387 están dimensionados para operar hasta ~100 kHz sin estrés térmico significativo. **No es restrictivo en nuestro rango (30-50 kHz).** Solo importaría si quisiéramos ir a 100+ kHz.

#### 6. Pérdida fraccionaria por dead-time (empuja hacia ABAJO)

Dead-time = 500 ns absoluto. Fracción del periodo PWM perdida a "tierra de nadie":

| f_PWM | T_s | DT / T_s |
|---|---|---|
| 30 kHz | 33 μs | 1.5% |
| **50 kHz** | **20 μs** | **2.5%** |
| 70 kHz | 14 μs | 3.6% |
| 100 kHz | 10 μs | 5.0% |

A 50 kHz perdemos 2.5% del periodo a dead-time. Esto se manifiesta como **distorsión periódica con el ángulo eléctrico** en la corriente — exactamente la perturbación que el ADALINE va a aprender. Argumento débil: subir a 50 kHz aumenta ligeramente la perturbación, pero también es **más trabajo útil para el ADALINE**.

### Tabla resumen

| Factor | 30 kHz | **50 kHz** | 70 kHz | Notas |
|---|---|---|---|---|
| Rizado relativo | 92% | **56%** | 40% | 50 kHz mejor |
| Modelo discreto válido | borderline | **OK** | OK | 50 kHz mejor |
| Presupuesto ISR | 5667 cy | **3400 cy** | 2380 cy | 30 kHz mejor, 50 cabe |
| AS5600 ciclos/update | 4.3 | **7.1** | 10.0 | 30 kHz mejor, 50 manejable |
| Pérdidas switching | bajas | **bajas** | medias | no restrictivo |
| DT como % de Ts | 1.5% | **2.5%** | 3.6% | 30 kHz mejor pero diferencia despreciable |

**Veredicto**: 50 kHz balancea mejor. Lo único en que 30 kHz gana es presupuesto computacional, donde 50 kHz "cabe" según literatura aunque sin tanto margen.

### Cálculo exacto del ARR

$$ARR = \frac{170 \times 10^6}{2 \times 50\,000} = 1700 \quad \text{(exacto, 0 ppm de error de cuantización)}$$

Bonus respecto a 30 kHz: 30 kHz pedía ARR=2833.33, redondeado a 2833 con error +82 ppm. **50 kHz cae justo en un entero.** Eliminamos un sub-bug potencial donde el muestreo cuasi-síncrono podría desincronizarse muy lentamente.

### Plan de validación en Semana 6

Cuando midamos la ISR real (Semana 6 según el plan de PROGRESO_HARDWARE.md), tendremos que verificar:

- **¿La ISR completa cabe en <15 μs (75% del Ts=20μs)?** Si no, hay margen para optimizar:
  - CORDIC para sin/cos en lugar de funciones de math.h.
  - Fixed-point Q1.15 para la predicción.
  - Reducir candidatos del FCS-M2PC (de 7 a 5).
- **¿La extrapolación del AS5600 mantiene θ_e con < 1° de error a velocidades nominales (~1500 rpm mecánicas, ~10500 rpm eléctricas con 7 pares de polos)?**

Si alguna de estas falla, hay 2 fallbacks documentados:
1. Bajar a 30 kHz como concesión.
2. Cambiar el AS5600 por AS5048A/AS5047P (SPI, ~6× ancho de banda).

### Por qué importa (meta-lección)

Esta nota es un caso clínico de **revisar decisiones antes de cementarlas en código**. La f_PWM=30 kHz estuvo en la memoria del proyecto durante varias sesiones con el argumento "ripple de corriente alto si Ts es grande" — argumento parcialmente correcto pero **subóptimo**, porque no se hizo el análisis cuantitativo.

Lección general: cuando una decisión técnica empieza a aparecer en código (constantes, comentarios, plan), es el último momento para preguntar "¿por qué este número y no otro?" — y exigir respuesta cuantitativa, no solo cualitativa.

---

## N1.9 — La trampa del Alternate Function: AF no es uniforme por periférico

### Panorama

Cuando configuras un pin en modo Alternate Function (AF), no solo eliges "este pin será controlado por un periférico". Eliges **cuál de hasta 16 funciones alternativas** ese pin específico ofrece. El número AF (0–15) **es propio del pin**, no del periférico.

La trampa: es **tentador asumir "todos los pines TIM1 son AF6"** porque ves que PA8, PA9, PA10 (TIM1_CH1/CH2/CH3) son AF6 y generalizas. Pero ST diseña la pin-mux **pin por pin**, priorizando las funciones más comunes en los números AF más bajos disponibles. **El mismo periférico puede caer en AF distintos según el pin.**

Si configuras un pin con el AF equivocado, **la silicio routea la salida a OTRA función** — no es un error de compilación, no hay warning, los registros se ven "correctos" (MODER=AF, AFR=6). Pero el pin físico está ejecutando algo distinto a lo que quieres.

### Analogía

Piensa en un edificio con 16 ascensores numerados 0–15. Cada planta (= pin) puede conectarse a cualquier ascensor, pero **qué destino llega a esa planta vía cada ascensor depende de la planta**:

- Planta PA8: ascensor 6 lleva a **TIM1_CH1**.
- Planta PA12: ascensor 6 lleva a **TIM1_CH2N**.
- Planta PB15: ascensor 6 lleva a **otra cosa**. TIM1_CH3N está en **ascensor 4**.
- Planta PC13: ascensor 6 lleva a **TIM8_CH4N**. TIM1_CH1N está en **ascensor 4**.

Tomar el ascensor 6 en todas las plantas te lleva a destinos diferentes. Asumir "ascensor 6 = TIM1 siempre" porque funcionó en algunas plantas es la causa del bug.

### Detalle técnico — el bug concreto en el banco

Cuando configuré los 6 pines TIM1 en `pwm.c` (sesión 7), asumí AF6 universal:

```c
gpio_set_af(GPIOA, 8U,  6U);   // PA8  CH1   → AF6 ✓
gpio_set_af(GPIOA, 9U,  6U);   // PA9  CH2   → AF6 ✓
gpio_set_af(GPIOA, 10U, 6U);   // PA10 CH3   → AF6 ✓
gpio_set_af(GPIOA, 12U, 6U);   // PA12 CH2N  → AF6 ✓
gpio_set_af(GPIOB, 15U, 6U);   // PB15 CH3N  → AF6 ❌ (debe ser AF4)
gpio_set_af(GPIOC, 13U, 6U);   // PC13 CH1N  → AF6 ❌ (debe ser AF4; AF6 es TIM8_CH4N)
```

DS12589 Tabla 13 — fila de PC13 (los 16 AFs):

| AF | Función en PC13 |
|---|---|
| AF0 | — |
| AF1 | — |
| AF2 | TIM1_BKIN |
| AF3 | — |
| **AF4** | **TIM1_CH1N** ← lo que necesitamos |
| AF5 | — |
| AF6 | TIM8_CH4N |
| ... | ... |

Y PB15:

| AF | Función en PB15 |
|---|---|
| AF1 | TIM15_CH2 |
| AF2 | TIM15_CH1N |
| AF3 | COMP3_OUT |
| **AF4** | **TIM1_CH3N** ← lo que necesitamos |
| AF5 | SPI2_MOSI/I2S2_SD |

### Síntomas del bug

Solo la fase B del puente funcionaba (sus dos pines PA9+PA12 están ambos en GPIOA con AF6 correcto para TIM1). Las fases A y C:

- **PC13 con AF6 → estaba routeada a TIM8_CH4N**, un periférico que no estaba habilitado → output del pin queda en estado indefinido (efectivamente alto-impedancia).
- **PB15 con AF6 → ninguna función específica en esa AF**, output indefinido.

Sin el low-side recibiendo señal del TIM1, el L6387 nunca conmutaba el low-side MOSFET → **bootstrap cap del high-side nunca se cargaba** → high-side tampoco podía conmutar → output del puente flotante en ~8V (mitad de Vbus por simetría del body diode).

Tiempo total perdido diagnosticando: ~2 sesiones de pruebas asumiendo problema de hardware (L6387 dañado, fuente baja, dead-time mal, etc.) cuando era un **3 (tres) en lugar de un 4 en cuatro bits de un registro**.

### Por qué fue difícil detectarlo

El dump diagnóstico que escribí mostraba:

```
PB15 AFR  = 6  (expected 6 = TIM1)
PC13 AFR  = 6  (expected 6 = TIM1)
```

**Y ahí estaba mi error**: el "expected" lo escribí yo basándome en mi misma asunción incorrecta. El bug existía **tanto en el código que configuraba el AF como en el código que validaba la configuración**, así que el diagnóstico decía "todo OK" cuando en realidad estaba mostrando exactamente el bug. Estabas usando el mismo mapa erróneo para preguntar si estás perdido.

### Por qué importa

1. **El AF es siempre por pin, nunca por periférico.** Siempre, siempre verificar la Tabla 13 del datasheet del chip específico (DS12589 para STM32G431) **pin por pin**. Ningún atajo.

2. **Los dumps de validación pueden esconder bugs si el "expected" está sacado del mismo modelo mental que el código que generó el bug.** Para validar realmente: el "expected" debe venir de **una fuente independiente** (el datasheet directamente, otro desarrollador, una herramienta de referencia como STM32CubeMX).

3. **Para futuros pines de la placa B-G431B-ESC1**: cuando llegue I²C1 para el AS5600 (Semana 7), verificar AF de PB6/PB7 directamente del datasheet. **No asumir nada.**

   > **Epílogo (2026-08-10): esta recomendación NO se siguió, y volvió a pasar.** AF4 en PB6
   > no es I²C1_SCL; SCL vive en PB8. Costó una sesión entera. El comentario del código
   > llegó a citar «DS12589 Table 13» sin que nadie abriera esa tabla — el datasheet ni
   > estaba en el repo. Ver [N1.16](#n116--bring-up-del-as5600-por-i²c1-el-sentido-de-posición).

4. **Si hay manera de detectar este error sin scope**: imprimir `GPIOx->ODR` (output data register) para los pines TIM1 mientras el counter corre. Si el AF está mal, GPIO no controla el output → puede dar lecturas raras. Pero esto es indirecto. La validación real es **comparar AFR contra la tabla 13** del datasheet con el código en una pantalla y la datasheet en otra.

### Tabla maestra de AFs para TIM1 en STM32G431 (B-G431B-ESC1)

Persistir aquí para referencia futura:

| Pin | Canal TIM1 | **AF correcto** |
|---|---|---|
| PA8 | CH1 (high-side fase A) | AF6 |
| PA9 | CH2 (high-side fase B) | AF6 |
| PA10 | CH3 (high-side fase C) | AF6 |
| PA12 | CH2N (low-side fase B) | AF6 |
| **PC13** | **CH1N (low-side fase A)** | **AF4** |
| **PB15** | **CH3N (low-side fase C)** | **AF4** |

---

## N1.10 — Cómo se mide la corriente del motor: shunt → OPAMP → ADC

### Panorama

El algoritmo FCS-M2PC quiere saber **qué corriente fluye por cada fase del motor** para predecir qué va a hacer la siguiente PWM. Pero la corriente no es algo que un microcontrolador pueda "ver" directamente. Solo puede medir **voltajes**, y solo en sus pines ADC.

La cadena para convertir "corriente del motor" en "número en memoria" tiene 3 etapas:

```
1. Shunt    : convierte corriente en voltaje pequeño (mV)
2. OPAMP    : amplifica ese voltaje a algo medible (V)
3. ADC      : convierte ese voltaje a un número digital (entero 0..4095)
```

Cada etapa introduce sus propios compromisos (resolución, ruido, latencia). Esta nota explica cada una y por qué fueron necesarias.

### Analogía

Es como medir la velocidad del viento. No puedes "ver" el viento directamente. Pones algo que el viento empuje (un molino), mides cuánto se mueve esa cosa, y de ahí calculas la velocidad. El molino convierte "movimiento de aire" en "movimiento mecánico". Después necesitas algo que convierta "movimiento mecánico" en "número" (un encoder, un tachómetro). Cada conversión introduce una ligera distorsión.

En nuestro caso: shunt convierte corriente en voltaje; OPAMP "agranda" ese voltaje; ADC convierte voltaje en número.

### Etapa 1: el shunt

Un **shunt** es una resistencia de valor **muy bajo** (típicamente 1-50 mΩ) colocada **en serie** con el camino de la corriente. En el B-G431B-ESC1 hay 3 shunts, uno en el camino del low-side de cada fase.

```
       Vbus
        │
       MOSFET high-side
        │
     ───●────→ a la fase del motor
        │
       MOSFET low-side
        │
       SHUNT (Rsense ~10 mΩ)
        │
       GND
       ───
```

Cuando la corriente del motor fluye a través del shunt, la **ley de Ohm** dice:

$$V_{shunt} = i_{motor} \times R_{sense}$$

Para `R_sense = 10 mΩ = 0.01 Ω` y corriente de 1 A:

$$V_{shunt} = 1 \text{ A} \times 0.01 \,\Omega = 10 \text{ mV}$$

### El compromiso del valor de R_sense

¿Por qué un valor tan bajo (10 mΩ)? Tres razones:

**1. Mínima caída de tensión en el motor.** Si R_sense fuera 1 Ω, a 1 A perdería 1 V — el motor solo "vería" Vbus − 1 V. Con 10 mΩ, la pérdida es 10 mV: invisible.

**2. Mínima potencia disipada como calor.** P = i² × R:
- Con R = 10 mΩ y 2 A pico: P = 4 × 0.01 = **40 mW** (no necesita disipador)
- Con R = 100 mΩ y 2 A pico: P = 4 × 0.1 = **400 mW** (ya pide disipación cuidadosa)

**3. Respuesta rápida en frecuencia.** Un shunt es básicamente una resistencia pura, sin inductancia parásita (si está bien diseñado). Su ancho de banda llega a MHz, mucho más que lo que necesitamos (50 kHz × 10 = ~500 kHz como Nyquist práctico).

### El problema del shunt: las señales son MUY pequeñas

Con `R = 10 mΩ` y corriente de motor en rango [−2 A, +2 A], el voltaje del shunt está en rango **[−20 mV, +20 mV]**.

Comparado con el rango de entrada del ADC (0 V a 3.3 V = **3300 mV**), nuestra señal usa apenas **40 mV / 3300 mV ≈ 1.2%** del rango. Si simplemente conectáramos el shunt al ADC:
- Solo usaríamos ~50 de los 4096 niveles del ADC.
- El ruido térmico del propio ADC (~1 mV) dominaría sobre la señal de motor.
- Resolución efectiva: ~6 bits útiles en lugar de 12.

**Solución**: amplificar la señal antes del ADC. Eso lo hace el OPAMP.

### Etapa 2: el OPAMP

Un **OPAMP** (operational amplifier) en motor control sirve para:

1. **Amplificar** la señal del shunt por un factor de ganancia G (típicamente 20-50).
2. **Sumar un offset** para que la señal AC bipolar (±20 mV) quede centrada en el medio del rango del ADC (~1.65 V) en lugar de oscilar alrededor de 0.

Con ganancia G = 50 y offset = 1.65 V, la señal después del OPAMP es:

$$V_{out} = 1.65 + (50 \times V_{shunt}) = 1.65 + (50 \times i \times 0.01) = 1.65 + 0.5 \cdot i$$

Para i = +2 A: V_out = 1.65 + 1.0 = **2.65 V**
Para i = 0:    V_out = 1.65 V
Para i = −2 A: V_out = 1.65 − 1.0 = **0.65 V**

Ahora la señal cubre **2 V de los 3.3 V** del ADC. Usamos ~60% del rango → resolución efectiva ~11 bits útiles → buena.

### ¿OPAMP interno o externo?

El STM32G431 tiene **3 OPAMPs internos** (OPAMP1, OPAMP2, OPAMP3) — uno por cada shunt. Esto es **clave del diseño de la placa**: ST escogió este chip específicamente porque integra todo lo que un motor controller necesita.

Ventajas vs OPAMPs externos discretos:
- **Sin retardos de PCB**: la señal del OPAMP va directo al ADC por silicio (10 ns de latencia), no por traces.
- **Sin componentes extra**: la placa tiene solo R's de feedback discretas; el OPAMP está en el MCU.
- **Calibración fácil**: ganancia y offset configurables por registros.

En la B-G431B-ESC1, las **R's de feedback** que definen la ganancia ya están soldadas en el PCB. La ganancia exacta hay que leerla del esquemático MB1419 — eso lo haremos en N1.11.

### Etapa 3: el ADC

El **ADC** (Analog-to-Digital Converter) toma un voltaje continuo de entrada y lo convierte a un número entero. Características clave del ADC del STM32G4:

| Parámetro | Valor |
|---|---|
| **Resolución** | 12 bits = **4096 niveles** |
| Rango de entrada | 0 V a Vref+ ≈ 3.3 V |
| **Resolución por bit** | 3.3 V / 4096 ≈ **0.806 mV/bit** |
| Tiempo de conversión típico | 6.5 + 12.5 = 19 ciclos del ADC clock |
| ADC clock máximo | 60 MHz |
| **Tiempo total mínimo por conversión** | 19 / 60 MHz ≈ **316 ns** |

Después del OPAMP, cada 1 A de corriente se traduce a 500 mV. Con 0.806 mV/bit:

$$\text{Bits por amperio} = \frac{500 \text{ mV}}{0.806 \text{ mV/bit}} \approx 620 \text{ bits/A}$$

Es decir, 1 A de corriente cambia el valor del ADC en ~620 cuentas. Sobreabundante para el rango del FCS-M2PC.

### ¿Por qué dos ADCs en paralelo (dual simultaneous)?

El STM32G431 tiene **2 ADCs independientes** (ADC1 y ADC2). En motor control trifásico, queremos las 3 corrientes "al mismo tiempo" — pero un solo ADC solo puede muestrear un canal a la vez.

Solución: **dual regular simultaneous mode**.
- ADC1 muestrea canal X.
- ADC2 muestrea canal Y **en el mismo instante**.
- Las dos conversiones suceden en paralelo, terminan a la vez.

Plan de distribución (tentativo, ajustaremos en N1.12):
- **ADC1**: i_a, i_c, Vbus, temperatura.
- **ADC2**: i_b, (lo que necesitemos).

i_a y i_b se muestrean **simultáneamente** (uno en ADC1, otro en ADC2). i_c lo calculamos: **i_c = −(i_a + i_b)** (suma de las 3 fases = 0 en un sistema trifásico balanceado, propiedad de Clarke).

### Sincronización con el TIM1

El ADC no muestrea "cuando se le antoja" — recibe un **trigger externo** del TIM1, exactamente en el pico/valle del contador (que coincide con el centro del rizado triangular de la corriente, como vimos en N1.6).

La configuración:
- `TIM1->CR2.MMS = 010` → update event sale como TRGO. ✅ (ya hecho)
- `ADC1->CFGR.EXTSEL = 9` → trigger es TIM1_TRGO.
- `ADC1->CFGR.EXTEN = 01` → trigger en rising edge.

Cuando el contador del TIM1 alcance su pico (CNT=ARR) o valle (CNT=0), TRGO sube → ADC arranca conversión → 316 ns después, los 12 bits están listos.

### Latencia total de la cadena

¿Cuánto tarda desde que la corriente cambia hasta que el FCS-M2PC ve el número?

| Etapa | Latencia |
|---|---|
| Shunt (resistivo, casi instantáneo) | ~10 ns |
| OPAMP interno (3 MHz BW típico) | ~50 ns |
| Conversión ADC (sample + convert) | ~316 ns |
| Trigger TRGO → ADC start | ~20 ns |
| **TOTAL** | **~400 ns** |

Esto es **0.4 μs de un periodo PWM de 20 μs** — solo 2% de latencia. Despreciable para FCS-M2PC.

### Por qué importa

1. **Sin medición precisa, FCS-MPC predice basura.** El algoritmo asume que la corriente medida ahora es exactamente la real. Si hay error (offset, ganancia mal calibrada, ruido), el modelo predictivo diverge.

2. **El timing es estructural, no opcional.** Si muestreáramos sin sincronización con el PWM, el rizado triangular de la corriente añadiría ruido aleatorio. Con TRGO en el pico, leemos el valor promedio real.

3. **La cadena entera es analógica hasta el ADC.** Cualquier ruido EMI que entre por el cable del motor o la fuente puede acoplarse al shunt y propagarse al ADC. Diseño de PCB importante: tracks cortos, planos de GND, etc. (Esto ya lo hizo ST, pero hay que saberlo para diagnosticar si hay ruido.)

4. **La calibración del offset es crítica.** El OPAMP tiene un offset DC pequeño pero no cero (~5 mV típico). Al ADC eso es ~6 cuentas de offset. Si no lo restamos, el FCS-M2PC va a "ver" una corriente DC que no existe. **Por eso en Semana 6 calibraremos**: 1000 muestras con motor desconectado → promedio → restar a futuras medidas.

5. **Conexión con el resto del pipeline**: i_a e i_b medidos → Clarke transform (en N1.13 o cerca) → i_α, i_β → entran al cálculo de FCS-M2PC. Es el "primer eslabón" entre hardware y algoritmo.

---

## N1.11 — OPAMPs internos del STM32G4: modos, PGA, calibración

### Panorama

El STM32G431 tiene **3 amplificadores operacionales integrados al silicio** (OPAMP1, OPAMP2, OPAMP3). No son "extras" — son periféricos de primera clase como el TIM1 o el ADC, con sus propios registros y modos de operación.

Cada uno tiene:
- 4 entradas posibles para `VINP` (entrada no inversora), mux interno.
- 2 entradas posibles para `VINM` (entrada inversora), o feedback interno.
- 1 salida `VOUT` que puede ir a un pin físico **y/o** directo al ADC.
- Auto-calibración de offset.

En la B-G431B-ESC1, los 3 OPAMPs se usan para amplificar las señales de los 3 shunts de corriente. **Esta nota explica cómo los configuramos.**

### Analogía

Un OPAMP discreto típico tiene 8 patas y va sobre el PCB con resistores externos a su alrededor. Los OPAMPs del G4 son lo mismo pero **dentro del chip**: las patas se reemplazan por pines del MCU configurables, y muchas resistencias internas opcionales. Lo que afuera serían 5–10 componentes discretos, adentro son 2 bits en un registro.

### Los 3 modos de operación (RM0440 §25.3.5)

#### 1. Standalone mode

Más parecido a un OPAMP discreto: VINP, VINM y VOUT son todos pines externos del MCU. Las R's de ganancia van afuera, en el PCB. Útil cuando:
- Necesitas una topología no soportada por los modos internos (ej. integrador con cap).
- El ancho de banda interno no alcanza para tu aplicación.

```
                 STM32 (silicio)
                ┌──────────────┐
   VINP   ───→  │      +       │
                │       \      │
                │        ●───────→ VOUT (pin)
                │       /      │
   VINM   ───→  │      -       │
                └──────────────┘
   (gain definida por R's externas entre VOUT y VINM)
```

#### 2. Follower mode

Ganancia 1, buffer de impedancia. La salida sigue la entrada. Útil para muestrear señales de alta impedancia antes del ADC. **No nos sirve para shunts** porque no amplifica.

#### 3. PGA mode (lo que nosotros vamos a usar)

El OPAMP tiene una red de feedback **interna** con resistores que definen la ganancia. Configuras `PGA_GAIN[2:0]` en `OPAMPx_CSR` y eliges:

| PGA_GAIN | Ganancia no-inversora | Ganancia inversora |
|---|---|---|
| 000 | x2 | x-1 |
| 001 | x4 | x-3 |
| 010 | x8 | x-7 |
| 011 | x16 | x-15 |
| 100 | x32 | x-31 |
| 101 | x64 | x-63 |

**Para corriente bipolar AC**, la ganancia inversora con offset al medio de Vrefint es lo típico, pero acá la placa MB1419 usa una topología distinta (verificable en el esquemático) que da ganancia de **~16** típicamente.

#### Sub-modo: PGA con feedback externo

Hay un caso especial: el bit `PGA_GAIN[3]` permite **routear el tap de feedback hacia el pin VINM externo**. Esto significa:
- La ganancia base del PGA interno funciona, pero...
- Una resistencia externa entre VINM (pin físico) y GND modifica esa ganancia.

**Esto es lo que usa la B-G431B-ESC1**: ST diseñó el PCB con R's externas entre los pines `Curr_fdbk*_OPAmp-` y GND, que ajustan finamente la ganancia y permiten sumar un offset (necesario para AC bipolar).

### Bandwidth del OPAMP

DS12589 sección 6.3.x — Operational Amplifiers electrical characteristics:

| Parámetro | Valor típico | Comentario |
|---|---|---|
| GBW (Gain-Bandwidth product) | 7–13 MHz | Producto ganancia × ancho de banda |
| Slew rate (modo normal) | ~5 V/μs | Velocidad máxima de cambio de la salida |
| Slew rate (modo high-speed, OPAHSM=1) | ~25 V/μs | Más velocidad, más consumo |
| Input offset (sin calibrar) | ±5 mV | Reducible a ±3 mV con auto-calibración |

A ganancia 4 (que es lo que esperamos en la placa), ancho de banda efectivo = GBW/4 ≈ **3.25 MHz**. Para muestrear a 50 kHz necesitas que el OPAMP esté establecido en ~10× el período de muestreo = 200 ns. 3.25 MHz → tiempo de respuesta ~300 ns. Justo en el borde, pero suficiente.

Si quisiéramos más margen: `OPAHSM = 1` (high-speed mode) extiende GBW a ~20 MHz con costo en consumo (~mA extra). Por defecto lo dejamos en modo normal y vemos si hay problemas.

### Auto-calibración del offset

El OPAMP físico tiene un offset DC pequeño pero no cero (típicamente ±5 mV). En motor control esto es importante porque:
- Offset de 5 mV en el OPAMP → ×4 ganancia → 20 mV en la salida → 25 cuentas del ADC.
- 25 cuentas falsas de corriente cada ciclo → FCS-M2PC ve corriente DC fantasma → controlador injerta corriente real para "corregir" el fantasma → torque DC inducido.

El OPAMP tiene **auto-calibración de offset** que reduce esto a ±3 mV. Procedimiento (RM0440 §25.3.7):

```c
// 1. Habilitar OPAMP
OPAMP1->CSR |= OPAMP_CSR_OPAEN;

// 2. Iniciar calibración del par diferencial P
OPAMP1->CSR |= OPAMP_CSR_CALON;       // arranca calibración
OPAMP1->CSR &= ~OPAMP_CSR_CALSEL;     // CALSEL=01 = P pair
OPAMP1->CSR |= 0x1U << OPAMP_CSR_CALSEL_Pos;

// 3. Incrementar TRIMOFFSETP de 0 a 31 hasta que CALOUT flip
for (uint32_t i = 0; i < 32; i++) {
    OPAMP1->CSR = ... // set TRIMOFFSETP = i
    delay_ms(2);  // CALOUT tarda hasta 2 ms en estabilizar
    if ((OPAMP1->CSR & OPAMP_CSR_CALOUT) == 0) break;
}

// 4. Repetir para par N (CALSEL=11)
// 5. Setear USERTRIM=1 para usar los valores calibrados
```

**Decisión**: vamos a hacer esta calibración **en el bring-up inicial** (una sola vez, después de pwm_init), no en cada arranque del sistema. Los valores quedan en SRAM y se pierden con cada reset, pero a 2 ms × 32 iteraciones × 2 pares = ~128 ms de calibración solo al arranque. Aceptable.

Hay también una **calibración a nivel de aplicación**: con todo OPAMP y ADC corriendo, motor desconectado, capturar 1000 muestras, promediar, y guardar como `i_offset`. Eso lo cubrimos en N1.14 (Semana 6 del planning).

### Conexión OPAMP → ADC

El bit `OPAINTOEN` (Output Internal connection) del CSR permite **routear la salida del OPAMP directo al ADC**, sin pasar por el pin físico:

```
OPAINTOEN = 0:  VOUT → pin OPAMPx_VOUT físico → ADC ve el pin (si tiene canal ahí)
OPAINTOEN = 1:  VOUT → conexión interna al ADC + pin OPAMPx_VOUT también activo
```

**Ventajas de OPAINTOEN = 1**:
- Ruta más corta → menos ruido EMI.
- El pin OPAMPx_VOUT queda libre para otra cosa si se quiere.
- En el G4 está optimizado: el ADC tiene canales internos dedicados que reciben el OPAMP directo.

Para la B-G431B-ESC1: ST usa `OPAINTOEN = 1` típicamente, porque los pines de los VOUT (`OP1_OUT` = PA2, etc.) están routeados también al ADC. Confirmar al implementar.

### Mapeo de OPAMPs a canales ADC internos

El silicio del G4 conecta cada OPAMP a un canal específico del ADC cuando `OPAINTOEN = 1`:

| OPAMP | Canal ADC interno |
|---|---|
| OPAMP1 | ADC1 IN13 |
| OPAMP2 | ADC2 IN16 |
| OPAMP3 | ADC2 IN18 |

(Verificable en DS12589 Tabla 14 "OPAMP output to ADC channel mapping" — voy a confirmar al implementar.)

Implicaciones:
- OPAMP1 (fase A) solo es accesible desde ADC1.
- OPAMP2 y OPAMP3 (fases B y C) accesibles desde ADC2.
- Esto **fuerza** la distribución: ADC1 para fase A, ADC2 para fases B y C.

**No podemos elegir libremente** qué ADC muestrea cuál fase — está determinado por el silicio. La distribución óptima es:
- ADC1: i_a (OPAMP1) + Vbus (PA0) + temperatura (PB14).
- ADC2: i_b (OPAMP2) + i_c (OPAMP3).

Para dual simultaneous, ADC1 e i_a + ADC2 e i_b se muestrean al mismo tiempo. i_c se mide en el siguiente slot del ADC2 (o se calcula: i_c = −(i_a + i_b)).

### Registros principales: OPAMPx_CSR

Vista de un golpe (32 bits):

```
[31] LOCK          : write-protect del registro (no usar)
[30] PGA_GAIN[4]
[29] OPAHSM        : high-speed mode (0=normal)
[28:24] TRIMOFFSETP : auto-cal value P
[23:19] TRIMOFFSETN : auto-cal value N
[18] USERTRIM      : 1 = usa valores calibrados
[17:14] PGA_GAIN[4:1]
[13] CALOUT        : output de la calibración (read only)
[12] CALSEL[1]
[11] CALON         : 1 = arranca calibración
[10:8] PGA_GAIN[3:0]: ganancia configurada
[7] VP_SEL[1]      : selecciona entrada VINP
[6] VP_SEL[0]
[5:4] VM_SEL       : selecciona entrada VINM
[3:2] FORCEVP, OPAINTOEN
[1] OPAEN          : enable
[0] (reservado)
```

Pasos típicos de inicialización para PGA con feedback externo:

```c
// 1. Habilitar clock al SYSCFG (en STM32G4, los OPAMPs están en SYSCFG bus)
RCC->APB2ENR |= RCC_APB2ENR_SYSCFGEN;

// 2. Configurar VP_SEL, VM_SEL, PGA_GAIN
OPAMP1->CSR = (vp_sel << OPAMP_CSR_VPSEL_Pos)
            | (vm_sel << OPAMP_CSR_VMSEL_Pos)
            | (pga_gain << OPAMP_CSR_PGA_GAIN_Pos)
            | OPAMP_CSR_OPAINTOEN;   // route output to ADC internally

// 3. Calibrar offset (procedimiento RM0440 §25.3.7)
// ... 128 ms

// 4. Habilitar
OPAMP1->CSR |= OPAMP_CSR_OPAEN;

// 5. Esperar a que se estabilice (~2 μs típico)
```

### Por qué importa

1. **Sin OPAMP, no hay control de corriente.** La señal del shunt (mV) no es medible por el ADC directamente. El OPAMP es estructural, no opcional.

2. **La ganancia exacta del OPAMP define la escala del FCS-M2PC.** Si configuras gain x4 pero la placa esperaba x8, todas tus corrientes están a la mitad del valor real → el algoritmo "ve" la mitad de torque del que pide → respuestas dinámicas distorsionadas.

3. **La calibración de offset es crítica al arranque.** Vale 128 ms de tiempo de boot a cambio de eliminar ~5 mV de error DC. Aceptable.

4. **El mapeo silicon-fijo a canales ADC** restringe la distribución. No es decisión nuestra, es decisión de ST. Saberlo evita perder tiempo planeando distribuciones imposibles.

5. **El bandwidth del OPAMP es suficiente pero ajustado.** Si en el futuro queremos PWM a 100 kHz (mucho más rápido), tendríamos que activar `OPAHSM = 1` para tener margen. Por ahora a 50 kHz no.

---

## N1.12 — El ADC del STM32G4: cómo se convierte voltaje en número

### Panorama

El ADC es el periférico más complejo de Semana 5. No por sus features básicas (todos los micros tienen ADCs), sino por sus **opciones de configuración** — modos, triggers, sequences, dual modes. Esta nota lo desarma en sus partes.

El STM32G431 tiene **2 ADCs independientes** (ADC1 y ADC2). Cada uno:
- Resolución 12 bits (4096 niveles).
- Hasta 19 canales de entrada (mux interno).
- Sample time configurable.
- Trigger por hardware (ej. TIM1_TRGO) o software.
- Modo **dual simultaneous**: ambos ADCs disparan al mismo tiempo, conversiones en paralelo.

### Analogía

Un ADC es como un cronista que va apuntando valores de un río: cada vez que alguien le da el "tic" (trigger), mira el nivel del agua, lo redondea al cm más cercano (en el caso del ADC: 0.806 mV), y lo apunta en su libreta (registro de salida).

El cronista necesita dos cosas:
1. **Tiempo para "ver" el nivel correctamente** (sample time): si miras muy rápido, ves un movimiento borroso.
2. **Tiempo para "redondear" al cm más cercano** (conversion time): comparar el voltaje con sus referencias internas.

### Anatomía interna: Sample-and-Hold + SAR

El ADC del G4 es de tipo **Successive Approximation Register (SAR)**. Funciona en dos fases:

**Fase 1: Sampling (acquisition)** — un capacitor interno (~5 pF) se carga al voltaje de entrada.

```
Pin de entrada → [Switch] ──[Cap interno]── GND
                   │
                   │ closed durante sample time
                   │ open durante conversion
                   ▼
                comparador del SAR
```

El capacitor necesita tiempo para cargarse a través de la impedancia de la fuente. **Si tu fuente tiene alta impedancia** (señal de un sensor débil sin buffer), el sample time tiene que ser largo. Si tiene baja impedancia (salida de OPAMP), puede ser corto.

**Fase 2: Conversion (SAR)** — el comparador interno hace 12 decisiones binarias secuenciales para determinar el voltaje:

```
Iteración 1: ¿V > 1.65V? (mitad de 3.3V) → bit 11
Iteración 2: ¿V > 1.65V + 0.825V? → bit 10
Iteración 3: ¿V > 1.65V + 0.4125V? → bit 9
...
Iteración 12: ¿V > ...? → bit 0
```

Después de 12 iteraciones tienes 12 bits = 4096 niveles posibles. Esto tarda **12.5 ciclos del ADC clock** (el 0.5 es overhead del SAR).

### Tiempo total de conversión

$$T_{conv} = T_{sample} + 12.5 \text{ ciclos del ADC clock}$$

`T_sample` es programable por canal en `SMPR1` / `SMPR2`:

| SMP | Sample time (ciclos) | Cuándo usarlo |
|---|---|---|
| 000 | 2.5 | Fuentes de baja impedancia (OPAMP interno) |
| 001 | 6.5 | Estándar para señales rápidas |
| 010 | 12.5 | Equilibrio precisión/velocidad |
| 011 | 24.5 | Señales lentas o con impedancia media |
| 100 | 47.5 | Sensor externo via PCB |
| 101 | 92.5 | Termistor, sensor lento |
| 110 | 247.5 | Vrefint, sensor temperatura interno |
| 111 | 640.5 | El máximo, para ruido muy bajo |

**Para nuestro caso** (señales de OPAMP interno, alta velocidad necesaria):

A `ADC_clock = 60 MHz` (típico) y `SMP=001` (6.5 ciclos):
- T_sample = 6.5 / 60 MHz = **108 ns**
- T_conv = (6.5 + 12.5) / 60 MHz = 19 / 60 MHz = **316 ns**

Con 5 canales secuenciales (i_a, i_b, i_c, Vbus, temp), tiempo total ~1.6 μs. Cabe holgadamente en los 20 μs del periodo PWM.

### ADC clock y de dónde viene

El reloj del ADC viene del **AHB1 clock** o de un clock dedicado, configurable en `RCC->CCIPR.ADC12SEL`:

| ADC12SEL | Fuente | Implicación |
|---|---|---|
| 00 | No clock (ADC apagado) | Después de reset |
| 01 | PLL "P" output | Independiente del HCLK |
| 10 | sysclock | Síncrono con CPU |

**Para nuestro caso**: usar el PLL output P (la salida adicional del PLL que se puede configurar a una frecuencia distinta de SYSCLK). Esto desacopla el reloj del ADC del CPU.

Luego un divisor adicional en `ADC12_COMMON->CCR.CKMODE` o `CCR.PRESC`:

| CKMODE | Significado |
|---|---|
| 00 | ADC clock asíncrono (desde RCC) |
| 01 | HCLK/1 (síncrono) |
| 10 | HCLK/2 |
| 11 | HCLK/4 |

A HCLK = 170 MHz, dividir por 4 da 42.5 MHz — debajo del máximo de 60 MHz. Es la opción más simple. O usar el modo asíncrono con el PLL P a 60 MHz exactos.

### Regular vs Injected channels

El ADC tiene **dos grupos de canales** completamente independientes:

#### Regular channels (group)
- Hasta 16 canales en una secuencia.
- Configuras `SQR1`, `SQR2`, `SQR3`, `SQR4` con la secuencia.
- Se ejecutan **en orden**, uno tras otro.
- Trigger por `EXTSEL` + `EXTEN`.
- Datos accesibles en `ADC_DR` (un registro). Si hay más de 1 canal en la secuencia, **necesitas DMA** o leer rápido el `DR` entre conversiones.
- Flag EOC se setea después de cada canal, EOS al final de la secuencia.

#### Injected channels (group)
- Hasta 4 canales.
- Configurados en `JSQR`.
- Tienen **mayor prioridad** que regulares: si un trigger injected llega durante una conversión regular, ésta pausa y se ejecuta primero el injected.
- Trigger por `JEXTSEL` + `JEXTEN`.
- Datos accesibles en `JDR1, JDR2, JDR3, JDR4` (un registro por canal). **Sin necesidad de DMA**.
- Útil para señales críticas en tiempo (corrientes del motor).

**Para motor control**:
- **Inyectados**: las 3 corrientes (i_a, i_b, i_c). Críticas para FCS-MPC.
- **Regulares**: Vbus, temperatura. Diagnóstico, no críticos.

Esto es el patrón estándar en STM32 motor control (lo usa MCSDK).

### Dual ADC modes

Dos ADCs pueden coordinarse en varios modos. Para motor control el relevante es:

#### Regular simultaneous mode

ADC1 (master) y ADC2 (slave) ejecutan sus secuencias **al mismo tiempo**. El trigger del master dispara también al slave. Cada uno tiene su propia secuencia (`SQR1` distinta), así que pueden muestrear canales **distintos** en paralelo.

```
trigger TIM1_TRGO  ──┬──→ ADC1 → muestrea SQR1 (i_a, Vbus, temp)
                     └──→ ADC2 → muestrea SQR1 (i_b, i_c)
                                 ↓                 ↓
                              ADC1_DR           ADC2_DR
                              (o combinado en ADC_CDR)
```

**Pero aquí está el truco**: con regular simultaneous, ambos ADCs usan la misma configuración de `CFGR` (el master). Significa que solo necesitas programar el master.

#### Injected simultaneous mode

Idéntico al anterior pero usando canales inyectados. **Lo que vamos a usar**.

Para motor control la combinación canónica es:
- **Injected simultaneous**: i_a en ADC1 inyectado, i_b en ADC2 inyectado. Trigger: TIM1_TRGO.
- **Regular**: Vbus, temperatura en cualquier ADC. Triger: software o segundo TRGO.

### Configuración de dual mode

El bit field `DUAL[4:0]` vive en `ADC_CCR` (registro común de los dos ADCs):

| DUAL | Modo |
|---|---|
| 00000 | Independent (default) |
| 00001 | Combined regular + injected simultaneous |
| 00010 | Combined regular + alternate trigger |
| 00101 | **Injected simultaneous only** ← nuestro caso |
| 00110 | Regular simultaneous only |
| 00111 | Interleaved only |
| 01001 | Alternate trigger only |

**DUAL = 00101** (injected simultaneous) o **DUAL = 00110** (regular simultaneous) según decidamos arriba.

### Trigger sources (EXTSEL / JEXTSEL)

`EXTSEL[4:0]` en `ADC_CFGR` selecciona qué señal interna dispara las conversiones regulares. RM0440 Tabla 162 mapea los valores:

| EXTSEL | Trigger source |
|---|---|
| 00000 | TIM1_CC1 |
| 00001 | TIM1_CC2 |
| 00010 | TIM1_CC3 |
| 00011 | TIM1_CC4 |
| **01001** | **TIM1_TRGO** ← lo que queremos |
| 01010 | TIM1_TRGO2 |
| ... | (muchos otros) |

Para `JEXTSEL` (inyectado): tabla análoga, mismos valores conceptualmente.

Y `EXTEN[1:0]` activa el trigger:

| EXTEN | Significado |
|---|---|
| 00 | Trigger deshabilitado (modo software con ADSTART) |
| 01 | Rising edge |
| 10 | Falling edge |
| 11 | Both edges |

### Flags importantes

| Flag | En registro | Set cuando |
|---|---|---|
| **ADRDY** | ISR | ADC listo después de habilitar (ADEN=1) |
| **EOC** | ISR | Una conversión terminó. Limpiar leyendo DR. |
| **EOS** | ISR | Secuencia completa terminó |
| **JEOC** | ISR | Una conversión inyectada terminó |
| **JEOS** | ISR | Secuencia inyectada completa terminó |
| OVR | ISR | Overrun (no se leyó DR a tiempo) |

Las interrupciones se habilitan con `EOCIE`, `JEOCIE` en `IER`. **Para nuestra ISR de FCS-MPC**, vamos a usar `JEOS` (interrupt cuando el sequence inyectado termina, indicando que i_a e i_b están listos).

### Calibración del ADC

Antes de habilitar el ADC, hay que **calibrarlo**. Es otra calibración distinta a la del OPAMP — esta es del ADC mismo. Cancela offsets internos del SAR comparator.

Procedimiento (RM0440 §21.4.8):

```c
// 1. Salir de Deep Power Down + arrancar regulador interno
ADC1->CR &= ~ADC_CR_DEEPPWD;
ADC1->CR |= ADC_CR_ADVREGEN;
delay_us(20);  // estabilización del regulador

// 2. Asegurar que ADEN = 0
// 3. Configurar ADCALDIF (0 = single-ended, 1 = differential). Usamos 0.
ADC1->CR &= ~ADC_CR_ADCALDIF;

// 4. Lanzar calibración
ADC1->CR |= ADC_CR_ADCAL;

// 5. Esperar a que termine (ADCAL = 0)
while (ADC1->CR & ADC_CR_ADCAL);

// 6. Los valores de calibración quedan en CALFACT, se aplican automáticamente
```

Tarda ~80 ciclos del ADC clock (~1.3 μs a 60 MHz). Una sola vez al arranque.

### Orden de inicialización del ADC

Pegado para referencia, el orden importa:

1. Habilitar clock del ADC (RCC).
2. Configurar clock source y prescaler (`CCIPR`, `CCR`).
3. Salir de Deep Power Down + arrancar regulador (`CR.DEEPPWD = 0`, `CR.ADVREGEN = 1`).
4. Calibrar (`CR.ADCAL = 1`, esperar).
5. Configurar canales: secuencias regulares e inyectadas (`SQR1`, `JSQR`, `SMPR1/2`).
6. Configurar trigger (`CFGR.EXTSEL`, `CFGR.EXTEN`, `JSQR.JEXTSEL`, `JSQR.JEXTEN`).
7. Configurar dual mode (`ADC_CCR.DUAL`).
8. Habilitar interrupts (`IER.JEOSIE`).
9. Habilitar ADC (`CR.ADEN = 1`, esperar `ISR.ADRDY = 1`).
10. **Recién entonces** habilitar el TIM1 (que dispara TRGO).

### Decisiones para nuestra implementación

| Parámetro | Valor decidido |
|---|---|
| ADC1 inyectados | i_a (canal OPAMP1 = IN13) |
| ADC2 inyectados | i_b (OPAMP2 = IN16), i_c (OPAMP3 = IN18) |
| ADC1 regulares | Vbus (PA0 = IN1), temperatura (PB14 = IN5) |
| ADC clock source | PLL P, asíncrono al CPU, ~30 MHz |
| Sample time | 6.5 ciclos (SMP=001) — fast, OPAMPs como fuente |
| Resolución | 12 bits |
| Dual mode | DUAL=00101 (injected simultaneous only) |
| Trigger inyectado | JEXTSEL=01001 (TIM1_TRGO), JEXTEN=01 (rising) |
| ISR | JEOS (end of injected sequence) → en cada periodo PWM |
| DMA | No por ahora (más simple); evaluar si la ISR no entra en presupuesto |

### Por qué importa

1. **El ADC es la fuente de "verdad" del controlador.** Toda la matemática del FCS-M2PC depende de números que vienen de aquí. Si los muestreas mal (sample time corto, trigger desincronizado), TODO el control diverge.

2. **Injected vs regular es decisión estructural.** Los canales críticos (corrientes) van en inyectados porque pueden interrumpir conversiones de menor prioridad. Si pusiéramos todo en regulares, una conversión de temperatura podría retrasar la corriente.

3. **Dual simultaneous mode existe específicamente para motor control.** Sin él, muestrearías i_a, esperar 316 ns, muestrear i_b — y entre uno y otro la corriente del motor cambió. Con dual, los dos son captados en el mismo instante.

4. **El timing total es ajustado pero cabe.** 5 canales × 316 ns + overhead ISR + transformación Clarke + algoritmo FCS-M2PC ≈ 5-15 μs. Cabe en 20 μs pero hay que vigilar.

5. **La calibración del ADC es separada de la del OPAMP.** Son dos calibraciones distintas, ambas necesarias. Sin ellas: offset DC permanente en las lecturas.

---

## N1.13 — Bring-up del ADC: tres trampas que encontramos

### Panorama

El bring-up de la cadena OPAMP + ADC en Semana 5 tomó más tiempo del esperado. Tres bugs distintos aparecieron, **todos relacionados con asunciones que parecían razonables pero no estaban verificadas contra la fuente autoritativa**. Esta nota los documenta para futuro.

### Trampa #1 — El clock "oculto" del SYSCFG

**Síntoma**: después de configurar `OPAMP1->CSR = ...`, una lectura back devolvía `0x00000000`. Los OPAMPs aparentaban no estar habilitados.

**Causa**: los registros de los OPAMPs viven en el bus APB2 a través del periférico SYSCFG. **Si el clock del SYSCFG no está habilitado, las escrituras a `OPAMPx_CSR` se ignoran silenciosamente** — sin error, sin warning, sin bus fault. Solo devuelven 0 al leer.

**Fix**: agregar antes de configurar los OPAMPs:

```c
RCC->APB2ENR |= RCC_APB2ENR_SYSCFGEN;
(void)RCC->APB2ENR;  // sync barrier
```

**Lección general**: cualquier periférico cuyas escrituras se ignoran silenciosamente probablemente le falta su clock. La regla heurística: **si lees back lo que escribiste y devuelve 0**, sospechar clock antes de cualquier otra cosa.

### Trampa #2 — Las tablas duplicadas: EXTSEL ≠ JEXTSEL

**Síntoma**: el TIM1_TRGO no estaba disparando las conversiones inyectadas del ADC, a pesar de que el counter del TIM1 corría y el TRGO estaba bien configurado en el TIM1 (MMS=010).

**Causa**: RM0440 tiene **dos tablas** distintas para mapear los triggers:
- **Tabla 162** — `EXTSEL[4:0]` para conversiones **regulares**: `01001 = TIM1_TRGO`.
- **Tabla 167** — `JEXTSEL[4:0]` para conversiones **inyectadas**: `00000 = TIM1_TRGO`.

Los valores son **completamente distintos**. El mismo TIM1_TRGO usa código `9` para regular y código `0` para inyectado.

Mi código tenía `JEXTSEL = 9` asumiendo que la tabla EXTSEL aplicaba. Resultado: el ADC esperaba TIM8_TRGO (que sí es 9 en JEXTSEL) y nunca disparaba.

**Fix**:

```c
ADC1->JSQR = ...
           | (0x0U << ADC_JSQR_JEXTSEL_Pos)  // 00000 = TIM1_TRGO en Tabla 167
           ...
```

**Lección general**: **siempre verifica la tabla específica del modo que estás usando**. Las tablas que mapean valores a fuentes no se reusan entre modos del mismo periférico — son matrices independientes que el silicio implementa con muxes separados. Mismo patrón que el bug AF de N1.9 (cada pin tiene su tabla AF).

### Trampa #3 — Standalone mode requiere conocer la topología del PCB

**Síntoma**: con OPAMPs en standalone mode (que es el modo "natural" cuando el PCB tiene R's de feedback discretas), las lecturas saturaban en el rail (~4093 raw = casi 3.3 V).

**Causa**: en standalone mode, **el OPAMP es solo el amplificador — no hay feedback interno**. La ganancia y el comportamiento dependen 100% de las R's externas y de qué pines exactos son VINP y VINM. Si configuras VP_SEL/VM_SEL a pines que el PCB no tiene en su red de feedback, el OPAMP queda efectivamente open-loop → salta a rail al primer estímulo.

La B-G431B-ESC1 tiene su red de R's diseñada para una asignación específica de VP_SEL/VM_SEL, pero ST no la documenta explícitamente en UM2516 — hay que leer el esquemático MB1419 con cuidado.

**Fix temporal**: switch a **PGA mode** con feedback interno del silicio. Output del OPAMP = gain × VINP, independiente del PCB. Con gain x2 las lecturas bajan a ~318 raw (consistente con offset DC pequeño y ausencia de saturación).

**Tradeoff aceptado**: PGA mode interno sin bias a Vrefint/2 → solo medimos corrientes **positivas** (output del OPAMP nunca baja de 0V). Para FCS-MPC con corrientes bipolares AC, eventualmente hay que:
- Volver a standalone mode descifrando bien el PCB, o
- Mantener PGA mode + restar offset DC en software (más simple).

**Lección general**: cuando dependes de hardware externo (PCB), **standalone mode es frágil** — depende de info que puede no estar bien documentada. **PGA mode interno** es más robusto para bring-up. Una vez la cadena funciona en PGA, optimizar a standalone si se necesita más rango dinámico.

### Meta-lección común a las tres trampas

Los tres bugs comparten un patrón: **asumir uniformidad donde el silicio diseñó variabilidad**.

- "Todos los clocks de periféricos se habilitan igual" → falso (SYSCFG es indirecto).
- "Las tablas de trigger son simétricas entre regular e injected" → falso (matrices independientes).
- "Standalone mode siempre funciona si pones VP_SEL/VM_SEL correctos" → falso (depende del PCB).

El antídoto en metodología:
1. **Para cada periférico nuevo**: ¿qué clocks necesita habilitar? Listar explícitamente, no asumir.
2. **Para cada modo de un periférico**: ¿hay tablas/registros distintos por modo? Verificar en el manual antes de codear.
3. **Para cada modo que depende de hardware externo**: ¿qué espera el PCB? Si no está claro, usar un modo "self-contained" primero.

### Tabla maestra del ADC + OPAMP para B-G431B-ESC1

Persistir para referencia:

| Item | Valor |
|---|---|
| Clock SYSCFG (para OPAMPs) | `RCC->APB2ENR \|= RCC_APB2ENR_SYSCFGEN` |
| Clock ADC | `RCC->AHB2ENR \|= RCC_AHB2ENR_ADC12EN` |
| ADC clock prescaler | CKMODE = 11 (HCLK/4 = 42.5 MHz) |
| JEXTSEL para TIM1_TRGO | **0x00** (Tabla 167, no 0x09) |
| JEXTEN | 01 (rising edge) |
| Dual mode | DUAL = 00101 (injected simultaneous only) |
| OPAMP mode actual | PGA interno gain x2 (provisional) |
| OPAMP1 routing | VINP0 = PA1, OPAMPINTEN → ADC1 IN13 |
| OPAMP2 routing | VINP0 = PA7, OPAMPINTEN → ADC2 IN16 |
| OPAMP3 routing | VINP0 = PB0 [VERIFY], OPAMPINTEN → ADC2 IN18 |
| Sample time | SMP = 001 (6.5 ciclos) |
| Resolución | 12 bits (default) |
| Vbus channel | ADC1_IN1 (PA0) |
| Temp channel | ADC1_IN5 (PB14) |

### Para mañana / Semana 6

Con la cadena funcional, lo que sigue:

1. **ISR JEOS**: callback cuando la secuencia inyectada termina. Es donde el FCS-MPC vivirá.
2. **Calibración de offset DC**: 1000 muestras motor off → promedio → restar a futuras lecturas. Esto convierte el 318 raw en "i_a = 0 sin corriente".
3. **Ganancia "raw → amperios"**: inyectar corriente conocida (multímetro + alimentación externa) → calibrar.
4. **OPAMP topology resolution**: decidir si volvemos a standalone con bias (mejor rango) o nos quedamos en PGA + offset SW (más simple).

---

## N1.14 — ISR JEOS: el latido del lazo de control

### Panorama

Hasta ahora todo el firmware corría en el `main()`: el while(1) imprime, hace polling al ADC, manda por VCP. El CPU está siempre "trabajando", pero **el trabajo es despacio y asíncrono respecto al PWM** — no hay garantía de que cuando leemos `JDR1` el dato corresponda a un ciclo PWM específico.

Para el control de un motor a 50 kHz, eso no sirve. El FCS-M2PC necesita que **cada 20 μs**, exactamente sincronizado con el pico/valle del contador del TIM1, ocurra una secuencia rígida:

```
t = 0:        TRGO levanta
t = 0-3 μs:   ADCs convierten i_a, i_b, i_c (paralelo)
t = ~3 μs:    JEOS levanta → IRQ → handler entra
t = 3-X μs:   handler lee corrientes, calcula control,
              actualiza CCR1/CCR2/CCR3
t = X-20 μs:  main() puede hacer otras cosas (UART, etc.)
t = 20 μs:    siguiente TRGO → ciclo se repite
```

La pieza que dispara ese ritmo es la **ISR** (*Interrupt Service Routine*): una función especial que el CPU ejecuta cuando un periférico levanta su línea de interrupción. **No la llama el main loop** — la dispara el silicio directamente.

Esta sesión solo construimos el esqueleto del handler. Sin lógica de control aún. El objetivo: confirmar que el lazo está vivo, late a 50 kHz exactos, y deja las corrientes accesibles en variables globales.

### Analogía — el médico de guardia

Imaginá un médico de guardia en un hospital:

- **Modo polling** (lo que hacemos ahora): el médico camina hasta el cuarto del paciente cada cierto tiempo, mide la presión, anota, vuelve a la sala de descanso. Si pasa algo entre visitas, no se entera. Si tarda en volver, la medición queda desactualizada.

- **Modo interrupción** (lo que vamos a hacer): el paciente tiene un botón rojo conectado al busca del médico. El médico está en la sala leyendo (haciendo otras cosas). Cuando el monitor del paciente termina de medir, **suena el busca**. El médico deja el libro, va al cuarto, lee el monitor, vuelve. Sabe **exactamente** cuándo medir y nunca pierde una medición.

La ISR es ese busca: una señal de hardware que **interrumpe** al CPU sin importar qué esté haciendo, lo manda al handler, y al volver, retoma exactamente donde estaba.

```
Sin ISR (polling):
main: ──read─work──work──read─work──work──read───
ADC:  ─█──────────█──────────█────────────█──────  (datos perdidos entre reads)

Con ISR:
main: ──work──work──work──work──work──work───────
ADC:  ─█──────█──────█──────█──────█──────█──────
       ↓      ↓      ↓      ↓      ↓      ↓
ISR:   ▌      ▌      ▌      ▌      ▌      ▌       (entra exacto a cada conv)
```

### Detalle 1 — Anatomía del flag JEOS

El ADC del STM32G4 tiene **cuatro tipos de flag** que pueden disparar interrupciones (RM0440 §21.4.31, registro `ADC_ISR`):

| Flag | Significado |
|---|---|
| `EOC` | End Of Conversion — terminó **un** canal individual |
| `EOS` | End Of Sequence — terminó la secuencia **regular** |
| `JEOC` | igual a EOC pero para inyectadas |
| **`JEOS`** | igual a EOS pero para inyectadas |

Para nuestro caso (las 3 corrientes vienen por la secuencia inyectada disparada por TIM1_TRGO), **JEOS** es el flag que importa: se levanta cuando ADC2 termina sus 2 conversiones (i_b → i_c). ADC1 ya terminó antes (1 sola conversión, i_a), pero el dato sigue ahí en `JDR1` esperando ser leído.

Cada flag tiene su bit "interrupt enable" correspondiente en `ADC_IER`. Para JEOS: bit `JEOSIE`. Sin setear ese bit, el flag se levanta pero **no genera IRQ** — sirve solo para polling (que es lo que estábamos haciendo).

### Detalle 2 — El NVIC y la línea compartida

El **NVIC** (Nested Vectored Interrupt Controller) es la pieza del Cortex-M4 que **routea las señales de IRQ desde los periféricos al CPU** y maneja sus prioridades.

Cada periférico tiene una "línea" asignada (un número de IRQ). El STM32G431 tiene 102 líneas (RM0440 §14.3, vector table). Cuando una línea se activa, el NVIC:

1. Pausa el CPU.
2. Guarda el contexto (registros R0-R3, R12, LR, PC, xPSR) en el stack.
3. Salta al handler asociado a esa línea.
4. Al return del handler, restaura el contexto y el CPU sigue como si nada.

**Detalle crítico para el ADC**: ADC1 **y** ADC2 comparten **una sola línea NVIC** (IRQ 18 en STM32G431, llamada `ADC1_2_IRQn`). El handler también es uno solo: `ADC1_2_IRQHandler`. Si JEOSIE está habilitada en ambos ADCs, ambos pueden disparar el mismo handler. Dentro del handler hay que mirar `ADC1->ISR` y `ADC2->ISR` para saber quién disparó.

Para nuestro caso, esto se simplifica: **solo habilitamos JEOSIE en ADC2** (el lento). ADC1 nunca dispara IRQ. El handler solo tiene que limpiar `ADC2->ISR`.

```
                          ┌──────────────┐
ADC1.JEOS ────[JEOSIE=0]──│              │
                          │  NVIC IRQ 18 │──→ CPU → ADC1_2_IRQHandler()
ADC2.JEOS ────[JEOSIE=1]──│              │
                          └──────────────┘
```

### Detalle 3 — Prioridades NVIC

El NVIC soporta prioridades configurables (0 = más alta, 15 = más baja en Cortex-M4 con 4 bits de prioridad). Si dos IRQs ocurren simultáneamente, el de mayor prioridad gana. Si una IRQ está corriendo y llega otra de **mayor** prioridad, ésta **preempta** (interrumpe) la primera.

Para el ADC1_2 vamos a usar **prioridad 1** (alta pero no la más alta). Reservamos prioridad 0 para faults catastróficos (HardFault, MemManage, etc., que ya están allí por default).

```c
NVIC_SetPriority(ADC1_2_IRQn, 1U);
NVIC_EnableIRQ(ADC1_2_IRQn);
```

**Por qué no la máxima**: si alguna vez agregamos un break input del TIM1 o un watchdog que detecta sobrecorriente, ese **sí** debe poder preemptar al lazo de control. Dejarle margen es disciplina, no over-engineering.

### Detalle 4 — Latencia y jitter

Cuando JEOS se levanta hasta que el primer instrucción del handler se ejecuta, hay un **retraso** llamado **latencia de IRQ**. En Cortex-M4 con stacking automático y FPU desactivada, la latencia mínima es **12 ciclos** (~70 ns a 170 MHz). Si la IRQ llega durante una instrucción multi-ciclo (e.g., `LDM`/`STM` con muchos registros), puede subir a ~20 ciclos.

El **jitter** es la variación de esa latencia entre disparos consecutivos. En general < 5 ciclos. Es despreciable para nuestro caso (50 kHz = 3400 ciclos por período → latencia < 1% del período).

**Lo que sí importa**: el handler debe **terminar antes del próximo TRGO** (20 μs = 3400 ciclos). Sino, el siguiente JEOS llega mientras el handler aún corre → o se pierde, o (si está en modo "pending") se ejecuta inmediatamente tras el actual y se rompe el timing.

Por eso instrumentamos con GPIO: **pulse width en PB8** = duración del handler. Si vemos que se acerca a 20 μs, ¡problema!

### Detalle 5 — El handler mínimo

Para esta sesión, el cuerpo del handler:

```c
void ADC1_2_IRQHandler(void) {
    GPIOB->BSRR = (1U << 8);              // PB8 HIGH (inicio scope)

    g_ia_raw = (uint16_t)ADC1->JDR1;       // i_a (1 canal de ADC1)
    g_ib_raw = (uint16_t)ADC2->JDR1;       // i_b (canal 1 de ADC2)
    g_ic_raw = (uint16_t)ADC2->JDR2;       // i_c (canal 2 de ADC2)

    g_isr_count++;                         // contador para validar 50 kHz

    ADC2->ISR = ADC_ISR_JEOS;              // limpia flag (write-1-to-clear)

    GPIOB->BSRR = (1U << (8 + 16));        // PB8 LOW (fin scope)
}
```

**Lo que NO hace** (intencionalmente):
- No calcula nada (sin Clarke, sin Park, sin FCS-MPC).
- No actualiza CCR1/CCR2/CCR3 (PWM duty queda 50% fijo).
- No llama `printf` (printf en ISR = catastrófico — UART polling tarda ~9000 ciclos por línea = 5 períodos PWM, rompe el lazo).
- No espera nada (no `while`, no `delay`).

El target de duración: **< 200 ns** (~30 ciclos) para esta versión mínima. Esto deja 99% del período libre — margen enorme para cuando agreguemos el control.

### Detalle 6 — `volatile` y `g_*_raw`

Las variables `g_ia_raw` etc. son **compartidas entre el handler y main()**. Sin `volatile`:

```c
uint16_t g_ia_raw;   // ❌ NO volatile
```

el compilador puede optimizar `printf("%u", g_ia_raw)` cargando el valor **una sola vez** en un registro y reusarlo. Como la ISR modifica la memoria pero no el registro, el printf nunca vería los updates.

Con `volatile`:

```c
volatile uint16_t g_ia_raw;   // ✓ cada lectura va a memoria
```

el compilador garantiza una lectura/escritura por cada acceso en el código C. **Mandatorio para cualquier variable compartida ISR ↔ main**.

(Aparte: para uint16_t en Cortex-M4, una lectura/escritura es atómica. Para uint32_t también. Para uint64_t o estructuras, hay que ser más cuidadoso.)

### Por qué importa

1. **Sin ISR, no hay lazo de control determinista.** El FCS-MPC necesita que cada decisión se tome a un intervalo fijo conocido (20 μs) — si lo hace el main loop, el intervalo varía con lo que el main esté haciendo (UART, lecturas, etc.). El control inestable o degradado es indistinguible de un control mal sintonizado: te vas a volver loco buscando el bug en el control cuando el bug está en el timing.

2. **El GPIO toggle no es decoración**. Es **el único medio** de verificar empíricamente la frecuencia y duración del handler. El scope te muestra inmediatamente:
   - ¿La ISR está corriendo? (pulsos visibles)
   - ¿A qué frecuencia? (período = 20 μs si todo bien)
   - ¿Cuánto tarda? (ancho del pulso)
   - ¿Hay jitter? (varianza del período)

   Sin esta instrumentación, debugar un control que "no converge" es disparar a ciegas.

3. **Decidir dónde habilitar JEOSIE (ADC1 vs ADC2) es la diferencia entre datos válidos y datos basura.** Habilitar JEOSIE en ADC1 hace que el handler entre antes de que ADC2 termine i_c → leemos `JDR2` con basura del ciclo anterior. El control con i_c desfasado un ciclo se vuelve un caos que tarda horas en identificar. Documentamos esto explícitamente para no pisarlo.

4. **Reglas de oro para el handler**:
   - **No printf, no UART, no delay, no while.**
   - **Variables compartidas siempre `volatile`.**
   - **Limpiar el flag al final** (sino se reentra inmediatamente).
   - **Toggle GPIO al entrar y salir** para diagnóstico.
   - **Duración objetivo < 50% del período PWM** (10 μs en nuestro caso). Si se acerca, simplificar la lógica o mover trabajo al main loop.

### Tabla maestra de la ISR para esta sesión

| Item | Valor |
|---|---|
| Trigger del lazo | TIM1_TRGO (update event, 50 kHz) |
| Flag que dispara IRQ | `ADC2.JEOS` (no ADC1 — ADC2 termina último) |
| Bit de habilitación | `ADC2.IER.JEOSIE = 1` |
| Línea NVIC | `ADC1_2_IRQn` (= IRQ 18) |
| Handler | `ADC1_2_IRQHandler` |
| Prioridad | 1 (alta, no máxima) |
| GPIO instrumentación | PB8 (Z+/H3 de J8, reservado en sesión 2) |
| Variables compartidas | `g_ia_raw, g_ib_raw, g_ic_raw, g_isr_count` (todas `volatile`) |
| Duración objetivo handler | < 200 ns (~30 ciclos @ 170 MHz) |
| Validación | Scope PB8: período 20 μs ± 0.1 μs, pulso < 200 ns. VCP: `g_isr_count` crece ~50000/s |

### Para sesión siguiente

Si la ISR valida bien:
- **Calibración de offset**: dentro del handler (o en función separada llamada desde main), promediar N=1000 muestras → guardar `i_offset_a/b/c`. Cada lectura posterior es `g_ia_raw - i_offset_a`.
- **Calibración de ganancia**: inyectar corriente DC conocida con fuente bench → leer raw → calcular escala raw → A.

Si la ISR muestra problemas (jitter alto, frecuencia distinta, pulsos perdidos), debug primero antes de avanzar.

### Sospechas pendientes — lo que NO confirmamos con scope explícito

En la sesión 10 (2026-05-22/23) el bring-up cerró con validación parcial:

- **Lo confirmado**:
  - `g_isr_count` crece a 50000/s exactos (descontando el delay del printf, 1007 ms efectivos entre prints).
  - Multímetro DC en pad Z+/H3 del J8 marca **44 mV** durante operación normal — consistente con un pulso de ~270 ns cada 20 μs (3.3 V × 270/20000 = 44.6 mV).
  - `ADC2.IER = 0x40` (JEOSIE encendido) post-init.
  - Lecturas i_a/i_b/i_c estables y consistentes con la sesión 9.

- **Lo NO confirmado** — sospechas latentes que pueden mordernos si el control no converge:

  1. **No vimos el pulso de PB8 en el scope explícitamente.** La evidencia es DC promedio + análisis aritmético. Si más adelante el FCS-MPC no converge, hay timing weird, o el handler parece más lento de lo esperado:
     - Reflashear y mirar con scope a **1 o 2 μs/div, trigger Edge/Rising, level 1.5 V, AUTO mode, probe x1, coupling DC**. Esos settings sí enganchan pulsos de 200-300 ns.
     - Esperado: pulsos angostos cada 20 μs exactos, amplitud 3.3 V.
     - Si el pulso es mucho más ancho que ~300 ns, el compilador no inlineó algo o hay código no obvio en el path — investigar con disassembly.

  2. **No medimos jitter ciclo a ciclo.** El promedio de 50000 ISRs/s no descarta que algunos ciclos lleguen tarde por preemption de otra IRQ. Hoy no hay otras IRQs habilitadas, pero cuando agreguemos UART RX, I²C, etc., habrá que verificar con persistence del scope.

  3. **No hicimos continuidad MCU pin 29 (PB8 físico) ↔ pad Z+/H3.** Asumimos la conexión basados en UM2516 Tabla 4 + esquemático MB1419 (página del J8 con R77 1.8k + pull-up). Los 44 mV son evidencia fuerte de que la conexión existe, pero si en algún punto los voltajes lateralmente cambian sin razón aparente, hacer continuidad con multímetro a placa apagada.

  4. **Otros pads del J8 mostraron voltajes raros** durante la sesión:
     - A+/H1 (PB6) = 2.0 V — esperado 3.3 V si solo está R71 pull-up de 10k a Vcc. Algo más en la red lo tira hacia abajo (¿AS5600 dormido?).
     - B+/H2 (PB7) = 2.9 V — más cerca de 3.3 V pero no llega.
     - **No nos afecta hoy** (I²C deshabilitado, AS5600 sin firmware todavía), pero **es una bandera para cuando arranquemos I²C en una sesión futura**. Si el bus I²C no arranca, recordar estos voltajes — el AS5600 puede tener un estado raro de power-up.

  5. **No medimos la frecuencia con instrumento externo**. La frecuencia "50 kHz exactos" viene del cálculo `g_isr_count / uptime`, donde `uptime` se mide con SysTick que a su vez está clockeado por el mismo SYSCLK que el TIM1. Si el HSE de la placa tiene un error de calibración, **no podemos detectarlo internamente** — todos los timers se desviarían juntos manteniendo la "consistencia interna". Para descartar: medir un PWM (e.g. salida de TIM1 en PA8) con frecuencímetro externo y comparar con 50 kHz nominales. Hoy no hace falta.

### Antídoto si algún día estos puntos importan

Cuando arranquemos el FCS-MPC real y el control no converja, **no asumir que el lazo está bien**. Volver a esta lista, validar visualmente con scope, y descartar cada item ANTES de hurgar en el control.

---

## N1.15 — Race condition latente: regular vs injected en el mismo ADC

### Panorama

En sesión 11 vimos que `adc_get_vbus_raw()` devolvía **552 raw**, lo cual asumimos era "12V" sin pensarlo. En sesión 12, con un handler más lento (stats agregadas), la misma función devolvió **1443 raw**. El número 1443 coincide perfectamente con la física del divisor de Vbus (12 V × 0.0963 / 3.3 × 4096 = 1434).

**El 552 era el bug. El 1443 es lo correcto.** La función no cambió — lo que cambió es el timing del handler que afecta la interacción entre las dos colas de conversión del ADC.

### Detalle — qué pasa cuando regular e injected comparten ADC

ADC1 está corriendo **dos secuencias simultáneamente**:

1. **Inyectada** (i_a en JDR1) — disparada por TIM1_TRGO a 50 kHz. JEOS levanta IRQ a través del NVIC.
2. **Regular** (Vbus en SQR1) — disparada por software desde `adc_get_vbus_raw()` cuando el main la llama.

Las dos comparten el mismo silicio. Por RM0440 §21.4.16, **las inyectadas tienen mayor prioridad y pausan a la regular** si coinciden temporalmente. Cuando la inyectada termina, la regular continúa donde quedó.

El problema sutil: el código de `adc_get_vbus_raw` hace polling de los flags `EOC` y `EOS` en `ADC1->ISR`, sin distinguir si el flag fue puesto por la **regular** que estamos esperando o por una **inyectada** que se intercaló:

```c
ADC1->ISR = ADC_ISR_EOS | ADC_ISR_EOC;   /* limpia ambos */
ADC1->CR |= ADC_CR_ADSTART;               /* arranca regular */
while ((ADC1->ISR & ADC_ISR_EOC) == 0U) { }   /* espera EOC */
uint16_t vbus = ADC1->DR;                 /* ⚠ puede ser de injected */
```

Si una inyectada termina entre el `ADSTART` y el primer break del `while`, EOC se levanta por la inyectada (no por la regular), y leemos `ADC1->DR` que tiene el valor **inyectado** (el del shunt amplificado por OPAMP), NO el del Vbus_sense.

### Por qué el bug se "auto-arregló" en sesión 12

El handler de sesión 11 era mínimo (~30 ciclos). El de sesión 12 agregó stats (~25 ciclos) → handler ~55-60 ciclos. Esta dilación cambió el momento exacto en el que el JEOS suelta el control del ADC, dándole ventana a la regular para completarse limpia antes del próximo TRGO.

El bug no está resuelto — está **escondido por timing accidental**. Si en una sesión futura el handler vuelve a ser corto (porque optimizamos algo), o la regular se llama desde otro hilo de ejecución, el síntoma vuelve.

### Tres formas de resolverlo correctamente

| Approach | Pro | Contra |
|---|---|---|
| **A. Mover Vbus a ADC3** (no usado hoy) | Aisla regular de injected | Habilitar otro ADC = más config + clock + canales |
| **B. Agregar Vbus a la cola inyectada** | Llega cada 50 kHz sincronizado con corrientes | ADC1 ya tiene 1 canal inyectado, agregar uno cambia el JL y el timing relativo |
| **C. DMA para la regular** | Independencia total entre cola regular e injected, sin polling | Más config, agrega complejidad |

Mi recomendación: **A o B** dependiendo de qué otros canales queramos sumar después. La AS5600 vía I²C no necesita ADC, así que probablemente nos sobra ADC3 para Vbus + temp + monitoring.

### Bandera de detección

Si en VCP en algún momento aparece `Vbus=552` o cualquier número que **no corresponda al voltaje real medido con multímetro**, el race está activo. El multímetro DC en J5 (Vbus) es la verificación de ground truth.

### Por qué importa

1. **Vbus es entrada de seguridad**. Si la lectura está mal y el control depende de Vbus (e.g., para normalizar duties, o como entrada del FCS-M2PC para predicción), el control se rompe **silenciosamente** — sin error, solo con resultados malos.

2. **Mismo patrón aplica a temperatura** (canal 5 en la regular). Hoy no lo usamos, pero cuando agreguemos protección térmica, el race afecta también la temp.

3. **Cualquier código que poll flags compartidos entre cargas concurrentes del mismo periférico tiene este patrón de bug**. Aplica también a I²C (`ISR.RXNE` con DMA + polling), USART, etc. Antídoto general: **no usar polling de flags si hay otro mecanismo (DMA, IRQ específica) corriendo en paralelo**.

---

## N1.16 — Bring-up del AS5600 por I²C1: el sentido de posición

### Panorama

Hasta aquí el banco sabía dos cosas: cuánta corriente circula (N1.10–N1.13) y cuándo
calcular (N1.14). Le faltaba la tercera, y sin ella el FCS-M²PC no existe: **dónde está
el rotor**.

El motivo es directo. La BEMF del motor es $\mathbf{e} = K_e\,\omega_m\,\mathbf{s}(\theta_e)$.
Toda la tesis se apoya en estimar esa forma $\mathbf{s}(\theta_e)$ con el ADALINE, y el
ADALINE se alimenta de un regresor de Fourier evaluado en $\theta_e$. Sin posición no hay
regresor, sin regresor no hay estimación, sin estimación no hay referencia de corriente.
La cadena entera cuelga de este dato.

El sensor es un **AS5600**: encoder magnético absoluto de 12 bits, integrado de fábrica al
motor 2804. "Absoluto" significa que al encender ya sabe su ángulo — no necesita una vuelta
de homing como un encoder incremental. Lee el campo de un imán diametral pegado al eje
mediante sensores Hall en el silicio, y reporta el ángulo por **I²C**.

Esta nota tiene dos partes. La primera es cómo funciona el bus y por qué el driver está
escrito como está. La segunda es **la trampa que costó la sesión entera**, que resultó no
ser ninguna de las cosas que parecía.

### Analogía — el bus de dos hilos como una conversación por radio

I²C es una radio compartida de dos cables donde todos escuchan y hablan por turnos. Un
cable es la voz (**SDA**, datos), el otro el metrónomo (**SCL**, reloj). El maestro —nuestro
STM32— es el único que marca el ritmo.

La particularidad está en la electrónica: nadie puede *empujar* la línea hacia arriba. Cada
participante solo tiene permiso para **tirarla a tierra**. El "1" lógico no lo genera nadie:
lo provee una resistencia de pull-up que mantiene el cable arriba cuando todos sueltan. Eso
es **open-drain**, y es lo que permite que varios dispositivos compartan el mismo par de
cables sin destruirse: si dos hablan a la vez, ambos tiran a cero y nadie pelea contra nadie
por imponer un 3.3 V.

Una transacción típica es una llamada corta con acuse de recibo. El maestro dice "atención,
dispositivo 0x36"; el esclavo contesta con un **ACK** (hunde SDA un ciclo). Si nadie
contesta, la línea se queda arriba: eso es un **NACK**, "aquí no hay nadie con ese nombre".

Reténgase esa distinción, porque más abajo es la que resuelve el caso: **NACK significa que
el reloj corrió**. Para llegar al bit de acuse hay que haber generado nueve pulsos de SCL.
Un bus que ni siquiera puede completar la condición de parada no da NACK: da *timeout*.
Son dos fallos completamente distintos que a simple vista parecen el mismo.

### Detalle 1 — Por qué el kernel clock es HSI16 y no PCLK1

En el STM32G4 casi todos los periféricos serie eligen de qué reloj se alimentan. El registro
es `RCC_CCIPR`, campo `I2C1SEL` (RM0440 §7.4.27): `00` = PCLK1, `01` = SYSCLK, `10` = HSI16.

Elegimos **HSI16** (`i2c.c`), y no es arbitrario:

1. **Desacople del PLL.** PCLK1 corre a 170 MHz derivado del HSE por el PLL. Si algún día
   tocamos M/N/R —para bajar consumo, para cambiar Ts, para lo que sea— la temporización del
   I²C se movería con él, en silencio. Con HSI16 el bus vive en su propio dominio.
2. **Timing tabulado.** ST publica valores de `TIMINGR` ya calculados para 16 MHz. Con
   170 MHz habría que derivar el preescalador a mano y el error de redondeo es más difícil
   de acotar.
3. **HSI16 es un RC interno**, ±1%. Sobra para I²C, que tolera desviación de reloj sin
   problema: el esclavo se sincroniza con SCL, no tiene reloj propio.

El HSI16 hay que **encenderlo explícitamente** (`RCC->CR |= RCC_CR_HSION`) y esperar
`HSIRDY` antes de enrutarlo. Está apagado si el sistema arrancó desde HSE, que es
exactamente nuestro caso.

### Detalle 2 — Anatomía del TIMINGR

`TIMINGR` junta cinco campos que definen la forma de onda de SCL. Nuestro valor es
`0x30420F13`:

| Campo | Bits | Valor | Significado | Cuenta |
|---|---|---|---|---|
| `PRESC` | 31:28 | `0x3` | preescalador del kernel clock | $t_{PRESC} = (3{+}1)/16\,\text{MHz} = 250$ ns |
| `SCLDEL` | 23:20 | `0x4` | setup de datos antes del flanco | $(4{+}1)\cdot250 = 1250$ ns |
| `SDADEL` | 19:16 | `0x2` | hold de datos tras el flanco | $2\cdot250 = 500$ ns |
| `SCLH` | 15:8 | `0x0F` | duración del nivel alto | $(15{+}1)\cdot250 = 4.00$ µs |
| `SCLL` | 7:0 | `0x13` | duración del nivel bajo | $(19{+}1)\cdot250 = 5.00$ µs |

El periodo resulta $t_{SCL} \approx 5.00 + 4.00 = 9.0$ µs más subida/bajada, o sea cerca de
100 kHz. La asimetría (bajo más largo que alto) no es capricho: el estándar Standard Mode
exige $t_{LOW} \ge 4.7$ µs y $t_{HIGH} \ge 4.0$ µs, porque el flanco de subida es lento —lo
hace el pull-up cargando la capacitancia del bus, no un transistor— y hay que darle tiempo.

Elegimos 100 kHz aunque el AS5600 llega a 1 MHz (Fast Mode Plus). Criterio de siempre: la
versión lenta y robusta primero.

**`TIMINGR` solo se escribe con `PE = 0`.** Por eso `i2c1_init()` limpia `PE`, escribe el
timing, y recién entonces habilita. Mismo patrón "enable al final" del TIM1 y del UART.

### Detalle 3 — Los pull-ups que NO pusimos, y la red oculta de J8

La decisión en el código es dejar `PUPDR = 00`, sin pull interno. El razonamiento: el módulo
AS5600 está alimentado a **5 V** y trae sus propios pull-ups a 5 V. Los pines PB6/PB7/PB8 son
**5V-tolerant** (tipo FT), así que ver 5 V en reposo no los daña. Pero si activáramos el
pull-up *interno*, ese va a **VDD = 3.3 V**: tendríamos la línea en reposo a 5 V por un lado
y una resistencia hacia 3.3 V por el otro, o sea un camino de corriente permanente desde el
bus hacia el riel de 3.3 V atravesando el pin. Funciona —los FT lo aguantan— pero es
exactamente el tipo de detalle que degrada un pin con el tiempo.

**Regla general: en un bus de 5 V, los pull-ups los pone el lado de 5 V, nunca el micro de
3.3 V.**

Lo que no sabíamos hasta esta sesión es que **la placa no conecta J8 directo al micro**.
Leyendo el esquemático MB1419 (hoja 5, bloque `HALL/ENCODER SENSOR`), cada línea lleva:

```
J8 pad ──[ R74/R75/R77 = 1.8 kΩ serie ]──┬── R71/R72/R73 = 10 kΩ pull-up
                                          ├── D17/D18/D19 = BAT30 (clamp Schottky)
                                          ├── C67/C68/C69 = 10 pF
                                          └── PB6 / PB7 / PB8
```

Esa red está pensada para **sensores Hall con salida push-pull**, no para un bus
open-drain. El resistor en serie forma un divisor con el pull-up del módulo y limita cuán
abajo puede llevar el maestro la línea **vista desde el esclavo**. Cuando el STM32 hunde su
pin a ~0.2 V, del otro lado del 1.8 kΩ la línea se queda en:

| Pull-up del módulo | V que ve el AS5600 en un '0' | Veredicto (V_IL de 5 V = 1.5 V) |
|---|---|---|
| 10 kΩ | 0.93 V | ✅ holgado |
| 4.7 kΩ | 1.53 V | ⚠️ al filo |
| 2.2 kΩ | 2.25 V | ❌ el esclavo nunca ve el cero |

**Lo contraintuitivo:** si el bus fallara por esto, la solución es **debilitar** los pull-ups
del módulo, no reforzarlos. El reflejo normal en I²C es bajar la resistencia; aquí eso
empeora el divisor.

En nuestro caso la red **no** resultó ser el problema —a 100 kHz pasa limpia, y el bit-bang
a 20 kHz también— pero queda documentada porque es la primera sospechosa el día que
subamos a Fast Mode Plus.

### Detalle 4 — La transacción de dos fases: repeated-START

Leer un registro de un dispositivo I²C no es una operación, son dos pegadas. Hay que decir
*qué* registro se quiere (una escritura) y después *leerlo*, sin soltar el bus en medio. Si
lo soltáramos, otro maestro podría meterse y mover el puntero.

La maniobra se llama **repeated-START**:

**Fase 1 — escribir el puntero.** `CR2` se escribe **de un golpe**, no con `|=` acumulativos,
para que todos los campos queden en estado conocido: `RD_WRN = 0` (escritura) y
`AUTOEND = 0` (no mandes STOP al terminar) caen implícitos en la asignación. El `addr7 << 1`
coloca la dirección de 7 bits en `SADD[7:1]`; el bit 0 lo llena el periférico con el sentido
de la transferencia.

Luego se esperan dos flags en orden: **`TXIS`** sube cuando el esclavo hizo ACK a la
dirección y `TXDR` está libre; **`TC`** sube cuando ese byte salió y, con `AUTOEND = 0`, el
bus queda **en hold** — SCL retenido, nadie más puede hablar.

**Fase 2 — leer.** Un `START` sobre un bus retenido **no** es un START nuevo: el hardware
emite un repeated-START. Ahora `RD_WRN = 1` y `AUTOEND = 1`, y el periférico manda el STOP
solo al completar los `n` bytes. Cada byte se recoge cuando `RXNE` sube; al final se espera
`STOPF` y se limpia por `ICR` — si no se limpia, la siguiente transacción arranca sobre un
flag viejo.

El detalle que abarata esto: **el AS5600 auto-incrementa su puntero interno**. Pedir 2 bytes
desde `0x0C` devuelve `0x0C` y luego `0x0D` en una sola transacción.

### Detalle 5 — RAW ANGLE (0x0C) y no ANGLE (0x0E)

El AS5600 expone el ángulo en dos sitios y elegimos el crudo:

- **`RAW ANGLE` (0x0C/0x0D)** — la medición directa, 12 bits, 0..4095.
- **`ANGLE` (0x0E/0x0F)** — la misma tras un filtro interno y el escalado por `ZPOS`/`MPOS`.

Queremos el crudo por **latencia** (el filtro interno añade retardo que no controlamos ni
documenta bien el datasheet, y en un lazo a 50 kHz cada microsegundo de retardo de fase se
paga en el margen del predictor) y porque **filtrar es trabajo nuestro**: el observador y la
extrapolación de θ necesitan el dato sin procesar para no filtrar dos veces.

La conversión a grados aprovecha que 4096 es potencia de dos:

```c
uint32_t deg10 = ((uint32_t)raw * 3600U) >> 12;   /* raw · 3600 / 4096 */
```

Grados×10 en entero, sin float. `raw · 3600 ≤ 4095 · 3600 < 2^24`, cabe holgado en 32 bits.
Coherente con la decisión de punto fijo para todo el path de control.

También leemos **`STATUS` (0x0B)**, que trae tres bits del control automático de ganancia:
`MD` (imán detectado), `ML` (demasiado lejos), `MH` (demasiado cerca). Es el diagnóstico
mecánico del montaje: si `MD = 0`, el problema no es el firmware, es que el imán no está
donde debería.

---

### La trampa — AF4 en PB6 no es I2C1_SCL

Aquí está la parte que costó la sesión.

**Síntoma:** al arrancar, el firmware imprimía las dos primeras líneas y moría. Con el
driver instrumentado, el cuadro era: `SCL(PB6) = LOW` en reposo, `SDA(PB7) = HIGH`, y las
112 direcciones del barrido dando **timeout**.

Un bus I²C en reposo debe estar alto en las dos líneas. Una clavada abajo apunta a corto,
esclavo reteniendo el bus, o falta de pull-ups. Ninguna de las tres era.

**Lo que descartamos, en orden, y con qué:**

| Hipótesis | Prueba | Resultado |
|---|---|---|
| Cable roto o módulo sin alimentar | PB6 como GPIO **entrada pura** | `HIGH` — el cable está sano |
| Corto a GND | Pull-up interno de ~40 kΩ en PB6 | Sube a `HIGH` — no hay camino de baja impedancia |
| La red de 1.8 kΩ de J8 | Es simétrica en las tres líneas | SDA reposa bien → no explica la asimetría |
| Kernel clock ausente | Probar las tres fuentes (PCLK1 / SYSCLK / HSI16) | Las tres dan `LOW` |
| Periférico en reset o sin clock gating | Volcado de registros | `I2C1EN=1`, `I2C1RST=0`, `TIMINGR` coincide exacto |
| El pin no puede manejar la línea | Bit-bang: hundir y soltar | Ambos `OK` |
| El AS5600 está muerto | I²C por software, con el periférico fuera | **`ACK`, `MD=1`, ángulo válido** |

Después de eso quedaba un hecho incómodo: el periférico sujetaba SCL con reloj presente,
`PE = 1`, `BUSY = 0` y sin transferencia pedida. Nada de eso es comportamiento documentado.

**La prueba que lo resolvió** fue comparar PB6 con **PB8**, que cuelga del pad `Z+/H3` de J8
—o sea, **exactamente la misma red**— y estaba libre:

```
[scltest] PB6(AF4)=LOW   PB8(AF4)=HIGH
```

Y luego, con SDA todavía conectado en PB7:

```
[auto] SCL en PB6: reposo=LOW  probe=TIMEOUT
[auto] SCL en PB8: reposo=HIGH probe=NACK
```

Ese **`NACK`** es la prueba. Para llegar al bit de acuse hay que generar nueve pulsos de
reloj; el `TIMEOUT` de PB6 significa que la transacción ni arrancaba. Con SCL en PB8 el
periférico generaba reloj de verdad y solo faltaba que el cable del sensor llegara ahí.

Movido el cable del pad 1 al pad 3, cerró:

```
[auto] SCL en PB8: reposo=HIGH probe=ACK
[scan]   ACK en 0x36   <-- AS5600
[as5600] STATUS=0x20  MD=1 ML=0 MH=0
```

**Causa raíz: `AF4` en `PB6` no es `I2C1_SCL` en el STM32G431.** Es otra función, y como
está inactiva emite 0 — con el pin en open-drain, eso hunde la línea permanentemente.

El barrido de AF0..AF15 dejó además una lectura útil: los AF **sin asignar** liberan el pin
(`HIGH`), los **asignados** lo emiten en 0 (`LOW`). AF15 = `EVENTOUT` sirvió de control
positivo, y cayó del lado `LOW` como debía.

### Por qué esto ya nos había pasado

Esto es **reincidencia exacta de [N1.9](#n19--la-trampa-del-alternate-function-af-no-es-uniforme-por-periférico)**. Aquella nota cierra diciendo:

> el AF para TIM1 NO es uniforme por periférico. Es propio de cada pin individual.

Y su recomendación explícita era:

> **Para futuros pines de la placa B-G431B-ESC1**: cuando llegue I²C1 para el AS5600
> (Semana 7), verificar AF de PB6/PB7 directamente del datasheet. **No asumir nada.**

No se hizo. El código llevaba escrito en el comentario `DS12589 Table 13 (AF mapping:
PB6/PB7 = I2C1 en AF4)` una cita a una tabla **que nunca se consultó** — el DS12589 no
estaba en el repo. La cita daba la apariencia de verificación sin la verificación.

Sesión 7-8: PB15 en AF4 y no AF6, PC13 en AF4 y no AF6. Sesión de hoy: PB6 no es SCL. Mismo
error, tres pines, dos periféricos, tres meses de diferencia.

**Antídoto que sí funciona:** tener el DS12589 en `papers/` y mirar la Tabla 13 antes de
escribir la línea de `AFR`. Cuesta dos minutos y hoy costó una sesión. Segundo antídoto,
para cuando el datasheet no esté: el barrido empírico de AF que quedó en el historial de
`i2c.c` — barrer AF0..15 y buscar cuál genera reloj es determinante y toma segundos.

### Tabla maestra del I²C1 + AS5600

| Concepto | Valor | Fuente |
|---|---|---|
| **SCL** | **PB8** — pad 3 de J8 (`Z+/H3`), **AF4** | verificado empíricamente 2026-08-10 |
| **SDA** | **PB7** — pad 2 de J8 (`B+/H2`), **AF4** | ídem |
| PB6 (`A+/H1`) | **libre** — AF4 aquí NO es I2C1_SCL | ídem |
| Modo GPIO | AF, **open-drain**, sin pull interno, OSPEED alto | `i2c.c` |
| Kernel clock | **HSI16** (`I2C1SEL = 0b10`) | RM0440 §7.4.27 |
| `TIMINGR` | `0x30420F13` → ~100 kHz | `i2c.c` |
| Dirección esclavo | **0x36** (7 bits) | Datasheet AS5600 |
| Registro STATUS | 0x0B — bits MD(5) / ML(4) / MH(3) | `as5600.h` |
| Registro RAW ANGLE | 0x0C (H, bits 3:0) / 0x0D (L) → 0..4095 | `as5600.c` |
| Transacción | write 1 byte (AUTOEND=0) → repeated-START → read n (AUTOEND=1) | `i2c.c` |
| Refresco interno | ~7 kHz (~150 µs) | Datasheet AS5600 |
| Red de J8 | 1.8 kΩ serie + 10 kΩ pull-up + clamp BAT30 por línea | MB1419 hoja 5 |
| Alimentación | el pad de 5 V de J8 sale del riel lógico: **basta el USB** | verificado 2026-08-10 |

### Validación — CERRADA 2026-08-10

- ✅ `STATUS = 0x20`, `MD=1`, `ML=0`, `MH=0`. Imán detectado y a buena distancia.
- ✅ `raw` recorre 0..4095 de forma monótona al girar el rotor a mano, con un solo salto por
  vuelta mecánica (se observó el wrap 358.2° → 0.4°).
- ✅ Barrido de direcciones: exactamente un dispositivo, en 0x36, sin timeouts.
- ✅ Lectura por hardware (I2C1), no solo por bit-bang.

Dos observaciones de los datos que **no** son problemas: los valores repetidos entre
muestras son la mano quieta a 10 Hz de muestreo, y los retrocesos de 6-8 LSB (≈0.6°) son el
rotor asentándose contra el cogging al soltar. Ambos son físicos.

**Resolución angular efectiva:** 4096 cuentas por vuelta **mecánica** con 7 pares de polos
son **585 cuentas por vuelta eléctrica**, es decir 0.615° eléctricos por LSB. Para el
regresor de Fourier con H=10 el mínimo por Nyquist son 20 muestras por periodo eléctrico;
tenemos 585. **La resolución del AS5600 no va a limitar al ADALINE.** La limitante es la
latencia, como sigue.

### Para la sesión siguiente

1. **Offset entre θ del AS5600 y la fase A.** El sensor mide un ángulo mecánico absoluto,
   pero su cero no coincide con el eje magnético de la fase A. El experimento es la
   alineación con el eje d: inyectar DC en una fase, dejar que el rotor se alinee solo, y
   declarar esa posición como $\theta_e = 0$. Sin esto el ADALINE aprende una BEMF rotada y
   el par sale mal aunque todo lo demás esté bien.

2. **Extrapolación de θ entre lecturas.** Ver el cálculo de abajo.

3. **Presupuesto de la ISR completa.** Medir ADC + I²C + Clarke + M²PC + LMS juntos. Target
   < 15 µs de los 20 µs de Ts. **Es el punto de decisión de Fase 1**: si no cabe, los
   fallbacks están en N1.8 (bajar a 30 kHz, CORDIC, o cambiar el AS5600 por un
   AS5048A/AS5047P por SPI).

### El problema que viene: esta lectura no cabe en la ISR

Una lectura de ángulo cuesta:

| Tramo | Clocks de SCL |
|---|---|
| START + dirección + ACK | 9 |
| byte de registro + ACK | 9 |
| repeated-START + dirección + ACK | 9 |
| 2 bytes de datos + ACK | 18 |
| **Total** | **~45** |

A 100 kHz cada clock son 10 µs, así que **una lectura son ~450 µs de bus**. El periodo de la
ISR es 20 µs: la transacción tarda **veintidós veces más que el ciclo de control completo**.

Leer el encoder de forma bloqueante dentro de la ISR no es difícil, es **imposible**. Ni
subir a Fast Mode Plus lo arregla: a 1 MHz son ~45 µs, todavía más del doble del presupuesto
entero. Las salidas son tres y hay que combinarlas: mover la transferencia a **DMA o IRQ**
para que avance en segundo plano, leer a **la tasa real del sensor** (~7 kHz, su refresco
interno) en vez de a la de la ISR, y **extrapolar** θ en los ciclos intermedios con la
velocidad estimada.

Conviene notar que la limitación no es del todo mala: el sensor solo tiene dato nuevo cada
~150 µs, así que leerlo a 50 kHz sería pedir la misma cifra siete veces. El diseño correcto
—leer a ~7 kHz en segundo plano y extrapolar— es el que el hardware pedía desde el principio.

**Esto no es optimización: es rediseño obligatorio antes de meter el control en la ISR.**

---

## N1.17 — Bisección: cómo se depura un síntoma que mezcla cuatro cosas

### Panorama

La sesión del AS5600 se fue en depurar, no en escribir código. Vale la pena destilar el
método, porque es transferible a cualquier periférico y porque **la sesión habría durado
veinte minutos si hubiera existido esta nota**.

El síntoma inicial era un firmware que arrancaba y moría en silencio. Un solo síntoma, y
detrás cuatro capas independientes que podían fallar:

```
  [ periférico I2C1 ]  ← ¿configurado bien? ¿tiene reloj? ¿está en reset?
          │
  [ pin / alternate function ]  ← ¿el AF conecta el periférico a ESTE pin?
          │
  [ línea física ]  ← ¿cable, pull-ups, cortos?
          │
  [ esclavo AS5600 ]  ← ¿alimentado? ¿vivo? ¿la dirección es esa?
```

Mirando el síntoma de frente, las cuatro son indistinguibles. Todas producen "no lee".

### Analogía — el electricista que no cambia bombillas al azar

Una lámpara no enciende. El aficionado prueba otra bombilla, luego otro enchufe, luego mueve
el cable, y si en algún momento funciona no sabe cuál de las tres cosas era. El electricista
mide: ¿hay tensión en el enchufe? Si sí, el problema está de la lámpara hacia dentro. ¿Pasa
corriente por el cable? Si sí, es la bombilla.

Cada medición **parte el espacio de causas en dos**. Con cuatro capas, cuatro mediciones bien
elegidas bastan; probando combinaciones al azar hacen falta muchas más y al final no se sabe
por qué funcionó.

### El principio — cada prueba debe aislar UNA capa

La regla operativa: **no preguntes "¿funciona?", pregunta "¿funciona esta capa con las demás
fuera del circuito?"**.

Las cuatro pruebas que usamos, y qué capa aísla cada una:

| Prueba | Cómo saca a las demás del circuito | Qué contesta |
|---|---|---|
| Pin como **GPIO de entrada pura** | El driver de salida queda desconectado: el micro no participa | ¿Cómo está la línea *físicamente*? |
| **Pull-up interno** de ~40 kΩ y releer | Compara contra una impedancia conocida | ¿Está flotando o hay un camino de baja impedancia a tierra? |
| **Bit-bang** por GPIO | El periférico entero queda fuera | ¿Sirven el pin y la línea? ¿Vive el esclavo? |
| **Volcado de registros** | No mueve nada, solo observa | ¿El periférico está como creemos? |

Ninguna de las cuatro depende de que las otras tres funcionen. Esa independencia es todo el
truco.

### Detalle 1 — El valor de las señales que distinguen

La medición más rentable de la sesión no midió voltajes: fue notar que **`NACK` y `TIMEOUT`
son fallos distintos**.

- **`NACK`** = el maestro generó los nueve pulsos de reloj y llegó al bit de acuse. El bus
  funciona eléctricamente; simplemente nadie contestó en esa dirección.
- **`TIMEOUT`** = la transacción no completó ni la condición de parada. El bus no arranca.

Confundirlos es fatal porque llevan a diagnósticos opuestos: uno dice "revisa el esclavo",
el otro "revisa el bus". El driver original no los distinguía —se colgaba en un `while`
desnudo— así que ese bit de información no existía.

**Lección general: instrumentar para distinguir modos de fallo vale más que instrumentar
para detectarlos.** Un `while(flag){}` detecta el fallo perfectamente; no dice nada sobre
cuál es.

### Detalle 2 — El control positivo

Al barrer AF0..AF15 buscando cuál era I²C1_SCL, el resultado bruto era ambiguo: varios AF
dejaban el pin en alto. ¿Significaba que alguno era el correcto, o que estaban todos sin
asignar?

Lo que lo resolvió fue reconocer un **control positivo**: AF15 es `EVENTOUT`, una función que
sabemos que existe y que emite 0 cuando está inactiva. Salió `LOW`. Eso confirmó la
interpretación —los AF asignados hunden el pin, los libres lo sueltan— y con ella el `LOW` de
AF4 dejó de ser un misterio y pasó a ser información.

**Un experimento sin control positivo no distingue "no hay señal" de "el instrumento no
mide".**

### Detalle 3 — La comparación con red idéntica

La prueba definitiva fue comparar PB6 contra PB8. Funcionó porque **ambos cuelgan de la
misma red de J8**: mismos 1.8 kΩ en serie, mismo pull-up de 10 kΩ, mismo clamp. Los niveles
eran directamente comparables sin corregir nada.

Si hubiéramos comparado contra un pin de otro conector, cualquier diferencia habría sido
atribuible a la red y no al AF, y la prueba no habría concluido nada.

**Al buscar un control, elige el que difiera en UNA sola variable.**

### Por qué importa

1. **El orden importa tanto como las pruebas.** Nosotros probamos de barato a caro: niveles
   con el multímetro y el GPIO antes que barridos de AF, y volcado de registros antes que
   reescribir el driver. Las primeras tres pruebas costaron minutos y descartaron la mitad
   del espacio.

2. **Instrumentar es más rápido que adivinar, aunque parezca lo contrario.** Escribir
   `i2c1_probe`, `i2c1_lines` y `swi2c` tomó tiempo que "podría" haberse ido en mirar el
   osciloscopio. Pero el scope habría mostrado la línea baja sin decir quién la hunde, que
   es exactamente lo que ya sabíamos.

3. **Este patrón aplica a cualquier periférico.** SPI, UART, ADC, CAN: siempre hay un
   periférico, un mapeo de pin, una línea física y un dispositivo al otro lado. Las cuatro
   pruebas se traducen casi literalmente.

4. **El código de diagnóstico se queda.** `probe`, `scan`, `lines` y `dump_regs` viven ahora
   en `i2c.c` de forma permanente, y `swi2c.c` se conserva como camino independiente. No es
   deuda: es el instrumental. La próxima vez que el bus falle, las cuatro respuestas están a
   un reset de distancia.

### La regla en una línea

**Antes de arreglar, aísla. Antes de aislar, busca la prueba que parta el problema en dos.**

---
