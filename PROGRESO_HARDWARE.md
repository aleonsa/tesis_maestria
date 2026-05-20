# Progreso — Implementación experimental FCS-M2PC + ADALINE en B-G431B-ESC1

> Bitácora cronológica del avance. Cada entrada documenta qué se hizo, qué se aprendió y qué quedó pendiente.
> Documentos guía: [PLANNING_HARDWARE.md](./PLANNING_HARDWARE.md), `papers/`, memoria del proyecto.

---

## Sesión 1 — 2026-05-16

**Hito**: Definición del planning y descarga de bibliografía bloqueante.

**Acciones**:
- Creado `PLANNING_HARDWARE.md` con 6 fases, ~290 h totales, 2 h/día → ~8 meses.
- Aclarado rol del ADALINE: estima **forma de BEMF** (21 coef. Fourier), NO parámetros eléctricos.
- Descargados a `papers/`:
  - UM2516 (manual de la placa).
  - RM0440 (Reference Manual STM32G4).
  - DS12589 / `STM32G431X6.PDF` (datasheet del chip).
  - `MB1419-G431CB-B01_schematic.pdf` (esquemático oficial).
  - AN5325 (CORDIC en STM32).
  - AN5305 (FMAC — opcional, opción de optimización para Fase 6).
  - Kouro 2009 — paper canónico de FCS-MPC.

**Pendiente al cerrar**: confirmar pinout de J8, asignación de fases, llegada de la placa.

---

## Sesión 2 — 2026-05-17

**Hito**: Placa recibida, hardware completamente cableado, toolchain de Pi confirmado.

### Verificaciones de documentación
- **J8 (sensor) — pinout confirmado por UM2516 Tabla 4 + DS12589 Tabla 13 (Alternate functions)**:

  | Pad J8 | Pin STM32 | Función AF4 |
  |---|---|---|
  | A+/H1 | PB6 | **I2C1_SCL** |
  | B+/H2 | PB7 | **I2C1_SDA** |
  | Z+/H3 | PB8 | libre (reservado instrumentación) |
  | 5V | — | alimentación |
  | GND | — | tierra |

- **J7 (motor)**: 3 terminales OUT1/OUT2/OUT3 (mismos pads accesibles desde top y bottom de la placa).

### Cableado físico (decisiones)

- **Fases del motor → J7**:
  - Blanco → OUT1 (fase A lógica)
  - Rojo → OUT2 (fase B lógica)
  - Negro → OUT3 (fase C lógica)
- **AS5600 (integrado al motor) → J8**:
  - VCC → 5V (confirmado por usuario: módulo acepta 5V)
  - GND → GND
  - SDA → B+/H2 (PB7)
  - SCL → A+/H1 (PB6)
  - PB8 (Z+/H3) sin conectar — reservado para instrumentación con osciloscopio.
- **J5/J6 (Vbus)**: NO conectado todavía.

### Hardware adicional confirmado

- **Fuente DC**: buck-boost ZK-4KX alimentada desde fuente de PC vieja, con modos CV/CC. Apta para Vbus 5–15 V.
- **Osciloscopio**: disponible.

### Workflow de desarrollo (decisión)

- Mac corporativa con JumpCloud → bloquea conexión USB de dispositivos.
- **Solución**: Raspberry Pi 4B 8GB + Ubuntu 22.04 LTS aarch64, conectada al ST-LINK por USB. Mac edita por Zed Remote SSH.
- Bonus: la Pi queda en el banco junto a osciloscopio + fuente + motor + placa. La Mac queda en el escritorio.

### Toolchain en la Pi — verificado

| Herramienta | Versión |
|---|---|
| arm-none-eabi-gcc | 10.3.1 (2021.07-4) |
| OpenOCD | 0.11.0 |
| make | instalado |
| cmake | instalado |

- Scripts de OpenOCD disponibles: `interface/stlink.cfg` (auto-detect, lo usaremos), `target/stm32g4x.cfg`.
- Grupos del usuario: `dialout` y `plugdev` ✓.
- Reglas udev para ST-LINK: **faltan**, se crean en sesión actual.

### ST-LINK detectado por la Pi

```
$ lsusb | grep -i stmicro
Bus 001 Device 004: ID 0483:374b STMicroelectronics ST-LINK/V2.1

$ ls /dev/ttyACM*
/dev/ttyACM0
```

VID:PID `0483:374b` = ST-LINK V2.1 ✓. VCP (puerto serial virtual sobre USB) en `/dev/ttyACM0` — útil para `printf` debug más adelante.

### Reglas udev creadas

Archivo `/etc/udev/rules.d/49-stlinkv2-1.rules`:

```
SUBSYSTEM=="usb", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="374b", MODE="660", GROUP="plugdev", TAG+="uaccess"
```

Verificado: `/dev/bus/usb/001/00X` con grupo `plugdev` y modo `crw-rw----`. Acceso al ST-LINK sin sudo confirmado.

### OpenOCD ↔ STM32G431 — cadena completa funcionando

Comando: `openocd -f interface/stlink.cfg -f target/stm32g4x.cfg`

Resultado clave:
- `STLINK V2J34M25 (API v2) VID:PID 0483:374B` — firmware del ST-LINK moderno.
- `Target voltage: 3.269681` — alimentación lógica del G431 dentro de spec (3.3V nominal).
- `stm32g4x.cpu: hardware has 6 breakpoints, 4 watchpoints` — SWD funcional, Cortex-M4 responde a queries de debug.
- `Listening on port 3333 for gdb connections` — servidor GDB activo.

**Conclusión sesión 2**: hardware y toolchain validados. Listos para flashear código.

### Pendiente al cerrar sesión 2

- Bajar CMSIS Device Pack para STM32G4 (headers de registros, startup, linker script).
- Crear estructura `firmware/blink/` con CMakeLists, main.c, archivo de OpenOCD para flash.
- Compilar y flashear blink en PC6 (LED user).

---

## Sesión 3 — 2026-05-18

**Hito**: Primer firmware bare-metal compilado, flasheado y funcionando. **Fase 0 completada.**

### Vendor pack instalado

- Clone `STM32CubeG4` en `~/projects/stm32g4/vendor/STM32CubeG4/` con `--depth=1`.
- Submódulo `Drivers/CMSIS/Device/ST/STM32G4xx` inicializado vía `git submodule update --init` (solo ese, no los demás).
- Archivos clave disponibles: `stm32g431xx.h`, `startup_stm32g431xx.s`, `core_cm4.h`, linker scripts del NUCLEO-G431KB (compatibles por igual layout Flash 128KB / RAM 32KB).

### Workflow Mac→Pi establecido

- **SSH alias `raspi`** configurado en `~/.ssh/config` apuntando a `192.168.68.52`.
- **Passwordless SSH** vía `ssh-copy-id` + `id_ed25519`.
- **Script `firmware/sync_to_pi.sh`** (rsync con `--delete`, excluye `build/`). Corregido bug de array vacío bajo `set -u` en bash 3.2 macOS (cambiado a variable string).
- **Source of truth**: Mac (`~/Documents/tesis_maestria/firmware/`). Build/flash: Pi.

### Proyecto `firmware/blink/` creado

Estructura:

```
firmware/
├── .gitignore
├── sync_to_pi.sh
└── blink/
    ├── CMakeLists.txt    # cross-compile Cortex-M4 + FPU hard FPv4-SP
    ├── README.md
    └── src/
        └── main.c        # toggle PC6 con HSI 16 MHz por defecto
```

Características técnicas:
- **Bare-metal CMSIS sin HAL**: solo headers de registros + Core M4.
- **`SystemInit()` vacío** (HSI default suficiente).
- **`delay()` por NOP loop** (~0.5s a 16 MHz, no preciso — pendiente migrar a SysTick).
- **Flags compilación**: `-Og -g3 -ffunction-sections -fdata-sections --gc-sections --specs=nano.specs --specs=nosys.specs`.

### Build y flash exitosos

Footprint:

```
text       data        bss        dec        hex    filename
 880          0       1568       2448        990    blink.elf
```

- **880 bytes Flash** (0.7% de los 128 KB disponibles) — código + constantes.
- **1568 bytes RAM** (4.8% de 32 KB) — mayormente stack/heap reservados por el linker script (no uso runtime real).

Flash con OpenOCD:
- `device idcode = 0x20036468 (STM32G43/G44xx)` confirma chip.
- `Programming Finished` + `Verified OK` + `Resetting Target`.

### LED parpadeando — Fase 0 cerrada

Cadena end-to-end funcional:

```
Mac (Zed) → rsync → Pi → cmake/arm-gcc → openocd → SWD → STM32G431 → LED PC6
```

### Pendiente al cerrar sesión 3 (Semana 2 del planning)

1. **Migrar `delay()` a SysTick** — base de tiempo precisa por hardware.
2. **Configurar PLL @170 MHz** — pasar de HSI 16 MHz a la velocidad máxima del G431. Indispensable para FCS-MPC.
3. **UART debug por USART2** (PA2/PA3) — `printf` desde firmware visible en `/dev/ttyACM0` de la Pi.

Con estos tres, terminamos infraestructura básica y entramos a Fase 1 (PWM + ADC).

---

## Sesión 4 — 2026-05-18/19

**Hitos**:
- Metodología de aprendizaje formalizada en memoria persistente.
- Migración de `delay()` NOP a **SysTick** (código en Mac, falta build/flash).
- Refactor del firmware a **estructura multi-app** (`apps/01_blink/` + top-level `CMakeLists`).
- Vendor `STM32CubeG4` clonado también en Mac, espejo de la Pi.
- Setup de **`.clangd`** para que Zed/clangd en Mac resuelva los headers de CMSIS.

### Metodología: explicación línea por línea

Guardado en `memory/feedback_deep_understanding.md`. Resumen:
- Cada cambio de código se explica línea por línea, cross-ref con UM2516/RM0440/DS12589/AN5325.
- Sin código defensivo para casos imposibles, sin abstracciones prematuras.
- Macros CMSIS (`*_Msk`, `*_Pos`) se inspeccionan en su header, no se usan como caja negra.
- Verificación: hardware (LED, scope, UART log) — no solo build.

### SysTick implementado

Cambios en `main.c`:
- Variable `static volatile uint32_t g_ticks` incrementada por `SysTick_Handler`.
- `SysTick_Handler` (sin `static`) sobreescribe el weak del startup ST.
- `systick_init(ticks_per_irq)` configura `LOAD = ticks - 1`, resetea `VAL`, habilita 3 bits del `CTRL`: CLKSOURCE (AHB), TICKINT (IRQ), ENABLE.
- `delay_ms(ms)` espera por diferencia unsigned de `g_ticks` (robusto a wrap-around).
- `main()` llama `systick_init(16000)` → 1 ms tick a HSI 16 MHz.

Se diseccionó la referencia `SysTick_Config()` de CMSIS para identificar qué líneas omitir:
- ❌ Defensivo `if ((ticks-1) > MASK)` — input hardcoded, no puede fallar.
- ❌ `NVIC_SetPriority` — sin otras IRQs aún; cuando entre ADC se agrega.
- ❌ `return error` — función `void`.

**Pendiente**: build + flash + verificación con osciloscopio de período = 1.000 s en PC6.

### Refactor a estructura multi-app

Estructura nueva:

```
firmware/
├── CMakeLists.txt          # config compartida (toolchain, vendor, flags)
├── README.md
├── .gitignore
├── .clangd                 # solo para Mac/clangd
├── sync_to_pi.sh
└── apps/
    └── 01_blink/
        ├── CMakeLists.txt  # solo declara el target (5 líneas)
        ├── README.md
        └── src/main.c
```

Cambios:
- Top-level `CMakeLists.txt` movido a `firmware/` con cross-compile setup, paths del vendor, flags y `include_directories`. Usa `add_subdirectory(apps/01_blink)`.
- App-level `CMakeLists.txt` reducido a 3 líneas funcionales (declarar target, link map, post-build .bin/.hex).
- Convención de nombres: `apps/NN_<nombre>/` con prefijo numérico para marcar progresión pedagógica.
- `lib/` reservada para cuando exista código compartido entre 2+ apps (no antes).

### Vendor clonado en Mac

Mismo espejo que en la Pi para que clangd resuelva headers localmente:

```
tesis_maestria/
├── firmware/   ← edición con Zed local
├── vendor/STM32CubeG4/   ← clone con submódulo cmsis_device_g4 inicializado
└── ...
```

- `.gitignore` raíz agrega `vendor/` (no se versiona — código de terceros).
- Simetría con Pi: `~/projects/stm32g4/{firmware,vendor}` ↔ `tesis_maestria/{firmware,vendor}`.
- Bug intermedio: primer clone quedó en `firmware/vendor/` por error de `cd`. Corregido con `mv`.

### `.clangd` config

Archivo `firmware/.clangd` con `CompileFlags.Add`:
- `-DSTM32G431xx` — selecciona variante del chip dentro de `stm32g4xx.h`.
- `-I` absolutos a `vendor/STM32CubeG4/Drivers/CMSIS/{Device/ST/STM32G4xx,Core}/Include`.

Decisiones:
- Paths absolutos en lugar de relativos para evitar ambigüedad sobre el "directorio raíz" según la versión de clangd.
- Sin `--target=arm-none-eabi` ni `-mcpu` — Apple clangd 21 parsea bien sin ellos; solo se requiere si aparecen errores con macros tipo `__ARM_ARCH_7EM__`.
- **No usa `compile_commands.json`** — alternativa más manual pero suficiente para nuestro caso (sin necesidad de arm-gcc en Mac).

### Pendiente al cerrar sesión 4

1. **Confirmar que squiggles desaparecen** tras `killall clangd` + reabrir `main.c` en Zed.
2. **Sincronizar y compilar SysTick** en Pi (`./sync_to_pi.sh` + `cmake -B build && cmake --build build`).
3. **Flashear y verificar visualmente** que LED parpadea a 1 Hz.
4. **Medir período con osciloscopio** en PC6 para validar precisión del SysTick vs el viejo NOP-delay.

Después: **UART por USART2** (PA2/PA3) con `printf` retargeting a `/dev/ttyACM0` de la Pi.

---

## Sesión 5 — 2026-05-19

**Hito**: **PLL @170 MHz operativo desde HSE 8 MHz**. SysTick recalibrado y validado por prueba cruzada visual. Cierre parcial de Semana 2 del planning.

### Apertura: validaciones previas

- **Squiggles de clangd**: desaparecieron tras `killall clangd` + reabrir en Zed. `.clangd` con paths absolutos confirmado como solución estable.
- **SysTick en Pi**: build + flash OK desde sesión anterior, LED parpadeando a 1 Hz visualmente correcto.
- **Medición con osciloscopio**: pospuesta. El LED de status (PC6) es físicamente muy pequeño, difícil de puntear. PC6 no sale por ningún conector (confirmado en UM2516 Tabla 4, fila 29: PC6 → STATUS, sin pad accesible). Decisión: la validación visual a ~1 Hz es suficiente prueba de que SysTick funciona (interrupt + LOAD + delay_ms son condiciones necesarias para parpadeo periódico). Precisión absoluta del clock se medirá cuando esté UART operativa (printf de `g_ticks` vs timestamps de `/dev/ttyACM0`).

### Verificación del esquemático MB1419 — habilitar HSE

Antes de configurar el PLL desde HSE, verificación del circuito del cristal en el esquemático oficial:

- **Y2 = cristal de 8 MHz** (NO 24 MHz como se asumió en un momento).
- **R27 = resistor poblado de 220 Ω** (NO solder bridge abierto). Configuración Pierce estándar: damping resistor en serie con OSC-OUT (PF1).
  - Limita corriente de excitación del oscilador hacia el cristal.
  - Atenúa armónicos espurios.
  - Reduce EMI.
- Conclusión: **HSE conectado y operativo sin necesidad de soldar nada**.

### Configuración del PLL — clock.c

Nuevo archivo `apps/01_blink/src/clock.c` con función única `clock_init_170mhz_hse()`. Filosofía:
- 8 pasos en orden estricto (no se puede reordenar — restricciones físicas + del silicio).
- Cada paso con comentario "qué + por qué" y cross-ref a RM0440.
- Sin código defensivo: si HSE no engancha, el `while ((RCC->CR & RCC_CR_HSERDY) == 0U) { }` atrapa la ejecución para siempre → señal clara de fallo de hardware.

**Cadena de clocks elegida**:

| Parámetro | Valor | Justificación |
|---|---|---|
| HSE | 8 MHz | Cristal Y2 |
| M | 2 → encoding `PLLM = 1` | f_PLL_IN = 4 MHz ∈ [2.66, 16] |
| N | 85 | f_VCO = 340 MHz ∈ [96, 344] |
| R | 2 → encoding `PLLR = 0b00` | SYSCLK = 170 MHz |
| HPRE | /1 | HCLK = 170 MHz |
| PPRE1, PPRE2 | /1 | PCLK1 = PCLK2 = 170 MHz |

**Orden de operaciones implementado**:

1. `RCC.APB1ENR1.PWREN = 1` — clock gating del periférico PWR.
2. `PWR.CR5.R1MODE = 0` — Range 1 Boost (obligatorio para > 150 MHz).
3. `FLASH.ACR.LATENCY = 4 WS` — RM0440 Table 9. **Antes** de subir el reloj.
4. `RCC.CR.HSEON = 1`, esperar `HSERDY`.
5. `RCC.PLLCFGR = ...` — escribir todo el registro en un solo write atómico (con PLL apagado).
6. `RCC.CR.PLLON = 1`, esperar `PLLRDY`.
7. `RCC.CFGR.SW = PLL`, esperar `SWS == PLL`.
8. Prescalers HPRE/PPRE1/PPRE2 = /1 (explícitos por autodocumentación).

### Cambios a main.c

- `#include "clock.h"`.
- Llamada a `clock_init_170mhz_hse()` como **primera** instrucción de `main()`.
- `systick_init(170000U)` (antes `16000U`) → mantiene tick de 1 ms con HCLK=170 MHz.
- `SystemInit()` se mantiene vacía (decisión consciente: clock setup en main para mayor control + debugging paso a paso).

### Cambios a CMakeLists.txt

- Agregada `src/clock.c` al `add_executable(blink.elf ...)`.

### Validación cruzada SysTick ↔ PLL

Resultado: **LED parpadea a 1 Hz visual exactamente igual que antes.**

Esto es prueba simultánea de que:
- PLL multiplicó correctamente (340 MHz VCO, 170 MHz SYSCLK).
- `LOAD = 170000` con HCLK efectivo de 170 MHz da 1 ms tick.
- Si cualquier divisor (M, N, R) estuviera mal, el período cambiaría perceptiblemente.

### Continuación sesión 5 — UART USART2 + cierre Fase 0

**Corrección importante**: PA2/PA3 NO son USART en esta placa. Están ocupados por **OP1_OUT** (PA2) y **Curr_fdbk1_OPAmp-** (PA3) — frontend analógico del current sensing del shunt 1. ST routeó USART2 a **PB3/PB4** (UM2516 Tabla 4 filas 41–42). Lección de método: la Tabla 4 es la fuente autoritativa; toda asunción "PA2/PA3 son USART2" basada en defaults genéricos del G4 hay que validarla contra la tabla específica de la placa.

### Verificación del cableado VCP

- **Esquemático MB1419 hoja ST-LINK**: existen las redes `USART2_TX_ST_LINK` y `USART2_RX_ST_LINK`, conectadas a PA2/PA3 del STM32F103 (ST-LINK) vía **R23/R24 = 0 Ω** (resistores poblados de fábrica, actúan como solder bridges normalmente cerrados — desoldables si se quiere usar PB3/PB4 del G431 para otra cosa).
- **Esquemático MB1419 hoja MCU principal**: los labels exactos de las redes sobre PB3/PB4 no se renderizan legibles en el PDF (etiquetas vectorizadas en zona vacía superior del símbolo). Decisión metodológica: aceptar la evidencia indirecta (UM2516 Tabla 4 oficial + redes existentes del lado F103 + convención ST para VCP en placas Discovery/Nucleo) y validar empíricamente con el firmware.
- **Validación empírica**: se ejecutó. Resultado abajo.

### Implementación

Dos archivos nuevos en `apps/01_blink/src/`:

- **`uart.h`**: declaraciones públicas (`uart2_init`, `uart2_putc`, `uart2_puts`).
- **`uart.c`**: 7 pasos de init en polling-mode + retargeting `_write` para `printf`.

**Cadena de configuración USART2**:

1. `RCC.AHB2ENR.GPIOBEN = 1` — clock al puerto B.
2. `RCC.APB1ENR1.USART2EN = 1` — clock al USART2.
3. `GPIOB.MODER` PB3/PB4 = `0b10` (Alternate Function).
4. `GPIOB.AFR[0]` PB3/PB4 = `7` (AF7 = USART2 en G4, DS12589 Table 13).
5. `USART2.CR1 = CR2 = CR3 = 0` (estado conocido, UE=0 → registros modificables).
6. `USART2.BRR = (170e6 + baud/2) / baud` — para 115200 → 1476, error −0.02%.
7. `CR1 = TE | RE`, luego `CR1 |= UE` (UE último, obligatorio).

**Retargeting `printf`**:

```c
int _write(int fd, char *buf, int len) {
    (void)fd;
    for (int i = 0; i < len; i++) uart2_putc(buf[i]);
    return len;
}
```

`--specs=nosys.specs` (en CMakeLists top-level) provee `_write` como weak stub; nuestra definición no-weak lo sobrescribe. `setvbuf(stdout, NULL, _IONBF, 0)` desactiva el buffering interno de newlib-nano para que cada `printf` salga inmediatamente.

### Resultado del flash — VCP confirmado

`stty -F /dev/ttyACM0 115200 cs8 -cstopb -parenb raw && cat /dev/ttyACM0`:

```
[boot] STM32G431 @170MHz, USART2 OK
tick 28  uptime=14050 ms
tick 29  uptime=14552 ms
tick 30  uptime=15054 ms
tick 31  uptime=15556 ms
```

### Validación cuádruple desde una sola observación

El delta entre ticks = **502 ms** (no 500 ms exactos) confirma simultáneamente:

| Sistema | Cómo se valida | Resultado |
|---|---|---|
| PLL @170 MHz | Si f_CK ≠ 170 MHz, baud sale mal → texto sería basura | ✅ Texto legible |
| BRR cálculo | Mismo razonamiento | ✅ Error −0.02% real |
| SysTick + delay_ms | Los 500 ms del `delay_ms` se respetan | ✅ |
| UART Tx polling | Los 2 ms extra = costo de `printf` bloqueando en TXE | ✅ |

**Predicción que dio la cifra exacta**: 24 chars × 10 bits × (1/115200 s) ≈ **2.08 ms**. Coincide al milisegundo con el delta observado (502 − 500 = 2 ms). Esto demuestra que el modelo mental de UART polling es correcto y que el clock está bien calibrado.

### Estado: Fase 0 cerrada ✅

| Hito Fase 0 | Estado |
|---|---|
| Toolchain Mac→Pi→ST-LINK→G431 | ✅ Sesión 2 |
| Vendor CMSIS Device Pack | ✅ Sesión 3 |
| Blink LED PC6 | ✅ Sesión 3 |
| Refactor multi-app + .clangd | ✅ Sesión 4 |
| SysTick 1 ms | ✅ Sesión 4–5 |
| PLL HSE → 170 MHz | ✅ Sesión 5 |
| UART2 + `printf` por VCP | ✅ Sesión 5 |

**~30 h presupuestadas para Fase 0 → realizadas en ~4 sesiones / mucho menos tiempo del estimado**. La estimación era conservadora, lo que nos da colchón para Fase 1.

---

## Sesión 6 — 2026-05-19/20

**Hito**: **Estudio TIM1 completado + skeleton `apps/02_pwm_adc/` + refactor `lib/`**. Apertura formal de Fase 1.

### Cuaderno pedagógico `FIELD_NOTES.md` creado

Nuevo archivo en raíz del proyecto: `FIELD_NOTES.md`. Es el **libro del razonamiento** (no bitácora de avance — eso vive aquí). Cada nota sigue: panorama → analogía → detalle técnico con cross-ref al manual → por qué importa.

Backfill de **Fase 0** (6 notas): bare-metal vs HAL, toolchain Mac→Pi→ST-LINK, CMSIS y vendor packs, árbol de relojes + PLL 170 MHz, SysTick, UART + VCP.

Estudio completo de **TIM1** (5 notas, N1.3–N1.7):
- N1.3 — Counter modes (edge vs center-aligned). Decisión: **CMS = 01 (center-aligned mode 1)**, ARR = 2833 para 30.0025 kHz exactos (error +82 ppm, despreciable).
- N1.4 — Cadena contador → pin: PWM mode 1 (OCxM=0110), polaridad active-high (CCxP=0), habilitación (CCxE + MOE), preload con shadow registers (OCxPE + ARPE + EGR.UG).
- N1.5 — Complementarios + dead-time. Encoding no-lineal del DTG[7:0]: 4 sub-rangos con resolución decreciente. Para 500 ns @ CKD=00: **DTG = 0x55**. La trampa de OISx/OISxN: estado de las salidas cuando MOE=0.
- N1.6 — TRGO al ADC. Decisión: **MMS = 010 (update event)** + RCR = 1 → 1 muestra por periodo PWM. EXTSEL = 9 en ADC para escuchar TIM1_TRGO. Truco avanzado de OC4REF como TRGO programable (no se usa en Fase 1).
- N1.7 — Break inputs (BKIN/BKIN2). Decisión: **no implementar en Fase 1**. La placa no expone BKIN externo; protección por software en la ISR del ADC. Documentado para futuras iteraciones (OPAMP → COMP interno → BKIN).

Convención: cuando alguien dé una explicación pedagógica nueva, va a `FIELD_NOTES.md` con la misma estructura. Workflow guardado en memoria como `feedback-field-notes-workflow`.

### Pinout TIM1 confirmado (UM2516 Tabla 4)

| Canal | High-side | Complementario (low-side) |
|---|---|---|
| TIM1_CH1 | **PA8** | **PC13** (compartido con TAMP/RTC en otros chips; aquí dedicado al gate driver) |
| TIM1_CH2 | **PA9** | **PA12** |
| TIM1_CH3 | **PA10** | **PB15** |

Todos en **AF6** (DS12589 Tabla 13).

### Correcciones de numeración de capítulos en RM0440

El plan original tenía mal los caps. Corregido:
- TIM1 = **Cap 29** (no Cap 28). Cap 28 es HRTIM (otro timer, no aplica).
- ADC = Cap 21 ✓
- OPAMP = **Cap 25** (no Cap 24).

### Refactor a `firmware/lib/`

Como ahora hay **2 apps** que comparten clock + UART, se cumplió la condición para crear `lib/`:

```
firmware/
├── CMakeLists.txt           ← + add_subdirectory(lib) y apps/02_pwm_adc
├── .clangd                  ← + path a firmware/lib/
├── lib/                     ← NUEVO
│   ├── CMakeLists.txt       (add_library stm32g4_lib STATIC)
│   ├── clock.{c,h}          (movidos desde apps/01_blink/src/)
│   └── uart.{c,h}           (movidos desde apps/01_blink/src/)
└── apps/
    ├── 01_blink/            ← target_link_libraries(blink.elf PRIVATE stm32g4_lib)
    │   ├── CMakeLists.txt
    │   ├── README.md
    │   └── src/main.c       (sin cambios; #include "clock.h" sigue funcionando)
    └── 02_pwm_adc/          ← NUEVO
        ├── CMakeLists.txt
        ├── README.md
        └── src/
            ├── main.c       (init clock+UART, llama stubs, heartbeat 1Hz)
            ├── pwm.h        (API: pwm_init, pwm_set_duties, pwm_enable, pwm_disable + constantes PWM_ARR, PWM_DEAD_DTG)
            └── pwm.c        (stubs vacíos + plan de implementación documentado en comentario)
```

`stm32g4_lib` es STATIC library con `target_include_directories ... PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}` — los consumers heredan automáticamente el include path.

### Build limpio en la Pi

```
[ 30%] Linking C static library libstm32g4_lib.a
[ 30%] Built target stm32g4_lib
[ 60%] Linking C executable blink.elf
   text       data        bss        dec        hex
   8632        100       1588      10320       2850   blink.elf       (idéntico antes y después del refactor)
[100%] Linking C executable pwm_adc.elf
   text       data        bss        dec        hex
   8680        100       1588      10368       2880   pwm_adc.elf     (+48 bytes flash vs blink; stubs + 1 printf extra)
```

Diferencia de 48 bytes Flash = 4 funciones stub (`pwm_init`, `pwm_set_duties`, `pwm_enable`, `pwm_disable`) + un `printf("[pwm_init] STUB ...")`. Razonable.

### Skeleton arrancando — validación quíntuple por VCP

```
[boot] STM32G431 @170MHz, USART2 OK (app: 02_pwm_adc)
[pwm_init] STUB — TIM1 todavía no configurado
alive tick=0 uptime=9 ms
alive tick=1 uptime=1011 ms
alive tick=2 uptime=2013 ms
```

Validaciones simultáneas en una sola lectura:
1. **PLL @ 170 MHz** OK (texto coherente → baud correcto → SYSCLK correcto).
2. **BRR del USART2** OK.
3. **SysTick 1 ms** OK (delta entre prints = 1002 ms; los 2 ms extra son `printf alive` de 30 chars × 87 μs).
4. **Retargeting de `printf`** funciona desde el nuevo target.
5. **Refactor a `lib/`** OK — clock.c y uart.c linkean correctamente desde un .elf distinto al blink.

Predicción teórica del primer `uptime`:
- 55 chars del `[boot]...` + 51 chars del `[pwm_init]...` = 106 chars
- A 87 μs/char (115200 baud) = **9.22 ms**
- Observado: **9 ms**. Coincide al ms.

### Pendiente al cerrar sesión 6

1. **Implementar `pwm_init()` registro por registro**, con cross-ref al manual y verificación incremental con osciloscopio:
   - Clock gating (RCC.AHB2ENR + RCC.APB2ENR).
   - GPIOs en AF6 (PA8, PA9, PA10, PA12, PB15, PC13).
   - TIM1 base (CR1.CMS, CR1.ARPE, ARR, RCR).
   - Cada canal x ∈ {1,2,3}: CCMRx, CCER, CCRx.
   - BDTR: DTG = 0x55, MOE = 0 todavía.
   - CR2: MMS = 010 (TRGO), OISx/OISxN = 0.
   - EGR.UG para cargar shadows.
2. **Validar PWM con osciloscopio** sobre J7 (puntas a OUT1 vs OUT1N): complementariedad + dead-time medido + frecuencia 30 kHz.
3. **Solo entonces** continuar a Semana 5 (OPAMPs + ADC).

---

## Próxima sesión — implementar `pwm_init()`

### Punto de partida

`apps/02_pwm_adc/src/pwm.c` tiene el plan documentado como comentario (9 pasos). Cada paso corresponde a una sección del `FIELD_NOTES.md` N1.3–N1.6.

### Estrategia de verificación incremental

Después de cada paso del plan, validar antes de seguir:

| Paso | Verificación |
|---|---|
| 1. Clock gating | Leer registros RCC.AHB2ENR, RCC.APB2ENR con GDB → bits seteados |
| 2. GPIOs en AF6 | GPIOA.MODER, GPIOA.AFRH, GPIOB.MODER, GPIOC.MODER con GDB |
| 3. TIM1 base | TIM1.CR1, TIM1.ARR, TIM1.RCR con GDB. Aún sin output. |
| 4. Canales con MOE=0 | TIM1.CCMR1/2, TIM1.CCER. Habilitar CEN=1 → contador corre, pero salidas todavía silenciadas por MOE=0. |
| 5. Dead-time + MOE=1 | Punteo de scope en PA8 (CH1): aparece PWM al 50% a 30 kHz. |
| 6. Complementario | Punteo simultáneo en PA8 y PC13: forma complementaria, dead-time visible. |
| 7. TRGO | Setear OC4 con CCR4=ARR-100 + MMS=0111 temporalmente como debug, ver TRGO en CH4 → confirma timing. (Luego revertir a MMS=010.) |

### Decisión de seguridad

`pwm_init()` deja MOE = 0. La primera vez que llamemos `pwm_enable()` (que pone MOE=1) las 6 salidas se activan instantáneamente. **Antes de esa llamada, hay que tener listo**:

- Duty cycles inicializados a 50% (medio del rango, voltaje promedio cero entre fases).
- Mecanismo para apagar (botón, watchdog, IRQ del UART).
- Motor desconectado en la primera prueba (verificar PWM con scope, no con motor cargado).

### Comandos rápidos de retoma

```bash
# Mac → Pi
cd ~/Documents/tesis_maestria/firmware
./sync_to_pi.sh

# Pi (vía SSH alias `raspi`)
ssh raspi
cd ~/projects/stm32g4/firmware
cmake -B build && cmake --build build

# Flashear (OJO: el .elf vive en build/, no en el directorio fuente)
openocd -f interface/stlink.cfg -f target/stm32g4x.cfg \
        -c "program build/apps/02_pwm_adc/pwm_adc.elf verify reset exit"

# Lectura VCP (terminal aparte)
stty -F /dev/ttyACM0 115200 cs8 -cstopb -parenb raw
cat /dev/ttyACM0
```

**Nota**: la versión anterior de los comandos en este archivo omitía el prefijo `build/` y solo funcionaba si se corría openocd desde dentro de `build/`. Esta es la versión explícita.

### Estado del firmware al cierre

```
firmware/
├── CMakeLists.txt            # toolchain + vendor + add_subdirectory(lib, apps/01_blink, apps/02_pwm_adc)
├── .clangd                   # paths absolutos para Zed/clangd, incluye lib/
├── .gitignore
├── sync_to_pi.sh
├── README.md
├── lib/
│   ├── CMakeLists.txt        # add_library(stm32g4_lib STATIC ...)
│   ├── clock.{c,h}           # PLL HSE 8 MHz → 170 MHz
│   └── uart.{c,h}            # USART2 polling + retargeting _write
└── apps/
    ├── 01_blink/             # validador de Fase 0 (LED + UART), todavía operativo
    │   ├── CMakeLists.txt
    │   ├── README.md
    │   └── src/main.c
    └── 02_pwm_adc/           # ← Fase 1: TIM1 PWM + ADC sync
        ├── CMakeLists.txt
        ├── README.md
        └── src/
            ├── main.c        # arranca pero pwm_init() es stub
            ├── pwm.h         # API + constantes PWM_ARR=2833, PWM_DEAD_DTG=0x55
            └── pwm.c         # STUBS + plan documentado en comentario
```

---
