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
