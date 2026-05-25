# Progreso — Implementación experimental FCS-M2PC + ADALINE en B-G431B-ESC1

> Bitácora cronológica del avance. Cada entrada documenta qué se hizo, qué se aprendió y qué quedó pendiente.
> Documentos guía: [PLANNING_HARDWARE.md](./PLANNING_HARDWARE.md), `papers/`, memoria del proyecto.

---

## Sesión 10 — 2026-05-22/23 — ISR JEOS funcionando (Semana 6 abre)

**Hito**: lazo de control a 50 kHz montado. ISR JEOS entra cada 20 μs, lee i_a/i_b/i_c en globals, toca PB8 para instrumentación. Cierre del primer paso del plan de Semana 6.

### Implementado

- `apps/02_pwm_adc/src/adc.h` — declara globals `g_ia_raw / g_ib_raw / g_ic_raw / g_isr_count` (volatile) y la API `adc_isr_init()`.
- `apps/02_pwm_adc/src/adc.c` — `adc_isr_init()` habilita JEOSIE en ADC2, configura PB8 como output, registra el handler con prioridad 1 en NVIC. `ADC1_2_IRQHandler()` togglea PB8, lee los 3 JDR, incrementa counter, limpia flag.
- `apps/02_pwm_adc/src/main.c` — llama `adc_isr_init()` entre `adc_init()` y `pwm_enable()`. El while reporta `isr_count` y Δ por segundo.
- `FIELD_NOTES.md` N1.14 — nota pedagógica: panorama, analogía (médico de guardia con busca), detalle de JEOS, NVIC, latencia/jitter, reglas de oro para handlers, decisión "JEOSIE en ADC2 no ADC1" justificada.

### Validación

- VCP estable: `Δ=50350` ISRs entre prints separados por 1.007 s → **50000 Hz exactos**.
- `ADC2.IER=0x40` (bit 6 JEOSIE) post-init ✓.
- Lecturas raw consistentes con sesión 9: `i_a/i_b/i_c ≈ 317-319`, `Vbus ≈ 552`.
- Multímetro DC en pad Z+/H3 del J8: **44 mV** → consistente con pulso de ~270 ns cada 20 μs (duty ~1.35%, coherente con un handler de ~45 ciclos del Cortex-M4 a 170 MHz). **Evidencia indirecta, no scope explícito**.
- Calibración de offset implementada (state machine en handler, N=4096, división por shift right 12). Offsets resultantes: `i_a=317, i_b=317, i_c=319` (diferencia max 2 raw → OPAMPs bien trimmed de fábrica).
- Lecturas compensadas (`cal`) estables en ±1 raw → dentro del ruido de cuantificación del ADC.
- **Sanity check con motor conectado**: consumo Vbus estable en **80 mA** (baseline antes y después del motor conectado, idéntico). Motor quieto, sin vibración, lecturas `cal` siguen en ±1.
- **Baseline de consumo Vbus a anotar** (post-fix del bug AF de sesión 8): **80 mA** con todo configurado (TIM1 corriendo + ADC + ISR + motor conectado, duty 50% balanceado). Cualquier desviación significativa en sesiones futuras es señal de problema.

### Drama del scope: bug que no era

Pasamos buena parte de la sesión persiguiendo "PB8 no se ve en scope". Razón real: pulso de 270 ns es demasiado angosto para los settings que probamos primero (5+ μs/div). El multímetro DC en 44 mV fue la pista que cerró el caso. Lección persistida en N1.14 sección "Sospechas pendientes" — si en algún momento futuro el control no converge, volver a validar visualmente con scope a 1-2 μs/div + auto trigger.

### Limpieza de código

Quitado el bloque temporal `[diag PB8]` del `main.c` (dump de GPIOB.MODER/OTYPER/AFR/PUPDR/ODR + blink lento). El código vuelve a su forma "limpia": init → enable → while loop con prints.

### Para sesión 11 — Semana 6 sigue

Plan original mantenido:

1. **Calibración de offset DC** — 1000 muestras motor off, promediar, guardar `i_offset_a/b/c`. Esto convierte el ~318 raw en "i = 0 sin corriente". Lo más simple: hacer la calibración dentro del while al boot, antes de habilitar el control.
2. **Calibración de ganancia raw → A** — inyectar corriente DC conocida con fuente bench → leer raw → escala.
3. **Medir tiempo de ISR con GPIO toggle + scope**. Pendiente del scope explícito de PB8.
4. **Decisión OPAMP topology** (PGA + offset SW vs standalone con bias del PCB) — la calibración da los datos para decidir.
5. **Dead-time empírico** (pendiente de sesión 8).

### Sesión 11 — 2026-05-23 — Ganancia teórica + PGA x16

**Hito**: cadena `raw → mA` lista con K teórico. Listo para excitación open-loop / test de motor.

**Decisiones**:

- **R_shunt confirmado**: 0.003 Ω (R54, R55, R56 según esquemático MB1419 página "SHUNT RESISTOR", 3 W). Los JP1/2/3 son jumpers de bypass, dejados abiertos = shunts activos.
- **PGA cambiado x2 → x16** en `adc.c` (PGGAIN=00011). Razón: con x2, sensibilidad era 7.45 raw/A — el sweep de 0-1 A daba apenas 7 raw, dominado por ruido. Con x16, 59.6 raw/A → 60 raw/A en 1 A, SNR razonable.
- **No subimos a x32**: el offset DC del front-end (318 raw con x2, ≈ 256 mV en VINP_equiv) escala con la ganancia. x32 lo llevaría a ~5000 raw → SATURA el ADC. x16 lo deja en ~2540 raw, queda headroom +26 A.
- **Constante K hardcoded en adc.h**: `ADC_FACTOR_RAW_TO_MA_Q12 = 68735` (= round(16.78 × 4096)). Inline `adc_raw_to_ma(int16_t) → int32_t` hace `(raw × FACTOR) >> 12`. 100% entero, ~3 ciclos por conversión.
- **Sweep de calibración física DEJADO en main.c, activable cambiando `#define CAL_TARGET_PHASE`** entre -1 (off, normal), 0/1/2 (calibrar A/B/C). Default = -1.

**Pendientes que registro explícitamente**:

1. **Validar K teórico vs medido** — el cálculo tiene ~5-8% de error teórico (tolerancias de R_shunt, G_PGA, V_ref). Para tesis riguroso: hacer al menos un sweep físico con multímetro en serie en una fase y validar. Si difiere < 10%, confiar en teórico para las 3 fases. Si difiere más, calibrar cada una por separado.
2. **Origen del offset DC alto (~256 mV con PGA x2)** — no esperado para input flotante / shunt sin corriente. Hipótesis: pin VINP del OPAMP conectado a un bias network del PCB no documentado en UM2516. Sin investigar todavía. Si el control no converge bien, esto puede ser un factor.
3. **Limitación bipolar conocida** — el offset alto restringe rango negativo. Corriente max negativa ≈ -42 A (no problema para nosotros), positiva ≈ +26 A (tampoco). Pero la asimetría puede afectar zero-crossing de corrientes AC. Para FCS-M2PC eventualmente: investigar bias correcto del PCB, o switch a standalone mode con bias a Vref/2.

### Para sesión 12 — qué sigue

Con `raw → mA` listo, las opciones son:

1. **Excitación open-loop básica** — generar 3 sinusoides desfasadas 120° en el handler, baja amplitud (5-10% duty), frecuencia eléctrica baja (e.g. 2 Hz). Confirmar que el motor gira. **Es el primer test que mueve el motor.**
2. **Validación de K con sweep físico** (multímetro en serie) — ya quedó armado el `CAL_TARGET_PHASE`.
3. **Medir tiempo de ISR con scope** (con settings agresivas, 1-2 μs/día).
4. **Dead-time empírico**.

El usuario quiere "conectar el motor" — opción 1 es el camino natural.

### Sesión 12 — 2026-05-23 — Motor gira por primera vez + paradoja de medición

**Hito**: el motor giró por primera vez 🎉. Datos confirman P=7 (no P=8 como decía CLAUDE.md). Pero quedó abierta una paradoja: las mediciones de corriente no parecen reflejar la corriente real del motor — siguiente sesión es debug.

#### Implementado

- **Módulo `openloop.{h,c}`**: excitación trifásica sinusoidal en open-loop. LUT Q15 de 256 entradas (generada offline con Python para no depender de libm). Theta Q32 con overflow natural = vuelta eléctrica. Desfase 120° por offset entero en índice (85, 170). Aritmética 100% entera.
- **Hook en handler**: `openloop_step()` llamado desde `ADC1_2_IRQHandler` después de leer JDR. Coste extra ~80-100 ciclos cuando running, ~3 ciclos cuando no.
- **Stats pico+RMS**: acumuladores `max(|i|)` y `Σi²` en uint64_t dentro del handler. Snapshot a globals cada N=50000 muestras (1 segundo). RMS calculado en main con `isqrt32` (entera, sin float). Coste extra del handler ~25 ciclos.
- **`adc_isqrt32`**: sqrt entera digit-by-digit base 4, ~20 ciclos M4.

#### Validaciones y descubrimientos

1. **Motor gira** a f_e=2 Hz: 3.5 s/vuelta → confirma f_m = 0.286 Hz → **P=7 (14 polos, 7 pares)**, NO P=8 como decía CLAUDE.md. A f_e=10 Hz: 0.65 s/vuelta → f_m = 1.54 Hz (predicho 1.43 Hz, 7% error del cronómetro).
2. **CLAUDE.md actualizado** con sección "Motores en el proyecto" — distinción Anaheim (paper, P=8, trap) vs 2804 (banco, P=14, sin). Memoria del banco también actualizada.
3. **R_fase medida = 2.7 Ω** (R_línea_línea = 5.4 Ω, Y-connection). Confirmó interpretación del datasheet x-teamrc.
4. **AS5600** confirmado por el usuario (datasheet x-teamrc menciona AS5048A pero el motor real tiene AS5600).
5. **Bug latente en `adc_get_vbus_raw`**: race condition entre conversiones regular e inyectada del mismo ADC. Sesión 11 mostraba Vbus_raw=552 (no físico, 4.6V); sesión 12 muestra Vbus_raw=1443 (= 11.6V real con divider 0.0963). El handler más lento "movió" el timing accidentalmente. Documentado en FIELD_NOTES N1.15 + Task #13.

#### Paradoja abierta — mediciones de corriente NO siguen al motor

Con `amp=170` (10% PWM) y motor girando:
- A 2 Hz: pico_raw=5-6, RMS_raw=1, consumo Vbus=127 mA.
- A 10 Hz: pico_raw=5-6, RMS_raw=1, consumo Vbus=123 mA.

**Mismas magnitudes a 2 y 10 Hz** — debería haber cambiado si midiéramos corriente real del motor. Si fuera sinusoide pura, Pico/RMS = √2; medimos Pico/RMS ≈ 5. **Lo que estamos midiendo NO refleja la corriente del motor.**

Hipótesis a evaluar en sesión 13 (en orden de probabilidad):
- **(A)** TRGO dispara en valle del PWM (high-side ON, shunts ven 0) en lugar del pico. RM0440 §29.4.5 no es 100% explícito sobre dónde cae el UEV con RCR=1 en center-aligned.
- **(B)** OPAMP con PGA x16 + sample time 6.5 ciclos (153 ns) no se estabiliza (tau OPAMP ≈ 200 ns con BW 0.8 MHz). Solución: subir SMP a 47.5 ciclos.
- **(C)** Ruido térmico/drift del front-end dominando sobre señal real (que es pequeña porque la corriente del motor no es tan grande).

#### Test definitivo propuesto (no ejecutado, para sesión 13)

Variar `OPENLOOP_AMP_TICKS` entre {0, 170, 510} y medir Pico/RMS de cada caso:
- amp=0: si pico>0, hay un baseline de ruido/drift.
- amp=170 (10%): pico esperado ~18 raw si medición OK.
- amp=510 (30%): pico esperado ~53 raw si medición OK.

Si las stats son **idénticas** entre los 3 casos → medición rota.
Si escalan proporcionalmente → medición OK pero corriente real es baja (problema distinto).

#### Estado del firmware al cierre

- `apps/02_pwm_adc/src/main.c`: OPENLOOP_ENABLE=1, AMP=170, DELTA=10 Hz_e.
- `apps/02_pwm_adc/src/openloop.{c,h}`: LUT + step + start/stop.
- `apps/02_pwm_adc/src/adc.{c,h}`: ISR + cal offset + stats + isqrt32.
- Sweep de ganancia armado pero desactivado (`CAL_TARGET_PHASE=-1`).

#### Tasks abiertas al cierre de sesión 12

- Task #8: validar K teórico con multímetro físico (no urgente).
- Task #9: investigar origen del offset DC alto (256 mV con PGA x2 = anómalo).
- Task #13: refactor adc_get_vbus_raw (race condition).
- Nuevas implícitas para sesión 13:
  - Test amp=0/170/510 para discriminar medición rota vs corriente real baja.
  - Si medición rota: probar SMP=47.5 ciclos primero (cambio menor); si persiste, cambiar MMS para forzar TRGO al pico.

### Para sesión 13 — punto de partida concreto

**Primera acción al sentarse**: implementar el test de 3 amps (0, 170, 510) con prints separados por amp, idealmente en un mismo flash con cambio automático cada 30 s. Eso descarta o confirma la hipótesis de medición rota en ~2 minutos.

Si medición rota:
1. Subir SMP del ADC de 6.5 a 47.5 ciclos (cambio de 1 línea en `adc.c`, paso 7 del adc_init). Re-test.
2. Si sigue mal: cambiar MMS a OC4REF con CCR4=ARR-1 para forzar TRGO explícitamente en el pico. Re-test.
3. Si sigue mal: instrumentar con scope. Mirar OUT1 vs PB8 (toggle del ISR) para inferir cuándo se hace el sampling.

Si medición OK pero corriente baja:
1. Calcular potencia disipada — debe cuadrar con consumo Vbus.
2. Considerar que la corriente real del motor outrunner pequeño con poca fricción es genuinamente baja.

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

## Revisión de f_PWM al cierre de Sesión 6 — 50 kHz en lugar de 30 kHz

**Decisión revisada el 2026-05-20**, antes de implementar registros del TIM1. Análisis cuantitativo de 6 factores (rizado de corriente, constante eléctrica, presupuesto computacional, AS5600, switching losses, dead-time fraccionario) → 50 kHz balancea mejor que 30 kHz. **Detalle completo en FIELD_NOTES.md N1.8.**

Resumen del veredicto:
- A 30 kHz, rizado de corriente = 92% relativo (borderline insuficiente para una L=0.86 mH).
- A 50 kHz, rizado baja a 56% y `ARR = 1700` cae exacto (vs 2833.33 → +82 ppm a 30 kHz).
- Presupuesto computacional: 3400 ciclos de CPU disponibles por ISR @ 50 kHz — cabe según literatura para FCS-M2PC (~1500-2500 ciclos típicos), pero hay que vigilar en Semana 6.
- Fallbacks si Semana 6 muestra que no cabe: bajar a 30 kHz, o cambiar AS5600 → AS5048A/AS5047P (SPI, ~6× ancho de banda).

Cambios concretos:
- `apps/02_pwm_adc/src/pwm.h`: `PWM_ARR` = 1700 (era 2833). Comentario explica el cálculo y referencia a N1.8.
- `FIELD_NOTES.md` N1.3 actualizada con el nuevo cálculo de ARR.
- `FIELD_NOTES.md` N1.8 nueva, dedicada al análisis cuantitativo de los 6 factores.

---

## Sesión 9 — 2026-05-20/21 — Bring-up ADC + OPAMP funcionando ✅ Cierre Semana 5

**Hito**: cadena completa **shunt → OPAMP → ADC → memoria** funcionando con motor desconectado. 3 corrientes leen valores consistentes (~318 raw c/u), trigger TIM1_TRGO disparando, dual injected simultaneous operativo.

### Lo que funciona

- `apps/02_pwm_adc/src/adc.{c,h}` implementado con todas las funciones (opamp_init, adc_init, get_currents_raw, get_vbus_raw).
- OPAMP1/2/3 en PGA mode interno gain x2, OPAMPINTEN=1 (salida directa a canales ADC internos 13/16/18).
- ADC1+ADC2 en dual injected simultaneous (DUAL=00101).
- Trigger inyectado: TIM1_TRGO en rising edge → conversiones a 50 kHz exactos.
- Vbus en canal regular ADC1_IN1, lectura ~553 raw = ~12 V correcto.
- Sample time 6.5 ciclos, resolución 12 bits.

### 3 bugs encontrados y resueltos durante el bring-up

Todos documentados en FIELD_NOTES.md N1.13:

1. **SYSCFG clock no habilitado** → writes a OPAMP_CSR se ignoraban silenciosamente (devolvían 0 al leer). Fix: `RCC->APB2ENR |= RCC_APB2ENR_SYSCFGEN`.

2. **JEXTSEL tiene tabla distinta a EXTSEL** (RM0440 Tabla 167 vs 162). Para TIM1_TRGO inyectado, JEXTSEL = 0x00, NO 0x09. Fix: cambiar valor en JSQR.

3. **Standalone mode sin topología clara del PCB** → OPAMPs saturaban a rail. Switch a PGA mode interno gain x2 para tener feedback definido internamente, no depende del board layout.

### Limitaciones aceptadas para esta iteración

- **PGA gain x2 sin bias a Vrefint/2** → solo medimos corrientes **positivas**. Para AC bipolar de FCS-M2PC, eventualmente: volver a standalone con bias correcto del PCB, o mantener PGA + offset DC restado en software.
- **Ganancia raw → amperios** no calibrada (pendiente Semana 6).
- **Offset DC ~318 raw** no compensado (pendiente Semana 6).

### Dead-time empíricamente NO verificado

Quedó pendiente de sesión 8: medir el dead-time programado (500 ns, DTG=0x55) con scope en el flanco de OUT2. Validar empíricamente queda como TODO para Semana 6 también.

### Para sesión 10 — Semana 6

Plan:
1. ISR JEOS (end of injected sequence) → callback a 50 kHz donde vivirá el FCS-M2PC.
2. Calibración de offset: 1000 muestras motor off → promediar → guardar `i_offset_a/b/c`.
3. Calibración de ganancia: inyectar corriente conocida (DC con fuente bench externa) → medir raw → calcular escala raw → A.
4. Medir tiempo de ISR con GPIO toggle + scope. Target: < 15 μs de los 20 μs disponibles a 50 kHz.
5. **Decisión final OPAMP topology**: PGA + offset SW vs standalone con bias del PCB.
6. (Si tiempo) verificar dead-time empíricamente.

---

## Sesión 8 — 2026-05-20 — Bug AF resuelto ✅ las 3 fases conmutan

**Hito**: el bug que dejó la sesión 7 abierta fue resuelto en ~30 minutos al inicio de la sesión 8. Las 3 fases del puente trifásico generan PWM 50 kHz, duty 50%, simétricas. **Cierre completo del bring-up del PWM.**

### Root cause

DS12589 Tabla 13 (Alternate Function table) es **por-pin**, no por periférico. Cada pin tiene su propio mapeo de qué función está en cada AF0–AF15. ST distribuye el TIM1 en distintos AFs según el pin:

- En GPIOA: TIM1_CH1/2/3/CH2N en AF6 (PA8, PA9, PA10, PA12).
- En GPIOB: TIM1_CH3N en **AF4** (PB15).
- En GPIOC: TIM1_CH1N en **AF4** (PC13).

Mi código en `pwm_init()` asumía AF6 universal. Resultado: PC13 quedó routeado a TIM8_CH4N (función no usada → output indefinido), y PB15 quedó en una AF sin función específica → ambos low-sides nunca recibían PWM → bootstrap caps de fase A y C nunca se cargaban → high-side tampoco conmutaba → output flotante en ~8V.

### Fix

Dos líneas modificadas en `apps/02_pwm_adc/src/pwm.c`:

```c
gpio_set_af(GPIOB, 15U, 4U);   // PB15 CH3N → AF4 (era 6)
gpio_set_af(GPIOC, 13U, 4U);   // PC13 CH1N → AF4 (era 6)
```

### Validación

- Build limpio, 0 warnings.
- Dump por VCP confirma `PB15 AFR=4` y `PC13 AFR=4`.
- Scope muestra las 3 OUTs (OUT1/OUT2/OUT3 en J7) con PWM idéntica: 50 kHz, duty 50%, amplitud 0–12V.
- Consumo de la fuente: ~30–50 mA estable.

### Lección persistida

`FIELD_NOTES.md` N1.9 — "La trampa del Alternate Function". Incluye:
- Tabla maestra de AFs para los 6 pines TIM1 de la placa.
- Meta-lección: los dumps de validación esconden bugs si el "expected" viene del mismo modelo mental erróneo del código.
- Recomendación: para pines futuros (I²C1 PB6/PB7 del AS5600), verificar AF directamente del datasheet pin por pin.

### Tareas cerradas

- Task #3 — Implementar TIM1 50 kHz center-aligned + 6 PWMs ✅
- Task #4 — TIM1 TRGO en update-event ✅ (parte del mismo pwm_init, validado por dump CR2=0x20).

### Próximo paso — Semana 5 del planning

Avanzar a **OPAMP1/2/3 + ADC1/ADC2 dual simultaneous** (Task #5). Lectura previa requerida:
- RM0440 Cap 21 (ADC) — JEXTSEL=TIM1_TRGO para sincronizar con pico/valle del PWM.
- RM0440 Cap 25 (OPAMP) — configuración como PGA con feedback externo.
- UM2516 — ruta de shunts → OPAMP → ADC en MB1419.

Crear nota N1.10 en FIELD_NOTES (estructura ADC con sus secciones panorama → analogía → detalle → por qué importa) antes de tocar código.

---

## Sesión 7 — 2026-05-20 — `pwm_init()` implementado, bug abierto

**Hito**: pwm_init() implementado y verificado por dump exhaustivo, pero **solo fase B (OUT2) conmuta**; OUT1 y OUT3 quedan flotantes en ~8V.

### Verificación del firmware (positivo)

Dump completo por VCP confirma:
- 12 registros TIM1 con valores esperados (CR1=0xB1 con bit DIR variable, CR2=0x20, ARR=1700, RCR=1, PSC=0, CCMR1=0x6868, CCMR2=0x68, CCER=0x555, BDTR=0x8C55 post-enable, CCRx=850).
- 6 pines GPIO con MODER=AF y AFR=6: PA8, PA9, PA10, PA12, PB15, PC13.
- Clock gating: RCC->AHB2ENR=0x07 (GPIOA/B/C), RCC->APB2ENR=0x800 (TIM1).
- CNT cambia entre prints → counter corriendo a 50 kHz.

### Comportamiento del puente (negativo)

- **OUT2** (CH2 = PA9 + PA12, ambos en GPIOA): PWM cuadrada limpia, 50 kHz, duty como configurado. ✅
- **OUT1** (CH1 = PA8 + PC13): output flotante en ~8 V constante. ❌
- **OUT3** (CH3 = PA10 + PB15): output flotante en ~8 V constante. ❌

Patrón: la única fase que funciona tiene **high-side Y low-side ambos en GPIOA**. Las que fallan tienen el low-side en otro puerto (GPIOB o GPIOC).

### Vbus operacional

- Fuente buck-boost ZK-4KX configurada CV=12V CC=500mA.
- Vbus medido en placa: 12 V estables.
- Consumo: ~30 mA estable (placa + L6387 quiescente). No entra en CC.

### Hipótesis priorizadas para sesión 8

1. **DBP backup domain** (PC13): falta habilitar acceso al backup domain antes de configurar PC13.
   ```c
   RCC->APB1ENR1 |= RCC_APB1ENR1_PWREN;
   PWR->CR1 |= PWR_CR1_DBP;
   ```
2. **RTC/TAMP**: leer `RTC->TAMPCR` y `RTC->CR` para descartar posesión de PC13.
3. **SYSCFG**: leer registros de SYSCFG para detectar routing especial de PB15 (también puede ser USB_DP en AF12).
4. **Reset state**: dumpear MODER/AFR ANTES de pwm_init() para ver estado de partida.
5. **MCSDK reference**: descargar X-CUBE-MCSDK ejemplo para B-G431B-ESC1, comparar línea por línea.

### Estado del firmware al cierre

- `apps/02_pwm_adc/src/pwm.c`: duties extremos para diagnóstico (CCR1=0, CCR2=850, CCR3=1690).
- `apps/02_pwm_adc/src/main.c`: dump diagnóstico GPIO + post-enable verification.
- ⚠ El test de duties extremos NO se llegó a medir — usuario pausó para retomar con cabeza fresca.

### Para arrancar sesión 8

1. Revisar firmware actual (pwm.c + main.c) antes de modificar.
2. Aplicar H1 (DBP) como primera prueba: agregar PWR_CR1_DBP en pwm.c antes de configurar PC13.
3. Si no funciona, ir a H2/H3 (dumps de RTC/SYSCFG).
4. Como último recurso, H5 (comparar con MCSDK).

### Documentación pedagógica pendiente

Cuando resolvamos el bug, escribir nota N1.9 en FIELD_NOTES.md: "Trampas de pines especiales en STM32G4 — PC13 y backup domain". Cualquier solución que encontremos vale la pena documentar.

---

## Próxima sesión — debug fase A/C del puente (continuación)

### Punto de partida

`apps/02_pwm_adc/src/pwm.c` tiene el plan documentado como comentario (9 pasos). Cada paso corresponde a una sección del `FIELD_NOTES.md` N1.3–N1.6, con la frecuencia revisada en N1.8.

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
