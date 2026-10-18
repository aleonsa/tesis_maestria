# firmware — STM32G431CB / B-G431B-ESC1

Firmware bare-metal CMSIS para el banco de pruebas FCS-M2PC + ADALINE.
Sin HAL, sin CubeIDE — solo `arm-none-eabi-gcc`, CMake y OpenOCD.

## Estructura

```
firmware/
├── CMakeLists.txt          # config compartida: toolchain, vendor, flags
├── README.md               # este archivo
├── sync_to_pi.sh           # rsync Mac → Pi
└── apps/                   # una subcarpeta por ejecutable
    └── 01_blink/           # toggle PC6 a 1 Hz (Fase 0)
        ├── CMakeLists.txt
        ├── README.md
        └── src/main.c
```

Cuando aparezca código compartido entre 2+ apps, se promueve a `lib/` (no antes).

## Workflow

**Local (desde 2026-10-11, PC propia)** — todo corre en esta máquina:

```bash
cd firmware/
cmake -B build && cmake --build build      # compila todas las apps
openocd -f interface/stlink.cfg -f target/stm32g4x.cfg \
    -c "reset_config srst_only connect_assert_srst" \
    -c "program build/apps/03_as5600/as5600.elf verify reset exit"
stty -F /dev/ttyACM0 115200 raw -echo && cat /dev/ttyACM0   # monitor serie
```

`connect_assert_srst` es el default del banco: engancha el core bajo reset y evita
"Unable to reset target" si el target está ocupado o en un estado raro (ver FIELD_NOTES
N1.21).

**Legado (Mac corporativa que bloquea USB):**

```
[Mac]  ~/Documents/tesis_maestria/firmware/
        │
        │ ./sync_to_pi.sh
        ▼
[Pi]   ~/projects/stm32g4/firmware/
        │
        │ cmake -B build && cmake --build build
        ▼
[Pi]   openocd ... program build/apps/01_blink/blink.elf
```

## Build

Desde la raíz `firmware/`:

```bash
cmake -B build
cmake --build build
```

Esto compila **todas** las apps. Para compilar una sola:

```bash
cmake --build build --target blink.elf
```

Salida de cada app queda en `build/apps/<nombre>/<target>.elf` (+ `.bin`, `.hex`, `.map`).

## Flash

```bash
openocd -f interface/stlink.cfg -f target/stm32g4x.cfg \
    -c "reset_config srst_only connect_assert_srst" \
    -c "program build/apps/01_blink/blink.elf verify reset exit"
```

## Apps actuales

| Folder | Fase del proyecto | Qué prueba |
|---|---|---|
| `01_blink` | 0 | Cadena toolchain → flash → LED. SysTick para base de tiempo precisa. |
| `02_pwm_adc` | 1 | TIM1 PWM 50 kHz + OPAMP/ADC de corriente + ISR JEOS + open-loop. Modos de diagnóstico: barrido de amplitudes, calibración K con DMM (`CAL_TARGET_PHASE`), stream DC. |
| `03_as5600` | 1 | Encoder por I²C1, polling e IRQ; throughput del bus. |
| `04_theta_pll` | 1 | Observador PLL tipo II de θ/ω a 50 kHz (TIM2), sin motor. |
| `05_phase_align` | 1 | Offset θ_e ↔ fase A y sentido del encoder (necesita Vbus y motor). |
| `06_openloop_sync` | 1→2 | Cadena completa en la ISR: PLL + offset + Clarke + Park (CORDIC) con el motor en open-loop. i_dq debe ser DC. Mide el costo del paso. Esqueleto del controlador. **Prueba de regresión obligatoria tras cualquier cambio en `lib/` o en los flags** (sesión 22: destapó el CORDIC roto); incluye autotest del CORDIC al arrancar. |
| `07_m2pc` | 3→5 | **FCS-M²PC en lazo cerrado** (`lib/m2pc`) + PI de velocidad a 1 kHz + logger a 1 kHz (921600 baud). `python_scripts/analyze_m2pc_log.py` reconstruye la BEMF del log. Offset de corriente en línea desde el valle del PWM (`pwm_valley_trigger_enable`, N1.26). Telemetría dq en la ISR (línea `E` a 100 Hz), escalones de i_q* con PI congelado y captura de 1024 muestras a 50 kHz (Exp. 1) y trayectoria senoidal de velocidad con inversión (`TRAJ_*`, Exp. 2); habilita el FPU en `SystemInit` (sesión 23). `-O2`. |
| `08_bemf_oc` | 2 | **BEMF en circuito abierto**: puente apagado (compuertas como GPIO en bajo), red de sensado de BEMF del 6‑step (PA4/PC4/PB11, `GPIO_BEMF` en bajo) + **10 kΩ externa de +12 V a OUT1** para fijar el modo común; PLL a 50 kHz, ADC2 regular a 10 kHz, stream a 2.5 kHz. Llanta a mano. `python_scripts/bemf_oc_analyze.py`. Sesión 24: THD < 0.7 %. |

`lib/m2pc` elige el sector por la dirección de la tensión deadbeat u* = e + R·i* + (L/Ts)(i* − i), no por la BEMF (θ_e + 90°) como el paper: con par negativo a baja velocidad la regla del paper no tiene candidatos que frenen/inviertan (sesión 23).

`lib/cordic` entrega sin y cos en **una sola lectura** de 16 bits (`RESSIZE=1`, `NRES=0`) y verifica el CSR por relectura: con dos lecturas de 32 bits la segunda se adelantaba y con la escritura tras habilitar el reloj se perdía (N1.27).

`lib/` contiene los drivers compartidos: `clock`, `uart`, `i2c`, `as5600`, `theta_pll`, `pwm`, `adc`, `openloop`, `cordic`, `m2pc`, `motor_params.h` (todos los parámetros físicos y constantes Q derivadas). La ISR de corriente llama al paso de control por hook (`adc_set_control_hook`). La lib compila en `-O2`.

Timing del PWM (sesión 20): muestreo en el **pico** (OC4REF, CCR4=ARR−1, todos los low-side ON) y UEV también en el pico (`RCR=0` antes del `UG`, `RCR=1` tras `CEN`), así que un comando calculado en el paso k se aplica durante [k+1, k+2]: retardo entero, compensado en `m2pc_step`. Presupuesto: 20 µs; el paso completo cuesta 13 µs.

Convenciones de signo fijadas (FIELD_NOTES N1.23/N1.24): secuencia positiva en `openloop` (b atrasa 120°), Clarke β = (b − c)/√3, el shunt lee −i_fase (`offset − raw` ya trae el signo), encoder `ENCODER_SIGN = +1` con `PHASE_A_OFFSET_RAW = 2558`.

## Dependencias

- `gcc-arm-none-eabi` ≥ 10
- `cmake` ≥ 3.20
- `openocd` ≥ 0.11
- Clone de `STM32CubeG4` en `tesis_maestria/vendor/STM32CubeG4` (junto a `firmware/`)
  con el submódulo `Drivers/CMSIS/Device/ST/STM32G4xx` inicializado.
