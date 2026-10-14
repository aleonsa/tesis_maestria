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

`lib/` contiene los drivers compartidos: `clock`, `uart`, `i2c`, `as5600`, `theta_pll`, `pwm`, `adc`, `openloop`. La ISR de corriente llama al paso de control por hook (`adc_set_control_hook`).

## Dependencias

- `gcc-arm-none-eabi` ≥ 10
- `cmake` ≥ 3.20
- `openocd` ≥ 0.11
- Clone de `STM32CubeG4` en `tesis_maestria/vendor/STM32CubeG4` (junto a `firmware/`)
  con el submódulo `Drivers/CMSIS/Device/ST/STM32G4xx` inicializado.
