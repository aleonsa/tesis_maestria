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

## Build (en la Pi)

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

## Flash (en la Pi)

```bash
openocd -f interface/stlink.cfg -f target/stm32g4x.cfg \
    -c "program build/apps/01_blink/blink.elf verify reset exit"
```

## Apps actuales

| Folder | Fase del proyecto | Qué prueba |
|---|---|---|
| `01_blink` | 0 | Cadena toolchain → flash → LED. SysTick para base de tiempo precisa. |

## Dependencias (en la Pi)

- `gcc-arm-none-eabi` ≥ 10
- `cmake` ≥ 3.20
- `openocd` ≥ 0.11
- Clone de `STM32CubeG4` en `~/projects/stm32g4/vendor/STM32CubeG4` con el submódulo `Drivers/CMSIS/Device/ST/STM32G4xx` inicializado.
