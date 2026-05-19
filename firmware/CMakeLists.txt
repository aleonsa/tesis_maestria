cmake_minimum_required(VERSION 3.20)

# ============================================================================
# Cross-compile a Cortex-M4 con FPU hard (FPv4-SP, single precision).
# Estas líneas DEBEN ir antes de project(), porque CMake las consulta al
# configurar los lenguajes.
# ============================================================================
set(CMAKE_SYSTEM_NAME       Generic)
set(CMAKE_SYSTEM_PROCESSOR  arm)
set(CMAKE_C_COMPILER        arm-none-eabi-gcc)
set(CMAKE_ASM_COMPILER      arm-none-eabi-gcc)
set(CMAKE_OBJCOPY           arm-none-eabi-objcopy)
set(CMAKE_SIZE              arm-none-eabi-size)

# Cross-compile: el try_compile() de CMake intenta linkear un binario y falla
# si no puede ejecutar el linker del host. Lo limitamos a static library.
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

project(stm32g4_firmware LANGUAGES C ASM)

# Genera build/compile_commands.json para clangd / language servers.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Vendor: clone de STM32CubeG4 con CMSIS Device Pack.
# Layout esperado (en la Pi):
#   ~/projects/stm32g4/firmware/   ← este CMakeLists
#   ~/projects/stm32g4/vendor/STM32CubeG4/
# ============================================================================
set(VENDOR "${CMAKE_SOURCE_DIR}/../vendor/STM32CubeG4"
    CACHE PATH "Path al clone de STM32CubeG4")

set(CMSIS_DEVICE   "${VENDOR}/Drivers/CMSIS/Device/ST/STM32G4xx")
set(CMSIS_CORE     "${VENDOR}/Drivers/CMSIS/Core")
set(STARTUP_SRC    "${CMSIS_DEVICE}/Source/Templates/gcc/startup_stm32g431xx.s")
set(LINKER_SCRIPT  "${VENDOR}/Projects/NUCLEO-G431KB/Examples/FLASH/FLASH_EraseProgram/STM32CubeIDE/STM32G431KBTX_FLASH.ld")

# ============================================================================
# Flags compartidos por todas las apps (heredados vía add_subdirectory).
# ============================================================================
set(CPU_FLAGS "-mcpu=cortex-m4 -mthumb -mfloat-abi=hard -mfpu=fpv4-sp-d16")

set(CMAKE_C_FLAGS   "${CPU_FLAGS} -DSTM32G431xx -Wall -Wextra -ffunction-sections -fdata-sections -Og -g3")
set(CMAKE_ASM_FLAGS "${CPU_FLAGS} -g3")
set(CMAKE_EXE_LINKER_FLAGS
    "${CPU_FLAGS} -T${LINKER_SCRIPT} -Wl,--gc-sections --specs=nano.specs --specs=nosys.specs")

include_directories(
    "${CMSIS_DEVICE}/Include"
    "${CMSIS_CORE}/Include"
)

# ============================================================================
# Apps — cada subdir es un ejecutable independiente.
# ============================================================================
add_subdirectory(apps/01_blink)
