# 05_phase_align — alineación θ_e ↔ fase A y verificación del sentido de giro.
# Prerequisito del control por campo: sin el offset θ_e(fase A) y el signo del
# encoder, Clarke/Park trabaja en un marco torcido y el lazo no cierra.
# Config compartida (toolchain, vendor, flags) viene del CMakeLists del padre.

add_executable(phase_align.elf
    src/main.c
    ${STARTUP_SRC}
)

target_link_libraries(phase_align.elf PRIVATE stm32g4_lib)

target_link_options(phase_align.elf PRIVATE -Wl,-Map=phase_align.map)

add_custom_command(TARGET phase_align.elf POST_BUILD
    COMMAND ${CMAKE_SIZE}    "$<TARGET_FILE:phase_align.elf>"
    COMMAND ${CMAKE_OBJCOPY} -O binary "$<TARGET_FILE:phase_align.elf>" "phase_align.bin"
    COMMAND ${CMAKE_OBJCOPY} -O ihex   "$<TARGET_FILE:phase_align.elf>" "phase_align.hex"
    COMMENT "Tamaño y generación de .bin/.hex"
)
