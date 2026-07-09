# 03_as5600 — bring-up del encoder magnético AS5600 vía I2C1 (PB6/PB7).
# Fase 1 del bring-up del banco: cierra el sensing de posición del rotor.
# Config compartida (toolchain, vendor, flags) viene del CMakeLists del padre.

add_executable(as5600.elf
    src/main.c
    src/i2c.c
    src/as5600.c
    ${STARTUP_SRC}
)

target_link_libraries(as5600.elf PRIVATE stm32g4_lib)

target_link_options(as5600.elf PRIVATE -Wl,-Map=as5600.map)

add_custom_command(TARGET as5600.elf POST_BUILD
    COMMAND ${CMAKE_SIZE}    "$<TARGET_FILE:as5600.elf>"
    COMMAND ${CMAKE_OBJCOPY} -O binary "$<TARGET_FILE:as5600.elf>" "as5600.bin"
    COMMAND ${CMAKE_OBJCOPY} -O ihex   "$<TARGET_FILE:as5600.elf>" "as5600.hex"
    COMMENT "Tamaño y generación de .bin/.hex"
)
