# 01_blink — toggle PC6 a 1 Hz con SysTick.
# Config compartida (toolchain, vendor, flags) viene del CMakeLists del padre.

add_executable(blink.elf
    src/main.c
    src/clock.c
    ${STARTUP_SRC}
)

target_link_options(blink.elf PRIVATE -Wl,-Map=blink.map)

add_custom_command(TARGET blink.elf POST_BUILD
    COMMAND ${CMAKE_SIZE}    "$<TARGET_FILE:blink.elf>"
    COMMAND ${CMAKE_OBJCOPY} -O binary "$<TARGET_FILE:blink.elf>" "blink.bin"
    COMMAND ${CMAKE_OBJCOPY} -O ihex   "$<TARGET_FILE:blink.elf>" "blink.hex"
    COMMENT "Tamaño y generación de .bin/.hex"
)
