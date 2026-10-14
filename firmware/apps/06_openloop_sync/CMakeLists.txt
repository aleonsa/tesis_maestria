# 06_openloop_sync — cadena de sensado completa en la ISR de control, con el
# motor en open-loop: PLL + offset de fase A + Clarke con signo + Park (CORDIC).
# Si θ_e es correcta, i_d e i_q son DC. Mide además el costo del paso en la ISR.
# Config compartida (toolchain, vendor, flags) viene del CMakeLists del padre.

add_executable(openloop_sync.elf
    src/main.c
    ${STARTUP_SRC}
)

target_link_libraries(openloop_sync.elf PRIVATE stm32g4_lib)

target_link_options(openloop_sync.elf PRIVATE -Wl,-Map=openloop_sync.map)

add_custom_command(TARGET openloop_sync.elf POST_BUILD
    COMMAND ${CMAKE_SIZE}    "$<TARGET_FILE:openloop_sync.elf>"
    COMMAND ${CMAKE_OBJCOPY} -O binary "$<TARGET_FILE:openloop_sync.elf>" "openloop_sync.bin"
    COMMAND ${CMAKE_OBJCOPY} -O ihex   "$<TARGET_FILE:openloop_sync.elf>" "openloop_sync.hex"
    COMMENT "Tamaño y generación de .bin/.hex"
)
