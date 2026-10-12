# 04_theta_pll — observador PLL tipo II para θ/ω del rotor sobre el AS5600.
# Banco de validación del estimador de posición (sin PWM/ADC/motor).
# Config compartida (toolchain, vendor, flags) viene del CMakeLists del padre.

add_executable(theta_pll.elf
    src/main.c
    ${STARTUP_SRC}
)

target_link_libraries(theta_pll.elf PRIVATE stm32g4_lib)

target_link_options(theta_pll.elf PRIVATE -Wl,-Map=theta_pll.map)

add_custom_command(TARGET theta_pll.elf POST_BUILD
    COMMAND ${CMAKE_SIZE}    "$<TARGET_FILE:theta_pll.elf>"
    COMMAND ${CMAKE_OBJCOPY} -O binary "$<TARGET_FILE:theta_pll.elf>" "theta_pll.bin"
    COMMAND ${CMAKE_OBJCOPY} -O ihex   "$<TARGET_FILE:theta_pll.elf>" "theta_pll.hex"
    COMMENT "Tamaño y generación de .bin/.hex"
)
