# 02_pwm_adc — TIM1 PWM 3-fásico complementario @ 30 kHz + ADC sincronizado.
# Fase 1 del bring-up del banco experimental FCS-M2PC.
# Config compartida (toolchain, vendor, flags) viene del CMakeLists del padre.

add_executable(pwm_adc.elf
    src/main.c
    src/pwm.c
    ${STARTUP_SRC}
)

target_link_libraries(pwm_adc.elf PRIVATE stm32g4_lib)

target_link_options(pwm_adc.elf PRIVATE -Wl,-Map=pwm_adc.map)

add_custom_command(TARGET pwm_adc.elf POST_BUILD
    COMMAND ${CMAKE_SIZE}    "$<TARGET_FILE:pwm_adc.elf>"
    COMMAND ${CMAKE_OBJCOPY} -O binary "$<TARGET_FILE:pwm_adc.elf>" "pwm_adc.bin"
    COMMAND ${CMAKE_OBJCOPY} -O ihex   "$<TARGET_FILE:pwm_adc.elf>" "pwm_adc.hex"
    COMMENT "Tamaño y generación de .bin/.hex"
)
