# stm32g4_lib — código compartido entre apps.
#
# STATIC library: se compila una vez, cada .elf consumidor la linkea.
# Los include dirs son PUBLIC: los consumidores heredan el path automáticamente
# y pueden hacer #include "clock.h" / "uart.h" sin path explícito.
#
# i2c/as5600 nacieron dentro de apps/03_as5600 y se promovieron aquí el
# 2026-08-18: el lazo de control (app 02 y siguientes) necesita el encoder, y
# duplicar el driver garantizaba que las dos copias divergieran. swi2c se queda
# en 03 — es la red de seguridad del bring-up, no código de producción.

add_library(stm32g4_lib STATIC
    clock.c
    uart.c
    i2c.c
    as5600.c
    theta_pll.c
    pwm.c
    adc.c
    openloop.c
    cordic.c
    m2pc.c
    adaline.c
)

# La lib va en -O2 desde el 2026-10-14: el controlador (07_m2pc) necesita que
# el camino caliente (PLL, I2C, Clarke, CORDIC, M2PC) quepa en medio periodo.
# Todo acceso a periféricos es volatile; -O2 no cambia la semántica.
target_compile_options(stm32g4_lib PRIVATE -O2)

# pwm/adc/openloop se promovieron desde apps/02_pwm_adc el 2026-10-13: la app
# 05_phase_align (y la futura app de control) los necesita, y duplicar
# drivers garantiza divergencia. adc.c ya no incluye openloop.h: la ISR de
# JEOS llama al paso de control por hook (adc_set_control_hook).

target_include_directories(stm32g4_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
