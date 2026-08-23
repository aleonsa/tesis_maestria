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
)

target_include_directories(stm32g4_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
