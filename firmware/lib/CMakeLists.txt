# stm32g4_lib — código compartido entre apps.
#
# STATIC library: se compila una vez, cada .elf consumidor la linkea.
# Los include dirs son PUBLIC: los consumidores heredan el path automáticamente
# y pueden hacer #include "clock.h" / "uart.h" sin path explícito.

add_library(stm32g4_lib STATIC
    clock.c
    uart.c
)

target_include_directories(stm32g4_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
